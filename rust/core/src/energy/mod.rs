//! Energy terms of a track polyline, each returning its value and gradient.
//!
//! Polylines are vertices in mm. Every term returns the gradient with
//! respect to all vertex coordinates, with the rows of fixed vertices
//! zeroed.
//!
//! The distance the obstacle terms take is a smooth minimum over the
//! obstacles near a point, -τ·log Σ exp(-d/τ) with τ = SMOOTH_WIDTH
//! (`field::smooth::smooth_minimum`), and not the nearest one's alone:
//! where a track lies equally near two obstacles the nearest-only distance
//! has a kink, its gradient that of one side, and a track centred between
//! its neighbours could not move. An obstacle is one chain or one track of
//! a section, one via, the board edge, or static copper as
//! `field::obstacles::Obstacles` groups it. Within one obstacle the
//! distance stays its nearest edge's, so a straight neighbour made of many
//! edges adds nothing, but where two parts of one obstacle are equally
//! near, the kink of the nearest-only distance remains: inside the corner
//! of a bent track, or between two arms of one, as in a U or a meander, or
//! of copper that one obstacle joins elsewhere. Where n obstacles are
//! equally near, the energy sees the point τ·ln n nearer than it is. Every
//! clearance judgement takes the exact distance (a field's
//! `signed_distance`).
//!
//! The optimizer scales its variables by each one's stiffness, which
//! `quadrature::edge_curvature` and `quadrature::point_stiffness` build
//! from an integrand's second derivative, its `integrands::Curvature`.
//! Nothing derives one from the other: a change to an integrand must change
//! its curvature too, which the tests check against central differences.
//!
//! The terms reproduce the numpy code they replaced operation by operation
//! (`numeric`), except where numpy left the order of a sum to its BLAS,
//! which differs between BLAS builds and even between the alignments of
//! one array's rows. There the order is fixed instead: an edge's mean sums
//! its samples in order (`quadrature::weighted_mean`), and a sum over a
//! polyline's edges weighted by their lengths is correctly rounded
//! (`numeric::exact_sum`). At a stationary point the line search's last
//! steps change the energy by about its round-off, so whether they fail or
//! take a step on noise turns on the last bits of the sum; summed in a
//! fixed order, it is a function of the edges' terms alone, the same on
//! every platform.

pub mod integrands;
pub mod polyline;
pub mod quadrature;

use crate::geometry::V2;

/// A term's value and its gradient at every vertex.
pub type TermValue = (f64, Vec<V2>);

/// Centre distance at which a ground via and a regular one stop attracting
/// each other, mm. Beyond the 1.8 mm over which two 0.6 mm ground vias of
/// one net with 0.2 mm clearance push each other at the default range, so
/// that one pushed out by its neighbours is still drawn back. At an
/// attraction of 0.25, a pair keeping up to 1.5 mm between centres still
/// balances beyond its clearance, 2.5 mm and 4 mm move that balance by
/// 0.02 mm, and stitching grids took about as many iterations to settle at
/// ranges from 2 mm to 3 mm.
pub const DEFAULT_ATTRACTION_RANGE: f64 = 3.0;

/// 1 nm, KiCad's coordinate resolution: shorter edges are treated as
/// degenerate.
pub const MIN_EDGE_LENGTH: f64 = 1e-6;

/// Beyond this turning angle the curvature 2·tan(φ/2) is continued linearly
/// in φ, keeping hairpins and full reversals finite, costly and
/// differentiable.
pub const MAX_SMOOTH_TURN: f64 = 0.95 * std::f64::consts::PI;

/// Weights and shape parameters of the total energy; length has weight 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyParams {
    pub repulsion: f64,
    /// Distance beyond the clearance boundary at which repulsion reaches 0,
    /// mm.
    pub falloff_range: f64,
    pub clearance_weight: f64,
    /// Distance from the clearance boundary at which the hinge starts, mm.
    pub standoff: f64,
    /// Interior quadrature points per edge, in addition to its endpoints.
    pub points_per_edge: usize,
    /// Weight of a ground via and a regular one pulling together; 0 turns
    /// it off (`integrands::Attraction`).
    pub attraction: f64,
    /// Centre distance at which that pull reaches 0, mm.
    pub attraction_range: f64,
}

impl EnergyParams {
    /// Distance from an obstacle beyond which both obstacle terms of the
    /// energy are zero, mm, so farther copper need not be looked up.
    pub fn obstacle_reach(&self) -> f64 {
        let falloff = if self.repulsion != 0.0 {
            self.falloff_range
        } else {
            0.0
        };
        self.standoff.max(falloff)
    }
}

fn zero_fixed(mut gradient: Vec<V2>, fixed: &[bool]) -> Vec<V2> {
    for (g, &f) in gradient.iter_mut().zip(fixed) {
        if f {
            *g = V2::default();
        }
    }
    gradient
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const FD_STEP: f64 = 1e-6;
    const GRADIENT_RTOL: f64 = 1e-5;

    pub fn params() -> EnergyParams {
        EnergyParams {
            repulsion: 0.7,
            falloff_range: 1.5,
            clearance_weight: 50.0,
            standoff: 0.2,
            points_per_edge: 3,
            attraction: 0.0,
            attraction_range: DEFAULT_ATTRACTION_RANGE,
        }
    }

    pub fn free(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    /// Nine vertices from (-2.5, 1.1) to (2.5, 0.9), each moved by up to
    /// 0.3 mm in a pattern `seed` picks: beside a unit disc at the origin,
    /// partly inside it, with tight bends.
    pub fn wiggly(seed: u64) -> Vec<V2> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..9)
            .map(|k| {
                let t = k as f64 / 8.0;
                V2::new(-2.5 + 5.0 * t + 0.6 * next(), 1.1 - 0.2 * t + 0.6 * next())
            })
            .collect()
    }

    /// The analytic gradient of `term` at `points` agrees with central
    /// differences of its value.
    pub fn assert_gradient_matches(term: impl Fn(&[V2], &[bool]) -> TermValue, points: &[V2]) {
        let fixed = free(points.len());
        let (_, analytic) = term(points, &fixed);
        let mut error = 0.0;
        let mut size = 0.0;
        for i in 0..points.len() {
            for axis in 0..2 {
                let shifted = |by: f64| {
                    let mut moved = points.to_vec();
                    if axis == 0 {
                        moved[i].x += by;
                    } else {
                        moved[i].y += by;
                    }
                    term(&moved, &fixed).0
                };
                let numeric = (shifted(FD_STEP) - shifted(-FD_STEP)) / (2.0 * FD_STEP);
                let exact = if axis == 0 {
                    analytic[i].x
                } else {
                    analytic[i].y
                };
                error += (exact - numeric).powi(2);
                size += numeric.powi(2);
            }
        }
        assert!(
            error.sqrt() < GRADIENT_RTOL * size.sqrt().max(1e-12),
            "gradient off by {} of {}",
            error.sqrt(),
            size.sqrt()
        );
    }

    #[test]
    fn the_obstacle_reach_leaves_out_the_repulsion_without_weight() {
        let p = params();
        assert_eq!(p.obstacle_reach(), 1.5);
        let without = EnergyParams {
            repulsion: 0.0,
            ..p
        };
        assert_eq!(without.obstacle_reach(), 0.2);
    }
}
