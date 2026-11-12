//! The terms that depend on the polyline alone: its length with the
//! short-edge barrier, and the bend-radius hinge.

use super::integrands::hinge;
use super::{EnergyParams, MAX_SMOOTH_TURN, MIN_EDGE_LENGTH, TermValue, zero_fixed};
use crate::geometry::V2;
use crate::numeric::{clip, exact_sum, maximum, numpy_sum};

/// A polyline's edge vectors, their lengths, and their unit directions,
/// zero for degenerate edges.
pub struct Edges {
    pub vectors: Vec<V2>,
    pub lengths: Vec<f64>,
    pub units: Vec<V2>,
}

impl Edges {
    pub fn of(points: &[V2]) -> Self {
        let vectors: Vec<V2> = points.windows(2).map(|w| w[1] - w[0]).collect();
        let lengths: Vec<f64> = vectors.iter().map(|v| v.norm()).collect();
        let units = vectors
            .iter()
            .zip(&lengths)
            .map(|(v, &l)| {
                let l = maximum(l, MIN_EDGE_LENGTH);
                V2::new(v.x / l, v.y / l)
            })
            .collect();
        Self {
            vectors,
            lengths,
            units,
        }
    }
}

/// The gradient at each of `n` vertices of a sum over edges, from its
/// derivatives by each edge's start and end.
pub fn scatter_edge_gradient(n: usize, start: &[V2], end: &[V2]) -> Vec<V2> {
    let mut gradient = vec![V2::default(); n];
    for (g, &s) in gradient.iter_mut().zip(start) {
        *g = *g + s;
    }
    for (g, &e) in gradient.iter_mut().skip(1).zip(end) {
        *g = *g + e;
    }
    gradient
}

/// Cumulative arc length at each vertex, starting at 0.
pub fn arc_lengths(points: &[V2]) -> Vec<f64> {
    let mut arc = Vec::with_capacity(points.len().max(1));
    arc.push(0.0);
    let mut total = 0.0;
    for (k, w) in points.windows(2).enumerate() {
        let step = (w[1] - w[0]).norm();
        total = if k == 0 { step } else { total + step };
        arc.push(total);
    }
    arc
}

/// The terms of a member's energy that depend on the polyline alone: its
/// length and, with a positive `short_edge`, mm, a barrier Σ (s - l)²/l
/// over the edges shorter than s = `short_edge` that have a free end.
///
/// Nothing else keeps neighbouring vertices apart: sliding a vertex along
/// the track barely changes length or repulsion, while an edge's
/// direction, and with it the gradient of its length and of the turns the
/// bend-radius hinge measures, is undefined as it vanishes. The barrier
/// grows without bound as an edge shrinks, so free vertices never pile up.
/// It is 0 with zero slope at s, so it leaves longer edges alone.
pub fn shape_energy(points: &[V2], fixed: &[bool], short_edge: f64) -> TermValue {
    let Edges {
        lengths, mut units, ..
    } = Edges::of(points);
    let mut value = numpy_sum(&lengths);
    if lengths.iter().any(|&l| l < short_edge) {
        let mut barrier = Vec::new();
        for (e, (&l, unit)) in lengths.iter().zip(units.iter_mut()).enumerate() {
            if l < short_edge && !(fixed[e] && fixed[e + 1]) {
                let clamped = maximum(l, MIN_EDGE_LENGTH);
                let gap = short_edge - clamped;
                barrier.push(gap * gap / clamped);
                let ratio = short_edge / clamped;
                let slope = 1.0 + (1.0 - ratio * ratio);
                *unit = V2::new(unit.x * slope, unit.y * slope);
            }
        }
        value += numpy_sum(&barrier);
    }
    let backward: Vec<V2> = units.iter().map(|&u| -u).collect();
    let gradient = scatter_edge_gradient(points.len(), &backward, &units);
    (value, zero_fixed(gradient, fixed))
}

/// 2·tan(φ/2) and its derivative, continued linearly beyond
/// MAX_SMOOTH_TURN, which keeps hairpins and full reversals finite, costly
/// and differentiable.
fn curvature(turn: f64) -> (f64, f64) {
    let clipped = clip(turn, -MAX_SMOOTH_TURN, MAX_SMOOTH_TURN);
    let tangent = libm::tan(clipped / 2.0);
    let slope = 1.0 + tangent * tangent;
    (2.0 * tangent + slope * (turn - clipped), slope)
}

/// A polyline's edges and the curvature at its interior vertices.
struct Bends {
    edges: Edges,
    /// 2·tan(φ/2) of each turn φ, as `curvature` continues it.
    curvature: Vec<f64>,
    curvature_slope: Vec<f64>,
}

impl Bends {
    fn of(points: &[V2]) -> Self {
        let edges = Edges::of(points);
        let (curvature, curvature_slope) = edges
            .vectors
            .windows(2)
            .map(|w| {
                let (incoming, outgoing) = (w[0], w[1]);
                let cross = incoming.x * outgoing.y - incoming.y * outgoing.x;
                let dot = incoming.x * outgoing.x + incoming.y * outgoing.y;
                curvature(libm::atan2(cross, dot))
            })
            .unzip();
        Self {
            edges,
            curvature,
            curvature_slope,
        }
    }
}

/// Each edge's bend radius, from the edges' `lengths` and tan(φ/2) of the
/// turn φ at each interior vertex, and the signed sum of those tangents at
/// each edge's two ends.
///
/// The bend radius of an edge is the largest offset at which its mitred
/// offset still has positive length: its length over the absolute sum,
/// infinite where the sum is 0.
pub fn edge_bend_radii(lengths: &[f64], half_turns: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut half_turn = Vec::with_capacity(half_turns.len() + 2);
    half_turn.push(0.0);
    half_turn.extend_from_slice(half_turns);
    half_turn.push(0.0);
    let spread: Vec<f64> = half_turn.windows(2).map(|w| w[0] + w[1]).collect();
    let radii = lengths
        .iter()
        .zip(&spread)
        .map(|(&l, &s)| {
            let size = s.abs();
            if size > 0.0 { l / size } else { f64::INFINITY }
        })
        .collect();
    (radii, spread)
}

/// The gradient at every vertex of a sum over interior vertices, given its
/// derivatives by each vertex's turn.
fn turn_gradient(n: usize, edges: &Edges, d_turn: &[f64]) -> Vec<V2> {
    let min_square = MIN_EDGE_LENGTH.powi(2);
    let perpendicular = |v: V2| V2::new(-v.y, v.x);
    let (d_incoming, d_outgoing): (Vec<V2>, Vec<V2>) = d_turn
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let incoming_square = maximum(edges.lengths[i] * edges.lengths[i], min_square);
            let outgoing_square = maximum(edges.lengths[i + 1] * edges.lengths[i + 1], min_square);
            let p = perpendicular(edges.vectors[i]);
            let q = perpendicular(edges.vectors[i + 1]);
            (
                V2::new(-d * p.x / incoming_square, -d * p.y / incoming_square),
                V2::new(d * q.x / outgoing_square, d * q.y / outgoing_square),
            )
        })
        .unzip();
    let mut gradient = vec![V2::default(); n];
    for (g, &d) in gradient.iter_mut().zip(&d_incoming) {
        *g = *g - d;
    }
    for (g, (&i, &o)) in gradient
        .iter_mut()
        .skip(1)
        .zip(d_incoming.iter().zip(&d_outgoing))
    {
        *g = *g + (i - o);
    }
    for (g, &o) in gradient.iter_mut().skip(2).zip(&d_outgoing) {
        *g = *g + o;
    }
    gradient
}

/// Σ lₑ·c(rₑ - `limit`) over the edges, weighted like the clearance hinge
/// c and with its standoff, where rₑ is edge e's bend radius: its length
/// lₑ over |tan(φ/2)| summed over its two ends' turns φ, signed, 0 at the
/// polyline's ends (`edge_bend_radii`).
///
/// That is the radius `pairs.check_bend_radius` measures: below it the
/// polyline's mitred offset to the inside folds back on itself, so with
/// `limit` set to `pairs.bend_radius_limit` it keeps a pair's inner track
/// from folding. A bend at the limit is as hard a constraint as the
/// clearance, and has nothing to do with how smooth a track should be,
/// hence the clearance's weight and standoff.
pub fn bend_radius_energy(
    points: &[V2],
    fixed: &[bool],
    limit: f64,
    params: &EnergyParams,
) -> TermValue {
    let n = points.len();
    if n < 2 {
        return (0.0, vec![V2::default(); n]);
    }
    let Bends {
        edges,
        curvature,
        curvature_slope,
    } = Bends::of(points);
    let half_turns: Vec<f64> = curvature.iter().map(|&c| c / 2.0).collect();
    let (radii, spread) = edge_bend_radii(&edges.lengths, &half_turns);
    let active: Vec<bool> = edges
        .lengths
        .iter()
        .zip(&spread)
        .map(|(&l, &s)| l < (limit + params.standoff) * s.abs())
        .collect();
    if !active.contains(&true) {
        return (0.0, vec![V2::default(); n]);
    }
    let weight = params.clearance_weight;
    let mut by_length = Vec::with_capacity(edges.lengths.len());
    let mut d_length = Vec::with_capacity(edges.lengths.len());
    let mut d_half_turn = Vec::with_capacity(edges.lengths.len());
    for e in 0..edges.lengths.len() {
        let length = edges.lengths[e];
        let (safe_size, radius) = if active[e] {
            (spread[e].abs(), radii[e])
        } else {
            (1.0, f64::INFINITY)
        };
        let (hinge_value, hinge_slope) = hinge(params.standoff, radius - limit);
        by_length.push(length * hinge_value);
        d_length.push(weight * (hinge_value + length * hinge_slope / safe_size));
        let d_size = if active[e] {
            let ratio = length / safe_size;
            -weight * (ratio * ratio) * hinge_slope
        } else {
            0.0
        };
        d_half_turn.push(numpy_sign(spread[e]) * d_size);
    }
    let value = weight * exact_sum(by_length);
    let d_turn: Vec<f64> = d_half_turn
        .windows(2)
        .zip(&curvature_slope)
        .map(|(w, &slope)| (w[0] + w[1]) * slope / 2.0)
        .collect();
    let backward: Vec<V2> = d_length
        .iter()
        .zip(&edges.units)
        .map(|(&d, &u)| -d * u)
        .collect();
    let forward: Vec<V2> = d_length
        .iter()
        .zip(&edges.units)
        .map(|(&d, &u)| d * u)
        .collect();
    let gradient = scatter_edge_gradient(n, &backward, &forward)
        .into_iter()
        .zip(turn_gradient(n, &edges, &d_turn))
        .map(|(a, b)| a + b)
        .collect();
    (value, zero_fixed(gradient, fixed))
}

/// numpy's `sign`: 0 at 0, NaN at NaN.
fn numpy_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::tests::{assert_gradient_matches, free, params, wiggly};

    fn arc(radius: f64, angle: f64, n: usize) -> Vec<V2> {
        (0..n)
            .map(|k| {
                let theta = angle * k as f64 / (n - 1) as f64;
                V2::new(radius * libm::cos(theta), radius * libm::sin(theta))
            })
            .collect()
    }

    #[test]
    fn the_length_is_the_sum_of_the_edges() {
        let points = [V2::new(0.0, 0.0), V2::new(3.0, 4.0), V2::new(3.0, 10.0)];
        assert_eq!(shape_energy(&points, &free(3), 0.0).0, 11.0);
    }

    #[test]
    fn the_length_s_gradient_matches_finite_differences() {
        for seed in 0..3 {
            let points = wiggly(seed);
            assert_gradient_matches(|p, f| shape_energy(p, f, 0.0), &points);
        }
    }

    #[test]
    fn the_short_edge_barrier_s_gradient_matches_finite_differences() {
        let mut points = arc(2.0, 1.0, 9);
        let middle = 0.5 * (points[4] + points[5]);
        let chord = points[5] - points[4];
        let along = (1.0 / chord.norm()) * chord;
        points[4] = middle - 0.015 * along;
        points[5] = middle + 0.015 * along;
        assert_gradient_matches(|p, f| shape_energy(p, f, 0.1), &points);
    }

    #[test]
    fn the_short_edge_barrier_leaves_an_edge_between_fixed_vertices_alone() {
        let mut points = arc(2.0, 1.0, 9);
        points[5] = points[4] + V2::new(1e-3, 0.0);
        let mut fixed = free(9);
        let plain = shape_energy(&points, &fixed, 0.0).0;
        fixed[4] = true;
        fixed[5] = true;
        assert_eq!(shape_energy(&points, &fixed, 0.1).0, plain);
    }

    #[test]
    fn the_bend_radius_s_gradient_matches_finite_differences() {
        for seed in 0..4 {
            let points = wiggly(seed);
            assert_gradient_matches(|p, f| bend_radius_energy(p, f, 1.8, &params()), &points);
        }
    }

    #[test]
    fn a_hairpin_beyond_the_smooth_turn_keeps_a_matching_gradient() {
        let hairpin = [
            V2::new(0.0, 0.0),
            V2::new(2.0, 0.0),
            V2::new(0.1, 0.05),
            V2::new(0.0, 1.0),
        ];
        assert_gradient_matches(|p, f| bend_radius_energy(p, f, 1.8, &params()), &hairpin);
    }

    #[test]
    fn a_full_reversal_is_finite() {
        let reversal = [V2::new(0.0, 0.0), V2::new(1.0, 0.0), V2::new(0.0, 0.0)];
        let (value, gradient) = bend_radius_energy(&reversal, &free(3), 1.8, &params());
        assert!(value.is_finite() && value > 0.0);
        assert!(gradient.iter().all(|g| g.x.is_finite() && g.y.is_finite()));
    }

    #[test]
    fn a_degenerate_edge_is_finite() {
        let points = [
            V2::new(-2.0, 1.1),
            V2::new(0.0, 1.1),
            V2::new(0.0, 1.1 + MIN_EDGE_LENGTH / 2.0),
            V2::new(2.0, 1.2),
        ];
        for (value, gradient) in [
            shape_energy(&points, &free(4), 0.1),
            bend_radius_energy(&points, &free(4), 1.8, &params()),
        ] {
            assert!(value.is_finite());
            assert!(gradient.iter().all(|g| g.x.is_finite() && g.y.is_finite()));
        }
    }

    #[test]
    fn wide_bends_and_straight_edges_cost_nothing() {
        let (value, gradient) = bend_radius_energy(&arc(1.0, 1.5, 33), &free(33), 0.5, &params());
        assert_eq!(value, 0.0);
        assert!(gradient.iter().all(|&g| g == V2::default()));
        let straight: Vec<V2> = (0..11).map(|k| V2::new(k as f64 * 1e-3, 0.0)).collect();
        assert_eq!(
            bend_radius_energy(&straight, &free(11), 1.8, &params()).0,
            0.0
        );
    }

    #[test]
    fn a_tighter_bend_costs_more() {
        let values: Vec<f64> = [1.0, 0.8, 0.5]
            .into_iter()
            .map(|r| bend_radius_energy(&arc(r, 1.5, 33), &free(33), 1.0, &params()).0)
            .collect();
        assert!(0.0 < values[0] && values[0] < values[1] && values[1] < values[2]);
    }

    #[test]
    fn fixed_vertices_get_no_gradient() {
        let points = wiggly(1);
        let mut fixed = free(points.len());
        fixed[0] = true;
        fixed[4] = true;
        let (_, gradient) = bend_radius_energy(&points, &fixed, 1.8, &params());
        assert_eq!(gradient[4], V2::default());
        let (_, gradient) = shape_energy(&points, &fixed, 0.0);
        assert_eq!(gradient[0], V2::default());
    }

    #[test]
    fn arc_lengths_accumulate_from_zero() {
        let points = [V2::new(0.0, 0.0), V2::new(3.0, 4.0), V2::new(3.0, 10.0)];
        assert_eq!(arc_lengths(&points), vec![0.0, 5.0, 11.0]);
    }

    #[test]
    fn bend_radii_divide_lengths_by_the_summed_half_turns() {
        let (radii, spread) = edge_bend_radii(&[1.0, 2.0, 3.0], &[0.5, -0.5]);
        assert_eq!(spread, vec![0.5, 0.0, -0.5]);
        assert_eq!(radii, vec![2.0, f64::INFINITY, 6.0]);
    }
}
