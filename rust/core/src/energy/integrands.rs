//! The functions of distance the energy integrates, their slopes and their
//! curvatures.

use super::EnergyParams;
use crate::numeric::{clip, maximum, minimum};

/// A function of the distance to an obstacle, giving its value and slope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrand {
    /// Repulsion plus clearance, weighted.
    Obstacle {
        repulsion: f64,
        falloff_range: f64,
        clearance_weight: f64,
        standoff: f64,
    },
    /// The clearance hinge alone, weighted: for what only limits where a
    /// thing may go and does not repel it.
    Hinge { weight: f64, standoff: f64 },
    /// The repulsion alone, weighted: for what pushes away without being a
    /// wall.
    Repulsion { weight: f64, falloff_range: f64 },
}

impl Integrand {
    pub fn obstacle(params: &EnergyParams) -> Self {
        Self::Obstacle {
            repulsion: params.repulsion,
            falloff_range: params.falloff_range,
            clearance_weight: params.clearance_weight,
            standoff: params.standoff,
        }
    }

    pub fn hinge(params: &EnergyParams) -> Self {
        Self::Hinge {
            weight: params.clearance_weight,
            standoff: params.standoff,
        }
    }

    pub fn repulsion(params: &EnergyParams) -> Self {
        Self::Repulsion {
            weight: params.repulsion,
            falloff_range: params.falloff_range,
        }
    }

    /// The value and the slope at `distance`.
    pub fn at(&self, distance: f64) -> (f64, f64) {
        match *self {
            Self::Obstacle {
                repulsion,
                falloff_range,
                clearance_weight,
                standoff,
            } => {
                let (falloff_value, falloff_slope) = falloff(falloff_range, distance);
                let (hinge_value, hinge_slope) = hinge(standoff, distance);
                (
                    repulsion * falloff_value + clearance_weight * hinge_value,
                    repulsion * falloff_slope + clearance_weight * hinge_slope,
                )
            }
            Self::Hinge { weight, standoff } => {
                let (value, slope) = hinge(standoff, distance);
                (weight * value, weight * slope)
            }
            Self::Repulsion {
                weight,
                falloff_range,
            } => {
                let (value, slope) = falloff(falloff_range, distance);
                (weight * value, weight * slope)
            }
        }
    }
}

/// An integrand's second derivative by the distance, and inside an
/// obstacle that at its boundary. Inside, the integrand is linear, but a
/// step back out meets the boundary's curvature, and a track pressed at
/// the clearance lies a round-off inside it as often as outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Curvature {
    /// `Integrand::Obstacle`'s.
    Obstacle {
        repulsion: f64,
        falloff_range: f64,
        clearance_weight: f64,
        standoff: f64,
    },
    /// `Integrand::Hinge`'s.
    Hinge { weight: f64, standoff: f64 },
}

impl Curvature {
    pub fn obstacle(params: &EnergyParams) -> Self {
        Self::Obstacle {
            repulsion: params.repulsion,
            falloff_range: params.falloff_range,
            clearance_weight: params.clearance_weight,
            standoff: params.standoff,
        }
    }

    pub fn hinge(params: &EnergyParams) -> Self {
        Self::Hinge {
            weight: params.clearance_weight,
            standoff: params.standoff,
        }
    }

    /// The curvature at `distance`.
    pub fn at(&self, distance: f64) -> f64 {
        let outside = maximum(distance, 0.0);
        match *self {
            Self::Obstacle {
                repulsion,
                falloff_range,
                clearance_weight,
                standoff,
            } => {
                repulsion * falloff_curvature(falloff_range, outside)
                    + clearance_weight * hinge_curvature(standoff, outside)
            }
            Self::Hinge { weight, standoff } => weight * hinge_curvature(standoff, outside),
        }
    }
}

/// The pull of two vias on each other at centre distance r with limit
/// distance r_min: the shifted-force -w·(1/r - 1/R) - w·(r - R)/R² of
/// weight w = `attraction` and range R = `attraction_range`, which has zero
/// value and slope at R and is 0 beyond, continued linearly below r_min,
/// so that it stays finite and leaves the clearance hinge the wall there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attraction {
    pub weight: f64,
    pub range: f64,
}

impl Attraction {
    pub fn new(params: &EnergyParams) -> Self {
        Self {
            weight: params.attraction,
            range: params.attraction_range,
        }
    }

    /// The value and the slope at centre distance `distance` and limit
    /// distance `limit`.
    pub fn at(&self, distance: f64, limit: f64) -> (f64, f64) {
        let (weight, cutoff) = (self.weight, self.range);
        if weight == 0.0 {
            return (0.0, 0.0);
        }
        let squared_cutoff = cutoff.powi(2);
        let at = minimum(maximum(distance, limit), cutoff);
        let slope = weight * (1.0 / (at * at) - 1.0 / squared_cutoff);
        let value = -weight * (1.0 / at - 1.0 / cutoff) - weight * (at - cutoff) / squared_cutoff;
        (value + slope * (distance - at), slope)
    }
}

/// (1 - d/range)³ on [0, range), 0 beyond, linear continuation below 0.
///
/// Zero value and slope at range, so farther copper has no effect. Inside
/// an obstacle the clearance hinge dominates; the linear continuation keeps
/// the repulsion bounded there instead of growing cubically.
pub fn falloff(falloff_range: f64, distance: f64) -> (f64, f64) {
    let remaining = 1.0 - clip(distance, 0.0, falloff_range) / falloff_range;
    let boundary_slope = -3.0 / falloff_range;
    let inside = minimum(distance, 0.0);
    // numpy's `**3` is libm's pow, not two products.
    let value = remaining.powf(3.0) + boundary_slope * inside;
    let slope = if distance < 0.0 {
        boundary_slope
    } else {
        boundary_slope * (remaining * remaining)
    };
    (value, slope)
}

/// CHOMP clearance hinge with standoff ε: 0 beyond ε, (ε-d)²/(2ε) on
/// [0, ε), ε/2 - d inside.
pub fn hinge(standoff: f64, distance: f64) -> (f64, f64) {
    let depth = standoff - minimum(distance, standoff);
    if distance < 0.0 {
        (standoff / 2.0 - distance, -1.0)
    } else {
        (depth * depth / (2.0 * standoff), -depth / standoff)
    }
}

/// The second derivative of `falloff`: 0 where it is linear or 0.
pub fn falloff_curvature(falloff_range: f64, distance: f64) -> f64 {
    if distance >= 0.0 && distance < falloff_range {
        let remaining = 1.0 - clip(distance, 0.0, falloff_range) / falloff_range;
        6.0 / falloff_range.powi(2) * remaining
    } else {
        0.0
    }
}

/// The second derivative of `hinge`: 1/ε on its quadratic part.
pub fn hinge_curvature(standoff: f64, distance: f64) -> f64 {
    if distance >= 0.0 && distance < standoff {
        1.0 / standoff
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f64 = 1e-6;

    fn params() -> EnergyParams {
        EnergyParams {
            repulsion: 0.7,
            falloff_range: 1.5,
            clearance_weight: 50.0,
            standoff: 0.2,
            points_per_edge: 3,
            attraction: 0.3,
            attraction_range: 3.0,
        }
    }

    /// Distances across every part of the integrands: inside, at the
    /// boundary, within the standoff and the falloff, and beyond.
    const SWEEP: [f64; 9] = [-0.5, -1e-3, 0.0, 0.05, 0.1, 0.199, 0.7, 1.4, 3.0];

    fn central_slope(f: impl Fn(f64) -> f64, d: f64) -> f64 {
        (f(d + STEP) - f(d - STEP)) / (2.0 * STEP)
    }

    #[test]
    fn each_slope_is_its_value_s_derivative() {
        let p = params();
        for integrand in [
            Integrand::obstacle(&p),
            Integrand::hinge(&p),
            Integrand::repulsion(&p),
        ] {
            for d in SWEEP {
                let numeric = central_slope(|x| integrand.at(x).0, d);
                let slope = integrand.at(d).1;
                assert!(
                    (numeric - slope).abs() <= 1e-4 * slope.abs().max(1.0),
                    "{integrand:?} at {d}: {slope} against {numeric}"
                );
            }
        }
    }

    #[test]
    fn each_curvature_is_its_slope_s_derivative_outside() {
        let p = params();
        for (integrand, curvature) in [
            (Integrand::obstacle(&p), Curvature::obstacle(&p)),
            (Integrand::hinge(&p), Curvature::hinge(&p)),
        ] {
            for d in SWEEP.into_iter().filter(|&d| d > 0.0 && d != 0.199) {
                let numeric = central_slope(|x| integrand.at(x).1, d);
                let exact = curvature.at(d);
                assert!(
                    (numeric - exact).abs() <= 1e-4 * exact.abs().max(1.0),
                    "{curvature:?} at {d}: {exact} against {numeric}"
                );
            }
        }
    }

    #[test]
    fn inside_an_obstacle_the_curvature_is_the_boundary_s() {
        let p = params();
        for curvature in [Curvature::obstacle(&p), Curvature::hinge(&p)] {
            assert_eq!(curvature.at(-0.05), curvature.at(0.0));
            assert!(curvature.at(0.0) > 0.0);
        }
    }

    #[test]
    fn the_integrands_vanish_beyond_their_reach() {
        let p = params();
        assert_eq!(Integrand::obstacle(&p).at(p.falloff_range), (0.0, 0.0));
        assert_eq!(Integrand::hinge(&p).at(p.standoff + 0.01), (0.0, -0.0));
    }

    #[test]
    fn the_attraction_vanishes_at_its_range_and_is_linear_inside_the_limit() {
        let attraction = Attraction::new(&params());
        assert_eq!(attraction.at(3.0, 0.8), (0.0, 0.0));
        assert_eq!(attraction.at(4.0, 0.8), (0.0, 0.0));
        let (at_limit, slope) = attraction.at(0.8, 0.8);
        let (inside, inside_slope) = attraction.at(0.3, 0.8);
        assert_eq!(inside_slope, slope);
        assert!((inside - (at_limit - 0.5 * slope)).abs() < 1e-12);
        let off = Attraction {
            weight: 0.0,
            range: 3.0,
        };
        assert_eq!(off.at(0.5, 0.8), (0.0, 0.0));
    }

    #[test]
    fn the_attraction_s_slope_is_its_value_s_derivative() {
        let attraction = Attraction::new(&params());
        for r in [0.3, 0.81, 1.7, 2.99] {
            let numeric = central_slope(|x| attraction.at(x, 0.8).0, r);
            let slope = attraction.at(r, 0.8).1;
            assert!((numeric - slope).abs() <= 1e-6 * slope.abs().max(1.0));
        }
    }
}
