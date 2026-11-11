//! The smooth minimum over obstacles that the energy takes as a point's
//! distance.
//!
//! Where n obstacles are equally near, the smooth minimum sees the point
//! τ·ln n nearer than it is, τ = SMOOTH_WIDTH. Obstacles farther than the
//! nearest by more than SMOOTH_REACH have no share, so that a search can
//! leave them out and give the same.

use crate::geometry::V2;

/// τ of the smooth minimum, mm.
pub const SMOOTH_WIDTH: f64 = 0.005;
/// How far beyond a point's nearest obstacle another still has a share of
/// its smooth minimum, mm: e⁻¹⁰ of the nearest's at most.
pub const SMOOTH_REACH: f64 = 10.0 * SMOOTH_WIDTH;

/// How far from a point the smooth distance looks for obstacles, for an
/// energy whose terms end at `reach`, mm.
pub fn smooth_reach(reach: f64) -> f64 {
    reach + SMOOTH_REACH
}

/// -τ·log Σ exp(-d/τ) at each of `count` points over the obstacles'
/// distances d, obstacle k's distance[k] at point[k], +inf at a point with
/// none; and each obstacle's share of the gradient there, exp(-d/τ) over
/// the sum, the shares of a point summing to 1. A point with one obstacle
/// gets its distance and a share of 1.
pub fn smooth_minimum(point: &[usize], distance: &[f64], count: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lowest = vec![f64::INFINITY; count];
    for (&p, &d) in point.iter().zip(distance) {
        lowest[p] = lowest[p].min(d);
    }
    // What each obstacle's distance is taken from, so that one at +inf
    // stays at +inf and none gives NaN.
    let base: Vec<f64> = lowest
        .iter()
        .map(|&l| if l.is_finite() { l } else { 0.0 })
        .collect();
    let mut share: Vec<f64> = point
        .iter()
        .zip(distance)
        .map(|(&p, &d)| {
            let above = d - base[p];
            if above <= SMOOTH_REACH {
                (-above / SMOOTH_WIDTH).exp()
            } else {
                0.0
            }
        })
        .collect();
    let mut total = vec![0.0; count];
    for (&p, &s) in point.iter().zip(&share) {
        total[p] += s;
    }
    let smooth = base
        .iter()
        .zip(&total)
        .map(|(&b, &t)| {
            if t > 0.0 {
                b - SMOOTH_WIDTH * t.ln()
            } else {
                f64::INFINITY
            }
        })
        .collect();
    for (s, &p) in share.iter_mut().zip(point) {
        if total[p] > 0.0 {
            *s /= total[p];
        }
    }
    (smooth, share)
}

/// At each of `count` points, the sum of the directions at it, each
/// weighted by its share; the smooth minimum's gradient.
pub fn weighted_sum(point: &[usize], share: &[f64], direction: &[V2], count: usize) -> Vec<V2> {
    let mut sum = vec![V2::default(); count];
    for ((&p, &s), &d) in point.iter().zip(share).zip(direction) {
        sum[p].x += s * d.x;
        sum[p].y += s * d.y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_equally_near_obstacles_read_tau_ln_2_nearer() {
        let (smooth, share) = smooth_minimum(&[0, 0], &[1.0, 1.0], 1);
        assert!((smooth[0] - (1.0 - SMOOTH_WIDTH * 2f64.ln())).abs() < 1e-15);
        assert_eq!(share, vec![0.5, 0.5]);
    }

    #[test]
    fn one_obstacle_is_its_own_minimum() {
        let (smooth, share) = smooth_minimum(&[1], &[0.3], 3);
        assert_eq!(smooth, vec![f64::INFINITY, 0.3, f64::INFINITY]);
        assert_eq!(share, vec![1.0]);
    }

    #[test]
    fn an_obstacle_beyond_the_reach_has_no_share() {
        let (smooth, share) = smooth_minimum(&[0, 0], &[0.0, SMOOTH_REACH * 1.01], 1);
        assert_eq!(smooth, vec![0.0]);
        assert_eq!(share, vec![1.0, 0.0]);
    }

    #[test]
    fn obstacles_at_infinity_leave_the_point_free() {
        let (smooth, share) = smooth_minimum(&[0], &[f64::INFINITY], 1);
        assert_eq!(smooth, vec![f64::INFINITY]);
        assert_eq!(share, vec![0.0]);
    }
}
