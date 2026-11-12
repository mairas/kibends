//! Measures and checks on a single polyline that the optimizer applies to
//! every member, independent of the problem, computed as the numpy code
//! they replaced computed them, to the last bit.
//!
//! `resample` lays each span between fixed vertices out at uniform
//! arc-length spacing, as every burst starts. `count_violations` counts
//! where a polyline enters its clearance away from its anchors, which the
//! best state may not add to. `shape_change` is how far a burst moved a
//! shape, not counting vertices sliding along it, which settles a problem.
//! `SelfCheck` is what a burst may not do to a coupled section's
//! centreline: bend an edge newly below the limit.

use crate::energy::MIN_EDGE_LENGTH;
use crate::energy::polyline::arc_lengths;
use crate::geometry::V2;
use crate::numeric::{clip, maximum, minimum, numpy_sum};
use crate::pairs::{bend_radii, too_tight};

/// A span only slightly longer than a multiple of the spacing gets no extra
/// vertex, so float noise in re-measured lengths does not change the count.
pub const RESAMPLE_SLACK: f64 = 1e-6;

/// Shape change is measured from each vertex to the other polyline's edges
/// within this many edges of the same index.
pub const SHAPE_WINDOW: isize = 3;

/// Offsets from edge k of the vertices its bend radius depends on: the
/// turns at both its ends take the vertices on either side.
const BEND_REACH: [isize; 4] = [-1, 0, 1, 2];

/// numpy's `interp` of `x` over the increasing `xp` with values `fp`, at a
/// point within them: the slope's product fused into the sum.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x > xp[last] {
        return fp[last];
    }
    if x < xp[0] {
        return fp[0];
    }
    let j = xp.partition_point(|&a| a <= x) - 1;
    if j == last || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut y = slope.mul_add(x - xp[j], fp[j]);
    if y.is_nan() {
        y = slope.mul_add(x - xp[j + 1], fp[j + 1]);
        if y.is_nan() && fp[j] == fp[j + 1] {
            y = fp[j];
        }
    }
    y
}

/// Each span between fixed vertices resampled to uniform arc-length
/// spacing, the fixed vertices kept exactly, and the new fixed mask. The
/// first vertex starts the result, fixed; vertices after the last fixed
/// one are dropped.
pub fn resample(points: &[V2], fixed: &[bool], spacing: f64) -> (Vec<V2>, Vec<bool>) {
    let anchors: Vec<usize> = (0..fixed.len()).filter(|&k| fixed[k]).collect();
    let mut out = vec![points[0]];
    let mut mask = vec![true];
    for pair in anchors.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let span = &points[start..=end];
        let arc = arc_lengths(span);
        let total = arc[arc.len() - 1];
        let count = ((total / spacing - RESAMPLE_SLACK).ceil() as i64).max(1) as usize;
        let step = total / count as f64;
        let xs: Vec<f64> = span.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = span.iter().map(|p| p.y).collect();
        for k in 1..count {
            // numpy's linspace, from 0: k·step, or k/count·total for a
            // step of 0.
            let target = if step == 0.0 {
                k as f64 / count as f64 * total
            } else {
                k as f64 * step
            };
            out.push(V2::new(
                interp(target, &arc, &xs),
                interp(target, &arc, &ys),
            ));
            mask.push(false);
        }
        out.push(points[end]);
        mask.push(true);
    }
    (out, mask)
}

/// Places where a polyline enters the clearance away from its `anchors`,
/// given the vertices `inside` the clearance and the edges that `cut` into
/// it: each corner cut, an edge with both ends clear that enters the
/// clearance, counts once, and so does each run of consecutive vertices
/// inside the clearance that holds no anchor. A run that holds an anchor in
/// the clearance cannot move away and is not counted.
pub fn count_violations(inside: &[bool], cut: &[bool], anchors: &[bool]) -> usize {
    let cuts = cut
        .iter()
        .enumerate()
        .filter(|&(k, &c)| c && !inside[k] && !inside[k + 1])
        .count();
    let mut runs = 0;
    let mut k = 0;
    while k < inside.len() {
        if !inside[k] {
            k += 1;
            continue;
        }
        let mut anchored = false;
        while k < inside.len() && inside[k] {
            anchored |= anchors[k];
            k += 1;
        }
        if !anchored {
            runs += 1;
        }
    }
    cuts + runs
}

/// The summed length of a polyline's edges, as numpy sums them.
pub fn length(points: &[V2]) -> f64 {
    let lengths: Vec<f64> = points.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
    numpy_sum(&lengths)
}

/// Local symmetric distance between two polylines, mm.
///
/// Vertices sliding along an unchanged shape do not count: resampling to
/// uniform spacing and the energy's indifference to where along the track
/// a vertex sits would otherwise keep moving vertices forever without
/// changing the track.
pub fn shape_change(before: &[V2], after: &[V2]) -> f64 {
    maximum(local_distance(after, before), local_distance(before, after))
}

/// Each vertex's arc length as a fraction of the whole.
fn arc_fractions(points: &[V2]) -> Vec<f64> {
    let arc = arc_lengths(points);
    let total = maximum(arc[arc.len() - 1], MIN_EDGE_LENGTH);
    arc.iter().map(|&a| a / total).collect()
}

/// Largest distance from each point to the polyline's edges near it:
/// within SHAPE_WINDOW edges of the edge at the same arc-length fraction,
/// so the far leg of a hairpin never counts as close.
fn local_distance(points: &[V2], polyline: &[V2]) -> f64 {
    let last_edge = polyline.len() as isize - 2;
    let fractions = arc_fractions(polyline);
    let edge_starts = &fractions[..fractions.len() - 1];
    let mut largest = f64::NEG_INFINITY;
    for (&p, f) in points.iter().zip(arc_fractions(points)) {
        let index = edge_starts.partition_point(|&s| s <= f) as isize - 1;
        let mut nearest = f64::INFINITY;
        for offset in -SHAPE_WINDOW..=SHAPE_WINDOW {
            let edge = (index + offset).max(0).min(last_edge) as usize;
            let start = polyline[edge];
            let delta = polyline[edge + 1] - start;
            let squared = delta.dot(delta);
            let t = (p - start).dot(delta) / maximum(squared, MIN_EDGE_LENGTH * MIN_EDGE_LENGTH);
            let c = clip(t, 0.0, 1.0);
            let closest = V2::new(start.x + c * delta.x, start.y + c * delta.y);
            nearest = minimum(nearest, (p - closest).norm());
        }
        largest = maximum(largest, nearest);
    }
    largest
}

/// What a burst may not do to a coupled section's centreline: bend an edge
/// below `bend_limit` that was not already there where the burst started
/// (`tight_before`).
#[derive(Clone, Debug)]
pub struct SelfCheck {
    pub bend_limit: f64,
    pub tight_before: Vec<bool>,
}

impl SelfCheck {
    pub fn of(original: &[V2], bend_limit: f64) -> Self {
        Self {
            bend_limit,
            tight_before: tight(original, bend_limit),
        }
    }

    /// The vertices to move back, and how many edges newly bent too
    /// tightly.
    pub fn check(&self, candidate: &[V2]) -> (Vec<bool>, usize) {
        let mut back = vec![false; candidate.len()];
        let last = candidate.len() as isize - 1;
        let mut count = 0;
        for (k, (&now, &before)) in tight(candidate, self.bend_limit)
            .iter()
            .zip(&self.tight_before)
            .enumerate()
        {
            if now && !before {
                count += 1;
                for reach in BEND_REACH {
                    back[(k as isize + reach).clamp(0, last) as usize] = true;
                }
            }
        }
        (back, count)
    }
}

fn tight(points: &[V2], limit: f64) -> Vec<bool> {
    bend_radii(points, None)
        .into_iter()
        .map(|r| too_tight(r, limit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(points: &[(f64, f64)]) -> Vec<V2> {
        points.iter().map(|&(x, y)| V2::new(x, y)).collect()
    }

    #[test]
    fn resampling_keeps_the_fixed_vertices_and_spaces_the_rest() {
        let points = line(&[(0.0, 0.0), (0.35, 0.0), (1.0, 0.0)]);
        let (out, fixed) = resample(&points, &[true, false, true], 0.25);
        assert_eq!(fixed, vec![true, false, false, false, true]);
        assert_eq!(out[0], points[0]);
        assert_eq!(out[4], points[2]);
        assert_eq!(out[2], V2::new(0.5, 0.0));
    }

    #[test]
    fn a_span_a_hair_over_the_spacing_gets_no_extra_vertex() {
        let points = line(&[(0.0, 0.0), (0.1 + 1e-9, 0.0)]);
        let (out, _) = resample(&points, &[true, true], 0.1);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_run_inside_that_holds_an_anchor_is_no_violation() {
        let inside = [false, true, true, false, true, false];
        let cut = [false, false, false, false, false];
        assert_eq!(
            count_violations(&inside, &cut, &[false, false, true, false, false, false]),
            1
        );
        // A corner cut by an edge with both ends clear counts once.
        assert_eq!(
            count_violations(&[false; 3], &[true, false], &[false; 3]),
            1
        );
    }

    #[test]
    fn sliding_along_a_shape_changes_nothing() {
        let before = line(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        let after = line(&[(0.0, 0.0), (0.4, 0.0), (2.0, 0.0)]);
        assert!(shape_change(&before, &after) < 1e-15);
        let moved = line(&[(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)]);
        assert_eq!(shape_change(&before, &moved), 0.5);
    }

    #[test]
    fn a_newly_tight_bend_moves_back_the_vertices_around_it() {
        let straight = line(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
        let check = SelfCheck::of(&straight, 2.0);
        let bent = line(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)]);
        let (back, count) = check.check(&bent);
        assert!(count > 0);
        assert!(back.iter().all(|&b| b));
    }
}
