//! Measures and checks on a single polyline that the optimizer applies to
//! every member, independent of the problem, in numpy's order of
//! operations (`numeric`), the same on every platform.
//!
//! `subdivide` splits a polyline's long edges, keeping every vertex, which
//! is how a chain is laid out at each level's start and each burst's.
//! `resample` lays each span between fixed vertices out at uniform
//! arc-length spacing, as a coupled section is between bursts.
//! `count_violations` counts
//! where a polyline enters its clearance away from its anchors, which the
//! best state may not add to. `shape_change` is how far a burst moved a
//! shape, not counting vertices sliding along it, which settles a problem.
//! `SelfCheck` is what a burst may not do to a coupled section's
//! centreline: bend an edge newly below the limit. `measure` gives a
//! polyline's length and tightest bend, which the run reports before and
//! after.
//!
//! Its bend radius is the discrete curvature radius l/(4·tan(φ/2)) of the
//! polyline resampled at a spacing with every vertex kept, l the two edges
//! meeting at a vertex and φ their turn, so that a sharp corner reports a
//! radius below the spacing however long its edges are. The turn's arctangent
//! and tangent are the `libm` crate's, so that it is the same on every
//! platform.

use crate::energy::MIN_EDGE_LENGTH;
use crate::energy::polyline::arc_lengths;
use crate::geometry::V2;
use crate::numeric::{clip, interp, maximum, minimum, numpy_sum};
use crate::pairs::{bend_radii, too_tight};

/// A span only slightly longer than a multiple of the spacing gets no extra
/// vertex, so float noise in re-measured lengths does not change the count.
pub const RESAMPLE_SLACK: f64 = 1e-6;

/// Below this turning angle a vertex counts as straight (infinite radius).
pub const STRAIGHT_TURN: f64 = 1e-9;

/// Shape change is measured from each vertex to the other polyline's edges
/// within this many edges of the same index.
pub const SHAPE_WINDOW: isize = 3;

/// Offsets from edge k of the vertices its bend radius depends on: the
/// turns at both its ends take the vertices on either side.
const BEND_REACH: [isize; 4] = [-1, 0, 1, 2];

/// Each span between fixed vertices resampled to uniform arc-length
/// spacing, the fixed vertices kept exactly, and the new fixed mask. The
/// first vertex starts the result, fixed; vertices after the last fixed
/// one are dropped.
pub fn resample(points: &[V2], fixed: &[bool], spacing: f64) -> (Vec<V2>, Vec<bool>) {
    resample_spans(points, fixed, |total| edges_at(total, spacing))
}

/// Each edge split evenly into edges no longer than its span's spacing,
/// every vertex kept where it is, as `resample` with every vertex fixed;
/// and per vertex of the result the vertex of `points` it is, None for one
/// a split added. A span runs between consecutive `anchors`, or the
/// polyline's ends. It takes `spacing`, unless that leaves it fewer than
/// `minimum_free` vertices between its ends: then a spacing that does, its
/// length over one more than `minimum_free`, but never finer than
/// `finest`.
pub fn subdivide(
    points: &[V2],
    anchors: &[bool],
    spacing: f64,
    minimum_free: usize,
    finest: f64,
) -> (Vec<V2>, Vec<Option<usize>>) {
    let last = points.len() - 1;
    let mut out = vec![points[0]];
    let mut source = vec![Some(0)];
    let mut start = 0;
    while start < last {
        let end = (start + 1..=last)
            .find(|&k| anchors[k] || k == last)
            .unwrap_or(last);
        let span = &points[start..=end];
        let lengths: Vec<f64> = span.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
        let edges = |s: f64| -> usize { lengths.iter().map(|&l| edges_at(l, s)).sum() };
        let wanted = (minimum_free + 1).min(edges(finest));
        let here = if edges(spacing) >= wanted {
            spacing
        } else {
            maximum(numpy_sum(&lengths) / wanted as f64, finest)
        };
        let (points, kept) = resample(span, &vec![true; span.len()], here);
        let mut at = start;
        for (p, k) in points.into_iter().zip(kept).skip(1) {
            out.push(p);
            source.push(k.then(|| {
                at += 1;
                at
            }));
        }
        start = end;
    }
    (out, source)
}

/// How many edges a span `total` long gets at `spacing`.
fn edges_at(total: f64, spacing: f64) -> usize {
    ((total / spacing - RESAMPLE_SLACK).ceil() as i64).max(1) as usize
}

/// Each span between fixed vertices resampled to uniform arc-length
/// spacing, as many edges as `edges` gives for its length.
fn resample_spans(
    points: &[V2],
    fixed: &[bool],
    edges: impl Fn(f64) -> usize,
) -> (Vec<V2>, Vec<bool>) {
    let anchors: Vec<usize> = (0..fixed.len()).filter(|&k| fixed[k]).collect();
    let mut out = vec![points[0]];
    let mut mask = vec![true];
    for pair in anchors.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let span = &points[start..=end];
        let arc = arc_lengths(span);
        let total = arc[arc.len() - 1];
        let count = edges(total);
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

/// A polyline's length, mm, and its tightest bend radius at `spacing`,
/// infinite for a straight one.
pub fn measure(polyline: &[V2], spacing: f64) -> (f64, f64) {
    let (resampled, _) = resample(polyline, &vec![true; polyline.len()], spacing);
    (length(polyline), tightest_bend_radius(&resampled))
}

fn tightest_bend_radius(points: &[V2]) -> f64 {
    let edges: Vec<V2> = points.windows(2).map(|w| w[1] - w[0]).collect();
    edges
        .windows(2)
        .filter_map(|pair| {
            let (incoming, outgoing) = (pair[0], pair[1]);
            let turn = libm::atan2(incoming.cross(outgoing), incoming.dot(outgoing)).abs();
            (turn > STRAIGHT_TURN)
                .then(|| (incoming.norm() + outgoing.norm()) / (4.0 * libm::tan(turn / 2.0)))
        })
        .reduce(minimum)
        .unwrap_or(f64::INFINITY)
}

/// Local symmetric distance between two polylines, mm.
///
/// Vertices sliding along an unchanged shape do not count: laying the
/// shapes out again between bursts and the energy's indifference to where
/// along the track a vertex sits would otherwise keep moving vertices
/// forever without changing the track.
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
    fn subdividing_keeps_every_vertex_and_splits_only_long_edges() {
        // A 90° corner with a sub-micrometre jog before it.
        let points = line(&[(0.0, 0.0), (0.25, 0.0), (0.25, 5e-4), (0.25, 0.1)]);
        let (out, source) = subdivide(&points, &[true, false, false, true], 0.1, 0, 0.1);
        let kept: Vec<V2> = out
            .iter()
            .zip(&source)
            .filter_map(|(&p, s)| s.map(|_| p))
            .collect();
        assert_eq!(kept, points);
        let sources: Vec<usize> = source.iter().flatten().copied().collect();
        assert_eq!(sources, vec![0, 1, 2, 3]);
        // The 0.25 mm edge splits in three, the others not at all.
        assert_eq!(out.len(), 6);
        assert_eq!(out[1], V2::new(0.25 / 3.0, 0.0));
        // As `resample` with every vertex fixed.
        assert_eq!(out, resample(&points, &[true; 4], 0.1).0);
    }

    #[test]
    fn a_coarse_span_too_short_keeps_the_minimum_free_vertices() {
        let points = line(&[(0.0, 0.0), (1.0, 0.0)]);
        // At 1.6 the span would be one chord; at 0.2 it keeps four free
        // vertices.
        let (out, source) = subdivide(&points, &[true, true], 1.6, 4, 0.1);
        assert_eq!(source, vec![Some(0), None, None, None, None, Some(1)]);
        assert!((out[1].x - 0.2).abs() < 1e-15);
        // Never finer than the finest spacing.
        let short = line(&[(0.0, 0.0), (0.25, 0.0)]);
        assert_eq!(subdivide(&short, &[true, true], 1.6, 4, 0.1).0.len(), 4);
        // A span long enough at the coarse spacing takes it.
        let long = line(&[(0.0, 0.0), (10.0, 0.0)]);
        assert_eq!(
            subdivide(&long, &[true, true], 1.6, 4, 0.1).0,
            resample(&long, &[true, true], 1.6).0
        );
        // Each span apart: a drawn vertex is free and counts, its two edges
        // split alike, three each; an anchor splits the polyline, and the
        // 4 mm span after it takes five edges where 1.6 gives three.
        let kinked = line(&[(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (5.0, 0.0)]);
        let (out, _) = subdivide(&kinked, &[true, false, true, true], 1.6, 4, 0.1);
        assert_eq!(out.len(), 1 + 6 + 5);
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
    fn a_right_angle_reports_its_discrete_radius_and_a_line_none() {
        let corner = line(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]);
        let (length, radius) = measure(&corner, 1.0);
        assert_eq!(length, 2.0);
        // Two unit edges turning by 90°: 2 / (4·tan 45°).
        assert!((radius - 0.5).abs() < 1e-15);
        let straight = line(&[(0.0, 0.0), (3.0, 0.0)]);
        assert_eq!(measure(&straight, 1.0), (3.0, f64::INFINITY));
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
