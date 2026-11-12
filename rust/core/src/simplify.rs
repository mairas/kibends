//! Obstacle-aware merging of polylines' segments.
//!
//! A chord around a convex obstacle always deviates toward it, so merging
//! by chord tolerance alone would cut into the clearance at every wrap;
//! each merged segment must also pass the exact edge check of the field
//! (`Field::segments_clear`). Merging runs from the polyline's first
//! vertex: each chord grows while every vertex it merges lies within its
//! own tolerance of it and the chord stays clear, and stops at a fixed
//! vertex (`kept_vertices`). A vertex that keeps a pad or via is fixed, so
//! merging keeps it where the optimizer left it, inside its region.
//!
//! `simplify_chains` simplifies a run's members in turn, each against its
//! static field plus the other members on its layer at their latest
//! shapes, those of the nets it ignores left out, by the rule the optimizer
//! applies (`Member::keeps_from`). Its field holds only the members whose
//! copper can reach the vertices it merges across (`Neighbours::field`):
//! those whose box, around what they show, comes within the widest
//! inflation it keeps from a track it sees of the box around the points
//! asked about. Any
//! other is farther from each of those points, and from each segment
//! between them, than the distance it is kept, so it neither blocks a
//! segment nor makes a distance negative, and the answers are those of a
//! field of every member on the layer. The field built and searched grows
//! only with the members near.
//!
//! A coupled section is simplified as its centreline, as the optimizer
//! moved it, keeping clear of the others as the diff pair's outer extent,
//! and the others see its two tracks offset from its latest shape, as in
//! the optimizer, so that both tracks keep equal parallel segments through
//! every bend. Its direction pins may be merged across only where they lie
//! on the chord, so that its end edges keep the ends' directions and both
//! tracks stay parallel there to the edges they meet. Its tracks' mitred
//! corners at kept vertices next to a merged edge reach farther out than
//! the edges: where one enters the field, the vertices on either side of
//! it are kept and the section merged again (`Neighbours::loose_corners`).

use std::sync::Arc;

use crate::field::Field;
use crate::field::chain::ChainField;
use crate::field::composite::CompositeField;
use crate::geometry::V2;
use crate::members::Member;
use crate::numeric::{clip, maximum, minimum};
use crate::optimize::INSIDE_TOLERANCE;

/// A member where the optimizer left it: its polyline, its fixed vertices,
/// and a section's direction pins, which stay where they started.
pub struct Optimised<'a> {
    pub polyline: &'a [V2],
    pub fixed: &'a [bool],
    pub directions: &'a [V2],
}

/// Each of `members` where the optimizer left it simplified within
/// `tolerance`, in order, each against its static field and the others on
/// its layer that it does not ignore, at their latest shapes. Its direction
/// pins keep within `INSIDE_TOLERANCE` of a chord merged across them.
///
/// Panics unless `optimised` holds one entry per member, each with a fixed
/// flag per vertex, and a moving section's polyline two vertices or more.
pub fn simplify_chains(
    members: &[Member],
    optimised: &[Optimised<'_>],
    tolerance: f64,
) -> Vec<Vec<V2>> {
    assert_eq!(
        members.len(),
        optimised.len(),
        "one optimised shape per member"
    );
    let mut seen = Neighbours::new(members, optimised);
    for (i, given) in optimised.iter().enumerate() {
        let polyline = given.polyline;
        // Each merged edge joins two of these vertices, so it stays in their box.
        let field = seen.field(i, polyline);
        let allowed: Vec<f64> = polyline
            .iter()
            .map(|p| {
                if given.directions.contains(p) {
                    INSIDE_TOLERANCE
                } else {
                    tolerance
                }
            })
            .collect();
        let mut fixed = given.fixed.to_vec();
        let kept = loop {
            let kept = kept_vertices(polyline, &fixed, field.as_ref(), &allowed);
            let loose = seen.loose_corners(i, &kept);
            if !loose.contains(&true) {
                break kept;
            }
            for (f, l) in fixed.iter_mut().zip(loose) {
                *f |= l;
            }
        };
        seen.relocate(i, kept.iter().map(|&k| polyline[k]).collect());
    }
    seen.current
}

/// The smallest and largest coordinates of `points`, NaN where one is.
fn bounds<'p>(points: impl IntoIterator<Item = &'p V2>) -> (V2, V2) {
    let mut low = V2::new(f64::INFINITY, f64::INFINITY);
    let mut high = V2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        low = V2::new(minimum(low.x, p.x), minimum(low.y, p.y));
        high = V2::new(maximum(high.x, p.x), maximum(high.y, p.y));
    }
    (low, high)
}

/// The members as they see each other while simplifying, each at its
/// latest shape.
struct Neighbours<'a> {
    members: &'a [Member],
    optimised: &'a [Optimised<'a>],
    current: Vec<Vec<V2>>,
    /// Per member, what the others see of it now.
    shown: Vec<Vec<Vec<V2>>>,
    /// Per member, the box around what it shows.
    low: Vec<V2>,
    high: Vec<V2>,
    /// Per member, the members on its layer, its own among them, in order.
    mates: Vec<Arc<[usize]>>,
    /// Per member, the widest inflation it keeps from a track on its layer
    /// that it sees (`Member::keeps_from`).
    reach: Vec<f64>,
}

impl<'a> Neighbours<'a> {
    fn new(members: &'a [Member], optimised: &'a [Optimised<'a>]) -> Self {
        let mut layers: Vec<(&str, Vec<usize>)> = Vec::new();
        for (j, member) in members.iter().enumerate() {
            match layers.iter_mut().find(|(layer, _)| *layer == member.layer) {
                Some((_, on)) => on.push(j),
                None => layers.push((&member.layer, vec![j])),
            }
        }
        let layers: Vec<(&str, Arc<[usize]>)> = layers
            .into_iter()
            .map(|(layer, on)| (layer, on.into()))
            .collect();
        let mates: Vec<Arc<[usize]>> = members
            .iter()
            .map(|member| {
                layers
                    .iter()
                    .find(|(layer, _)| *layer == member.layer)
                    .map(|(_, on)| Arc::clone(on))
                    .expect("every member's layer is listed")
            })
            .collect();
        let reach = members
            .iter()
            .zip(&mates)
            .map(|(member, on)| {
                on.iter()
                    .flat_map(|&j| members[j].shown.iter().map(move |t| (j, t)))
                    .filter_map(|(j, track)| member.keeps_from(&members[j], track))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let mut seen = Self {
            members,
            optimised,
            current: optimised.iter().map(|o| o.polyline.to_vec()).collect(),
            shown: Vec::with_capacity(members.len()),
            low: Vec::with_capacity(members.len()),
            high: Vec::with_capacity(members.len()),
            mates,
            reach,
        };
        for (j, given) in optimised.iter().enumerate() {
            let shown = seen.shows(j, given.polyline);
            let (low, high) = bounds(shown.iter().flatten());
            seen.shown.push(shown);
            seen.low.push(low);
            seen.high.push(high);
        }
        seen
    }

    /// What the others see of member j at `points`: a section's tracks.
    fn shows(&self, j: usize, points: &[V2]) -> Vec<Vec<V2>> {
        self.members[j]
            .shown(points)
            .into_iter()
            .map(|part| part.into_owned())
            .collect()
    }

    /// Member j is at `points` from now on.
    fn relocate(&mut self, j: usize, points: Vec<V2>) {
        let shown = self.shows(j, &points);
        (self.low[j], self.high[j]) = bounds(shown.iter().flatten());
        self.shown[j] = shown;
        self.current[j] = points;
    }

    /// Member i's field, with the others as they are now, for questions
    /// about `points` and the segments between them.
    fn field(&self, i: usize, points: &[V2]) -> Arc<dyn Field> {
        let (low, high) = bounds(points);
        let reach = self.reach[i];
        let member = &self.members[i];
        let mut parts: Vec<(&[V2], f64, usize)> = Vec::new();
        for &j in self.mates[i].iter() {
            let near = j != i
                && self.low[j].x <= high.x + reach
                && self.low[j].y <= high.y + reach
                && self.high[j].x >= low.x - reach
                && self.high[j].y >= low.y - reach;
            if !near {
                continue;
            }
            let other = &self.members[j];
            for (track, part) in other.shown.iter().zip(&self.shown[j]) {
                if let Some(inflation) = member.keeps_from(other, track) {
                    parts.push((part, inflation, parts.len()));
                }
            }
        }
        if parts.is_empty() {
            return Arc::clone(&member.field);
        }
        Arc::new(CompositeField::new(vec![
            Arc::clone(&member.field),
            Arc::new(ChainField::from_polylines(parts)),
        ]))
    }

    /// For a moving section, the vertices of its optimised centreline to
    /// keep on either side of each `kept` vertex next to a merged edge
    /// where a track's corner enters its field; none for a chain.
    fn loose_corners(&self, i: usize, kept: &[usize]) -> Vec<bool> {
        let polyline = self.optimised[i].polyline;
        let mut loose = vec![false; polyline.len()];
        let member = &self.members[i];
        let Some(section) = member.section.as_ref().filter(|s| !s.offset.held) else {
            return loose;
        };
        let merged: Vec<bool> = kept.windows(2).map(|w| w[1] - w[0] > 1).collect();
        let beside: Vec<bool> = merged.windows(2).map(|w| w[0] || w[1]).collect();
        if !beside.contains(&true) {
            return loose;
        }
        // Each corner keeps the clearance of a track, half the pair's
        // extent less half a track width inside the centreline's.
        let inside = (member.viewer.width - section.offset.width) / 2.0;
        let at: Vec<V2> = kept.iter().map(|&k| polyline[k]).collect();
        let (a, b) = section.offset.offset(&at);
        let corners = [&a[1..a.len() - 1], &b[1..b.len() - 1]];
        let field = self.field(i, &corners.concat());
        for corner in corners {
            let (distance, _) = field.signed_distance(corner, f64::INFINITY);
            for ((&vertex, &d), &next_to_merged) in kept[1..].iter().zip(&distance).zip(&beside) {
                if next_to_merged && d < -inside - INSIDE_TOLERANCE {
                    loose[vertex - 1] = true;
                    loose[vertex + 1] = true;
                }
            }
        }
        loose
    }
}

/// The vertices of `polyline` that merging keeps, in order, the first and
/// the last always among them: no vertex in `fixed` is merged, and each
/// merged vertex lies within its `tolerances` of the chord that replaces
/// it, which is clear in `field`.
pub fn kept_vertices(
    polyline: &[V2],
    fixed: &[bool],
    field: &dyn Field,
    tolerances: &[f64],
) -> Vec<usize> {
    let mut kept = vec![0];
    let Some(last) = polyline.len().checked_sub(1) else {
        return kept;
    };
    let mut start = 0;
    while start < last {
        let mut end = start + 1;
        while end < last && !fixed[end] && mergeable(polyline, start, end + 1, field, tolerances) {
            end += 1;
        }
        kept.push(end);
        start = end;
    }
    kept
}

/// Whether the chord from vertex `start` to vertex `end` may replace the
/// vertices between them.
fn mergeable(
    polyline: &[V2],
    start: usize,
    end: usize,
    field: &dyn Field,
    tolerances: &[f64],
) -> bool {
    let a = polyline[start];
    let b = polyline[end];
    let chord = b - a;
    let squared = chord.dot(chord).max(f64::MIN_POSITIVE);
    let near = polyline[start + 1..end]
        .iter()
        .zip(&tolerances[start + 1..end])
        .all(|(&p, &tolerance)| {
            let t = clip((p - a).dot(chord) / squared, 0.0, 1.0);
            let off = p - (a + t * chord);
            // A NaN deviation is no deviation beyond the tolerance.
            off.dot(off).sqrt().partial_cmp(&tolerance) != Some(std::cmp::Ordering::Greater)
        });
    near && field.segments_clear(&[a], &[b], 0.0)[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::members::Trace;
    use crate::testing::{Disc, FreeSpace};

    const WIDTH: f64 = 0.2;
    const CLEARANCE: f64 = 0.2;
    const BUMP: f64 = 0.005;
    const TOLERANCE: f64 = 0.05;

    fn chain(net: &str, layer: &str) -> Member {
        let trace = Trace {
            net: net.into(),
            width: WIDTH,
            clearance: CLEARANCE,
        };
        Member {
            layer: layer.into(),
            viewer: trace.clone(),
            ignored: vec![net.into()],
            shown: vec![trace],
            field: Arc::new(FreeSpace),
            width: WIDTH,
            contacts: Vec::new(),
            section: None,
        }
    }

    /// A chain bumped by `BUMP` in its middle, and another of `net` on
    /// `layer` running straight `below` its chord.
    fn simplified_beside(net: &str, layer: &str, below: f64) -> Vec<Vec<V2>> {
        let bumped = [V2::new(0.0, 0.0), V2::new(2.5, BUMP), V2::new(5.0, 0.0)];
        let other = [V2::new(0.0, -below), V2::new(5.0, -below)];
        let ends = [true, false, true];
        let members = [chain("A", "F.Cu"), chain(net, layer)];
        let optimised = [
            Optimised {
                polyline: &bumped,
                fixed: &ends,
                directions: &[],
            },
            Optimised {
                polyline: &other,
                fixed: &[true, true],
                directions: &[],
            },
        ];
        simplify_chains(&members, &optimised, TOLERANCE)
    }

    #[test]
    fn a_neighbour_within_clearance_of_the_chord_keeps_the_vertex() {
        let kept = WIDTH + CLEARANCE;
        assert_eq!(simplified_beside("B", "F.Cu", kept - 0.002)[0].len(), 3);
        assert_eq!(simplified_beside("B", "F.Cu", kept + 0.002)[0].len(), 2);
    }

    #[test]
    fn a_neighbour_of_its_net_or_another_layer_is_no_obstacle() {
        let inside = WIDTH + CLEARANCE - 0.002;
        assert_eq!(simplified_beside("A", "F.Cu", inside)[0].len(), 2);
        assert_eq!(simplified_beside("B", "B.Cu", inside)[0].len(), 2);
    }

    #[test]
    fn a_direction_pin_off_the_chord_is_kept() {
        // BUMP lies within the chord tolerance, so only the direction
        // pin's own tolerance keeps the vertex.
        let bumped = [V2::new(0.0, 0.0), V2::new(2.5, BUMP), V2::new(5.0, 0.0)];
        let ends = [true, false, true];
        let members = [chain("A", "F.Cu")];
        let simplified = |directions: &[V2]| {
            let optimised = [Optimised {
                polyline: &bumped,
                fixed: &ends,
                directions,
            }];
            simplify_chains(&members, &optimised, TOLERANCE)
        };
        assert_eq!(simplified(&[bumped[1]])[0].len(), 3);
        assert_eq!(simplified(&[])[0].len(), 2);
    }

    #[test]
    fn a_straight_route_merges_into_one_segment_and_fixed_vertices_stay() {
        let line: Vec<V2> = (0..6).map(|k| V2::new(f64::from(k), 0.0)).collect();
        let tolerances = vec![0.01; line.len()];
        let free = vec![false; line.len()];
        assert_eq!(
            kept_vertices(&line, &free, &FreeSpace, &tolerances),
            vec![0, 5]
        );
        let mut fixed = free.clone();
        fixed[2] = true;
        assert_eq!(
            kept_vertices(&line, &fixed, &FreeSpace, &tolerances),
            vec![0, 2, 5]
        );
    }

    #[test]
    fn a_vertex_whose_merge_would_break_clearance_is_kept() {
        // A route around a disc: its chord deviates by less than the
        // tolerance but cuts into the disc.
        let route = [V2::new(-2.0, 1.05), V2::new(0.0, 1.1), V2::new(2.0, 1.05)];
        let tolerances = [1.0; 3];
        let free = [false; 3];
        let disc = Disc {
            cx: 0.0,
            cy: 0.0,
            radius: 1.06,
        };
        assert_eq!(
            kept_vertices(&route, &free, &disc, &tolerances),
            vec![0, 1, 2]
        );
        assert_eq!(
            kept_vertices(&route, &free, &FreeSpace, &tolerances),
            vec![0, 2]
        );
    }
}
