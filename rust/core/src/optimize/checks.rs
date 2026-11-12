//! What a burst checks: the checks after L-BFGS compare where the burst
//! ends with where it started, and move back what they reject.
//!
//! An edge that starts clear of static copper and ends crossing it moves
//! its free ends back to the burst's start, and the check repeats, as
//! moving one member's vertex back can put another member's moved edge
//! across it; moved back whole, every member is at its start, clear
//! (`Check::revert`). A junction moves back when any of its branches'
//! edges at it does. The other members are judged track by track and via
//! by via (`seen::EdgesClear::obstacle`): an edge that ends across one it
//! did not cross at the start, or that takes its member deeper inside one
//! than the member's deepest edge was there at the start
//! (`seen::EdgesClear::depth`), moves back with the edge or via it
//! entered, so that an edge across one member is still kept out of the
//! others. Depth is judged by member, not by edge: an edge that takes over
//! part of its neighbour, as a vertex sliding along inside a clearance
//! does, leaves the member's copper and its deepest point where they were,
//! and an edge other than the deepest going deeper while the deepest stays
//! so does not worsen the clearance. An edge across a centreline is as
//! deep inside it as an edge can be, so its depth does not show it sliding
//! along that track; the track's edges it newly enters do. The violation
//! guard on the best state counts the rest (`state`).
//!
//! A section moves as its centreline and is seen as its two tracks, which
//! the others' edge checks see where the burst leaves them; its own edges
//! are checked as the pair's outer extent. A burst also moves back what
//! `polylines::SelfCheck` finds on a section: an edge newly bent below its
//! limit.
//!
//! A moving via is a member whose one vertex is its centre, inflated by
//! `via_chain_inflation` against a member of another net and seen by it
//! so, and against another via by `via_via_inflation`, hole-to-hole alone
//! for one of its net. The chain ends it carries move with it, and their
//! edges are checked as any. The copper of other nets is judged obstacle by
//! obstacle, by the via's nearest distance to each: each obstacle of its
//! static copper, connected copper of one net on one layer as one, each
//! member and each via. A via that ends inside its clearance of one it
//! started the burst clear of, or deeper inside one than it started, moves
//! back with the ends it carries, so that a via inside one is still kept
//! out of the others. What only stops it is judged the same way, obstacle
//! by obstacle of each part: outside where its zones, taken together on
//! each layer where one reaches the via's ring, connect it at the next
//! refill, its own net's stops, and each moving via of its net. On the
//! zones' layers its inflation also holds the zone's clearance and margin,
//! against static copper and members alike, so that the refill reaches the
//! via wherever the others move. The chains a via carries are of its net,
//! so it does not see them, nor they it.
//!
//! A crossing between two members, a via and a member or two vias moves
//! back both sides: a member or via that a via newly or more deeply enters
//! moves back too, the member's edge nearest the via, and a member's edge
//! or a moving via whose copper a member's edge newly or more deeply enters
//! moves back with that edge. A crossing whose moving sides are all back
//! already, as some that start inside something can be, moves the whole
//! burst back.
//!
//! A vertex that keeps a pad or a via that stays is free: it has a stop,
//! the clearance hinge at its distance to the region's edge
//! (`contacts::contact_stops`), which limits without repelling, and the
//! check moves it back if it newly ends outside (`contacts::contacts_left`).
//! Every state a burst leaves, and so the best state, has it inside.

use std::sync::Arc;

use super::problem::Layout;
use super::state::Shared;
use super::{DEEPER_ROUND_OFF, INSIDE_TOLERANCE, Settings};
use crate::geometry::V2;
use crate::numeric::minimum;
use crate::polylines::SelfCheck;
use crate::seen::{EdgesClear, Seen};
use crate::vias::free::{FreeDistances, newly_inside};

/// What one check of a burst found: per member, its edges that newly
/// crossed; per via, the edges at the ends it carries that newly crossed,
/// and the times it ended newly or deeper inside its clearance. A crossing
/// is counted on the side that found it, once, though both sides move
/// back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub members: Vec<usize>,
    pub vias: Vec<usize>,
}

/// What a member's edges cross: each (edge, obstacle) pair as one number,
/// the obstacles numbered below the stride, ascending, and the member's
/// depth in each obstacle at its deepest edge, +inf in those it does not
/// cross.
#[derive(Clone, Debug)]
struct Crossed {
    key: Vec<usize>,
    deepest: Vec<f64>,
}

impl Crossed {
    fn of(edges: &EdgesClear, stride: usize) -> Self {
        let mut deepest = vec![f64::INFINITY; stride];
        for (&o, &d) in edges.obstacle.iter().zip(&edges.depth) {
            deepest[o] = minimum(deepest[o], d);
        }
        let mut key: Vec<usize> = edges
            .edge
            .iter()
            .zip(&edges.obstacle)
            .map(|(&e, &o)| e * stride + o)
            .collect();
        key.sort_unstable();
        Self { key, deepest }
    }

    /// Which crossing pieces of `edges` cross for their edge an obstacle it
    /// did not cross here, or take the member deeper inside one than its
    /// deepest edge was here, beyond round-off.
    ///
    /// An edge newly across an obstacle is new copper inside it even where
    /// the member is deeper elsewhere, so it is judged by edge; depth is
    /// judged by member, since an edge taking over part of its neighbour,
    /// such as a vertex sliding along inside a clearance, leaves the copper
    /// and its deepest point where they were.
    fn new_or_deeper(&self, edges: &EdgesClear, stride: usize) -> Vec<bool> {
        edges
            .edge
            .iter()
            .zip(&edges.obstacle)
            .zip(&edges.depth)
            .map(|((&e, &o), &d)| {
                let new = self.key.binary_search(&(e * stride + o)).is_err();
                let deeper = d < minimum(self.deepest[o] - DEEPER_ROUND_OFF, -INSIDE_TOLERANCE);
                new || deeper
            })
            .collect()
    }
}

fn refs(polylines: &[Vec<V2>]) -> Vec<&[V2]> {
    polylines.iter().map(Vec::as_slice).collect()
}

/// The checks of one burst, measured at its start.
pub struct Check {
    shared: Arc<Shared>,
    layout: Layout,
    contact_of: Vec<Vec<Option<usize>>>,
    seen: Arc<Seen>,
    static_before: Vec<Vec<bool>>,
    crossed_before: Vec<Crossed>,
    stride: usize,
    member_of: Vec<usize>,
    vertex_of: Vec<usize>,
    // Only a vertex that newly leaves its contact moves back, so that
    // moving back always ends.
    left_before: Vec<Vec<bool>>,
    self_checks: Vec<Option<SelfCheck>>,
    free_before: FreeDistances,
    stopped_before: FreeDistances,
    vias_from: usize,
    /// The burst's start, flat.
    start: Vec<f64>,
    /// The first round a check found, which the burst counts.
    pub found: Option<Round>,
}

impl Check {
    /// The checks of a burst of the problem `shared` from `starts` and the
    /// vias at `positions`, laid out as `layout`, which starts at `start`,
    /// with each vertex's contact by `contact_of`; `seen` is what the
    /// members see at `starts`.
    pub fn new(
        shared: Arc<Shared>,
        layout: Layout,
        contact_of: Vec<Vec<Option<usize>>>,
        seen: Arc<Seen>,
        starts: &[Vec<V2>],
        positions: &[V2],
        start: Vec<f64>,
    ) -> Self {
        let members = &shared.setup.energy.members.members;
        let flat_start = shared.flat(starts, positions);
        let static_before = members
            .iter()
            .zip(starts)
            .map(|(m, s)| m.field.segments_clear(&s[..s.len() - 1], &s[1..], 0.0))
            .collect();
        let others_before = seen.segments_clear(&refs(starts), &flat_start, 0.0);
        let stride = seen.obstacle().iter().max().map_or(0, |&o| o + 1);
        let crossed_before = others_before
            .iter()
            .map(|edges| Crossed::of(edges, stride))
            .collect();
        let (member_of, vertex_of) = shared.shown_vertices(starts);
        let left_before = starts
            .iter()
            .enumerate()
            .map(|(i, s)| shared.left(i, s, &contact_of[i]))
            .collect();
        let self_checks = shared
            .setup
            .members
            .iter()
            .zip(starts)
            .map(|(m, s)| m.limit.map(|limit| SelfCheck::of(s, limit)))
            .collect();
        let free_before = shared.via_distances(starts, positions, -INSIDE_TOLERANCE);
        let stopped_before = shared.stop_distances(starts, positions, -INSIDE_TOLERANCE);
        let vias_from = flat_start.len() - 2 * positions.len();
        Self {
            shared,
            layout,
            contact_of,
            seen,
            static_before,
            crossed_before,
            stride,
            member_of,
            vertex_of,
            left_before,
            self_checks,
            free_before,
            stopped_before,
            vias_from,
            start,
            found: None,
        }
    }

    pub fn settings(&self) -> &Settings {
        self.shared.settings()
    }

    /// The burst's start, flat.
    pub fn start(&self) -> &[f64] {
        &self.start
    }

    /// What `Round` counts at `x`, and the coordinates of the variables to
    /// move back. A crossing between two members, a via and a member or
    /// two vias moves back what both its sides can move.
    pub fn crossings(&self, x: &[f64]) -> (Round, Vec<bool>) {
        let shared = &self.shared;
        let layout = &self.layout;
        let index = &layout.index;
        let members = &shared.setup.energy.members.members;
        let (polylines, centres) = layout.unflatten(x);
        let flat = shared.flat(&polylines, &centres);
        let mut back = vec![false; layout.count];
        let mut counts = Vec::with_capacity(polylines.len());
        let mut via_counts = vec![0; centres.len()];
        let others = self.seen.segments_clear(&refs(&polylines), &flat, 0.0);
        let move_edge = |back: &mut [bool], member: usize, vertex: usize| {
            for i in index[member].iter().skip(vertex).take(2).flatten() {
                back[*i] = true;
            }
        };
        for (i, points) in polylines.iter().enumerate() {
            let edges = &others[i];
            let new = self.crossed_before[i].new_or_deeper(edges, self.stride);
            let mut crossing = vec![false; points.len() - 1];
            for (k, &n) in new.iter().enumerate() {
                if n {
                    crossing[edges.edge[k]] = true;
                }
            }
            let now_clear =
                members[i]
                    .field
                    .segments_clear(&points[..points.len() - 1], &points[1..], 0.0);
            for ((c, &before), &clear) in crossing
                .iter_mut()
                .zip(&self.static_before[i])
                .zip(&now_clear)
            {
                *c |= before && !clear;
            }
            for (k, &n) in new.iter().enumerate() {
                if !n {
                    continue;
                }
                let entered = edges.vertex[k];
                // A via's piece is its first vertex in `flat`.
                if entered >= self.vias_from {
                    back[layout.via_index[(entered - self.vias_from) / 2]] = true;
                } else {
                    move_edge(&mut back, self.member_of[entered], self.vertex_of[entered]);
                }
            }
            let mut ends = vec![false; points.len()];
            for (k, &c) in crossing.iter().enumerate() {
                if c {
                    ends[k] = true;
                    ends[k + 1] = true;
                }
            }
            for &(k, end) in &shared.carried_by[i] {
                via_counts[k] += usize::from(ends[end.of(points.len())]);
            }
            let left = shared.left(i, points, &self.contact_of[i]);
            let mut outside = 0;
            for ((e, &l), &before) in ends.iter_mut().zip(&left).zip(&self.left_before[i]) {
                if l && !before {
                    *e = true;
                    outside += 1;
                }
            }
            let mut rejected = crossing.iter().filter(|&&c| c).count() + outside;
            if let Some(check) = &self.self_checks[i] {
                let (own, more) = check.check(points);
                for (e, o) in ends.iter_mut().zip(own) {
                    *e |= o;
                }
                rejected += more;
            }
            counts.push(rejected);
            for (&e, v) in ends.iter().zip(&index[i]) {
                if let (true, Some(v)) = (e, v) {
                    back[*v] = true;
                }
            }
        }
        let mut entering = vec![false; centres.len()];
        for (now, before) in [
            (
                shared.via_distances(&polylines, &centres, -INSIDE_TOLERANCE),
                &self.free_before,
            ),
            (
                shared.stop_distances(&polylines, &centres, -INSIDE_TOLERANCE),
                &self.stopped_before,
            ),
        ] {
            let (fixed, inside) = newly_inside(&now, before, INSIDE_TOLERANCE, DEEPER_ROUND_OFF);
            let judged = &now.by_obstacle;
            for (k, &f) in inside.iter().enumerate() {
                if !f {
                    continue;
                }
                let obstacle = judged.obstacle[k];
                if obstacle < members.len() {
                    move_edge(&mut back, obstacle, judged.vertex[k]);
                } else {
                    back[layout.via_index[obstacle - members.len()]] = true;
                }
                entering[judged.via[k]] = true;
            }
            for (k, &f) in fixed.iter().enumerate() {
                if f {
                    entering[now.fixed.via[k]] = true;
                }
            }
        }
        for (k, &e) in entering.iter().enumerate() {
            if e {
                back[layout.via_index[k]] = true;
                via_counts[k] += 1;
            }
        }
        (
            Round {
                members: counts,
                vias: via_counts,
            },
            back.into_iter().flat_map(|b| [b, b]).collect(),
        )
    }

    /// Moves coordinates of `candidate` back to the burst's start, in
    /// place, until the crossings find no edge newly crossing an obstacle
    /// (`revert_until_clear`), and keeps the first round found.
    pub fn revert(&mut self, candidate: &mut [f64]) -> Reverted {
        let start = std::mem::take(&mut self.start);
        let reverted = revert_until_clear(candidate, &start, |x| {
            let (round, back) = self.crossings(x);
            let count = round.members.iter().sum::<usize>() + round.vias.iter().sum::<usize>();
            if self.found.is_none() {
                self.found = Some(round);
            }
            (count, back)
        });
        self.start = start;
        reverted
    }
}

/// What `revert_until_clear` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reverted {
    /// The coordinates it moved back.
    pub mask: Vec<bool>,
    /// The crossings in the candidate as given.
    pub crossings: usize,
    /// Whether a crossing named nothing more to move back, so that it
    /// moved the whole candidate back.
    pub fell_back: bool,
}

/// Moves coordinates of `candidate` back to `start`, in place, until
/// `crossings` finds no edge newly crossing an obstacle. `crossings` gives
/// the number of crossings and the coordinates to move back for them.
/// Moving back can make a neighbouring edge cross, hence the loop; it ends
/// because the start, judged against itself, crosses nothing. A crossing
/// whose moving coordinates are all back already moves the whole candidate
/// back.
pub fn revert_until_clear(
    candidate: &mut [f64],
    start: &[f64],
    mut crossings: impl FnMut(&[f64]) -> (usize, Vec<bool>),
) -> Reverted {
    let mut mask = vec![false; candidate.len()];
    let mut initial = None;
    let mut fell_back = false;
    loop {
        let (count, mut back) = crossings(candidate);
        let initial = *initial.get_or_insert(count);
        if count == 0 {
            return Reverted {
                mask,
                crossings: initial,
                fell_back,
            };
        }
        for (b, &r) in back.iter_mut().zip(&mask) {
            *b &= !r;
        }
        if !back.iter().any(|&b| b) {
            // A crossing whose moving coordinates are all back already, so
            // that what crosses does not move back. Moving everything back
            // ends the loop, as the start crosses nothing.
            fell_back = true;
            back = mask.iter().map(|&r| !r).collect();
        }
        for (k, &b) in back.iter().enumerate() {
            if b {
                candidate[k] = start[k];
                mask[k] = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossings_naming_nothing_to_move_back_move_all_back() {
        let start = vec![0.0; 4];
        let mut candidate = vec![1.0; 4];
        // Crosses until coordinate 3 is back, but names only coordinate 0.
        let reverted = revert_until_clear(&mut candidate, &start, |x| {
            if x[3] == 0.0 {
                (0, vec![false; 4])
            } else {
                (2, vec![true, false, false, false])
            }
        });
        assert_eq!(reverted.crossings, 2);
        assert!(reverted.fell_back);
        assert_eq!(reverted.mask, vec![true; 4]);
        assert_eq!(candidate, start);
    }

    #[test]
    fn moving_back_what_crosses_ends_once_nothing_does() {
        let start = vec![0.0, 0.0];
        let mut candidate = vec![1.0, 2.0];
        let reverted = revert_until_clear(&mut candidate, &start, |x| {
            if x[1] == 0.0 {
                (0, vec![false, false])
            } else {
                (1, vec![false, true])
            }
        });
        assert_eq!(reverted.mask, vec![false, true]);
        assert!(!reverted.fell_back);
        assert_eq!(candidate, vec![1.0, 0.0]);
    }
}
