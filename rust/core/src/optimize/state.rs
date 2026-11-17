//! One problem of a run: the chains, coupled sections and vias of the
//! layers that moving vias tie together, or of one layer, optimised
//! together; their current shapes, the best state so far, and whether the
//! problem has settled. The chains come first, then the sections, as their
//! centrelines.
//!
//! Layout
//! ------
//! A problem is built at its inputs (`Shared::inputs`): each member's drawn
//! vertices, every one kept where it is, with each edge longer than the
//! final spacing split evenly (`polylines::subdivide`). Entering the
//! coarsest level lays the drawn vertices out the same way at its spacing,
//! each span between anchors keeping MIN_COARSE_FREE free vertices. So the
//! coarsest level, and each unit a level starts from the inputs instead
//! (`State::enter`), starts from the drawn geometry: no chord cuts a drawn
//! corner, and a re-run starts from the shape the last run committed. A
//! finer level starts from the coarser one's best state subdivided
//! (`State::bring`), which keeps its geometry.
//!
//! Each burst starts from the shapes laid out again (`State::simplified`,
//! then `State::bring`): a member is simplified within SIMPLIFY_FRACTION of
//! the settled change, as the run's last simplification merges, and
//! across its edges shorter than MERGE_EDGE, each merge clear of what it
//! sees and of the moving vias' rings where they are, then its edges
//! longer than the level's spacing are split evenly. So the vertices the
//! bursts bunched or left on a straight go: a chain carries no short-edge
//! barrier, its free vertices may bunch within a burst, and a chain that
//! shortens loses vertices here rather than crumpling. The
//! simplification moves a shape by no more than its tolerance, a merge
//! across a short edge by up to that edge, and a split not at all; a
//! burst's change counts from the shapes so laid out, so that laying out
//! alone never keeps a problem from settling.
//!
//! A coupled section is laid out as a chain is, its centreline from track
//! a's drawn vertices, but its fold length (`members::Section::layout_limits`)
//! takes MERGE_EDGE's place, its short-edge barrier starts there
//! (`energy::polyline::shape_energy`), and no split leaves an edge shorter
//! than it (`polylines::subdivide`). Its
//! tracks are its centreline's mitred offsets, whose inner side folds at an
//! edge shorter than that between turns as sharp as its bend-radius limit
//! allows at the spacing. The fold length is always above half the spacing,
//! so a section's edges run from it to under twice it, longer than the
//! spacing where splitting would make them shorter, and longer still beside
//! a turn too sharp for pieces that short (below).
//!
//! Nor does laying a section out bend an edge below its limit that was not
//! below it already, as `polylines::SelfCheck` judges a burst, or one that
//! was any tighter: merging keeps the vertices around an edge it would
//! (`simplify::simplified_vertices`), and a split leaves the piece beside a
//! turn long enough (`polylines::subdivide`). A split vertex does not turn, so a piece beside
//! a turn is tighter than the edge it came from; a merge moves the turn of
//! the vertex it drops onto its neighbours. Either would start a burst with
//! the inner track folded where `SelfCheck` does not look, and the burst's
//! repair would count as its change, so that the problem could never
//! settle on it.
//!
//! The shapes as a burst leaves them are what the energy, the guard and the
//! best state see.
//!
//! The best state is limited to no more violations per member than the
//! inputs (`polylines::count_violations`), and to no via newly or deeper
//! inside its clearance of any obstacle, or its stops, than it started
//! (`vias::free::newly_inside`), each obstacle judged apart. Each problem
//! keeps its best state, taken only from a burst that does not raise its
//! energy and passes that guard, at a coarse level unit by unit. Laying
//! out renumbers the edges, so the guard compares counts, not edges.
//!
//! The run refuses inputs inside a clearance by the rules a user can see
//! (`Shared::insides`, judged again by Python's `kibends.inside`), but for
//! a run inside that holds an anchor, which `count_violations` does not
//! count, an arc's chords on its concave side within the arc's offset
//! from them, and what is inside only the optimizer's own keep-outs: a
//! section's widened stand-in, a via's zone clearance and margin. What
//! starts inside so moves like any other: the clearance hinge pushes it
//! out. No burst takes another member or a via newly or deeper inside
//! than it starts, and no burst takes an edge newly across static copper;
//! one that starts across static copper may go deeper into it, which only
//! the guard, measured at the inputs, then limits (`checks`).
//!
//! A section held within the run (`State::go_on_from`) builds its problem
//! again, which measures its guard and its energy with the sections held, so the best state is never compared with one that
//! had them moving.
//!
//! Coarse levels
//! -------------
//! With coarse spacings (`Settings::coarse`), the chains run at each coarse
//! spacing in turn, coarsest first, then at the final one, a level handing
//! over when it settles. A problem measures its guard on its inputs at the
//! final spacing and starts from its inputs at the coarsest spacing, where
//! its first step enters the coarsest level before its burst
//! (`State::unstarted`). A section stays at the
//! final spacing throughout. A 45 mm track at 0.1 mm has about 450
//! vertices, and its bow across the whole span is about N² softer than one
//! vertex, so a few coarse vertices straighten it in a few bursts where the
//! final spacing needs a hundred. Each level's stiffness reference takes
//! its spacing.
//!
//! The best state is always one that passes the guard at the current
//! level's spacing, and energies are compared only within a level
//! (`State::enter`). Entering the next, finer level brings the best state
//! there (`State::bring`): subdivided, keeping every vertex, so the
//! geometry does not change. What fails the guard there starts from the
//! inputs laid out at the level instead, unit by unit (`layout::units`).
//! The energy and the stall history start over there.
//!
//! Entering the final level takes the coarse levels' best state,
//! subdivided, only if some coarse burst ended normally and it passes the
//! guard; else it takes the inputs (`State::final_state`). So a run none of
//! whose coarse bursts ended normally does not end at a layout of its
//! inputs that no burst reached. A burst ends normally here when L-BFGS
//! did not end it abnormally and, if a stop or the time limit cut it, it
//! accepted an iteration first: one cut before that is where it started.
//!
//! At a coarse level each unit is kept apart: after a burst, a unit that
//! fails the guard goes back to its best shape and the others keep theirs
//! (`State::merge`), and what is left becomes the best state if its energy
//! did not rise, so that a bus pressed into clearance does not void a free
//! track's progress. A lone chain put back is held there for the rest of
//! the level, and the change that settles the level is that of what it
//! kept. The final level keeps the whole-problem rule. A run stopped at a
//! coarse level ends at the final spacing by the same rule as entering the
//! final level (`State::finish`). A time limit ends a coarse level as
//! settling does (`State::hand_over`), so by the same rules: a level the
//! limit ends after no normal burst hands over a state no descent reached,
//! and the final level then starts from the inputs unless some earlier
//! coarse burst ended normally.
//!
//! The checks judge where a burst leaves the shapes, not the way there, so
//! at a coarse level, where few vertices take long steps, a chain may pass
//! through another moving member in one burst and settle on its far side
//! where that route is shorter and keeps clearance, as the long arch of the
//! drape scenario (`tests/scenarios.py`) does past the short one under it,
//! or a loop across a pad it went round. A re-run on converged output runs
//! the coarse levels again, and keeps what they reach only if it costs no
//! more than its inputs.
//!
//! The first burst at a level a coarser one handed over to settles it if
//! it moves nothing, even if L-BFGS ended it abnormally, provided the
//! coarser level descended: one of its bursts ended normally, or its own
//! first burst settled it so. It starts from where that descent left off,
//! and a vertex resting on a kink of the energy, such as a chain end at a
//! pad's corner, fails the first line search of every burst from there, so
//! the coarser level may well have given up there. A level none of whose
//! bursts ended normally hands over a state no descent reached, and the
//! next level needs a streak of its own to give up: a run whose every
//! burst ends abnormally gives up at the final level, as at one spacing.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use super::checks::{Check, Round};
use super::layout::{self, units};
use super::problem::{Layout, Lookups, Objective, Problem};
use super::{
    BurstReport, Bursts, DEEPER_ROUND_OFF, End, Expired, INSIDE_TOLERANCE, MAX_ABNORMAL_BURSTS,
    NOTHING_MOVED, NotFinite, Notice, Settings, asked, stalled,
};
use crate::contacts::contacts_left;
use crate::field::Field;
use crate::field::chain::ChainField;
use crate::geometry::V2;
use crate::numeric::{python_max, python_min, python_sum};
use crate::polylines::{self, LayoutLimits, count_violations, shape_change, subdivide};
use crate::seen::Seen;
use crate::simplify::{Merging, Optimised, simplified_vertices};
use crate::vias::free::{FreeDistances, free_distances, newly_inside};
use crate::vias::sight::ViaSight;

/// Whose copper is inside a clearance (`Shared::insides`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    Member(usize),
    Via(usize),
}

/// Each distinct `Against` of `entered` with its deepest depth, in the
/// order first met.
fn deepest_of_each<'a>(
    entered: impl IntoIterator<Item = &'a (f64, Against)>,
) -> Vec<(f64, Against)> {
    let mut out: Vec<(f64, Against)> = Vec::new();
    for &(d, against) in entered {
        match out.iter_mut().find(|(_, a)| *a == against) {
            Some(seen) if d < seen.0 => seen.0 = d,
            Some(_) => {}
            None => out.push((d, against)),
        }
    }
    out
}

/// What copper is inside the clearance of (`Shared::insides`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Against {
    /// The static copper, holes and keepouts of its field, or the board
    /// edge.
    Static,
    Member(usize),
    Via(usize),
    /// What only stops a via: outside where its zones connect it, or its
    /// net's stops.
    Stops,
}

/// A place where copper is inside a clearance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inside {
    pub subject: Subject,
    /// The member's vertex, `from` and `to` alike, or its edge; a via's
    /// centre.
    pub from: V2,
    pub to: V2,
    /// The member's copper the place stands for, from its first point to
    /// its last along it: a vertex's run inside with the clear vertex on
    /// either side, or the edge; a via's centre.
    pub run_from: V2,
    pub run_to: V2,
    /// Its signed distance, below the tolerance; NaN for an edge
    /// cutting into static copper, which the field does not measure.
    pub depth: f64,
    pub against: Against,
}

/// A member of a problem as the run takes it, beside what its energy takes
/// (`problem::Problem::members`).
#[derive(Clone, Debug)]
pub struct MemberSetup {
    /// Its centreline as given.
    pub original: Vec<V2>,
    /// Its pinned vertices.
    pub anchors: Vec<bool>,
    /// Per contact it keeps, the vertex that keeps it.
    pub contacts: Vec<usize>,
    /// Whether it is pinned throughout and never laid out again.
    pub held: bool,
}

/// A moving via of a problem as the run takes it.
#[derive(Clone)]
pub struct ViaSetup {
    pub start: V2,
    /// The chain ends it carries: the member, which end, and the end's
    /// offset from the via's centre.
    pub carried: Vec<(usize, End, V2)>,
    /// What stands still that it keeps its clearance from.
    pub copper: Arc<dyn Field>,
    /// What only stops it, in parts, each judged apart.
    pub stop_parts: Vec<Option<Arc<dyn Field>>>,
}

/// What a problem is built from.
pub struct Setup {
    pub energy: Arc<Problem>,
    pub members: Vec<MemberSetup>,
    pub vias: Vec<ViaSetup>,
    /// The loose chain ends that share a point, by member.
    pub groups: Vec<Vec<(usize, End)>>,
    /// What the vias' clearance checks see of the members and of each
    /// other.
    pub sight: ViaSight,
    pub settings: Settings,
}

/// What does not change in a problem's run.
pub struct Shared {
    pub setup: Setup,
    /// Per member, its loose ends.
    pub loose: Vec<Vec<End>>,
    /// Per member, the vias that carry its ends, and which.
    pub carried_by: Vec<Vec<(usize, End)>>,
    /// Per member, each vertex's contact as given, None for none.
    pub given_contact_of: Vec<Vec<Option<usize>>>,
    /// Per via, the chain ends it carries.
    pub carried: Vec<Vec<(usize, End, V2)>>,
}

/// The members' shapes and the vias' centres of a problem.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub points: Vec<Vec<V2>>,
    pub masks: Vec<Vec<bool>>,
    /// Per member, each vertex's contact among its member's, None for none.
    pub contact_of: Vec<Vec<Option<usize>>>,
    pub positions: Vec<V2>,
}

impl Shape {
    /// Member `i` of `other` in place of its own.
    fn take_member(&mut self, i: usize, other: &Shape) {
        self.points[i] = other.points[i].clone();
        self.masks[i] = other.masks[i].clone();
        self.contact_of[i] = other.contact_of[i].clone();
    }
}

fn refs(polylines: &[Vec<V2>]) -> Vec<&[V2]> {
    polylines.iter().map(Vec::as_slice).collect()
}

impl Shared {
    fn count(&self) -> usize {
        self.setup.members.len()
    }

    pub fn settings(&self) -> &Settings {
        &self.setup.settings
    }

    /// What each member sees of the others and of the vias with the
    /// members at `polylines`.
    pub fn seeing(&self, polylines: &[Vec<V2>]) -> Seen {
        self.setup.energy.seen(&refs(polylines))
    }

    /// What the others see of every member, then each via's centre twice.
    pub fn flat(&self, polylines: &[Vec<V2>], positions: &[V2]) -> Vec<V2> {
        self.setup
            .energy
            .members
            .shown_flat(&refs(polylines), positions)
    }

    /// For each vertex the members at `polylines` show the others, its
    /// member and its vertex there, in the order of `flat`: a section's
    /// tracks have a vertex for each of its centreline's, a held section's
    /// tracks their own (`Member::shown_lengths`). A held section's
    /// centreline is fixed but for an end a via carries, so a vertex of
    /// its tracks moves back at most that via, and past the centreline's
    /// end nothing.
    pub fn shown_vertices(&self, polylines: &[Vec<V2>]) -> (Vec<usize>, Vec<usize>) {
        let mut member = Vec::new();
        let mut vertex = Vec::new();
        for (i, (m, points)) in self
            .setup
            .energy
            .members
            .members
            .iter()
            .zip(polylines)
            .enumerate()
        {
            for length in m.shown_lengths(points) {
                member.extend(std::iter::repeat_n(i, length));
                vertex.extend(0..length);
            }
        }
        (member, vertex)
    }

    /// Each moving via's free distance to each obstacle of the copper of
    /// other nets: its static copper, and those below `below` of the
    /// members near it at `polylines` and the vias of other nets at
    /// `positions`.
    pub fn via_distances(
        &self,
        polylines: &[Vec<V2>],
        positions: &[V2],
        below: f64,
    ) -> FreeDistances {
        let statics: Vec<Vec<Option<Arc<dyn Field>>>> = self
            .setup
            .vias
            .iter()
            .map(|v| vec![Some(v.copper.clone())])
            .collect();
        free_distances(
            &statics,
            &self.setup.sight,
            &refs(polylines),
            positions,
            below,
        )
    }

    /// Each moving via's distance to each of what only stops it: each part
    /// of its stops, and each of the other vias of its net at `positions`
    /// nearer than `below`.
    pub fn stop_distances(
        &self,
        polylines: &[Vec<V2>],
        positions: &[V2],
        below: f64,
    ) -> FreeDistances {
        let statics: Vec<Vec<Option<Arc<dyn Field>>>> = self
            .setup
            .vias
            .iter()
            .map(|v| v.stop_parts.clone())
            .collect();
        free_distances(
            &statics,
            &self.setup.energy.stop,
            &refs(polylines),
            positions,
            below,
        )
    }

    /// Member i's vertices at `points` that lie outside the contact they
    /// keep by `contact_of`.
    pub fn left(&self, i: usize, points: &[V2], contact_of: &[Option<usize>]) -> Vec<bool> {
        let contacts = &self.setup.energy.members.members[i].contacts;
        if contacts.is_empty() {
            return vec![false; points.len()];
        }
        let within: Vec<&dyn Field> = contacts.iter().map(Arc::as_ref).collect();
        contacts_left(points, contact_of, &within, INSIDE_TOLERANCE)
    }

    /// `masks` with the ends the moving vias carry loose.
    pub fn energy_masks(&self, masks: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let mut loose = masks.to_vec();
        for (mask, carried) in loose.iter_mut().zip(&self.carried_by) {
            for &(_, end) in carried {
                let at = end.of(mask.len());
                mask[at] = false;
            }
        }
        loose
    }

    /// The spacing of `level` of the schedule: its coarse spacings, then
    /// the final one.
    pub fn spacing_at(&self, level: usize) -> f64 {
        let settings = self.settings();
        settings
            .coarse
            .as_slice()
            .get(level)
            .copied()
            .unwrap_or(settings.spacing)
    }

    /// The final level of the schedule.
    pub fn final_level(&self) -> usize {
        self.settings().coarse.as_slice().len()
    }

    /// The spacing member `i` is laid out at with the chains at `spacing`:
    /// a section's is always the final spacing, since its bend-radius limit
    /// and direction pins are laid out at it.
    fn member_spacing(&self, i: usize, spacing: f64) -> f64 {
        if self.setup.energy.members.members[i].section.is_some() {
            self.settings().spacing
        } else {
            spacing
        }
    }

    /// What laying member `i` out keeps if it is a section with a
    /// bend-radius limit, whether or not its energy has the hinge
    /// (`members::Section::layout_limits`).
    fn layout_limits(&self, i: usize) -> Option<LayoutLimits> {
        self.setup.energy.members.members[i].layout_limits()
    }

    /// Member `i` at `points`, its pinned vertices `mask` and contacts
    /// `contact_of`, laid out by `polylines::subdivide` at `spacing`, its
    /// spans between the pinned vertices, its loose ends and the vertices
    /// that keep a contact, each with at least `minimum_free` free vertices
    /// above the final spacing, and a section within its layout limits; and
    /// the new mask and contacts.
    fn lay_out(
        &self,
        i: usize,
        points: &[V2],
        mask: &[bool],
        contact_of: &[Option<usize>],
        spacing: f64,
        minimum_free: usize,
    ) -> (Vec<V2>, Vec<bool>, Vec<Option<usize>>) {
        let anchors = anchors(mask, contact_of, &self.loose[i]);
        let (points, source) = subdivide(
            points,
            &anchors,
            spacing,
            minimum_free,
            self.settings().spacing,
            self.layout_limits(i),
        );
        let fixed = source.iter().map(|s| s.is_some_and(|k| mask[k])).collect();
        let moved = source
            .iter()
            .map(|s| s.and_then(|k| contact_of[k]))
            .collect();
        (points, fixed, moved)
    }

    /// The inputs with the chains laid out at `spacing`: each member's
    /// drawn vertices, all kept, its long edges subdivided at its spacing
    /// (`Shared::lay_out`), and the vias at their starts. A member pinned
    /// throughout stays as given.
    pub fn inputs(&self, spacing: f64) -> Shape {
        let count = self.count();
        let mut points = Vec::with_capacity(count);
        let mut masks = Vec::with_capacity(count);
        let mut contact_of = Vec::with_capacity(count);
        for (i, (m, given)) in self
            .setup
            .members
            .iter()
            .zip(&self.given_contact_of)
            .enumerate()
        {
            let here = self.member_spacing(i, spacing);
            let (p, mask, c) =
                self.lay_out(i, &m.original, &m.anchors, given, here, MIN_COARSE_FREE);
            // Subdividing would put free vertices between a held member's
            // pins.
            if m.held || mask.iter().all(|&f| f) {
                points.push(m.original.clone());
                masks.push(m.anchors.clone());
                contact_of.push(given.clone());
            } else {
                points.push(p);
                masks.push(mask);
                contact_of.push(c);
            }
        }
        Shape {
            points,
            masks,
            contact_of,
            positions: self.setup.vias.iter().map(|v| v.start).collect(),
        }
    }

    /// Per member, the rings of the moving vias at `positions` that it
    /// keeps its centreline out of (`problem::Problem::rings`), None for
    /// none.
    fn via_rings(&self, positions: &[V2]) -> Vec<Option<Arc<dyn Field>>> {
        let mut centres = vec![Vec::new(); self.count()];
        let mut rings = vec![Vec::new(); self.count()];
        for (&at, near) in positions.iter().zip(&self.setup.energy.rings) {
            for &(i, ring) in near {
                centres[i].push(at);
                rings[i].push(ring);
            }
        }
        centres
            .iter()
            .zip(rings)
            .map(|(centres, rings)| {
                // Zero-length edges, each via an obstacle of its own.
                (!centres.is_empty()).then(|| {
                    let obstacles = (0..centres.len()).collect();
                    Arc::new(ChainField::new(centres, centres, rings, obstacles)) as Arc<dyn Field>
                })
            })
            .collect()
    }

    /// The problem's energy at `shape`, with what the members see of each
    /// other at its points given as `seen`.
    pub fn energy_of(&self, shape: &Shape, seen: &Seen) -> Result<f64, NotFinite> {
        let masks = self.energy_masks(&shape.masks);
        self.setup
            .energy
            .total(
                &refs(&shape.points),
                &shape.positions,
                &masks.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &shape
                    .contact_of
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>(),
                seen,
                None,
            )
            .map(|e| e.value)
            .map_err(NotFinite)
    }

    /// The summed length of the selected tracks at `polylines`, a
    /// section's as its two tracks, mm.
    pub fn length(&self, polylines: &[Vec<V2>]) -> f64 {
        python_sum(
            self.setup
                .energy
                .members
                .members
                .iter()
                .zip(polylines)
                .flat_map(|(m, points)| {
                    m.shown(points)
                        .into_iter()
                        .map(|track| polylines::length(&track))
                        .collect::<Vec<_>>()
                }),
        )
    }

    /// Per member at `polylines`, its violations of its static copper, the
    /// other members and the vias at `positions` together
    /// (`polylines::count_violations`), every member's at once through
    /// `seen`, what they see there.
    pub fn violations(
        &self,
        polylines: &[Vec<V2>],
        masks: &[Vec<bool>],
        positions: &[V2],
        seen: &Seen,
    ) -> Vec<usize> {
        let flat = self.flat(polylines, positions);
        let queries = refs(polylines);
        let near = seen.distances(&queries, &flat, -INSIDE_TOLERANCE);
        let clear = seen.segments_clear(&queries, &flat, INSIDE_TOLERANCE);
        let mut counts = Vec::with_capacity(polylines.len());
        for (((member, points), mask), (distance, others)) in self
            .setup
            .energy
            .members
            .members
            .iter()
            .zip(polylines)
            .zip(masks)
            .zip(near.iter().zip(&clear))
        {
            let field = &member.field;
            let (fixed, _) = field.signed_distance(points, f64::INFINITY);
            let inside: Vec<bool> = fixed
                .iter()
                .zip(distance)
                .map(|(&d, &n)| d < -INSIDE_TOLERANCE || n < -INSIDE_TOLERANCE)
                .collect();
            let mut cut: Vec<bool> = others.clear.iter().map(|&c| !c).collect();
            let ends_clear: Vec<usize> = (0..cut.len())
                .filter(|&k| !inside[k] && !inside[k + 1])
                .collect();
            let starts: Vec<V2> = ends_clear.iter().map(|&k| points[k]).collect();
            let ends: Vec<V2> = ends_clear.iter().map(|&k| points[k + 1]).collect();
            let static_clear = field.segments_clear(&starts, &ends, INSIDE_TOLERANCE);
            for (&k, &c) in ends_clear.iter().zip(&static_clear) {
                cut[k] |= !c;
            }
            counts.push(count_violations(&inside, &cut, mask));
        }
        counts
    }

    /// Each place where the copper of a member at `polylines` or of a via
    /// at `positions` is inside a clearance: for each violation
    /// `violations` counts, a run inside at its deepest vertex or an edge
    /// cutting in, one for each class of copper it enters, static copper and
    /// each member or via its edges enter, each with its own depth; then one
    /// for each obstacle a via is inside or each of its stops it is outside
    /// of, a crossing of a member and a via, or of two vias, once. A run
    /// that holds a pinned vertex counts as any other, though `violations`
    /// lets it be in a burst: a run that starts inside a clearance would be
    /// free to move into another. A pin on copper of its own nets that its
    /// field holds, such as a section's end on its pair's via, lies inside
    /// that copper's clearance by design, and is no violation by the rules
    /// the caller judges each place by. Inside means by more than
    /// `tolerance`. `seen` is what the members see at `polylines`.
    pub fn insides(
        &self,
        polylines: &[Vec<V2>],
        positions: &[V2],
        seen: &Seen,
        tolerance: f64,
    ) -> Vec<Inside> {
        let count = self.count();
        let flat = self.flat(polylines, positions);
        let queries = refs(polylines);
        let near = seen.distances(&queries, &flat, -tolerance);
        let clear = seen.segments_clear(&queries, &flat, tolerance);
        let (member_of, _) = self.shown_vertices(polylines);
        let vias_from = flat.len() - 2 * positions.len();
        // A via's piece is its first vertex in `flat`.
        let piece = |v: usize| {
            if v >= vias_from {
                Against::Via((v - vias_from) / 2)
            } else {
                Against::Member(member_of[v])
            }
        };
        let mut found = Vec::new();
        for (i, ((member, points), (distance, others))) in self
            .setup
            .energy
            .members
            .members
            .iter()
            .zip(polylines)
            .zip(near.iter().zip(&clear))
            .enumerate()
        {
            let field = &member.field;
            let (fixed, _) = field.signed_distance(points, f64::INFINITY);
            let depth: Vec<f64> = fixed
                .iter()
                .zip(distance)
                .map(|(&d, &n)| python_min(d, n))
                .collect();
            let inside: Vec<bool> = depth.iter().map(|&d| d < -tolerance).collect();
            let mut entering: Vec<Vec<(f64, Against)>> = vec![Vec::new(); points.len() - 1];
            for ((&e, &v), &d) in others.edge.iter().zip(&others.vertex).zip(&others.depth) {
                entering[e].push((d, piece(v)));
            }
            let mut k = 0;
            while k < inside.len() {
                if !inside[k] {
                    k += 1;
                    continue;
                }
                let first = k;
                while k < inside.len() && inside[k] {
                    k += 1;
                }
                let at = (first..k)
                    .reduce(|a, b| if depth[b] < depth[a] { b } else { a })
                    .unwrap_or(first);
                let run_from = points[first.saturating_sub(1)];
                let run_to = points[k.min(points.len() - 1)];
                let place = |depth: f64, against: Against| Inside {
                    subject: Subject::Member(i),
                    from: points[at],
                    to: points[at],
                    run_from,
                    run_to,
                    depth,
                    against,
                };
                // Static copper by its deepest vertex in the run, and each
                // member or via the run's edges enter by its deepest, so
                // that no one hides another.
                let fixed_at = (first..k)
                    .reduce(|a, b| if fixed[b] < fixed[a] { b } else { a })
                    .unwrap_or(first);
                if fixed[fixed_at] < -tolerance {
                    found.push(place(fixed[fixed_at], Against::Static));
                }
                let edges = first.saturating_sub(1)..k.min(points.len() - 1);
                for (d, against) in deepest_of_each(edges.flat_map(|e| &entering[e])) {
                    found.push(place(d, against));
                }
            }
            let ends_clear: Vec<usize> = (0..points.len() - 1)
                .filter(|&k| !inside[k] && !inside[k + 1])
                .collect();
            let starts: Vec<V2> = ends_clear.iter().map(|&k| points[k]).collect();
            let ends: Vec<V2> = ends_clear.iter().map(|&k| points[k + 1]).collect();
            let static_clear = field.segments_clear(&starts, &ends, tolerance);
            for (&k, &c) in ends_clear.iter().zip(&static_clear) {
                let mut classes = Vec::new();
                if !c {
                    classes.push((f64::NAN, Against::Static));
                }
                if !others.clear[k] {
                    classes.extend(deepest_of_each(&entering[k]));
                }
                for (depth, against) in classes {
                    found.push(Inside {
                        subject: Subject::Member(i),
                        from: points[k],
                        to: points[k + 1],
                        run_from: points[k],
                        run_to: points[k + 1],
                        depth,
                        against,
                    });
                }
            }
        }
        let mut crossed: BTreeSet<(usize, usize)> = found
            .iter()
            .filter_map(|f| match (f.subject, f.against) {
                (Subject::Member(i), Against::Via(k)) => Some((i, count + k)),
                _ => None,
            })
            .collect();
        for (stops, now) in [
            (false, self.via_distances(polylines, positions, -tolerance)),
            (true, self.stop_distances(polylines, positions, -tolerance)),
        ] {
            for (&k, &d) in now.fixed.via.iter().zip(&now.fixed.distance) {
                if d < -tolerance {
                    found.push(Inside {
                        subject: Subject::Via(k),
                        from: positions[k],
                        to: positions[k],
                        run_from: positions[k],
                        run_to: positions[k],
                        depth: d,
                        against: if stops {
                            Against::Stops
                        } else {
                            Against::Static
                        },
                    });
                }
            }
            let judged = &now.by_obstacle;
            for ((&k, &o), &d) in judged
                .via
                .iter()
                .zip(&judged.obstacle)
                .zip(&judged.distance)
            {
                let pair = if o < count {
                    (o, count + k)
                } else {
                    (count + k.min(o - count), count + k.max(o - count))
                };
                if d >= -tolerance || !crossed.insert(pair) {
                    continue;
                }
                found.push(Inside {
                    subject: Subject::Via(k),
                    from: positions[k],
                    to: positions[k],
                    run_from: positions[k],
                    run_to: positions[k],
                    depth: d,
                    against: if o < count {
                        Against::Member(o)
                    } else {
                        Against::Via(o - count)
                    },
                });
            }
        }
        found
    }

    /// Each (member, member or via) whose clearance the member's edges at
    /// `polylines` enter, the vias at `positions`, numbered after the
    /// members. `seen` is what the members see at `polylines`.
    pub fn entered(
        &self,
        polylines: &[Vec<V2>],
        positions: &[V2],
        seen: &Seen,
    ) -> BTreeSet<(usize, usize)> {
        let count = self.count();
        let flat = self.flat(polylines, positions);
        let (mut of, _) = self.shown_vertices(polylines);
        for k in 0..positions.len() {
            of.push(count + k);
            of.push(count + k);
        }
        let mut entered = BTreeSet::new();
        for (i, edges) in seen
            .segments_clear(&refs(polylines), &flat, INSIDE_TOLERANCE)
            .iter()
            .enumerate()
        {
            for &v in &edges.vertex {
                entered.insert((i, of[v]));
            }
        }
        entered
    }
}

/// What a burst of the problem reached and what it moved back.
struct ProblemBurst {
    polylines: Vec<Vec<V2>>,
    positions: Vec<V2>,
    /// Per member, its edges whose moves were undone.
    rejected_edges: Vec<usize>,
    /// Per via, the edges at the ends it carries that newly crossed, and
    /// the times it ended newly inside its clearance.
    via_rejected: Vec<usize>,
    ran: Ran,
}

use super::Ran;

/// One problem's state in a run.
pub struct State {
    shared: Arc<Shared>,
    pub now: Shape,
    /// What the members see of each other at `now`.
    pub seen: Arc<Seen>,
    /// The best state that passes the guard at the current level.
    pub best: Shape,
    /// The inputs laid out at the final spacing (`Shared::inputs`): what
    /// the guard was measured on, so it always passes it.
    pub input: Shape,
    /// The energy of `now` and of `best`, laid out at the current level's
    /// spacing, compared only within it.
    pub energy: f64,
    pub best_energy: f64,
    /// The energy the run started from, of its inputs laid out at its first
    /// level's spacing.
    pub start_energy: f64,
    /// The level of the schedule it runs at (`Shared::spacing_at`).
    pub level: usize,
    /// Whether it has coarse levels and has taken no step: it is still at
    /// its inputs at the coarsest spacing, where its starting energy and
    /// length are measured, and its first step enters level 0.
    unstarted: bool,
    /// Whether its next burst is the first at a level a coarser one that
    /// descended handed over to.
    handed_over: bool,
    /// Whether the current level descended: one of its bursts ended
    /// normally, or its first burst settled it after a handover.
    descended: bool,
    /// Whether some burst at a coarse level ended normally (as the module
    /// docs count it), without which
    /// the coarse levels' best state is only a layout of what they started
    /// from.
    coarse_descended: bool,
    /// Whether it has run a burst at the level it runs at: not before its
    /// first step, nor since it last entered a level, by settling or
    /// `hand_over`. A problem built again (`go_on_from`) keeps its
    /// earlier one's.
    ran: bool,
    /// Per member, whether it is held at its best shape for the rest of
    /// the coarse level it runs at (`State::step`).
    held: Vec<bool>,
    /// Per member, the violations its inputs have, which the guard holds
    /// it to.
    pub allowed: Vec<usize>,
    inside: FreeDistances,
    stopped: FreeDistances,
    /// The members that move: those with a free vertex and those a via
    /// carries.
    pub free: Vec<usize>,
    pub moving: bool,
    /// Whether some burst changed a shape by more than NOTHING_MOVED.
    pub moved: bool,
    lowest: Vec<f64>,
    lengths: Vec<f64>,
    pub rejected_edges: Vec<usize>,
    pub via_rejected: Vec<usize>,
    pub abnormal_bursts: u32,
    pub abnormal_streak: u32,
    pub settled: bool,
}

/// At a coarse level, the fewest free vertices a span between anchors
/// keeps, so that a short kinked chain between pins never collapses to a
/// chord (`polylines::subdivide`).
pub const MIN_COARSE_FREE: usize = 4;

/// The tolerance a burst's start is simplified within, as a fraction of
/// `Settings::settled_change`: below it, so that the next burst moving the
/// vertices back does not keep the problem from settling.
pub const SIMPLIFY_FRACTION: f64 = 0.5;

/// The shortest edge a burst's start keeps between free vertices, mm:
/// across a shorter one a vertex is merged
/// (`simplify::merge_short_edges`). A chain has no short-edge barrier, so
/// this is what keeps its free vertices from piling up: a chain that
/// shortens loses vertices rather than crumpling them.
pub const MERGE_EDGE: f64 = 0.02;

/// A member's pinned vertices, its `loose` ends and the vertices that keep
/// a contact (`contact_of`): what laying it out keeps where it is.
fn anchors(mask: &[bool], contact_of: &[Option<usize>], loose: &[End]) -> Vec<bool> {
    let mut anchors: Vec<bool> = mask
        .iter()
        .zip(contact_of)
        .map(|(&m, c)| m || c.is_some())
        .collect();
    for end in loose {
        let at = end.of(anchors.len());
        anchors[at] = true;
    }
    anchors
}

/// Bursts each level runs before a rule settles it on how little it
/// changes or gains, and that `levelled_off` looks back over. Giving up, an
/// abnormal first burst that moves nothing at a level a descending one
/// handed over to, and a coarse level with every chain held and no via
/// settle sooner.
pub const LEVEL_WINDOW: usize = 5;
/// The share of the run's energy decrease so far below which a level's
/// last `LEVEL_WINDOW` bursts count as levelled off.
pub const LEVEL_GAIN: f64 = 0.001;

/// Whether a level whose bursts' lowest energies since it began are
/// `lowest`, its first entry where the level began, has levelled off: its
/// last `LEVEL_WINDOW` bursts lowered the energy by less than `LEVEL_GAIN`
/// of how far it has fallen from `start_energy`, where the run began. Measured
/// against the whole run's decrease, it ends a coarse level once its large
/// early gains are past and leaves the finer levels to fine-tuning
/// (https://github.com/mairas/kibends/issues/233).
pub fn levelled_off(lowest: &[f64], start_energy: f64) -> bool {
    let n = lowest.len();
    if n <= LEVEL_WINDOW {
        return false;
    }
    let last = lowest[n - 1];
    let fallen = start_energy - last;
    fallen > 0.0 && lowest[n - 1 - LEVEL_WINDOW] - last < LEVEL_GAIN * fallen
}

impl State {
    /// The problem `setup` makes, at its inputs (`Shared::inputs`); Err if
    /// its energy there is not finite.
    pub fn new(setup: Setup) -> Result<Self, NotFinite> {
        let count = setup.members.len();
        let mut loose = vec![Vec::new(); count];
        for group in &setup.groups {
            for &(i, end) in group {
                loose[i].push(end);
            }
        }
        let mut carried_by = vec![Vec::new(); count];
        for (k, via) in setup.vias.iter().enumerate() {
            for &(i, end, _) in &via.carried {
                carried_by[i].push((k, end));
            }
        }
        let given_contact_of: Vec<Vec<Option<usize>>> = setup
            .members
            .iter()
            .map(|m| {
                let mut contact_of = vec![None; m.original.len()];
                for (k, &v) in m.contacts.iter().enumerate() {
                    contact_of[v] = Some(k);
                }
                for (c, &a) in contact_of.iter_mut().zip(&m.anchors) {
                    if a {
                        *c = None;
                    }
                }
                contact_of
            })
            .collect();
        let carried = setup.vias.iter().map(|v| v.carried.clone()).collect();
        let spacing = setup.settings.spacing;
        let shared = Arc::new(Shared {
            setup,
            loose,
            carried_by,
            given_contact_of,
            carried,
        });
        // The guard is measured on the inputs at the final spacing.
        let input = shared.inputs(spacing);
        let free: Vec<usize> = (0..count)
            .filter(|&i| !input.masks[i].iter().all(|&f| f) || !shared.carried_by[i].is_empty())
            .collect();
        let moving = !free.is_empty() || !shared.setup.vias.is_empty();
        let seen = shared.seeing(&input.points);
        let allowed = shared.violations(&input.points, &input.masks, &input.positions, &seen);
        let inside = shared.via_distances(&input.points, &input.positions, -INSIDE_TOLERANCE);
        let stopped = shared.stop_distances(&input.points, &input.positions, -INSIDE_TOLERANCE);
        let vias = shared.setup.vias.len();
        let level = shared.final_level();
        let unstarted = moving && level > 0;
        // The run starts from the inputs at the spacing of its first level.
        let (now, seen) = if unstarted {
            let now = shared.inputs(shared.spacing_at(0));
            let seen = shared.seeing(&now.points);
            (now, seen)
        } else {
            (input.clone(), seen)
        };
        let energy = shared.energy_of(&now, &seen)?;
        let length = shared.length(&now.points);
        Ok(Self {
            best: now.clone(),
            input,
            now,
            seen: Arc::new(seen),
            energy,
            best_energy: energy,
            start_energy: energy,
            level,
            unstarted,
            handed_over: false,
            descended: false,
            coarse_descended: false,
            ran: false,
            held: vec![false; count],
            allowed,
            inside,
            stopped,
            free,
            moving,
            moved: false,
            lowest: vec![energy],
            lengths: vec![length],
            rejected_edges: vec![0; count],
            via_rejected: vec![0; vias],
            abnormal_bursts: 0,
            abnormal_streak: 0,
            settled: !moving,
            shared,
        })
    }

    pub fn shared(&self) -> &Shared {
        &self.shared
    }

    /// Where its inputs, the drawn copper, are inside a clearance by more
    /// than `tolerance` (`Shared::insides`).
    pub fn start_inside(&self, tolerance: f64) -> Vec<Inside> {
        let input = &self.input;
        let seen = self.shared.seeing(&input.points);
        self.shared
            .insides(&input.points, &input.positions, &seen, tolerance)
    }

    /// The spacing of the level it runs at.
    pub fn level_spacing(&self) -> f64 {
        self.shared.spacing_at(self.level)
    }

    /// Whether it runs at a coarse level.
    pub fn coarse(&self) -> bool {
        self.level < self.shared.final_level()
    }

    /// Whether it has not reached the final level: it runs at a coarse
    /// one, or has taken no step, which leaves `level` at the final one.
    pub fn before_final(&self) -> bool {
        self.unstarted || self.coarse()
    }

    /// Whether it owes a burst at its level, which a time limit waits for:
    /// it has not settled and has run none at a level a handover took it to.
    /// Level 0 is never one: the coarsest is entered at a first step, which
    /// bursts, and a single spacing starts from the inputs.
    pub fn owes_burst(&self) -> bool {
        self.moving && !self.unstarted && self.level > 0 && !self.settled && !self.ran
    }

    /// `shape` with each free member that is not held simplified within
    /// SIMPLIFY_FRACTION of the settled change, keeping its pinned
    /// vertices, loose ends and the vertices that keep a contact, as the
    /// run's last simplification merges them (`simplify`), and merged
    /// across its edges shorter than MERGE_EDGE, a section's than its fold
    /// length (`members::Section::layout_limits`), so that the vertices the
    /// bursts bunched or left on a straight go. Each merged edge keeps
    /// clear of what its member sees and of the rings of the moving vias at
    /// `shape`'s positions.
    fn simplified(&self, shape: &Shape) -> Shape {
        let shared = &self.shared;
        let count = shared.count();
        let fixed: Vec<Vec<bool>> = (0..count)
            .map(|i| {
                if self.free.contains(&i) && !self.held[i] {
                    anchors(&shape.masks[i], &shape.contact_of[i], &shared.loose[i])
                } else {
                    vec![true; shape.points[i].len()]
                }
            })
            .collect();
        let optimised: Vec<Optimised<'_>> = shape
            .points
            .iter()
            .zip(&fixed)
            .map(|(points, fixed)| Optimised {
                polyline: points,
                fixed,
                directions: &[],
            })
            .collect();
        let tolerance = SIMPLIFY_FRACTION * shared.settings().settled_change;
        let merging: Vec<Merging> = (0..count)
            .map(|i| match shared.layout_limits(i) {
                Some(limits) => Merging {
                    shortest: limits.shortest,
                    bend_limit: Some(limits.bend_limit),
                },
                None => Merging {
                    shortest: MERGE_EDGE,
                    bend_limit: None,
                },
            })
            .collect();
        let kept = simplified_vertices(
            &shared.setup.energy.members.members,
            &optimised,
            &shared.via_rings(&shape.positions),
            tolerance,
            &merging,
        );
        let mut simplified = shape.clone();
        for (i, kept) in kept.iter().enumerate() {
            simplified.points[i] = kept.iter().map(|&k| shape.points[i][k]).collect();
            simplified.masks[i] = kept.iter().map(|&k| shape.masks[i][k]).collect();
            simplified.contact_of[i] = kept.iter().map(|&k| shape.contact_of[i][k]).collect();
        }
        simplified
    }

    /// `shape` with each free member's edges longer than `spacing` for it
    /// split evenly, every vertex kept, so that the geometry does not
    /// change and no new edge is shorter than half the spacing, nor a
    /// section's than its fold length.
    fn bring(&self, shape: &Shape, spacing: f64) -> Shape {
        let shared = &self.shared;
        let mut brought = shape.clone();
        for &i in &self.free {
            let here = shared.member_spacing(i, spacing);
            let (p, m, c) = shared.lay_out(
                i,
                &shape.points[i],
                &shape.masks[i],
                &shape.contact_of[i],
                here,
                0,
            );
            brought.points[i] = p;
            brought.masks[i] = m;
            brought.contact_of[i] = c;
        }
        brought
    }

    /// `candidate`, with each unit (`layout::units`) that `failing` names
    /// there, and each it names once those are, taken from `fallback`
    /// instead, until it names none; all of `fallback` if what is left
    /// still fails the guard. `fallback` must pass it. Returns the state,
    /// what the members see there, and per member, then per via, whether
    /// it came from `fallback`.
    fn merge(
        &self,
        candidate: &Shape,
        fallback: &Shape,
        failing: impl Fn(&Shape, &Seen) -> Vec<bool>,
    ) -> (Shape, Seen, Vec<bool>) {
        let shared = &self.shared;
        let count = shared.count();
        let unit = units(count, &shared.setup.groups, &shared.carried);
        let state = |back: &[bool]| -> Shape {
            let mut shape = candidate.clone();
            for i in 0..count {
                if back[unit[i]] {
                    shape.take_member(i, fallback);
                }
            }
            for (k, at) in shape.positions.iter_mut().enumerate() {
                if back[unit[count + k]] {
                    *at = fallback.positions[k];
                }
            }
            shape
        };
        let mut back = vec![false; unit.len()];
        let (mut now, mut seen) = loop {
            let now = state(&back);
            let seen = shared.seeing(&now.points);
            let failing = failing(&now, &seen);
            let mut roots: Vec<usize> = (0..failing.len())
                .filter(|&k| failing[k])
                .map(|k| unit[k])
                .filter(|&r| !back[r])
                .collect();
            roots.sort_unstable();
            roots.dedup();
            if roots.is_empty() {
                break (now, seen);
            }
            for r in roots {
                back[r] = true;
            }
        };
        if !self.guards(&now, &seen) {
            back.fill(true);
            now = state(&back);
            seen = shared.seeing(&now.points);
        }
        let from_fallback = unit.iter().map(|&r| back[r]).collect();
        (now, seen, from_fallback)
    }

    /// `candidate` with each unit that oversteps (`overstepping`) taken
    /// from `fallback` (`merge`).
    fn merge_guarded(&self, candidate: &Shape, fallback: &Shape) -> (Shape, Seen, Vec<bool>) {
        self.merge(candidate, fallback, |s, seen| {
            self.overstepping(&s.points, &s.masks, &s.positions, seen)
        })
    }

    /// Start `level`. A coarse level starts from the best state brought to
    /// its spacing (`bring`), each unit that fails the guard there from the
    /// inputs laid out at it (`merge`, `Shared::inputs`); the first, entered
    /// before any burst, from the inputs laid out at it. The final level
    /// starts from `final_state`. The
    /// best state, the energies and the stall history start there, so that
    /// no energy is compared across spacings, and the abnormal streak
    /// starts over.
    ///
    /// Of the schedule it sets `level`, `handed_over` (whether a coarser
    /// level that descended hands over), `descended` and `ran` (not yet)
    /// and `held` (none), and clears `settled` and `unstarted`;
    /// `coarse_descended` carries.
    pub fn enter(&mut self, level: usize) -> Result<(), NotFinite> {
        let (now, seen) = if level < self.shared.final_level() {
            self.level = level;
            let inputs = self.shared.inputs(self.level_spacing());
            if self.unstarted {
                let seen = self.shared.seeing(&inputs.points);
                (inputs, seen)
            } else {
                let candidate = self.bring(&self.best, self.level_spacing());
                let (now, seen, _) = self.merge_guarded(&candidate, &inputs);
                (now, seen)
            }
        } else {
            self.final_state()
        };
        self.settled = false;
        self.unstarted = false;
        self.handed_over = self.descended;
        self.descended = false;
        self.ran = false;
        self.held.fill(false);
        self.reseed(now, seen)
    }

    /// Leave the coarse levels for the final level, setting `level` to it.
    /// Returns the state the run goes on from there and what the members
    /// see there: the coarse best state subdivided (`bring`) if some coarse
    /// burst ended normally (`coarse_descended`) and it passes the guard;
    /// else the inputs. So the coarse levels never leave a state only
    /// laying out produced.
    fn final_state(&mut self) -> (Shape, Seen) {
        self.level = self.shared.final_level();
        if self.coarse_descended {
            let candidate = self.bring(&self.best, self.level_spacing());
            let seen = self.shared.seeing(&candidate.points);
            if self.guards(&candidate, &seen) {
                return (candidate, seen);
            }
        }
        let seen = self.shared.seeing(&self.input.points);
        (self.input.clone(), seen)
    }

    /// Take `now` as the current and best state, and start the stall
    /// history and the abnormal streak over.
    fn reseed(&mut self, now: Shape, seen: Seen) -> Result<(), NotFinite> {
        self.now = now;
        self.seen = Arc::new(seen);
        self.energy = self.shared.energy_of(&self.now, &self.seen)?;
        self.keep();
        self.lowest = vec![self.energy];
        self.lengths = vec![self.length()];
        self.abnormal_streak = 0;
        Ok(())
    }

    /// End a run stopped before the final level at the final spacing, at
    /// the state `final_state` gives, which becomes `now` and the best
    /// state. Of the schedule it sets `level`, to the final one, and clears
    /// `unstarted`; a run it ends takes no more bursts.
    pub fn finish(&mut self) -> Result<(), NotFinite> {
        if !self.before_final() {
            return Ok(());
        }
        let (best, seen) = self.final_state();
        let energy = self.shared.energy_of(&best, &seen)?;
        self.unstarted = false;
        self.best = best;
        self.best_energy = energy;
        self.now = self.best.clone();
        self.seen = Arc::new(seen);
        self.energy = energy;
        Ok(())
    }

    /// End the level it runs at for the run's time limit: before the final
    /// level it enters the next as a level that settled hands over to it
    /// (`enter`), from level 0 if it has taken no step. Returns the level
    /// it left; None at the final level, which it leaves as it is.
    pub fn hand_over(&mut self) -> Result<Option<usize>, NotFinite> {
        if !self.before_final() {
            return Ok(None);
        }
        let left = if self.unstarted { 0 } else { self.level };
        self.enter(left + 1)?;
        Ok(Some(left))
    }

    /// Take the current state as the best.
    fn keep(&mut self) {
        self.best = self.now.clone();
        self.best_energy = self.energy;
    }

    /// Whether it settled on MAX_ABNORMAL_BURSTS abnormal bursts in a row.
    pub fn gave_up(&self) -> bool {
        self.abnormal_streak >= MAX_ABNORMAL_BURSTS
    }

    /// The summed length of the selected tracks at their current shapes,
    /// a section's as its two tracks, mm.
    pub fn length(&self) -> f64 {
        self.shared.length(&self.now.points)
    }

    /// For each member, then each via, whether at `polylines` and
    /// `positions` it has more
    /// violations than `allowed`, or, for a via, is newly inside, or deeper
    /// inside, its clearance of any obstacle or any of its stops than it
    /// started, or is a member or via a via is so inside. `seen` is what
    /// the members see at `polylines`.
    pub fn overstepping(
        &self,
        polylines: &[Vec<V2>],
        masks: &[Vec<bool>],
        positions: &[V2],
        seen: &Seen,
    ) -> Vec<bool> {
        let count = self.shared.count();
        let mut over = vec![false; count + self.shared.setup.vias.len()];
        let violations = self.shared.violations(polylines, masks, positions, seen);
        for ((o, &v), &a) in over.iter_mut().zip(&violations).zip(&self.allowed) {
            *o = v > a;
        }
        for (now, start) in [
            (
                self.shared
                    .via_distances(polylines, positions, -INSIDE_TOLERANCE),
                &self.inside,
            ),
            (
                self.shared
                    .stop_distances(polylines, positions, -INSIDE_TOLERANCE),
                &self.stopped,
            ),
        ] {
            let (fixed, inside) = newly_inside(&now, start, INSIDE_TOLERANCE, DEEPER_ROUND_OFF);
            for (k, &f) in fixed.iter().enumerate() {
                if f {
                    over[count + now.fixed.via[k]] = true;
                }
            }
            for (k, &f) in inside.iter().enumerate() {
                if f {
                    over[count + now.by_obstacle.via[k]] = true;
                    over[now.by_obstacle.obstacle[k]] = true;
                }
            }
        }
        over
    }

    /// For each member, then each via, whether at `polylines` and
    /// `positions` it oversteps (`overstepping`), enters a member or via
    /// that it did not enter at the start, `at_start`
    /// (`Shared::entered`), or is entered so. `seen` is what the members
    /// see at `polylines`.
    pub fn failing(
        &self,
        polylines: &[Vec<V2>],
        masks: &[Vec<bool>],
        positions: &[V2],
        seen: &Seen,
        at_start: &BTreeSet<(usize, usize)>,
    ) -> Vec<bool> {
        let mut failing = self.overstepping(polylines, masks, positions, seen);
        for (i, j) in self
            .shared
            .entered(polylines, positions, seen)
            .difference(at_start)
        {
            failing[*i] = true;
            failing[*j] = true;
        }
        failing
    }

    /// Whether nothing oversteps at `polylines` and `positions`
    /// (`overstepping`).
    pub fn guard(
        &self,
        polylines: &[Vec<V2>],
        masks: &[Vec<bool>],
        positions: &[V2],
        seen: &Seen,
    ) -> bool {
        !self
            .overstepping(polylines, masks, positions, seen)
            .iter()
            .any(|&o| o)
    }

    /// Whether nothing oversteps at `shape`.
    fn guards(&self, shape: &Shape, seen: &Seen) -> bool {
        self.guard(&shape.points, &shape.masks, &shape.positions, seen)
    }

    /// The variables of a burst from `starts` and the current centres.
    pub fn layout(&self, starts: Vec<Vec<V2>>) -> Layout {
        layout::layout(
            starts,
            &self.now.masks,
            &self.shared.setup.groups,
            &self.shared.carried,
        )
    }

    /// Each of `layout`'s variables' stiffness at its start: that of the
    /// length term, of the members' obstacle terms and of the moving vias'
    /// own (`problem::Problem::stiffness`), summed over the vertices it
    /// moves. `seen` and `lookups` are as for the objective.
    pub fn stiffness(
        &self,
        layout: &Layout,
        seen: &Seen,
        lookups: Option<&mut Lookups>,
    ) -> Vec<f64> {
        let (per_member, per_via) = self.shared.setup.energy.stiffness(
            &refs(&layout.starts),
            &self.now.positions,
            seen,
            lookups,
        );
        let mut stiffness = layout::length_stiffness(layout);
        for (per_vertex, index) in per_member.iter().zip(&layout.index) {
            for (&s, i) in per_vertex.iter().zip(index) {
                if let Some(i) = *i {
                    stiffness[i] += s;
                }
            }
        }
        for (&v, &s) in layout.via_index.iter().zip(&per_via) {
            stiffness[v] += s;
        }
        stiffness
    }

    /// One burst of every member and via together, after entering level 0
    /// if `unstarted`; returns its largest change of a shape or a via's
    /// centre. A burst `expired` cut short does not settle the problem.
    /// `expired` is asked before the step's long parts too: as it begins,
    /// after entering level 0, after laying the shapes out, and before and
    /// after building the burst's check; answering true there ends the
    /// step at once with no burst, None, the problem as it was but for a
    /// level entered. `n` is the problem's place among those the run
    /// advances, by which `report` hears what the burst cost and each
    /// level it enters.
    pub fn step<E: From<NotFinite>>(
        &mut self,
        mut expired: Option<&mut Expired<'_, E>>,
        bursts: &mut dyn Bursts<E>,
        report: &mut dyn FnMut(Notice) -> Result<(), E>,
    ) -> Result<Option<f64>, E> {
        if asked(&mut expired)? {
            return Ok(None);
        }
        if self.unstarted {
            self.enter(0)?;
            report(Notice::Stage { level: 0 })?;
            if asked(&mut expired)? {
                return Ok(None);
            }
        }
        let began = Instant::now();
        let shared = self.shared.clone();
        let spacing = self.level_spacing();
        let laid_out = self.bring(&self.simplified(&self.now), spacing);
        if asked(&mut expired)? {
            return Ok(None);
        }
        self.now.masks = laid_out.masks;
        self.now.contact_of = laid_out.contact_of;
        let starts = laid_out.points;
        // A burst's change counts from where it started, so that laying
        // the shapes out does not count as moving them.
        let mut before = starts.clone();
        let previous = self.now.points.clone();
        let mut burst_masks = self.now.masks.clone();
        for &i in &self.free {
            if self.held[i] {
                burst_masks[i].fill(true);
            }
        }
        let Some(burst) = self.burst(starts, &burst_masks, expired, bursts)? else {
            return Ok(None);
        };
        self.ran = true;
        let positions = std::mem::replace(&mut self.now.positions, burst.positions);
        self.now.points = burst.polylines;
        self.seen = Arc::new(shared.seeing(&self.now.points));
        for (r, c) in self.rejected_edges.iter_mut().zip(&burst.rejected_edges) {
            *r += c;
        }
        for (r, c) in self.via_rejected.iter_mut().zip(&burst.via_rejected) {
            *r += c;
        }
        if self.coarse() {
            // Each unit that fails the guard goes back to its best shape,
            // so that one cutting into clearance does not void the
            // progress of the others, and a lone chain that does is held
            // there for the rest of the level: restarted from coarse chords
            // cut into clearance, it would take every burst's L-BFGS steps
            // from the others. A chain of a junction or via is not, since
            // the shared end it moves with stays a variable.
            let (now, seen, back) = self.merge_guarded(&self.now, &self.best);
            for &i in &self.free {
                if back[i] && shared.loose[i].is_empty() && shared.carried_by[i].is_empty() {
                    self.held[i] = true;
                }
                if back[i] {
                    before[i] = previous[i].clone();
                }
            }
            self.now = now;
            self.seen = Arc::new(seen);
            self.energy = self.shared.energy_of(&self.now, &self.seen)?;
            if self.energy <= self.best_energy {
                self.keep();
            }
        } else {
            self.energy = self.shared.energy_of(&self.now, &self.seen)?;
            if self.energy <= self.best_energy && self.guards(&self.now, &self.seen) {
                self.keep();
            }
        }
        // At a coarse level the change is that of what the level kept: a
        // unit put back every burst has not moved, and does not keep the
        // others from settling, so its change counts from its shape before
        // the layout.
        let changes: Vec<f64> = self
            .free
            .iter()
            .map(|&i| shape_change(&before[i], &self.now.points[i]))
            .collect();
        let change = changes
            .iter()
            .copied()
            .chain(
                self.now
                    .positions
                    .iter()
                    .zip(&positions)
                    .map(|(&after, &before)| (after - before).norm()),
            )
            .reduce(python_max)
            .unwrap_or(f64::NAN);
        self.moved |= change > NOTHING_MOVED;
        let lowest = python_min(self.lowest[self.lowest.len() - 1], self.energy);
        self.lowest.push(lowest);
        let length = self.length();
        self.lengths.push(length);
        let ran = burst.ran;
        if ran.abnormal {
            self.abnormal_bursts += 1;
            self.abnormal_streak += 1;
            report(Notice::Abnormal {
                streak: self.abnormal_streak,
            })?;
        } else {
            self.abnormal_streak = 0;
        }
        // An abnormal burst may have stopped anywhere, so only the streak
        // settles it. A burst cut short by moves back changed little
        // because of them, not because the shapes settled.
        let cut_short = burst.via_rejected.iter().any(|&c| c > 0)
            || burst.rejected_edges.iter().any(|&c| c > 0);
        // The first burst at a level a coarser one that descended handed
        // over to starts from that level's best state, subdivided, so an
        // abnormal burst that moves nothing there settles at once, before
        // LEVEL_WINDOW bursts: L-BFGS found no descent from it, as at a
        // pad's corner, where the contact stop has a kink and every burst
        // from the end resting there fails its first line search, so
        // waiting would only reach giving up.
        let settings = shared.settings();
        let handed_over = std::mem::take(&mut self.handed_over);
        let unchanged = !cut_short && change < settings.settled_change;
        let ran_enough = self.lowest.len() > LEVEL_WINDOW;
        self.settled = self.gave_up()
            || (!ran.cut
                && ((handed_over && ran.abnormal && unchanged)
                    || (ran_enough
                        && !ran.abnormal
                        && (unchanged
                            || levelled_off(&self.lowest, self.start_energy)
                            || stalled(
                                &self.lowest,
                                &self.lengths,
                                settings,
                                self.free.len(),
                                shared.setup.vias.len(),
                            )))));
        // A burst a stop or the time limit cut before it accepted an
        // iteration has not descended: it is where it started.
        let descended = !ran.abnormal && !(ran.cut && ran.iterations == 0);
        self.descended |= descended || (handed_over && self.settled);
        self.coarse_descended |= self.coarse() && descended;
        let rest = began.elapsed().as_secs_f64() - ran.evaluating;
        report(Notice::Burst(BurstReport {
            level: self.level,
            iterations: ran.iterations,
            evaluations: ran.evaluations,
            evaluating: ran.evaluating,
            rest,
            message: ran.message,
            lowest,
            length,
            members: self.free.clone(),
            lengths: self
                .free
                .iter()
                .map(|&i| polylines::length(&self.now.points[i]))
                .collect(),
            changes,
        }))?;
        // With every member held and no via, a burst would have nothing to
        // move.
        if self.coarse() && shared.setup.vias.is_empty() && self.free.iter().all(|&i| self.held[i])
        {
            self.settled = true;
        }
        if self.settled && self.coarse() {
            let level = self.level + 1;
            self.enter(level)?;
            report(Notice::Stage { level })?;
        }
        Ok(Some(change))
    }

    /// One L-BFGS run over every vertex `masks` leaves free and every
    /// moving via of the problem, with what moved back where edges newly crossed static copper, or
    /// newly or more deeply another member's new shape, a section newly
    /// bent below its limit or a via ended newly or deeper inside its
    /// clearance of an obstacle (`checks`). None if `expired` answers
    /// true before or after the check is built.
    fn burst<E: From<NotFinite>>(
        &self,
        starts: Vec<Vec<V2>>,
        masks: &[Vec<bool>],
        mut expired: Option<&mut Expired<'_, E>>,
        bursts: &mut dyn Bursts<E>,
    ) -> Result<Option<ProblemBurst>, E> {
        let shared = &self.shared;
        let layout = layout::layout(starts.clone(), masks, &shared.setup.groups, &shared.carried);
        let seen = Arc::new(shared.seeing(&starts));
        let mut lookups = shared.setup.energy.lookups(&seen);
        let stiffness = self.stiffness(&layout, &seen, Some(&mut lookups));
        let scale = layout::scale(&stiffness, self.level_spacing());
        let x0 = layout::start_of(&layout, &self.now.positions);
        let objective = Objective {
            problem: shared.setup.energy.clone(),
            seen: seen.clone(),
            layout: layout.clone(),
            masks: shared.energy_masks(masks),
            contact_of: self.now.contact_of.clone(),
        };
        if asked(&mut expired)? {
            return Ok(None);
        }
        let check = Check::new(
            self.shared.clone(),
            layout.clone(),
            self.now.contact_of.clone(),
            seen,
            &starts,
            &self.now.positions,
            x0.clone(),
        );
        if asked(&mut expired)? {
            return Ok(None);
        }
        let (ran, found) = bursts.run(objective, lookups, x0, scale, check)?;
        let (polylines, positions) = layout.unflatten(&ran.point);
        // An abnormally ended burst stops at its check without counting it.
        let counted = match found {
            Some(round) if !ran.abnormal => round,
            _ => Round {
                members: vec![0; starts.len()],
                vias: vec![0; shared.setup.vias.len()],
            },
        };
        Ok(Some(ProblemBurst {
            polylines,
            positions,
            rejected_edges: counted.members,
            via_rejected: counted.vias,
            ran,
        }))
    }

    /// Take `earlier`'s best state, `earlier` being this problem with fewer
    /// sections held, but keep the sections held only here as drawn, and
    /// put back where they started the members and vias that would then
    /// come too close: each that would fail the guard, or whose edges or
    /// clearance would enter a member or via they did not enter at the
    /// start, and what that entered, each with what moves with it
    /// (`layout::units`), until none would. The state that is left is the
    /// best, its energy measured with the sections held. If it fails the
    /// guard still, as a member put back that a neighbour's violation count
    /// hides may leave it, everything goes back to the start.
    ///
    /// The counts of rejected edges and abnormal bursts carry over; the
    /// abnormal streak starts over, so that the problem built again is not
    /// given up on at once. At a coarse level the state is brought to its
    /// spacing (`bring`), each unit that fails the guard there from the
    /// inputs (`merge`).
    ///
    /// Of the schedule, `level`, `unstarted`, `handed_over`, `descended`,
    /// `coarse_descended`, `ran` and `moved` carry from `earlier`; `held`
    /// starts over, so a lone chain held at a coarse level is tried again.
    /// Returns how many members that move and how many vias went back.
    pub fn go_on_from(&mut self, earlier: &State) -> Result<(usize, usize), NotFinite> {
        self.put_back(earlier, State::failing)
    }

    /// `go_on_from`, putting back what `failing` names.
    fn put_back(
        &mut self,
        earlier: &State,
        failing: impl Fn(
            &State,
            &[Vec<V2>],
            &[Vec<bool>],
            &[V2],
            &Seen,
            &BTreeSet<(usize, usize)>,
        ) -> Vec<bool>,
    ) -> Result<(usize, usize), NotFinite> {
        let shared = self.shared.clone();
        let count = shared.count();
        let ahead: Vec<bool> = shared
            .setup
            .members
            .iter()
            .zip(&earlier.shared.setup.members)
            .map(|(m, before)| !m.held || before.held)
            .collect();
        // What goes back goes to the inputs at the final spacing, where the
        // guard is measured.
        let start = self.input.clone();
        let at_start = shared.entered(
            &start.points,
            &start.positions,
            &shared.seeing(&start.points),
        );
        let mut candidate = earlier.best.clone();
        for i in (0..count).filter(|&i| !ahead[i]) {
            candidate.take_member(i, &start);
        }
        let (now, seen, back) = self.merge(&candidate, &start, |s, seen| {
            failing(self, &s.points, &s.masks, &s.positions, seen, &at_start)
        });
        let from_input: Vec<bool> = (0..count).map(|i| !ahead[i] || back[i]).collect();
        let members = self.free.iter().filter(|&&i| ahead[i] && back[i]).count();
        let vias = back[count..].iter().filter(|&&b| b).count();
        self.level = earlier.level;
        self.unstarted = earlier.unstarted;
        self.handed_over = earlier.handed_over;
        self.descended = earlier.descended;
        self.coarse_descended = earlier.coarse_descended;
        self.ran = earlier.ran;
        self.moved = earlier.moved;
        self.rejected_edges = earlier.rejected_edges.clone();
        self.via_rejected = earlier.via_rejected.clone();
        self.abnormal_bursts = earlier.abnormal_bursts;
        let (now, seen) = if self.coarse() {
            // What went back is laid out at the final spacing and what a
            // burst moved is at the level's: what went back is laid out at
            // the level's too, all kept where that passes the guard.
            let inputs = self.shared.inputs(self.level_spacing());
            let mut laid_out = now;
            for i in (0..count).filter(|&i| from_input[i]) {
                laid_out.take_member(i, &inputs);
            }
            let (now, seen, _) = self.merge_guarded(&laid_out, &inputs);
            (now, seen)
        } else {
            (now, seen)
        };
        self.reseed(now, seen)?;
        Ok((members, vias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burst;
    use crate::energy::EnergyParams;
    use crate::members::{Bend, Member, Members, Section, Trace};
    use crate::optimize::{CoarseSpacings, Outcome, advance};
    use crate::pair_blocks::{PairOffset, Pins};
    use crate::testing::{Disc, FreeSpace};

    fn params() -> EnergyParams {
        EnergyParams {
            repulsion: 0.5,
            falloff_range: 1.0,
            clearance_weight: 10.0,
            standoff: 0.05,
            points_per_edge: 3,
            attraction: 0.0,
            attraction_range: 3.0,
        }
    }

    fn settings() -> Settings {
        Settings {
            spacing: 0.5,
            coarse: crate::optimize::CoarseSpacings::default(),
            settled_change: 0.005,
            stall_window: 3,
            stall_tolerance: 1e-3,
            burst: burst::Settings {
                max_iterations: 30,
                ftol: 1e-9,
                gtol: 1e-10,
            },
        }
    }

    const VIA_AT: V2 = V2 { x: 5.0, y: -5.0 };

    /// A chain from (0, 0) to (10, 0) bowed up to `rise` at its middle, in
    /// `field`, with a stitching via at `VIA_AT` in its static copper `via`,
    /// if given.
    fn problem(field: Arc<dyn Field>, rise: f64, via: Option<Arc<dyn Field>>) -> State {
        chains(&[(field, rise, 10.0)], via, settings())
    }

    /// `problem` with `coarse` spacings first, without a via.
    fn coarse_problem(field: Arc<dyn Field>, rise: f64, coarse: &[f64]) -> State {
        chains(&[(field, rise, 10.0)], None, coarse_settings(coarse))
    }

    fn coarse_settings(coarse: &[f64]) -> Settings {
        Settings {
            coarse: CoarseSpacings::new(coarse).expect("few enough"),
            ..settings()
        }
    }

    /// Per (field, rise, length), a chain of its own net on a layer of its
    /// own from (0, 0) to (length, 0) bowed up to `rise` at its middle, in
    /// `field`; with a stitching via at `VIA_AT` in its static copper
    /// `via`, if given.
    fn chains(
        given: &[(Arc<dyn Field>, f64, f64)],
        via: Option<Arc<dyn Field>>,
        settings: Settings,
    ) -> State {
        let members: Vec<Member> = given
            .iter()
            .enumerate()
            .map(|(k, (field, _, _))| {
                let trace = Trace {
                    net: format!("N{k}"),
                    width: 0.2,
                    clearance: 0.2,
                };
                Member {
                    layer: format!("L{k}"),
                    viewer: trace.clone(),
                    ignored: vec![trace.net.clone()],
                    shown: vec![trace],
                    field: field.clone(),
                    width: 0.2,
                    contacts: vec![],
                    section: None,
                }
            })
            .collect();
        let setups = given
            .iter()
            .map(|&(_, rise, length)| MemberSetup {
                original: vec![
                    V2::new(0.0, 0.0),
                    V2::new(length / 2.0, rise),
                    V2::new(length, 0.0),
                ],
                anchors: vec![true, false, true],
                contacts: vec![],
                held: false,
            })
            .collect();
        let vias: Vec<(V2, Arc<dyn Field>)> =
            via.into_iter().map(|copper| (VIA_AT, copper)).collect();
        let rings = vias.iter().map(|_| vec![]).collect();
        state(members, setups, vias, rings, settings)
    }

    /// The state of `members` set up as `setups`, with a via per (start,
    /// static copper) in `vias` whose ring each member of `rings` reaches,
    /// none seeing another or stopping at another.
    fn state(
        members: Vec<Member>,
        setups: Vec<MemberSetup>,
        vias: Vec<(V2, Arc<dyn Field>)>,
        rings: Vec<Vec<(usize, f64)>>,
        settings: Settings,
    ) -> State {
        let vias: Vec<ViaSetup> = vias
            .into_iter()
            .map(|(start, copper)| ViaSetup {
                start,
                carried: vec![],
                copper,
                stop_parts: vec![None, None],
            })
            .collect();
        joined(members, setups, vias, rings, vec![], settings)
    }

    /// `state`, its vias set up as `vias` and the loose ends of `groups`
    /// joined.
    fn joined(
        members: Vec<Member>,
        setups: Vec<MemberSetup>,
        vias: Vec<ViaSetup>,
        rings: Vec<Vec<(usize, f64)>>,
        groups: Vec<Vec<(usize, End)>>,
        settings: Settings,
    ) -> State {
        let count = members.len();
        let none: Vec<Vec<(usize, f64)>> = vias.iter().map(|_| vec![]).collect();
        let energy = Problem {
            members: Members::new(members),
            rings,
            vias: vias
                .iter()
                .map(|v| super::super::problem::ProblemVia {
                    diameter: 0.6,
                    field: v.copper.clone(),
                    stops: None,
                })
                .collect(),
            repel: ViaSight::new(&none, &none, count),
            stop: ViaSight::new(&none, &none, count),
            pairs: None,
            params: params(),
        };
        State::new(Setup {
            energy: Arc::new(energy),
            members: setups,
            vias,
            groups,
            sight: ViaSight::new(&none, &none, count),
            settings,
        })
        .expect("finite")
    }

    /// Bursts that end where they start, at `point` if given, reporting
    /// `rejected` edges per member and `moved_back` per via, cut if `cut`,
    /// abnormal if `abnormal`, after `iterations` accepted iterations.
    struct Still {
        rejected: usize,
        moved_back: usize,
        point: Option<Vec<f64>>,
        cut: bool,
        abnormal: bool,
        iterations: u64,
    }

    impl Bursts<NotFinite> for Still {
        fn run(
            &mut self,
            objective: Objective,
            _lookups: Lookups,
            start: Vec<f64>,
            _scale: Vec<f64>,
            _check: Check,
        ) -> Result<(Ran, Option<Round>), NotFinite> {
            let members = objective.layout.starts.len();
            let vias = objective.layout.via_index.len();
            Ok((
                Ran {
                    point: self.point.clone().unwrap_or(start),
                    abnormal: self.abnormal,
                    iterations: self.iterations,
                    evaluations: 1,
                    message: String::new(),
                    cut: self.cut,
                    evaluating: 0.0,
                },
                Some(Round {
                    members: vec![self.rejected; members],
                    vias: vec![self.moved_back; vias],
                }),
            ))
        }
    }

    fn still(rejected: usize, moved_back: usize) -> Still {
        Still {
            rejected,
            moved_back,
            point: None,
            cut: false,
            abnormal: false,
            iterations: 1,
        }
    }

    fn run(problem: &mut State, bursts: &mut dyn Bursts<NotFinite>, cap: u64) -> Outcome {
        let mut iteration = 0;
        advance(
            problem,
            &mut iteration,
            cap,
            false,
            None,
            bursts,
            &mut |_| Ok(()),
        )
        .expect("finite")
    }

    // Enough bursts for a level to settle on an unchanged burst.
    const STILL_ITERATIONS: u64 = LEVEL_WINDOW as u64 + 1;

    /// `problem`'s settings with a stall window too long for
    /// `STILL_ITERATIONS` bursts to stall.
    fn unstalling() -> Settings {
        Settings {
            stall_window: STILL_ITERATIONS as usize + 1,
            ..settings()
        }
    }

    /// `problem` with `unstalling` settings.
    fn still_problem(via: Option<Arc<dyn Field>>) -> State {
        chains(&[(Arc::new(FreeSpace), 1.0, 10.0)], via, unstalling())
    }

    #[test]
    fn a_problem_whose_bursts_move_back_does_not_settle() {
        let mut moving = still_problem(None);
        assert_eq!(
            run(&mut moving, &mut still(5, 0), STILL_ITERATIONS),
            Outcome::IterationCap
        );
        let mut settling = still_problem(None);
        assert_ne!(
            run(&mut settling, &mut still(0, 0), STILL_ITERATIONS),
            Outcome::IterationCap
        );
    }

    #[test]
    fn a_via_moved_back_keeps_its_problem_from_settling() {
        let mut moving = still_problem(Some(Arc::new(FreeSpace)));
        assert_eq!(
            run(&mut moving, &mut still(0, 1), STILL_ITERATIONS),
            Outcome::IterationCap
        );
        let mut settling = still_problem(Some(Arc::new(FreeSpace)));
        assert_ne!(
            run(&mut settling, &mut still(0, 0), STILL_ITERATIONS),
            Outcome::IterationCap
        );
    }

    /// How far the chain bows up from the post, mm: beyond the repulsion's
    /// reach, as far as cutting under it saves more than crossing the post
    /// costs.
    const BOW: f64 = 3.0;

    /// A small disc on the chord under the bow: straight across it is
    /// shorter, and lower in energy, than the bow, and a violation.
    fn post() -> Arc<dyn Field> {
        Arc::new(Disc {
            cx: 5.0,
            cy: 0.0,
            radius: 0.02,
        })
    }

    fn straight(problem: &State) -> Vec<f64> {
        let layout = problem.layout(problem.now.points.clone());
        let mut x = layout::start_of(&layout, &problem.now.positions);
        for y in x.iter_mut().skip(1).step_by(2) {
            *y = 0.0;
        }
        x
    }

    #[test]
    fn a_cut_burst_the_guard_rejects_is_not_kept() {
        for (field, kept) in [
            (post(), false),
            (Arc::new(FreeSpace) as Arc<dyn Field>, true),
        ] {
            let mut p = problem(field, BOW, None);
            let start = p.now.points.clone();
            let mut cut = Still {
                rejected: 0,
                moved_back: 0,
                point: Some(straight(&p)),
                cut: true,
                abnormal: false,
                iterations: 1,
            };
            assert_eq!(run(&mut p, &mut cut, 1), Outcome::IterationCap);
            assert!(p.energy < p.start_energy);
            assert!(!p.settled);
            assert_eq!(p.best.points == start, !kept);
        }
    }

    #[test]
    fn what_putting_back_misses_puts_the_whole_problem_back() {
        let mut earlier = problem(post(), BOW, None);
        let crossing = straight(&earlier);
        let layout = earlier.layout(earlier.now.points.clone());
        let (points, _) = layout.unflatten(&crossing);
        earlier.best.points = points.clone();
        let mut again = problem(post(), BOW, None);
        let start = again.now.points.clone();
        let nothing = |p: &State,
                       _: &[Vec<V2>],
                       _: &[Vec<bool>],
                       _: &[V2],
                       _: &Seen,
                       _: &BTreeSet<(usize, usize)>| {
            vec![false; p.shared.count() + p.shared.setup.vias.len()]
        };
        again.put_back(&earlier, nothing).expect("finite");
        // The guard after putting back finds the crossing.
        assert_eq!(again.now.points, start);
        let mut ahead = problem(Arc::new(FreeSpace), BOW, None);
        let mut clear = problem(Arc::new(FreeSpace), BOW, None);
        clear.best.points = points.clone();
        ahead.put_back(&clear, nothing).expect("finite");
        assert_eq!(ahead.now.points, points);
    }

    /// The point of `f`'s one sign change on [`low`, `high`].
    fn bisect(mut low: f64, mut high: f64, f: impl Fn(f64) -> f64) -> f64 {
        let below = f(low) < 0.0;
        while high - low > f64::EPSILON * high.abs().max(1.0) {
            let middle = (low + high) / 2.0;
            if (f(middle) < 0.0) == below {
                low = middle;
            } else {
                high = middle;
            }
        }
        (low + high) / 2.0
    }

    /// How far a burst may move the via along y, mm.
    const VIA_REACH: f64 = 0.5;

    /// Bursts that leave the problem, after its `k`th, `k` times `shorten`
    /// shorter and `k` times `lower` lower in energy than the first burst
    /// started: the members' free vertices drawn towards y = 0, then the
    /// via moved along y. Each reports a rejected edge, so that only the
    /// stall rule settles the problem.
    struct Aimed {
        shorten: f64,
        lower: f64,
        first: Option<(f64, f64)>,
        bursts: u32,
    }

    fn aimed(shorten: f64, lower: f64) -> Aimed {
        Aimed {
            shorten,
            lower,
            first: None,
            bursts: 0,
        }
    }

    impl Bursts<NotFinite> for Aimed {
        fn run(
            &mut self,
            objective: Objective,
            _lookups: Lookups,
            start: Vec<f64>,
            _scale: Vec<f64>,
            _check: Check,
        ) -> Result<(Ran, Option<Round>), NotFinite> {
            let layout = &objective.layout;
            let energy = |x: &[f64]| objective.evaluate(x, None).expect("finite").0;
            let length = |x: &[f64]| {
                let (polylines, _) = layout.unflatten(x);
                polylines.iter().map(|p| polylines::length(p)).sum::<f64>()
            };
            let (energy0, length0) = *self
                .first
                .get_or_insert_with(|| (energy(&start), length(&start)));
            self.bursts += 1;
            let k = f64::from(self.bursts);
            let ys: Vec<usize> = layout
                .index
                .iter()
                .flatten()
                .flatten()
                .map(|&i| 2 * i + 1)
                .collect();
            let drawn = |f: f64| {
                let mut x = start.clone();
                for &y in &ys {
                    x[y] *= f;
                }
                x
            };
            let shape = if self.shorten > 0.0 {
                bisect(0.0, 1.0, |f| {
                    length(&drawn(f)) - (length0 - k * self.shorten)
                })
            } else {
                1.0
            };
            let via = 2 * layout.via_index[0] + 1;
            let moved = |t: f64| {
                let mut x = drawn(shape);
                x[via] = start[via] + t;
                x
            };
            let t = bisect(-VIA_REACH, VIA_REACH, |t| {
                energy(&moved(t)) - (energy0 - k * self.lower)
            });
            Ok((
                Ran {
                    point: moved(t),
                    abnormal: false,
                    iterations: 1,
                    evaluations: 1,
                    message: String::new(),
                    cut: false,
                    evaluating: 0.0,
                },
                Some(Round {
                    members: vec![1; layout.starts.len()],
                    vias: vec![0; layout.via_index.len()],
                }),
            ))
        }
    }

    /// Copper of the via's own static field just below it, within the
    /// repulsion's reach of its ring: the via's energy falls as it moves
    /// up, and rises as it moves down.
    fn under_the_via() -> Arc<dyn Field> {
        Arc::new(Disc {
            cx: VIA_AT.x,
            cy: VIA_AT.y - 0.8,
            radius: 0.1,
        })
    }

    /// How far the problem's lowest energy and length fell over the last
    /// stall window.
    fn over_the_window(p: &State) -> (f64, f64) {
        let window = settings().stall_window;
        let fell = |v: &[f64]| v[v.len() - 1 - window] - v[v.len() - 1];
        (fell(&p.lowest), fell(&p.lengths))
    }

    /// A window's energy gain, in stall tolerances: past the one member's
    /// share, within the share of the member and the via.
    const VIA_SHARE_GAIN: f64 = 1.5;
    /// A window's shortening, in stall tolerances: past the length side's.
    const SHORTENING: f64 = 1.8;
    /// Bursts enough for several stall windows.
    const CAP: u64 = 8;

    #[test]
    fn a_problem_stalls_on_the_share_its_vias_add() {
        let tolerance = settings().stall_tolerance;
        let window = settings().stall_window as f64;
        let mut p = problem(Arc::new(FreeSpace), 0.0, Some(under_the_via()));

        let outcome = run(
            &mut p,
            &mut aimed(0.0, VIA_SHARE_GAIN * tolerance / window),
            CAP,
        );

        assert_ne!(outcome, Outcome::IterationCap);
        assert_eq!(p.lowest.len(), LEVEL_WINDOW + 1);
        let members = p.free.len() as f64;
        let vias = p.shared.setup.vias.len() as f64;
        let (gain, shortened) = over_the_window(&p);
        assert!(members * tolerance < gain && gain <= (members + vias) * tolerance);
        assert!(shortened <= tolerance);
    }

    #[test]
    fn a_problem_still_shortening_on_a_flat_energy_levels_off() {
        let tolerance = settings().stall_tolerance;
        let window = settings().stall_window as f64;
        let mut p = problem(Arc::new(FreeSpace), 1.0, Some(under_the_via()));

        let outcome = run(
            &mut p,
            &mut aimed(SHORTENING * tolerance / window, 0.0),
            CAP,
        );

        // The stall rule's length side would keep it running; the energy,
        // the run's one goal, has levelled off.
        assert_eq!(outcome, Outcome::Converged);
        let (gain, shortened) = over_the_window(&p);
        assert!(gain <= tolerance);
        assert!(shortened > tolerance);
        assert!(levelled_off(&p.lowest, p.start_energy));
    }

    /// Coarse spacings above `settings`' final 0.5 mm.
    const LEVELS: [f64; 2] = [2.0, 1.0];

    /// Edge lengths may exceed the spacing they were laid out at by this
    /// much, mm: round-off in measuring them.
    const EDGE_ROUND_OFF: f64 = 1e-12;

    fn longest_edge(points: &[V2]) -> f64 {
        points
            .windows(2)
            .map(|w| (w[1] - w[0]).norm())
            .fold(0.0, f64::max)
    }

    /// Bursts that leave every free vertex at `factor` times its height
    /// above y = 0, abnormal if `abnormal`, recording each burst's vertex
    /// count per member and its variables.
    struct Toward {
        factor: f64,
        abnormal: bool,
        counts: Vec<Vec<usize>>,
        variables: Vec<usize>,
    }

    fn toward(factor: f64) -> Toward {
        Toward {
            factor,
            abnormal: false,
            counts: Vec::new(),
            variables: Vec::new(),
        }
    }

    impl Bursts<NotFinite> for Toward {
        fn run(
            &mut self,
            objective: Objective,
            _lookups: Lookups,
            start: Vec<f64>,
            _scale: Vec<f64>,
            _check: Check,
        ) -> Result<(Ran, Option<Round>), NotFinite> {
            let layout = &objective.layout;
            self.counts
                .push(layout.starts.iter().map(Vec::len).collect());
            self.variables.push(layout.count);
            let mut point = start;
            for y in point.iter_mut().skip(1).step_by(2) {
                *y *= self.factor;
            }
            Ok((
                Ran {
                    point,
                    abnormal: self.abnormal,
                    iterations: 1,
                    evaluations: 1,
                    message: String::new(),
                    cut: false,
                    evaluating: 0.0,
                },
                Some(Round {
                    members: vec![0; layout.starts.len()],
                    vias: vec![0; layout.via_index.len()],
                }),
            ))
        }
    }

    fn passes(p: &State, shape: &Shape) -> bool {
        p.guards(shape, &p.shared.seeing(&shape.points))
    }

    fn energy_of_shape(p: &State, shape: &Shape) -> f64 {
        let seen = p.shared.seeing(&shape.points);
        p.shared.energy_of(shape, &seen).expect("finite")
    }

    #[test]
    fn a_problem_runs_its_coarse_levels_then_ends_at_the_final_spacing() {
        let mut p = coarse_problem(Arc::new(FreeSpace), 1.0, &LEVELS);
        let fine = p.input.points[0].len();
        let mut bursts = toward(0.5);

        assert_eq!(run(&mut p, &mut bursts, 100), Outcome::Converged);

        assert_eq!(p.level, LEVELS.len());
        let counts: Vec<usize> = bursts.counts.iter().map(|c| c[0]).collect();
        // The levels, coarsest first, then the final spacing; within a
        // level a straighter shape may lay out with fewer vertices.
        assert!(counts[0] * 3 < fine, "{counts:?}");
        assert_eq!(counts.iter().min(), Some(&counts[0]), "{counts:?}");
        assert_eq!(counts.iter().max(), counts.last(), "{counts:?}");
        assert!(
            counts
                .iter()
                .any(|&c| counts[0] < c && c < counts[counts.len() - 1]),
            "{counts:?}"
        );
        assert!(longest_edge(&p.best.points[0]) <= settings().spacing + EDGE_ROUND_OFF);
    }

    /// The levels a run of `p` reports entering, and those its bursts ran at.
    fn stages(p: &mut State, bursts: &mut dyn Bursts<NotFinite>) -> (Vec<usize>, Vec<usize>) {
        let (mut entered, mut ran) = (Vec::new(), Vec::new());
        let mut iteration = 0;
        advance(p, &mut iteration, 100, false, None, bursts, &mut |notice| {
            match notice {
                Notice::Stage { level } => entered.push(level),
                Notice::Burst(b) => ran.push(b.level),
                _ => {}
            }
            Ok(())
        })
        .expect("finite");
        (entered, ran)
    }

    #[test]
    fn a_problem_reports_entering_each_level_once() {
        let mut p = coarse_problem(Arc::new(FreeSpace), 1.0, &LEVELS);

        let (entered, ran) = stages(&mut p, &mut toward(0.5));

        assert_eq!(entered, (0..=LEVELS.len()).collect::<Vec<_>>());
        assert_eq!(ran[0], 0);
        assert!(ran.windows(2).all(|w| w[0] <= w[1]), "{ran:?}");
    }

    #[test]
    fn a_problem_with_coarse_levels_enters_the_coarsest_before_its_first_burst() {
        let mut p = coarse_problem(Arc::new(FreeSpace), 1.0, &LEVELS);
        let fine = p.input.points[0].len();
        assert!(p.unstarted);
        let mut notices = Vec::new();
        let mut bursts = toward(0.5);

        p.step(None, &mut bursts, &mut |notice| {
            notices.push(notice);
            Ok(())
        })
        .expect("finite");

        assert_eq!(notices[0], Notice::Stage { level: 0 });
        assert!(matches!(&notices[1], Notice::Burst(b) if b.level == 0));
        assert!(bursts.counts[0][0] * 3 < fine);
        assert!(!p.unstarted);
    }

    #[test]
    fn a_problem_without_coarse_levels_reports_no_level() {
        let mut single = coarse_problem(Arc::new(FreeSpace), 1.0, &[]);

        assert!(!single.unstarted);
        assert_eq!(stages(&mut single, &mut toward(0.5)).0, Vec::<usize>::new());
    }

    /// An `expired` that answers true from its `from`-th question on,
    /// counting its questions in `asked`.
    fn expiring(
        from: usize,
        asked: &std::cell::Cell<usize>,
    ) -> impl FnMut() -> Result<bool, NotFinite> + '_ {
        move || {
            asked.set(asked.get() + 1);
            Ok(asked.get() >= from)
        }
    }

    /// The questions a started problem's step asks `expired` before its
    /// burst: as it begins, after the layout, and before and after building
    /// the check.
    const PRELUDE_QUESTIONS: usize = 4;

    #[test]
    fn a_step_expired_before_its_burst_runs_none() {
        // An unstarted problem asks once more, after entering level 0.
        for (started, questions) in [(false, PRELUDE_QUESTIONS + 1), (true, PRELUDE_QUESTIONS)] {
            let fresh = || {
                let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
                if started {
                    p.enter(0).expect("finite");
                }
                p
            };
            let asked = std::cell::Cell::new(0);
            let change = fresh()
                .step(
                    Some(&mut expiring(usize::MAX, &asked)),
                    &mut toward(0.5),
                    &mut |_| Ok(()),
                )
                .expect("finite");
            assert!(change.is_some());
            assert_eq!(asked.get(), questions, "started: {started}");
            for from in 1..=questions {
                asked.set(0);
                let mut p = fresh();
                let mut bursts = toward(0.5);

                let change = p
                    .step(Some(&mut expiring(from, &asked)), &mut bursts, &mut |_| {
                        Ok(())
                    })
                    .expect("finite");

                assert_eq!(change, None, "from question {from}");
                assert_eq!(asked.get(), from);
                assert!(bursts.counts.is_empty());
                assert!(!p.ran && !p.settled);
                assert_eq!(p.now, p.best);
            }
        }
    }

    #[test]
    fn an_advance_expired_from_the_start_runs_no_burst() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        let mut bursts = toward(0.5);
        let mut iteration = 0;
        let mut expired = || Ok::<_, NotFinite>(true);

        let outcome = advance(
            &mut p,
            &mut iteration,
            100,
            false,
            Some(&mut expired),
            &mut bursts,
            &mut |_| Ok(()),
        )
        .expect("finite");

        assert_eq!(outcome, Outcome::TimeLimit);
        assert!(bursts.counts.is_empty());
    }

    #[test]
    fn each_level_starts_from_a_best_state_that_passes_the_guard_at_its_spacing() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        for level in 0..=LEVELS.len() {
            p.enter(level).expect("finite");

            assert!(passes(&p, &p.best));
            assert_eq!(p.best_energy, energy_of_shape(&p, &p.best));
            assert_eq!(p.now, p.best);
            assert_eq!(p.lowest, vec![p.best_energy]);
            assert!(longest_edge(&p.best.points[0]) <= p.level_spacing() + EDGE_ROUND_OFF);
        }
    }

    #[test]
    fn handing_over_subdivides_keeping_every_coarse_vertex() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        p.enter(0).expect("finite");
        let coarse = p.best.points[0].clone();

        p.enter(1).expect("finite");

        let fine = &p.best.points[0];
        assert!(fine.len() > coarse.len());
        assert!(coarse.iter().all(|v| fine.contains(v)));
        assert!(shape_change(&coarse, fine) < EDGE_ROUND_OFF);
    }

    #[test]
    fn a_handover_whose_subdivided_state_fails_the_guard_starts_from_the_inputs() {
        let mut p = coarse_problem(post(), BOW, &LEVELS);
        p.enter(LEVELS.len() - 1).expect("finite");
        // A best state across the post, as the guard would never keep.
        let (crossing, _) = p.layout(p.best.points.clone()).unflatten(&straight(&p));
        p.best.points = crossing;

        p.enter(LEVELS.len()).expect("finite");

        assert_eq!(p.best, p.input);
        assert_eq!(p.best_energy, energy_of_shape(&p, &p.input));
    }

    #[test]
    fn a_coarse_burst_keeps_the_units_that_pass_the_guard_and_puts_back_the_rest() {
        let mut p = chains(
            &[
                (post(), BOW, 10.0),
                (Arc::new(FreeSpace) as Arc<dyn Field>, BOW, 10.0),
            ],
            None,
            coarse_settings(&LEVELS),
        );
        p.enter(0).expect("finite");
        let entered = p.best.clone();

        let mut straighten = toward(0.0);
        p.step(None, &mut straighten, &mut |_| Ok(()))
            .expect("finite");

        // Straight across the post fails the guard; straight in free space
        // does not.
        assert_eq!(p.best.points[0], entered.points[0]);
        assert!(p.best.points[1].iter().all(|v| v.y == 0.0));
        assert!(passes(&p, &p.best));
        assert_eq!(p.now, p.best);
        // The chain put back is held for the rest of the level: the next
        // burst moves only the other.
        p.step(None, &mut straighten, &mut |_| Ok(()))
            .expect("finite");
        let pinned_ends = 2;
        assert_eq!(
            straighten.variables[1],
            straighten.counts[1][1] - pinned_ends
        );
        // The level settles once it has run its bursts, and the next tries
        // the chain again.
        for _ in 2..LEVEL_WINDOW {
            p.step(None, &mut straighten, &mut |_| Ok(()))
                .expect("finite");
        }
        assert_eq!(p.level, 1);
        assert!(!p.held.iter().any(|&h| h));
    }

    #[test]
    fn a_unit_put_back_after_a_merge_counts_no_change() {
        // A bow over the post with a notch 10 µm deep of 0.014 mm edges at
        // its apex, which laying out merges before each burst; straight
        // across the post fails the guard, so the chain goes back to its
        // notched best shape.
        let (mut member, setup) = own_chain(
            0,
            vec![
                V2::new(0.0, 0.0),
                V2::new(4.99, BOW),
                V2::new(5.0, BOW + 0.01),
                V2::new(5.01, BOW),
                V2::new(10.0, 0.0),
            ],
        );
        member.field = post();
        let mut p = state(
            vec![member],
            vec![setup],
            vec![],
            vec![],
            coarse_settings(&LEVELS),
        );
        p.enter(0).expect("finite");

        let change = p
            .step(None, &mut toward(0.0), &mut |_| Ok(()))
            .expect("finite")
            .expect("its burst ran");

        assert!(p.best.points[0].contains(&V2::new(5.0, BOW + 0.01)));
        assert!(change < settings().settled_change, "{change}");
    }

    #[test]
    fn a_chain_shorter_than_the_coarse_spacing_keeps_the_fewest_free_vertices() {
        let mut p = chains(
            &[(Arc::new(FreeSpace) as Arc<dyn Field>, 0.5, 3.0)],
            None,
            coarse_settings(&[4.0]),
        );

        p.enter(0).expect("finite");

        // Each of its two drawn edges splits into as many edges, so the
        // fewest free vertices above the minimum is one more, the drawn
        // middle vertex among them, on edges longer than the final spacing.
        let best = &p.best.points[0];
        let free = p.best.masks[0].iter().filter(|&&f| !f).count();
        assert_eq!(free, MIN_COARSE_FREE + 1);
        assert_eq!(best[best.len() / 2], V2::new(1.5, 0.5));
        let spacing = settings().spacing;
        assert!(best.windows(2).all(|w| (w[1] - w[0]).norm() > spacing));
    }

    fn advanced(p: &mut State, bursts: &mut dyn Bursts<NotFinite>) -> (Outcome, u64) {
        let mut iteration = 0;
        let outcome =
            advance(p, &mut iteration, 100, false, None, bursts, &mut |_| Ok(())).expect("finite");
        (outcome, iteration)
    }

    #[test]
    fn a_straight_chain_with_coarse_levels_has_nothing_to_do_after_its_least_bursts() {
        let mut p = coarse_problem(Arc::new(FreeSpace), 0.0, &LEVELS);

        let bursts = (LEVELS.len() + 1) as u64 * LEVEL_WINDOW as u64;
        assert_eq!(
            advanced(&mut p, &mut still(0, 0)),
            (Outcome::NothingToDo, bursts)
        );
        assert_eq!(p.level, LEVELS.len());
        assert!(p.best_energy <= energy_of_shape(&p, &p.input));
        assert!(passes(&p, &p.best));
    }

    /// A rise each burst straightens by less than the settle threshold, mm.
    const SETTLED_RISE: f64 = 0.004;

    #[test]
    fn a_problem_each_burst_settles_runs_its_least_bursts_per_level() {
        let mut p = coarse_problem(Arc::new(FreeSpace), SETTLED_RISE, &LEVELS);
        let mut bursts = toward(0.0);

        let least = (LEVELS.len() + 1) as u64 * LEVEL_WINDOW as u64;
        assert_eq!(advanced(&mut p, &mut bursts), (Outcome::Converged, least));
        assert_eq!(p.level, LEVELS.len());
        assert!(p.best_energy <= energy_of_shape(&p, &p.input));
    }

    /// How much of its height each abnormal burst leaves a vertex: enough
    /// that no burst of the run moves less than the settle threshold.
    const ABNORMAL_FACTOR: f64 = 0.8;

    #[test]
    fn abnormal_bursts_hand_a_coarse_level_over_and_give_up_only_at_the_final_level() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        let mut abnormal = Toward {
            abnormal: true,
            ..toward(ABNORMAL_FACTOR)
        };

        let (outcome, iterations) = advanced(&mut p, &mut abnormal);

        assert_eq!(outcome, Outcome::GaveUp);
        // A streak at each level, the coarsest first.
        let levels = LEVELS.len() as u64 + 1;
        assert_eq!(iterations, levels * u64::from(MAX_ABNORMAL_BURSTS));
        assert_eq!(p.level, LEVELS.len());
    }

    /// A rise whose chain the coarsest level lays out with fewer vertices
    /// than its inputs have, mm.
    const CUT_RISE: f64 = 4.0;

    /// A burst that a stop or the time limit cut after `iterations`
    /// accepted iterations, where it started.
    fn cut_after(iterations: u64) -> Still {
        Still {
            cut: true,
            iterations,
            ..still(0, 0)
        }
    }

    #[test]
    fn a_burst_cut_before_any_iteration_does_not_descend() {
        for (iterations, at_inputs) in [(0, true), (1, false)] {
            let mut p = coarse_problem(Arc::new(FreeSpace), CUT_RISE, &LEVELS);
            p.step(None, &mut cut_after(iterations), &mut |_| Ok(()))
                .expect("finite");

            p.finish().expect("finite");

            // A burst that descended leaves the coarsest layout, subdivided,
            // with more vertices than the inputs have.
            assert_eq!(p.best == p.input, at_inputs, "after {iterations}");
        }
    }

    #[test]
    fn a_handover_after_a_burst_cut_before_any_iteration_hands_no_descent() {
        for (iterations, descended) in [(0, false), (1, true)] {
            let mut p = coarse_problem(Arc::new(FreeSpace), CUT_RISE, &LEVELS);
            p.step(None, &mut cut_after(iterations), &mut |_| Ok(()))
                .expect("finite");

            assert_eq!(p.hand_over().expect("finite"), Some(0));

            assert_eq!(p.handed_over, descended, "after {iterations}");
        }
    }

    #[test]
    fn abnormal_bursts_that_move_nothing_give_up_at_the_final_level() {
        let mut p = coarse_problem(Arc::new(FreeSpace), CUT_RISE, &LEVELS);
        let mut abnormal = Still {
            abnormal: true,
            ..still(0, 0)
        };

        let (outcome, iterations) = advanced(&mut p, &mut abnormal);

        assert_eq!(outcome, Outcome::GaveUp);
        // A streak at each level, the coarsest first: a level that gave up
        // hands over a state no burst settled.
        let levels = LEVELS.len() as u64 + 1;
        assert_eq!(iterations, levels * u64::from(MAX_ABNORMAL_BURSTS));
        assert_eq!(p.level, LEVELS.len());
        // No burst descended, so it ends at its inputs, not at their
        // layout at the coarsest spacing; the final level's bursts move them
        // by round-off in the layout's offsets.
        assert_eq!(p.best.points[0].len(), p.input.points[0].len());
        assert!(shape_change(&p.best.points[0], &p.input.points[0]) < EDGE_ROUND_OFF);
    }

    #[test]
    fn the_final_level_takes_a_coarse_best_the_guard_passes_whatever_its_energy() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        assert_eq!(p.level, 0);
        // A coarse best bowed twice as high as the inputs, as the guard
        // passes.
        let mut x = layout::start_of(&p.layout(p.best.points.clone()), &p.best.positions);
        for y in x.iter_mut().skip(1).step_by(2) {
            *y *= 4.0;
        }
        (p.best.points, _) = p.layout(p.best.points.clone()).unflatten(&x);

        let coarse = p.best.clone();

        p.enter(LEVELS.len()).expect("finite");

        assert_eq!(p.best, p.bring(&coarse, p.level_spacing()));
    }

    #[test]
    fn a_run_with_coarse_levels_starts_from_its_inputs_at_the_coarsest_spacing() {
        let p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        let coarsest = p.shared.inputs(LEVELS[0]);

        assert_eq!(p.now, coarsest);
        assert_ne!(p.now, p.input);
        assert_eq!(p.start_energy, energy_of_shape(&p, &coarsest));
    }

    #[test]
    fn a_run_stopped_before_its_first_burst_ends_at_its_inputs() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        // It reports the final level, yet never ran at it.
        assert!(p.before_final());

        p.finish().expect("finite");

        assert!(!p.coarse());
        assert_eq!(p.best, p.input);
        assert_eq!(p.best_energy, energy_of_shape(&p, &p.input));
    }

    #[test]
    fn only_a_problem_at_the_final_level_is_past_the_coarse_ones() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        p.enter(0).expect("finite");
        assert!(p.before_final());

        p.enter(LEVELS.len()).expect("finite");

        assert!(!p.before_final());
    }

    #[test]
    fn an_abnormal_burst_that_moves_nothing_after_a_descent_settles_its_level() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        p.enter(0).expect("finite");
        // A normal burst descends at the coarsest level, then a streak that
        // moves nothing gives it up, as at a pad's corner.
        p.step(None, &mut toward(ABNORMAL_FACTOR), &mut |_| Ok(()))
            .expect("finite");
        assert!(!p.settled);
        let mut abnormal = Still {
            abnormal: true,
            ..still(0, 0)
        };
        for _ in 0..MAX_ABNORMAL_BURSTS {
            p.step(None, &mut abnormal, &mut |_| Ok(()))
                .expect("finite");
        }
        assert_eq!(p.level, 1);

        // Settled so, a level hands the descent over in turn.
        for level in 2..=LEVELS.len() {
            p.step(None, &mut abnormal, &mut |_| Ok(()))
                .expect("finite");
            assert_eq!(p.level, level);
        }
        p.step(None, &mut abnormal, &mut |_| Ok(()))
            .expect("finite");

        assert!(p.settled);
        assert!(!p.gave_up());
        // Only the first burst after a handover.
        p.settled = false;
        p.step(None, &mut abnormal, &mut |_| Ok(()))
            .expect("finite");
        assert!(!p.settled);
    }

    #[test]
    fn a_run_stopped_at_a_coarse_level_finishes_at_the_final_spacing() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        assert_eq!(run(&mut p, &mut toward(0.5), 3), Outcome::IterationCap);
        assert!(p.coarse());

        p.finish().expect("finite");

        assert!(!p.coarse());
        assert!(longest_edge(&p.best.points[0]) <= settings().spacing + EDGE_ROUND_OFF);
        assert!(passes(&p, &p.best));
        assert_eq!(p.best_energy, energy_of_shape(&p, &p.best));
        assert!(p.best_energy < energy_of_shape(&p, &p.input));
    }

    #[test]
    fn a_stopped_run_whose_best_fails_the_guard_ends_at_the_inputs() {
        let mut p = coarse_problem(post(), BOW, &LEVELS);
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        // Straight across the post.
        let mut x = layout::start_of(&p.layout(p.best.points.clone()), &p.best.positions);
        for y in x.iter_mut().skip(1).step_by(2) {
            *y = 0.0;
        }
        (p.best.points, _) = p.layout(p.best.points.clone()).unflatten(&x);

        p.finish().expect("finite");

        assert_eq!(p.best, p.input);
        assert_eq!(p.best_energy, energy_of_shape(&p, &p.input));
    }

    #[test]
    fn a_level_the_time_limit_ends_hands_over_as_one_that_settled() {
        let mut limited = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        let mut settled = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        for p in [&mut limited, &mut settled] {
            p.step(None, &mut toward(0.5), &mut |_| Ok(()))
                .expect("finite");
            assert_eq!(p.level, 0);
        }

        assert_eq!(limited.hand_over().expect("finite"), Some(0));
        settled.enter(1).expect("finite");

        assert_eq!(limited.level, 1);
        assert!(!limited.settled);
        assert_eq!(limited.best, settled.best);
        assert_eq!(limited.best_energy, settled.best_energy);
    }

    #[test]
    fn a_problem_the_time_limit_reaches_before_its_first_step_leaves_the_coarsest_level() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);

        assert_eq!(p.hand_over().expect("finite"), Some(0));

        assert_eq!(p.level, 1);
        assert!(p.coarse());
        assert_eq!(p.best, p.shared.inputs(LEVELS[1]));
    }

    #[test]
    fn a_coarse_level_the_time_limit_ends_without_a_normal_burst_hands_the_inputs_on() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        let mut abnormal = Toward {
            abnormal: true,
            ..toward(0.5)
        };
        p.step(None, &mut abnormal, &mut |_| Ok(()))
            .expect("finite");
        p.enter(LEVELS.len() - 1).expect("finite");

        assert_eq!(p.hand_over().expect("finite"), Some(LEVELS.len() - 1));

        assert!(!p.before_final());
        assert_eq!(p.best, p.input);
        assert_eq!(p.best_energy, energy_of_shape(&p, &p.input));
    }

    #[test]
    fn a_problem_has_run_at_its_level_from_its_first_burst_there() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        assert!(!p.ran);
        p.enter(0).expect("finite");
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        assert_eq!(p.level, 0);
        assert!(p.ran);

        // The burst that settles its level hands over to one it has not run
        // at.
        for _ in 1..LEVEL_WINDOW {
            p.step(None, &mut still(0, 0), &mut |_| Ok(()))
                .expect("finite")
                .expect("its burst ran");
        }
        assert_eq!(p.level, 1);
        assert!(!p.ran);
        p.hand_over().expect("finite");
        assert!(!p.ran);
    }

    #[test]
    fn a_run_reports_whether_some_problem_owes_a_burst_as_that_changes() {
        let mut p = coarse_problem(Arc::new(FreeSpace), SETTLED_RISE, &LEVELS);
        let mut owing = Vec::new();
        let mut iteration = 0;

        advance(
            &mut p,
            &mut iteration,
            100,
            false,
            None,
            &mut toward(0.0),
            &mut |notice| {
                if let Notice::Owing { owing: o } = notice {
                    owing.push(o);
                }
                Ok(())
            },
        )
        .expect("finite");

        // The problem owes a burst at each level it is handed to, until its
        // first burst there.
        assert_eq!(owing, vec![false, true, false, true, false]);
    }

    #[test]
    fn a_problem_owes_a_burst_only_at_a_level_a_handover_took_it_to() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        // Not started, it has no level of its own yet.
        assert!(!p.owes_burst());
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        assert!(!p.owes_burst());

        p.hand_over().expect("finite");
        assert!(p.owes_burst());
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        assert!(!p.owes_burst());

        let single = coarse_problem(Arc::new(FreeSpace), BOW, &[]);
        assert!(!single.owes_burst());
    }

    #[test]
    fn the_time_limit_at_the_final_level_hands_nothing_over() {
        let mut p = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        p.enter(LEVELS.len()).expect("finite");
        p.step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        let (best, now) = (p.best.clone(), p.now.clone());

        assert_eq!(p.hand_over().expect("finite"), None);

        assert_eq!(p.level, LEVELS.len());
        assert_eq!((p.best.clone(), p.now.clone()), (best, now));
    }

    #[test]
    fn a_problem_built_again_at_a_coarse_level_goes_on_at_it() {
        let mut earlier = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        earlier.enter(0).expect("finite");
        earlier
            .step(None, &mut toward(0.0), &mut |_| Ok(()))
            .expect("finite");
        // Built again with the post under the straightened chain, which
        // puts it back at its input.
        for (field, back) in [
            (Arc::new(FreeSpace) as Arc<dyn Field>, false),
            (post(), true),
        ] {
            let mut again = coarse_problem(field, BOW, &LEVELS);

            again.go_on_from(&earlier).expect("finite");

            assert_eq!(again.level, 0);
            assert!(!again.unstarted);
            assert!(passes(&again, &again.best));
            assert_eq!(again.best_energy, energy_of_shape(&again, &again.best));
            assert!(longest_edge(&again.best.points[0]) <= LEVELS[0] + EDGE_ROUND_OFF);
            assert_eq!(again.best.points == earlier.best.points, !back);
        }
    }

    #[test]
    fn a_problem_built_again_at_the_final_level_puts_back_at_the_final_spacing() {
        let mut earlier = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        earlier.enter(LEVELS.len()).expect("finite");
        earlier
            .step(None, &mut toward(0.0), &mut |_| Ok(()))
            .expect("finite");
        // The post under the straightened chain puts it back at its input.
        let mut again = coarse_problem(post(), BOW, &LEVELS);

        again.go_on_from(&earlier).expect("finite");

        assert_eq!(again.level, LEVELS.len());
        assert_eq!(again.best, again.input);
        assert_eq!(again.best_energy, energy_of_shape(&again, &again.input));
    }

    #[test]
    fn a_problem_built_again_owes_the_burst_its_earlier_one_owed() {
        let mut earlier = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        earlier
            .step(None, &mut toward(0.5), &mut |_| Ok(()))
            .expect("finite");
        earlier.hand_over().expect("finite");
        for burst in [false, true] {
            if burst {
                earlier
                    .step(None, &mut toward(0.5), &mut |_| Ok(()))
                    .expect("finite");
            }
            let mut again = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);

            again.go_on_from(&earlier).expect("finite");

            assert_eq!(again.level, 1);
            assert_eq!(again.owes_burst(), !burst);
        }
    }

    #[test]
    fn a_problem_built_again_after_a_descent_keeps_it() {
        let mut earlier = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        earlier.enter(0).expect("finite");
        earlier
            .step(None, &mut toward(ABNORMAL_FACTOR), &mut |_| Ok(()))
            .expect("finite");
        assert_eq!(earlier.level, 0);
        assert!(earlier.descended);

        let mut again = coarse_problem(Arc::new(FreeSpace), BOW, &LEVELS);
        again.go_on_from(&earlier).expect("finite");

        assert!(again.descended);
        assert_eq!(again.handed_over, earlier.handed_over);
    }

    /// The vertices of the held section's tracks in `beside_held`: track a
    /// has as many as its centreline, track b more.
    const HELD_A: usize = 3;
    const HELD_B: usize = 6;

    /// A held section above two chains on its layer, all in free space:
    /// chain 1 from (0, 0) to (10, 0), chain 2 from (0, 3) to (10, 3), both
    /// straight. A via at `via`, if given, is seen by chain 1.
    fn beside_held(via: Option<V2>) -> State {
        let trace = |net: &str| Trace {
            net: net.to_string(),
            width: 0.2,
            clearance: 0.2,
        };
        let line = |y: f64, n: usize| -> Vec<V2> {
            (0..n)
                .map(|k| V2::new(10.0 * k as f64 / (n - 1) as f64, y))
                .collect()
        };
        let chain = |net: &str| Member {
            layer: "L".to_string(),
            viewer: trace(net),
            ignored: vec![net.to_string()],
            shown: vec![trace(net)],
            field: Arc::new(FreeSpace),
            width: 0.2,
            contacts: vec![],
            section: None,
        };
        let section = Member {
            layer: "L".to_string(),
            viewer: trace("A"),
            ignored: vec!["A".to_string(), "B".to_string()],
            shown: vec![trace("A"), trace("B")],
            field: Arc::new(FreeSpace),
            width: 0.6,
            contacts: vec![],
            section: Some(Section {
                offset: PairOffset {
                    gap: 0.2,
                    width: 0.2,
                    a_on_right: true,
                    held: true,
                    pins: Pins {
                        a_at: vec![],
                        a: vec![],
                        b_at: vec![],
                        b: vec![],
                    },
                    a: line(9.8, HELD_A),
                    b: line(10.2, HELD_B),
                },
                bend: None,
                hinged: false,
            }),
        };
        let setup = |original: Vec<V2>, held: bool| MemberSetup {
            anchors: (0..original.len())
                .map(|k| held || k == 0 || k == original.len() - 1)
                .collect(),
            original,
            contacts: vec![],
            held,
        };
        let vias: Vec<(V2, Arc<dyn Field>)> = via
            .into_iter()
            .map(|start| (start, Arc::new(FreeSpace) as Arc<dyn Field>))
            .collect();
        let rings = vias.iter().map(|_| vec![(1, 0.5)]).collect();
        state(
            vec![section, chain("N1"), chain("N2")],
            vec![
                setup(line(10.0, HELD_A), true),
                setup(line(0.0, 3), false),
                setup(line(3.0, 3), false),
            ],
            vias,
            rings,
            settings(),
        )
    }

    #[test]
    fn a_held_section_shows_each_of_its_tracks_own_vertices() {
        let p = beside_held(None);
        let points = &p.now.points;
        let (member, vertex) = p.shared().shown_vertices(points);
        let flat = p.shared().flat(points, &[]);
        let mut expected: Vec<(usize, usize)> = Vec::new();
        expected.extend((0..HELD_A).map(|k| (0, k)));
        expected.extend((0..HELD_B).map(|k| (0, k)));
        for (i, line) in points.iter().enumerate().skip(1) {
            expected.extend((0..line.len()).map(|k| (i, k)));
        }
        assert_eq!(expected.len(), flat.len());
        let got: Vec<(usize, usize)> = member.into_iter().zip(vertex).collect();
        assert_eq!(got, expected);
    }

    /// Bursts that move chain 1's free vertices of `beside_held` to `x` and `y`,
    /// then let the check move back what that crossed, keeping chain 1 as
    /// it was moved back and whether the held section had a variable.
    struct MoveChain1 {
        /// Where chain 1's free vertices go: x, or each keeps its own, and y.
        x: Option<f64>,
        y: f64,
        reverted: Option<Vec<V2>>,
        held_free: Option<bool>,
    }

    impl MoveChain1 {
        fn to(to: V2) -> Self {
            Self {
                x: Some(to.x),
                y: to.y,
                reverted: None,
                held_free: None,
            }
        }

        fn to_y(y: f64) -> Self {
            Self {
                x: None,
                y,
                reverted: None,
                held_free: None,
            }
        }
    }

    impl Bursts<NotFinite> for MoveChain1 {
        fn run(
            &mut self,
            objective: Objective,
            _lookups: Lookups,
            start: Vec<f64>,
            _scale: Vec<f64>,
            mut check: Check,
        ) -> Result<(Ran, Option<Round>), NotFinite> {
            let layout = &objective.layout;
            let mut point = start;
            for &k in layout.index[1].iter().flatten() {
                if let Some(x) = self.x {
                    point[2 * k] = x;
                }
                point[2 * k + 1] = self.y;
            }
            self.held_free = Some(layout.index[0].iter().any(Option::is_some));
            check.revert(&mut point);
            self.reverted = Some(layout.unflatten(&point).0[1].clone());
            Ok((
                Ran {
                    point,
                    abnormal: false,
                    iterations: 1,
                    evaluations: 1,
                    message: String::new(),
                    cut: false,
                    evaluating: 0.0,
                },
                check.found.take(),
            ))
        }
    }

    #[test]
    fn a_chain_entering_the_member_after_a_held_section_moves_back() {
        let mut p = beside_held(None);
        let start = p.now.points[1].clone();
        let mut bursts = MoveChain1::to_y(3.0);
        run(&mut p, &mut bursts, 1);
        assert_eq!(bursts.reverted, Some(start));
    }

    #[test]
    fn a_chain_entering_a_held_sections_extra_vertices_moves_back() {
        let mut p = beside_held(None);
        let start = p.now.points.clone();
        let tracks = HELD_A + HELD_B;
        let held_before = p.shared().flat(&start, &[])[..tracks].to_vec();
        // Track b's vertex 4, past the centreline's last vertex.
        let mut bursts = MoveChain1::to(V2::new(8.0, 10.2));
        run(&mut p, &mut bursts, 1);
        assert_eq!(bursts.held_free, Some(false));
        assert_eq!(bursts.reverted.as_ref(), Some(&start[1]));
        assert_eq!(p.now.points[0], start[0]);
        let held_after = p.shared().flat(&p.now.points, &[])[..tracks].to_vec();
        assert_eq!(held_after, held_before);
    }

    #[test]
    fn a_via_after_a_held_section_is_numbered_after_the_members() {
        let p = beside_held(Some(V2::new(5.0, 0.0)));
        let points = &p.now.points;
        let seen = p.shared().seeing(points);
        let entered = p.shared().entered(points, &p.now.positions, &seen);
        assert!(entered.contains(&(1, 3)), "{entered:?}");
    }

    /// How far a moving via's ring keeps the chain of `over_a_via` from its
    /// centre, mm.
    const RING: f64 = 0.5;
    /// The radius the chain of `over_a_via` arcs over the via at, mm.
    const ARCED_AT: f64 = 0.5002;
    /// The arc's edges, each about 0.0196 mm, shorter than MERGE_EDGE; a
    /// chord across two of them passes 0.4998 mm from the via.
    const ARC_EDGES: u32 = 80;

    /// A chain up from (4.495, -2), over a moving via at (5, 0) on a
    /// half-circle ARCED_AT from it, and down to (5.505, -2), the via's
    /// ring keeping it RING off. The via starts at `start`.
    fn over_a_via(start: V2) -> State {
        let arc = (0..=ARC_EDGES).map(|k| {
            let turn = std::f64::consts::PI * (f64::from(k) / f64::from(ARC_EDGES) - 0.5);
            V2::new(5.0 + ARCED_AT * libm::sin(turn), ARCED_AT * libm::cos(turn))
        });
        let original: Vec<V2> = std::iter::once(V2::new(5.0 - ARCED_AT, -2.0))
            .chain(arc)
            .chain(std::iter::once(V2::new(5.0 + ARCED_AT, -2.0)))
            .collect();
        let (member, setup) = own_chain(0, original);
        state(
            vec![member],
            vec![setup],
            vec![(start, Arc::new(FreeSpace) as Arc<dyn Field>)],
            vec![vec![(0, RING)]],
            settings(),
        )
    }

    /// A chain of net `Nk` alone on layer `Lk` through `original`, its ends
    /// pinned, in free space.
    fn own_chain(k: usize, original: Vec<V2>) -> (Member, MemberSetup) {
        let trace = Trace {
            net: format!("N{k}"),
            width: 0.2,
            clearance: 0.2,
        };
        let member = Member {
            layer: format!("L{k}"),
            viewer: trace.clone(),
            ignored: vec![trace.net.clone()],
            shown: vec![trace],
            field: Arc::new(FreeSpace),
            width: 0.2,
            contacts: vec![],
            section: None,
        };
        let setup = MemberSetup {
            anchors: (0..original.len())
                .map(|k| k == 0 || k == original.len() - 1)
                .collect(),
            original,
            contacts: vec![],
            held: false,
        };
        (member, setup)
    }

    /// `own_chain` as a moving section of pair `Nk`/`Pk`, gap and track
    /// width 0.2 mm, with `bend`.
    fn own_section(k: usize, original: Vec<V2>, bend: Option<Bend>) -> (Member, MemberSetup) {
        let (mut m, s) = own_chain(k, original);
        m.width = 0.6;
        let other = Trace {
            net: format!("P{k}"),
            ..m.viewer.clone()
        };
        m.ignored.push(other.net.clone());
        m.shown.push(other);
        m.section = Some(Section {
            offset: PairOffset {
                gap: 0.2,
                width: 0.2,
                a_on_right: true,
                held: false,
                pins: Pins {
                    a_at: vec![],
                    a: vec![],
                    b_at: vec![],
                    b: vec![],
                },
                a: vec![],
                b: vec![],
            },
            bend,
            hinged: true,
        });
        (m, s)
    }

    #[test]
    fn a_section_merges_across_edges_shorter_than_its_inner_track_folds() {
        // Two 45° corners 0.042 mm apart: longer than MERGE_EDGE, shorter
        // than the section's fold length, 0.2 · 0.1 / 0.3 mm.
        let drawn = vec![
            V2::new(0.0, 0.0),
            V2::new(4.0, 0.0),
            V2::new(4.03, 0.03),
            V2::new(4.03, 4.0),
        ];
        let bend = Bend {
            limit: 0.3,
            spacing: 0.1,
        };
        let (chain, chain_setup) = own_chain(0, drawn.clone());
        let (section, section_setup) = own_section(1, drawn.clone(), Some(bend));
        let p = state(
            vec![chain, section],
            vec![chain_setup, section_setup],
            vec![],
            vec![],
            settings(),
        );
        let given = Shape {
            points: vec![drawn.clone(), drawn.clone()],
            masks: vec![vec![true, false, false, true]; 2],
            contact_of: vec![vec![None; 4]; 2],
            positions: vec![],
        };

        let got = p.simplified(&given);

        assert_eq!(got.points[0], drawn);
        assert_eq!(got.points[1], vec![drawn[0], drawn[1], drawn[3]]);
        assert_eq!(got.masks[1], vec![true, false, true]);
    }

    const PIN_LIMIT: f64 = 0.3;

    /// A section through `drawn`, fixed where `fixed` is true, alone in free
    /// space with bend limit PIN_LIMIT at spacing 0.1, and `drawn` laid out
    /// between bursts: merged as the run merges it, and as it would merge
    /// without the bend limit.
    fn merged_beside_a_pin(drawn: &[V2], fixed: &[bool]) -> (Vec<V2>, Vec<V2>) {
        let bend = Bend {
            limit: PIN_LIMIT,
            spacing: 0.1,
        };
        let (section, setup) = own_section(0, drawn.to_vec(), Some(bend));
        let p = state(vec![section], vec![setup], vec![], vec![], settings());
        let given = Shape {
            points: vec![drawn.to_vec()],
            masks: vec![fixed.to_vec()],
            contact_of: vec![vec![None; drawn.len()]],
            positions: vec![],
        };
        let guarded = p.simplified(&given).points.remove(0);
        let members = &p.shared().setup.energy.members.members;
        let limits = members[0].layout_limits().expect("a limit");
        let unguarded = simplified_vertices(
            members,
            &[Optimised {
                polyline: drawn,
                fixed,
                directions: &[],
            }],
            &[None],
            SIMPLIFY_FRACTION * settings().settled_change,
            &[Merging {
                shortest: limits.shortest,
                bend_limit: None,
            }],
        )
        .remove(0)
        .iter()
        .map(|&k| drawn[k])
        .collect();
        (guarded, unguarded)
    }

    /// The tightest bend radius of `points`.
    fn tightest(points: &[V2]) -> f64 {
        crate::pairs::bend_radii(points, None)
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn a_section_merge_beside_a_direction_pin_bends_nothing_below_its_limit() {
        // An end and its direction pin, then a free vertex 0.05 mm on,
        // shorter than the fold length, where an arc of radius 0.32 mm,
        // just wider than the limit, starts.
        const ARC: f64 = 0.32;
        let mut drawn = vec![V2::new(0.0, 0.0), V2::new(0.1, 0.0), V2::new(0.15, 0.0)];
        for k in 1..=8 {
            let turn = f64::from(k) * 0.1 / ARC;
            drawn.push(V2::new(
                0.15 + ARC * libm::sin(turn),
                ARC * (1.0 - libm::cos(turn)),
            ));
        }
        let mut fixed = vec![false; drawn.len()];
        fixed[0] = true;
        fixed[1] = true;
        *fixed.last_mut().expect("vertices") = true;
        assert!(tightest(&drawn) >= PIN_LIMIT);

        let (guarded, unguarded) = merged_beside_a_pin(&drawn, &fixed);

        // Without the limit the merge drops the free vertex and puts its
        // turn on the pin, too tight for the inner track.
        assert!(!unguarded.contains(&drawn[2]));
        assert!(tightest(&unguarded) < PIN_LIMIT);
        assert!(guarded.contains(&drawn[2]));
        assert!(tightest(&guarded) >= PIN_LIMIT);
    }

    #[test]
    fn a_section_merge_beside_a_pin_already_too_tight_bends_it_no_tighter() {
        // A pin halfway along a 0.1 mm straight between two right angles,
        // with its direction pins halfway to each corner: the edges from
        // them to the corners start far below the limit. The legs leave
        // the corners upwards, a vertex every 0.1 mm.
        let mut drawn: Vec<V2> = (0..=10)
            .rev()
            .map(|k| V2::new(0.0, 0.1 * f64::from(k)))
            .collect();
        drawn.extend([0.025, 0.05, 0.075].map(|x| V2::new(x, 0.0)));
        drawn.extend((0..=10).map(|k| V2::new(0.1, 0.1 * f64::from(k))));
        let pins = [0, 11, 12, 13, drawn.len() - 1];
        let fixed: Vec<bool> = (0..drawn.len()).map(|k| pins.contains(&k)).collect();
        let before = tightest(&drawn);
        assert!(before < PIN_LIMIT);

        let (guarded, unguarded) = merged_beside_a_pin(&drawn, &fixed);

        // Unguarded, the merge drops each corner and turns its direction
        // pin instead, which bends the pin's own straight, the edge from
        // the direction pin to the pin, from no turn at all to below the
        // limit. Guarded, the corners stay, as tight as drawn and no
        // tighter, for the bursts' hinge to round.
        assert!(!unguarded.contains(&V2::new(0.0, 0.0)), "{unguarded:?}");
        let straight = [drawn[11], drawn[12]];
        let at = unguarded
            .windows(2)
            .position(|w| w == straight)
            .expect("the pin's straight");
        let radii = crate::pairs::bend_radii(&unguarded, None);
        assert!(radii[at] < PIN_LIMIT, "{radii:?}");
        assert_eq!(guarded, drawn);
        assert_eq!(tightest(&guarded), before);
    }

    /// `count` vertices evenly from `from` to `to`.
    fn straight_line(from: V2, to: V2, count: usize) -> Vec<V2> {
        (0..count)
            .map(|k| from + (k as f64 / (count - 1) as f64) * (to - from))
            .collect()
    }

    #[test]
    fn merging_between_bursts_keeps_a_chain_clear_of_a_moving_via_where_it_is() {
        let via = V2::new(5.0, 0.0);
        let ring = ChainField::new(&[via], &[via], vec![RING], vec![0]);
        let clear = |points: &[V2]| {
            ring.segments_clear(&points[..points.len() - 1], &points[1..], 0.0)
                .iter()
                .all(|&c| c)
        };
        // Started far off, the via has moved under the arc.
        let p = over_a_via(V2::new(5.0, -5.0));
        assert!(clear(&p.now.points[0]));
        let mut there = p.now.clone();
        there.positions = vec![via];
        let kept = p.simplified(&there);
        assert!(clear(&kept.points[0]), "{:?}", kept.points[0]);
        // Where the via started, nothing keeps the arc.
        let away = p.simplified(&p.now);
        assert!(!clear(&away.points[0]), "{:?}", away.points[0]);
    }

    #[test]
    fn a_burst_whose_only_change_is_a_merge_across_short_edges_settles() {
        // A straight chain with a notch 10 µm deep of 0.014 mm edges, which
        // the merge between bursts takes out.
        let (member, setup) = own_chain(
            0,
            vec![
                V2::new(0.0, 0.0),
                V2::new(4.99, 0.0),
                V2::new(5.0, 0.01),
                V2::new(5.01, 0.0),
                V2::new(10.0, 0.0),
            ],
        );
        let mut p = state(vec![member], vec![setup], vec![], vec![], settings());
        let drawn = p.now.points[0].clone();

        let change = p
            .step(None, &mut still(0, 0), &mut |_| Ok(()))
            .expect("finite")
            .expect("its burst ran");

        assert!(shape_change(&drawn, &p.now.points[0]) > settings().settled_change);
        assert!(change < settings().settled_change, "{change}");
        for _ in 1..LEVEL_WINDOW {
            p.step(None, &mut still(0, 0), &mut |_| Ok(()))
                .expect("finite");
        }
        assert!(p.settled);
    }

    /// The members laying out keeps as they are or in part, each alone on
    /// its layer, straight but for a section: 0, a chain with a pinned
    /// vertex 6 and a contact at vertex 14; 1 and 2, chains joined at a
    /// loose end; 3, a chain whose last end a via carries; 4, a chain
    /// pinned throughout; 5, a chain to hold; 6, a section drawn with
    /// uneven edges.
    fn kept_by_simplifying() -> State {
        let line = |k: usize, y: f64, from: f64, to: f64| {
            own_chain(k, straight_line(V2::new(from, y), V2::new(to, y), 21))
        };
        let mut members = Vec::new();
        let mut setups = Vec::new();
        let (mut m, mut s) = line(0, 0.0, 0.0, 10.0);
        m.contacts = vec![Arc::new(Disc {
            cx: 7.0,
            cy: 0.0,
            radius: 0.3,
        })];
        s.anchors[6] = true;
        s.contacts = vec![14];
        members.push(m);
        setups.push(s);
        for (k, (from, to)) in [(0.0, 5.0), (5.0, 10.0)].into_iter().enumerate() {
            let (m, mut s) = line(1 + k, 1.0, from, to);
            let joined = if k == 0 { s.anchors.len() - 1 } else { 0 };
            s.anchors[joined] = false;
            members.push(m);
            setups.push(s);
        }
        let (m, mut s) = line(3, 2.0, 0.0, 10.0);
        *s.anchors.last_mut().expect("vertices") = false;
        members.push(m);
        setups.push(s);
        let (m, mut s) = line(4, 3.0, 0.0, 10.0);
        s.held = true;
        s.anchors.fill(true);
        members.push(m);
        setups.push(s);
        let (m, s) = line(5, 4.0, 0.0, 10.0);
        members.push(m);
        setups.push(s);
        let (m, s) = own_section(
            6,
            vec![
                V2::new(0.0, 5.0),
                V2::new(1.0, 5.0),
                V2::new(1.3, 5.0),
                V2::new(4.0, 5.0),
                V2::new(10.0, 5.0),
            ],
            None,
        );
        members.push(m);
        setups.push(s);
        let carrier = ViaSetup {
            start: V2::new(10.0, 2.0),
            carried: vec![(3, End::Last, V2::default())],
            copper: Arc::new(FreeSpace),
            stop_parts: vec![None, None],
        };
        let rings = vec![vec![]];
        joined(
            members,
            setups,
            vec![carrier],
            rings,
            vec![vec![(1, End::Last), (2, End::First)]],
            settings(),
        )
    }

    #[test]
    fn laying_out_keeps_pins_contacts_shared_ends_and_what_is_held() {
        let mut p = kept_by_simplifying();
        p.held[5] = true;
        let given = p.now.clone();

        let got = p.simplified(&given);

        let ends = |i: usize| {
            vec![
                given.points[i][0],
                *given.points[i].last().expect("vertices"),
            ]
        };
        let at = |i: usize, k: &[usize]| k.iter().map(|&k| given.points[i][k]).collect::<Vec<_>>();
        assert_eq!(got.points[0], at(0, &[0, 6, 14, 20]));
        assert_eq!(got.masks[0], vec![true, true, false, true]);
        assert_eq!(got.contact_of[0], vec![None, None, Some(0), None]);
        for i in 1..=3 {
            assert_eq!(got.points[i], ends(i), "member {i}");
        }
        assert_eq!(got.masks[3], vec![true, false]);
        for i in [4, 5] {
            assert_eq!(got.points[i], given.points[i], "member {i}");
        }
        // A free section is simplified as a chain is.
        assert_eq!(got.points[6], ends(6));
    }

    /// A level's lowest energies: it began at `start` and its last
    /// `LEVEL_WINDOW` bursts lowered the energy by `gain` in all.
    fn level_gaining(start: f64, gain: f64) -> Vec<f64> {
        let step = gain / LEVEL_WINDOW as f64;
        (0..=LEVEL_WINDOW)
            .map(|i| start - step * i as f64)
            .collect()
    }

    #[test]
    fn a_level_levels_off_once_its_window_gains_under_its_share_of_the_fall() {
        // The run fell from 200 to 100 before the window; 0.1 % of that is 0.1.
        assert!(levelled_off(&level_gaining(100.0, 0.09), 200.0));
        assert!(!levelled_off(&level_gaining(100.0, 0.11), 200.0));
    }

    #[test]
    fn a_level_has_not_levelled_off_before_its_window_has_run() {
        let short = &level_gaining(100.0, 0.0)[1..];
        assert!(!levelled_off(short, 200.0));
    }

    #[test]
    fn a_run_whose_energy_has_not_fallen_never_levels_off() {
        assert!(!levelled_off(&level_gaining(100.0, 0.0), 100.0));
        assert!(!levelled_off(&level_gaining(100.0, 0.0), 90.0));
    }
}
