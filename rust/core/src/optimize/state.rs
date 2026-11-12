//! One problem of a run: the chains, coupled sections and vias of the
//! layers that moving vias tie together, or of one layer, optimised
//! together; their current shapes, the best state so far, and whether the
//! problem has settled. The chains come first, then the sections, as their
//! centrelines.
//!
//! A burst starts from the members resampled; the shapes as the burst left
//! them are what the energy, the guard and the best state see. Resampling
//! can lay a chord across an obstacle corner, and a burst only rejects
//! crossings its own moves create (`checks`), so the best state is limited
//! to no more violations per member than the resampled start
//! (`polylines::count_violations`), and to no via newly or deeper inside
//! its clearance of any obstacle, or its stops, than it started
//! (`vias::free::newly_inside`), each obstacle judged apart. Each problem
//! keeps its best state, taken only from a burst that does not raise its
//! energy and passes that guard. Resampling renumbers the edges, so the
//! guard compares counts, not edges.
//!
//! A vertex or via that starts inside its clearance moves like any other:
//! the clearance hinge pushes it out, and the check after a burst moves
//! back only what was clear at the burst's start. Resampling keeps a vertex
//! more than PRESSED_DEPTH inside the others' clearance where it is, as a
//! free vertex (`State::pressed`): spread along a shape that the free
//! vertices beside them bend, such vertices would creep out of the
//! clearance a little every burst, and the run would settle later.
//!
//! A section held within the run (`State::go_on_from`) builds its problem
//! again, which measures its guard, its energy and its inputs' energy with
//! the sections held, so the best state is never compared with one that
//! had them moving.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use super::checks::{Check, Round};
use super::layout::{self, units};
use super::problem::{Layout, Lookups, Objective, Problem};
use super::{
    BurstReport, Bursts, DEEPER_ROUND_OFF, End, INSIDE_TOLERANCE, MAX_ABNORMAL_BURSTS,
    NOTHING_MOVED, NotFinite, Notice, PRESSED_DEPTH, Settings, stalled,
};
use crate::contacts::contacts_left;
use crate::field::Field;
use crate::geometry::V2;
use crate::numeric::{python_max, python_min, python_sum};
use crate::polylines::{self, count_violations, resample, shape_change};
use crate::seen::Seen;
use crate::vias::free::{FreeDistances, free_distances, newly_inside};
use crate::vias::sight::ViaSight;

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
    /// Whether it is pinned throughout and not resampled.
    pub held: bool,
    /// A section's bend-radius limit, which its bursts may not newly bend
    /// it below (`polylines::SelfCheck`).
    pub limit: Option<f64>,
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
    /// member and its vertex there: a section's tracks have a vertex for
    /// each of its centreline's.
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
            let times = if m.section.is_some() { 2 } else { 1 };
            for _ in 0..times {
                member.extend(std::iter::repeat_n(i, points.len()));
                vertex.extend(0..points.len());
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
    pub best: Shape,
    pub energy: f64,
    pub best_energy: f64,
    /// The energy of the inputs, resampled.
    pub input_energy: f64,
    /// Per member, the violations the resampled start has, which the guard
    /// holds it to.
    pub allowed: Vec<usize>,
    inside: FreeDistances,
    stopped: FreeDistances,
    /// The members that move: those with a free vertex and those a via
    /// carries.
    pub free: Vec<usize>,
    pub moving: bool,
    /// Whether some burst changed a shape by more than NOTHING_MOVED.
    pub moved: bool,
    /// Per member, whether its next burst starts from its shape as it is,
    /// resampled already.
    resampled: Vec<bool>,
    lowest: Vec<f64>,
    lengths: Vec<f64>,
    pub rejected_edges: Vec<usize>,
    pub via_rejected: Vec<usize>,
    pub abnormal_bursts: u32,
    pub abnormal_streak: u32,
    pub settled: bool,
}

/// `polylines::resample`, keeping the `loose` ends, the vertices that keep
/// a contact (`contact_of`) and the `kept` vertices where they are, as free
/// vertices, and where each contact went.
fn resample_member(
    points: &[V2],
    mask: &[bool],
    contact_of: &[Option<usize>],
    loose: &[End],
    spacing: f64,
    kept: Option<&[bool]>,
) -> (Vec<V2>, Vec<bool>, Vec<Option<usize>>) {
    let mut anchors: Vec<bool> = mask
        .iter()
        .zip(contact_of)
        .map(|(&m, c)| m || c.is_some())
        .collect();
    for end in loose {
        let at = end.of(anchors.len());
        anchors[at] = true;
    }
    if let Some(kept) = kept {
        for (a, &k) in anchors.iter_mut().zip(kept) {
            *a |= k;
        }
    }
    let (points, mut fixed) = resample(points, &anchors, spacing);
    // Resampling keeps the anchors, in order.
    let at: Vec<usize> = (0..fixed.len()).filter(|&k| fixed[k]).collect();
    let anchored = (0..anchors.len()).filter(|&k| anchors[k]);
    let mut moved = vec![None; points.len()];
    for (&slot, original) in at.iter().zip(anchored) {
        if !mask[original] {
            fixed[slot] = false;
        }
        moved[slot] = contact_of[original];
    }
    (points, fixed, moved)
}

impl State {
    /// The problem `setup` makes, at its members resampled; Err if its
    /// energy there is not finite.
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
        let mut points = Vec::with_capacity(count);
        let mut masks = Vec::with_capacity(count);
        let mut contact_of = Vec::with_capacity(count);
        for ((m, given), ends) in setup.members.iter().zip(&given_contact_of).zip(&loose) {
            let (p, mask, c) = if m.held {
                (m.original.clone(), m.anchors.clone(), given.clone())
            } else {
                resample_member(&m.original, &m.anchors, given, ends, spacing, None)
            };
            // Resampling would put free vertices between a held member's
            // pins.
            if mask.iter().all(|&f| f) {
                points.push(m.original.clone());
                masks.push(m.anchors.clone());
                contact_of.push(given.clone());
            } else {
                points.push(p);
                masks.push(mask);
                contact_of.push(c);
            }
        }
        let positions: Vec<V2> = setup.vias.iter().map(|v| v.start).collect();
        let free: Vec<usize> = (0..count)
            .filter(|&i| !masks[i].iter().all(|&f| f) || !carried_by[i].is_empty())
            .collect();
        let moving = !free.is_empty() || !setup.vias.is_empty();
        let shared = Arc::new(Shared {
            setup,
            loose,
            carried_by,
            given_contact_of,
            carried,
        });
        let now = Shape {
            points,
            masks,
            contact_of,
            positions,
        };
        let seen = shared.seeing(&now.points);
        let allowed = shared.violations(&now.points, &now.masks, &now.positions, &seen);
        let inside = shared.via_distances(&now.points, &now.positions, -INSIDE_TOLERANCE);
        let stopped = shared.stop_distances(&now.points, &now.positions, -INSIDE_TOLERANCE);
        let energy = shared.energy_of(&now, &seen)?;
        let length = shared.length(&now.points);
        let vias = shared.setup.vias.len();
        Ok(Self {
            best: now.clone(),
            now,
            seen: Arc::new(seen),
            energy,
            best_energy: energy,
            input_energy: energy,
            allowed,
            inside,
            stopped,
            free,
            moving,
            moved: false,
            // The first burst starts from the shapes `allowed` was
            // measured on: resampled again, they would cut corners deeper
            // still.
            resampled: vec![true; count],
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

    /// Per member, its vertices at `polylines` more than PRESSED_DEPTH
    /// inside the clearance of the other members and vias at `positions`,
    /// which resampling keeps where they are, as free vertices that the
    /// next burst moves. Vertices deep inside static copper are left to
    /// resampling, which lays them there where a chord cuts a corner and
    /// moves them out again. `seen` is what the members see at
    /// `polylines`.
    pub fn pressed(&self, polylines: &[Vec<V2>], positions: &[V2], seen: &Seen) -> Vec<Vec<bool>> {
        let flat = self.shared.flat(polylines, positions);
        seen.distances(&refs(polylines), &flat, -PRESSED_DEPTH)
            .into_iter()
            .map(|d| d.into_iter().map(|d| d < -PRESSED_DEPTH).collect())
            .collect()
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

    /// One burst of every member and via together; returns its largest
    /// change of a shape or a via's centre. A burst `expired` cut short
    /// does not settle the problem. `n` is the problem's place among those
    /// the run advances, by which `report` hears what the burst cost.
    pub fn step<E: From<NotFinite>>(
        &mut self,
        n: usize,
        bursts: &mut dyn Bursts<E>,
        report: &mut dyn FnMut(Notice) -> Result<(), E>,
    ) -> Result<f64, E> {
        let began = Instant::now();
        let shared = self.shared.clone();
        let spacing = shared.settings().spacing;
        // The change counts from the shapes as given, so that resampling
        // the inputs does not count as moving them.
        let before: Vec<Vec<V2>> = (0..shared.count())
            .map(|i| {
                if self.resampled[i] {
                    shared.setup.members[i].original.clone()
                } else {
                    self.now.points[i].clone()
                }
            })
            .collect();
        let pressed = self.pressed(&self.now.points, &self.now.positions, &self.seen);
        let mut starts = Vec::with_capacity(shared.count());
        let mut masks = Vec::with_capacity(shared.count());
        let mut contact_of = Vec::with_capacity(shared.count());
        for i in 0..shared.count() {
            let (points, mask, kept) = (
                &self.now.points[i],
                &self.now.masks[i],
                &self.now.contact_of[i],
            );
            if self.free.contains(&i) && !self.resampled[i] {
                let (p, m, c) = resample_member(
                    points,
                    mask,
                    kept,
                    &shared.loose[i],
                    spacing,
                    Some(&pressed[i]),
                );
                starts.push(p);
                masks.push(m);
                contact_of.push(c);
            } else {
                starts.push(points.clone());
                masks.push(mask.clone());
                contact_of.push(kept.clone());
            }
        }
        self.now.masks = masks;
        self.now.contact_of = contact_of;
        self.resampled = vec![false; starts.len()];
        let burst = self.burst(starts, bursts, report)?;
        let changes: Vec<f64> = self
            .free
            .iter()
            .map(|&i| shape_change(&before[i], &burst.polylines[i]))
            .collect();
        let change = changes
            .iter()
            .copied()
            .chain(
                burst
                    .positions
                    .iter()
                    .zip(&self.now.positions)
                    .map(|(&after, &before)| (after - before).norm()),
            )
            .reduce(python_max)
            .unwrap_or(f64::NAN);
        self.moved |= change > NOTHING_MOVED;
        self.now.points = burst.polylines;
        self.now.positions = burst.positions;
        self.seen = Arc::new(shared.seeing(&self.now.points));
        for (r, c) in self.rejected_edges.iter_mut().zip(&burst.rejected_edges) {
            *r += c;
        }
        for (r, c) in self.via_rejected.iter_mut().zip(&burst.via_rejected) {
            *r += c;
        }
        self.energy = shared.energy_of(&self.now, &self.seen)?;
        if self.energy <= self.best_energy && self.guards(&self.now, &self.seen) {
            self.keep();
        }
        let lowest = python_min(self.lowest[self.lowest.len() - 1], self.energy);
        self.lowest.push(lowest);
        let length = self.length();
        self.lengths.push(length);
        let ran = burst.ran;
        if ran.abnormal {
            self.abnormal_bursts += 1;
            self.abnormal_streak += 1;
            report(Notice::Abnormal {
                problem: n,
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
        let settings = shared.settings();
        self.settled = self.gave_up()
            || (!ran.abnormal
                && !ran.cut
                && ((!cut_short && change < settings.settled_change)
                    || stalled(
                        &self.lowest,
                        &self.lengths,
                        settings,
                        self.free.len(),
                        shared.setup.vias.len(),
                    )));
        let rest = began.elapsed().as_secs_f64() - ran.evaluating;
        report(Notice::Burst(BurstReport {
            problem: n,
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
        Ok(change)
    }

    /// One L-BFGS run over every free vertex and moving via of the problem,
    /// with what moved back where edges newly crossed static copper, or
    /// newly or more deeply another member's new shape, a section newly
    /// bent below its limit or a via ended newly or deeper inside its
    /// clearance of an obstacle (`checks`).
    fn burst<E: From<NotFinite>>(
        &self,
        starts: Vec<Vec<V2>>,
        bursts: &mut dyn Bursts<E>,
        report: &mut dyn FnMut(Notice) -> Result<(), E>,
    ) -> Result<ProblemBurst, E> {
        let shared = &self.shared;
        let layout = self.layout(starts.clone());
        let seen = Arc::new(shared.seeing(&starts));
        let mut lookups = shared.setup.energy.lookups(&seen);
        let stiffness = self.stiffness(&layout, &seen, Some(&mut lookups));
        let scale = layout::scale(&stiffness, shared.settings().spacing);
        let x0 = layout::start_of(&layout, &self.now.positions);
        let objective = Objective {
            problem: shared.setup.energy.clone(),
            seen: seen.clone(),
            layout: layout.clone(),
            masks: shared.energy_masks(&self.now.masks),
            contact_of: self.now.contact_of.clone(),
        };
        let check = Check::new(
            self.shared.clone(),
            layout.clone(),
            self.now.contact_of.clone(),
            seen,
            &starts,
            &self.now.positions,
            x0.clone(),
        );
        let (ran, found) = bursts.run(objective, lookups, x0, scale, check, report)?;
        let (polylines, positions) = layout.unflatten(&ran.point);
        // An abnormally ended burst stops at its check without counting it.
        let counted = match found {
            Some(round) if !ran.abnormal => round,
            _ => Round {
                members: vec![0; starts.len()],
                vias: vec![0; shared.setup.vias.len()],
            },
        };
        Ok(ProblemBurst {
            polylines,
            positions,
            rejected_edges: counted.members,
            via_rejected: counted.vias,
            ran,
        })
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
    /// given up on at once. Returns how many members that move and how many
    /// vias went back.
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
        let unit = units(count, &shared.setup.groups, &shared.carried);
        let at_start = shared.entered(&self.now.points, &self.now.positions, &self.seen);
        let start = self.now.clone();
        let starts: Vec<V2> = shared.setup.vias.iter().map(|v| v.start).collect();
        let state = |back: &[bool]| -> Shape {
            let mut shape = start.clone();
            for i in 0..count {
                if ahead[i] && !back[unit[i]] {
                    shape.points[i] = earlier.best.points[i].clone();
                    shape.masks[i] = earlier.best.masks[i].clone();
                    shape.contact_of[i] = earlier.best.contact_of[i].clone();
                }
            }
            shape.positions = earlier
                .best
                .positions
                .iter()
                .enumerate()
                .map(|(k, &at)| if back[unit[count + k]] { starts[k] } else { at })
                .collect();
            shape
        };
        let mut back = vec![false; unit.len()];
        let (mut now, mut seen) = loop {
            let now = state(&back);
            let seen = shared.seeing(&now.points);
            let failing = failing(
                self,
                &now.points,
                &now.masks,
                &now.positions,
                &seen,
                &at_start,
            );
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
        self.now = now;
        self.seen = Arc::new(seen);
        self.resampled = (0..count).map(|i| !ahead[i] || back[unit[i]]).collect();
        self.moved = earlier.moved;
        self.energy = shared.energy_of(&self.now, &self.seen)?;
        self.keep();
        self.lowest = vec![self.energy];
        self.lengths = vec![self.length()];
        self.rejected_edges = earlier.rejected_edges.clone();
        self.via_rejected = earlier.via_rejected.clone();
        self.abnormal_bursts = earlier.abnormal_bursts;
        let members = self
            .free
            .iter()
            .filter(|&&i| ahead[i] && back[unit[i]])
            .count();
        let vias = unit[count..].iter().filter(|&&r| back[r]).count();
        Ok((members, vias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burst;
    use crate::energy::EnergyParams;
    use crate::members::{Member, Members, Trace};
    use crate::optimize::{Outcome, advance};
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

    /// A chain from (0, 0) to (10, 0) bowed up to `rise` at its middle, in
    /// `field`, with a stitching via at (5, -5) if `via`.
    fn problem(field: Arc<dyn Field>, rise: f64, via: bool) -> State {
        let trace = Trace {
            net: "A".into(),
            width: 0.2,
            clearance: 0.2,
        };
        let member = Member {
            layer: "F.Cu".into(),
            viewer: trace.clone(),
            ignored: vec!["A".into()],
            shown: vec![trace],
            field,
            width: 0.2,
            contacts: vec![],
            section: None,
        };
        let vias: Vec<ViaSetup> = if via {
            vec![ViaSetup {
                start: V2::new(5.0, -5.0),
                carried: vec![],
                copper: Arc::new(FreeSpace),
                stop_parts: vec![None, None],
            }]
        } else {
            vec![]
        };
        let none: Vec<Vec<(usize, f64)>> = vias.iter().map(|_| vec![]).collect();
        let energy = Problem {
            members: Members::new(vec![member]),
            rings: none.clone(),
            vias: vias
                .iter()
                .map(|v| super::super::problem::ProblemVia {
                    diameter: 0.6,
                    field: v.copper.clone(),
                    stops: None,
                })
                .collect(),
            repel: ViaSight::new(&none, &none, 1),
            stop: ViaSight::new(&none, &none, 1),
            pairs: None,
            params: params(),
            short_edge: 0.05,
        };
        State::new(Setup {
            energy: Arc::new(energy),
            members: vec![MemberSetup {
                original: vec![V2::new(0.0, 0.0), V2::new(5.0, rise), V2::new(10.0, 0.0)],
                anchors: vec![true, false, true],
                contacts: vec![],
                held: false,
                limit: None,
            }],
            vias,
            groups: vec![],
            sight: ViaSight::new(&none, &none, 1),
            settings: settings(),
        })
        .expect("finite")
    }

    /// Bursts that end where they start, at `point` if given, reporting
    /// `rejected` edges per member and `moved_back` per via, cut if `cut`.
    struct Still {
        rejected: usize,
        moved_back: usize,
        point: Option<Vec<f64>>,
        cut: bool,
    }

    impl Bursts<NotFinite> for Still {
        fn run(
            &mut self,
            objective: Objective,
            _lookups: Lookups,
            start: Vec<f64>,
            _scale: Vec<f64>,
            _check: Check,
            _report: &mut dyn FnMut(Notice) -> Result<(), NotFinite>,
        ) -> Result<(Ran, Option<Round>), NotFinite> {
            let members = objective.layout.starts.len();
            let vias = objective.layout.via_index.len();
            Ok((
                Ran {
                    point: self.point.clone().unwrap_or(start),
                    abnormal: false,
                    iterations: 1,
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
        }
    }

    fn run(problem: &mut State, bursts: &mut dyn Bursts<NotFinite>, cap: u64) -> Outcome {
        let mut iteration = 0;
        advance(
            &mut [problem],
            &mut iteration,
            cap,
            false,
            None,
            bursts,
            &mut |_| Ok(()),
        )
        .expect("finite")
    }

    // Too few bursts for the best energy to stall.
    const STILL_ITERATIONS: u64 = 2;

    #[test]
    fn a_problem_whose_bursts_move_back_does_not_settle() {
        let mut moving = problem(Arc::new(FreeSpace), 1.0, false);
        assert_eq!(
            run(&mut moving, &mut still(5, 0), STILL_ITERATIONS),
            Outcome::IterationCap
        );
        let mut settling = problem(Arc::new(FreeSpace), 1.0, false);
        assert_ne!(
            run(&mut settling, &mut still(0, 0), STILL_ITERATIONS),
            Outcome::IterationCap
        );
    }

    #[test]
    fn a_via_moved_back_keeps_its_problem_from_settling() {
        let mut moving = problem(Arc::new(FreeSpace), 1.0, true);
        assert_eq!(
            run(&mut moving, &mut still(0, 1), STILL_ITERATIONS),
            Outcome::IterationCap
        );
        let mut settling = problem(Arc::new(FreeSpace), 1.0, true);
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
            let mut p = problem(field, BOW, false);
            let start = p.now.points.clone();
            let mut cut = Still {
                rejected: 0,
                moved_back: 0,
                point: Some(straight(&p)),
                cut: true,
            };
            assert_eq!(run(&mut p, &mut cut, 1), Outcome::IterationCap);
            assert!(p.energy < p.input_energy);
            assert!(!p.settled);
            assert_eq!(p.best.points == start, !kept);
        }
    }

    #[test]
    fn what_putting_back_misses_puts_the_whole_problem_back() {
        let mut earlier = problem(post(), BOW, false);
        let crossing = straight(&earlier);
        let layout = earlier.layout(earlier.now.points.clone());
        let (points, _) = layout.unflatten(&crossing);
        earlier.best.points = points.clone();
        let mut again = problem(post(), BOW, false);
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
        let mut ahead = problem(Arc::new(FreeSpace), BOW, false);
        let mut clear = problem(Arc::new(FreeSpace), BOW, false);
        clear.best.points = points.clone();
        ahead.put_back(&clear, nothing).expect("finite");
        assert_eq!(ahead.now.points, points);
    }
}
