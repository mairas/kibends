//! The run loop as Python sees it (`kibends_core::optimize`): a problem's
//! state (`ProblemState`), the outer iterations that advance the problems
//! (`advance`) and what they tell Python as they go (`FallbackNotice`,
//! `AbnormalNotice`, `BurstNotice`, `ProgressNotice`).
//!
//! Each burst runs through the Python callable `advance` takes, given the
//! burst's objective, start, scale and check (`Revert`), so that the
//! Python side keeps `bursts.run_burst` as the one place a burst runs; the
//! evaluations stay in Rust.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{BoolArg, BoolArray, IntpArray};
use crate::field::{Flags, Floats, Indices, Points, floats, points_of, vectors};
use crate::fields::FieldArg;
use crate::problem::{
    FreeDistances, Lookups, Objective, Polyline, Problem, Seen, ViaSight, check_count, masks_of,
    not_finite, polylines_of, vectors_list,
};
use kibends_core::burst;
use kibends_core::geometry::V2;
use kibends_core::optimize::checks::{Check, Round};
use kibends_core::optimize::layout;
use kibends_core::optimize::problem::{self, Layout};
use kibends_core::optimize::state::{MemberSetup, Setup, State, ViaSetup};
use kibends_core::optimize::{
    self as core, Bursts, End, NotFinite, Notice, Outcome, Ran, Settings,
};

/// An error of the run: Python's, or an energy that is not finite.
enum Failure {
    Python(PyErr),
    NotFinite(f64),
}

impl From<PyErr> for Failure {
    fn from(e: PyErr) -> Self {
        Failure::Python(e)
    }
}

impl From<NotFinite> for Failure {
    fn from(e: NotFinite) -> Self {
        Failure::NotFinite(e.0)
    }
}

impl Failure {
    fn into_py(self, py: Python<'_>) -> PyErr {
        match self {
            Failure::Python(e) => e,
            Failure::NotFinite(v) => not_finite(py, v),
        }
    }
}

/// The chain ends a via carries: per end its member, whether it is the
/// member's last point, and its offset from the via's centre.
type Carried = Vec<(usize, bool, (f64, f64))>;

fn end(last: bool) -> End {
    if last { End::Last } else { End::First }
}

fn contacts<'py>(py: Python<'py>, contact_of: &[Option<usize>]) -> Indices<'py> {
    // An index into a Rust slice fits an isize.
    IntpArray(
        contact_of
            .iter()
            .map(|c| c.map_or(-1, |k| k as isize))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )
}

fn flags<'py>(py: Python<'py>, values: &[bool]) -> Flags<'py> {
    BoolArray(values.to_vec().into_pyarray(py))
}

/// One problem of a run (`kibends_core::optimize::state::State`): its
/// current shapes, its best state, and whether it has settled.
#[gen_stub_pyclass]
#[pyclass(module = "kibends._core")]
pub struct ProblemState {
    inner: State,
}

impl ProblemState {
    fn state(
        &self,
        polylines: &[Polyline<'_>],
        positions: &Polyline<'_>,
    ) -> PyResult<(Vec<Vec<V2>>, Vec<V2>)> {
        let polylines = polylines_of(polylines)?;
        let positions = points_of(positions)?;
        let shared = self.inner.shared();
        check_count("polylines", polylines.len(), shared.setup.members.len())?;
        check_count("positions", positions.len(), shared.setup.vias.len())?;
        Ok((polylines, positions))
    }

    fn layout_of(&self, starts: &[Polyline<'_>]) -> PyResult<Layout> {
        let starts = polylines_of(starts)?;
        check_count(
            "starts",
            starts.len(),
            self.inner.shared().setup.members.len(),
        )?;
        for (s, mask) in starts.iter().zip(&self.inner.now.masks) {
            check_count("start vertices", s.len(), mask.len())?;
        }
        Ok(self.inner.layout(starts))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ProblemState {
    /// The problem of `energy`'s members and moving vias: per member, its
    /// centreline as given (`originals`), its pinned vertices (`anchors`),
    /// the vertex that keeps each of its contacts (`contacts`), whether it
    /// is `held` and a section's bend-radius limit (`limits`); per via, its
    /// centre (`starts`), the chain ends it `carried`, each the member,
    /// whether it is its last point and its offset from the centre, its
    /// static `copper` and what only stops it in parts (`stop_parts`); the
    /// loose ends that share a point (`groups`, each member and whether it
    /// is its last point), what the vias' checks see (`sight`) and the
    /// run's settings. Raises ValueError if its energy is not finite.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        energy: PyRef<'_, Problem>,
        originals: Vec<Polyline<'_>>,
        anchors: Vec<BoolArg<'_>>,
        contacts: Vec<Vec<usize>>,
        held: Vec<bool>,
        limits: Vec<Option<f64>>,
        starts: Polyline<'_>,
        carried: Vec<Carried>,
        copper: Vec<FieldArg>,
        stop_parts: Vec<Vec<Option<FieldArg>>>,
        groups: Vec<Vec<(usize, bool)>>,
        sight: PyRef<'_, ViaSight>,
        spacing: f64,
        settled_change: f64,
        stall_window: usize,
        stall_tolerance: f64,
        burst_maxiter: u64,
        ftol: f64,
        gtol: f64,
    ) -> PyResult<Self> {
        let originals = polylines_of(&originals)?;
        let count = energy.inner.members.len();
        let anchors = masks_of(&anchors, &originals)?;
        for (what, got) in [
            ("centrelines", originals.len()),
            ("contacts", contacts.len()),
            ("holds", held.len()),
            ("limits", limits.len()),
        ] {
            check_count(what, got, count)?;
        }
        let starts = points_of(&starts)?;
        let vias = energy.inner.vias.len();
        for (what, got) in [
            ("via centres", starts.len()),
            ("carried ends", carried.len()),
            ("via copper", copper.len()),
            ("via stops", stop_parts.len()),
            ("vias seen by the checks", sight.vias),
        ] {
            check_count(what, got, vias)?;
        }
        check_count("members seen by the checks", sight.members, count)?;
        let members = originals
            .into_iter()
            .zip(anchors)
            .zip(contacts)
            .zip(held.into_iter().zip(limits))
            .map(
                |(((original, anchors), contacts), (held, limit))| MemberSetup {
                    original,
                    anchors,
                    contacts,
                    held,
                    limit,
                },
            )
            .collect();
        let vias = starts
            .into_iter()
            .zip(carried)
            .zip(copper.into_iter().zip(stop_parts))
            .map(|((start, carried), (copper, stops))| ViaSetup {
                start,
                carried: carried
                    .into_iter()
                    .map(|(i, last, (x, y))| (i, end(last), V2::new(x, y)))
                    .collect(),
                copper: copper.0,
                stop_parts: stops.into_iter().map(|s| s.map(|f| f.0)).collect(),
            })
            .collect();
        let setup = Setup {
            energy: energy.inner.clone(),
            members,
            vias,
            groups: groups
                .into_iter()
                .map(|g| g.into_iter().map(|(i, last)| (i, end(last))).collect())
                .collect(),
            sight: sight.inner.clone(),
            settings: Settings {
                spacing,
                settled_change,
                stall_window,
                stall_tolerance,
                burst: burst::Settings {
                    max_iterations: burst_maxiter,
                    ftol,
                    gtol,
                },
            },
        };
        let inner = State::new(setup).map_err(|e| not_finite(py, e.0))?;
        Ok(Self { inner })
    }

    /// Each member's current shape.
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Vec<Points<'py>>> {
        vectors_list(py, &self.inner.now.points)
    }

    /// Each member's pinned vertices.
    #[getter]
    fn masks<'py>(&self, py: Python<'py>) -> Vec<Flags<'py>> {
        self.inner.now.masks.iter().map(|m| flags(py, m)).collect()
    }

    /// Each member's vertices' contacts, -1 for none.
    #[getter]
    fn contact_of<'py>(&self, py: Python<'py>) -> Vec<Indices<'py>> {
        self.inner
            .now
            .contact_of
            .iter()
            .map(|c| contacts(py, c))
            .collect()
    }

    /// Each moving via's current centre.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.now.positions)
    }

    /// What the members see of each other at their current shapes.
    #[getter]
    fn seen(&self) -> Seen {
        Seen::shared(self.inner.seen.clone())
    }

    /// Per member, the violations its resampled start has, which the guard
    /// holds it to.
    #[getter]
    fn allowed(&self) -> Vec<usize> {
        self.inner.allowed.clone()
    }

    /// The members that move.
    #[getter]
    fn free(&self) -> Vec<usize> {
        self.inner.free.clone()
    }

    #[getter]
    fn moving(&self) -> bool {
        self.inner.moving
    }

    /// Whether some burst changed a shape by more than NOTHING_MOVED.
    #[getter]
    fn moved(&self) -> bool {
        self.inner.moved
    }

    #[getter]
    fn settled(&self) -> bool {
        self.inner.settled
    }

    /// Whether it settled on MAX_ABNORMAL_BURSTS abnormal bursts in a row.
    #[getter]
    fn gave_up(&self) -> bool {
        self.inner.gave_up()
    }

    /// Its energy at its current shapes.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    #[getter]
    fn best_energy(&self) -> f64 {
        self.inner.best_energy
    }

    /// Its energy at its inputs, resampled.
    #[getter]
    fn input_energy(&self) -> f64 {
        self.inner.input_energy
    }

    /// Its bursts that L-BFGS ended abnormally.
    #[getter]
    fn abnormal_bursts(&self) -> u32 {
        self.inner.abnormal_bursts
    }

    /// Its abnormal bursts in a row, up to the last.
    #[getter]
    fn abnormal_streak(&self) -> u32 {
        self.inner.abnormal_streak
    }

    /// The summed length of the selected tracks at their current shapes,
    /// a section's as its two tracks, mm.
    fn length(&self) -> f64 {
        self.inner.length()
    }

    /// Member k's best state, with `best`, else its input: its polyline,
    /// its pinned vertices, each vertex's contact, -1 for none, and its
    /// rejected edges.
    fn member<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        best: bool,
    ) -> PyResult<(Points<'py>, Flags<'py>, Indices<'py>, usize)> {
        if k >= self.inner.shared().setup.members.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "no member {k}"
            )));
        }
        let m = self.inner.member(k, best);
        Ok((
            vectors(py, &m.polyline)?,
            flags(py, &m.mask),
            contacts(py, &m.contact_of),
            m.rejected_edges,
        ))
    }

    /// Via k's best centre, with `best`, else its input, and its rejected
    /// edges.
    fn via(&self, k: usize, best: bool) -> PyResult<((f64, f64), usize)> {
        if k >= self.inner.shared().setup.vias.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "no via {k}"
            )));
        }
        let v = self.inner.via(k, best);
        Ok(((v.position.x, v.position.y), v.rejected_edges))
    }

    /// Whether nothing oversteps at `polylines` and `positions`: no member
    /// has more violations than it started with, and no via is newly or
    /// deeper inside its clearance or its stops. `seen` is what the
    /// members see at `polylines`.
    fn guard(
        &self,
        polylines: Vec<Polyline<'_>>,
        masks: Vec<BoolArg<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
    ) -> PyResult<bool> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        let masks = masks_of(&masks, &polylines)?;
        Ok(self
            .inner
            .guard(&polylines, &masks, &positions, &seen.inner))
    }

    /// Per member at `polylines`, its violations of its static copper, the
    /// other members and the vias at `positions` together, as `seen` shows
    /// them.
    fn violations(
        &self,
        polylines: Vec<Polyline<'_>>,
        masks: Vec<BoolArg<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
    ) -> PyResult<Vec<usize>> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        let masks = masks_of(&masks, &polylines)?;
        Ok(self
            .inner
            .shared()
            .violations(&polylines, &masks, &positions, &seen.inner))
    }

    /// Per member, its vertices at `polylines` deeper than PRESSED_DEPTH
    /// inside the clearance of the other members and the vias at
    /// `positions`, which resampling keeps where they are.
    fn pressed<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
    ) -> PyResult<Vec<Flags<'py>>> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        Ok(self
            .inner
            .pressed(&polylines, &positions, &seen.inner)
            .iter()
            .map(|p| flags(py, p))
            .collect())
    }

    /// Each moving via's free distance below `below` to each obstacle of
    /// the copper of other nets at `polylines` and `positions`.
    fn via_distances(
        &self,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        below: f64,
    ) -> PyResult<FreeDistances> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        Ok(FreeDistances {
            inner: self
                .inner
                .shared()
                .via_distances(&polylines, &positions, below),
        })
    }

    /// The variables of a burst from `starts` and the current centres.
    fn layout(&self, starts: Vec<Polyline<'_>>) -> PyResult<BurstLayout> {
        Ok(BurstLayout {
            inner: self.layout_of(&starts)?,
            positions: self.inner.now.positions.clone(),
        })
    }

    /// Each (member, member or via) whose clearance the member's edges at
    /// `polylines` enter, the vias at `positions` numbered after the
    /// members, as `seen` shows them.
    fn entered(
        &self,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
    ) -> PyResult<Vec<(usize, usize)>> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        Ok(self
            .inner
            .shared()
            .entered(&polylines, &positions, &seen.inner)
            .into_iter()
            .collect())
    }

    /// For each member, then each via, whether at `polylines` and
    /// `positions` it oversteps, enters a member or via it did not enter
    /// at the start, `at_start` (`entered`), or is entered so.
    fn failing<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        masks: Vec<BoolArg<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
        at_start: Vec<(usize, usize)>,
    ) -> PyResult<Flags<'py>> {
        let (polylines, positions) = self.state(&polylines, &positions)?;
        let masks = masks_of(&masks, &polylines)?;
        let failing = self.inner.failing(
            &polylines,
            &masks,
            &positions,
            &seen.inner,
            &at_start.into_iter().collect(),
        );
        Ok(flags(py, &failing))
    }

    /// Each flat coordinate's scale for a burst from `starts`, the
    /// stiffness taken as `seen` shows the members there, searching
    /// through `lookups` if given.
    #[pyo3(signature = (starts, seen, lookups = None))]
    fn scale<'py>(
        &self,
        py: Python<'py>,
        starts: Vec<Polyline<'_>>,
        seen: PyRef<'_, Seen>,
        lookups: Option<PyRefMut<'_, Lookups>>,
    ) -> PyResult<Floats<'py>> {
        let layout = self.layout_of(&starts)?;
        let mut lookups = lookups;
        let stiffness =
            self.inner
                .stiffness(&layout, &seen.inner, lookups.as_mut().map(|l| &mut l.inner));
        Ok(floats(
            py,
            layout::scale(&stiffness, self.inner.shared().settings().spacing),
        ))
    }

    /// Take `earlier`'s best state, keeping the sections held only here as
    /// drawn, and put back where they started the members and vias that
    /// would come too close; returns how many members that move and how
    /// many vias went back. Raises ValueError if the energy there is not
    /// finite.
    fn go_on_from(
        &mut self,
        py: Python<'_>,
        earlier: PyRef<'_, ProblemState>,
    ) -> PyResult<(usize, usize)> {
        let shared = self.inner.shared();
        let other = earlier.inner.shared();
        check_count(
            "members",
            other.setup.members.len(),
            shared.setup.members.len(),
        )?;
        check_count("vias", other.setup.vias.len(), shared.setup.vias.len())?;
        self.inner
            .go_on_from(&earlier.inner)
            .map_err(|e| not_finite(py, e.0))
    }
}

/// How a burst's variables make the members' polylines and the vias'
/// centres (`kibends_core::optimize::problem::Layout`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct BurstLayout {
    inner: Layout,
    positions: Vec<V2>,
}

#[gen_stub_pymethods]
#[pymethods]
impl BurstLayout {
    /// Each member's start.
    #[getter]
    fn starts<'py>(&self, py: Python<'py>) -> PyResult<Vec<Points<'py>>> {
        vectors_list(py, &self.inner.starts)
    }

    /// The vias' centres at the start.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.positions)
    }

    /// Per member, each vertex's variable, -1 where it stays.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> Vec<Indices<'py>> {
        self.inner.index.iter().map(|i| contacts(py, i)).collect()
    }

    /// Per member, each vertex's offset from its variable.
    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> PyResult<Vec<Points<'py>>> {
        vectors_list(py, &self.inner.offsets)
    }

    /// Each via's variable.
    #[getter]
    fn via_index<'py>(&self, py: Python<'py>) -> Indices<'py> {
        crate::field::indices(py, &self.inner.via_index)
    }

    /// How many variables there are.
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    /// The variables at the start, flat.
    #[getter]
    fn x0<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, layout::start_of(&self.inner, &self.positions))
    }

    /// The members' polylines and the vias' centres at `x`, flat.
    fn unflatten<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(Vec<Points<'py>>, Vec<Floats<'py>>)> {
        let x = x.as_array().to_vec();
        check_count("coordinates", x.len(), 2 * self.inner.count)?;
        let (polylines, positions) = self.inner.unflatten(&x);
        Ok((
            vectors_list(py, &polylines)?,
            positions
                .iter()
                .map(|p| floats(py, vec![p.x, p.y]))
                .collect(),
        ))
    }
}

/// What a burst's check moves back (`kibends_core::optimize::checks`):
/// called with a point, flat, it moves back in place what newly crosses an
/// obstacle and returns how many crossings the point had.
#[gen_stub_pyclass]
#[pyclass(module = "kibends._core")]
pub struct Revert {
    check: Check,
    report: Py<PyAny>,
    /// The coordinates its last call moved back.
    reverted: Vec<bool>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Revert {
    fn __call__(
        &mut self,
        py: Python<'_>,
        mut candidate: PyReadwriteArray1<'_, f64>,
    ) -> PyResult<usize> {
        let mut x = candidate.as_array().to_vec();
        check_count("coordinates", x.len(), self.check.start().len())?;
        let reverted = self.check.revert(&mut x);
        for (c, v) in candidate.as_array_mut().iter_mut().zip(x) {
            *c = v;
        }
        self.reverted = reverted.mask;
        if reverted.fell_back {
            self.report.call1(py, (FallbackNotice,))?;
        }
        Ok(reverted.crossings)
    }

    /// The coordinates its last call moved back.
    #[getter]
    fn reverted<'py>(&self, py: Python<'py>) -> Flags<'py> {
        flags(py, &self.reverted)
    }
}

/// A burst's crossings named nothing more to move back, and the whole
/// burst moved back.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct FallbackNotice;

#[gen_stub_pymethods]
#[pymethods]
impl FallbackNotice {
    #[new]
    fn new() -> Self {
        FallbackNotice
    }
}

/// L-BFGS ended a burst of a problem abnormally, `streak` in a row.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct AbnormalNotice {
    /// The problem, by its place among those advanced.
    #[pyo3(get)]
    problem: usize,
    #[pyo3(get)]
    streak: u32,
}

/// What one burst cost and moved (`kibends_core::optimize::BurstReport`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct BurstNotice {
    /// The problem, by its place among those advanced.
    #[pyo3(get)]
    problem: usize,
    /// L-BFGS's iterations.
    #[pyo3(get)]
    iterations: u64,
    /// Its energy evaluations.
    #[pyo3(get)]
    evaluations: u64,
    /// Seconds the evaluations took.
    #[pyo3(get)]
    evaluating: f64,
    /// Seconds the rest of the step took.
    #[pyo3(get)]
    rest: f64,
    /// What L-BFGS says ended it; empty for a burst that did not run it.
    #[pyo3(get)]
    message: String,
    /// The lowest energy the problem's bursts have reached, as the stall
    /// rule takes it.
    #[pyo3(get)]
    lowest: f64,
    /// The problem's tracks' summed length after the burst, mm.
    #[pyo3(get)]
    length: f64,
    /// The problem's free members, and each one's length and shape change.
    #[pyo3(get)]
    members: Vec<usize>,
    #[pyo3(get)]
    lengths: Vec<f64>,
    #[pyo3(get)]
    changes: Vec<f64>,
}

/// After an outer iteration: its number, its largest change, and each
/// problem's length and energy.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ProgressNotice {
    #[pyo3(get)]
    iteration: u64,
    #[pyo3(get)]
    change: f64,
    #[pyo3(get)]
    lengths: Vec<f64>,
    #[pyo3(get)]
    energies: Vec<f64>,
}

fn notice(py: Python<'_>, notice: Notice) -> PyResult<Py<PyAny>> {
    Ok(match notice {
        Notice::Abnormal { problem, streak } => {
            Py::new(py, AbnormalNotice { problem, streak })?.into_any()
        }
        Notice::Burst(b) => Py::new(
            py,
            BurstNotice {
                problem: b.problem,
                iterations: b.iterations,
                evaluations: b.evaluations,
                evaluating: b.evaluating,
                rest: b.rest,
                message: b.message,
                lowest: b.lowest,
                length: b.length,
                members: b.members,
                lengths: b.lengths,
                changes: b.changes,
            },
        )?
        .into_any(),
        Notice::Progress {
            iteration,
            change,
            lengths,
            energies,
        } => Py::new(
            py,
            ProgressNotice {
                iteration,
                change,
                lengths,
                energies,
            },
        )?
        .into_any(),
    })
}

/// Bursts that run through a Python callable.
struct PythonBursts<'a, 'py> {
    py: Python<'py>,
    burst: &'a Bound<'py, PyAny>,
    report: &'a Py<PyAny>,
}

impl Bursts<Failure> for PythonBursts<'_, '_> {
    fn run(
        &mut self,
        objective: problem::Objective,
        lookups: problem::Lookups,
        start: Vec<f64>,
        scale: Vec<f64>,
        check: Check,
    ) -> Result<(Ran, Option<Round>), Failure> {
        let py = self.py;
        let seen = objective.seen.clone();
        let lookups = Py::new(
            py,
            Lookups {
                inner: lookups,
                seen,
            },
        )?;
        let objective = Py::new(py, Objective::of_problem(objective, Some(lookups)))?;
        let revert = Bound::new(
            py,
            Revert {
                check,
                report: self.report.clone_ref(py),
                reverted: Vec::new(),
            },
        )?;
        let result = self.burst.call1((
            objective,
            floats(py, start),
            floats(py, scale),
            revert.clone(),
        ))?;
        let points: PyReadonlyArray1<'_, f64> = result.getattr("points")?.extract()?;
        let ran = Ran {
            point: points.as_array().to_vec(),
            abnormal: result.getattr("abnormal")?.extract()?,
            iterations: result.getattr("iterations")?.extract()?,
            evaluations: result.getattr("evaluations")?.extract()?,
            message: result.getattr("message")?.extract()?,
            cut: result.getattr("cut")?.extract()?,
            evaluating: result.getattr("evaluating")?.extract()?,
        };
        let found = revert.borrow().check.found.clone();
        Ok((ran, found))
    }
}

/// Outer iterations of `problems` from `iteration` until `cap`, each giving
/// every problem that has not settled one burst, until every problem has
/// settled or `expired`, asked after each burst, answers true. Each burst
/// runs through `burst`, called with its objective, start, per-coordinate
/// scale and check; `report` hears each notice as it comes. `gave_up` says
/// some problem was given up on before. Returns how the run ended, as
/// `kibends.results.Termination`'s value names it, and the iteration it
/// reached.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (problems, iteration, cap, gave_up, expired, burst, report))]
#[allow(clippy::too_many_arguments)]
fn advance(
    py: Python<'_>,
    problems: Vec<Bound<'_, ProblemState>>,
    iteration: u64,
    cap: u64,
    gave_up: bool,
    #[gen_stub(override_type(
        type_repr = "typing.Optional[collections.abc.Callable[[], builtins.bool]]",
        imports = ("typing", "collections.abc", "builtins")
    ))]
    expired: Option<Bound<'_, PyAny>>,
    #[gen_stub(override_type(
        type_repr = "collections.abc.Callable[[Objective, numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], Revert], bursts.Burst]",
        imports = ("collections.abc", "numpy", "numpy.typing", "kibends.bursts")
    ))]
    burst: Bound<'_, PyAny>,
    #[gen_stub(override_type(
        type_repr = "collections.abc.Callable[[FallbackNotice | AbnormalNotice | BurstNotice | ProgressNotice], object]",
        imports = ("collections.abc",)
    ))]
    report: Py<PyAny>,
) -> PyResult<(&'static str, u64)> {
    let mut borrowed: Vec<PyRefMut<'_, ProblemState>> = problems
        .iter()
        .map(|p| p.try_borrow_mut())
        .collect::<Result<_, _>>()?;
    let mut states: Vec<&mut State> = borrowed.iter_mut().map(|p| &mut p.inner).collect();
    let mut iteration = iteration;
    let mut asked =
        expired.map(|e| move || -> Result<bool, Failure> { Ok(e.call0()?.is_truthy()?) });
    let mut told = |n: Notice| -> Result<(), Failure> {
        report.call1(py, (notice(py, n)?,))?;
        Ok(())
    };
    let mut bursts = PythonBursts {
        py,
        burst: &burst,
        report: &report,
    };
    let outcome = core::advance(
        &mut states,
        &mut iteration,
        cap,
        gave_up,
        asked
            .as_mut()
            .map(|a| a as &mut dyn FnMut() -> Result<bool, Failure>),
        &mut bursts,
        &mut told,
    )
    .map_err(|e| e.into_py(py))?;
    let word = match outcome {
        Outcome::NothingToDo => "nothing to do",
        Outcome::Converged => "converged",
        Outcome::GaveUp => "stopped on abnormal bursts",
        Outcome::IterationCap => "iteration cap reached",
        Outcome::TimeLimit => "time limit reached",
    };
    Ok((word, iteration))
}

/// Whether a problem of `members` moving members and `vias` moving vias
/// has stalled: over `window` bursts its `lowest` energy fell by no more
/// than `tolerance` for each member and via, and its tracks' summed
/// `lengths` by no more than one tolerance
/// (`kibends_core::optimize::stalled`).
#[gen_stub_pyfunction]
#[pyfunction]
fn stalled(
    lowest: Vec<f64>,
    lengths: Vec<f64>,
    window: usize,
    tolerance: f64,
    members: usize,
    vias: usize,
) -> PyResult<bool> {
    check_count("lengths", lengths.len(), lowest.len())?;
    let settings = Settings {
        spacing: 1.0,
        settled_change: 0.0,
        stall_window: window,
        stall_tolerance: tolerance,
        burst: burst::Settings {
            max_iterations: 0,
            ftol: 0.0,
            gtol: 0.0,
        },
    };
    Ok(core::stalled(&lowest, &lengths, &settings, members, vias))
}

/// Registers the run loop's classes and function in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ProblemState>()?;
    m.add_class::<Revert>()?;
    m.add_class::<BurstLayout>()?;
    m.add_class::<FallbackNotice>()?;
    m.add_class::<AbnormalNotice>()?;
    m.add_class::<BurstNotice>()?;
    m.add_class::<ProgressNotice>()?;
    m.add_function(wrap_pyfunction!(advance, m)?)?;
    m.add_function(wrap_pyfunction!(stalled, m)?)?;
    m.add("MAX_ABNORMAL_BURSTS", core::MAX_ABNORMAL_BURSTS)?;
    m.add("PRESSED_DEPTH", core::PRESSED_DEPTH)?;
    m.add("INSIDE_TOLERANCE", core::INSIDE_TOLERANCE)?;
    Ok(())
}

pyo3_stub_gen::module_variable!(
    "kibends._core",
    "MAX_ABNORMAL_BURSTS",
    u32,
    core::MAX_ABNORMAL_BURSTS
);
pyo3_stub_gen::module_variable!("kibends._core", "PRESSED_DEPTH", f64, core::PRESSED_DEPTH);
pyo3_stub_gen::module_variable!(
    "kibends._core",
    "INSIDE_TOLERANCE",
    f64,
    core::INSIDE_TOLERANCE
);
