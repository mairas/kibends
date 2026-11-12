//! The energy of a problem as Python sees it: its members and coupled
//! sections (`Member`, `PairOffset`, `Members`), what they see of each
//! other (`Seen`, `NeighbourList`), what the moving vias see and their free
//! distances (`ViaSight`, `free_distances`, `newly_inside`), the whole
//! problem with a burst's objective (`Problem`, `Lookups`, `Objective`) and
//! the statistics of what the energy cost (`kibends_core::stats`).

use std::collections::BTreeMap;
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{BoolArg, BoolArray, IntpArg};
use crate::energy::Params;
use crate::field::{Flags, Floats, Indices, Points, floats, indices, points_of, vectors};
use crate::fields::FieldArg;
use crate::vias::ViaPairs;
use kibends_core::field::pairs::SmoothAnswer as CoreSmoothAnswer;
use kibends_core::geometry::V2;
use kibends_core::members::{self, Trace};
use kibends_core::neighbours;
use kibends_core::pair_blocks::{self, Pins};
use kibends_core::problem::{self, Layout};
use kibends_core::seen;
use kibends_core::stats;
use kibends_core::vias::{free, sight};

type Polyline<'py> = PyReadonlyArray2<'py, f64>;

fn polylines_of(arrays: &[Polyline<'_>]) -> PyResult<Vec<Vec<V2>>> {
    arrays.iter().map(points_of).collect()
}

fn refs(polylines: &[Vec<V2>]) -> Vec<&[V2]> {
    polylines.iter().map(Vec::as_slice).collect()
}

fn vectors_list<'py>(py: Python<'py>, values: &[Vec<V2>]) -> PyResult<Vec<Points<'py>>> {
    values.iter().map(|v| vectors(py, v)).collect()
}

fn invalid(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// The value of an energy that is not finite, as Python says it.
fn not_finite(py: Python<'_>, value: f64) -> PyErr {
    let said = PyFloat::new(py, value)
        .str()
        .map_or_else(|_| value.to_string(), |s| s.to_string());
    invalid(format!("energy is not finite: {said}"))
}

/// Per vertex, its index, -1 for none.
fn optional_indices(array: &IntpArg<'_>, below: usize) -> PyResult<Vec<Option<usize>>> {
    array
        .0
        .as_array()
        .iter()
        .map(|&i| match usize::try_from(i) {
            Ok(i) if i < below => Ok(Some(i)),
            _ if i == -1 => Ok(None),
            _ => Err(invalid(format!("index {i} out of range -1..{below}"))),
        })
        .collect()
}

pub(crate) fn check_count(what: &str, got: usize, want: usize) -> PyResult<()> {
    if got == want {
        Ok(())
    } else {
        Err(invalid(format!("{got} {what} for {want}")))
    }
}

/// Each polyline has as many entries of `what` as it has vertices.
fn check_rows<T>(what: &str, rows: &[Vec<T>], polylines: &[Vec<V2>]) -> PyResult<()> {
    check_count(what, rows.len(), polylines.len())?;
    for (row, points) in rows.iter().zip(polylines) {
        check_count(what, row.len(), points.len())?;
    }
    Ok(())
}

fn rings_of(rings: &[BTreeMap<usize, f64>], members: usize) -> PyResult<Vec<Vec<(usize, f64)>>> {
    rings
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|(&i, &d)| {
                    if i < members {
                        Ok((i, d))
                    } else {
                        Err(invalid(format!("member {i} out of range 0..{members}")))
                    }
                })
                .collect()
        })
        .collect()
}

/// A moving section's centreline needs an edge to offset.
fn check_centreline(offset: &pair_blocks::PairOffset, points: &[V2]) -> PyResult<()> {
    if !offset.held && points.len() < 2 {
        return Err(invalid("a section's centreline needs two vertices".into()));
    }
    Ok(())
}

fn check_sections(members: &members::Members, polylines: &[Vec<V2>]) -> PyResult<()> {
    for (member, points) in members.members.iter().zip(polylines) {
        if let Some(section) = &member.section {
            check_centreline(&section.offset, points)?;
        }
    }
    Ok(())
}

/// A coupled section's tracks as the offset of its centreline
/// (`kibends_core::pair_blocks::PairOffset`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct PairOffset {
    inner: pair_blocks::PairOffset,
}

#[gen_stub_pymethods]
#[pymethods]
impl PairOffset {
    /// Its `gap` and track `width`, mm, whether track a lies to the right
    /// of the centreline, whether it is `held`, the centreline's vertices
    /// `a_at` and `b_at` holding the tracks' own points `a` and `b`, and the
    /// tracks as given, which a held section keeps.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gap: f64,
        width: f64,
        a_on_right: bool,
        held: bool,
        a_at: Polyline<'_>,
        a: Polyline<'_>,
        b_at: Polyline<'_>,
        b: Polyline<'_>,
        track_a: Polyline<'_>,
        track_b: Polyline<'_>,
    ) -> PyResult<Self> {
        let pins = Pins {
            a_at: points_of(&a_at)?,
            a: points_of(&a)?,
            b_at: points_of(&b_at)?,
            b: points_of(&b)?,
        };
        check_count("track a pins", pins.a.len(), pins.a_at.len())?;
        check_count("track b pins", pins.b.len(), pins.b_at.len())?;
        Ok(Self {
            inner: pair_blocks::PairOffset {
                gap,
                width,
                a_on_right,
                held,
                pins,
                a: points_of(&track_a)?,
                b: points_of(&track_b)?,
            },
        })
    }

    /// Tracks a and b along the centreline at `points`, through the
    /// section's own ends and pins.
    fn offset<'py>(
        &self,
        py: Python<'py>,
        points: Polyline<'_>,
    ) -> PyResult<(Points<'py>, Points<'py>)> {
        let points = points_of(&points)?;
        check_centreline(&self.inner, &points)?;
        let (a, b) = self.inner.offset(&points);
        Ok((vectors(py, &a)?, vectors(py, &b)?))
    }

    /// The gradient of a function of `offset(points)`, `at_a` and `at_b` at
    /// the vertices of tracks a and b, taken back to the centreline's
    /// vertices.
    fn offset_pullback<'py>(
        &self,
        py: Python<'py>,
        points: Polyline<'_>,
        at_a: Polyline<'_>,
        at_b: Polyline<'_>,
    ) -> PyResult<Points<'py>> {
        let points = points_of(&points)?;
        let (at_a, at_b) = (points_of(&at_a)?, points_of(&at_b)?);
        check_count("gradients of track a", at_a.len(), points.len())?;
        check_count("gradients of track b", at_b.len(), points.len())?;
        vectors(py, &self.inner.offset_pullback(&points, &at_a, &at_b))
    }
}

/// A track's net, width and clearance.
type TraceArg = (String, f64, f64);

fn trace((net, width, clearance): TraceArg) -> Trace {
    Trace {
        net,
        width,
        clearance,
    }
}

/// A chain, or a coupled section as its centreline
/// (`kibends_core::members::Member`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Member {
    inner: members::Member,
}

#[gen_stub_pymethods]
#[pymethods]
impl Member {
    /// On `layer`, keeping clear of the others as the track `viewer`,
    /// ignoring the tracks of the nets `ignored`, showing the others the
    /// tracks `shown`, in its static `field`, its contact stops weighed by
    /// `width` and its contacts' regions `contacts`; a coupled section with
    /// its `section` offset and the bend-radius `limit` its hinge keeps,
    /// none without the hinge.
    #[new]
    #[pyo3(signature = (layer, viewer, ignored, shown, field, width, contacts, section = None, limit = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        layer: String,
        viewer: TraceArg,
        ignored: Vec<String>,
        shown: Vec<TraceArg>,
        field: FieldArg,
        width: f64,
        contacts: Vec<FieldArg>,
        section: Option<PyRef<'_, PairOffset>>,
        limit: Option<f64>,
    ) -> PyResult<Self> {
        let parts = if section.is_some() { 2 } else { 1 };
        check_count("shown tracks", shown.len(), parts)?;
        Ok(Self {
            inner: members::Member {
                layer,
                viewer: trace(viewer),
                ignored,
                shown: shown.into_iter().map(trace).collect(),
                field: field.0,
                width,
                contacts: contacts.into_iter().map(|f| f.0).collect(),
                section: section.map(|s| members::Section {
                    offset: s.inner.clone(),
                    limit,
                }),
            },
        })
    }
}

/// The members of a problem (`kibends_core::members::Members`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Members {
    inner: Arc<members::Members>,
}

impl Members {
    fn check_polylines(&self, polylines: &[Vec<V2>]) -> PyResult<()> {
        check_count("polylines", polylines.len(), self.inner.len())?;
        check_sections(&self.inner, polylines)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Members {
    #[new]
    fn new(members: Vec<PyRef<'_, Member>>) -> Self {
        Self {
            inner: Arc::new(members::Members::new(
                members.iter().map(|m| m.inner.clone()).collect(),
            )),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// What each member sees of the others with the members at
    /// `polylines`, each via of `rings` seen by the members it names,
    /// inflated by the distance it gives.
    #[pyo3(signature = (polylines, rings = Vec::new()))]
    fn seen(
        &self,
        polylines: Vec<Polyline<'_>>,
        rings: Vec<BTreeMap<usize, f64>>,
    ) -> PyResult<Seen> {
        let polylines = polylines_of(&polylines)?;
        self.check_polylines(&polylines)?;
        let rings = rings_of(&rings, self.inner.len())?;
        Ok(Seen::of(self.inner.seen(&refs(&polylines), &rings)))
    }

    /// What the others see of every member at `polylines`, one after
    /// another, then each via's centre at `positions` twice.
    fn shown_flat<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
    ) -> PyResult<Points<'py>> {
        let polylines = polylines_of(&polylines)?;
        self.check_polylines(&polylines)?;
        let positions = points_of(&positions)?;
        vectors(py, &self.inner.shown_flat(&refs(&polylines), &positions))
    }

    /// The members' summed energy at `polylines` as `seen` shows them, with
    /// vias at `positions` among them; its gradient at every vertex, the
    /// vertices of `masks` zeroed, and at each via's centre. Raises
    /// ValueError if it is not finite.
    #[allow(clippy::too_many_arguments)]
    fn energy<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        masks: Vec<BoolArg<'_>>,
        params: Params,
        short_edge: f64,
        seen: PyRef<'_, Seen>,
        positions: Polyline<'_>,
    ) -> PyResult<(f64, Vec<Points<'py>>, Points<'py>)> {
        let polylines = polylines_of(&polylines)?;
        self.check_polylines(&polylines)?;
        let masks: Vec<Vec<bool>> = masks.iter().map(|m| m.0.as_array().to_vec()).collect();
        check_rows("mask entries", &masks, &polylines)?;
        let positions = points_of(&positions)?;
        let flat = self.inner.shown_flat(&refs(&polylines), &positions);
        seen.check(self.inner.len(), flat.len())?;
        let mask_refs: Vec<&[bool]> = masks.iter().map(Vec::as_slice).collect();
        let energy = self
            .inner
            .energy(
                &refs(&polylines),
                &mask_refs,
                &params.0,
                short_edge,
                &seen.inner,
                &positions,
                None,
                None,
            )
            .map_err(|v| not_finite(py, v))?;
        Ok((
            energy.value,
            vectors_list(py, &energy.gradients)?,
            vectors(py, &energy.via_gradients)?,
        ))
    }
}

/// Whether each edge of a member stays out of the pieces it sees, and what
/// those that do not enter (`kibends_core::seen::EdgesClear`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct EdgesClear {
    inner: seen::EdgesClear,
}

#[gen_stub_pymethods]
#[pymethods]
impl EdgesClear {
    /// Whether each edge stays out.
    #[getter]
    fn clear<'py>(&self, py: Python<'py>) -> Flags<'py> {
        BoolArray(self.inner.clear.clone().into_pyarray(py))
    }

    /// Per (edge, piece) pair entered, the edge.
    #[getter]
    fn edge<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.edge)
    }

    /// The first vertex of the piece's edge, numbered across what the
    /// members show, so that a via's is its first vertex there.
    #[getter]
    fn vertex<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.vertex)
    }

    /// The piece's obstacle, its track or its via (`Seen.obstacle`).
    #[getter]
    fn obstacle<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.obstacle)
    }

    /// How deep the edge is inside the piece.
    #[getter]
    fn depth<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.depth.clone())
    }
}

/// The smooth distance at each point and its gradient, and what answers
/// it: per obstacle that has a share at a point, the point, the piece
/// nearest it of that obstacle, its share and its unit gradient.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct SmoothAnswer {
    inner: CoreSmoothAnswer,
}

#[gen_stub_pymethods]
#[pymethods]
impl SmoothAnswer {
    #[getter]
    fn distance<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.distance.clone())
    }

    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.gradient)
    }

    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.point)
    }

    #[getter]
    fn edge<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.edge)
    }

    #[getter]
    fn share<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.share.clone())
    }

    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.direction)
    }
}

/// What each member of a problem sees of the others at one set of shapes
/// (`kibends_core::seen::Seen`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Seen {
    inner: Arc<seen::Seen>,
    /// How many vertices of what the members show it refers to.
    extent: usize,
}

impl Seen {
    fn of(inner: seen::Seen) -> Self {
        let extent = inner.vertex().iter().max().map_or(0, |v| v + 2);
        Self {
            inner: Arc::new(inner),
            extent,
        }
    }

    /// Whether it was made for `members` members showing at least as many
    /// vertices as `flat` holds.
    fn check(&self, members: usize, flat: usize) -> PyResult<()> {
        check_count("members", members, self.inner.member_count())?;
        if flat < self.extent {
            return Err(invalid(format!(
                "{flat} vertices shown for a view of {}",
                self.extent
            )));
        }
        Ok(())
    }

    fn queries<'a>(&self, queries: &'a [Vec<V2>]) -> PyResult<Vec<&'a [V2]>> {
        check_count("members' queries", queries.len(), self.inner.member_count())?;
        Ok(refs(queries))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Seen {
    /// How many layers its members are on.
    #[getter]
    fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    /// Each piece's obstacle for the smooth minimum: its track, or its via.
    #[getter]
    fn obstacle<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.obstacle())
    }

    /// Per member, the signed distance from each of its `queries` to the
    /// nearest piece it sees, +inf where it sees none; exact below `below`,
    /// elsewhere at least `below`. `flat` is what the members show.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<Polyline<'_>>,
        flat: Polyline<'_>,
        below: f64,
    ) -> PyResult<Vec<Floats<'py>>> {
        let queries = polylines_of(&queries)?;
        let flat = points_of(&flat)?;
        self.check(self.inner.member_count(), flat.len())?;
        Ok(self
            .inner
            .distances(&self.queries(&queries)?, &flat, below)
            .into_iter()
            .map(|d| floats(py, d))
            .collect())
    }

    /// Per member at `polylines`, whether each of its edges stays out of
    /// the pieces it sees at `flat`, entering by at most `tolerance`, and
    /// what each edge that does not enters.
    #[pyo3(signature = (polylines, flat, tolerance = 0.0))]
    fn segments_clear(
        &self,
        polylines: Vec<Polyline<'_>>,
        flat: Polyline<'_>,
        tolerance: f64,
    ) -> PyResult<Vec<EdgesClear>> {
        let polylines = polylines_of(&polylines)?;
        let flat = points_of(&flat)?;
        self.check(self.inner.member_count(), flat.len())?;
        Ok(self
            .inner
            .segments_clear(&self.queries(&polylines)?, &flat, tolerance)
            .into_iter()
            .map(|inner| EdgesClear { inner })
            .collect())
    }

    /// Per member, the smooth distance from each of its `queries` to the
    /// pieces it sees at `flat`, with `reach`, and what answers it; with
    /// `listed`, made for this, each layer looks only at the pairs it lists.
    #[pyo3(signature = (queries, flat, reach, listed = None))]
    fn smooth_answers(
        &self,
        queries: Vec<Polyline<'_>>,
        flat: Polyline<'_>,
        reach: f64,
        listed: Option<PyRefMut<'_, NeighbourList>>,
    ) -> PyResult<Vec<SmoothAnswer>> {
        let queries = polylines_of(&queries)?;
        let flat = points_of(&flat)?;
        self.check(self.inner.member_count(), flat.len())?;
        let mut listed = listed;
        let list = match listed.as_deref_mut() {
            Some(l) => {
                l.check(&self.inner)?;
                Some(&mut l.inner)
            }
            None => None,
        };
        Ok(self
            .inner
            .smooth_answers(&self.queries(&queries)?, &flat, reach, list)
            .into_iter()
            .map(|inner| SmoothAnswer { inner })
            .collect())
    }
}

/// A Verlet list of the pairs of a `Seen` (`kibends_core::neighbours::
/// NeighbourList`).
#[gen_stub_pyclass]
#[pyclass(module = "kibends._core")]
pub struct NeighbourList {
    inner: neighbours::NeighbourList,
    seen: Arc<seen::Seen>,
}

impl NeighbourList {
    fn check(&self, seen: &Arc<seen::Seen>) -> PyResult<()> {
        if Arc::ptr_eq(&self.seen, seen) {
            Ok(())
        } else {
            Err(invalid(
                "a neighbour list answers only the view it was made for".into(),
            ))
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl NeighbourList {
    /// A list for `seen`'s layers, for an energy whose terms end at `reach`.
    #[new]
    fn new(seen: PyRef<'_, Seen>, reach: f64) -> Self {
        Self {
            inner: neighbours::NeighbourList::new(&seen.inner, reach),
            seen: seen.inner.clone(),
        }
    }
}

/// (via, obstacle) pairs and each via's exact distance to its obstacle
/// (`kibends_core::vias::sight::ObstacleDistances`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ObstacleDistances {
    inner: sight::ObstacleDistances,
}

#[gen_stub_pymethods]
#[pymethods]
impl ObstacleDistances {
    #[getter]
    fn key<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.key)
    }

    #[getter]
    fn via<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.via)
    }

    /// A member by its index, a via by the number of members plus its.
    #[getter]
    fn obstacle<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.obstacle)
    }

    #[getter]
    fn distance<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.distance.clone())
    }

    /// The member's edge's first vertex, or the via's index.
    #[getter]
    fn vertex<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.vertex)
    }
}

/// Each moving via's smooth distance and its gradient, and what answers it
/// (`kibends_core::vias::sight::ViaSmooth`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ViaSmooth {
    inner: sight::ViaSmooth,
}

#[gen_stub_pymethods]
#[pymethods]
impl ViaSmooth {
    #[getter]
    fn distance<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.distance.clone())
    }

    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.gradient)
    }

    /// The via each answer is for.
    #[getter]
    fn point<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.point)
    }

    /// The member that answers, -1 for a via.
    #[getter]
    fn member<'py>(&self, py: Python<'py>) -> Indices<'py> {
        let member: Vec<isize> = self
            .inner
            .member
            .iter()
            .map(|m| m.and_then(|m| isize::try_from(m).ok()).unwrap_or(-1))
            .collect();
        crate::arrays::IntpArray(member.into_pyarray(py))
    }

    /// The member's edge's first vertex, or the via.
    #[getter]
    fn vertex<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.vertex)
    }

    #[getter]
    fn starts<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.starts)
    }

    #[getter]
    fn ends<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.ends)
    }

    #[getter]
    fn share<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.share.clone())
    }

    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> PyResult<Points<'py>> {
        vectors(py, &self.inner.direction)
    }
}

/// What each moving via sees of the members and the other vias
/// (`kibends_core::vias::sight::ViaSight`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ViaSight {
    inner: sight::ViaSight,
    members: usize,
    vias: usize,
}

impl ViaSight {
    fn inputs(
        &self,
        polylines: &[Polyline<'_>],
        positions: &Polyline<'_>,
    ) -> PyResult<(Vec<Vec<V2>>, Vec<V2>)> {
        let polylines = polylines_of(polylines)?;
        let positions = points_of(positions)?;
        check_count("polylines", polylines.len(), self.members)?;
        check_count("positions", positions.len(), self.vias)?;
        Ok((polylines, positions))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ViaSight {
    /// Each via, by its index, seeing the members its `rings` names and the
    /// vias its `apart` names, at those distances, among `members`.
    #[new]
    fn new(
        rings: Vec<BTreeMap<usize, f64>>,
        apart: Vec<BTreeMap<usize, f64>>,
        members: usize,
    ) -> PyResult<Self> {
        check_count("vias apart", apart.len(), rings.len())?;
        let rings = rings_of(&rings, members)?;
        let apart = rings_of(&apart, rings.len())?;
        Ok(Self {
            inner: sight::ViaSight::new(&rings, &apart, members),
            members,
            vias: rings.len(),
        })
    }

    /// Each via's exact distance to the nearest of what it sees, with the
    /// members at `polylines` and the vias at `positions`, within `reach`.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        reach: f64,
    ) -> PyResult<Floats<'py>> {
        let (polylines, positions) = self.inputs(&polylines, &positions)?;
        Ok(floats(
            py,
            self.inner.distances(&refs(&polylines), &positions, reach),
        ))
    }

    /// Each via's exact distance to each member and each via it sees, where
    /// it is below `below`.
    fn obstacle_distances(
        &self,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        below: f64,
    ) -> PyResult<ObstacleDistances> {
        let (polylines, positions) = self.inputs(&polylines, &positions)?;
        Ok(ObstacleDistances {
            inner: self
                .inner
                .obstacle_distances(&refs(&polylines), &positions, below),
        })
    }

    /// Each via's smooth distance to what it sees, with `reach`, and what
    /// answers it.
    fn smooth_answers(
        &self,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        reach: f64,
    ) -> PyResult<ViaSmooth> {
        let (polylines, positions) = self.inputs(&polylines, &positions)?;
        Ok(ViaSmooth {
            inner: self
                .inner
                .smooth_answers(&refs(&polylines), &positions, reach),
        })
    }
}

/// (via, obstacle) pairs of what stands still for each via and each via's
/// exact distance to its obstacle (`kibends_core::vias::free::StaticFree`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct StaticFree {
    inner: free::StaticFree,
}

#[gen_stub_pymethods]
#[pymethods]
impl StaticFree {
    #[getter]
    fn key<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.key)
    }

    #[getter]
    fn via<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, &self.inner.via)
    }

    #[getter]
    fn distance<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.distance.clone())
    }
}

/// The free distance of each via to each obstacle
/// (`kibends_core::vias::free::FreeDistances`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct FreeDistances {
    inner: free::FreeDistances,
}

#[gen_stub_pymethods]
#[pymethods]
impl FreeDistances {
    /// To each of what stands still.
    #[getter]
    #[pyo3(name = "static")]
    fn fixed(&self) -> StaticFree {
        StaticFree {
            inner: self.inner.fixed.clone(),
        }
    }

    /// To each member and via its sight finds.
    #[getter]
    fn by_obstacle(&self) -> ObstacleDistances {
        ObstacleDistances {
            inner: self.inner.by_obstacle.clone(),
        }
    }
}

/// The free distance of each via to each obstacle nearer than `below`: to
/// each part of its `statics`, and to each of the members at `polylines`
/// and the vias at `positions` that `sight` finds
/// (`kibends_core::vias::free::free_distances`).
#[gen_stub_pyfunction]
#[pyfunction]
fn free_distances(
    statics: Vec<Vec<Option<FieldArg>>>,
    sight: PyRef<'_, ViaSight>,
    polylines: Vec<Polyline<'_>>,
    positions: Polyline<'_>,
    below: f64,
) -> PyResult<FreeDistances> {
    let (polylines, positions) = sight.inputs(&polylines, &positions)?;
    check_count("vias' statics", statics.len(), positions.len())?;
    let statics: Vec<free::StaticParts> = statics
        .into_iter()
        .map(|parts| parts.into_iter().map(|f| f.map(|f| f.0)).collect())
        .collect();
    Ok(FreeDistances {
        inner: free::free_distances(&statics, &sight.inner, &refs(&polylines), &positions, below),
    })
}

/// Which of the pairs of what stands still and of what the vias see, at
/// `now`, are newly inside by more than `tolerance`, or more than `slack`
/// deeper inside than at `start` (`kibends_core::vias::free::newly_inside`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (now, start, tolerance, slack = 0.0))]
fn newly_inside<'py>(
    py: Python<'py>,
    now: PyRef<'_, FreeDistances>,
    start: PyRef<'_, FreeDistances>,
    tolerance: f64,
    slack: f64,
) -> (Flags<'py>, Flags<'py>) {
    let (fixed, seen) = free::newly_inside(&now.inner, &start.inner, tolerance, slack);
    (
        BoolArray(fixed.into_pyarray(py)),
        BoolArray(seen.into_pyarray(py)),
    )
}

/// What one objective's evaluations search through
/// (`kibends_core::problem::Lookups`).
#[gen_stub_pyclass]
#[pyclass(module = "kibends._core")]
pub struct Lookups {
    inner: problem::Lookups,
    seen: Arc<seen::Seen>,
}

/// The members, moving vias and terms of one problem
/// (`kibends_core::problem::Problem`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Problem {
    inner: Arc<problem::Problem>,
}

/// The polylines and positions of a problem, checked against it.
struct State {
    polylines: Vec<Vec<V2>>,
    positions: Vec<V2>,
}

impl Problem {
    fn state(&self, polylines: &[Polyline<'_>], positions: &Polyline<'_>) -> PyResult<State> {
        let polylines = polylines_of(polylines)?;
        let positions = points_of(positions)?;
        check_count("polylines", polylines.len(), self.inner.members.len())?;
        check_count("positions", positions.len(), self.inner.vias.len())?;
        check_sections(&self.inner.members, &polylines)?;
        Ok(State {
            polylines,
            positions,
        })
    }

    fn check_seen(&self, seen: &Seen, state: &State) -> PyResult<()> {
        let flat = self
            .inner
            .members
            .shown_flat(&refs(&state.polylines), &state.positions);
        seen.check(self.inner.members.len(), flat.len())
    }

    fn contact_of(
        &self,
        contact_of: &[IntpArg<'_>],
        polylines: &[Vec<V2>],
    ) -> PyResult<Vec<Vec<Option<usize>>>> {
        let rows = contact_of
            .iter()
            .zip(&self.inner.members.members)
            .map(|(c, m)| optional_indices(c, m.contacts.len()))
            .collect::<PyResult<Vec<_>>>()?;
        check_count("contacts", contact_of.len(), polylines.len())?;
        check_rows("contacts", &rows, polylines)?;
        Ok(rows)
    }

    fn lookups_of<'a>(
        seen: &Seen,
        lookups: Option<&'a mut PyRefMut<'_, Lookups>>,
    ) -> PyResult<Option<&'a mut problem::Lookups>> {
        match lookups {
            Some(l) => {
                if !Arc::ptr_eq(&l.seen, &seen.inner) {
                    return Err(invalid(
                        "lookups answer only the view they were made for".into(),
                    ));
                }
                Ok(Some(&mut l.inner))
            }
            None => Ok(None),
        }
    }
}

fn masks_of(masks: &[BoolArg<'_>], polylines: &[Vec<V2>]) -> PyResult<Vec<Vec<bool>>> {
    let masks: Vec<Vec<bool>> = masks.iter().map(|m| m.0.as_array().to_vec()).collect();
    check_rows("mask entries", &masks, polylines)?;
    Ok(masks)
}

#[gen_stub_pymethods]
#[pymethods]
impl Problem {
    /// The problem of `members`, the vias seen by the members `rings`
    /// names at those distances, each moving via with its `diameter`, its
    /// static `fields` and its `stops`, what the vias' full terms see
    /// (`repel`) and what their stops see (`stop`), the pairwise term
    /// `pairs`, the energy's `params` and its `short_edge`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        members: PyRef<'_, Members>,
        rings: Vec<BTreeMap<usize, f64>>,
        diameters: Vec<f64>,
        fields: Vec<FieldArg>,
        stops: Vec<Option<FieldArg>>,
        repel: PyRef<'_, ViaSight>,
        stop: PyRef<'_, ViaSight>,
        pairs: Option<PyRef<'_, ViaPairs>>,
        params: Params,
        short_edge: f64,
    ) -> PyResult<Self> {
        let count = members.inner.len();
        let vias = diameters.len();
        for (what, got) in [
            ("rings", rings.len()),
            ("via fields", fields.len()),
            ("via stops", stops.len()),
            ("vias seen by the full terms", repel.vias),
            ("vias seen by the stops", stop.vias),
        ] {
            check_count(what, got, vias)?;
        }
        check_count("members seen by the vias", repel.members, count)?;
        check_count("members seen by the stops", stop.members, count)?;
        if let Some(pairs) = &pairs {
            check_count("paired vias", pairs.inner.via_count(), vias)?;
        }
        let vias = diameters
            .into_iter()
            .zip(fields)
            .zip(stops)
            .map(|((diameter, field), stops)| problem::ProblemVia {
                diameter,
                field: field.0,
                stops: stops.map(|s| s.0),
            })
            .collect();
        Ok(Self {
            inner: Arc::new(problem::Problem {
                members: members.inner.as_ref().clone(),
                rings: rings_of(&rings, count)?,
                vias,
                repel: repel.inner.clone(),
                stop: stop.inner.clone(),
                pairs: pairs.map(|p| p.inner.clone()),
                params: params.0,
                short_edge,
            }),
        })
    }

    /// What each member sees of the others and of the vias with the
    /// members at `polylines`.
    fn seen(&self, polylines: Vec<Polyline<'_>>) -> PyResult<Seen> {
        let polylines = polylines_of(&polylines)?;
        check_count("polylines", polylines.len(), self.inner.members.len())?;
        Ok(Seen::of(self.inner.seen(&refs(&polylines))))
    }

    /// What the others see of every member at `polylines`, one after
    /// another, then each via's centre at `positions` twice.
    fn shown_flat<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
    ) -> PyResult<Points<'py>> {
        let state = self.state(&polylines, &positions)?;
        vectors(
            py,
            &self
                .inner
                .members
                .shown_flat(&refs(&state.polylines), &state.positions),
        )
    }

    /// The lists for the evaluations of one objective, made for `seen`.
    fn lookups(&self, seen: PyRef<'_, Seen>) -> PyResult<Lookups> {
        check_count(
            "members",
            seen.inner.member_count(),
            self.inner.members.len(),
        )?;
        Ok(Lookups {
            inner: self.inner.lookups(&seen.inner),
            seen: seen.inner.clone(),
        })
    }

    /// The problem's energy at `polylines` and `positions`, the members'
    /// terms with the vertices of `masks` zeroed and each vertex's contact
    /// by `contact_of`, -1 for none, as `seen` shows them, and its gradient
    /// at every vertex and every via's centre; searching through
    /// `lookups`, made for `seen`, if given. Raises ValueError if the
    /// members' energy is not finite.
    #[pyo3(signature = (polylines, positions, masks, contact_of, seen, lookups = None))]
    #[allow(clippy::too_many_arguments)]
    fn total<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        masks: Vec<BoolArg<'_>>,
        contact_of: Vec<IntpArg<'_>>,
        seen: PyRef<'_, Seen>,
        lookups: Option<PyRefMut<'_, Lookups>>,
    ) -> PyResult<(f64, Vec<Points<'py>>, Points<'py>)> {
        let state = self.state(&polylines, &positions)?;
        self.check_seen(&seen, &state)?;
        let masks = masks_of(&masks, &state.polylines)?;
        let contact_of = self.contact_of(&contact_of, &state.polylines)?;
        let mut lookups = lookups;
        let lookups = Self::lookups_of(&seen, lookups.as_mut())?;
        let energy = self
            .inner
            .total(
                &refs(&state.polylines),
                &state.positions,
                &masks.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &contact_of.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &seen.inner,
                lookups,
            )
            .map_err(|v| not_finite(py, v))?;
        Ok((
            energy.value,
            vectors_list(py, &energy.gradients)?,
            vectors(py, &energy.via_gradients)?,
        ))
    }

    /// The moving vias' own obstacle terms at `polylines` and `positions`,
    /// and their gradient at every vertex and at every via's centre.
    fn via_energy<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
    ) -> PyResult<(f64, Vec<Points<'py>>, Points<'py>)> {
        let state = self.state(&polylines, &positions)?;
        let energy = self
            .inner
            .via_energy(&refs(&state.polylines), &state.positions, None);
        Ok((
            energy.value,
            vectors_list(py, &energy.gradients)?,
            vectors(py, &energy.via_gradients)?,
        ))
    }

    /// Each moving via's stiffness under its own obstacle terms.
    fn via_stiffness<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
    ) -> PyResult<Floats<'py>> {
        let state = self.state(&polylines, &positions)?;
        Ok(floats(
            py,
            self.inner
                .via_stiffness(&refs(&state.polylines), &state.positions, None),
        ))
    }

    /// Per member, each vertex's stiffness under its obstacle term, and
    /// each via's under its own terms, at `polylines` and `positions` as
    /// `seen` shows them, searching through `lookups` if given.
    #[pyo3(signature = (polylines, positions, seen, lookups = None))]
    fn stiffness<'py>(
        &self,
        py: Python<'py>,
        polylines: Vec<Polyline<'_>>,
        positions: Polyline<'_>,
        seen: PyRef<'_, Seen>,
        lookups: Option<PyRefMut<'_, Lookups>>,
    ) -> PyResult<(Vec<Floats<'py>>, Floats<'py>)> {
        let state = self.state(&polylines, &positions)?;
        self.check_seen(&seen, &state)?;
        let mut lookups = lookups;
        let lookups = Self::lookups_of(&seen, lookups.as_mut())?;
        let (members, vias) = self.inner.stiffness(
            &refs(&state.polylines),
            &state.positions,
            &seen.inner,
            lookups,
        );
        Ok((
            members.into_iter().map(|s| floats(py, s)).collect(),
            floats(py, vias),
        ))
    }

    /// The problem's energy as a function of a burst's variables, laid out
    /// from `starts` with each vertex's variable in `index`, -1 where it
    /// stays, and its offset from it in `offsets`, each via's variable in
    /// `via_index`, of `count` variables; with what each member sees of the
    /// others as `seen`, the energy `masks` and each vertex's contact by
    /// `contact_of`, searching through `lookups` if given.
    #[pyo3(signature = (starts, index, offsets, via_index, count, masks, contact_of, seen, lookups = None))]
    #[allow(clippy::too_many_arguments)]
    fn objective(
        &self,
        py: Python<'_>,
        starts: Vec<Polyline<'_>>,
        index: Vec<IntpArg<'_>>,
        offsets: Vec<Polyline<'_>>,
        via_index: IntpArg<'_>,
        count: usize,
        masks: Vec<BoolArg<'_>>,
        contact_of: Vec<IntpArg<'_>>,
        seen: PyRef<'_, Seen>,
        lookups: Option<Py<Lookups>>,
    ) -> PyResult<Objective> {
        let starts = polylines_of(&starts)?;
        check_count("polylines", starts.len(), self.inner.members.len())?;
        let index = index
            .iter()
            .map(|i| optional_indices(i, count))
            .collect::<PyResult<Vec<_>>>()?;
        check_rows("variables", &index, &starts)?;
        let offsets = polylines_of(&offsets)?;
        check_rows("offsets", &offsets, &starts)?;
        let via_index = crate::field::indices_of(&via_index, count)?;
        check_count("via variables", via_index.len(), self.inner.vias.len())?;
        let masks = masks_of(&masks, &starts)?;
        let contact_of = self.contact_of(&contact_of, &starts)?;
        let positions = vec![V2::default(); via_index.len()];
        self.check_seen(
            &seen,
            &State {
                polylines: starts.clone(),
                positions,
            },
        )?;
        if let Some(l) = &lookups
            && !Arc::ptr_eq(&l.borrow(py).seen, &seen.inner)
        {
            return Err(invalid(
                "lookups answer only the view they were made for".into(),
            ));
        }
        Ok(Objective {
            kind: ObjectiveKind::Problem {
                inner: problem::Objective {
                    problem: self.inner.clone(),
                    seen: seen.inner.clone(),
                    layout: Layout {
                        starts,
                        index,
                        offsets,
                        via_index,
                        count,
                    },
                    masks,
                    contact_of,
                },
                lookups,
            },
        })
    }
}

/// A burst's energy as a function of its variables
/// (`kibends_core::problem::Objective`), or a test's
/// (`kibends._core.testing.quadratic`): called with the variables, flat,
/// it gives the energy and its gradient.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Objective {
    kind: ObjectiveKind,
}

enum ObjectiveKind {
    Problem {
        inner: problem::Objective,
        lookups: Option<Py<Lookups>>,
    },
    /// `kibends_core::testing::quadratic` with these curvatures.
    Quadratic { curvatures: Vec<f64>, uphill: bool },
}

impl Objective {
    pub(crate) fn quadratic(curvatures: Vec<f64>, uphill: bool) -> Self {
        Objective {
            kind: ObjectiveKind::Quadratic { curvatures, uphill },
        }
    }

    /// The energy at `x` and its gradient; ValueError for an `x` of the
    /// wrong length or an energy that is not finite.
    pub(crate) fn evaluate(&self, py: Python<'_>, x: &[f64]) -> PyResult<(f64, Vec<f64>)> {
        match &self.kind {
            ObjectiveKind::Problem { inner, lookups } => {
                check_count("coordinates", x.len(), 2 * inner.layout.count)?;
                let mut lookups = lookups.as_ref().map(|l| l.borrow_mut(py));
                inner
                    .evaluate(x, lookups.as_deref_mut().map(|l| &mut l.inner))
                    .map_err(|v| not_finite(py, v))
            }
            ObjectiveKind::Quadratic { curvatures, uphill } => {
                check_count("coordinates", x.len(), curvatures.len())?;
                Ok(kibends_core::testing::quadratic(curvatures, *uphill, x))
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Objective {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(f64, Floats<'py>)> {
        let (value, gradient) = self.evaluate(py, &x.as_array().to_vec())?;
        Ok((value, floats(py, gradient)))
    }
}

/// What the optimizer's energy cost since the last `reset_statistics`
/// (`kibends_core::stats`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Statistics {
    /// Energy evaluations of the bursts' objectives.
    #[pyo3(get)]
    evaluations: u64,
    /// Searches of a layer's neighbour list for its pairs.
    #[pyo3(get)]
    list_searches: u64,
}

/// The energy's statistics so far.
#[gen_stub_pyfunction]
#[pyfunction]
fn statistics() -> Statistics {
    let counted = stats::statistics();
    Statistics {
        evaluations: counted.evaluations,
        list_searches: counted.list_searches,
    }
}

/// Every statistic back to zero.
#[gen_stub_pyfunction]
#[pyfunction]
fn reset_statistics() {
    stats::reset();
}

/// The points the energy asked `field` about since the last
/// `reset_statistics`.
#[gen_stub_pyfunction]
#[pyfunction]
fn field_queries(field: FieldArg) -> u64 {
    stats::queries(&field.0)
}

/// Tracks a and b of a pair at ±(gap + width)/2 from `centreline`, to its
/// right and to its left (`kibends_core::pair_blocks::offset_pair`).
#[gen_stub_pyfunction]
#[pyfunction]
fn offset_pair<'py>(
    py: Python<'py>,
    centreline: Polyline<'_>,
    gap: f64,
    width: f64,
) -> PyResult<(Points<'py>, Points<'py>)> {
    let centreline = points_of(&centreline)?;
    if centreline.len() < 2 {
        return Err(invalid("a centreline needs two vertices".into()));
    }
    let (right, left) = pair_blocks::offset_pair(&centreline, gap, width);
    Ok((vectors(py, &right)?, vectors(py, &left)?))
}

/// The gradient of a function of `offset_pair`'s two tracks, `right` and
/// `left` at each of their vertices, taken back to the centreline's
/// vertices (`kibends_core::pair_blocks::offset_pair_pullback`).
#[gen_stub_pyfunction]
#[pyfunction]
fn offset_pair_pullback<'py>(
    py: Python<'py>,
    centreline: Polyline<'_>,
    gap: f64,
    width: f64,
    right: Polyline<'_>,
    left: Polyline<'_>,
) -> PyResult<Points<'py>> {
    let centreline = points_of(&centreline)?;
    let (right, left) = (points_of(&right)?, points_of(&left)?);
    check_count("right gradients", right.len(), centreline.len())?;
    check_count("left gradients", left.len(), centreline.len())?;
    vectors(
        py,
        &pair_blocks::offset_pair_pullback(&centreline, gap, width, &right, &left),
    )
}

/// Registers the problem's classes and functions in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PairOffset>()?;
    m.add_class::<Member>()?;
    m.add_class::<Members>()?;
    m.add_class::<EdgesClear>()?;
    m.add_class::<SmoothAnswer>()?;
    m.add_class::<Seen>()?;
    m.add_class::<NeighbourList>()?;
    m.add_class::<ObstacleDistances>()?;
    m.add_class::<ViaSmooth>()?;
    m.add_class::<ViaSight>()?;
    m.add_class::<StaticFree>()?;
    m.add_class::<FreeDistances>()?;
    m.add_class::<Lookups>()?;
    m.add_class::<Problem>()?;
    m.add_class::<Objective>()?;
    m.add_class::<Statistics>()?;
    m.add_function(wrap_pyfunction!(free_distances, m)?)?;
    m.add_function(wrap_pyfunction!(newly_inside, m)?)?;
    m.add_function(wrap_pyfunction!(statistics, m)?)?;
    m.add_function(wrap_pyfunction!(reset_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(field_queries, m)?)?;
    m.add_function(wrap_pyfunction!(offset_pair, m)?)?;
    m.add_function(wrap_pyfunction!(offset_pair_pullback, m)?)?;
    m.add("SKIN_LEAST", neighbours::SKIN_LEAST)?;
    m.add("SKIN_MOST", neighbours::SKIN_MOST)?;
    m.add("STATIC_SKIN", neighbours::STATIC_SKIN)?;
    Ok(())
}

pyo3_stub_gen::module_variable!("kibends._core", "SKIN_LEAST", f64, neighbours::SKIN_LEAST);
pyo3_stub_gen::module_variable!("kibends._core", "SKIN_MOST", f64, neighbours::SKIN_MOST);
pyo3_stub_gen::module_variable!("kibends._core", "STATIC_SKIN", f64, neighbours::STATIC_SKIN);
