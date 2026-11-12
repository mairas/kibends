//! The energy terms as Python sees them (`kibends_core::energy`): the
//! integrands and curvatures as callables over numpy arrays, and the terms
//! over polylines, for the Python side of the optimizer until it moves here
//! too.

use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3_stub_gen::{PyStubType, TypeInfo};

use crate::arrays::{BoolArg, IntpArg};
use crate::field::{
    Floats, Points, flags_of, floats, floats_of, indices_of, points_of, same_lengths, vectors,
};
use kibends_core::energy::integrands::{
    Attraction, Curvature as CoreCurvature, Integrand as CoreIntegrand,
};
use kibends_core::energy::{
    DEFAULT_ATTRACTION_RANGE, EnergyParams, MAX_SMOOTH_TURN, MIN_EDGE_LENGTH, polyline, quadrature,
};
use kibends_core::field::smooth::{self, SMOOTH_REACH, SMOOTH_WIDTH};
use kibends_core::geometry::V2;

type FloatsDyn<'py> = Bound<'py, PyArrayDyn<f64>>;

/// `kibends.energy.EnergyParams`, read by its attributes.
pub struct Params(pub EnergyParams);

impl<'a, 'py> FromPyObject<'a, 'py> for Params {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let float = |name: &str| -> PyResult<f64> { obj.getattr(name)?.extract() };
        Ok(Self(EnergyParams {
            repulsion: float("repulsion")?,
            falloff_range: float("falloff_range")?,
            clearance_weight: float("clearance_weight")?,
            standoff: float("standoff")?,
            points_per_edge: obj.getattr("points_per_edge")?.extract()?,
            attraction: float("attraction")?,
            attraction_range: float("attraction_range")?,
        }))
    }
}

impl PyStubType for Params {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: "energy.EnergyParams".into(),
            source_module: None,
            import: HashSet::from(["kibends.energy".into()]),
            type_refs: HashMap::new(),
        }
    }
}

/// A function of the distance to an obstacle: called with distances, it
/// gives its value and slope at each.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Integrand {
    pub inner: CoreIntegrand,
}

#[gen_stub_pymethods]
#[pymethods]
impl Integrand {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        distance: PyReadonlyArrayDyn<'_, f64>,
    ) -> (FloatsDyn<'py>, FloatsDyn<'py>) {
        let distance = distance.as_array();
        let value = distance.map(|&d| self.inner.at(d).0);
        let slope = distance.map(|&d| self.inner.at(d).1);
        (value.into_pyarray(py), slope.into_pyarray(py))
    }
}

/// An integrand's second derivative by the distance: called with
/// distances, it gives the curvature at each.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Curvature {
    pub inner: CoreCurvature,
}

#[gen_stub_pymethods]
#[pymethods]
impl Curvature {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        distance: PyReadonlyArrayDyn<'_, f64>,
    ) -> FloatsDyn<'py> {
        distance
            .as_array()
            .map(|&d| self.inner.at(d))
            .into_pyarray(py)
    }
}

/// A function of two vias' centre distance and their limit distance:
/// called with both, it gives its value and slope at each.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct PairIntegrand {
    inner: Attraction,
}

#[gen_stub_pymethods]
#[pymethods]
impl PairIntegrand {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        distance: PyReadonlyArray1<'_, f64>,
        limit: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(Floats<'py>, Floats<'py>)> {
        let distance = floats_of(&distance);
        let limit = floats_of(&limit);
        same_lengths(&[distance.len(), limit.len()])?;
        let (value, slope) = distance
            .iter()
            .zip(&limit)
            .map(|(&d, &l)| self.inner.at(d, l))
            .unzip();
        Ok((floats(py, value), floats(py, slope)))
    }
}

/// Repulsion plus clearance, weighted, and its slope, at given distances.
#[gen_stub_pyfunction]
#[pyfunction]
fn obstacle_integrand(params: Params) -> Integrand {
    Integrand {
        inner: CoreIntegrand::obstacle(&params.0),
    }
}

/// The clearance hinge of `obstacle_integrand` alone, weighted: for what
/// only limits where a thing may go and does not repel it.
#[gen_stub_pyfunction]
#[pyfunction]
fn hinge_integrand(params: Params) -> Integrand {
    Integrand {
        inner: CoreIntegrand::hinge(&params.0),
    }
}

/// The repulsion of `obstacle_integrand` alone, weighted: for what pushes
/// away without being a wall.
#[gen_stub_pyfunction]
#[pyfunction]
fn repulsion_integrand(params: Params) -> Integrand {
    Integrand {
        inner: CoreIntegrand::repulsion(&params.0),
    }
}

/// The pull of two vias on each other, and its slope, at centre distances
/// r with limit distances r_min (`kibends_core::energy::integrands::Attraction`).
#[gen_stub_pyfunction]
#[pyfunction]
fn attraction_integrand(params: Params) -> PairIntegrand {
    PairIntegrand {
        inner: Attraction::new(&params.0),
    }
}

/// The second derivative of `obstacle_integrand` by the distance, and
/// inside an obstacle that at its boundary.
#[gen_stub_pyfunction]
#[pyfunction]
fn obstacle_curvature(params: Params) -> Curvature {
    Curvature {
        inner: CoreCurvature::obstacle(&params.0),
    }
}

/// The second derivative of `hinge_integrand` by the distance, inside an
/// obstacle taken at its boundary.
#[gen_stub_pyfunction]
#[pyfunction]
fn hinge_curvature(params: Params) -> Curvature {
    Curvature {
        inner: CoreCurvature::hinge(&params.0),
    }
}

/// Distance from an obstacle beyond which both obstacle terms of the
/// energy are zero, mm, so farther copper need not be looked up.
#[gen_stub_pyfunction]
#[pyfunction]
fn obstacle_reach(params: Params) -> f64 {
    params.0.obstacle_reach()
}

/// How far from a point the smooth distance looks for obstacles, for an
/// energy whose terms end at `reach`, mm (SMOOTH_REACH).
#[gen_stub_pyfunction]
#[pyfunction]
fn smooth_reach(reach: f64) -> f64 {
    smooth::smooth_reach(reach)
}

fn polyline_of(
    points: &PyReadonlyArray2<'_, f64>,
    fixed: &BoolArg<'_>,
) -> PyResult<(Vec<V2>, Vec<bool>)> {
    let points = points_of(points)?;
    let fixed = flags_of(fixed);
    same_lengths(&[points.len(), fixed.len()])?;
    Ok((points, fixed))
}

/// The terms of a member's energy that depend on the polyline alone: its
/// length and, with a positive `short_edge`, mm, the short-edge barrier
/// (`kibends_core::energy::polyline::shape_energy`); and the gradient,
/// zero at `fixed` vertices.
#[gen_stub_pyfunction]
#[pyfunction]
fn shape_energy<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    fixed: BoolArg<'_>,
    short_edge: f64,
) -> PyResult<(f64, Points<'py>)> {
    let (points, fixed) = polyline_of(&points, &fixed)?;
    let (value, gradient) = polyline::shape_energy(&points, &fixed, short_edge);
    Ok((value, vectors(py, &gradient)?))
}

/// The bend-radius hinge at `limit` (`kibends_core::energy::polyline::
/// bend_radius_energy`), and its gradient, zero at `fixed` vertices.
#[gen_stub_pyfunction]
#[pyfunction]
fn bend_radius_energy<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    fixed: BoolArg<'_>,
    limit: f64,
    params: Params,
) -> PyResult<(f64, Points<'py>)> {
    let (points, fixed) = polyline_of(&points, &fixed)?;
    let (value, gradient) = polyline::bend_radius_energy(&points, &fixed, limit, &params.0);
    Ok((value, vectors(py, &gradient)?))
}

/// Each edge's bend radius, from the edges' `lengths` and tan(φ/2) of the
/// turn φ at each interior vertex, and the signed sum of those tangents at
/// each edge's two ends: its length over the absolute sum, infinite where
/// the sum is 0.
#[gen_stub_pyfunction]
#[pyfunction]
fn edge_bend_radii<'py>(
    py: Python<'py>,
    lengths: PyReadonlyArray1<'_, f64>,
    half_turns: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Floats<'py>, Floats<'py>)> {
    let lengths = floats_of(&lengths);
    let half_turns = floats_of(&half_turns);
    same_lengths(&[lengths.len(), half_turns.len() + 1])?;
    let (radii, spread) = polyline::edge_bend_radii(&lengths, &half_turns);
    Ok((floats(py, radii), floats(py, spread)))
}

/// Cumulative arc length at each vertex, starting at 0.
#[gen_stub_pyfunction]
#[pyfunction]
fn arc_lengths<'py>(py: Python<'py>, points: PyReadonlyArray2<'_, f64>) -> PyResult<Floats<'py>> {
    Ok(floats(py, polyline::arc_lengths(&points_of(&points)?)))
}

/// The points at which `edge_integral` takes distances: the vertices, then
/// each edge's interior points in order.
#[gen_stub_pyfunction]
#[pyfunction]
fn quadrature_points<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    points_per_edge: usize,
) -> PyResult<Points<'py>> {
    let points = points_of(&points)?;
    vectors(py, &quadrature::quadrature_points(&points, points_per_edge))
}

/// The distances and their gradients at the `quadrature_points` of a
/// polyline, checked against it.
fn samples_of(
    points: &PyReadonlyArray2<'_, f64>,
    distance: &PyReadonlyArray1<'_, f64>,
    direction: &PyReadonlyArray2<'_, f64>,
    points_per_edge: usize,
) -> PyResult<(Vec<V2>, Vec<f64>, Vec<V2>)> {
    let points = points_of(points)?;
    let distance = floats_of(distance);
    let direction = points_of(direction)?;
    let expected = points.len() + points.len().saturating_sub(1) * points_per_edge;
    same_lengths(&[distance.len(), direction.len(), expected])?;
    Ok((points, distance, direction))
}

/// ∫ f(d(s)) ds along the polyline by the trapezoid rule on each edge,
/// from the distances and their unit gradients at `quadrature_points`
/// (`kibends_core::energy::quadrature::edge_integral`): the value, its
/// gradient at every vertex, and its derivative by the distance at each
/// quadrature point.
#[gen_stub_pyfunction]
#[pyfunction]
fn edge_integral<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    distance: PyReadonlyArray1<'_, f64>,
    direction: PyReadonlyArray2<'_, f64>,
    integrand: PyRef<'_, Integrand>,
    points_per_edge: usize,
) -> PyResult<(f64, Points<'py>, Floats<'py>)> {
    let (points, distance, direction) =
        samples_of(&points, &distance, &direction, points_per_edge)?;
    let (value, gradient, slopes) = quadrature::edge_integral(
        &points,
        &distance,
        &direction,
        &integrand.inner,
        points_per_edge,
    );
    Ok((value, vectors(py, &gradient)?, floats(py, slopes)))
}

/// Each vertex's stiffness under `edge_integral` of `integrand`
/// (`kibends_core::energy::quadrature::edge_curvature`).
#[gen_stub_pyfunction]
#[pyfunction]
fn edge_curvature<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    distance: PyReadonlyArray1<'_, f64>,
    direction: PyReadonlyArray2<'_, f64>,
    integrand: PyRef<'_, Integrand>,
    curvature: PyRef<'_, Curvature>,
    points_per_edge: usize,
) -> PyResult<Floats<'py>> {
    let (points, distance, direction) =
        samples_of(&points, &distance, &direction, points_per_edge)?;
    Ok(floats(
        py,
        quadrature::edge_curvature(
            &points,
            &distance,
            &direction,
            &integrand.inner,
            &curvature.inner,
            points_per_edge,
        ),
    ))
}

/// The diagonal second derivative of `integrand` at points by x and by y,
/// (n, 2), from their smooth distances (n,) and gradients (n, 2)
/// (`kibends_core::energy::quadrature::point_stiffness`).
#[gen_stub_pyfunction]
#[pyfunction]
fn point_stiffness<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray1<'_, f64>,
    direction: PyReadonlyArray2<'_, f64>,
    integrand: PyRef<'_, Integrand>,
    curvature: PyRef<'_, Curvature>,
) -> PyResult<Points<'py>> {
    let distance = floats_of(&distance);
    let direction = points_of(&direction)?;
    same_lengths(&[distance.len(), direction.len()])?;
    let stiffness: Vec<V2> = distance
        .iter()
        .zip(&direction)
        .map(|(&d, &g)| quadrature::point_stiffness(d, g, &integrand.inner, &curvature.inner))
        .collect();
    vectors(py, &stiffness)
}

/// -τ·log Σ exp(-d/τ), τ = SMOOTH_WIDTH, at each of `count` points over the
/// obstacles' distances d, obstacle k's distance[k] at point[k], +inf at a
/// point with none; and each obstacle's share of the gradient there.
#[gen_stub_pyfunction]
#[pyfunction]
fn smooth_minimum<'py>(
    py: Python<'py>,
    point: IntpArg<'_>,
    distance: PyReadonlyArray1<'_, f64>,
    count: usize,
) -> PyResult<(Floats<'py>, Floats<'py>)> {
    let point = indices_of(&point, count)?;
    let distance = floats_of(&distance);
    same_lengths(&[point.len(), distance.len()])?;
    let (smooth, share) = smooth::smooth_minimum(&point, &distance, count);
    Ok((floats(py, smooth), floats(py, share)))
}

/// At each of `count` points, the sum of the (m, 2) `direction`s at it,
/// each weighted by its `share`; the smooth minimum's gradient.
#[gen_stub_pyfunction]
#[pyfunction]
fn weighted_sum<'py>(
    py: Python<'py>,
    point: IntpArg<'_>,
    share: PyReadonlyArray1<'_, f64>,
    direction: PyReadonlyArray2<'_, f64>,
    count: usize,
) -> PyResult<Points<'py>> {
    let point = indices_of(&point, count)?;
    let share = floats_of(&share);
    let direction = points_of(&direction)?;
    same_lengths(&[point.len(), share.len(), direction.len()])?;
    vectors(py, &smooth::weighted_sum(&point, &share, &direction, count))
}

/// The smooth minimum over fields, each of `distances` one field's (n,)
/// smooth distances, and each field's share at each point.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (*distances))]
fn combine_fields<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy", "numpy.typing")
    ))]
    distances: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<(Floats<'py>, Vec<Floats<'py>>)> {
    let distances: Vec<Vec<f64>> = distances.iter().map(floats_of).collect();
    if distances.is_empty() {
        return Err(PyValueError::new_err("combine_fields needs a field"));
    }
    same_lengths(&distances.iter().map(Vec::len).collect::<Vec<_>>())?;
    let slices: Vec<&[f64]> = distances.iter().map(Vec::as_slice).collect();
    let (smooth, shares) = smooth::combine_fields(&slices);
    Ok((
        floats(py, smooth),
        shares.into_iter().map(|s| floats(py, s)).collect(),
    ))
}

/// Registers the energy's classes, functions and constants in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Integrand>()?;
    m.add_class::<Curvature>()?;
    m.add_class::<PairIntegrand>()?;
    m.add_function(wrap_pyfunction!(obstacle_integrand, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_integrand, m)?)?;
    m.add_function(wrap_pyfunction!(repulsion_integrand, m)?)?;
    m.add_function(wrap_pyfunction!(attraction_integrand, m)?)?;
    m.add_function(wrap_pyfunction!(obstacle_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(obstacle_reach, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_reach, m)?)?;
    m.add_function(wrap_pyfunction!(shape_energy, m)?)?;
    m.add_function(wrap_pyfunction!(bend_radius_energy, m)?)?;
    m.add_function(wrap_pyfunction!(edge_bend_radii, m)?)?;
    m.add_function(wrap_pyfunction!(arc_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(quadrature_points, m)?)?;
    m.add_function(wrap_pyfunction!(edge_integral, m)?)?;
    m.add_function(wrap_pyfunction!(edge_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(point_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_minimum, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_sum, m)?)?;
    m.add_function(wrap_pyfunction!(combine_fields, m)?)?;
    m.add("MIN_EDGE_LENGTH", MIN_EDGE_LENGTH)?;
    m.add("MAX_SMOOTH_TURN", MAX_SMOOTH_TURN)?;
    m.add("SMOOTH_WIDTH", SMOOTH_WIDTH)?;
    m.add("SMOOTH_REACH", SMOOTH_REACH)?;
    m.add("DEFAULT_ATTRACTION_RANGE", DEFAULT_ATTRACTION_RANGE)?;
    Ok(())
}

pyo3_stub_gen::module_variable!("kibends._core", "MIN_EDGE_LENGTH", f64, MIN_EDGE_LENGTH);
pyo3_stub_gen::module_variable!("kibends._core", "MAX_SMOOTH_TURN", f64, MAX_SMOOTH_TURN);
pyo3_stub_gen::module_variable!("kibends._core", "SMOOTH_WIDTH", f64, SMOOTH_WIDTH);
pyo3_stub_gen::module_variable!("kibends._core", "SMOOTH_REACH", f64, SMOOTH_REACH);
pyo3_stub_gen::module_variable!(
    "kibends._core",
    "DEFAULT_ATTRACTION_RANGE",
    f64,
    DEFAULT_ATTRACTION_RANGE
);
