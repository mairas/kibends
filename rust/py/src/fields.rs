//! The fields the optimizer composes, as Python sees them: other chains as
//! obstacles (`ChainField`), the union of several fields
//! (`CompositeField`), and the queries every field answers.

use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3_stub_gen::{PyStubType, TypeInfo};

use crate::arrays::{BoolArray, IntpArg};
use crate::field::{
    Flags, Floats, Indices, Obstacles, Points, StaticEdges, floats, floats_of, indices, indices_of,
    points_of, same_lengths, vectors,
};
use crate::testing;
use kibends_core::field::Field;
use kibends_core::field::chain;
use kibends_core::field::composite;

/// The distance at (n, 2) points and its unit gradient, from `field`.
pub(crate) fn signed_distance<'py>(
    py: Python<'py>,
    field: &dyn Field,
    points: &PyReadonlyArray2<'_, f64>,
    reach: f64,
) -> PyResult<(Floats<'py>, Points<'py>)> {
    let points = points_of(points)?;
    let (distance, gradient) = field.signed_distance(&points, reach);
    Ok((floats(py, distance), vectors(py, &gradient)?))
}

/// The smooth distance at (n, 2) points and its gradient, from `field`.
pub(crate) fn smooth_distance<'py>(
    py: Python<'py>,
    field: &dyn Field,
    points: &PyReadonlyArray2<'_, f64>,
    reach: f64,
) -> PyResult<(Floats<'py>, Points<'py>)> {
    let points = points_of(points)?;
    let (distance, gradient) = field.smooth_distance(&points, reach);
    Ok((floats(py, distance), vectors(py, &gradient)?))
}

/// Whether each segment from starts[i] to ends[i] is clear in `field`.
pub(crate) fn segments_clear<'py>(
    py: Python<'py>,
    field: &dyn Field,
    starts: &PyReadonlyArray2<'_, f64>,
    ends: &PyReadonlyArray2<'_, f64>,
    tolerance: f64,
) -> PyResult<Flags<'py>> {
    let starts = points_of(starts)?;
    let ends = points_of(ends)?;
    same_lengths(&[starts.len(), ends.len()])?;
    Ok(BoolArray(
        field
            .segments_clear(&starts, &ends, tolerance)
            .into_pyarray(py),
    ))
}

/// A field implemented here, of any kind: `Obstacles`, `ChainField`,
/// `CompositeField` or a field of `kibends._core.testing`.
pub struct FieldArg(pub Arc<dyn Field>);

impl<'a, 'py> FromPyObject<'a, 'py> for FieldArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(f) = obj.cast::<Obstacles>() {
            return Ok(Self(f.get().inner.clone()));
        }
        if let Ok(f) = obj.cast::<ChainField>() {
            return Ok(Self(f.get().inner.clone()));
        }
        if let Ok(f) = obj.cast::<CompositeField>() {
            return Ok(Self(f.get().inner.clone()));
        }
        testing::field_of(&obj).map(Self).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{} is not a field of kibends._core",
                obj.get_type()
                    .name()
                    .map_or_else(|_| "?".into(), |n| n.to_string())
            ))
        })
    }
}

impl PyStubType for FieldArg {
    fn type_output() -> TypeInfo {
        crate::Field::type_output()
    }
}

/// Signed distance to other chains' inflated centrelines, and edge checks
/// (`kibends_core::field::chain`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ChainField {
    pub(crate) inner: Arc<chain::ChainField>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ChainField {
    /// Edges from starts[i] to ends[i], (m, 2) in mm, each inflated by
    /// inflations[i] and part of the obstacle obstacles[i].
    #[new]
    fn new(
        starts: PyReadonlyArray2<'_, f64>,
        ends: PyReadonlyArray2<'_, f64>,
        inflations: PyReadonlyArray1<'_, f64>,
        obstacles: IntpArg<'_>,
    ) -> PyResult<Self> {
        let starts = points_of(&starts)?;
        let ends = points_of(&ends)?;
        let inflations = floats_of(&inflations);
        let obstacles = indices_of(&obstacles, usize::MAX)?;
        same_lengths(&[starts.len(), ends.len(), inflations.len(), obstacles.len()])?;
        Ok(Self {
            inner: Arc::new(chain::ChainField::new(
                &starts, &ends, inflations, obstacles,
            )),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The field as `StaticEdges`, each edge an item of its own.
    #[getter]
    fn static_edges(&self) -> StaticEdges {
        StaticEdges {
            inner: self.inner.static_edges().clone(),
        }
    }

    /// Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn signed_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        signed_distance(py, self.inner.as_ref(), &points, reach)
    }

    /// `signed_distance` as the energy takes it: the smooth minimum over
    /// the obstacles of each one's nearest edge.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn smooth_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        smooth_distance(py, self.inner.as_ref(), &points, reach)
    }

    /// `smooth_distance`, and what answers it: per obstacle that has a
    /// share at a point, the point, its nearest edge by index, its share
    /// and its unit gradient.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    #[allow(clippy::type_complexity)]
    fn smooth_answers<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(
        Floats<'py>,
        Points<'py>,
        Indices<'py>,
        Indices<'py>,
        Floats<'py>,
        Points<'py>,
    )> {
        let found = self.inner.smooth_answers(&points_of(&points)?, reach);
        Ok((
            floats(py, found.distance),
            vectors(py, &found.gradient)?,
            indices(py, &found.point),
            indices(py, &found.edge),
            floats(py, found.share),
            vectors(py, &found.direction)?,
        ))
    }

    /// Whether each segment from starts[i] to ends[i] stays out of every
    /// inflated edge, by at most `tolerance`.
    #[pyo3(signature = (starts, ends, tolerance = 0.0))]
    fn segments_clear<'py>(
        &self,
        py: Python<'py>,
        starts: PyReadonlyArray2<'_, f64>,
        ends: PyReadonlyArray2<'_, f64>,
        tolerance: f64,
    ) -> PyResult<Flags<'py>> {
        segments_clear(py, self.inner.as_ref(), &starts, &ends, tolerance)
    }
}

/// The polylines, (n, 2) each, as a field: each split into edges of at
/// most `MAX_EDGE_LENGTH`, inflated by inflations[i] and part of the
/// obstacle obstacles[i].
#[gen_stub_pyfunction]
#[pyfunction]
fn chains_field(
    polylines: Vec<PyReadonlyArray2<'_, f64>>,
    inflations: PyReadonlyArray1<'_, f64>,
    obstacles: IntpArg<'_>,
) -> PyResult<ChainField> {
    let polylines = polylines
        .iter()
        .map(points_of)
        .collect::<PyResult<Vec<_>>>()?;
    let inflations = floats_of(&inflations);
    let obstacles = indices_of(&obstacles, usize::MAX)?;
    same_lengths(&[polylines.len(), inflations.len(), obstacles.len()])?;
    let field = chain::ChainField::from_polylines(
        polylines
            .iter()
            .zip(inflations)
            .zip(obstacles)
            .map(|((p, i), o)| (p.as_slice(), i, o)),
    );
    Ok(ChainField {
        inner: Arc::new(field),
    })
}

/// The union of several fields: the nearest of them answers each point,
/// and the smooth distance is over all their obstacles
/// (`kibends_core::field::composite`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct CompositeField {
    pub(crate) inner: Arc<composite::CompositeField>,
}

#[gen_stub_pymethods]
#[pymethods]
impl CompositeField {
    #[new]
    fn new(fields: Vec<FieldArg>) -> Self {
        Self {
            inner: Arc::new(composite::CompositeField::new(
                fields.into_iter().map(|f| f.0).collect(),
            )),
        }
    }

    /// Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn signed_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        signed_distance(py, self.inner.as_ref(), &points, reach)
    }

    /// The smooth minimum of the fields' smooth distances.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn smooth_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        smooth_distance(py, self.inner.as_ref(), &points, reach)
    }

    /// Whether each segment from starts[i] to ends[i] is clear in every
    /// field, each allowing `tolerance` inside its clearance.
    #[pyo3(signature = (starts, ends, tolerance = 0.0))]
    fn segments_clear<'py>(
        &self,
        py: Python<'py>,
        starts: PyReadonlyArray2<'_, f64>,
        ends: PyReadonlyArray2<'_, f64>,
        tolerance: f64,
    ) -> PyResult<Flags<'py>> {
        segments_clear(py, self.inner.as_ref(), &starts, &ends, tolerance)
    }
}

/// Registers the composed fields in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ChainField>()?;
    m.add_class::<CompositeField>()?;
    m.add_function(wrap_pyfunction!(chains_field, m)?)?;
    Ok(())
}
