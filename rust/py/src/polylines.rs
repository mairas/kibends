//! The measures and checks on a single polyline as Python sees them
//! (`kibends_core::polylines`).

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{BoolArg, BoolArray, Flags, Points, check_count, flags_of, points_of, vectors};

use kibends_core::polylines;

/// Each span between fixed vertices resampled to uniform arc-length
/// `spacing`, the fixed vertices kept exactly, and the new fixed mask.
#[gen_stub_pyfunction]
#[pyfunction]
fn resample<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    fixed: BoolArg<'_>,
    spacing: f64,
) -> PyResult<(Points<'py>, Flags<'py>)> {
    let points = points_of(&points)?;
    let fixed = flags_of(&fixed);
    check_count("fixed flags", fixed.len(), points.len())?;
    if !fixed.first().copied().unwrap_or(false) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a resampled polyline starts at a fixed vertex",
        ));
    }
    let (points, fixed) = polylines::resample(&points, &fixed, spacing);
    Ok((vectors(py, &points)?, BoolArray(fixed.into_pyarray(py))))
}

/// Places where a polyline enters the clearance away from its `anchors`,
/// given the vertices `inside` the clearance and the edges that `cut`
/// into it (`kibends_core::polylines::count_violations`).
#[gen_stub_pyfunction]
#[pyfunction]
fn count_violations(
    inside: BoolArg<'_>,
    cut: BoolArg<'_>,
    anchors: BoolArg<'_>,
) -> PyResult<usize> {
    let (inside, cut, anchors) = (flags_of(&inside), flags_of(&cut), flags_of(&anchors));
    check_count("anchors", anchors.len(), inside.len())?;
    check_count("edges", cut.len(), inside.len().saturating_sub(1))?;
    Ok(polylines::count_violations(&inside, &cut, &anchors))
}

/// Local symmetric distance between two polylines, mm, not counting
/// vertices sliding along an unchanged shape.
#[gen_stub_pyfunction]
#[pyfunction]
fn shape_change(
    before: PyReadonlyArray2<'_, f64>,
    after: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let (before, after) = (points_of(&before)?, points_of(&after)?);
    if before.len() < 2 || after.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a shape change needs polylines of two vertices",
        ));
    }
    Ok(polylines::shape_change(&before, &after))
}

/// A polyline's length, mm, and its tightest bend radius at `spacing`,
/// infinite for a straight one (`kibends_core::polylines::measure`).
#[gen_stub_pyfunction]
#[pyfunction]
fn measure(polyline: PyReadonlyArray2<'_, f64>, spacing: f64) -> PyResult<(f64, f64)> {
    let polyline = points_of(&polyline)?;
    if polyline.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a polyline of no vertex has no measure",
        ));
    }
    Ok(polylines::measure(&polyline, spacing))
}

/// What a burst may not do to a coupled section's centreline: bend an edge
/// below its limit that was not already there where the burst started
/// (`kibends_core::polylines::SelfCheck`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct SelfCheck {
    inner: polylines::SelfCheck,
}

#[gen_stub_pymethods]
#[pymethods]
impl SelfCheck {
    /// The check of a burst that starts at `original`.
    #[staticmethod]
    fn of(original: PyReadonlyArray2<'_, f64>, bend_limit: f64) -> PyResult<Self> {
        Ok(Self {
            inner: polylines::SelfCheck::of(&points_of(&original)?, bend_limit),
        })
    }

    /// The vertices of `candidate` to move back, and how many edges newly
    /// bent too tightly.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        candidate: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(Flags<'py>, usize)> {
        let candidate = points_of(&candidate)?;
        check_count(
            "edges",
            candidate.len().saturating_sub(1),
            self.inner.tight_before.len(),
        )?;
        let (back, count) = self.inner.check(&candidate);
        Ok((BoolArray(back.into_pyarray(py)), count))
    }
}

/// Registers the polyline functions and class in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SelfCheck>()?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(count_violations, m)?)?;
    m.add_function(wrap_pyfunction!(shape_change, m)?)?;
    m.add_function(wrap_pyfunction!(measure, m)?)?;
    Ok(())
}
