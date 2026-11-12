//! Arrays crossing to and from Python, independent of what they hold: the
//! conversions between numpy arrays and the core's values, and the checks
//! of their lengths and indices. numpy arrays of indices and flags carry
//! their stub types here: pyo3-stub-gen types no array of `isize` or
//! `bool`, which Python's index arrays (`numpy.intp`) and masks are.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use pyo3_stub_gen::{PyStubType, TypeInfo};

use kibends_core::geometry::V2;

fn ndarray_of(scalar: &str) -> TypeInfo {
    TypeInfo {
        name: format!("numpy.typing.NDArray[numpy.{scalar}]"),
        source_module: None,
        import: HashSet::from(["numpy".into(), "numpy.typing".into()]),
        type_refs: HashMap::new(),
    }
}

/// A numpy array of indices, `numpy.intp`, returned to Python.
pub struct IntpArray<'py>(pub Bound<'py, PyArray1<isize>>);

/// A numpy array of flags, `numpy.bool_`, returned to Python.
pub struct BoolArray<'py>(pub Bound<'py, PyArray1<bool>>);

/// A numpy array of indices, `numpy.intp`, taken from Python.
pub struct IntpArg<'py>(pub PyReadonlyArray1<'py, isize>);

/// A numpy array of flags, `numpy.bool_`, taken from Python.
pub struct BoolArg<'py>(pub PyReadonlyArray1<'py, bool>);

impl<'py> IntoPyObject<'py> for IntpArray<'py> {
    type Target = PyArray1<isize>;
    type Output = Bound<'py, PyArray1<isize>>;
    type Error = Infallible;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

impl<'py> IntoPyObject<'py> for BoolArray<'py> {
    type Target = PyArray1<bool>;
    type Output = Bound<'py, PyArray1<bool>>;
    type Error = Infallible;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for IntpArg<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract()?))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for BoolArg<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract()?))
    }
}

impl PyStubType for IntpArray<'_> {
    fn type_output() -> TypeInfo {
        ndarray_of("intp")
    }
}

impl PyStubType for BoolArray<'_> {
    fn type_output() -> TypeInfo {
        ndarray_of("bool_")
    }
}

impl PyStubType for IntpArg<'_> {
    fn type_output() -> TypeInfo {
        ndarray_of("intp")
    }
}

impl PyStubType for BoolArg<'_> {
    fn type_output() -> TypeInfo {
        ndarray_of("bool_")
    }
}

pub(crate) type Floats<'py> = Bound<'py, PyArray1<f64>>;

pub(crate) type Points<'py> = Bound<'py, PyArray2<f64>>;

pub(crate) type Indices<'py> = IntpArray<'py>;

pub(crate) type Flags<'py> = BoolArray<'py>;

pub(crate) fn points_of(array: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<V2>> {
    if array.shape()[1] != 2 {
        return Err(PyValueError::new_err("points must be an (n, 2) array"));
    }
    Ok(array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| V2::new(r[0], r[1]))
        .collect())
}

pub(crate) fn floats_of(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    array.as_array().to_vec()
}

pub(crate) fn flags_of(array: &BoolArg<'_>) -> Vec<bool> {
    array.0.as_array().to_vec()
}

pub(crate) fn indices_of(array: &IntpArg<'_>, below: usize) -> PyResult<Vec<usize>> {
    array
        .0
        .as_array()
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .filter(|&i| i < below)
                .ok_or_else(|| PyValueError::new_err(format!("index {i} out of range 0..{below}")))
        })
        .collect()
}

pub(crate) fn same_lengths(lengths: &[usize]) -> PyResult<()> {
    if lengths.windows(2).any(|w| w[0] != w[1]) {
        return Err(PyValueError::new_err(format!(
            "arrays of unequal lengths {lengths:?}"
        )));
    }
    Ok(())
}

pub(crate) fn floats<'py>(py: Python<'py>, values: Vec<f64>) -> Floats<'py> {
    values.into_pyarray(py)
}

pub(crate) fn indices<'py>(py: Python<'py>, values: &[usize]) -> Indices<'py> {
    // An index into a Rust slice fits an isize.
    IntpArray(
        values
            .iter()
            .map(|&i| i as isize)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )
}

pub(crate) fn vectors<'py>(py: Python<'py>, values: &[V2]) -> PyResult<Points<'py>> {
    let flat: Vec<f64> = values.iter().flat_map(|v| [v.x, v.y]).collect();
    let array = Array2::from_shape_vec((values.len(), 2), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

pub(crate) type Polyline<'py> = PyReadonlyArray2<'py, f64>;

pub(crate) fn polylines_of(arrays: &[Polyline<'_>]) -> PyResult<Vec<Vec<V2>>> {
    arrays.iter().map(points_of).collect()
}

pub(crate) fn refs(polylines: &[Vec<V2>]) -> Vec<&[V2]> {
    polylines.iter().map(Vec::as_slice).collect()
}

pub(crate) fn vectors_list<'py>(py: Python<'py>, values: &[Vec<V2>]) -> PyResult<Vec<Points<'py>>> {
    values.iter().map(|v| vectors(py, v)).collect()
}

pub(crate) fn invalid(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// The value of an energy that is not finite, as Python says it.
pub(crate) fn not_finite(py: Python<'_>, value: f64) -> PyErr {
    let said = PyFloat::new(py, value)
        .str()
        .map_or_else(|_| value.to_string(), |s| s.to_string());
    invalid(format!("energy is not finite: {said}"))
}

/// Per vertex, its index, -1 for none.
pub(crate) fn optional_indices(array: &IntpArg<'_>, below: usize) -> PyResult<Vec<Option<usize>>> {
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
pub(crate) fn check_rows<T>(what: &str, rows: &[Vec<T>], polylines: &[Vec<V2>]) -> PyResult<()> {
    check_count(what, rows.len(), polylines.len())?;
    for (row, points) in rows.iter().zip(polylines) {
        check_count(what, row.len(), points.len())?;
    }
    Ok(())
}

pub(crate) fn masks_of(masks: &[BoolArg<'_>], polylines: &[Vec<V2>]) -> PyResult<Vec<Vec<bool>>> {
    let masks: Vec<Vec<bool>> = masks.iter().map(|m| m.0.as_array().to_vec()).collect();
    check_rows("mask entries", &masks, polylines)?;
    Ok(masks)
}

pub(crate) fn flags<'py>(py: Python<'py>, values: &[bool]) -> Flags<'py> {
    BoolArray(values.to_vec().into_pyarray(py))
}
