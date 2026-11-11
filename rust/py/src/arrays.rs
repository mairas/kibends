//! numpy arrays of indices and flags with their stub types: pyo3-stub-gen
//! types no array of `isize` or `bool`, which Python's index arrays
//! (`numpy.intp`) and masks are.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_stub_gen::{PyStubType, TypeInfo};

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
