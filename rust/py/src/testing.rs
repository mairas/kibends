//! The submodule `kibends._core.testing`: the analytic fields and the
//! quadratic objective of `kibends_core::testing`, for tests.

use std::sync::Arc;

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{Flags, Floats, Points};
use crate::fields::{FieldArg, segments_clear, signed_distance, smooth_distance};
use crate::problem::Objective;
use kibends_core::field::Field;
use kibends_core::testing;

/// A test field's class: its getters as given before a `;` and its other
/// fields after it, its methods as given, and the queries every field
/// answers.
macro_rules! test_field {
    (
        $(#[$doc:meta])*
        $name:ident { $($field:ident: $ty:ty),* $(; $($hidden:ident: $hidden_ty:ty),*)? }
        $($methods:tt)*
    ) => {
        $(#[$doc])*
        #[gen_stub_pyclass]
        #[pyclass(frozen, module = "kibends._core.testing")]
        pub struct $name {
            $(#[pyo3(get)] $field: $ty,)*
            $($($hidden: $hidden_ty,)*)?
            inner: Arc<dyn Field>,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl $name {
            $($methods)*

            /// Distance (n,) and its unit gradient (n, 2) at (n, 2) points
            /// in mm; every distance is exact, whatever the reach.
            #[pyo3(signature = (points, reach = f64::INFINITY))]
            fn signed_distance<'py>(
                &self,
                py: Python<'py>,
                points: PyReadonlyArray2<'_, f64>,
                reach: f64,
            ) -> PyResult<(Floats<'py>, Points<'py>)> {
                signed_distance(py, self.inner.as_ref(), &points, reach)
            }

            /// `signed_distance` as the energy takes it.
            #[pyo3(signature = (points, reach = f64::INFINITY))]
            fn smooth_distance<'py>(
                &self,
                py: Python<'py>,
                points: PyReadonlyArray2<'_, f64>,
                reach: f64,
            ) -> PyResult<(Floats<'py>, Points<'py>)> {
                smooth_distance(py, self.inner.as_ref(), &points, reach)
            }

            /// Whether each segment from starts[i] to ends[i] stays out of
            /// the field's obstacles, by at most `tolerance`.
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
    };
}

test_field! {
    /// Nothing anywhere: every point at `FAR`.
    FreeSpace {}

    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(testing::FreeSpace),
        }
    }
}

test_field! {
    /// A disc of `radius` at (`cx`, `cy`).
    Disc { cx: f64, cy: f64, radius: f64 }

    #[new]
    fn new(cx: f64, cy: f64, radius: f64) -> Self {
        Self {
            cx,
            cy,
            radius,
            inner: Arc::new(testing::Disc { cx, cy, radius }),
        }
    }
}

test_field! {
    /// An axis-aligned box given by its centre and half-sizes. A corner
    /// radius rounds the corners, as inflating a sharp box by that radius
    /// would; the half-sizes include the rounding.
    Box { cx: f64, cy: f64, half_x: f64, half_y: f64, corner_radius: f64 }

    #[new]
    #[pyo3(signature = (cx, cy, half_x, half_y, corner_radius = 0.0))]
    fn new(cx: f64, cy: f64, half_x: f64, half_y: f64, corner_radius: f64) -> Self {
        Self {
            cx,
            cy,
            half_x,
            half_y,
            corner_radius,
            inner: Arc::new(testing::Box {
                cx,
                cy,
                half_x,
                half_y,
                corner_radius,
            }),
        }
    }
}

test_field! {
    /// Two walls parallel to the x axis at y = ±half_gap, each an obstacle;
    /// free space between.
    Walls { half_gap: f64 }

    #[new]
    fn new(half_gap: f64) -> Self {
        Self {
            half_gap,
            inner: Arc::new(testing::Walls {
                half_gap,
                misleading: false,
            }),
        }
    }
}

test_field! {
    /// `Walls` whose distance gradient points the wrong way.
    MisleadingWalls { half_gap: f64 }

    #[new]
    fn new(half_gap: f64) -> Self {
        Self {
            half_gap,
            inner: Arc::new(testing::Walls {
                half_gap,
                misleading: true,
            }),
        }
    }
}

test_field! {
    /// The nearest of several fields at each point, exact in free space;
    /// the smooth distance over theirs.
    Union { ; shapes: Vec<Py<PyAny>> }

    #[new]
    #[gen_stub(override_return_type(type_repr = "Union"))]
    fn new(
        #[gen_stub(override_type(
            type_repr = "typing.Sequence[_core.Field]",
            imports = ("typing", "kibends._core")
        ))]
        shapes: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let fields = shapes
            .iter()
            .map(|s| Ok(s.extract::<FieldArg>()?.0))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            shapes: shapes.into_iter().map(Bound::unbind).collect(),
            inner: Arc::new(testing::Union { shapes: fields }),
        })
    }

    /// The fields joined, in order.
    #[getter]
    #[gen_stub(override_return_type(
        type_repr = "builtins.list[_core.Field]",
        imports = ("builtins", "kibends._core")
    ))]
    fn shapes(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.shapes.iter().map(|s| s.clone_ref(py)).collect()
    }
}

test_field! {
    /// `field` answering every query exactly, whatever the reach, and
    /// giving no static edges.
    ReachIgnored { ; field: Py<PyAny> }

    #[new]
    #[gen_stub(override_return_type(type_repr = "ReachIgnored"))]
    fn new(
        #[gen_stub(override_type(type_repr = "_core.Field", imports = ("kibends._core",)))]
        field: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let inner = field.extract::<FieldArg>()?.0;
        Ok(Self {
            field: field.unbind(),
            inner: Arc::new(testing::ReachIgnored { field: inner }),
        })
    }

    /// The field it answers for.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "_core.Field", imports = ("kibends._core",)))]
    fn field(&self, py: Python<'_>) -> Py<PyAny> {
        self.field.clone_ref(py)
    }
}

test_field! {
    /// A field that answers NaN everywhere.
    NanField {}

    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(testing::NanField),
        }
    }
}

/// The core field of a test field, None for anything else.
pub fn field_of(obj: &Bound<'_, PyAny>) -> Option<Arc<dyn Field>> {
    macro_rules! of {
        ($($class:ident),*) => {
            $(if let Ok(f) = obj.cast::<$class>() {
                return Some(f.get().inner.clone());
            })*
        };
    }
    of!(
        FreeSpace,
        Disc,
        Box,
        Walls,
        MisleadingWalls,
        Union,
        ReachIgnored,
        NanField
    );
    None
}

/// Σ c·x² over the coordinates x, with `curvatures` c, as a burst's
/// objective; `uphill`, its gradient the opposite, so that no line search
/// along it succeeds (`kibends_core::testing::quadratic`).
#[gen_stub_pyfunction(module = "kibends._core.testing")]
#[pyfunction]
#[pyo3(signature = (curvatures, uphill = false))]
fn quadratic(curvatures: Vec<f64>, uphill: bool) -> Objective {
    Objective::quadratic(curvatures, uphill)
}

/// Calls `work` on a thread of a rayon pool of `threads` threads of its
/// own, so that the evaluations it makes spread over them whatever pool the
/// process has (`kibends_core::parallel`).
#[gen_stub_pyfunction(module = "kibends._core.testing")]
#[pyfunction]
fn in_pool(
    py: Python<'_>,
    threads: usize,
    #[gen_stub(override_type(
        type_repr = "collections.abc.Callable[[], None]",
        imports = ("collections.abc")
    ))]
    work: Py<PyAny>,
) -> PyResult<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    py.detach(|| pool.install(|| Python::attach(|py| work.call0(py).map(drop))))
}

pyo3_stub_gen::module_variable!("kibends._core.testing", "FAR", f64, testing::FAR);

/// The submodule `kibends._core.testing`, registered in `sys.modules` so
/// that it imports by its name.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "testing")?;
    m.add_class::<FreeSpace>()?;
    m.add_class::<Disc>()?;
    m.add_class::<Box>()?;
    m.add_class::<Walls>()?;
    m.add_class::<MisleadingWalls>()?;
    m.add_class::<Union>()?;
    m.add_class::<ReachIgnored>()?;
    m.add_class::<NanField>()?;
    m.add_function(wrap_pyfunction!(quadratic, &m)?)?;
    m.add_function(wrap_pyfunction!(in_pool, &m)?)?;
    m.add("FAR", testing::FAR)?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("kibends._core.testing", &m)?;
    Ok(())
}
