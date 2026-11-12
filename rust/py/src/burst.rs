//! One L-BFGS burst (`kibends_core::burst`) over an `Objective`, evaluated
//! without leaving Rust; Python is asked only once per iteration, for
//! signals and for the caller's `stop`.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction};

use crate::arrays::{Floats, floats, floats_of};
use crate::problem::Objective;
use kibends_core::burst::{self, Settings};

/// What a burst reached (`kibends_core::burst::Burst`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct BurstResult {
    /// The last accepted point, x.
    points: Vec<f64>,
    /// The energy there.
    #[pyo3(get)]
    energy: f64,
    /// The accepted L-BFGS iterations.
    #[pyo3(get)]
    iterations: u64,
    /// The objective's evaluations.
    #[pyo3(get)]
    evaluations: u64,
    /// Seconds the evaluations took.
    #[pyo3(get)]
    evaluating: f64,
    /// What ended it, as scipy's L-BFGS-B words it.
    #[pyo3(get)]
    message: String,
    /// Whether an iteration was not accepted.
    #[pyo3(get)]
    abnormal: bool,
    /// Whether `stop` ended it before its iteration cap.
    #[pyo3(get)]
    cut: bool,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl BurstResult {
    /// The last accepted point, x.
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.points.clone())
    }
}

/// Minimises `objective` from `start` with L-BFGS, over y with x =
/// `scale` ⊙ y if a scale is given, until `max_iterations`, `ftol` or
/// `gtol` ends it (`kibends_core::burst`). After each iteration it checks
/// for signals and asks `stop`, if given, which ends the burst by
/// answering true before the cap. An exception raised there or by the
/// objective ends the burst and propagates.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (objective, start, scale, max_iterations, ftol, gtol, stop = None))]
#[allow(clippy::too_many_arguments)]
fn run_burst(
    py: Python<'_>,
    objective: PyRef<'_, Objective>,
    start: PyReadonlyArray1<'_, f64>,
    scale: Option<PyReadonlyArray1<'_, f64>>,
    max_iterations: u64,
    ftol: f64,
    gtol: f64,
    #[gen_stub(override_type(
        type_repr = "typing.Optional[collections.abc.Callable[[], builtins.bool]]",
        imports = ("typing", "collections.abc", "builtins")
    ))]
    stop: Option<Py<PyAny>>,
) -> PyResult<BurstResult> {
    let start = floats_of(&start);
    let scale = scale.map(|s| floats_of(&s));
    if let Some(s) = &scale {
        crate::arrays::check_count("scales", s.len(), start.len())?;
    }
    let settings = Settings {
        max_iterations,
        ftol,
        gtol,
    };
    let mut evaluate = |x: &[f64]| objective.evaluate(py, x);
    let mut asked = || -> PyResult<bool> {
        py.check_signals()?;
        match &stop {
            Some(stop) => stop.call0(py)?.is_truthy(py),
            None => Ok(false),
        }
    };
    let b = burst::run(
        &mut evaluate,
        &start,
        scale.as_deref(),
        &settings,
        &mut asked,
    )?;
    Ok(BurstResult {
        message: b.ending.message(),
        abnormal: b.abnormal(),
        cut: b.cut(),
        points: b.point,
        energy: b.energy,
        iterations: b.iterations,
        evaluations: b.evaluations,
        evaluating: b.evaluating.as_secs_f64(),
    })
}

/// Registers the burst's class and function in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BurstResult>()?;
    m.add_function(wrap_pyfunction!(run_burst, m)?)?;
    Ok(())
}
