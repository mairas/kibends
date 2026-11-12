//! Simplification of a run's members as Python sees it
//! (`kibends_core::simplify::simplify_chains`).

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::arrays::{
    BoolArg, Points, Polyline, check_count, invalid, masks_of, polylines_of, vectors_list,
};
use crate::problem::Member;
use kibends_core::members;
use kibends_core::simplify::{self, Optimised};

/// Each of `members` where the optimizer left it, at `polylines` with its
/// `fixed` vertices and a section's `directions` pins, simplified within
/// `tolerance`, in order, each against its static field and the others on
/// its layer that it does not ignore, at their latest shapes.
#[gen_stub_pyfunction]
#[pyfunction]
fn simplify_chains<'py>(
    py: Python<'py>,
    members: Vec<PyRef<'_, Member>>,
    polylines: Vec<Polyline<'_>>,
    fixed: Vec<BoolArg<'_>>,
    directions: Vec<Polyline<'_>>,
    tolerance: f64,
) -> PyResult<Vec<Points<'py>>> {
    let polylines = polylines_of(&polylines)?;
    let fixed = masks_of(&fixed, &polylines)?;
    let directions = polylines_of(&directions)?;
    check_count("polylines", polylines.len(), members.len())?;
    check_count("direction pin lists", directions.len(), members.len())?;
    let members: Vec<members::Member> = members.iter().map(|m| m.inner.clone()).collect();
    for (member, points) in members.iter().zip(&polylines) {
        let moving_section = member.section.as_ref().is_some_and(|s| !s.offset.held);
        let least = if moving_section { 2 } else { 1 };
        if points.len() < least {
            return Err(invalid(format!(
                "a polyline of {} vertices where {least} are needed",
                points.len()
            )));
        }
    }
    let optimised: Vec<Optimised<'_>> = polylines
        .iter()
        .zip(&fixed)
        .zip(&directions)
        .map(|((polyline, fixed), directions)| Optimised {
            polyline,
            fixed,
            directions,
        })
        .collect();
    let simplified = simplify::simplify_chains(&members, &optimised, tolerance);
    vectors_list(py, &simplified)
}

/// Registers the simplification in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify_chains, m)?)?;
    Ok(())
}
