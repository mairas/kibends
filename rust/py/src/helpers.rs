//! The diff pair helpers, segment merging and the contact terms as Python
//! sees them (`kibends_core::pairs`, `kibends_core::simplify`,
//! `kibends_core::contacts`).

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::arrays::{BoolArg, BoolArray, IntpArg};
use crate::energy::Integrand;
use crate::field::{
    Flags, Floats, Points, flags_of, floats, floats_of, points_of, same_lengths, vectors,
};
use crate::fields::FieldArg;
use kibends_core::field::Field;
use kibends_core::{contacts, pairs, simplify};

/// The name the other net of a diff pair would have, by KiCad's router's
/// rule, or None (`kibends_core::pairs::complementary_net`).
#[gen_stub_pyfunction]
#[pyfunction]
fn complementary_net(name: &str) -> Option<String> {
    pairs::complementary_net(name)
}

/// `net` and the other net of its diff pair, if it has one.
#[gen_stub_pyfunction]
#[gen_stub(override_return_type(type_repr = "builtins.frozenset[builtins.str]", imports = ("builtins")))]
#[pyfunction]
fn diff_pair_nets<'py>(py: Python<'py>, net: &str) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, pairs::diff_pair_nets(net))
}

/// The smallest centreline bend radius at which the inner track's inner
/// edge still has a radius of at least zero.
#[gen_stub_pyfunction]
#[pyfunction]
fn bend_radius_limit(gap: f64, width: f64) -> f64 {
    pairs::bend_radius_limit(gap, width)
}

fn turning_of(centreline: usize, turning: Option<&BoolArg<'_>>) -> PyResult<Option<Vec<bool>>> {
    turning
        .map(|t| {
            let t = flags_of(t);
            same_lengths(&[t.len(), centreline])?;
            Ok(t)
        })
        .transpose()
}

/// Each edge's bend radius, 0 beside a fold and for an edge of no length;
/// only the turns and folds at the vertices `turning` count, every vertex
/// by default (`kibends_core::pairs::bend_radii`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (centreline, turning = None))]
fn bend_radii<'py>(
    py: Python<'py>,
    centreline: PyReadonlyArray2<'_, f64>,
    turning: Option<BoolArg<'_>>,
) -> PyResult<Floats<'py>> {
    let centreline = points_of(&centreline)?;
    let turning = turning_of(centreline.len(), turning.as_ref())?;
    Ok(floats(
        py,
        pairs::bend_radii(&centreline, turning.as_deref()),
    ))
}

/// Whether each bend radius is below `limit`, beyond round-off.
#[gen_stub_pyfunction]
#[pyfunction]
fn too_tight<'py>(py: Python<'py>, radius: PyReadonlyArray1<'_, f64>, limit: f64) -> Flags<'py> {
    let tight: Vec<bool> = floats_of(&radius)
        .into_iter()
        .map(|r| pairs::too_tight(r, limit))
        .collect();
    BoolArray(tight.into_pyarray(py))
}

/// The centreline's smallest bend radius and whether it is below the limit
/// for tracks of `width` `gap` apart, counting the turns at the vertices
/// `turning` (`kibends_core::pairs::check_bend_radius`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (centreline, gap, width, turning = None))]
fn check_bend_radius(
    centreline: PyReadonlyArray2<'_, f64>,
    gap: f64,
    width: f64,
    turning: Option<BoolArg<'_>>,
) -> PyResult<(f64, bool)> {
    let centreline = points_of(&centreline)?;
    let turning = turning_of(centreline.len(), turning.as_ref())?;
    pairs::check_bend_radius(&centreline, gap, width, turning.as_deref())
        .ok_or_else(|| PyValueError::new_err("a centreline of no edge has no bend radius"))
}

/// The vertices of `polyline` that merging keeps: no vertex of `fixed` is
/// merged, and each merged vertex lies within its `tolerances` of a chord
/// clear in `field` (`kibends_core::simplify::kept_vertices`).
#[gen_stub_pyfunction]
#[pyfunction]
fn kept_vertices(
    polyline: PyReadonlyArray2<'_, f64>,
    fixed: BoolArg<'_>,
    field: FieldArg,
    tolerances: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<usize>> {
    let polyline = points_of(&polyline)?;
    let fixed = flags_of(&fixed);
    let tolerances = floats_of(&tolerances);
    same_lengths(&[polyline.len(), fixed.len(), tolerances.len()])?;
    Ok(simplify::kept_vertices(
        &polyline,
        &fixed,
        field.0.as_ref(),
        &tolerances,
    ))
}

/// Each vertex's contact as an index among `count` regions, -1 for none.
fn contacts_of(
    contact_of: &IntpArg<'_>,
    points: usize,
    count: usize,
) -> PyResult<Vec<Option<usize>>> {
    let contact_of: Vec<Option<usize>> = contact_of
        .0
        .as_array()
        .iter()
        .map(|&c| usize::try_from(c).ok())
        .collect();
    same_lengths(&[contact_of.len(), points])?;
    if contact_of.iter().flatten().any(|&c| c >= count) {
        return Err(PyValueError::new_err("a contact is not among the regions"));
    }
    Ok(contact_of)
}

/// The stops that keep each vertex of `points` that keeps a contact inside
/// its region among `within`, and their gradient at every vertex
/// (`kibends_core::contacts::contact_stops`).
#[gen_stub_pyfunction]
#[pyfunction]
fn contact_stops<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    contact_of: IntpArg<'_>,
    within: Vec<FieldArg>,
    width: f64,
    limit: PyRef<'_, Integrand>,
) -> PyResult<(f64, Points<'py>)> {
    let points = points_of(&points)?;
    let contact_of = contacts_of(&contact_of, points.len(), within.len())?;
    let regions: Vec<&dyn Field> = within.iter().map(|f| f.0.as_ref()).collect();
    let (value, gradient) =
        contacts::contact_stops(&points, &contact_of, &regions, width, &limit.inner);
    Ok((value, vectors(py, &gradient)?))
}

/// The vertices of `points` that keep a contact and lie outside its region
/// among `within` by more than `tolerance`
/// (`kibends_core::contacts::contacts_left`).
#[gen_stub_pyfunction]
#[pyfunction]
fn contacts_left<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    contact_of: IntpArg<'_>,
    within: Vec<FieldArg>,
    tolerance: f64,
) -> PyResult<Flags<'py>> {
    let points = points_of(&points)?;
    let contact_of = contacts_of(&contact_of, points.len(), within.len())?;
    let regions: Vec<&dyn Field> = within.iter().map(|f| f.0.as_ref()).collect();
    let left = contacts::contacts_left(&points, &contact_of, &regions, tolerance);
    Ok(BoolArray(left.into_pyarray(py)))
}

/// Registers the helpers in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(complementary_net, m)?)?;
    m.add_function(wrap_pyfunction!(diff_pair_nets, m)?)?;
    m.add_function(wrap_pyfunction!(bend_radius_limit, m)?)?;
    m.add_function(wrap_pyfunction!(bend_radii, m)?)?;
    m.add_function(wrap_pyfunction!(too_tight, m)?)?;
    m.add_function(wrap_pyfunction!(check_bend_radius, m)?)?;
    m.add_function(wrap_pyfunction!(kept_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(contact_stops, m)?)?;
    m.add_function(wrap_pyfunction!(contacts_left, m)?)?;
    Ok(())
}
