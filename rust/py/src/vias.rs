//! The moving vias and the pairwise via term as Python sees them
//! (`kibends_core::vias`).

use std::collections::{HashMap, HashSet};

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::build::{Via, Zone, zones_of};
use crate::energy::Params;
use crate::field::{Points, floats_of, points_of, same_lengths, vectors};
use crate::fields::ChainField;

/// `k` if it is below `count`.
fn index(k: usize, count: usize, what: &str) -> PyResult<usize> {
    if k < count {
        Ok(k)
    } else {
        Err(PyIndexError::new_err(format!("no {what} {k} of {count}")))
    }
}
use kibends_core::geometry::V2;
use kibends_core::rules::HoleRules;
use kibends_core::vias::{moves, pairs};

/// How `via` and `other`, each a ground via or not, interact in the
/// pairwise term, as `vias.PairKind`'s value; None if they do not
/// (`kibends_core::vias::pairs::pair_kind`).
#[gen_stub_pyfunction]
#[pyfunction]
fn pair_kind(
    via: PyRef<'_, Via>,
    ground: bool,
    other: PyRef<'_, Via>,
    other_ground: bool,
) -> Option<u8> {
    pairs::pair_kind(&via.inner, ground, &other.inner, other_ground).map(|k| k as u8)
}

fn pair_vias(
    vias: &[PyRef<'_, Via>],
    grounds: &[bool],
    clearances: &[f64],
    zones: &[Vec<PyRef<'_, Zone>>],
) -> PyResult<Vec<pairs::PairVia>> {
    same_lengths(&[vias.len(), grounds.len(), clearances.len(), zones.len()])?;
    Ok(vias
        .iter()
        .zip(grounds)
        .zip(clearances)
        .zip(zones)
        .map(|(((v, &ground), &clearance), zones)| pairs::PairVia {
            via: v.inner.clone(),
            ground,
            clearance,
            zones: zones_of(zones),
        })
        .collect())
}

/// The pairwise term of a problem's moving vias, with each other and with
/// the vias standing still (`kibends_core::vias::pairs::ViaPairs`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct ViaPairs {
    pub(crate) inner: std::sync::Arc<pairs::ViaPairs>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ViaPairs {
    /// The moving `vias`, each a ground via or not, with its netclass
    /// clearance and zones, keeping the distances `apart` gives from each
    /// other; and the `statics`, each a ground via or not with its netclass
    /// clearance.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vias: Vec<PyRef<'_, Via>>,
        grounds: Vec<bool>,
        clearances: Vec<f64>,
        zones: Vec<Vec<PyRef<'_, Zone>>>,
        apart: Vec<HashMap<usize, f64>>,
        statics: Vec<PyRef<'_, Via>>,
        static_grounds: Vec<bool>,
        static_clearances: Vec<f64>,
        params: Params,
        hole_clearance: f64,
        hole_to_hole: f64,
    ) -> PyResult<Self> {
        let moving = pair_vias(&vias, &grounds, &clearances, &zones)?;
        same_lengths(&[moving.len(), apart.len()])?;
        let none: Vec<Vec<PyRef<'_, Zone>>> = statics.iter().map(|_| Vec::new()).collect();
        let still = pair_vias(&statics, &static_grounds, &static_clearances, &none)?;
        Ok(Self {
            inner: std::sync::Arc::new(pairs::ViaPairs::new(
                moving,
                &apart,
                still,
                &params.0,
                HoleRules {
                    hole_clearance,
                    hole_to_hole,
                },
            )),
        })
    }

    /// The distance moving via k's centre keeps from `other`, of netclass
    /// `clearance`, as its static field would count it; None if they share
    /// no layer.
    fn kept(&self, k: usize, other: PyRef<'_, Via>, clearance: f64) -> PyResult<Option<f64>> {
        let k = index(k, self.inner.via_count(), "via")?;
        Ok(self.inner.kept(k, &other.inner, clearance))
    }

    /// Whether the term has no pair.
    #[getter]
    fn empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Per via, the problem's vias it pairs with.
    #[getter]
    fn paired(&self) -> Vec<HashSet<usize>> {
        self.inner.paired().to_vec()
    }

    /// The term's value with the problem's vias at (n, 2) `positions`, and
    /// its gradient at each via's centre.
    fn energy<'py>(
        &self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<(f64, Points<'py>)> {
        let positions = points_of(&positions)?;
        same_lengths(&[positions.len(), self.inner.via_count()])?;
        let (value, gradient) = self.inner.energy(&positions);
        Ok((value, vectors(py, &gradient)?))
    }
}

/// A carried chain end: its chain's index, whether it is the chain's last
/// point, and its offset from the via's centre.
type CarriedEnd = (usize, bool, (f64, f64));

/// The moving vias of a run and the chains they move among
/// (`kibends_core::vias::moves::MovingVias`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct MovingVias {
    inner: moves::MovingVias,
}

#[gen_stub_pymethods]
#[pymethods]
impl MovingVias {
    /// The chains, by their nets, layer ordinals, widths, netclass
    /// clearances and first and last points, (n, 2) each; and the vias,
    /// each with its netclass clearance, its zones, and the ends it
    /// carries, each a chain index and whether it is the chain's last
    /// point.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nets: Vec<String>,
        layers: Vec<i32>,
        widths: PyReadonlyArray1<'_, f64>,
        clearances: PyReadonlyArray1<'_, f64>,
        firsts: PyReadonlyArray2<'_, f64>,
        lasts: PyReadonlyArray2<'_, f64>,
        vias: Vec<PyRef<'_, Via>>,
        via_clearances: Vec<f64>,
        zones: Vec<Vec<PyRef<'_, Zone>>>,
        ends: Vec<Vec<(usize, bool)>>,
        hole_clearance: f64,
        hole_to_hole: f64,
    ) -> PyResult<Self> {
        let widths = floats_of(&widths);
        let clearances = floats_of(&clearances);
        let firsts = points_of(&firsts)?;
        let lasts = points_of(&lasts)?;
        same_lengths(&[
            nets.len(),
            layers.len(),
            widths.len(),
            clearances.len(),
            firsts.len(),
            lasts.len(),
        ])?;
        same_lengths(&[vias.len(), via_clearances.len(), zones.len(), ends.len()])?;
        if ends.iter().flatten().any(|&(chain, _)| chain >= nets.len()) {
            return Err(PyValueError::new_err("a via carries the end of no chain"));
        }
        let chains = (0..nets.len())
            .map(|i| moves::Chain {
                net: nets[i].clone(),
                layer: layers[i],
                width: widths[i],
                clearance: clearances[i],
                first: firsts[i],
                last: lasts[i],
            })
            .collect();
        let vias = vias
            .iter()
            .zip(via_clearances)
            .zip(&zones)
            .zip(ends)
            .map(|(((v, clearance), zones), ends)| moves::Moving {
                via: v.inner.clone(),
                clearance,
                zones: zones_of(zones),
                ends,
            })
            .collect();
        Ok(Self {
            inner: moves::MovingVias::new(
                chains,
                vias,
                HoleRules {
                    hole_clearance,
                    hole_to_hole,
                },
            ),
        })
    }

    /// The chain ends via v carries: each chain's index, whether it is the
    /// chain's last point, and its offset from the via's centre.
    fn ends(&self, v: usize) -> PyResult<Vec<CarriedEnd>> {
        let v = index(v, self.inner.via_count(), "via")?;
        Ok(self
            .inner
            .ends(v)
            .iter()
            .map(|e| (e.chain, e.last, (e.offset.x, e.offset.y)))
            .collect())
    }

    /// The vias near chain i, at (n, 2) `positions`, as obstacles to it;
    /// None when none is one.
    fn via_obstacles_for_chain(
        &self,
        i: usize,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Option<ChainField>> {
        let i = index(i, self.inner.chain_count(), "chain")?;
        let positions: Vec<V2> = points_of(&positions)?;
        same_lengths(&[positions.len(), self.inner.via_count()])?;
        Ok(self
            .inner
            .via_obstacles_for_chain(i, &positions)
            .map(|field| ChainField {
                inner: std::sync::Arc::new(field),
            }))
    }

    /// The distance via v's centre keeps from via u's; None if no rule
    /// applies.
    fn apart(&self, v: usize, u: usize) -> PyResult<Option<f64>> {
        let count = self.inner.via_count();
        Ok(self
            .inner
            .apart(index(v, count, "via")?, index(u, count, "via")?))
    }

    /// The chains near via v, and how far each keeps its centreline from
    /// v's centre.
    fn rings(&self, v: usize) -> PyResult<Vec<(usize, f64)>> {
        Ok(self.inner.rings(index(v, self.inner.via_count(), "via")?))
    }
}

/// Registers the via terms in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pair_kind, m)?)?;
    m.add_class::<ViaPairs>()?;
    m.add_class::<MovingVias>()?;
    Ok(())
}
