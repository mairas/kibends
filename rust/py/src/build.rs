//! The clearance rules and the static fields' builders as Python sees
//! them (`kibends_core::rules`, `kibends_core::field::build`): vias,
//! zones and a layer's copper read from the snapshot into plain records,
//! and the builders, which return each field with the sources of its items.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{
    BoolArg, Floats, Indices, IntpArg, IntpArray, flags_of, floats, floats_of, points_of,
    same_lengths,
};
use crate::field::{Obstacles, Shapes};
use crate::fields::ChainField;
use kibends_core::field::build::{self, BuiltField, Layer, MovingVia, PairTrack, ViaRules};
use kibends_core::geometry::{Shape, V2};
use kibends_core::rules::{self, HoleRules, Span};

/// A via, as the rules take it: its id, net, centre, ring diameter and
/// drill, and the span of its drill as copper layer ordinals.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Via {
    pub(crate) inner: rules::Via,
}

#[gen_stub_pymethods]
#[pymethods]
impl Via {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        net: String,
        x: f64,
        y: f64,
        diameter: f64,
        drill: f64,
        low: i32,
        high: i32,
    ) -> Self {
        Self {
            inner: rules::Via {
                id,
                net,
                position: V2::new(x, y),
                diameter,
                drill,
                span: Span { low, high },
            },
        }
    }
}

/// A zone a via stays inside, as the rules take it: its clearance to other
/// nets' copper, the margin the via's centre keeps inside it, and the
/// ordinals of the layers on which it reaches the via's ring.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Zone {
    pub(crate) inner: rules::Zone,
}

#[gen_stub_pymethods]
#[pymethods]
impl Zone {
    #[new]
    fn new(clearance: f64, margin: f64, layers: Vec<i32>) -> Self {
        Self {
            inner: rules::Zone {
                clearance,
                margin,
                layers,
            },
        }
    }
}

pub(crate) fn zones_of(zones: &[PyRef<'_, Zone>]) -> Vec<rules::Zone> {
    zones.iter().map(|z| z.inner.clone()).collect()
}

/// The copper of one layer as the builders take it
/// (`kibends_core::field::build::Layer`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct BoardLayer {
    inner: Arc<Layer>,
}

/// The shapes of `shapes`, one per item of `count` items.
fn shapes_of(shapes: &Shapes, count: usize) -> PyResult<Vec<Shape>> {
    same_lengths(&[shapes.shapes.len(), count])?;
    Ok(shapes.shapes.clone())
}

#[gen_stub_pymethods]
#[pymethods]
impl BoardLayer {
    /// The layer `name` of `ordinal`: its pads, each with its net,
    /// position, drill, whether plated, clearance override (NaN for none)
    /// and the index of its copper among `pad_shapes`, -1 for none; its
    /// tracks' nets, widths and centrelines; its vias; its graphics' nets
    /// and shapes; its keepouts of tracks and of vias; the board outline,
    /// None if Edge.Cuts is not closed; and the netclass clearance per net,
    /// `default_clearance` for any other.
    #[new]
    #[pyo3(signature = (
        name, ordinal, pad_nets, pad_positions, pad_drills, pad_plated,
        pad_overrides, pad_shape_of, pad_shapes, track_nets, track_widths,
        track_shapes, vias, graphic_nets, graphic_shapes, track_keepouts,
        via_keepouts, outline, clearances, default_clearance,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        ordinal: i32,
        pad_nets: Vec<String>,
        pad_positions: PyReadonlyArray2<'_, f64>,
        pad_drills: PyReadonlyArray1<'_, f64>,
        pad_plated: BoolArg<'_>,
        pad_overrides: PyReadonlyArray1<'_, f64>,
        pad_shape_of: IntpArg<'_>,
        pad_shapes: PyRef<'_, Shapes>,
        track_nets: Vec<String>,
        track_widths: PyReadonlyArray1<'_, f64>,
        track_shapes: PyRef<'_, Shapes>,
        vias: Vec<PyRef<'_, Via>>,
        graphic_nets: Vec<String>,
        graphic_shapes: PyRef<'_, Shapes>,
        track_keepouts: PyRef<'_, Shapes>,
        via_keepouts: PyRef<'_, Shapes>,
        outline: Option<PyRef<'_, Shapes>>,
        clearances: HashMap<String, f64>,
        default_clearance: f64,
    ) -> PyResult<Self> {
        let positions = points_of(&pad_positions)?;
        let drills = floats_of(&pad_drills);
        let plated = flags_of(&pad_plated);
        let overrides = floats_of(&pad_overrides);
        let shape_of = pad_shape_of.0.as_array().to_vec();
        same_lengths(&[
            pad_nets.len(),
            positions.len(),
            drills.len(),
            plated.len(),
            overrides.len(),
            shape_of.len(),
        ])?;
        let shape = |k: isize| -> PyResult<Option<Shape>> {
            if k < 0 {
                return Ok(None);
            }
            usize::try_from(k)
                .ok()
                .and_then(|k| pad_shapes.shapes.get(k))
                .cloned()
                .map(Some)
                .ok_or_else(|| PyValueError::new_err("a pad's shape index is out of range"))
        };
        let pads = pad_nets
            .into_iter()
            .enumerate()
            .map(|(k, net)| {
                Ok(build::Pad {
                    net,
                    position: positions[k],
                    drill: drills[k],
                    plated: plated[k],
                    shape: shape(shape_of[k])?,
                    clearance_override: (!overrides[k].is_nan()).then_some(overrides[k]),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let widths = floats_of(&track_widths);
        same_lengths(&[track_nets.len(), widths.len()])?;
        let tracks = track_nets
            .into_iter()
            .zip(widths)
            .zip(shapes_of(&track_shapes, track_shapes.shapes.len())?)
            .map(|((net, width), centreline)| build::Track {
                net,
                width,
                centreline,
            })
            .collect::<Vec<_>>();
        if tracks.len() != track_shapes.shapes.len() {
            return Err(PyValueError::new_err("one centreline per track"));
        }
        let graphics: Vec<build::Graphic> = graphic_nets
            .iter()
            .zip(shapes_of(&graphic_shapes, graphic_nets.len())?)
            .map(|(net, shape)| build::Graphic {
                net: net.clone(),
                shape,
            })
            .collect();
        let outline = match outline {
            None => None,
            Some(o) => match o.shapes.as_slice() {
                [one] => Some(one.clone()),
                _ => return Err(PyValueError::new_err("the outline must be one shape")),
            },
        };
        Ok(Self {
            inner: Arc::new(Layer {
                name,
                ordinal,
                pads,
                tracks,
                vias: vias.iter().map(|v| v.inner.clone()).collect(),
                graphics,
                track_keepouts: track_keepouts.shapes.clone(),
                via_keepouts: via_keepouts.shapes.clone(),
                outline,
                clearances,
                default_clearance,
            }),
        })
    }

    /// Whether the layer has the board outline: Edge.Cuts is closed.
    #[getter]
    fn has_outline(&self) -> bool {
        self.inner.outline.is_some()
    }
}

/// A built field's items' sources as arrays: each item's kind (`Source`),
/// the layer of its record and its index there, the layer of its copper
/// (-1 for none), and its inflation.
type PickedArrays<'py> = (
    Indices<'py>,
    Indices<'py>,
    Indices<'py>,
    Indices<'py>,
    Floats<'py>,
);

/// A built field and its items' sources.
type Built<'py> = (Obstacles, PickedArrays<'py>);

fn built<'py>(py: Python<'py>, built: BuiltField) -> PyResult<Built<'py>> {
    let index = |values: Vec<isize>| IntpArray(values.into_pyarray(py));
    let picked = &built.picked;
    // An index into a Rust slice fits an isize, and a source's code is small.
    let source = picked.iter().map(|p| p.source as isize).collect();
    let layer = picked.iter().map(|p| p.layer as isize).collect();
    let item = picked.iter().map(|p| p.index as isize).collect();
    let copper = picked
        .iter()
        .map(|p| p.copper.map_or(-1, |c| c as isize))
        .collect();
    let inflations = (0..picked.len())
        .map(|k| built.field.edges().inflations()[k])
        .collect();
    Ok((
        Obstacles::of(py, built.field)?,
        (
            index(source),
            index(layer),
            index(item),
            index(copper),
            floats(py, inflations),
        ),
    ))
}

/// Obstacles for a track of `net` and `width` mm on `layer`
/// (`kibends_core::field::build::chain_obstacles`).
#[gen_stub_pyfunction]
#[pyfunction]
fn build_obstacles<'py>(
    py: Python<'py>,
    layer: PyRef<'_, BoardLayer>,
    net: &str,
    width: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> PyResult<Built<'py>> {
    built(
        py,
        build::chain_obstacles(&layer.inner, net, width, edge_clearance, hole_clearance),
    )
}

/// Obstacles for the centreline of a coupled section of a differential
/// pair on `layer`, its tracks of `width`, `gap` mm apart, of nets `a_net`
/// and `b_net` at `a_points` and `b_points`
/// (`kibends_core::field::build::pair_obstacles`).
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_pair_obstacles<'py>(
    py: Python<'py>,
    layer: PyRef<'_, BoardLayer>,
    a_net: &str,
    a_points: PyReadonlyArray2<'_, f64>,
    b_net: &str,
    b_points: PyReadonlyArray2<'_, f64>,
    width: f64,
    gap: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> PyResult<Built<'py>> {
    let a = points_of(&a_points)?;
    let b = points_of(&b_points)?;
    built(
        py,
        build::pair_obstacles(
            &layer.inner,
            PairTrack {
                net: a_net,
                points: &a,
            },
            PairTrack {
                net: b_net,
                points: &b,
            },
            width,
            gap,
            edge_clearance,
            hole_clearance,
        ),
    )
}

fn layers_of<'a>(layers: &'a [PyRef<'_, BoardLayer>]) -> Vec<&'a Layer> {
    layers.iter().map(|l| l.inner.as_ref()).collect()
}

fn ids(ids: &[String]) -> HashSet<&str> {
    ids.iter().map(String::as_str).collect()
}

/// A moving via's field over the layers of `layers` it spans, and the vias
/// whose ids are in `split` apart, None if none is one
/// (`kibends_core::field::build::via_obstacles`).
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn split_via_obstacles<'py>(
    py: Python<'py>,
    layers: Vec<PyRef<'_, BoardLayer>>,
    via: PyRef<'_, Via>,
    split: Vec<String>,
    moving: Vec<String>,
    edge_clearance: f64,
    hole_clearance: f64,
    hole_to_hole: f64,
    zones: Vec<PyRef<'_, Zone>>,
) -> PyResult<(Built<'py>, Option<Built<'py>>)> {
    let fields = build::via_obstacles(
        &layers_of(&layers),
        &via.inner,
        &ids(&split),
        &ids(&moving),
        ViaRules {
            edge_clearance,
            holes: HoleRules {
                hole_clearance,
                hole_to_hole,
            },
        },
        &zones_of(&zones),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        built(py, fields.field)?,
        fields.split.map(|s| built(py, s)).transpose()?,
    ))
}

/// What of `via`'s net on the layers of `layers` it spans stops it: its
/// pads and holes, and its vias, each None if none is one
/// (`kibends_core::field::build::own_net_stops`).
#[gen_stub_pyfunction]
#[pyfunction]
fn own_net_stops<'py>(
    py: Python<'py>,
    layers: Vec<PyRef<'_, BoardLayer>>,
    via: PyRef<'_, Via>,
    moving: Vec<String>,
    hole_clearance: f64,
    hole_to_hole: f64,
    allow_via_in_pad: bool,
) -> PyResult<(Option<Built<'py>>, Option<ChainField>)> {
    let stops = build::own_net_stops(
        &layers_of(&layers),
        &via.inner,
        &ids(&moving),
        HoleRules {
            hole_clearance,
            hole_to_hole,
        },
        allow_via_in_pad,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        stops.items.map(|s| built(py, s)).transpose()?,
        stops.vias.map(|v| ChainField { inner: Arc::new(v) }),
    ))
}

/// `vias`, at their positions, as obstacles to a track of `net`, `width`
/// and netclass `clearance` on the layer of `ordinal`, or None if none is
/// one; `clearances` and `zones` give each via's netclass clearance and
/// the zones it stays inside
/// (`kibends_core::field::build::moving_via_obstacles`).
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn moving_via_obstacles(
    ordinal: i32,
    net: &str,
    width: f64,
    clearance: f64,
    vias: Vec<PyRef<'_, Via>>,
    clearances: Vec<f64>,
    hole_clearance: f64,
    zones: Vec<Vec<PyRef<'_, Zone>>>,
) -> PyResult<Option<ChainField>> {
    same_lengths(&[vias.len(), clearances.len(), zones.len()])?;
    let zones: Vec<Vec<rules::Zone>> = zones.iter().map(|z| zones_of(z)).collect();
    let moving: Vec<MovingVia<'_>> = vias
        .iter()
        .zip(clearances)
        .zip(&zones)
        .map(|((v, clearance), zones)| MovingVia {
            via: &v.inner,
            clearance,
            zones,
        })
        .collect();
    Ok(
        build::moving_via_obstacles(ordinal, net, width, clearance, &moving, hole_clearance).map(
            |field| ChainField {
                inner: Arc::new(field),
            },
        ),
    )
}

/// The distance `via`'s centre keeps from `other`'s, None if no rule
/// applies (`kibends_core::rules::via_via_inflation`).
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn via_via_inflation(
    via: PyRef<'_, Via>,
    clearance: f64,
    other: PyRef<'_, Via>,
    other_clearance: f64,
    hole_clearance: f64,
    hole_to_hole: f64,
    zones: Vec<PyRef<'_, Zone>>,
    other_zones: Vec<PyRef<'_, Zone>>,
) -> Option<f64> {
    rules::via_via_inflation(
        &via.inner,
        clearance,
        &other.inner,
        other_clearance,
        HoleRules {
            hole_clearance,
            hole_to_hole,
        },
        &zones_of(&zones),
        &zones_of(&other_zones),
    )
}

/// The distance `via`'s centre keeps from the centreline of a chain of
/// `net`, `width` and netclass `chain_clearance`, None for a chain of its
/// net (`kibends_core::rules::via_chain_inflation`).
#[gen_stub_pyfunction]
#[pyfunction]
fn via_chain_inflation(
    via: PyRef<'_, Via>,
    clearance: f64,
    net: &str,
    width: f64,
    chain_clearance: f64,
    hole_clearance: f64,
    zones: Vec<PyRef<'_, Zone>>,
) -> Option<f64> {
    rules::via_chain_inflation(
        &via.inner,
        clearance,
        rules::Track {
            net,
            width,
            clearance: chain_clearance,
        },
        hole_clearance,
        &zones_of(&zones),
    )
}

/// The distance a track of `width` and `clearance` keeps its centreline
/// from that of a track of another net (`kibends_core::rules::track_inflation`).
#[gen_stub_pyfunction]
#[pyfunction]
fn track_inflation(width: f64, clearance: f64, other_width: f64, other_clearance: f64) -> f64 {
    rules::track_inflation(width, clearance, other_width, other_clearance)
}

/// Registers the rules and builders in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Via>()?;
    m.add_class::<Zone>()?;
    m.add_class::<BoardLayer>()?;
    m.add_function(wrap_pyfunction!(build_obstacles, m)?)?;
    m.add_function(wrap_pyfunction!(build_pair_obstacles, m)?)?;
    m.add_function(wrap_pyfunction!(split_via_obstacles, m)?)?;
    m.add_function(wrap_pyfunction!(own_net_stops, m)?)?;
    m.add_function(wrap_pyfunction!(moving_via_obstacles, m)?)?;
    m.add_function(wrap_pyfunction!(via_via_inflation, m)?)?;
    m.add_function(wrap_pyfunction!(via_chain_inflation, m)?)?;
    m.add_function(wrap_pyfunction!(track_inflation, m)?)?;
    Ok(())
}
