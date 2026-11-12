//! The static field as Python sees it: shapes built from shapely's plain
//! arrays, `Obstacles` and its `StaticEdges`, and zone depth.

use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{BoolArg, BoolArray, IntpArg, IntpArray};
use kibends_core::field::edges::StaticDistances;
use kibends_core::field::obstacles::Item;
use kibends_core::field::{Kind, edges, obstacles};
use kibends_core::geometry::{PlainShapes, Shape, V2, zone_depth as core_zone_depth};

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

fn distances_arrays<'py>(
    py: Python<'py>,
    found: StaticDistances,
) -> (Indices<'py>, Indices<'py>, Floats<'py>) {
    (
        indices(py, &found.point),
        indices(py, &found.obstacle),
        floats(py, found.distance),
    )
}

/// Geometries as plain coordinates: each shape's parts, each part a polygon
/// of rings, its exterior first, or a line, a point a line of one
/// coordinate. Built from shapely's arrays (`obstacles.plain_shapes`):
/// the coordinates, and where each ring ends in them, each part in the
/// rings, and each shape in the parts; whether each part is a polygon, and
/// each shape's shapely type id.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Shapes {
    pub(crate) shapes: Vec<Shape>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Shapes {
    #[new]
    fn new(
        coords: PyReadonlyArray2<'_, f64>,
        ring_ends: IntpArg<'_>,
        part_ends: IntpArg<'_>,
        part_polygon: BoolArg<'_>,
        shape_ends: IntpArg<'_>,
        type_ids: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<Self> {
        let coords: Vec<[f64; 2]> = points_of(&coords)?.iter().map(|p| p.array()).collect();
        let ends = |array: &IntpArg<'_>, count: usize| -> PyResult<Vec<usize>> {
            let ends = indices_of(array, count + 1)?;
            if ends.windows(2).any(|w| w[1] < w[0]) || ends.last().is_some_and(|&e| e != count) {
                return Err(PyValueError::new_err(
                    "ends must ascend to the count they end",
                ));
            }
            Ok(ends)
        };
        let ring_ends = ends(&ring_ends, coords.len())?;
        let part_ends = ends(&part_ends, ring_ends.len())?;
        let part_polygon = flags_of(&part_polygon);
        let shape_ends = ends(&shape_ends, part_ends.len())?;
        let type_ids = type_ids.as_array().to_vec();
        same_lengths(&[part_ends.len(), part_polygon.len()])?;
        same_lengths(&[shape_ends.len(), type_ids.len()])?;
        let plain = PlainShapes {
            coords: &coords,
            ring_ends: &ring_ends,
            part_ends: &part_ends,
            part_polygon: &part_polygon,
            shape_ends: &shape_ends,
            type_ids: &type_ids,
        };
        Ok(Self {
            shapes: Shape::from_plain(&plain),
        })
    }

    fn __len__(&self) -> usize {
        self.shapes.len()
    }
}

/// A static field's obstacles as edges (`kibends_core::field::edges`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct StaticEdges {
    pub(crate) inner: Arc<edges::StaticEdges>,
}

#[gen_stub_pymethods]
#[pymethods]
impl StaticEdges {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Each item's obstacle.
    #[getter]
    fn obstacles<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.obstacles())
    }

    /// Each item's obstacle as a clearance check judges it: its obstacle,
    /// or for an `Obstacles` field its connected copper of one net on one
    /// layer.
    #[getter]
    fn clearance_obstacles<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.clearance_obstacles())
    }

    /// Each point's exact signed distance to each obstacle nearer than
    /// `below`, as a clearance check judges obstacles: point, obstacle and
    /// distance, by point then obstacle.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<'_, f64>,
        below: f64,
    ) -> PyResult<(Indices<'py>, Indices<'py>, Floats<'py>)> {
        let coords = points_of(&coords)?;
        Ok(distances_arrays(py, self.inner.distances(&coords, below)))
    }
}

/// Signed distance to the nearest inflated obstacle, and segment checks:
/// the board's items and its edge (`kibends_core::field::obstacles`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct Obstacles {
    pub(crate) inner: Arc<obstacles::Obstacles>,
    edges: Py<StaticEdges>,
}

impl Obstacles {
    /// The Python field of `inner`.
    pub(crate) fn of(py: Python<'_>, inner: obstacles::Obstacles) -> PyResult<Self> {
        let inner = Arc::new(inner);
        let edges = Py::new(
            py,
            StaticEdges {
                inner: inner.edges().clone(),
            },
        )?;
        Ok(Self { inner, edges })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Obstacles {
    /// The items of `shapes`, each with its inflation and, for copper, its
    /// net, extent and layer; and the outside of `outline`, inflated by
    /// `edge_inflation`, as the board edge.
    #[new]
    #[pyo3(signature = (shapes, inflations, nets, extents, layers, outline, edge_inflation))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        shapes: PyRef<'_, Shapes>,
        inflations: PyReadonlyArray1<'_, f64>,
        nets: Vec<Option<String>>,
        extents: PyReadonlyArray1<'_, f64>,
        layers: Vec<Option<String>>,
        outline: Option<PyRef<'_, Shapes>>,
        edge_inflation: f64,
    ) -> PyResult<Self> {
        let inflations = floats_of(&inflations);
        let extents = floats_of(&extents);
        same_lengths(&[
            shapes.shapes.len(),
            inflations.len(),
            nets.len(),
            extents.len(),
            layers.len(),
        ])?;
        let items = shapes
            .shapes
            .iter()
            .zip(inflations)
            .zip(nets)
            .zip(extents)
            .zip(layers)
            .map(|((((shape, inflation), net), extent), layer)| Item {
                shape: shape.clone(),
                inflation,
                net,
                extent,
                layer,
            })
            .collect();
        let outline = match outline {
            None => None,
            Some(o) => match o.shapes.as_slice() {
                [one] => Some(one.clone()),
                _ => return Err(PyValueError::new_err("the outline must be one shape")),
            },
        };
        let inner = Arc::new(obstacles::Obstacles::new(items, outline, edge_inflation));
        let edges = Py::new(
            py,
            StaticEdges {
                inner: inner.edges().clone(),
            },
        )?;
        Ok(Self { inner, edges })
    }

    /// The field as edges, its items in order, then the board edge.
    #[getter]
    fn static_edges(&self, py: Python<'_>) -> Py<StaticEdges> {
        self.edges.clone_ref(py)
    }

    /// Distance (n,) and its unit gradient (n, 2) at (n, 2) points in mm.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn signed_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        let points = points_of(&points)?;
        let (distance, gradient) = self.inner.signed_distance(&points, reach);
        Ok((floats(py, distance), vectors(py, &gradient)?))
    }

    /// `signed_distance` as the energy takes it: the smooth minimum over the
    /// obstacles, from the field's edges.
    #[pyo3(signature = (points, reach = f64::INFINITY))]
    fn smooth_distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        reach: f64,
    ) -> PyResult<(Floats<'py>, Points<'py>)> {
        let points = points_of(&points)?;
        let (distance, gradient) = self.inner.smooth_distance(&points, reach);
        Ok((floats(py, distance), vectors(py, &gradient)?))
    }

    /// Whether each segment from starts[i] to ends[i] stays out of every
    /// inflated obstacle, by at most `tolerance`, and inside the inflated
    /// board edge.
    #[pyo3(signature = (starts, ends, tolerance = 0.0))]
    fn segments_clear<'py>(
        &self,
        py: Python<'py>,
        starts: PyReadonlyArray2<'_, f64>,
        ends: PyReadonlyArray2<'_, f64>,
        tolerance: f64,
    ) -> PyResult<Flags<'py>> {
        let starts = points_of(&starts)?;
        let ends = points_of(&ends)?;
        same_lengths(&[starts.len(), ends.len()])?;
        Ok(BoolArray(
            self.inner
                .segments_clear(&starts, &ends, tolerance)
                .into_pyarray(py),
        ))
    }
}

/// How far each point lies inside `area`, one shape, negative outside it,
/// -inf for an empty area: the zone depth both `vias._depth` and the
/// containment field judge by.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn zone_depth<'py>(
    py: Python<'py>,
    area: PyRef<'_, Shapes>,
    points: PyReadonlyArray2<'_, f64>,
) -> PyResult<Floats<'py>> {
    let [shape] = area.shapes.as_slice() else {
        return Err(PyValueError::new_err("the area must be one shape"));
    };
    let points = points_of(&points)?;
    Ok(floats(
        py,
        points.iter().map(|&p| core_zone_depth(shape, p)).collect(),
    ))
}

pyo3_stub_gen::module_variable!("kibends._core", "LINE", usize, Kind::Line as usize);
pyo3_stub_gen::module_variable!("kibends._core", "AREA", usize, Kind::Area as usize);
pyo3_stub_gen::module_variable!("kibends._core", "OUTSIDE", usize, Kind::Outside as usize);
pyo3_stub_gen::module_variable!(
    "kibends._core",
    "INDEXED_EDGES",
    usize,
    kibends_core::field::INDEXED_EDGES
);

/// Registers the field's classes, functions and constants in `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Shapes>()?;
    m.add_class::<StaticEdges>()?;
    m.add_class::<Obstacles>()?;
    m.add_function(wrap_pyfunction!(zone_depth, m)?)?;
    m.add("LINE", Kind::Line as usize)?;
    m.add("AREA", Kind::Area as usize)?;
    m.add("OUTSIDE", Kind::Outside as usize)?;
    m.add("INDEXED_EDGES", kibends_core::field::INDEXED_EDGES)?;
    Ok(())
}
