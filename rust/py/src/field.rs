//! The static field as Python sees it: shapes built from shapely's plain
//! arrays, `Obstacles`, `StaticEdges` and `StaticItems`, and the searches
//! and pair helpers over numpy arrays that the Python side of the
//! optimizer calls until it moves here too.

use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::arrays::{BoolArg, BoolArray, IntpArg, IntpArray};
use kibends_core::field::edges::{Found, StaticDistances};
use kibends_core::field::obstacles::Item;
use kibends_core::field::{Kind, edges, obstacles, pairs, search};
use kibends_core::geometry::{PlainShapes, Shape, V2, zone_depth as core_zone_depth};

pub(crate) type Floats<'py> = Bound<'py, PyArray1<f64>>;
pub(crate) type Points<'py> = Bound<'py, PyArray2<f64>>;
pub(crate) type Indices<'py> = IntpArray<'py>;
pub(crate) type Flags<'py> = BoolArray<'py>;
pub(crate) type Complexes<'py> = Bound<'py, PyArray1<Complex64>>;

/// `StaticEdges.search`'s pairs as arrays, in `obstacles.Found`'s order.
type FoundArrays<'py> = (
    Indices<'py>,
    Complexes<'py>,
    Complexes<'py>,
    Floats<'py>,
    Floats<'py>,
    Indices<'py>,
    Indices<'py>,
    Flags<'py>,
    Flags<'py>,
);

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

pub(crate) fn complex_of(array: &PyReadonlyArray1<'_, Complex64>) -> Vec<V2> {
    array
        .as_array()
        .iter()
        .map(|c| V2::new(c.re, c.im))
        .collect()
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

pub(crate) fn complexes<'py>(py: Python<'py>, values: &[V2]) -> Complexes<'py> {
    values
        .iter()
        .map(|v| Complex64::new(v.x, v.y))
        .collect::<Vec<_>>()
        .into_pyarray(py)
}

fn found_arrays<'py>(py: Python<'py>, found: Found) -> FoundArrays<'py> {
    (
        indices(py, &found.point),
        complexes(py, &found.starts),
        complexes(py, &found.ends),
        floats(py, found.inflation),
        floats(py, found.limit),
        indices(py, &found.obstacle),
        indices(py, &found.item),
        BoolArray(found.inside.into_pyarray(py)),
        BoolArray(found.unsure.into_pyarray(py)),
    )
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
    shapes: Vec<Shape>,
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

/// A static field's obstacles as edges, with the search that finds the
/// edges near points (`kibends_core::field::edges`).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct StaticEdges {
    pub(crate) inner: Arc<edges::StaticEdges>,
}

#[gen_stub_pymethods]
#[pymethods]
impl StaticEdges {
    /// Edges from starts[i] to ends[i], (m, 2) in mm, each an item of its
    /// own, a `LINE`, inflated by inflations[i] and part of obstacle
    /// obstacles[i].
    #[staticmethod]
    fn from_segments(
        starts: PyReadonlyArray2<'_, f64>,
        ends: PyReadonlyArray2<'_, f64>,
        inflations: PyReadonlyArray1<'_, f64>,
        obstacles: IntpArg<'_>,
    ) -> PyResult<Self> {
        let starts = points_of(&starts)?;
        let ends = points_of(&ends)?;
        let inflations = floats_of(&inflations);
        let obstacles = indices_of(&obstacles, usize::MAX)?;
        same_lengths(&[starts.len(), ends.len(), inflations.len(), obstacles.len()])?;
        Ok(Self {
            inner: Arc::new(edges::StaticEdges::from_segments(
                &starts, &ends, inflations, obstacles,
            )),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Each item's obstacle.
    #[getter]
    fn obstacles<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.obstacles())
    }

    /// Each item's kind: `LINE`, `AREA` or `OUTSIDE`.
    #[getter]
    fn kinds<'py>(&self, py: Python<'py>) -> Indices<'py> {
        let kinds: Vec<usize> = self.inner.kinds().iter().map(|&k| k as usize).collect();
        indices(py, &kinds)
    }

    #[getter]
    fn inflations<'py>(&self, py: Python<'py>) -> Floats<'py> {
        floats(py, self.inner.inflations().to_vec())
    }

    /// How many obstacles the items make.
    #[getter]
    fn obstacle_count(&self) -> usize {
        self.inner.obstacle_count()
    }

    /// Each item's obstacle as a clearance check judges it, numbered below
    /// `obstacle_count`: its obstacle, or for an `Obstacles` field its
    /// connected copper of one net on one layer.
    #[getter]
    fn clearance_obstacles<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.clearance_obstacles())
    }

    /// The pairs that can answer the points `coords` within `reach` once
    /// each has moved at most `slack` mm, as `obstacles.Found`'s fields.
    fn search<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<'_, f64>,
        reach: f64,
        slack: f64,
    ) -> PyResult<FoundArrays<'py>> {
        let coords = points_of(&coords)?;
        Ok(found_arrays(py, self.inner.search(&coords, reach, slack)))
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

    /// Whether each point lies inside its item, items[i] of coords[i]: in
    /// an `AREA`'s area or outside an `OUTSIDE`'s, never for a `LINE`.
    fn inside<'py>(
        &self,
        py: Python<'py>,
        items: IntpArg<'_>,
        coords: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Flags<'py>> {
        let items = indices_of(&items, self.inner.len())?;
        let coords = points_of(&coords)?;
        same_lengths(&[items.len(), coords.len()])?;
        let inside: Vec<bool> = items
            .iter()
            .zip(&coords)
            .map(|(&k, &p)| self.inner.inside(k, p))
            .collect();
        Ok(BoolArray(inside.into_pyarray(py)))
    }

    /// The unit gradient at a point lying on the item's edges.
    fn normal(&self, item: usize, x: f64, y: f64) -> PyResult<(f64, f64)> {
        if item >= self.inner.len() {
            return Err(PyValueError::new_err(format!("no item {item}")));
        }
        let n = self.inner.normal(item, V2::new(x, y));
        Ok((n.x, n.y))
    }
}

/// The items of several `StaticEdges`, numbered one part after another.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "kibends._core")]
pub struct StaticItems {
    inner: search::StaticItems,
}

#[gen_stub_pymethods]
#[pymethods]
impl StaticItems {
    #[new]
    fn new(parts: Vec<PyRef<'_, StaticEdges>>) -> Self {
        Self {
            inner: search::StaticItems::new(parts.iter().map(|p| p.inner.clone()).collect()),
        }
    }

    /// Where each part's items start, and then their count.
    #[getter]
    fn starts<'py>(&self, py: Python<'py>) -> Indices<'py> {
        indices(py, self.inner.starts())
    }

    /// Whether each point lies inside its item, numbered across the parts.
    fn inside<'py>(
        &self,
        py: Python<'py>,
        items: IntpArg<'_>,
        coords: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Flags<'py>> {
        let items = indices_of(&items, usize::MAX)?;
        let coords = points_of(&coords)?;
        same_lengths(&[items.len(), coords.len()])?;
        let inside: Vec<bool> = items
            .iter()
            .zip(&coords)
            .map(|(&k, &p)| self.inner.inside(k, p))
            .collect();
        Ok(BoolArray(inside.into_pyarray(py)))
    }

    /// The unit gradient at a point lying on an item's edges, numbered
    /// across the parts.
    fn normal(&self, item: usize, x: f64, y: f64) -> (f64, f64) {
        let n = self.inner.normal(item, V2::new(x, y));
        (n.x, n.y)
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

/// `StaticEdges.search` over several parts, each with the number its items
/// start at, their obstacles numbered one part after another.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn search_parts<'py>(
    py: Python<'py>,
    parts: Vec<(PyRef<'_, StaticEdges>, usize)>,
    coords: PyReadonlyArray2<'_, f64>,
    reach: f64,
    slack: f64,
) -> PyResult<FoundArrays<'py>> {
    let coords = points_of(&coords)?;
    let numbered: Vec<(&edges::StaticEdges, usize)> = parts
        .iter()
        .map(|(p, first)| (p.inner.as_ref(), *first))
        .collect();
    Ok(found_arrays(
        py,
        search::search_parts(&numbered, &coords, reach, slack),
    ))
}

/// `StaticEdges.distances` over several parts, their obstacles numbered one
/// part after another.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn static_distances<'py>(
    py: Python<'py>,
    parts: Vec<PyRef<'_, StaticEdges>>,
    coords: PyReadonlyArray2<'_, f64>,
    below: f64,
) -> PyResult<(Indices<'py>, Indices<'py>, Floats<'py>)> {
    let coords = points_of(&coords)?;
    let parts: Vec<&edges::StaticEdges> = parts.iter().map(|p| p.inner.as_ref()).collect();
    Ok(distances_arrays(
        py,
        search::static_distances(&parts, &coords, below),
    ))
}

/// The smooth distance over the obstacles of `parts`, from one search.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn parts_smooth_distance<'py>(
    py: Python<'py>,
    parts: Vec<PyRef<'_, StaticEdges>>,
    points: PyReadonlyArray2<'_, f64>,
    reach: f64,
) -> PyResult<(Floats<'py>, Points<'py>)> {
    let points = points_of(&points)?;
    let parts: Vec<Arc<edges::StaticEdges>> = parts.iter().map(|p| p.inner.clone()).collect();
    let (distance, gradient) = search::parts_smooth_distance(&parts, &points, reach);
    Ok((floats(py, distance), vectors(py, &gradient)?))
}

/// The smooth distance at each point of `query` and its gradient over the
/// pairs of `obstacles.Found`'s fields, their items numbered as `items`
/// numbers them.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn static_smooth<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, Complex64>,
    point: IntpArg<'_>,
    starts: PyReadonlyArray1<'_, Complex64>,
    ends: PyReadonlyArray1<'_, Complex64>,
    inflation: PyReadonlyArray1<'_, f64>,
    limit: PyReadonlyArray1<'_, f64>,
    obstacle: IntpArg<'_>,
    item: IntpArg<'_>,
    inside: BoolArg<'_>,
    unsure: BoolArg<'_>,
    items: PyRef<'_, StaticItems>,
) -> PyResult<(Floats<'py>, Points<'py>)> {
    let query = complex_of(&query);
    let found = Found {
        point: indices_of(&point, query.len())?,
        starts: complex_of(&starts),
        ends: complex_of(&ends),
        inflation: floats_of(&inflation),
        limit: floats_of(&limit),
        obstacle: indices_of(&obstacle, usize::MAX)?,
        item: indices_of(&item, usize::MAX)?,
        inside: flags_of(&inside),
        unsure: flags_of(&unsure),
    };
    same_lengths(&[
        found.point.len(),
        found.starts.len(),
        found.ends.len(),
        found.inflation.len(),
        found.limit.len(),
        found.obstacle.len(),
        found.item.len(),
        found.inside.len(),
        found.unsure.len(),
    ])?;
    let (distance, gradient) = search::static_smooth(&query, &found, &items.inner);
    Ok((floats(py, distance), vectors(py, &gradient)?))
}

/// The smooth distance at each point of `query` over (point, segment)
/// pairs, each of an obstacle: distance, gradient, and per answering
/// obstacle the point, the pair, its share and its unit gradient.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn smooth_of_pairs<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, Complex64>,
    point: IntpArg<'_>,
    starts: PyReadonlyArray1<'_, Complex64>,
    along: PyReadonlyArray1<'_, Complex64>,
    inflations: PyReadonlyArray1<'_, f64>,
    limit: PyReadonlyArray1<'_, f64>,
    obstacle: IntpArg<'_>,
) -> PyResult<(
    Floats<'py>,
    Points<'py>,
    Indices<'py>,
    Indices<'py>,
    Floats<'py>,
    Points<'py>,
)> {
    let query = complex_of(&query);
    let point = indices_of(&point, query.len())?;
    let starts = complex_of(&starts);
    let along = complex_of(&along);
    let inflations = floats_of(&inflations);
    let limit = floats_of(&limit);
    let obstacle = indices_of(&obstacle, usize::MAX)?;
    same_lengths(&[
        point.len(),
        starts.len(),
        along.len(),
        inflations.len(),
        limit.len(),
        obstacle.len(),
    ])?;
    let found = pairs::smooth_of_pairs(
        &query,
        &point,
        &starts,
        &along,
        &inflations,
        &limit,
        &obstacle,
    );
    Ok((
        floats(py, found.distance),
        vectors(py, &found.gradient)?,
        indices(py, &found.point),
        indices(py, &found.edge),
        floats(py, found.share),
        vectors(py, &found.direction)?,
    ))
}

/// The nearest inflated segment at each point of `query` over (point,
/// segment) pairs: distance, gradient, and the pair that answers, -1 where
/// none does.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn nearest_of_pairs<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, Complex64>,
    point: IntpArg<'_>,
    starts: PyReadonlyArray1<'_, Complex64>,
    along: PyReadonlyArray1<'_, Complex64>,
    inflations: PyReadonlyArray1<'_, f64>,
    limit: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Floats<'py>, Points<'py>, Indices<'py>)> {
    let query = complex_of(&query);
    let point = indices_of(&point, query.len())?;
    let starts = complex_of(&starts);
    let along = complex_of(&along);
    let inflations = floats_of(&inflations);
    let limit = floats_of(&limit);
    same_lengths(&[
        point.len(),
        starts.len(),
        along.len(),
        inflations.len(),
        limit.len(),
    ])?;
    let found = pairs::nearest_of_pairs(&query, &point, &starts, &along, &inflations, &limit);
    // An index into a Rust slice fits an isize.
    let edge: Vec<isize> = found
        .edge
        .iter()
        .map(|e| e.map_or(-1, |k| k as isize))
        .collect();
    Ok((
        floats(py, found.distance),
        vectors(py, &found.gradient)?,
        IntpArray(edge.into_pyarray(py)),
    ))
}

/// Which (point, segment) pairs can answer their point within within[k]
/// once the distance from each point to each segment has changed by at most
/// `slack`, mm.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn candidate_pairs<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'_, Complex64>,
    point: IntpArg<'_>,
    starts: PyReadonlyArray1<'_, Complex64>,
    along: PyReadonlyArray1<'_, Complex64>,
    inflations: PyReadonlyArray1<'_, f64>,
    within: PyReadonlyArray1<'_, f64>,
    slack: f64,
) -> PyResult<Flags<'py>> {
    let query = complex_of(&query);
    let point = indices_of(&point, query.len())?;
    let starts = complex_of(&starts);
    let along = complex_of(&along);
    let inflations = floats_of(&inflations);
    let within = floats_of(&within);
    same_lengths(&[
        point.len(),
        starts.len(),
        along.len(),
        inflations.len(),
        within.len(),
    ])?;
    Ok(BoolArray(
        pairs::candidate_pairs(&query, &point, &starts, &along, &inflations, &within, slack)
            .into_pyarray(py),
    ))
}

/// Whether the segment from a[k] along ab[k] enters the edge from
/// starts[k] along along[k], inflated by inflations[k], by more than
/// `tolerance`.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn pairs_inside<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'_, Complex64>,
    ab: PyReadonlyArray1<'_, Complex64>,
    starts: PyReadonlyArray1<'_, Complex64>,
    along: PyReadonlyArray1<'_, Complex64>,
    inflations: PyReadonlyArray1<'_, f64>,
    tolerance: f64,
) -> PyResult<Flags<'py>> {
    let (a, ab, starts, along) = (
        complex_of(&a),
        complex_of(&ab),
        complex_of(&starts),
        complex_of(&along),
    );
    let inflations = floats_of(&inflations);
    same_lengths(&[
        a.len(),
        ab.len(),
        starts.len(),
        along.len(),
        inflations.len(),
    ])?;
    Ok(BoolArray(
        pairs::pairs_inside(&a, &ab, &starts, &along, &inflations, tolerance).into_pyarray(py),
    ))
}

/// The gap between each segment and edge of `pairs_inside` less the
/// edge's inflation.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn pair_depths<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'_, Complex64>,
    ab: PyReadonlyArray1<'_, Complex64>,
    starts: PyReadonlyArray1<'_, Complex64>,
    along: PyReadonlyArray1<'_, Complex64>,
    inflations: PyReadonlyArray1<'_, f64>,
) -> PyResult<Floats<'py>> {
    let (a, ab, starts, along) = (
        complex_of(&a),
        complex_of(&ab),
        complex_of(&starts),
        complex_of(&along),
    );
    let inflations = floats_of(&inflations);
    same_lengths(&[
        a.len(),
        ab.len(),
        starts.len(),
        along.len(),
        inflations.len(),
    ])?;
    Ok(floats(
        py,
        pairs::pair_depths(&a, &ab, &starts, &along, &inflations),
    ))
}

/// The order of (point, segment) pairs by point, then obstacle, then
/// tiebreak.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn obstacle_order<'py>(
    py: Python<'py>,
    point: IntpArg<'_>,
    obstacle: IntpArg<'_>,
    tiebreak: IntpArg<'_>,
) -> PyResult<Indices<'py>> {
    let point = indices_of(&point, usize::MAX)?;
    let obstacle = indices_of(&obstacle, usize::MAX)?;
    let tiebreak = indices_of(&tiebreak, usize::MAX)?;
    same_lengths(&[point.len(), obstacle.len(), tiebreak.len()])?;
    Ok(indices(
        py,
        &pairs::obstacle_order(&point, &obstacle, &tiebreak),
    ))
}

/// (point, segment) index pairs, by point then segment, among them every
/// pair whose segment from starts[k] to ends[k] comes within `radius` of
/// points[i].
#[gen_stub_pyfunction]
#[pyfunction]
pub fn near_segments<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray2<'_, f64>,
    ends: PyReadonlyArray2<'_, f64>,
    points: PyReadonlyArray2<'_, f64>,
    radius: f64,
) -> PyResult<(Indices<'py>, Indices<'py>)> {
    let starts = points_of(&starts)?;
    let ends = points_of(&ends)?;
    let points = points_of(&points)?;
    same_lengths(&[starts.len(), ends.len()])?;
    let (point, segment) = pairs::near_segments(&starts, &ends, &points, radius);
    Ok((indices(py, &point), indices(py, &segment)))
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
    m.add_class::<StaticItems>()?;
    m.add_class::<Obstacles>()?;
    m.add_function(wrap_pyfunction!(search_parts, m)?)?;
    m.add_function(wrap_pyfunction!(static_distances, m)?)?;
    m.add_function(wrap_pyfunction!(parts_smooth_distance, m)?)?;
    m.add_function(wrap_pyfunction!(static_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_of_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_of_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(pairs_inside, m)?)?;
    m.add_function(wrap_pyfunction!(pair_depths, m)?)?;
    m.add_function(wrap_pyfunction!(obstacle_order, m)?)?;
    m.add_function(wrap_pyfunction!(near_segments, m)?)?;
    m.add_function(wrap_pyfunction!(zone_depth, m)?)?;
    m.add("LINE", Kind::Line as usize)?;
    m.add("AREA", Kind::Area as usize)?;
    m.add("OUTSIDE", Kind::Outside as usize)?;
    m.add("INDEXED_EDGES", kibends_core::field::INDEXED_EDGES)?;
    Ok(())
}
