//! A static field's obstacles as edges, with the search that finds the
//! edges near points.
//!
//! Each item is a geometry of a kind (`Kind`), with its inflation and its
//! obstacle. It is looked for as far beyond the reach as the largest
//! inflation of the items but the `Outside` ones, or for the outside of an
//! area as far as its own. Its edges are its geometry's, or its
//! boundary's, and an item's are numbered together, items in order.
//!
//! A tree over the items' bounding boxes finds those near a point; the
//! `Outside` items are few and asked at every point. Whether a point is
//! inside an area is tested exactly, and only where the point is inside
//! the area's bounding box. Of an item with more than INDEXED_EDGES edges,
//! a tree over the middles of pieces of its edges finds those near a point
//! (`StaticEdges::item_edges_near`).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;

use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};

use super::pairs::PointTree;
use super::smooth::SMOOTH_REACH;
use super::union_find::UnionFind;
use super::{DEGENERATE_DISTANCE, INDEXED_EDGES, Kind, MAX_EDGE_LENGTH, TOUCH_TOLERANCE};
use crate::geometry::{Shape, V2, closest_point, offset, outward_normal, segment_distance, segment_to_segment};

/// (point, edge) pairs of static fields that can answer points, as
/// `search::static_smooth` takes them: by point, then obstacle, then edge,
/// so that the pairs of one item and those of one obstacle at a point lie
/// together. Each pair's edge, where it starts and ends, its inflation,
/// how far from the point it counts unless the point is inside its item,
/// its obstacle and item; whether the point was inside the item when
/// found, and whether it may have crossed the item's boundary since, so
/// that it is tested again.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Found {
    pub point: Vec<usize>,
    pub starts: Vec<V2>,
    pub ends: Vec<V2>,
    pub inflation: Vec<f64>,
    pub limit: Vec<f64>,
    pub obstacle: Vec<usize>,
    pub item: Vec<usize>,
    pub inside: Vec<bool>,
    pub unsure: Vec<bool>,
}

impl Found {
    pub fn len(&self) -> usize {
        self.point.len()
    }

    pub fn is_empty(&self) -> bool {
        self.point.is_empty()
    }

    /// The pairs `order` indexes, in that order.
    pub fn taken(&self, order: &[usize]) -> Found {
        Found {
            point: order.iter().map(|&k| self.point[k]).collect(),
            starts: order.iter().map(|&k| self.starts[k]).collect(),
            ends: order.iter().map(|&k| self.ends[k]).collect(),
            inflation: order.iter().map(|&k| self.inflation[k]).collect(),
            limit: order.iter().map(|&k| self.limit[k]).collect(),
            obstacle: order.iter().map(|&k| self.obstacle[k]).collect(),
            item: order.iter().map(|&k| self.item[k]).collect(),
            inside: order.iter().map(|&k| self.inside[k]).collect(),
            unsure: order.iter().map(|&k| self.unsure[k]).collect(),
        }
    }

    /// `self` then `other`, its points, obstacles and items shifted.
    pub fn extend(&mut self, other: Found, point: usize, obstacle: usize, item: usize) {
        self.point.extend(other.point.iter().map(|p| p + point));
        self.starts.extend(other.starts);
        self.ends.extend(other.ends);
        self.inflation.extend(other.inflation);
        self.limit.extend(other.limit);
        self.obstacle.extend(other.obstacle.iter().map(|o| o + obstacle));
        self.item.extend(other.item.iter().map(|i| i + item));
        self.inside.extend(other.inside);
        self.unsure.extend(other.unsure);
    }
}

/// (point, obstacle) pairs of a static field, by point then obstacle, and
/// each point's exact signed distance to its obstacle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticDistances {
    pub point: Vec<usize>,
    pub obstacle: Vec<usize>,
    pub distance: Vec<f64>,
}

/// What a clearance check joins of the items: copper of one net on one
/// layer that touches is one obstacle to it. Nets and layers are numbered,
/// None for an item that is not copper of one net on one layer.
#[derive(Clone, Debug, Default)]
pub struct Copper {
    pub net: Vec<Option<usize>>,
    pub layer: Vec<Option<usize>>,
    /// How far each item's copper reaches beyond its geometry, mm.
    pub extent: Vec<f64>,
}

/// An item's edges split into pieces of at most MAX_EDGE_LENGTH: a tree
/// over their middles, each piece's edge, numbered from the item's first,
/// and half the longest piece.
struct Pieces {
    tree: PointTree,
    edge_of: Vec<usize>,
    half_piece: f64,
}

type Boxed = GeomWithData<Rectangle<[f64; 2]>, usize>;

pub struct StaticEdges {
    shapes: Vec<Shape>,
    kinds: Vec<Kind>,
    inflations: Vec<f64>,
    obstacles: Vec<usize>,
    obstacle_count: usize,
    /// The largest inflation of the items but the `Outside` ones.
    inflation: f64,
    tree: RTree<Boxed>,
    outside: Vec<usize>,
    first: Vec<usize>,
    count: Vec<usize>,
    starts: Vec<V2>,
    ends: Vec<V2>,
    bounds: Vec<Option<[f64; 4]>>,
    pieces: Vec<OnceLock<Pieces>>,
    indexed_edges: usize,
    copper: Option<Copper>,
    clearance: OnceLock<Vec<usize>>,
}

impl StaticEdges {
    /// Items of `shapes`, each of its kind, inflation and obstacle; with
    /// `copper`, a clearance check joins touching copper of one net on one
    /// layer (`clearance_obstacles`).
    pub fn new(
        shapes: Vec<Shape>,
        kinds: Vec<Kind>,
        inflations: Vec<f64>,
        obstacles: Vec<usize>,
        copper: Option<Copper>,
    ) -> Self {
        let mut first = Vec::with_capacity(shapes.len());
        let mut count = Vec::with_capacity(shapes.len());
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for shape in &shapes {
            let edges = shape.edges();
            first.push(starts.len());
            count.push(edges.len());
            for (a, b) in edges {
                starts.push(a);
                ends.push(b);
            }
        }
        let bounds: Vec<Option<[f64; 4]>> = shapes.iter().map(Shape::bounds).collect();
        let boxed = (0..shapes.len())
            .filter(|&k| kinds[k] != Kind::Outside)
            .filter_map(|k| {
                bounds[k].map(|[x0, y0, x1, y1]| {
                    GeomWithData::new(Rectangle::from_corners([x0, y0], [x1, y1]), k)
                })
            })
            .collect();
        let inflation = (0..shapes.len())
            .filter(|&k| kinds[k] != Kind::Outside)
            .map(|k| inflations[k])
            .fold(0.0, f64::max);
        Self {
            outside: (0..shapes.len()).filter(|&k| kinds[k] == Kind::Outside).collect(),
            obstacle_count: obstacles.iter().max().map_or(0, |&m| m + 1),
            pieces: (0..shapes.len()).map(|_| OnceLock::new()).collect(),
            tree: RTree::bulk_load(boxed),
            shapes,
            kinds,
            inflations,
            obstacles,
            inflation,
            first,
            count,
            starts,
            ends,
            bounds,
            indexed_edges: INDEXED_EDGES,
            copper,
            clearance: OnceLock::new(),
        }
    }

    /// Edges from starts[i] to ends[i], each an item of its own inflated by
    /// inflations[i] and part of obstacle obstacles[i].
    pub fn from_segments(starts: &[V2], ends: &[V2], inflations: Vec<f64>, obstacles: Vec<usize>) -> Self {
        let shapes = starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| Shape::line(vec![a, b]))
            .collect::<Vec<_>>();
        let kinds = vec![Kind::Line; shapes.len()];
        Self::new(shapes, kinds, inflations, obstacles, None)
    }

    /// The same field, with items of more than `edges` edges indexed.
    pub fn with_indexed_edges(mut self, edges: usize) -> Self {
        self.indexed_edges = edges;
        self
    }

    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }

    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    pub fn inflations(&self) -> &[f64] {
        &self.inflations
    }

    pub fn obstacles(&self) -> &[usize] {
        &self.obstacles
    }

    pub fn obstacle_count(&self) -> usize {
        self.obstacle_count
    }

    /// Each item's obstacle as a clearance check judges it, numbered below
    /// `obstacle_count`: its obstacle, or with copper given, the obstacles
    /// joined where copper of one net overlaps on one layer, as at a
    /// track's width change or a T-junction, each numbered as its lowest
    /// obstacle. Connected copper is one piece of copper, whatever the
    /// energy splits it into; copper on two layers is two, even where a via
    /// joins them, since a via crossing from one to the other's clearance
    /// enters new copper. Taken only when a moving via's clearance asks.
    pub fn clearance_obstacles(&self) -> &[usize] {
        self.clearance.get_or_init(|| match &self.copper {
            None => self.obstacles.clone(),
            Some(copper) => self.joined_copper(copper),
        })
    }

    fn joined_copper(&self, copper: &Copper) -> Vec<usize> {
        let is_copper = |k: usize| copper.net[k].is_some() && copper.layer[k].is_some();
        let widest = (0..self.len())
            .filter(|&k| is_copper(k))
            .map(|k| copper.extent[k])
            .fold(0.0, f64::max);
        let mut sets = UnionFind::new(self.obstacle_count);
        for a in (0..self.len()).filter(|&k| is_copper(k)) {
            let Some([x0, y0, x1, y1]) = self.bounds[a] else {
                continue;
            };
            // Two pieces overlap within both their extents; the box reaches
            // as far as the widest piece could.
            let r = copper.extent[a] + widest + TOUCH_TOLERANCE;
            let envelope = AABB::from_corners([x0 - r, y0 - r], [x1 + r, y1 + r]);
            for found in self.tree.locate_in_envelope_intersecting(envelope) {
                let b = found.data;
                if b <= a
                    || !is_copper(b)
                    || copper.net[a] != copper.net[b]
                    || copper.layer[a] != copper.layer[b]
                {
                    continue;
                }
                let gap = self.shapes[a].shape_distance(&self.shapes[b]);
                if gap <= copper.extent[a] + copper.extent[b] + TOUCH_TOLERANCE {
                    sets.join(self.obstacles[a], self.obstacles[b]);
                }
            }
        }
        let mut lowest: HashMap<usize, usize> = HashMap::new();
        for o in 0..self.obstacle_count {
            let root = sets.root(o);
            lowest.entry(root).or_insert(o);
        }
        self.obstacles
            .iter()
            .map(|&o| {
                let root = sets.root(o);
                lowest.get(&root).copied().unwrap_or(o)
            })
            .collect()
    }

    fn edge_range(&self, item: usize) -> Range<usize> {
        self.first[item]..self.first[item] + self.count[item]
    }

    /// The edge of `edge`, numbered across the items.
    pub fn edge(&self, edge: usize) -> (V2, V2) {
        (self.starts[edge], self.ends[edge])
    }

    fn pieces(&self, item: usize) -> &Pieces {
        self.pieces[item].get_or_init(|| {
            let mut middles = Vec::new();
            let mut edge_of = Vec::new();
            let mut longest: f64 = 0.0;
            for (local, e) in self.edge_range(item).enumerate() {
                let a = self.starts[e];
                let along = self.ends[e] - a;
                let length = along.hypot();
                let parts = (length / MAX_EDGE_LENGTH).ceil().max(1.0);
                longest = longest.max(length / parts);
                // An f64 count of pieces of a finite edge is a small whole
                // number.
                let pieces = parts as usize;
                for part in 0..pieces {
                    middles.push(a + ((part as f64 + 0.5) / parts) * along);
                    edge_of.push(local);
                }
            }
            Pieces {
                tree: PointTree::new(&middles),
                edge_of,
                half_piece: longest / 2.0,
            }
        })
    }

    /// The edges of `item`, numbered across the items and ascending, among
    /// them every edge within 2·slack of the nearest to `p`: all of them,
    /// or of an item of more than INDEXED_EDGES edges those of the pieces
    /// whose middles lie within the nearest piece's middle's distance,
    /// 2·slack and half a piece. The nearest piece's middle is no nearer
    /// than the nearest edge, and every edge has a piece whose middle is
    /// within half the longest piece of the edge's nearest point.
    pub fn item_edges_near(&self, item: usize, p: V2, slack: f64) -> Vec<usize> {
        let range = self.edge_range(item);
        if range.len() <= self.indexed_edges {
            return range.collect();
        }
        let pieces = self.pieces(item);
        let Some((_, bound)) = pieces.tree.nearest(p) else {
            return Vec::new();
        };
        // Round-off in the distances, far below the nanometre allowed.
        let radius = bound + 2.0 * slack + pieces.half_piece + 3.0 * DEGENERATE_DISTANCE;
        let mut edges: Vec<usize> = pieces
            .tree
            .within(p, radius)
            .into_iter()
            .map(|piece| range.start + pieces.edge_of[piece])
            .collect();
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// The edges of `item`, ascending, among them every one within
    /// `radius` of the segment from `a` to `b`.
    fn item_edges_near_segment(&self, item: usize, a: V2, b: V2, radius: f64) -> Vec<usize> {
        let range = self.edge_range(item);
        if range.len() <= self.indexed_edges {
            return range.collect();
        }
        let pieces = self.pieces(item);
        let r = radius + pieces.half_piece + DEGENERATE_DISTANCE;
        // Every point of the segment lies within half its length of its
        // middle.
        let middle = 0.5 * (a + b);
        let half = 0.5 * (b - a).norm();
        let mut edges: Vec<usize> = pieces
            .tree
            .within(middle, r + half)
            .into_iter()
            .map(|piece| range.start + pieces.edge_of[piece])
            .collect();
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// Whether `p` lies inside `item`: in an `Area`'s area or outside an
    /// `Outside`'s, never for a `Line`.
    pub fn inside(&self, item: usize, p: V2) -> bool {
        let kind = self.kinds[item];
        if kind == Kind::Line {
            return false;
        }
        let boxed = self.bounds[item]
            .is_some_and(|[x0, y0, x1, y1]| p.x >= x0 && p.y >= y0 && p.x <= x1 && p.y <= y1);
        let within = boxed && self.shapes[item].contains(p);
        if kind == Kind::Outside { !within } else { within }
    }

    /// The unit gradient at a point lying on the item's edges, where the
    /// direction to them is noise.
    pub fn normal(&self, item: usize, p: V2) -> V2 {
        let kind = self.kinds[item];
        let n = outward_normal(&self.shapes[item], p, kind != Kind::Line);
        if kind == Kind::Outside { -n } else { n }
    }

    /// The distance from `p` to the item's geometry, zero inside an area.
    fn item_distance(&self, item: usize, p: V2) -> f64 {
        if self.kinds[item] == Kind::Area && self.shapes[item].covers(p) {
            return 0.0;
        }
        self.item_edges_near(item, p, 0.0)
            .into_iter()
            .map(|e| segment_distance(p, self.starts[e], self.ends[e]))
            .fold(f64::INFINITY, f64::min)
    }

    /// The item of the tree nearest `p` by its geometry, and that distance.
    pub fn nearest_item(&self, p: V2) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (found, d2) in self.tree.nearest_neighbor_iter_with_distance_2(p.array()) {
            if let Some((_, least)) = best
                && d2 > least * least
            {
                break;
            }
            let d = self.item_distance(found.data, p);
            if best.is_none_or(|(_, least)| d < least) {
                best = Some((found.data, d));
            }
        }
        best
    }

    /// The items of the tree whose bounding boxes lie within `radius` of
    /// `p`, ascending.
    pub fn items_within(&self, p: V2, radius: f64) -> Vec<usize> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        let mut found: Vec<usize> = self
            .tree
            .locate_within_distance(p.array(), radius * radius)
            .map(|g| g.data)
            .collect();
        found.sort_unstable();
        found
    }

    /// The items of the tree whose bounding boxes come within `radius` of
    /// the segment from `a` to `b`'s bounding box, ascending.
    pub fn items_near_segment(&self, a: V2, b: V2, radius: f64) -> Vec<usize> {
        let envelope = AABB::from_corners(
            [a.x.min(b.x) - radius, a.y.min(b.y) - radius],
            [a.x.max(b.x) + radius, a.y.max(b.y) + radius],
        );
        let mut found: Vec<usize> = self
            .tree
            .locate_in_envelope_intersecting(envelope)
            .map(|g| g.data)
            .collect();
        found.sort_unstable();
        found
    }

    /// The items that can come within `reach` of `p` once inflated and once
    /// the point has moved `slack`, or without a finite reach within
    /// SMOOTH_REACH of its nearest; and every `Outside` item.
    fn near(&self, p: V2, reach: f64, slack: f64) -> Vec<usize> {
        let radius = if reach.is_finite() {
            reach + self.inflation + slack
        } else {
            // An item farther than the nearest can still be nearer once
            // inflated, by at most the spread of the inflations, once the
            // nearest has come `slack` nearer and another gone `slack`
            // farther.
            self.nearest_item(p).map_or(f64::NEG_INFINITY, |(item, d)| {
                d - self.inflations[item] + self.inflation + SMOOTH_REACH + 2.0 * slack
            })
        };
        let mut items = self.items_within(p, radius + DEGENERATE_DISTANCE);
        items.extend(&self.outside);
        items
    }

    /// The pairs that can answer the points `coords` within `reach` once
    /// each has moved at most `slack` mm: of an item's edges those that can
    /// be its nearest, of the items those that can come within
    /// SMOOTH_REACH of the point's nearest, as `search::static_smooth`
    /// measures them. Without a finite reach, those within SMOOTH_REACH of
    /// the nearest.
    pub fn search(&self, coords: &[V2], reach: f64, slack: f64) -> Found {
        let mut found = Found::default();
        for (i, &p) in coords.iter().enumerate() {
            self.search_point(&mut found, i, p, reach, slack);
        }
        found
    }

    fn search_point(&self, found: &mut Found, i: usize, p: V2, reach: f64, slack: f64) {
        struct Near {
            item: usize,
            inside: bool,
            unsure: bool,
            nearest: f64,
            distance: f64,
            limit: f64,
            edges: Vec<(usize, f64)>,
        }
        let mut near: Vec<Near> = Vec::new();
        let mut bound = f64::INFINITY;
        for item in self.near(p, reach, slack) {
            if self.count[item] == 0 {
                continue;
            }
            let kind = self.kinds[item];
            let inside = self.inside(item, p);
            let edges: Vec<(usize, f64)> = self
                .item_edges_near(item, p, slack)
                .into_iter()
                .map(|e| (e, offset(p, self.starts[e], self.ends[e] - self.starts[e]).hypot()))
                .collect();
            let nearest = edges.iter().map(|&(_, raw)| raw).fold(f64::INFINITY, f64::min);
            let limit = reach
                + if kind == Kind::Outside {
                    self.inflations[item]
                } else {
                    self.inflation
                };
            let distance = if inside { -nearest } else { nearest } - self.inflations[item];
            let unsure = slack > 0.0 && kind != Kind::Line && nearest <= slack + DEGENERATE_DISTANCE;
            // Distance changes no faster than the point moves, so an item
            // that stays in reach bounds how far the point's nearest can lie.
            if (inside && !unsure) || nearest + slack <= limit {
                bound = bound.min(distance + slack);
            }
            near.push(Near {
                item,
                inside,
                unsure,
                nearest,
                distance,
                limit,
                edges,
            });
        }
        let tolerance = 2.0 * DEGENERATE_DISTANCE;
        let mut kept: Vec<(usize, usize, usize)> = Vec::new();
        for (n, at) in near.iter().enumerate() {
            let reachable =
                at.inside || at.unsure || at.nearest - slack <= at.limit + DEGENERATE_DISTANCE;
            if !reachable || at.distance - slack > bound + SMOOTH_REACH + tolerance {
                continue;
            }
            for &(edge, raw) in &at.edges {
                if raw <= at.nearest + 2.0 * slack + tolerance {
                    kept.push((self.obstacles[at.item], edge, n));
                }
            }
        }
        kept.sort_unstable();
        for (obstacle, edge, n) in kept {
            let at = &near[n];
            found.point.push(i);
            found.starts.push(self.starts[edge]);
            found.ends.push(self.ends[edge]);
            found.inflation.push(self.inflations[at.item]);
            found.limit.push(at.limit);
            found.obstacle.push(obstacle);
            found.item.push(at.item);
            found.inside.push(at.inside);
            found.unsure.push(at.unsure);
        }
    }

    /// Each point's exact signed distance to each obstacle nearer than
    /// `below`, as a clearance check judges obstacles
    /// (`clearance_obstacles`), measured as `search::static_smooth`
    /// measures: its nearest item's nearest edge. Without a finite `below`,
    /// to at least each obstacle within SMOOTH_REACH of the point's
    /// nearest.
    pub fn distances(&self, coords: &[V2], below: f64) -> StaticDistances {
        let clearance = self.clearance_obstacles();
        let mut out = StaticDistances::default();
        for (i, &p) in coords.iter().enumerate() {
            let mut measured: Vec<(usize, f64)> = Vec::new();
            for item in self.near(p, below.max(0.0), 0.0) {
                if self.count[item] == 0 {
                    continue;
                }
                let (length, _) = self.nearest_offset(item, p);
                let inside = self.inside(item, p);
                let distance = if inside { -length } else { length } - self.inflations[item];
                measured.push((clearance[item], distance));
            }
            measured.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            let mut last = None;
            for (obstacle, distance) in measured {
                if last == Some(obstacle) {
                    continue;
                }
                last = Some(obstacle);
                if distance < below {
                    out.point.push(i);
                    out.obstacle.push(obstacle);
                    out.distance.push(distance);
                }
            }
        }
        out
    }

    /// The length of the offset to `p` from the item's nearest point, as
    /// GEOS measures it from the first of its nearest edges, and that
    /// offset.
    pub fn nearest_offset(&self, item: usize, p: V2) -> (f64, V2) {
        let mut best: Option<(usize, f64)> = None;
        for e in self.item_edges_near(item, p, 0.0) {
            let d = segment_distance(p, self.starts[e], self.ends[e]);
            if best.is_none_or(|(_, least)| d < least) {
                best = Some((e, d));
            }
        }
        let Some((e, _)) = best else {
            return (f64::INFINITY, V2::default());
        };
        let toward = p - closest_point(p, self.starts[e], self.ends[e]);
        (toward.norm(), toward)
    }

    /// The least distance from the segment from `a` to `b` to the item's
    /// edges, considering only edges that can be nearer than `below`;
    /// infinite if none is.
    pub fn segment_edge_distance(&self, item: usize, a: V2, b: V2, below: f64) -> f64 {
        self.item_edges_near_segment(item, a, b, below)
            .into_iter()
            .map(|e| segment_to_segment(a, b, self.starts[e], self.ends[e]))
            .fold(f64::INFINITY, f64::min)
    }

    /// The item's edges that the segment from `a` to `b` may touch.
    pub fn edges_touching_segment(&self, item: usize, a: V2, b: V2) -> Vec<(V2, V2)> {
        self.item_edges_near_segment(item, a, b, DEGENERATE_DISTANCE)
            .into_iter()
            .map(|e| (self.starts[e], self.ends[e]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(centre: V2, radius: f64, edges: usize) -> Shape {
        let ring = (0..=edges)
            .map(|k| {
                let angle = std::f64::consts::TAU * (k % edges) as f64 / edges as f64;
                V2::new(centre.x + radius * angle.cos(), centre.y + radius * angle.sin())
            })
            .collect();
        Shape::polygon(vec![ring])
    }

    fn scattered(n: usize, seed: u64) -> Vec<V2> {
        // A small linear congruential generator: the points only need to
        // be spread, not random.
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| V2::new(-1.5 + 9.0 * next(), -1.5 + 9.0 * next())).collect()
    }

    fn scene(indexed: usize) -> StaticEdges {
        let mut outline = disc(V2::new(3.0, 2.0), 6.0, 200);
        outline.type_id = 3;
        let shapes = vec![
            disc(V2::new(1.0, 1.0), 0.8, 300),
            disc(V2::new(4.0, 1.0), 1.5, 64),
            Shape::line(vec![V2::new(0.0, 3.0), V2::new(6.0, 3.5)]),
            Shape::line(vec![V2::new(5.0, 5.0)]),
            outline,
        ];
        let kinds = vec![Kind::Area, Kind::Area, Kind::Line, Kind::Line, Kind::Outside];
        StaticEdges::new(shapes, kinds, vec![0.3, 0.2, 0.35, 0.4, 0.3], vec![0, 1, 2, 3, 4], None)
            .with_indexed_edges(indexed)
    }

    #[test]
    fn an_indexed_search_finds_what_measuring_every_edge_finds() {
        let points = scattered(300, 3);
        for reach in [0.3, f64::INFINITY] {
            for slack in [0.0, 0.05] {
                let every = scene(usize::MAX).search(&points, reach, slack);
                let indexed = scene(0).search(&points, reach, slack);
                assert_eq!(every, indexed);
            }
        }
    }

    #[test]
    fn a_many_edged_item_measures_only_edges_near_each_point() {
        let field = scene(INDEXED_EDGES);
        let p = V2::new(1.0, 1.9);
        let near = field.item_edges_near(0, p, 0.05);
        assert!(!near.is_empty());
        assert!(near.len() * 10 < field.count[0]);
    }

    #[test]
    fn distances_are_found_through_the_index_as_without_it() {
        let points = scattered(200, 7);
        for below in [0.3, f64::INFINITY] {
            assert_eq!(
                scene(usize::MAX).distances(&points, below),
                scene(0).distances(&points, below)
            );
        }
    }
}
