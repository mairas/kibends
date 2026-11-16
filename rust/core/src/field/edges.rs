//! A static field's obstacles as edges, with the search that finds the
//! edges near points.
//!
//! A field's items are members of item sets (`items::ItemSet`) that
//! several fields share: one set, or several, as a moving via's field
//! spans the sets of its layers. The field holds which of the sets' items
//! it has, in its own order, and each one's inflation and obstacle; a set
//! holds the rest. An item is a geometry of a kind (`Kind`). It is looked
//! for as far beyond the reach as the largest inflation of the field's
//! items but the `Outside` ones, or for the outside of an area as far as
//! its own. Its edges are its geometry's, or its boundary's.
//!
//! Each set's tree over its items' bounding boxes finds those near a
//! point, and the field keeps those that are its own, in its order; the
//! `Outside` items are few and asked at every point. The field's items
//! answer in its order wherever their sets give them, so that it answers
//! exactly as a field of the same items in one set would. Whether a point
//! is inside an area is tested exactly, and only where the point is inside
//! the area's bounding box. Of an item with more than INDEXED_EDGES edges,
//! a tree over the middles of pieces of its edges finds those near a point
//! (`StaticEdges::item_edges_near`).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::items::{Copper, ItemSet, Picked, Source};
use super::smooth::SMOOTH_REACH;
use super::union_find::UnionFind;
use super::{DEGENERATE_DISTANCE, INDEXED_EDGES, Kind, TOUCH_TOLERANCE};
use crate::geometry::{
    Shape, V2, closest_point, offset, outward_normal, segment_distance, segment_to_segment,
};

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
        self.obstacle
            .extend(other.obstacle.iter().map(|o| o + obstacle));
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

/// A field's item: the set it lies in, by its place among the field's
/// sets, and its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub part: u32,
    pub index: u32,
}

/// How a field takes an item otherwise than its set holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Override {
    /// Copper counted on the layer of this number rather than its set's.
    Layer(u32),
    /// A pad's copper held as no copper, as what only stops a via.
    Stop,
}

/// A set's item that is not the field's.
const ABSENT: u32 = u32::MAX;

/// A static field's items, each a member of one of its sets, with the
/// field's own inflation and obstacle of each.
pub struct StaticEdges {
    parts: Vec<Arc<ItemSet>>,
    members: Vec<Member>,
    /// Each part's items' places among the field's, ABSENT for the items
    /// not there; None for a field of every item of one set, in order.
    lookup: Option<Vec<Vec<u32>>>,
    /// Whether a part's items found in its tree, taken in the order it
    /// gives them, are the field's in its order.
    ordered: bool,
    inflations: Vec<f64>,
    obstacles: Vec<u32>,
    obstacle_count: usize,
    /// The largest inflation of the items but the `Outside` ones.
    inflation: f64,
    outside: Vec<usize>,
    /// The field's items it takes otherwise than their sets hold them,
    /// ascending.
    overrides: Vec<(u32, Override)>,
    indexed_edges: usize,
    clearance: OnceLock<Vec<u32>>,
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
        let set = Arc::new(ItemSet::new(shapes, kinds, copper, None));
        let obstacles = obstacles.into_iter().map(number).collect();
        Self::whole(set, inflations, obstacles)
    }

    /// Every item of `set`, in order, each of its inflation and obstacle.
    pub fn whole(set: Arc<ItemSet>, inflations: Vec<f64>, obstacles: Vec<u32>) -> Self {
        let members = (0..set.len())
            .map(|k| Member {
                part: 0,
                index: number(k),
            })
            .collect();
        Self::build(vec![set], members, None, inflations, obstacles, Vec::new())
    }

    /// The items `members` of `parts`, in that order, each of its
    /// inflation and obstacle, the items `overrides` names taken as it
    /// says, ascending. The parts are sets of one interner, and no item is
    /// a member twice.
    pub fn view(
        parts: Vec<Arc<ItemSet>>,
        members: Vec<Member>,
        inflations: Vec<f64>,
        obstacles: Vec<u32>,
        overrides: Vec<(u32, Override)>,
    ) -> Self {
        assert!(
            parts.windows(2).all(|w| w[0].token() == w[1].token()),
            "a field's sets are of one interner"
        );
        let mut lookup: Vec<Vec<u32>> = parts.iter().map(|p| vec![ABSENT; p.len()]).collect();
        for (k, m) in members.iter().enumerate() {
            let slot = &mut lookup[m.part as usize][m.index as usize];
            assert_eq!(*slot, ABSENT, "an item is a member of a field once");
            *slot = number(k);
        }
        assert!(
            overrides.windows(2).all(|w| w[0].0 < w[1].0),
            "overrides ascend by item"
        );
        Self::build(
            parts,
            members,
            Some(lookup),
            inflations,
            obstacles,
            overrides,
        )
    }

    fn build(
        parts: Vec<Arc<ItemSet>>,
        members: Vec<Member>,
        lookup: Option<Vec<Vec<u32>>>,
        inflations: Vec<f64>,
        obstacles: Vec<u32>,
        overrides: Vec<(u32, Override)>,
    ) -> Self {
        let kind = |m: &Member| parts[m.part as usize].kind(m.index as usize);
        let inflation = members
            .iter()
            .zip(&inflations)
            .filter(|(m, _)| kind(m) != Kind::Outside)
            .map(|(_, &i)| i)
            .fold(0.0, f64::max);
        let outside = (0..members.len())
            .filter(|&k| kind(&members[k]) == Kind::Outside)
            .collect();
        let ordered = parts.len() == 1 && members.windows(2).all(|w| w[0].index < w[1].index);
        Self {
            obstacle_count: obstacles.iter().max().map_or(0, |&m| m as usize + 1),
            parts,
            members,
            lookup,
            ordered,
            inflations,
            obstacles,
            inflation,
            outside,
            overrides,
            indexed_edges: INDEXED_EDGES,
            clearance: OnceLock::new(),
        }
    }

    /// Edges from starts[i] to ends[i], each an item of its own inflated by
    /// inflations[i] and part of obstacle obstacles[i].
    pub fn from_segments(
        starts: &[V2],
        ends: &[V2],
        inflations: Vec<f64>,
        obstacles: Vec<usize>,
    ) -> Self {
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
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The set of `item` and its index there.
    fn at(&self, item: usize) -> (&ItemSet, usize) {
        let m = self.members[item];
        (&self.parts[m.part as usize], m.index as usize)
    }

    pub fn shape(&self, item: usize) -> &Shape {
        let (set, k) = self.at(item);
        set.shape(k)
    }

    pub fn kind(&self, item: usize) -> Kind {
        let (set, k) = self.at(item);
        set.kind(k)
    }

    pub fn inflations(&self) -> &[f64] {
        &self.inflations
    }

    pub fn obstacles(&self) -> &[u32] {
        &self.obstacles
    }

    pub fn obstacle_count(&self) -> usize {
        self.obstacle_count
    }

    /// Where each item comes from, None for a field of items without
    /// records.
    pub fn picked(&self) -> Option<Vec<Picked>> {
        let mut overrides = self.overrides.iter().peekable();
        let mut picked = Vec::with_capacity(self.len());
        for (item, m) in self.members.iter().enumerate() {
            let set = &self.parts[m.part as usize];
            let k = m.index as usize;
            let record = set.record(k)?;
            let layer = set.copper().and_then(|c| c.layer[k]);
            let how = overrides
                .next_if(|(at, _)| *at as usize == item)
                .map(|&(_, how)| how);
            let (source, copper) = match how {
                None => (record.source, layer),
                Some(Override::Layer(layer)) => (record.source, Some(layer)),
                Some(Override::Stop) => (Source::PadStop, None),
            };
            picked.push(Picked {
                source,
                layer: record.layer as usize,
                index: record.index as usize,
                copper: copper.map(|c| c as usize),
            });
        }
        Some(picked)
    }

    /// The field's items among the items `found` gives of each of its
    /// sets, ascending.
    fn mapped(&self, found: impl Fn(&ItemSet) -> Vec<usize>) -> Vec<usize> {
        let Some(lookup) = &self.lookup else {
            return found(&self.parts[0]);
        };
        let mut items = Vec::new();
        for (part, places) in self.parts.iter().zip(lookup) {
            items.extend(
                found(part)
                    .into_iter()
                    .map(|k| places[k])
                    .filter(|&f| f != ABSENT)
                    .map(|f| f as usize),
            );
        }
        if !self.ordered {
            items.sort_unstable();
        }
        items
    }

    /// The place among the field's of item `k` of part `part`, None for
    /// an item that is not the field's.
    fn place(&self, part: usize, k: usize) -> Option<usize> {
        match &self.lookup {
            None => Some(k),
            Some(lookup) => {
                let f = lookup[part][k];
                (f != ABSENT).then_some(f as usize)
            }
        }
    }

    /// Each item's obstacle as a clearance check judges it, numbered below
    /// `obstacle_count`: its obstacle, or with copper given, the obstacles
    /// joined where copper of one net overlaps on one layer, as at a
    /// track's width change or a T-junction, each numbered as its lowest
    /// obstacle. Connected copper is one piece of copper, whatever the
    /// energy splits it into; copper on two layers is two, even where a via
    /// joins them, since a via crossing from one to the other's clearance
    /// enters new copper. Taken only when a moving via's clearance asks.
    pub fn clearance_obstacles(&self) -> &[u32] {
        self.clearance.get_or_init(|| {
            if self.parts.iter().any(|p| p.copper().is_none()) {
                self.obstacles.clone()
            } else {
                self.joined_copper()
            }
        })
    }

    fn joined_copper(&self) -> Vec<u32> {
        let mut nets = Vec::with_capacity(self.len());
        let mut layers = Vec::with_capacity(self.len());
        let mut extents = Vec::with_capacity(self.len());
        for item in 0..self.len() {
            let (set, k) = self.at(item);
            let copper = set.copper().expect("every part has copper");
            nets.push(copper.net[k]);
            layers.push(copper.layer[k]);
            extents.push(copper.extent[k]);
        }
        for &(item, how) in &self.overrides {
            layers[item as usize] = match how {
                Override::Layer(layer) => Some(layer),
                Override::Stop => None,
            };
        }
        let is_copper = |k: usize| nets[k].is_some() && layers[k].is_some();
        let widest = (0..self.len())
            .filter(|&k| is_copper(k))
            .map(|k| extents[k])
            .fold(0.0, f64::max);
        let mut sets = UnionFind::new(self.obstacle_count);
        for a in (0..self.len()).filter(|&k| is_copper(k)) {
            let (set, k) = self.at(a);
            let Some([x0, y0, x1, y1]) = set.bounds(k) else {
                continue;
            };
            // Two pieces overlap within both their extents; the box reaches
            // as far as the widest piece could.
            let r = extents[a] + widest + TOUCH_TOLERANCE;
            let meeting = self.mapped(|s| s.tree().meeting(V2::new(x0, y0), V2::new(x1, y1), r));
            for b in meeting {
                if b <= a || !is_copper(b) || nets[a] != nets[b] || layers[a] != layers[b] {
                    continue;
                }
                let gap = self.shape(a).shape_distance(self.shape(b));
                if gap <= extents[a] + extents[b] + TOUCH_TOLERANCE {
                    sets.join(self.obstacles[a] as usize, self.obstacles[b] as usize);
                }
            }
        }
        let mut lowest: HashMap<usize, u32> = HashMap::new();
        for o in 0..self.obstacle_count {
            let root = sets.root(o);
            lowest.entry(root).or_insert(number(o));
        }
        self.obstacles
            .iter()
            .map(|&o| {
                let root = sets.root(o as usize);
                lowest.get(&root).copied().unwrap_or(o)
            })
            .collect()
    }

    /// The edges of `item`, numbered across its set's items and ascending,
    /// among them every edge within 2·slack of the nearest to `p`: all of
    /// them, or of an item of more than INDEXED_EDGES edges those of the
    /// pieces whose middles lie within the nearest piece's middle's
    /// distance, 2·slack and half a piece. The nearest piece's middle is no
    /// nearer than the nearest edge, and every edge has a piece whose
    /// middle is within half the longest piece of the edge's nearest point.
    pub fn item_edges_near(&self, item: usize, p: V2, slack: f64) -> Vec<usize> {
        let (set, k) = self.at(item);
        let range = set.edge_range(k);
        if range.len() <= self.indexed_edges {
            return range.collect();
        }
        let pieces = set.pieces(k);
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

    /// The edges of `item`, numbered across its set's items and ascending,
    /// among them every one within `radius` of the segment from `a` to `b`.
    fn item_edges_near_segment(&self, item: usize, a: V2, b: V2, radius: f64) -> Vec<usize> {
        let (set, k) = self.at(item);
        let range = set.edge_range(k);
        if range.len() <= self.indexed_edges {
            return range.collect();
        }
        let pieces = set.pieces(k);
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
        let (set, k) = self.at(item);
        let kind = set.kind(k);
        if kind == Kind::Line {
            return false;
        }
        let boxed = set
            .bounds(k)
            .is_some_and(|[x0, y0, x1, y1]| p.x >= x0 && p.y >= y0 && p.x <= x1 && p.y <= y1);
        let within = boxed && set.shape(k).contains(p);
        if kind == Kind::Outside {
            !within
        } else {
            within
        }
    }

    /// The unit gradient at a point lying on the item's edges, where the
    /// direction to them is noise.
    pub fn normal(&self, item: usize, p: V2) -> V2 {
        let (set, k) = self.at(item);
        let kind = set.kind(k);
        let n = outward_normal(set.shape(k), p, kind != Kind::Line);
        if kind == Kind::Outside { -n } else { n }
    }

    /// The distance from `p` to the item's geometry, zero inside an area.
    fn item_distance(&self, item: usize, p: V2) -> f64 {
        let (set, k) = self.at(item);
        if set.kind(k) == Kind::Area && set.shape(k).covers(p) {
            return 0.0;
        }
        self.item_edges_near(item, p, 0.0)
            .into_iter()
            .map(|e| segment_distance(p, set.start(e), set.end(e)))
            .fold(f64::INFINITY, f64::min)
    }

    /// The item of the trees nearest `p` by its geometry, and that
    /// distance. Of equally near items, the least inflated, whose
    /// inflation leaves the widest search around the point, in whatever
    /// order the trees give them: every item whose box lies within
    /// DEGENERATE_DISTANCE of the nearest is measured, since a box's
    /// squared distance may exceed the square of an equal distance by
    /// round-off, as for a via's ring and its hole, which are one point.
    pub fn nearest_item(&self, p: V2) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (part, set) in self.parts.iter().enumerate() {
            for (k, d2) in set.tree().nearest_first(p) {
                if let Some((_, least)) = best
                    && d2 > (least + DEGENERATE_DISTANCE) * (least + DEGENERATE_DISTANCE)
                {
                    break;
                }
                let Some(item) = self.place(part, k) else {
                    continue;
                };
                let d = self.item_distance(item, p);
                let nearer = best.is_none_or(|(other, least)| {
                    d < least || d == least && self.inflations[item] < self.inflations[other]
                });
                if nearer {
                    best = Some((item, d));
                }
            }
        }
        best
    }

    /// The items of the trees whose bounding boxes lie within `radius` of
    /// `p`, ascending; none for a NaN or negative radius.
    pub fn items_within(&self, p: V2, radius: f64) -> Vec<usize> {
        self.mapped(|set| set.tree().within(p, radius))
    }

    /// The items of the trees whose bounding boxes come within `radius` of
    /// the segment from `a` to `b`'s bounding box, ascending; none for a
    /// NaN or negative radius.
    pub fn items_near_segment(&self, a: V2, b: V2, radius: f64) -> Vec<usize> {
        let low = V2::new(a.x.min(b.x), a.y.min(b.y));
        let high = V2::new(a.x.max(b.x), a.y.max(b.y));
        self.mapped(|set| set.tree().meeting(low, high, radius))
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
        let within = radius + DEGENERATE_DISTANCE;
        let mut items = self.items_within(p, within);
        if !reach.is_finite() {
            // Within a finite reach every item past it is farther than the
            // reach from the point once inflated, which each caller drops;
            // without one, nothing else drops the items the box only
            // brings in.
            items.retain(|&item| self.item_distance(item, p) <= within);
        }
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
            let (set, k) = self.at(item);
            if set.edge_count(k) == 0 {
                continue;
            }
            let kind = set.kind(k);
            let inside = self.inside(item, p);
            let edges: Vec<(usize, f64)> = self
                .item_edges_near(item, p, slack)
                .into_iter()
                .map(|e| {
                    let a = set.start(e);
                    (e, offset(p, a, set.end(e) - a).hypot())
                })
                .collect();
            let nearest = edges
                .iter()
                .map(|&(_, raw)| raw)
                .fold(f64::INFINITY, f64::min);
            let limit = reach
                + if kind == Kind::Outside {
                    self.inflations[item]
                } else {
                    self.inflation
                };
            let distance = if inside { -nearest } else { nearest } - self.inflations[item];
            let unsure =
                slack > 0.0 && kind != Kind::Line && nearest <= slack + DEGENERATE_DISTANCE;
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
        // By obstacle, then item and edge: the edges numbered across the
        // field's items in order.
        let mut kept: Vec<(u32, usize, usize, usize)> = Vec::new();
        for (n, at) in near.iter().enumerate() {
            let reachable =
                at.inside || at.unsure || at.nearest - slack <= at.limit + DEGENERATE_DISTANCE;
            if !reachable || at.distance - slack > bound + SMOOTH_REACH + tolerance {
                continue;
            }
            for &(edge, raw) in &at.edges {
                if raw <= at.nearest + 2.0 * slack + tolerance {
                    kept.push((self.obstacles[at.item], at.item, edge, n));
                }
            }
        }
        kept.sort_unstable();
        for (obstacle, item, edge, n) in kept {
            let at = &near[n];
            let (set, _) = self.at(item);
            found.point.push(i);
            found.starts.push(set.start(edge));
            found.ends.push(set.end(edge));
            found.inflation.push(self.inflations[item]);
            found.limit.push(at.limit);
            found.obstacle.push(obstacle as usize);
            found.item.push(item);
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
                let (set, k) = self.at(item);
                if set.edge_count(k) == 0 {
                    continue;
                }
                let (length, _) = self.nearest_offset(item, p);
                let inside = self.inside(item, p);
                let distance = if inside { -length } else { length } - self.inflations[item];
                measured.push((clearance[item] as usize, distance));
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
        let (set, _) = self.at(item);
        let mut best: Option<(usize, f64)> = None;
        for e in self.item_edges_near(item, p, 0.0) {
            let d = segment_distance(p, set.start(e), set.end(e));
            if best.is_none_or(|(_, least)| d < least) {
                best = Some((e, d));
            }
        }
        let Some((e, _)) = best else {
            return (f64::INFINITY, V2::default());
        };
        let toward = p - closest_point(p, set.start(e), set.end(e));
        (toward.norm(), toward)
    }

    /// The least distance from the segment from `a` to `b` to the item's
    /// edges, considering only edges that can be nearer than `below`;
    /// infinite if none is.
    pub fn segment_edge_distance(&self, item: usize, a: V2, b: V2, below: f64) -> f64 {
        let (set, _) = self.at(item);
        self.item_edges_near_segment(item, a, b, below)
            .into_iter()
            .map(|e| segment_to_segment(a, b, set.start(e), set.end(e)))
            .fold(f64::INFINITY, f64::min)
    }

    /// The item's edges that the segment from `a` to `b` may touch.
    pub fn edges_touching_segment(&self, item: usize, a: V2, b: V2) -> Vec<(V2, V2)> {
        let (set, _) = self.at(item);
        self.item_edges_near_segment(item, a, b, DEGENERATE_DISTANCE)
            .into_iter()
            .map(|e| (set.start(e), set.end(e)))
            .collect()
    }
}

/// An index or count as the u32 a field holds it in.
pub fn number(k: usize) -> u32 {
    u32::try_from(k).expect("fewer than 2³² items")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::items::Interner;

    fn disc(centre: V2, radius: f64, edges: usize) -> Shape {
        let ring = (0..=edges)
            .map(|k| {
                let angle = std::f64::consts::TAU * (k % edges) as f64 / edges as f64;
                V2::new(
                    centre.x + radius * libm::cos(angle),
                    centre.y + radius * libm::sin(angle),
                )
            })
            .collect();
        Shape::polygon(vec![ring])
    }

    fn scattered(n: usize, seed: u64) -> Vec<V2> {
        // A small linear congruential generator: the points only need to
        // be spread, not random.
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| V2::new(-1.5 + 9.0 * next(), -1.5 + 9.0 * next()))
            .collect()
    }

    const INFLATIONS: [f64; 5] = [0.3, 0.2, 0.35, 0.4, 0.3];

    fn scene_items() -> Vec<(Shape, Kind)> {
        let mut outline = disc(V2::new(3.0, 2.0), 6.0, 200);
        outline.type_id = 3;
        vec![
            (disc(V2::new(1.0, 1.0), 0.8, 300), Kind::Area),
            (disc(V2::new(4.0, 1.0), 1.5, 64), Kind::Area),
            (
                Shape::line(vec![V2::new(0.0, 3.0), V2::new(6.0, 3.5)]),
                Kind::Line,
            ),
            (Shape::line(vec![V2::new(5.0, 5.0)]), Kind::Line),
            (outline, Kind::Outside),
        ]
    }

    fn scene(indexed: usize) -> StaticEdges {
        let (shapes, kinds) = scene_items().into_iter().unzip();
        StaticEdges::new(
            shapes,
            kinds,
            INFLATIONS.to_vec(),
            vec![0, 1, 2, 3, 4],
            None,
        )
        .with_indexed_edges(indexed)
    }

    /// `scene`'s items in two sets, out of order and among items the field
    /// does not hold.
    fn split_scene(indexed: usize) -> StaticEdges {
        let items = scene_items();
        let decoy = (
            Shape::line(vec![V2::new(2.0, 2.0), V2::new(2.5, 2.0)]),
            Kind::Line,
        );
        let mut interner = Interner::new();
        let mut set = |picked: Vec<(Shape, Kind)>| {
            let (shapes, kinds) = picked.into_iter().unzip();
            Arc::new(ItemSet::new(shapes, kinds, None, Some(&mut interner)))
        };
        let first = set(vec![decoy.clone(), items[1].clone(), items[3].clone()]);
        let second = set(vec![
            items[4].clone(),
            items[2].clone(),
            decoy,
            items[0].clone(),
        ]);
        let member = |part, index| Member { part, index };
        StaticEdges::view(
            vec![first, second],
            vec![
                member(1, 3),
                member(0, 1),
                member(1, 1),
                member(0, 2),
                member(1, 0),
            ],
            INFLATIONS.to_vec(),
            vec![0, 1, 2, 3, 4],
            Vec::new(),
        )
        .with_indexed_edges(indexed)
    }

    #[test]
    fn of_a_ring_and_its_hole_the_less_inflated_is_nearest_in_either_order() {
        // From this point the tree's squared distance to the centre exceeds
        // the square of the measured one by an ulp.
        let p = V2::new(0.125, 0.625);
        let centre = || Shape::line(vec![V2::new(0.0, 0.0)]);
        for inflations in [vec![0.825, 0.725], vec![0.725, 0.825]] {
            let field = StaticEdges::new(
                vec![centre(), centre()],
                vec![Kind::Line; 2],
                inflations.clone(),
                vec![0, 0],
                None,
            );
            let (item, _) = field.nearest_item(p).unwrap();
            assert_eq!(inflations[item], 0.725);
        }
    }

    #[test]
    fn a_field_over_two_sets_answers_as_one_over_its_own() {
        let points = scattered(300, 11);
        for indexed in [0, usize::MAX] {
            let (one, two) = (scene(indexed), split_scene(indexed));
            for reach in [0.3, f64::INFINITY] {
                for slack in [0.0, 0.05] {
                    assert_eq!(
                        one.search(&points, reach, slack),
                        two.search(&points, reach, slack)
                    );
                }
                assert_eq!(one.distances(&points, reach), two.distances(&points, reach));
            }
            for &p in &points {
                assert_eq!(one.nearest_item(p), two.nearest_item(p));
                assert_eq!(one.items_within(p, 0.5), two.items_within(p, 0.5));
            }
        }
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
        assert!(near.len() * 10 < field.shape(0).edges().len());
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
