//! A static field of the board's items and its edge: signed distance to
//! the nearest inflated obstacle, and segment checks.
//!
//! Positive in free space, zero exactly at the required clearance, negative
//! inside an inflated obstacle or outside the board. Where obstacles
//! overlap, the deepest one counts.

use std::collections::HashMap;
use std::sync::Arc;

use super::edges::{Copper, StaticEdges};
use super::search::parts_smooth_distance;
use super::union_find::UnionFind;
use super::{DEGENERATE_DISTANCE, Field, Kind};
use crate::geometry::{LINE_STRING, Part, Shape, V2, outward_normal};

/// An obstacle as the field takes it: raw geometry and the inflation at
/// which what keeps clear of it is at clearance.
#[derive(Clone, Debug)]
pub struct Item {
    /// Polygonal copper, or a centreline or centre point to be inflated.
    pub shape: Shape,
    pub inflation: f64,
    /// The net of copper, None for what is not copper.
    pub net: Option<String>,
    /// How far copper reaches beyond `shape`, mm.
    pub extent: f64,
    /// The layer of copper, None for what is not copper of one layer.
    pub layer: Option<String>,
}

pub struct Obstacles {
    /// The items in order, then the board edge as an `Outside` item.
    edges: Arc<StaticEdges>,
    items: usize,
    outline: bool,
    edge_inflation: f64,
    max_inflation: f64,
}

impl Obstacles {
    /// `items`, and the outside of `outline` inflated by `edge_inflation`
    /// as the board edge.
    pub fn new(items: Vec<Item>, outline: Option<Shape>, edge_inflation: f64) -> Self {
        let mut ids = obstacle_ids(&items);
        let edge_obstacle = ids.iter().max().map_or(0, |&m| m + 1);
        let max_inflation = items.iter().map(|i| i.inflation).fold(0.0, f64::max);
        let count = items.len();
        let mut numbered: HashMap<String, usize> = HashMap::new();
        let mut number = |name: &Option<String>| -> Option<usize> {
            name.as_ref().map(|n| {
                let next = numbered.len();
                *numbered.entry(n.clone()).or_insert(next)
            })
        };
        let mut copper = Copper::default();
        let mut shapes = Vec::with_capacity(count + 1);
        let mut kinds = Vec::with_capacity(count + 1);
        let mut inflations = Vec::with_capacity(count + 1);
        for item in items {
            copper.net.push(number(&item.net));
            copper.layer.push(number(&item.layer));
            copper.extent.push(item.extent);
            kinds.push(if item.shape.is_area() {
                Kind::Area
            } else {
                Kind::Line
            });
            inflations.push(item.inflation);
            shapes.push(item.shape);
        }
        let has_outline = outline.is_some();
        if let Some(outline) = outline {
            shapes.push(outline);
            kinds.push(Kind::Outside);
            inflations.push(edge_inflation);
            ids.push(edge_obstacle);
            copper.net.push(None);
            copper.layer.push(None);
            copper.extent.push(0.0);
        }
        Self {
            edges: Arc::new(StaticEdges::new(
                shapes,
                kinds,
                inflations,
                ids,
                Some(copper),
            )),
            items: count,
            outline: has_outline,
            edge_inflation,
            max_inflation,
        }
    }

    /// The field as edges, its items in order, then the board edge.
    pub fn edges(&self) -> &Arc<StaticEdges> {
        &self.edges
    }

    /// Distance and its unit gradient at each point, mm.
    ///
    /// With no obstacles and no board outline the distance is +inf and the
    /// gradient zero. Points farther than `reach` from every inflated
    /// obstacle and from the inflated board edge get +inf and a zero
    /// gradient; points outside the board are always reported.
    pub fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let mut distance = vec![f64::INFINITY; points.len()];
        let mut gradient = vec![V2::default(); points.len()];
        for (i, &p) in points.iter().enumerate() {
            if self.items > 0 {
                let search = if reach.is_finite() {
                    // An item is within reach once inflated only if its raw
                    // geometry is within reach plus the largest inflation.
                    reach + self.max_inflation
                } else {
                    // An item farther than the nearest raw geometry can
                    // still be nearer once inflated, by at most the spread
                    // of the inflations.
                    self.edges
                        .nearest_item(p)
                        .map_or(f64::NEG_INFINITY, |(item, d)| {
                            d - self.edges.inflations()[item] + self.max_inflation
                        })
                };
                let within = search + DEGENERATE_DISTANCE;
                for item in self.edges.items_within(p, within) {
                    if self.raw_distance(item, p) > within {
                        continue;
                    }
                    let (raw, toward) = self.signed_to(item, p);
                    let d = raw - self.edges.inflations()[item];
                    if d < distance[i] {
                        distance[i] = d;
                        gradient[i] = toward;
                    }
                }
            }
            if self.outline {
                let edge = self.items;
                let (raw, toward) = self.signed_to(edge, p);
                // Inside the board the raw distance is the negative of the
                // distance to its edge.
                let near = !reach.is_finite() || raw > 0.0 || -raw - self.edge_inflation <= reach;
                let d = -raw - self.edge_inflation;
                if near && d < distance[i] {
                    distance[i] = d;
                    gradient[i] = -toward;
                }
            }
        }
        (distance, gradient)
    }

    /// The distance from `p` to the item's raw geometry, zero inside an
    /// area.
    fn raw_distance(&self, item: usize, p: V2) -> f64 {
        let shape = &self.edges.shapes()[item];
        if shape.is_area() && shape.covers(p) {
            return 0.0;
        }
        self.edges.nearest_offset(item, p).0
    }

    /// Signed distance from `p` to the item's geometry, negative inside an
    /// area, and the unit gradient.
    fn signed_to(&self, item: usize, p: V2) -> (f64, V2) {
        let shape = &self.edges.shapes()[item];
        let area = shape.is_area();
        let inside = area && shape.covers(p);
        let (length, toward) = self.edges.nearest_offset(item, p);
        let sign = if inside { -1.0 } else { 1.0 };
        let gradient = if length < DEGENERATE_DISTANCE {
            outward_normal(shape, p, area)
        } else {
            V2::new(sign * toward.x / length, sign * toward.y / length)
        };
        (sign * length, gradient)
    }

    /// `signed_distance` as the energy takes it: the smooth minimum over the
    /// obstacles, from the field's edges.
    pub fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        parts_smooth_distance(std::slice::from_ref(&self.edges), points, reach)
    }

    /// Whether each segment from starts[i] to ends[i] stays out of every
    /// inflated obstacle and inside the inflated board edge.
    ///
    /// Touching an inflated boundary, which is exactly at clearance, is
    /// clear, and so is entering it by at most `tolerance` mm. The
    /// tolerance does not extend the board outline itself, which only
    /// matters with an edge clearance below it.
    pub fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| self.segment_clear(a, b, tolerance))
            .collect()
    }

    fn segment_clear(&self, a: V2, b: V2, tolerance: f64) -> bool {
        if self.items > 0 {
            for item in self.edges.items_near_segment(a, b, self.max_inflation) {
                let gap = self.edges.shapes()[item].segment_distance(a, b);
                if gap <= self.max_inflation && gap < self.edges.inflations()[item] - tolerance {
                    return false;
                }
            }
        }
        if !self.outline {
            return true;
        }
        let edge = self.items;
        let outline = &self.edges.shapes()[edge];
        let below = self.edge_inflation - tolerance;
        outline.covers_segment(a, b, &self.edges.edges_touching_segment(edge, a, b))
            && self.edges.segment_edge_distance(edge, a, b, below) >= below
    }
}

/// A key under which items are one obstacle.
#[derive(Hash, PartialEq, Eq)]
enum Key {
    /// The same geometry, as its type and coordinates tell it.
    Geometry(Vec<u64>),
    /// An end of a centreline, with its inflation.
    End([u64; 3]),
}

fn geometry_key(shape: &Shape) -> Vec<u64> {
    let mut key = vec![
        u64::from(shape.type_id.unsigned_abs()),
        shape.parts.len() as u64,
    ];
    let points = |points: &[V2], key: &mut Vec<u64>| {
        key.push(points.len() as u64);
        for p in points {
            key.push(p.x.to_bits());
            key.push(p.y.to_bits());
        }
    };
    for part in &shape.parts {
        match part {
            Part::Polygon(rings) => {
                key.extend([0, rings.len() as u64]);
                for ring in rings {
                    points(ring, &mut key);
                }
            }
            Part::Line(line) => {
                key.push(1);
                points(line, &mut key);
            }
        }
    }
    key
}

impl Field for Obstacles {
    fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        Obstacles::signed_distance(self, points, reach)
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        Obstacles::smooth_distance(self, points, reach)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        Obstacles::segments_clear(self, starts, ends, tolerance)
    }

    fn static_parts(&self) -> Option<Vec<Arc<StaticEdges>>> {
        Some(vec![self.edges().clone()])
    }
}

/// Each item's obstacle, numbered from 0. Items of one geometry are one
/// obstacle: a pad or keepout on each of the layers a via spans, a via's
/// ring and its hole. So are centrelines of one inflation that meet end to
/// end, as the segments of a track do, so that its joints and bends add
/// nothing; the joining is transitive, so a connected run of them is one
/// obstacle however it branches.
///
/// Centrelines of two inflations stay two obstacles where they meet, as at
/// a width change, and so does a centreline whose end lands on the middle
/// of another, a T-junction: opposite its stem a point reads up to τ·ln 2
/// nearer than it is. A clearance check joins them again
/// (`StaticEdges::clearance_obstacles`).
pub fn obstacle_ids(items: &[Item]) -> Vec<usize> {
    let mut sets = UnionFind::new(items.len());
    let mut first: HashMap<Key, usize> = HashMap::new();
    for (k, item) in items.iter().enumerate() {
        let mut keys = vec![Key::Geometry(geometry_key(&item.shape))];
        if item.shape.type_id == LINE_STRING
            && let Some(Part::Line(line)) = item.shape.parts.first()
            && let (Some(a), Some(b)) = (line.first(), line.last())
        {
            let inflation = item.inflation.to_bits();
            keys.push(Key::End([a.x.to_bits(), a.y.to_bits(), inflation]));
            keys.push(Key::End([b.x.to_bits(), b.y.to_bits(), inflation]));
        }
        for key in keys {
            let f = *first.entry(key).or_insert(k);
            sets.join(k, f);
        }
    }
    let mut numbered: HashMap<usize, usize> = HashMap::new();
    (0..items.len())
        .map(|k| {
            let root = sets.root(k);
            let next = numbered.len();
            *numbered.entry(root).or_insert(next)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(points: &[(f64, f64)], inflation: f64) -> Item {
        Item {
            shape: Shape::line(points.iter().map(|&(x, y)| V2::new(x, y)).collect()),
            inflation,
            net: Some("N".into()),
            extent: 0.1,
            layer: Some("F.Cu".into()),
        }
    }

    fn square(x0: f64, y0: f64, size: f64) -> Shape {
        Shape::polygon(vec![vec![
            V2::new(x0, y0),
            V2::new(x0 + size, y0),
            V2::new(x0 + size, y0 + size),
            V2::new(x0, y0 + size),
            V2::new(x0, y0),
        ]])
    }

    #[test]
    fn centrelines_meeting_end_to_end_are_one_obstacle() {
        let items = vec![
            line(&[(0.0, 0.0), (1.0, 0.0)], 0.3),
            line(&[(1.0, 0.0), (2.0, 1.0)], 0.3),
            line(&[(2.0, 1.0), (3.0, 1.0)], 0.4),
            line(&[(5.0, 0.0), (6.0, 0.0)], 0.3),
        ];
        assert_eq!(obstacle_ids(&items), vec![0, 0, 1, 2]);
    }

    #[test]
    fn a_width_change_is_one_piece_of_copper_to_a_clearance_check() {
        let items = vec![
            line(&[(0.0, 0.0), (2.0, 0.0)], 0.3),
            line(&[(2.0, 0.0), (4.0, 0.0)], 0.4),
        ];
        let field = Obstacles::new(items, None, 0.0);
        assert_eq!(field.edges().obstacles(), &[0, 1]);
        assert_eq!(field.edges().clearance_obstacles(), &[0, 0]);
    }

    #[test]
    fn the_board_edge_is_measured_from_inside() {
        let field = Obstacles::new(vec![], Some(square(0.0, 0.0, 10.0)), 0.3);
        let (d, g) = field.signed_distance(&[V2::new(1.0, 5.0), V2::new(-1.0, 5.0)], f64::INFINITY);
        assert!((d[0] - 0.7).abs() < 1e-12);
        assert!((d[1] + 1.3).abs() < 1e-12);
        assert_eq!(g[0], V2::new(1.0, 0.0));
        assert_eq!(g[1], V2::new(1.0, 0.0));
    }

    #[test]
    fn a_point_inside_an_area_is_at_minus_its_depth() {
        let items = vec![Item {
            shape: square(0.0, 0.0, 2.0),
            inflation: 0.25,
            net: None,
            extent: 0.0,
            layer: None,
        }];
        let field = Obstacles::new(items, None, 0.0);
        let (d, _) = field.signed_distance(&[V2::new(0.5, 1.0), V2::new(0.0, 1.0)], f64::INFINITY);
        assert!((d[0] + 0.75).abs() < 1e-12);
        assert!((d[1] + 0.25).abs() < 1e-12);
        let (smooth, _) = field.smooth_distance(&[V2::new(0.5, 1.0)], f64::INFINITY);
        assert!((smooth[0] + 0.75).abs() < 1e-12);
    }

    #[test]
    fn segments_keep_clear_of_items_and_inside_the_board() {
        let items = vec![line(&[(5.0, 0.0), (5.0, 10.0)], 0.5)];
        let field = Obstacles::new(items, Some(square(0.0, -1.0, 12.0)), 0.3);
        let clear = field.segments_clear(
            &[V2::new(1.0, 1.0), V2::new(1.0, 1.0), V2::new(1.0, 1.0)],
            &[V2::new(4.0, 1.0), V2::new(4.6, 1.0), V2::new(1.0, -1.0)],
            0.0,
        );
        assert_eq!(clear, vec![true, false, false]);
    }
}
