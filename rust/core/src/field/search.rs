//! The search and smooth distance over several static fields at once: their
//! items numbered one part after another, so that the smooth minimum is
//! over all their obstacles.

use std::sync::Arc;

use super::DEGENERATE_DISTANCE;
use super::edges::{Found, StaticDistances, StaticEdges};
use super::pairs::{nearest_in_runs, nearest_then_smooth};
use super::smooth::{smooth_reach, weighted_sum};
use crate::geometry::{V2, closest_point, segment_distance};

/// The items of several `StaticEdges`, numbered one part after another,
/// for what `static_smooth` asks of an item.
#[derive(Clone)]
pub struct StaticItems {
    parts: Vec<Arc<StaticEdges>>,
    /// Where each part's items start, and then their count.
    starts: Vec<usize>,
}

impl StaticItems {
    pub fn new(parts: Vec<Arc<StaticEdges>>) -> Self {
        let mut starts = vec![0];
        for part in &parts {
            starts.push(starts[starts.len() - 1] + part.len());
        }
        Self { parts, starts }
    }

    pub fn parts(&self) -> &[Arc<StaticEdges>] {
        &self.parts
    }

    pub fn starts(&self) -> &[usize] {
        &self.starts
    }

    /// The part of `item` and its number there; None past the last.
    fn part_of(&self, item: usize) -> Option<(&StaticEdges, usize)> {
        let p = self.starts.partition_point(|&s| s <= item).checked_sub(1)?;
        self.parts
            .get(p)
            .map(|part| (part.as_ref(), item - self.starts[p]))
    }

    /// `StaticEdges::inside` of an item numbered across the parts.
    pub fn inside(&self, item: usize, p: V2) -> bool {
        self.part_of(item)
            .is_some_and(|(part, k)| part.inside(k, p))
    }

    /// `StaticEdges::normal` of an item numbered across the parts.
    pub fn normal(&self, item: usize, p: V2) -> V2 {
        self.part_of(item)
            .map_or(crate::geometry::FALLBACK_DIRECTION, |(part, k)| {
                part.normal(k, p)
            })
    }
}

/// `StaticEdges::search` over several parts, each with the number its items
/// start at, their obstacles numbered one part after another.
pub fn search_parts(
    parts: &[(&StaticEdges, usize)],
    coords: &[V2],
    reach: f64,
    slack: f64,
) -> Found {
    let mut joined = Found::default();
    let mut obstacles = 0;
    for &(part, first) in parts {
        joined.extend(part.search(coords, reach, slack), 0, obstacles, first);
        obstacles += part.obstacle_count();
    }
    if parts.len() <= 1 {
        return joined;
    }
    let mut order: Vec<usize> = (0..joined.len()).collect();
    order.sort_by_key(|&k| (joined.point[k], joined.obstacle[k]));
    joined.taken(&order)
}

/// `StaticEdges::distances` over several parts, their obstacles numbered
/// one part after another, as `search_parts` numbers them.
pub fn static_distances(parts: &[&StaticEdges], coords: &[V2], below: f64) -> StaticDistances {
    let mut all: Vec<(usize, usize, f64)> = Vec::new();
    let mut first = 0;
    for part in parts {
        let found = part.distances(coords, below);
        for k in 0..found.point.len() {
            all.push((found.point[k], found.obstacle[k] + first, found.distance[k]));
        }
        first += part.obstacle_count();
    }
    all.sort_by_key(|&(p, o, _)| (p, o));
    StaticDistances {
        point: all.iter().map(|a| a.0).collect(),
        obstacle: all.iter().map(|a| a.1).collect(),
        distance: all.iter().map(|a| a.2).collect(),
    }
}

/// The smooth distance over the obstacles of `parts`, from one search.
pub fn parts_smooth_distance(
    parts: &[Arc<StaticEdges>],
    points: &[V2],
    reach: f64,
) -> (Vec<f64>, Vec<V2>) {
    let items = StaticItems::new(parts.to_vec());
    let numbered: Vec<(&StaticEdges, usize)> = parts
        .iter()
        .zip(items.starts())
        .map(|(part, &first)| (part.as_ref(), first))
        .collect();
    let found = search_parts(&numbered, points, smooth_reach(reach), 0.0);
    static_smooth(points, &found, &items)
}

/// The smooth distance at each point of `query` and its gradient over the
/// pairs `found` gives, its items numbered as `items` numbers them: of each
/// item the nearest edge to a point, the exact signed distance, then of
/// each obstacle the nearest item, then the smooth minimum over the
/// obstacles. A point inside an area is at the negative of its distance to
/// the area's boundary, and a point on an item's edge takes its normal.
pub fn static_smooth(query: &[V2], found: &Found, items: &StaticItems) -> (Vec<f64>, Vec<V2>) {
    let count = query.len();
    if found.is_empty() {
        return (vec![f64::INFINITY; count], vec![V2::default(); count]);
    }
    let n = found.len();
    let inside: Vec<bool> = (0..n)
        .map(|k| {
            if found.unsure[k] {
                items.inside(found.item[k], query[found.point[k]])
            } else {
                found.inside[k]
            }
        })
        .collect();
    let counted: Vec<f64> = (0..n)
        .map(|k| {
            let key = segment_distance(query[found.point[k]], found.starts[k], found.ends[k]);
            if inside[k] || key <= found.limit[k] + DEGENERATE_DISTANCE {
                key
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let best = nearest_in_runs(&found.point, &found.item, &counted);
    let measured: Vec<(V2, f64, f64)> = best
        .iter()
        .map(|&k| {
            let at = query[found.point[k]];
            let toward = at - closest_point(at, found.starts[k], found.ends[k]);
            let length = toward.norm();
            let sign = if inside[k] { -1.0 } else { 1.0 };
            let distance = if counted[k].is_finite() {
                sign * length
            } else {
                f64::INFINITY
            } - found.inflation[k];
            (toward, length, distance)
        })
        .collect();
    let point: Vec<usize> = best.iter().map(|&k| found.point[k]).collect();
    let obstacle: Vec<usize> = best.iter().map(|&k| found.obstacle[k]).collect();
    let distance: Vec<f64> = measured.iter().map(|m| m.2).collect();
    let (chosen, smooth, share) = nearest_then_smooth(&point, &obstacle, &distance, count);
    let direction: Vec<V2> = chosen
        .iter()
        .map(|&c| {
            let k = best[c];
            let (toward, length, _) = measured[c];
            if length < DEGENERATE_DISTANCE {
                items.normal(found.item[k], query[found.point[k]])
            } else {
                let sign = if inside[k] { -1.0 } else { 1.0 };
                V2::new(sign * toward.x / length, sign * toward.y / length)
            }
        })
        .collect();
    let at: Vec<usize> = chosen.iter().map(|&c| point[c]).collect();
    (smooth, weighted_sum(&at, &share, &direction, count))
}
