//! Other selected chains as obstacles: signed distance to their inflated
//! centrelines, and edge checks.
//!
//! The same answers as `Obstacles` with each centreline as an obstacle,
//! except that zero-length edges and zero-length query segments count like
//! any other, where shapely's tree query skips them. The optimizer rebuilds
//! fields like this for every burst and check. A tree over the edge
//! middles finds the edges near each query, and the exact distances to
//! those edges are measured over (point, edge) pairs (`pairs`), as the
//! members' view of each other measures them for many members at once.
//!
//! Its smooth distance takes the edges of one obstacle, a chain or a via,
//! as one: the nearest of them, then the smooth minimum over the obstacles
//! (`pairs::smooth_of_pairs`).

use std::sync::{Arc, OnceLock};

use super::edges::StaticEdges;
use super::pairs::{
    NearestEdge, PointTree, SmoothAnswer, nearest_of_pairs, obstacle_order, pairs_inside,
    smooth_of_pairs,
};
use super::smooth::{SMOOTH_REACH, smooth_reach};
use super::{DEGENERATE_DISTANCE, Field, MAX_EDGE_LENGTH};
use crate::geometry::{V2, offset};

/// Edges, each inflated by its own amount and part of an obstacle.
pub struct ChainField {
    starts: Vec<V2>,
    along: Vec<V2>,
    middles: Vec<V2>,
    inflations: Vec<f64>,
    obstacles: Vec<usize>,
    max_inflation: f64,
    max_half_length: f64,
    /// Built on first use: a field whose checks all end early needs none.
    tree: OnceLock<PointTree>,
    edges: OnceLock<Arc<StaticEdges>>,
}

/// (point, edge) pairs to measure, and how far from its point each is
/// looked at.
struct Pairs {
    point: Vec<usize>,
    edge: Vec<usize>,
    limit: Vec<f64>,
}

impl ChainField {
    /// Edges from starts[i] to ends[i], each inflated by inflations[i] and
    /// part of the obstacle obstacles[i].
    pub fn new(starts: &[V2], ends: &[V2], inflations: Vec<f64>, obstacles: Vec<usize>) -> Self {
        let along: Vec<V2> = starts.iter().zip(ends).map(|(&a, &b)| b - a).collect();
        let max_inflation = inflations.iter().copied().fold(0.0, f64::max);
        let max_half_length = along.iter().map(|a| a.hypot()).fold(0.0, f64::max) / 2.0;
        let middles = starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| {
                let sum = a + b;
                V2::new(sum.x / 2.0, sum.y / 2.0)
            })
            .collect();
        Self {
            starts: starts.to_vec(),
            along,
            middles,
            inflations,
            obstacles,
            max_inflation,
            max_half_length,
            tree: OnceLock::new(),
            edges: OnceLock::new(),
        }
    }

    /// The polylines as a field, each split into edges of at most
    /// MAX_EDGE_LENGTH (`split_edges`), inflated by its inflation and part
    /// of its obstacle.
    pub fn from_polylines<'a>(polylines: impl IntoIterator<Item = (&'a [V2], f64, usize)>) -> Self {
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut inflations = Vec::new();
        let mut obstacles = Vec::new();
        for (points, inflation, obstacle) in polylines {
            let (a, b) = split_edges(points);
            inflations.extend(std::iter::repeat_n(inflation, a.len()));
            obstacles.extend(std::iter::repeat_n(obstacle, a.len()));
            starts.extend(a);
            ends.extend(b);
        }
        Self::new(&starts, &ends, inflations, obstacles)
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    fn tree(&self) -> &PointTree {
        self.tree.get_or_init(|| PointTree::new(&self.middles))
    }

    /// The field as `StaticEdges`, each edge an item of its own, for a field
    /// that stands still. Its distances differ from `smooth_distance`'s by
    /// round-off, since `StaticEdges` measures by GEOS's point-to-segment
    /// formula.
    pub fn static_edges(&self) -> &Arc<StaticEdges> {
        self.edges.get_or_init(|| {
            let ends: Vec<V2> = self
                .starts
                .iter()
                .zip(&self.along)
                .map(|(&s, &a)| s + a)
                .collect();
            Arc::new(StaticEdges::from_segments(
                &self.starts,
                &ends,
                self.inflations.clone(),
                self.obstacles.clone(),
            ))
        })
    }

    /// `signed_distance`, and the edge that answers each point, None where
    /// none does.
    pub fn nearest(&self, points: &[V2], reach: f64) -> NearestEdge {
        if self.is_empty() || points.is_empty() {
            return NearestEdge {
                distance: vec![f64::INFINITY; points.len()],
                gradient: vec![V2::default(); points.len()],
                edge: vec![None; points.len()],
            };
        }
        let pairs = self.search(points, reach, 0.0);
        let found = nearest_of_pairs(
            points,
            &pairs.point,
            &gather(&self.starts, &pairs.edge),
            &gather(&self.along, &pairs.edge),
            &gather(&self.inflations, &pairs.edge),
            &pairs.limit,
        );
        NearestEdge {
            edge: found
                .edge
                .iter()
                .map(|e| e.map(|k| pairs.edge[k]))
                .collect(),
            ..found
        }
    }

    /// `smooth_distance`, and the edges that answer each point, by their
    /// index, with their shares.
    pub fn smooth_answers(&self, points: &[V2], reach: f64) -> SmoothAnswer {
        if self.is_empty() || points.is_empty() {
            return SmoothAnswer::free(points.len());
        }
        let pairs = self.search(points, smooth_reach(reach), SMOOTH_REACH);
        let obstacle = gather(&self.obstacles, &pairs.edge);
        let order = obstacle_order(&pairs.point, &obstacle, &pairs.edge);
        let edge = gather(&pairs.edge, &order);
        let found = smooth_of_pairs(
            points,
            &gather(&pairs.point, &order),
            &gather(&self.starts, &edge),
            &gather(&self.along, &edge),
            &gather(&self.inflations, &edge),
            &gather(&pairs.limit, &order),
            &gather(&obstacle, &order),
        );
        SmoothAnswer {
            edge: found.edge.iter().map(|&k| edge[k]).collect(),
            ..found
        }
    }

    /// (point, edge) pairs among which are the edges within `reach` of each
    /// point once inflated, or without a finite reach those within `beyond`
    /// of its nearest.
    fn search(&self, points: &[V2], reach: f64, beyond: f64) -> Pairs {
        let tree = self.tree();
        let mut pairs = Pairs {
            point: Vec::new(),
            edge: Vec::new(),
            limit: Vec::new(),
        };
        for (i, &p) in points.iter().enumerate() {
            let limit = if reach.is_finite() {
                // An edge is within reach once inflated only if it is within
                // reach plus the largest inflation.
                reach + self.max_inflation
            } else {
                let Some((nearest, _)) = tree.nearest(p) else {
                    continue;
                };
                let raw = offset(p, self.starts[nearest], self.along[nearest]).hypot();
                // An edge farther than the nearest one can still be nearer
                // once inflated, by at most the spread of the inflations.
                raw - self.inflations[nearest] + self.max_inflation + beyond
            };
            for k in tree.within(p, limit + self.max_half_length + DEGENERATE_DISTANCE) {
                pairs.point.push(i);
                pairs.edge.push(k);
                pairs.limit.push(limit);
            }
        }
        pairs
    }
}

impl Field for ChainField {
    /// With no edges the distance is +inf and the gradient zero everywhere.
    /// Points farther than `reach` from every inflated edge get +inf and a
    /// zero gradient.
    fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let found = self.nearest(points, reach);
        (found.distance, found.gradient)
    }

    /// `signed_distance` as the energy takes it: the smooth minimum over
    /// the obstacles of each one's nearest edge.
    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let found = self.smooth_answers(points, reach);
        (found.distance, found.gradient)
    }

    /// Whether each segment stays out of every inflated edge. Touching an
    /// inflated boundary, which is exactly at clearance, is clear, and so
    /// is entering it by at most `tolerance` mm.
    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        let mut clear = vec![true; starts.len()];
        let reach = self.max_inflation - tolerance;
        if self.is_empty() || reach <= 0.0 || starts.is_empty() {
            return clear;
        }
        let tree = self.tree();
        let mut segment = Vec::new();
        let mut edge = Vec::new();
        for (i, (&a, &b)) in starts.iter().zip(ends).enumerate() {
            let sum = a + b;
            let middle = V2::new(sum.x / 2.0, sum.y / 2.0);
            let radius = reach + (b - a).hypot() / 2.0 + self.max_half_length + DEGENERATE_DISTANCE;
            for k in tree.within(middle, radius) {
                segment.push(i);
                edge.push(k);
            }
        }
        let a: Vec<V2> = segment.iter().map(|&i| starts[i]).collect();
        let ab: Vec<V2> = segment.iter().map(|&i| ends[i] - starts[i]).collect();
        let inside = pairs_inside(
            &a,
            &ab,
            &gather(&self.starts, &edge),
            &gather(&self.along, &edge),
            &gather(&self.inflations, &edge),
            tolerance,
        );
        for (&i, inside) in segment.iter().zip(inside) {
            if inside {
                clear[i] = false;
            }
        }
        clear
    }

    fn static_parts(&self) -> Option<Vec<Arc<StaticEdges>>> {
        Some(vec![self.static_edges().clone()])
    }
}

fn gather<T: Copy>(values: &[T], at: &[usize]) -> Vec<T> {
    at.iter().map(|&k| values[k]).collect()
}

/// The polyline's edges, each split evenly into parts of at most
/// MAX_EDGE_LENGTH, so that a point near an edge is also near a part's
/// middle, where the tree finds it. Chains resampled at the default 0.1 mm
/// spacing are never split.
pub fn split_edges(vertices: &[V2]) -> (Vec<V2>, Vec<V2>) {
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for w in vertices.windows(2) {
        let (a, along) = (w[0], w[1] - w[0]);
        let parts = (libm::hypot(along.x, along.y) / MAX_EDGE_LENGTH)
            .ceil()
            .max(1.0) as usize;
        for part in 0..parts {
            let first = part as f64 / parts as f64;
            let last = (part + 1) as f64 / parts as f64;
            starts.push(a + first * along);
            ends.push(a + last * along);
        }
    }
    (starts, ends)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight track along y = 0 of 100 edges 0.1 mm long, one
    /// obstacle, kept 0.2 mm from.
    fn track() -> ChainField {
        let points: Vec<V2> = (0..=100).map(|k| V2::new(k as f64 * 0.1, 0.0)).collect();
        ChainField::from_polylines([(points.as_slice(), 0.2, 0)])
    }

    #[test]
    fn a_straight_track_of_many_edges_is_one_obstacle() {
        let field = track();
        let at = [V2::new(5.0, 0.4)];
        let (smooth, toward) = field.smooth_distance(&at, 1.0);
        let (exact, exact_toward) = field.signed_distance(&at, f64::INFINITY);
        assert!((smooth[0] - exact[0]).abs() < 1e-12);
        assert!((toward[0] - exact_toward[0]).hypot() < 1e-12);
        assert!((exact[0] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn points_beyond_the_reach_are_free() {
        let (distance, gradient) = track().signed_distance(&[V2::new(5.0, 3.0)], 1.0);
        assert_eq!(distance, vec![f64::INFINITY]);
        assert_eq!(gradient, vec![V2::default()]);
    }

    #[test]
    fn two_chains_equally_near_share_a_point() {
        let below = [V2::new(0.0, -1.0), V2::new(2.0, -1.0)];
        let above = [V2::new(0.0, 1.0), V2::new(2.0, 1.0)];
        let field = ChainField::from_polylines([(&below[..], 0.5, 0), (&above[..], 0.5, 1)]);
        let found = field.smooth_answers(&[V2::new(1.0, 0.0)], f64::INFINITY);
        assert_eq!(found.share, vec![0.5, 0.5]);
        assert_eq!(found.edge.len(), 2);
        assert!(found.gradient[0].hypot() < 1e-15);
    }

    #[test]
    fn a_segment_crossing_an_inflated_edge_is_not_clear() {
        let field = track();
        let clear = field.segments_clear(
            &[V2::new(5.0, 1.0), V2::new(5.0, 1.0)],
            &[V2::new(5.0, 0.1), V2::new(6.0, 1.0)],
            0.0,
        );
        assert_eq!(clear, vec![false, true]);
    }

    #[test]
    fn an_empty_field_is_free_and_clear() {
        let field = ChainField::new(&[], &[], Vec::new(), Vec::new());
        let at = [V2::new(0.0, 0.0)];
        assert_eq!(
            field.signed_distance(&at, f64::INFINITY).0,
            vec![f64::INFINITY]
        );
        assert_eq!(field.smooth_distance(&at, 1.0).0, vec![f64::INFINITY]);
        assert_eq!(field.segments_clear(&at, &at, 0.0), vec![true]);
    }

    #[test]
    fn long_edges_are_split_into_parts_of_at_most_the_maximum() {
        let (starts, ends) = split_edges(&[V2::new(0.0, 0.0), V2::new(1.2, 0.0)]);
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[0], V2::new(0.0, 0.0));
        assert!((ends[2].x - 1.2).abs() < 1e-15);
        assert!(
            starts
                .iter()
                .zip(&ends)
                .all(|(a, b)| (*b - *a).norm() <= MAX_EDGE_LENGTH)
        );
    }

    #[test]
    fn the_nearest_edge_is_reported() {
        let field = track();
        let found = field.nearest(&[V2::new(5.05, 0.5)], f64::INFINITY);
        assert_eq!(found.edge, vec![Some(50)]);
    }
}
