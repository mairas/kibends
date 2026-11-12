//! Distances over (point, segment) pairs: a point and a segment it may be
//! answered by, each segment inflated by its own amount and part of an
//! obstacle. A chain field, the members' view of each other and the moving
//! vias' view search for their pairs and measure them here, all at once.

use rstar::RTree;
use rstar::primitives::GeomWithData;

use super::smooth::{SMOOTH_REACH, smooth_minimum, weighted_sum};
use super::{DEGENERATE_DISTANCE, ROUND_OFF};
use crate::geometry::{FALLBACK_DIRECTION, V2, offset, segment_gap};

/// The nearest inflated segment at each point, and which pair answers it.
#[derive(Debug, Default)]
pub struct NearestEdge {
    pub distance: Vec<f64>,
    pub gradient: Vec<V2>,
    /// The pair that answers each point, None where none is in reach.
    pub edge: Vec<Option<usize>>,
}

/// The smooth distance at each point and its gradient, and what answers
/// it: per obstacle that has a share at a point, the point, the pair
/// nearest it of that obstacle, its share and its unit gradient, in the
/// order of their points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmoothAnswer {
    pub distance: Vec<f64>,
    pub gradient: Vec<V2>,
    pub point: Vec<usize>,
    pub edge: Vec<usize>,
    pub share: Vec<f64>,
    pub direction: Vec<V2>,
}

impl SmoothAnswer {
    /// The answer at `count` points that nothing answers.
    pub fn free(count: usize) -> Self {
        Self {
            distance: vec![f64::INFINITY; count],
            gradient: vec![V2::default(); count],
            ..Self::default()
        }
    }
}

/// The direction away from a segment along `along` at `offset` from it;
/// its normal, a quarter turn of its tangent, on the segment. Scaled by the
/// reciprocal of the length, as numpy divides a complex number by a real
/// one, which differs from dividing in the last bit.
pub fn unit_gradient(offset: V2, length: f64, along: V2) -> V2 {
    if length >= DEGENERATE_DISTANCE {
        let scale = 1.0 / length;
        return V2::new(offset.x * scale, offset.y * scale);
    }
    let norm = along.hypot();
    if norm > 0.0 {
        let scale = 1.0 / norm;
        V2::new(-along.y * scale, along.x * scale)
    } else {
        FALLBACK_DIRECTION
    }
}

/// The nearest inflated segment at each of `query` over (point, segment)
/// pairs: pair k is the segment from starts[k] along along[k], inflated by
/// inflations[k], looked at within limit[k] of query[point[k]]. Of equally
/// near pairs at a point, the last answers.
pub fn nearest_of_pairs(
    query: &[V2],
    point: &[usize],
    starts: &[V2],
    along: &[V2],
    inflations: &[f64],
    limit: &[f64],
) -> NearestEdge {
    let mut distance = vec![f64::INFINITY; query.len()];
    let mut gradient = vec![V2::default(); query.len()];
    let mut edge = vec![None; query.len()];
    let candidates: Vec<(V2, f64, f64)> = (0..point.len())
        .map(|k| {
            let o = offset(query[point[k]], starts[k], along[k]);
            let raw = o.hypot();
            let candidate = if raw <= limit[k] + DEGENERATE_DISTANCE {
                raw - inflations[k]
            } else {
                f64::INFINITY
            };
            (o, raw, candidate)
        })
        .collect();
    for (k, &(_, _, candidate)) in candidates.iter().enumerate() {
        let p = point[k];
        distance[p] = distance[p].min(candidate);
    }
    for (k, &(o, raw, candidate)) in candidates.iter().enumerate() {
        let p = point[k];
        if candidate == distance[p] && candidate < f64::INFINITY {
            gradient[p] = unit_gradient(o, raw, along[k]);
            edge[p] = Some(k);
        }
    }
    NearestEdge {
        distance,
        gradient,
        edge,
    }
}

/// The smooth distance at each of `query` over (point, segment) pairs, as
/// `nearest_of_pairs` takes them, pair k's segment part of obstacle[k]: of
/// each obstacle the nearest segment to a point, of equal ones the first,
/// then the smooth minimum over the obstacles. The pairs come in the order
/// `obstacle_order` gives. The edge that answers is its pair's index.
pub fn smooth_of_pairs(
    query: &[V2],
    point: &[usize],
    starts: &[V2],
    along: &[V2],
    inflations: &[f64],
    limit: &[f64],
    obstacle: &[usize],
) -> SmoothAnswer {
    let count = query.len();
    if point.is_empty() {
        return SmoothAnswer::free(count);
    }
    let offsets: Vec<V2> = (0..point.len())
        .map(|k| offset(query[point[k]], starts[k], along[k]))
        .collect();
    let raw: Vec<f64> = offsets.iter().map(|o| o.hypot()).collect();
    let distance: Vec<f64> = (0..point.len())
        .map(|k| {
            if raw[k] <= limit[k] + DEGENERATE_DISTANCE {
                raw[k] - inflations[k]
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let (best, smooth, share) = nearest_then_smooth(point, obstacle, &distance, count);
    let at: Vec<usize> = best.iter().map(|&k| point[k]).collect();
    let direction: Vec<V2> = best
        .iter()
        .map(|&k| unit_gradient(offsets[k], raw[k], along[k]))
        .collect();
    SmoothAnswer {
        distance: smooth,
        gradient: weighted_sum(&at, &share, &direction, count),
        point: at,
        edge: best,
        share,
        direction,
    }
}

/// Of candidates in the order `obstacle_order` gives, candidate k at
/// distance[k] from point[k] and part of obstacle[k]: each obstacle's
/// nearest at each point, of equal ones the first, those that have a
/// share; the smooth minimum over them at each of `count` points; and
/// their shares.
pub fn nearest_then_smooth(
    point: &[usize],
    obstacle: &[usize],
    distance: &[f64],
    count: usize,
) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let best = nearest_in_runs(point, obstacle, distance);
    let at: Vec<usize> = best.iter().map(|&k| point[k]).collect();
    let nearest: Vec<f64> = best.iter().map(|&k| distance[k]).collect();
    let (smooth, share) = smooth_minimum(&at, &nearest, count);
    let (best, share): (Vec<usize>, Vec<f64>) = best
        .into_iter()
        .zip(share)
        .filter(|&(_, s)| s > 0.0)
        .unzip();
    (best, smooth, share)
}

/// Of each run of consecutive candidates of one point and one group, the
/// first of the nearest by `distance`, by index.
pub fn nearest_in_runs(point: &[usize], group: &[usize], distance: &[f64]) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for k in 0..point.len() {
        let starts = k == 0 || point[k] != point[k - 1] || group[k] != group[k - 1];
        if starts {
            best.push(k);
        } else if let Some(b) = best.last_mut()
            && distance[k] < distance[*b]
        {
            *b = k;
        }
    }
    best
}

/// Which (point, segment) pairs, as `smooth_of_pairs` takes them, can
/// answer their point within within[k] once the distance from each point
/// to each segment has changed by at most `slack`, mm.
pub fn candidate_pairs(
    query: &[V2],
    point: &[usize],
    starts: &[V2],
    along: &[V2],
    inflations: &[f64],
    within: &[f64],
    slack: f64,
) -> Vec<bool> {
    let raw: Vec<f64> = (0..point.len())
        .map(|k| offset(query[point[k]], starts[k], along[k]).hypot())
        .collect();
    // Only obstacles within SMOOTH_REACH of the nearest answer a point. An
    // edge that stays within reach however its distance changes bounds how
    // far away the nearest is, and an edge that cannot come within
    // SMOOTH_REACH of the bound never answers.
    let mut bound = vec![f64::INFINITY; query.len()];
    for k in 0..point.len() {
        if raw[k] + slack <= within[k] {
            let p = point[k];
            bound[p] = bound[p].min(raw[k] - inflations[k] + slack);
        }
    }
    (0..point.len())
        .map(|k| {
            raw[k] <= within[k] + slack + 2.0 * DEGENERATE_DISTANCE
                && raw[k] - inflations[k] - slack
                    <= bound[point[k]] + SMOOTH_REACH + 2.0 * DEGENERATE_DISTANCE
        })
        .collect()
}

/// Over (segment, edge) pairs: whether the segment from a[k] along ab[k]
/// enters the edge from starts[k] along along[k], inflated by
/// inflations[k], by more than `tolerance`.
pub fn pairs_inside(
    a: &[V2],
    ab: &[V2],
    starts: &[V2],
    along: &[V2],
    inflations: &[f64],
    tolerance: f64,
) -> Vec<bool> {
    (0..a.len())
        .map(|k| {
            segment_gap(a[k], ab[k], starts[k], along[k]) < inflations[k] - tolerance - ROUND_OFF
        })
        .collect()
}

/// The signed distance between each pair of `pairs_inside`: the gap
/// between the segment and the edge, less the edge's inflation; negative
/// inside, and least, -inflation, where the two cross.
pub fn pair_depths(
    a: &[V2],
    ab: &[V2],
    starts: &[V2],
    along: &[V2],
    inflations: &[f64],
) -> Vec<f64> {
    (0..a.len())
        .map(|k| segment_gap(a[k], ab[k], starts[k], along[k]) - inflations[k])
        .collect()
}

/// The order of (point, segment) pairs, pair k's segment part of
/// obstacle[k], that `smooth_of_pairs` takes them in: by point, then by
/// obstacle, then by tiebreak[k], unique among a point's pairs, such as the
/// segment's index, so that of equally near segments of an obstacle the
/// lowest tiebreak answers.
pub fn obstacle_order(point: &[usize], obstacle: &[usize], tiebreak: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..point.len()).collect();
    order.sort_unstable_by_key(|&k| (point[k], obstacle[k], tiebreak[k]));
    order
}

type Indexed = GeomWithData<[f64; 2], usize>;

/// A tree over points, each by its index.
pub struct PointTree {
    tree: RTree<Indexed>,
}

impl PointTree {
    pub fn new(points: &[V2]) -> Self {
        let items = points
            .iter()
            .enumerate()
            .map(|(k, p)| GeomWithData::new(p.array(), k))
            .collect();
        Self {
            tree: RTree::bulk_load(items),
        }
    }

    /// The indices of the points within `radius` of `at`, ascending.
    pub fn within(&self, at: V2, radius: f64) -> Vec<usize> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        let mut found: Vec<usize> = self
            .tree
            .locate_within_distance(at.array(), radius * radius)
            .map(|g| g.data)
            .collect();
        found.sort_unstable();
        found
    }

    /// The point nearest `at`, by its index, and its distance.
    pub fn nearest(&self, at: V2) -> Option<(usize, f64)> {
        self.tree
            .nearest_neighbor_iter_with_distance_2(at.array())
            .next()
            .map(|(g, d2)| (g.data, d2.sqrt()))
    }
}

/// A tree over segments' middles, which finds every segment that may come
/// within a distance of a point.
pub struct SegmentTree {
    tree: PointTree,
    half_length: f64,
}

impl SegmentTree {
    /// The segments from starts[k] to ends[k].
    pub fn new(starts: &[V2], ends: &[V2]) -> Self {
        let half_length = starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| (b - a).norm())
            .fold(0.0, f64::max)
            / 2.0;
        let middles: Vec<V2> = starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| 0.5 * (a + b))
            .collect();
        Self {
            tree: PointTree::new(&middles),
            half_length,
        }
    }

    /// The segments, by index ascending, among them every one that comes
    /// within `radius` of `p`.
    pub fn near(&self, p: V2, radius: f64) -> Vec<usize> {
        self.tree
            .within(p, radius + self.half_length + DEGENERATE_DISTANCE)
    }
}

/// (point, segment) index pairs, by point then segment, among them every
/// pair whose segment from starts[k] to ends[k] comes within `radius` of
/// points[i], found with one tree over the segments' middles.
pub fn near_segments(
    starts: &[V2],
    ends: &[V2],
    points: &[V2],
    radius: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut point = Vec::new();
    let mut segment = Vec::new();
    if starts.is_empty() || points.is_empty() {
        return (point, segment);
    }
    let tree = SegmentTree::new(starts, ends);
    for (i, &p) in points.iter().enumerate() {
        for k in tree.near(p, radius) {
            point.push(i);
            segment.push(k);
        }
    }
    (point, segment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_nearest_segment_answers_with_its_inflation_taken_off() {
        let query = [V2::new(0.0, 1.0)];
        let found = nearest_of_pairs(
            &query,
            &[0, 0],
            &[V2::new(-1.0, 0.0), V2::new(-1.0, 3.0)],
            &[V2::new(2.0, 0.0), V2::new(2.0, 0.0)],
            &[0.25, 0.25],
            &[10.0, 10.0],
        );
        assert_eq!(found.distance, vec![0.75]);
        assert_eq!(found.edge, vec![Some(0)]);
        assert_eq!(found.gradient[0], V2::new(0.0, 1.0));
    }

    #[test]
    fn a_point_on_a_segment_takes_its_normal() {
        assert_eq!(
            unit_gradient(V2::default(), 0.0, V2::new(2.0, 0.0)),
            V2::new(-0.0, 1.0)
        );
        assert_eq!(
            unit_gradient(V2::default(), 0.0, V2::default()),
            FALLBACK_DIRECTION
        );
    }

    #[test]
    fn two_equally_near_obstacles_share_the_point() {
        let query = [V2::new(0.0, 0.0)];
        let found = smooth_of_pairs(
            &query,
            &[0, 0],
            &[V2::new(-1.0, 1.0), V2::new(-1.0, -1.0)],
            &[V2::new(2.0, 0.0), V2::new(2.0, 0.0)],
            &[0.0, 0.0],
            &[10.0, 10.0],
            &[0, 1],
        );
        let expected = 1.0 - super::super::smooth::SMOOTH_WIDTH * 2f64.ln();
        assert!((found.distance[0] - expected).abs() < 1e-15);
        assert_eq!(found.share, vec![0.5, 0.5]);
        assert!(found.gradient[0].hypot() < 1e-15);
    }

    #[test]
    fn crossing_segments_are_inside_by_the_whole_inflation() {
        let depth = pair_depths(
            &[V2::new(0.0, -1.0)],
            &[V2::new(0.0, 2.0)],
            &[V2::new(-1.0, 0.0)],
            &[V2::new(2.0, 0.0)],
            &[0.3],
        );
        assert_eq!(depth, vec![-0.3]);
    }

    #[test]
    fn near_segments_finds_every_segment_within_the_radius() {
        let starts = [V2::new(0.0, 0.0), V2::new(5.0, 0.0)];
        let ends = [V2::new(4.0, 0.0), V2::new(5.0, 0.0)];
        let (point, segment) = near_segments(&starts, &ends, &[V2::new(0.0, 0.5)], 0.6);
        assert_eq!(point, vec![0]);
        assert_eq!(segment, vec![0]);
    }

    #[test]
    fn obstacle_order_sorts_by_point_obstacle_and_tiebreak() {
        let order = obstacle_order(&[1, 0, 0, 1], &[0, 2, 1, 0], &[5, 0, 1, 2]);
        assert_eq!(order, vec![2, 1, 3, 0]);
    }
}
