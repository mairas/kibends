//! Distances over (point, segment) pairs: a point and a segment it may be
//! answered by, each segment inflated by its own amount and part of an
//! obstacle. A chain field, the members' view of each other and the moving
//! vias' view search for their pairs and measure them here, all at once.
//! Their smooth distance blends every segment, less the vertex two
//! consecutive segments of one obstacle share (`smooth_of_pairs`), so that
//! it has no kink where a point is equally near two segments of one
//! obstacle.

use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};

use super::smooth::{SMOOTH_REACH, signed_smooth_minimum, smooth_minimum, weighted_sum};
use super::{DEGENERATE_DISTANCE, ROUND_OFF};
use crate::geometry::{FALLBACK_DIRECTION, V2, foot, offset, segment_gap};

/// The nearest inflated segment at each point, and which pair answers it.
#[derive(Debug, Default)]
pub struct NearestEdge {
    pub distance: Vec<f64>,
    pub gradient: Vec<V2>,
    /// The pair that answers each point, None where none is in reach.
    pub edge: Vec<Option<usize>>,
}

/// The smooth distance at each point and its gradient, and what answers
/// it: per pair that has a share at a point, the point, the pair, its share
/// and its unit gradient, in the order of their points.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmoothAnswer {
    pub distance: Vec<f64>,
    pub gradient: Vec<V2>,
    pub point: Vec<usize>,
    pub edge: Vec<usize>,
    pub share: Vec<f64>,
    pub direction: Vec<V2>,
    /// Whether the answer is the first vertex of its pair's segment alone,
    /// which that segment shares with the one before it in its obstacle,
    /// its share negative (`smooth_of_pairs`).
    pub vertex: Vec<bool>,
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
/// `nearest_of_pairs` takes them, pair k's segment part of obstacle[k]: the
/// smooth minimum over every segment, less, where two segments of one
/// obstacle join, the vertex they share, inflated by the lesser of their
/// inflations. The pairs come in the order `obstacle_order` gives. Two
/// segments within their limits join when the first ends where the
/// second starts and only pairs beyond their limits and zero-length
/// segments lie between them, so that the answers do not depend on which
/// pairs beyond the limits a search lists, as long as every segment of one
/// obstacle is inflated alike: a segment inflated less can lie within its
/// limit yet beyond the smooth reach, and still take a join from a
/// zero-length segment inflated more.
/// The edge that answers is its pair's index, a shared vertex by the
/// segment it starts.
///
/// Taking the vertex off makes an obstacle's sum the nearest segment's
/// term wherever both segments' nearest point is that vertex, as outside a
/// bend or beside a straight join, so that the distance there is exact.
/// Where a point's nearest points lie inside two segments, as on the
/// bisector inside a bend, the segments share the point and its distance
/// has a slope, where the nearest segment's alone has a kink. A segment
/// whose nearest point is the vertex, inflated as much as the vertex, has
/// the vertex's term, and both are left out of the sum: the same sum, but
/// beside a straight run of many segments one term instead of two per
/// segment.
///
/// A vertex is no nearer than either of its segments, so the sum is at
/// least its largest segment's term and the distance never farther than
/// the exact one; it is nearer by at most τ·ln m, of the m segments that
/// have a share. Inside one bend that is τ·ln 2, but near the centre of a
/// curve resampled into n segments, whose vertices lie hardly farther than
/// the segments and take little off, nearly τ·ln n.
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
    let (offsets, along_at): (Vec<V2>, Vec<f64>) = (0..point.len())
        .map(|k| foot(query[point[k]], starts[k], along[k]))
        .unzip();
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
    // A segment whose nearest point is the vertex it shares, inflated as
    // much as the vertex, has the vertex's term, so the two cancel; each
    // vertex cancels one segment at most, and each segment meets it at one
    // end only.
    //
    // A segment joins the last one before it of its point and obstacle
    // within the limit, past any zero-length one, which lies on the vertex
    // the two share; a zero-length one only when there is no other. So
    // pairs beyond the limit, which a search may list or not, and the
    // zero-length pieces of repeated vertices change no join.
    let mut segment = vec![true; point.len()];
    // The inflation of the vertex each segment shares at its start.
    let mut shared: Vec<Option<f64>> = vec![None; point.len()];
    let (mut long, mut short): (Option<usize>, Option<usize>) = (None, None);
    for k in 0..point.len() {
        if k > 0 && (point[k - 1] != point[k] || obstacle[k - 1] != obstacle[k]) {
            (long, short) = (None, None);
        }
        if distance[k] == f64::INFINITY {
            continue;
        }
        let before = long.or(short);
        if along[k] == V2::new(0.0, 0.0) {
            short = Some(k);
        } else {
            long = Some(k);
        }
        let Some(j) = before else { continue };
        if (starts[j] + along[j] - starts[k]).hypot() > ROUND_OFF {
            continue;
        }
        let vertex = inflations[j].min(inflations[k]);
        if along_at[k] == 0.0 && inflations[k] == vertex {
            segment[k] = false;
        } else if along_at[j] == 1.0 && inflations[j] == vertex && segment[j] {
            segment[j] = false;
        } else {
            shared[k] = Some(vertex);
        }
    }
    let mut terms = Terms::default();
    for k in 0..point.len() {
        if distance[k] == f64::INFINITY {
            continue;
        }
        if segment[k] {
            terms.push(
                point[k],
                k,
                distance[k],
                unit_gradient(offsets[k], raw[k], along[k]),
                false,
            );
        }
        if let Some(vertex) = shared[k] {
            let from = query[point[k]] - starts[k];
            let length = from.hypot();
            terms.push(
                point[k],
                k,
                length - vertex,
                unit_gradient(from, length, along[k]),
                true,
            );
        }
    }
    let (smooth, share) =
        signed_smooth_minimum(&terms.point, &terms.distance, &terms.vertex, count);
    let mut found = SmoothAnswer {
        distance: smooth,
        gradient: weighted_sum(&terms.point, &share, &terms.direction, count),
        ..SmoothAnswer::default()
    };
    for (n, &s) in share.iter().enumerate() {
        if s != 0.0 {
            found.point.push(terms.point[n]);
            found.edge.push(terms.pair[n]);
            found.share.push(s);
            found.direction.push(terms.direction[n]);
            found.vertex.push(terms.vertex[n]);
        }
    }
    found
}

/// The terms of `smooth_of_pairs`' sums, segments and the vertices they
/// share, each by its point and its pair.
#[derive(Default)]
struct Terms {
    point: Vec<usize>,
    pair: Vec<usize>,
    distance: Vec<f64>,
    direction: Vec<V2>,
    vertex: Vec<bool>,
}

impl Terms {
    fn push(&mut self, point: usize, pair: usize, distance: f64, direction: V2, vertex: bool) {
        self.point.push(point);
        self.pair.push(pair);
        self.distance.push(distance);
        self.direction.push(direction);
        self.vertex.push(vertex);
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

type Boxed = GeomWithData<Rectangle<[f64; 2]>, usize>;

/// A tree over boxes, each by its index, which finds the boxes near a point
/// or a box. It adds no margin of its own, and finds none for a NaN or
/// negative reach.
pub struct BoxTree {
    tree: RTree<Boxed>,
}

impl BoxTree {
    /// Each box by its index and two opposite corners.
    pub fn new(boxes: impl IntoIterator<Item = (usize, V2, V2)>) -> Self {
        let items = boxes
            .into_iter()
            .map(|(k, a, b)| GeomWithData::new(Rectangle::from_corners(a.array(), b.array()), k))
            .collect();
        Self {
            tree: RTree::bulk_load(items),
        }
    }

    /// The indices of the boxes within `reach` of `p`, ascending.
    pub fn within(&self, p: V2, reach: f64) -> Vec<usize> {
        if reach.is_nan() || reach < 0.0 {
            return Vec::new();
        }
        let mut found: Vec<usize> = self
            .tree
            .locate_within_distance(p.array(), reach * reach)
            .map(|g| g.data)
            .collect();
        found.sort_unstable();
        found
    }

    /// The indices of the boxes that meet the box from `low` to `high`
    /// grown by `reach`, ascending.
    pub fn meeting(&self, low: V2, high: V2, reach: f64) -> Vec<usize> {
        if reach.is_nan() || reach < 0.0 {
            return Vec::new();
        }
        let envelope = AABB::from_corners(
            [low.x - reach, low.y - reach],
            [high.x + reach, high.y + reach],
        );
        let mut found: Vec<usize> = self
            .tree
            .locate_in_envelope_intersecting(envelope)
            .map(|g| g.data)
            .collect();
        found.sort_unstable();
        found
    }

    /// The boxes from the nearest `p` outward, each by its index and its
    /// squared distance.
    pub fn nearest_first(&self, p: V2) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.tree
            .nearest_neighbor_iter_with_distance_2(p.array())
            .map(|(g, d2)| (g.data, d2))
    }
}

/// A tree over segments' bounding boxes, which finds every segment that may
/// come within a distance of a point or of a box. A segment lies in its
/// box, so a segment near a query has its box near it too, and a long
/// segment is found only near its own box.
pub struct SegmentTree {
    boxes: BoxTree,
}

impl SegmentTree {
    /// The segments from starts[k] to ends[k].
    pub fn new(starts: &[V2], ends: &[V2]) -> Self {
        Self {
            boxes: BoxTree::new(
                starts
                    .iter()
                    .zip(ends)
                    .enumerate()
                    .map(|(k, (&a, &b))| (k, a, b)),
            ),
        }
    }

    /// The segments, by index ascending, whose boxes lie within `radius`
    /// and DEGENERATE_DISTANCE of `p`: among them every one that comes
    /// within `radius` of it. None for a NaN or negative radius.
    pub fn near(&self, p: V2, radius: f64) -> Vec<usize> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        self.boxes.within(p, radius + DEGENERATE_DISTANCE)
    }

    /// The segments, by index ascending, whose boxes meet the box from
    /// `low` to `high` grown by `radius` and DEGENERATE_DISTANCE: among
    /// them every one that comes within `radius` of a point in it. None
    /// for a NaN or negative radius.
    pub fn near_box(&self, low: V2, high: V2, radius: f64) -> Vec<usize> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        self.boxes.meeting(low, high, radius + DEGENERATE_DISTANCE)
    }
}

/// (point, segment) index pairs, by point then segment, among them every
/// pair whose segment from starts[k] to ends[k] comes within `radius` of
/// points[i], found with one tree over the segments' bounding boxes.
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
    use crate::field::smooth::SMOOTH_WIDTH;
    use crate::testing::Random;

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

    /// How far `to_polyline` looks from its point, mm: beyond every
    /// polyline its tests take.
    const LOOK: f64 = 10.0;

    /// The smooth distance from `query` to one obstacle, the polyline
    /// through `vertices` inflated by `inflation`.
    fn to_polyline(query: V2, vertices: &[V2], inflation: f64) -> SmoothAnswer {
        to_widened(query, vertices, &vec![inflation; vertices.len() - 1])
    }

    /// `to_polyline` with each edge inflated by its own of `inflations`.
    fn to_widened(query: V2, vertices: &[V2], inflations: &[f64]) -> SmoothAnswer {
        let edges = vertices.len() - 1;
        let along: Vec<V2> = vertices.windows(2).map(|w| w[1] - w[0]).collect();
        smooth_of_pairs(
            &[query],
            &vec![0; edges],
            &vertices[..edges],
            &along,
            inflations,
            &vec![LOOK; edges],
            &vec![0; edges],
        )
    }

    /// The exact distance from `query` to the polyline through `vertices`,
    /// each edge inflated by its own of `inflations`.
    fn exact(query: V2, vertices: &[V2], inflations: &[f64]) -> f64 {
        vertices
            .windows(2)
            .zip(inflations)
            .map(|(w, &inflation)| offset(query, w[0], w[1] - w[0]).hypot() - inflation)
            .fold(f64::INFINITY, f64::min)
    }

    /// How much nearer than the exact distance the smooth one reads at
    /// `query`, with each edge inflated by its own of `inflations`.
    fn nearer(query: V2, vertices: &[V2], inflations: &[f64]) -> f64 {
        exact(query, vertices, inflations) - to_widened(query, vertices, inflations).distance[0]
    }

    #[test]
    fn a_point_on_the_bisector_inside_a_bend_has_a_slope() {
        // Inside the bend, its feet on both edges, so that the nearest
        // edge's distance has a kink there: its slope along x jumps by
        // twice the sine of the edges' angle to the x axis.
        let bend = [V2::new(-2.0, 1.5), V2::new(0.0, 1.0), V2::new(2.0, 1.5)];
        let on = V2::new(0.0, 2.0);
        let inflation = 0.1;
        let step = 1e-7;
        let at = |dx: f64| to_polyline(V2::new(on.x + dx, on.y), &bend, inflation).distance[0];
        let right = (at(step) - at(0.0)) / step;
        let left = (at(0.0) - at(-step)) / step;
        let nearest_only_jump = 2.0 * libm::sin(libm::atan(0.25));
        let curvature_part = 1e-5;
        assert!((right - left).abs() < curvature_part * nearest_only_jump);
        let found = to_polyline(on, &bend, inflation);
        assert!(found.gradient[0].x.abs() < 1e-15);
        // Nearer than the exact distance, by at most τ·ln 2 of two equal
        // terms.
        let exact = 1.0 / libm::hypot(1.0, 0.25) - inflation;
        let nearer = exact - found.distance[0];
        assert!(nearer > 0.0 && nearer <= SMOOTH_WIDTH * libm::log(2.0));
    }

    #[test]
    fn a_point_nearest_a_shared_vertex_takes_its_distance_alone() {
        // Outside the bend and beside a straight join, where both edges'
        // nearest point is the vertex they share.
        let bend = [V2::new(-2.0, 1.5), V2::new(0.0, 1.0), V2::new(2.0, 1.5)];
        let found = to_polyline(V2::new(0.0, 0.0), &bend, 0.25);
        assert!((found.distance[0] - 0.75).abs() < 1e-15);
        assert!((found.gradient[0] - V2::new(0.0, -1.0)).hypot() < 1e-15);
        let straight = [V2::new(-1.0, 0.0), V2::new(0.0, 0.0), V2::new(1.0, 0.0)];
        let found = to_polyline(V2::new(0.0, 1.0), &straight, 0.25);
        assert!((found.distance[0] - 0.75).abs() < 1e-15);
        assert!((found.share.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_segment_nearest_at_the_vertex_it_shares_cancels_with_it() {
        let bend = [V2::new(-2.0, 1.5), V2::new(0.0, 1.0), V2::new(2.0, 1.5)];
        let found = to_polyline(V2::new(0.0, 0.0), &bend, 0.25);
        assert_eq!(found.edge, vec![0]);
        assert_eq!(found.share, vec![1.0]);
    }

    #[test]
    fn a_shared_vertex_inside_a_bend_takes_its_share_off() {
        let bend = [V2::new(-2.0, 1.5), V2::new(0.0, 1.0), V2::new(2.0, 1.5)];
        let found = to_polyline(V2::new(0.0, 2.0), &bend, 0.1);
        // Both edges and, taken off, the vertex they share.
        assert_eq!(found.edge, vec![0, 1, 1]);
        assert_eq!(found.vertex, vec![false, false, true]);
        assert!(found.share[2] < 0.0);
    }

    #[test]
    fn at_a_width_change_the_distance_is_no_farther_and_within_tau_ln_2() {
        // Beside the vertex of a straight run, with both feet on it or one
        // inside an edge, and around a bend, inside with both feet inside
        // its edges and outside with both on its vertex; narrower first and
        // wider first, so that each cancellation, and its refusal where the
        // segment is inflated more than the vertex, is taken.
        let straight = [V2::new(-1.0, 0.0), V2::new(0.0, 0.0), V2::new(1.0, 0.0)];
        let bend = [V2::new(-2.0, 1.5), V2::new(0.0, 1.0), V2::new(2.0, 1.5)];
        let beside = [V2::new(0.0, 1.0), V2::new(-0.01, 1.0), V2::new(0.01, 1.0)];
        let around = [V2::new(0.0, 2.0), V2::new(0.05, 2.0), V2::new(0.0, 0.0)];
        let two = SMOOTH_WIDTH * libm::log(2.0);
        for inflations in [[0.25, 0.26], [0.26, 0.25]] {
            for (vertices, points) in [(&straight, &beside), (&bend, &around)] {
                for &query in points {
                    let by = nearer(query, vertices, &inflations);
                    assert!(
                        by >= -ROUND_OFF && by <= two,
                        "{query:?} {inflations:?} {by}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_centre_of_a_resampled_curve_reads_at_most_tau_ln_n_nearer() {
        // A half circle in edges of about 0.1 mm, whose vertices lie only
        // s²/8R farther from the centre than its edges, so that they take
        // little off and every edge has a share.
        let radius = 1.0;
        let edges = 31;
        let arc: Vec<V2> = (0..=edges)
            .map(|k| {
                let angle = core::f64::consts::PI * k as f64 / edges as f64;
                V2::new(radius * libm::cos(angle), radius * libm::sin(angle))
            })
            .collect();
        let by = nearer(V2::new(0.0, 0.0), &arc, &vec![0.1; edges]);
        assert!(by >= 0.0 && by <= SMOOTH_WIDTH * libm::log(edges as f64));
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
        let expected = 1.0 - super::super::smooth::SMOOTH_WIDTH * libm::log(2.0);
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

    /// A 30 mm segment along y = 0, then short ones along y = 9 and y = 20.
    fn one_long_and_short_segments() -> SegmentTree {
        SegmentTree::new(
            &[V2::new(0.0, 0.0), V2::new(14.0, 9.0), V2::new(20.0, 20.0)],
            &[V2::new(30.0, 0.0), V2::new(16.0, 9.0), V2::new(21.0, 20.0)],
        )
    }

    #[test]
    fn a_long_segment_far_from_the_point_is_not_near() {
        let tree = one_long_and_short_segments();
        assert_eq!(tree.near(V2::new(15.0, 10.0), 1.5), vec![1]);
    }

    #[test]
    fn a_long_segment_passing_close_is_near_though_its_middle_is_far() {
        let tree = SegmentTree::new(&[V2::new(0.0, 0.5)], &[V2::new(24.0, 0.5)]);
        assert_eq!(tree.near(V2::new(0.0, 0.0), 1.0), vec![0]);
    }

    #[test]
    fn a_segment_of_no_length_is_near_within_the_radius_only() {
        let tree = SegmentTree::new(&[V2::new(2.0, 0.0)], &[V2::new(2.0, 0.0)]);
        assert_eq!(tree.near(V2::default(), 2.0), vec![0]);
        assert!(tree.near(V2::default(), 1.99).is_empty());
    }

    #[test]
    fn the_box_query_lists_the_boxes_that_meet_the_grown_box_ascending() {
        let tree = SegmentTree::new(
            &[
                V2::new(0.0, 0.0),
                V2::new(3.0, 12.5),
                V2::new(0.0, 12.0),
                V2::new(20.0, 20.0),
            ],
            &[
                V2::new(30.0, 0.0),
                V2::new(3.0, 13.0),
                V2::new(1.0, 12.0),
                V2::new(21.0, 20.0),
            ],
        );
        let found = tree.near_box(V2::new(0.5, 11.0), V2::new(2.5, 11.5), 1.0);
        assert_eq!(found, vec![1, 2]);
    }

    #[test]
    fn a_nan_or_negative_radius_finds_nothing() {
        let tree = one_long_and_short_segments();
        let (low, high) = (V2::new(14.0, 8.0), V2::new(16.0, 10.0));
        assert!(tree.near(V2::new(15.0, 9.0), f64::NAN).is_empty());
        assert!(tree.near(V2::new(15.0, 9.0), -1.0).is_empty());
        assert!(tree.near_box(low, high, f64::NAN).is_empty());
        assert!(tree.near_box(low, high, -1.0).is_empty());
    }

    #[test]
    fn a_box_tree_finds_nothing_for_a_nan_or_negative_reach() {
        let tree = BoxTree::new([(7, V2::new(0.0, 0.0), V2::new(1.0, 1.0))]);
        let (low, high) = (V2::new(0.0, 0.0), V2::new(1.0, 1.0));
        assert_eq!(tree.within(V2::new(0.5, 0.5), 0.0), vec![7]);
        assert_eq!(tree.meeting(low, high, 0.0), vec![7]);
        for reach in [f64::NAN, -1.0] {
            assert!(tree.within(V2::new(0.5, 0.5), reach).is_empty());
            assert!(tree.meeting(low, high, reach).is_empty());
        }
    }

    #[test]
    fn near_lists_every_segment_within_the_radius_ascending_on_random_segments() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let starts: Vec<V2> = (0..200).map(|_| random.point(20.0)).collect();
        let ends: Vec<V2> = starts
            .iter()
            .map(|&a| {
                let length = random.uniform(0.0, 1.0);
                a + length * length * length * V2::new(random.uniform(-15.0, 15.0), 0.0)
                    + V2::new(0.0, random.uniform(-3.0, 3.0))
            })
            .collect();
        let tree = SegmentTree::new(&starts, &ends);
        for _ in 0..200 {
            let p = random.point(20.0);
            let radius = random.uniform(0.0, 2.0);
            let found = tree.near(p, radius);
            assert!(found.windows(2).all(|w| w[0] < w[1]));
            for k in 0..starts.len() {
                if segment_gap(p, V2::default(), starts[k], ends[k] - starts[k]) <= radius {
                    assert!(found.contains(&k), "segment {k} missing at {p:?}");
                }
            }
        }
    }

    #[test]
    fn obstacle_order_sorts_by_point_obstacle_and_tiebreak() {
        let order = obstacle_order(&[1, 0, 0, 1], &[0, 2, 1, 0], &[5, 0, 1, 2]);
        assert_eq!(order, vec![2, 1, 3, 0]);
    }
}
