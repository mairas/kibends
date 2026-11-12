//! What each moving via sees of the members and the other vias.
//!
//! A moving via sees whole members and the other vias (`ViaSight`): one
//! tree over their edges and centres answers every via, each pair inflated
//! by the distance its via keeps from that member or via, each member and
//! each via an obstacle of the smooth distance. A member is seen as its
//! own polyline, a section as its centreline.

use crate::field::pairs::{near_segments, nearest_of_pairs, smooth_of_pairs};
use crate::field::smooth::smooth_reach;
use crate::geometry::V2;
use crate::numeric::maximum;
use crate::seen::ROUND_OFF;

/// What each moving via sees, for its energy and its free distance: every
/// edge of the members it names, and the other vias it names as edges of
/// no length, each inflated by the distance it gives; searched with one
/// tree over the edges of every member some via sees and every via.
#[derive(Clone, Debug)]
pub struct ViaSight {
    /// Per via, the distance it keeps from each member it sees.
    rings: Vec<Vec<Option<f64>>>,
    /// Per via, the distance it keeps from each other via it sees.
    apart: Vec<Vec<Option<f64>>>,
    /// Each via's largest inflation.
    within: Vec<f64>,
    /// The members some via sees, ascending.
    seen: Vec<usize>,
}

/// (via, obstacle) pairs, in ascending order of `key`, and each via's exact
/// distance to its obstacle, a member or a via. Of a member, the first
/// vertex of its edge nearest the via comes too.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObstacleDistances {
    /// Each pair's via times the number of obstacles, plus its obstacle.
    pub key: Vec<usize>,
    pub via: Vec<usize>,
    /// A member by its index, a via by the number of members plus its.
    pub obstacle: Vec<usize>,
    pub distance: Vec<f64>,
    /// The member's edge's first vertex, or the via's index.
    pub vertex: Vec<usize>,
}

/// Each moving via's smooth distance and its gradient, and what answers it:
/// per member or via that has a share, the via it answers, the member and
/// its edge's first vertex, or None and the via, where that edge starts and
/// ends, its share and its unit gradient.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViaSmooth {
    pub distance: Vec<f64>,
    pub gradient: Vec<V2>,
    pub point: Vec<usize>,
    pub member: Vec<Option<usize>>,
    pub vertex: Vec<usize>,
    pub starts: Vec<V2>,
    pub ends: Vec<V2>,
    pub share: Vec<f64>,
    pub direction: Vec<V2>,
}

/// What the vias see: every edge some via may see, from a to b, each a
/// member's edge by its first vertex or a via, and the (via, edge) pairs
/// within `limit` of each via that it sees, and their inflations.
struct View {
    at: Vec<V2>,
    a: Vec<V2>,
    b: Vec<V2>,
    member: Vec<Option<usize>>,
    vertex: Vec<usize>,
    point: Vec<usize>,
    edge: Vec<usize>,
    inflation: Vec<f64>,
    limit: Vec<f64>,
}

impl View {
    /// Where each pair's edge starts and the way to its end.
    fn shapes(&self, edges: &[usize]) -> (Vec<V2>, Vec<V2>) {
        edges
            .iter()
            .map(|&e| (self.a[e], self.b[e] - self.a[e]))
            .unzip()
    }

    /// Each pair's obstacle: its member, or after `members` its via.
    fn obstacle(&self, edge: usize, members: usize) -> usize {
        self.member[edge].unwrap_or(members + self.vertex[edge])
    }
}

impl ViaSight {
    /// Each via, by its index, seeing the members its `rings` names and the
    /// vias its `apart` names, at those distances, among `members`.
    pub fn new(rings: &[Vec<(usize, f64)>], apart: &[Vec<(usize, f64)>], members: usize) -> Self {
        let vias = rings.len();
        let mut seen_rings = vec![vec![None; members]; vias];
        let mut seen_apart = vec![vec![None; vias]; vias];
        let mut within = Vec::with_capacity(vias);
        for (k, (seen, kept)) in rings.iter().zip(apart).enumerate() {
            let mut widest = 0.0f64;
            for &(j, ring) in seen {
                seen_rings[k][j] = Some(ring);
                widest = if ring > widest { ring } else { widest };
            }
            for &(u, distance) in kept {
                seen_apart[k][u] = Some(distance);
                widest = if distance > widest { distance } else { widest };
            }
            within.push(widest);
        }
        let seen = (0..members)
            .filter(|&j| seen_rings.iter().any(|row| row[j].is_some()))
            .collect();
        Self {
            rings: seen_rings,
            apart: seen_apart,
            within,
            seen,
        }
    }

    fn view(&self, polylines: &[&[V2]], positions: &[V2], reach: f64) -> View {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut member = Vec::new();
        let mut vertex = Vec::new();
        for &j in &self.seen {
            let points = polylines[j];
            for v in 0..points.len().saturating_sub(1) {
                a.push(points[v]);
                b.push(points[v + 1]);
                member.push(Some(j));
                vertex.push(v);
            }
        }
        for (k, &at) in positions.iter().enumerate() {
            a.push(at);
            b.push(at);
            member.push(None);
            vertex.push(k);
        }
        let limit: Vec<f64> = self.within.iter().map(|w| reach + w).collect();
        let radius = limit.iter().copied().fold(0.0, maximum) + ROUND_OFF;
        let (found_point, found_edge) = near_segments(&a, &b, positions, radius);
        let mut point = Vec::new();
        let mut edge = Vec::new();
        let mut inflation = Vec::new();
        for (p, e) in found_point.into_iter().zip(found_edge) {
            let kept = match member[e] {
                Some(j) => self.rings[p][j],
                None => self.apart[p][vertex[e]],
            };
            if let Some(kept) = kept {
                point.push(p);
                edge.push(e);
                inflation.push(kept);
            }
        }
        View {
            at: positions.to_vec(),
            a,
            b,
            member,
            vertex,
            point,
            edge,
            inflation,
            limit,
        }
    }

    /// Each via's exact distance, in order, to the nearest of what it sees,
    /// with the members at `polylines` and the vias at `positions`, as a
    /// chain field of the via's own view with `reach` gives it.
    pub fn distances(&self, polylines: &[&[V2]], positions: &[V2], reach: f64) -> Vec<f64> {
        let view = self.view(polylines, positions, reach);
        let (starts, along) = view.shapes(&view.edge);
        let limit: Vec<f64> = view.point.iter().map(|&p| view.limit[p]).collect();
        nearest_of_pairs(
            &view.at,
            &view.point,
            &starts,
            &along,
            &view.inflation,
            &limit,
        )
        .distance
    }

    /// Each via's exact distance to each member and each via it sees, as
    /// `distances` gives it over that one alone, where it is below `below`.
    pub fn obstacle_distances(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        below: f64,
    ) -> ObstacleDistances {
        let view = self.view(polylines, positions, below);
        let (starts, along) = view.shapes(&view.edge);
        let query: Vec<V2> = view.point.iter().map(|&p| view.at[p]).collect();
        let pairs: Vec<usize> = (0..view.point.len()).collect();
        let limit: Vec<f64> = view.point.iter().map(|&p| view.limit[p]).collect();
        let distance =
            nearest_of_pairs(&query, &pairs, &starts, &along, &view.inflation, &limit).distance;
        let count = polylines.len() + view.at.len();
        let mut near: Vec<(usize, f64, usize, usize, usize)> = (0..pairs.len())
            .filter(|&k| distance[k] < below)
            .map(|k| {
                let e = view.edge[k];
                let via = view.point[k];
                let obstacle = view.obstacle(e, polylines.len());
                (
                    via * count + obstacle,
                    distance[k],
                    via,
                    obstacle,
                    view.vertex[e],
                )
            })
            .collect();
        // By key, then distance, NaN last, as numpy's lexsort orders them.
        near.sort_by(|x, y| {
            x.0.cmp(&y.0).then_with(|| {
                x.1.partial_cmp(&y.1)
                    .unwrap_or(x.1.is_nan().cmp(&y.1.is_nan()))
            })
        });
        let mut found = ObstacleDistances::default();
        for (n, &(key, distance, via, obstacle, vertex)) in near.iter().enumerate() {
            if n > 0 && near[n - 1].0 == key {
                continue;
            }
            found.key.push(key);
            found.via.push(via);
            found.obstacle.push(obstacle);
            found.distance.push(distance);
            found.vertex.push(vertex);
        }
        found
    }

    /// Each via's smooth distance, in order, to what it sees, each member
    /// and each via an obstacle, and what answers it, as a chain field of
    /// the via's own view with `reach` gives it.
    pub fn smooth_answers(&self, polylines: &[&[V2]], positions: &[V2], reach: f64) -> ViaSmooth {
        let view = self.view(polylines, positions, smooth_reach(reach));
        let members = polylines.len();
        let mut order: Vec<usize> = (0..view.point.len()).collect();
        order.sort_unstable_by_key(|&k| {
            (
                view.point[k],
                view.obstacle(view.edge[k], members),
                view.edge[k],
            )
        });
        let point: Vec<usize> = order.iter().map(|&k| view.point[k]).collect();
        let edge: Vec<usize> = order.iter().map(|&k| view.edge[k]).collect();
        let (starts, along) = view.shapes(&edge);
        let inflation: Vec<f64> = order.iter().map(|&k| view.inflation[k]).collect();
        let limit: Vec<f64> = point.iter().map(|&p| view.limit[p]).collect();
        let obstacle: Vec<usize> = edge.iter().map(|&e| view.obstacle(e, members)).collect();
        let found = smooth_of_pairs(
            &view.at, &point, &starts, &along, &inflation, &limit, &obstacle,
        );
        let answering: Vec<usize> = found.edge.iter().map(|&k| edge[k]).collect();
        ViaSmooth {
            distance: found.distance,
            gradient: found.gradient,
            point: found.point,
            member: answering.iter().map(|&e| view.member[e]).collect(),
            vertex: answering.iter().map(|&e| view.vertex[e]).collect(),
            starts: answering.iter().map(|&e| view.a[e]).collect(),
            ends: answering.iter().map(|&e| view.b[e]).collect(),
            share: found.share,
            direction: found.direction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_via_sees_the_members_and_vias_it_names_and_not_itself() {
        // Via 0 sees member 0 at 0.4 and via 1 at 0.6; via 1 sees nothing.
        let sight = ViaSight::new(&[vec![(0, 0.4)], vec![]], &[vec![(1, 0.6)], vec![]], 2);
        let line = [V2::new(-1.0, 1.0), V2::new(1.0, 1.0)];
        let other = [V2::new(-1.0, -1.0), V2::new(1.0, -1.0)];
        let polylines: [&[V2]; 2] = [&line, &other];
        let positions = [V2::new(0.0, 0.0), V2::new(2.0, 0.0)];
        let distances = sight.distances(&polylines, &positions, 10.0);
        assert_eq!(distances[0], 0.6);
        assert_eq!(distances[1], f64::INFINITY);
        let near = sight.obstacle_distances(&polylines, &positions, 10.0);
        assert_eq!(near.via, vec![0, 0]);
        // Member 0, then via 1 numbered after the two members.
        assert_eq!(near.obstacle, vec![0, 3]);
        assert_eq!(near.distance, vec![0.6, 2.0 - 0.6]);
    }
}
