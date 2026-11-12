//! A moving via's free distances, judged obstacle by obstacle.
//!
//! A via's clearance is judged obstacle by obstacle (`newly_inside`) in
//! the optimizer's checks, so that a via inside one obstacle is still kept
//! out of the others: each member, each other via, and each obstacle of its
//! static copper and of each part of its stops, connected copper of one net
//! on one layer as one (`StaticFree`).

use std::sync::Arc;

use super::sight::{ObstacleDistances, ViaSight};
use crate::field::Field;
use crate::field::edges::StaticEdges;
use crate::field::search::static_distances;
use crate::geometry::V2;
use crate::numeric::minimum;

/// (via, obstacle) pairs of what stands still for each via, in ascending
/// order of `key`, and each via's exact distance to its obstacle. A via's
/// obstacles are those of each of its parts, numbered one part after
/// another: of a field that gives edges (`Field::static_parts`), each as a
/// clearance check judges it, its connected copper of one net on one layer
/// or other item (`StaticEdges::clearance_obstacles`), and the whole of one
/// that does not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticFree {
    /// Each pair's via times the most obstacles any via has, plus its
    /// obstacle.
    pub key: Vec<usize>,
    pub via: Vec<usize>,
    pub distance: Vec<f64>,
}

/// The free distance of each via to each obstacle: each of what stands
/// still, and each member and via its sight finds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FreeDistances {
    pub fixed: StaticFree,
    pub by_obstacle: ObstacleDistances,
}

/// What stands still for a via, in parts, each None where it has none.
pub type StaticParts = Vec<Option<Arc<dyn Field>>>;

/// The obstacles of what stands still for one via that are near it, in
/// ascending order, their distances, and how many obstacles there are in
/// all.
struct ViaStatic {
    obstacle: Vec<usize>,
    distance: Vec<f64>,
    obstacles: usize,
}

/// The obstacles of `fields` nearer than `below` to a via at `at`.
fn static_free(fields: &StaticParts, at: V2, below: f64) -> ViaStatic {
    let mut found = ViaStatic {
        obstacle: Vec::new(),
        distance: Vec::new(),
        obstacles: 0,
    };
    for field in fields.iter().flatten() {
        let (obstacle, distance, count) = match field.static_parts() {
            None => {
                let (distance, _) = field.signed_distance(&[at], f64::INFINITY);
                (vec![0], distance, 1)
            }
            Some(parts) => {
                let edges: Vec<&StaticEdges> = parts.iter().map(Arc::as_ref).collect();
                let near = static_distances(&edges, &[at], below);
                let count = edges.iter().map(|e| e.obstacle_count()).sum();
                (near.obstacle, near.distance, count)
            }
        };
        for (o, d) in obstacle.into_iter().zip(distance) {
            if d < below {
                found.obstacle.push(o + found.obstacles);
                found.distance.push(d);
            }
        }
        found.obstacles += count;
    }
    found
}

/// The free distance of each via to each obstacle nearer than `below`: to
/// each of its `statics` (`StaticFree`), and to each of the chains or
/// members at `polylines` and the vias at `positions` that `sight` finds.
/// Without a finite `below`, of each part of its statics at least the
/// nearest obstacle.
pub fn free_distances(
    statics: &[StaticParts],
    sight: &ViaSight,
    polylines: &[&[V2]],
    positions: &[V2],
    below: f64,
) -> FreeDistances {
    let found: Vec<ViaStatic> = statics
        .iter()
        .zip(positions)
        .map(|(fields, &at)| static_free(fields, at, below))
        .collect();
    let stride = found.iter().map(|f| f.obstacles).max().unwrap_or(0);
    let mut fixed = StaticFree::default();
    for (v, f) in found.iter().enumerate() {
        for (&o, &d) in f.obstacle.iter().zip(&f.distance) {
            fixed.key.push(v * stride + o);
            fixed.via.push(v);
            fixed.distance.push(d);
        }
    }
    FreeDistances {
        fixed,
        by_obstacle: sight.obstacle_distances(polylines, positions, below),
    }
}

/// Which pairs, by their keys `now_key` at distances `now`, are newly
/// inside by more than `tolerance`, or more than `slack` deeper inside than
/// the pair of the same key among `start_key` at `start`, ascending.
pub fn deeper_inside(
    now_key: &[usize],
    now: &[f64],
    start_key: &[usize],
    start: &[f64],
    tolerance: f64,
    slack: f64,
) -> Vec<bool> {
    now_key
        .iter()
        .zip(now)
        .map(|(&key, &distance)| {
            let at = start_key.partition_point(|&k| k < key);
            let before = if start_key.get(at) == Some(&key) {
                start[at]
            } else {
                f64::INFINITY
            };
            distance < minimum(before - slack, -tolerance)
        })
        .collect()
}

/// Which of the (via, obstacle) pairs of what stands still for the vias and
/// of the members and vias they see, at `now`, are newly inside by more
/// than `tolerance`, or more than `slack` deeper inside than at `start`.
/// Each obstacle is judged by the via's nearest distance to it against that
/// distance at the start, so that a via inside one is still kept out of
/// the others; an edge of the obstacle other than its nearest may move
/// deeper while the nearest stays the deepest. `start` holds every pair
/// inside by more than `tolerance`.
pub fn newly_inside(
    now: &FreeDistances,
    start: &FreeDistances,
    tolerance: f64,
    slack: f64,
) -> (Vec<bool>, Vec<bool>) {
    (
        deeper_inside(
            &now.fixed.key,
            &now.fixed.distance,
            &start.fixed.key,
            &start.fixed.distance,
            tolerance,
            slack,
        ),
        deeper_inside(
            &now.by_obstacle.key,
            &now.by_obstacle.distance,
            &start.by_obstacle.key,
            &start.by_obstacle.distance,
            tolerance,
            slack,
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_newly_inside_or_deeper_inside_is_judged() {
        let inside = deeper_inside(
            &[1, 2, 5],
            &[-0.2, -0.1, -0.3],
            &[2, 5],
            &[-0.05, -0.3],
            1e-6,
            1e-9,
        );
        // Key 1 is new; key 2 went deeper; key 5 stayed as deep.
        assert_eq!(inside, vec![true, true, false]);
    }

    #[test]
    fn a_field_without_edges_is_one_obstacle() {
        let disc: Arc<dyn Field> = Arc::new(crate::testing::Disc {
            cx: 0.0,
            cy: 0.0,
            radius: 1.0,
        });
        let sight = ViaSight::new(&[vec![]], &[vec![]], 0);
        let free = free_distances(
            &[vec![Some(disc), None]],
            &sight,
            &[],
            &[V2::new(0.5, 0.0)],
            0.0,
        );
        assert_eq!(free.fixed.key, vec![0]);
        assert_eq!(free.fixed.distance, vec![-0.5]);
    }
}
