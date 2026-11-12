//! What the members of a problem see of each other, searched for all of
//! them at once.
//!
//! A member sees the edges the others show (`members::Member::shown`) by a
//! rule with sparse exceptions (`Seen`). By the rule it sees every edge, on
//! its layer, of a track of a net it does not ignore, inflated by the
//! distance the two kinds keep, a kind being a width and a clearance
//! (`Seen::table`). A chain ignores its own net and the other net of its
//! diff pair, a coupled section its two nets. The exceptions are edges,
//! each inflated by its own amount (`Pieces`): a member sees the vias near
//! it with their rings.
//!
//! Every question about what the members see, the smooth distance from a
//! point for the energy, each track and each via an obstacle
//! (`field::smooth::smooth_minimum`), how far a point lies from the
//! nearest piece for the checks, or whether a segment enters one, is
//! answered for all members of a layer with one tree over the edges they
//! see: each pair of a point and an edge stands for the pieces of that edge
//! the point's member sees, by the rule or its exceptions. The pairs within
//! reach are those a tree over the member's own pieces would find,
//! measured the same way, so the answers are the same as a field of each
//! member's own pieces gives. A distance wanted however far is found
//! beyond that tree from fields that hold only what a member sees
//! (`Seen::distances`), so that many edges a member does not see, such as
//! its own net's, cost nothing.
//!
//! What the members and moving vias see of static copper is searched once
//! per asker and kept between evaluations (`neighbours::StaticList`), as
//! `neighbours::NeighbourList` keeps what the members see of each other.

use std::collections::HashMap;

use crate::field::pairs::{
    SmoothAnswer, candidate_pairs, near_segments, nearest_of_pairs, pair_depths, pairs_inside,
    smooth_of_pairs,
};
use crate::field::smooth::smooth_reach;
use crate::geometry::V2;
use crate::neighbours::NeighbourList;
use crate::numeric::maximum;

/// The trees are searched a little beyond the limits, so that rounding in
/// a sum of them drops no pair that the exact distance keeps.
pub(crate) const ROUND_OFF: f64 = 1e-9;

/// A track a member shows the others.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    /// Its first vertex, numbered across what every member shows, one
    /// after another, then two for each via (`members::Members::shown_flat`).
    pub first: usize,
    pub vertices: usize,
    pub layer: String,
    /// Its net, numbered.
    pub net: usize,
    /// Its width and clearance, numbered as `Seen::table` takes them.
    pub kind: usize,
}

/// A member as it sees the others.
#[derive(Clone, Debug, PartialEq)]
pub struct Viewer {
    pub layer: String,
    pub net: usize,
    pub kind: usize,
    /// The nets besides its own whose tracks it does not see: the other net
    /// of a chain's diff pair, a section's two. Its own tracks are of its
    /// net or of these.
    pub exempt: Vec<usize>,
}

/// Edges, each inflated by its own amount.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pieces {
    /// Each piece's edge's first vertex, numbered as `Track::first`.
    pub vertex: Vec<usize>,
    pub inflation: Vec<f64>,
}

/// (point, piece) pairs, and the inflation each pair's member sees its
/// piece with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pairs {
    pub point: Vec<usize>,
    pub piece: Vec<usize>,
    pub inflation: Vec<f64>,
}

impl Pairs {
    fn push(&mut self, point: usize, piece: usize, inflation: f64) {
        self.point.push(point);
        self.piece.push(piece);
        self.inflation.push(inflation);
    }

    fn taken(&self, order: impl IntoIterator<Item = usize>) -> Pairs {
        let mut pairs = Pairs::default();
        for k in order {
            pairs.push(self.point[k], self.piece[k], self.inflation[k]);
        }
        pairs
    }

    pub fn len(&self) -> usize {
        self.point.len()
    }

    pub fn is_empty(&self) -> bool {
        self.point.is_empty()
    }
}

/// Whether each edge of a member stays out of the pieces it sees, and what
/// those that do not enter: per (edge, piece) pair, the edge, the first
/// vertex of the piece's edge, numbered as `Track::first`, so that a via's
/// is its first vertex in what the members show, the piece's obstacle, its
/// track or its via (`Seen::obstacle`), and how deep the edge is inside the
/// piece (`field::pairs::pair_depths`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgesClear {
    pub clear: Vec<bool>,
    pub edge: Vec<usize>,
    pub vertex: Vec<usize>,
    pub obstacle: Vec<usize>,
    pub depth: Vec<f64>,
}

/// The members on one layer, the edges they see and what decides which
/// pieces of them each sees.
#[derive(Clone, Debug)]
pub(crate) struct SeenLayer {
    pub(crate) members: Vec<usize>,
    /// Each edge some member sees, by its first vertex, ascending.
    pub(crate) edges: Vec<usize>,
    /// Each edge's piece covering all of it, by its index in `Seen`.
    whole: Vec<usize>,
    /// Each edge's track's net and kind, None for a via.
    net: Vec<Option<usize>>,
    kind: Vec<Option<usize>>,
    /// Per member, its net, its kind and the nets besides its own whose
    /// tracks it does not see.
    viewer_net: Vec<usize>,
    viewer_kind: Vec<usize>,
    exempt: Vec<Vec<usize>>,
    /// The members' exceptions, each at member * edges + the position of
    /// its edge, ascending, each one's piece by its index in `Seen` and its
    /// inflation.
    slot: Vec<usize>,
    piece: Vec<usize>,
    inflation: Vec<f64>,
}

/// What each member of a problem sees of the others at one set of shapes,
/// and one search per layer that answers every member on it.
#[derive(Clone, Debug)]
pub struct Seen {
    table: Vec<Vec<f64>>,
    /// Each member's largest inflation.
    within: Vec<f64>,
    pub(crate) layers: Vec<SeenLayer>,
    /// Each piece's edge's first vertex, numbered as `Track::first`.
    vertex: Vec<usize>,
    /// Each piece's obstacle for the smooth minimum: its track, or its via.
    obstacle: Vec<usize>,
}

/// The `queries` of `members` one after another, each one's member by its
/// place among them, and where each member's lie.
fn gathered(queries: &[&[V2]], members: &[usize]) -> (Vec<V2>, Vec<usize>, Vec<(usize, usize)>) {
    let mut points = Vec::new();
    let mut owner = Vec::new();
    let mut bounds = Vec::with_capacity(members.len());
    for (place, &i) in members.iter().enumerate() {
        let first = points.len();
        points.extend_from_slice(queries[i]);
        owner.extend(std::iter::repeat_n(place, queries[i].len()));
        bounds.push((first, points.len()));
    }
    (points, owner, bounds)
}

/// The largest of `values`, NaN if any is, as numpy's `max` takes it;
/// `empty` if there are none.
fn largest(values: impl IntoIterator<Item = f64>, empty: f64) -> f64 {
    let mut values = values.into_iter();
    values
        .next()
        .map_or(empty, |first| values.fold(first, maximum))
}

impl Seen {
    /// The members as `viewers` and their `exceptions`, in order, seeing
    /// the `tracks`, two tracks of kinds k and l of two nets kept
    /// table[k][l] apart.
    pub fn new(
        tracks: &[Track],
        viewers: &[Viewer],
        table: Vec<Vec<f64>>,
        exceptions: &[Pieces],
    ) -> Self {
        let size = tracks
            .iter()
            .map(|t| t.first + t.vertices)
            .chain(
                exceptions
                    .iter()
                    .filter_map(|p| p.vertex.iter().max().map(|v| v + 2)),
            )
            .max()
            .unwrap_or(0);
        let mut net: Vec<Option<usize>> = vec![None; size];
        let mut kind: Vec<Option<usize>> = vec![None; size];
        // A vertex of no track is a via's first, an obstacle of its own.
        let mut obstacle: Vec<usize> = (0..size).map(|v| tracks.len() + v).collect();
        for (k, t) in tracks.iter().enumerate() {
            let edges = t.first..t.first + t.vertices.saturating_sub(1);
            net[edges.clone()].fill(Some(t.net));
            kind[edges].fill(Some(t.kind));
            obstacle[t.first..t.first + t.vertices].fill(k);
        }
        let mut within = vec![0.0; viewers.len()];
        let mut vertex: Vec<usize> = Vec::new();
        let mut layer_order: Vec<String> = Vec::new();
        let mut on: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, viewer) in viewers.iter().enumerate() {
            if !on.contains_key(viewer.layer.as_str()) {
                layer_order.push(viewer.layer.clone());
            }
            on.entry(viewer.layer.as_str()).or_default().push(i);
        }
        let mut layers = Vec::with_capacity(layer_order.len());
        let mut pieces = 0;
        for layer in &layer_order {
            let members = on[layer.as_str()].clone();
            let shown: Vec<&Track> = tracks.iter().filter(|t| &t.layer == layer).collect();
            let mut blocked: HashMap<usize, usize> = HashMap::new();
            for &i in &members {
                let v = &viewers[i];
                let mut nets = vec![v.net];
                for &n in &v.exempt {
                    if !nets.contains(&n) {
                        nets.push(n);
                    }
                }
                for n in nets {
                    *blocked.entry(n).or_default() += 1;
                }
            }
            let ex_vertex: Vec<usize> = members
                .iter()
                .flat_map(|&i| exceptions[i].vertex.iter().copied())
                .collect();
            let mut edges = ex_vertex.clone();
            for t in &shown {
                if blocked.get(&t.net).copied().unwrap_or(0) < members.len() {
                    edges.extend(t.first..t.first + t.vertices.saturating_sub(1));
                }
            }
            edges.sort_unstable();
            edges.dedup();
            let count = edges.len();
            let mut slots: Vec<(usize, usize, f64)> = Vec::with_capacity(ex_vertex.len());
            for (place, &i) in members.iter().enumerate() {
                for (&v, &inflation) in exceptions[i].vertex.iter().zip(&exceptions[i].inflation) {
                    let at = edges.partition_point(|&e| e < v);
                    let piece = pieces + count + slots.len();
                    slots.push((place * count + at, piece, inflation));
                }
            }
            slots.sort_by_key(|&(slot, _, _)| slot);
            vertex.extend_from_slice(&edges);
            vertex.extend_from_slice(&ex_vertex);
            let mut nets_of: Vec<(usize, Vec<usize>)> = Vec::new();
            for t in &shown {
                match nets_of.iter_mut().find(|(k, _)| *k == t.kind) {
                    Some((_, nets)) => {
                        if !nets.contains(&t.net) {
                            nets.push(t.net);
                        }
                    }
                    None => nets_of.push((t.kind, vec![t.net])),
                }
            }
            for &i in &members {
                let v = &viewers[i];
                let excluded = |n: usize| n == v.net || v.exempt.contains(&n);
                let mut widest = 0.0f64;
                for (k, nets) in &nets_of {
                    if nets.iter().any(|&n| !excluded(n)) {
                        widest = python_max(widest, table[v.kind][*k]);
                    }
                }
                let own = largest(exceptions[i].inflation.iter().copied(), 0.0);
                within[i] = python_max(widest, maximum(own, 0.0));
            }
            layers.push(SeenLayer {
                whole: (pieces..pieces + count).collect(),
                net: edges.iter().map(|&e| net[e]).collect(),
                kind: edges.iter().map(|&e| kind[e]).collect(),
                viewer_net: members.iter().map(|&i| viewers[i].net).collect(),
                viewer_kind: members.iter().map(|&i| viewers[i].kind).collect(),
                exempt: members.iter().map(|&i| viewers[i].exempt.clone()).collect(),
                slot: slots.iter().map(|s| s.0).collect(),
                piece: slots.iter().map(|s| s.1).collect(),
                inflation: slots.iter().map(|s| s.2).collect(),
                edges,
                members,
            });
            pieces += count + ex_vertex.len();
        }
        let obstacle = vertex.iter().map(|&v| obstacle[v]).collect();
        Seen {
            table,
            within,
            layers,
            vertex,
            obstacle,
        }
    }

    /// How many layers its members are on.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// How many members it was made for.
    pub fn member_count(&self) -> usize {
        self.within.len()
    }

    /// Each piece's edge's first vertex, numbered as `Track::first`.
    pub fn vertex(&self) -> &[usize] {
        &self.vertex
    }

    /// Each piece's obstacle for the smooth minimum: its track, or its via.
    pub fn obstacle(&self) -> &[usize] {
        &self.obstacle
    }

    /// Where piece `p` starts and the way to its end, with what the
    /// members show at `flat`.
    fn shape(&self, flat: &[V2], p: usize) -> (V2, V2) {
        let a = flat[self.vertex[p]];
        let b = a + (flat[self.vertex[p] + 1] - a);
        (a, b - a)
    }

    fn shapes(&self, flat: &[V2], pieces: &[usize]) -> (Vec<V2>, Vec<V2>) {
        pieces.iter().map(|&p| self.shape(flat, p)).unzip()
    }

    /// How far from a point of each member of `layer` the smooth search
    /// with `reach` looks at what the member sees, its largest inflation
    /// included: the same for the search and the neighbour list, so that
    /// both give the same answers.
    fn smooth_limit(&self, layer: &SeenLayer, reach: f64) -> Vec<f64> {
        layer
            .members
            .iter()
            .map(|&i| smooth_reach(reach) + self.within[i])
            .collect()
    }

    /// `pairs` in the order `smooth_of_pairs` takes them, searched or
    /// listed alike: by point, then obstacle, then piece.
    fn in_obstacle_order(&self, pairs: &Pairs) -> Pairs {
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_unstable_by_key(|&k| {
            (
                pairs.point[k],
                self.obstacle[pairs.piece[k]],
                pairs.piece[k],
            )
        });
        pairs.taken(order)
    }

    /// (point, piece) pairs of `layer`, among them every piece within
    /// `radius` of a point that the point's member, by its place on the
    /// layer in `owner`, sees, and each pair's inflation.
    fn pairs(
        &self,
        layer: &SeenLayer,
        points: &[V2],
        owner: &[usize],
        flat: &[V2],
        radius: f64,
    ) -> Pairs {
        let mut pairs = Pairs::default();
        if layer.edges.is_empty() || points.is_empty() {
            return pairs;
        }
        let starts: Vec<V2> = layer.edges.iter().map(|&v| flat[v]).collect();
        let ends: Vec<V2> = layer.edges.iter().map(|&v| flat[v + 1]).collect();
        let (point, edge) = near_segments(&starts, &ends, points, radius + ROUND_OFF);
        let count = layer.edges.len();
        for (p, e) in point.into_iter().zip(edge) {
            let member = owner[p];
            let ruled = match (layer.kind[e], layer.net[e]) {
                (Some(kind), Some(net))
                    if net != layer.viewer_net[member] && !layer.exempt[member].contains(&net) =>
                {
                    Some(kind)
                }
                _ => None,
            };
            if let Some(kind) = ruled {
                pairs.push(
                    p,
                    layer.whole[e],
                    self.table[layer.viewer_kind[member]][kind],
                );
                continue;
            }
            let slot = member * count + e;
            let low = layer.slot.partition_point(|&s| s < slot);
            let high = layer.slot.partition_point(|&s| s <= slot);
            for j in low..high {
                pairs.push(p, layer.piece[j], layer.inflation[j]);
            }
        }
        pairs
    }

    /// Per member, the smooth distance from each of its `queries` to the
    /// pieces it sees, each track and each via an obstacle, and what
    /// answers it, as a chain field of the member's own pieces with `reach`
    /// gives it at `flat`; each answer's point is by its place among the
    /// member's queries and its edge is the piece's index. With `listed`,
    /// each layer looks only at the pairs it lists.
    pub fn smooth_answers(
        &self,
        queries: &[&[V2]],
        flat: &[V2],
        reach: f64,
        mut listed: Option<&mut NeighbourList>,
    ) -> Vec<SmoothAnswer> {
        let mut answers: Vec<SmoothAnswer> = (0..self.within.len())
            .map(|_| SmoothAnswer::default())
            .collect();
        for (n, layer) in self.layers.iter().enumerate() {
            let (points, owner, bounds) = gathered(queries, &layer.members);
            let limit = self.smooth_limit(layer, reach);
            let searched;
            let pairs = match listed.as_deref_mut() {
                Some(list) => list.pairs(self, n, &points, &owner, flat),
                None => {
                    let radius = largest(limit.iter().copied(), f64::NEG_INFINITY);
                    searched =
                        self.in_obstacle_order(&self.pairs(layer, &points, &owner, flat, radius));
                    &searched
                }
            };
            let (starts, along) = self.shapes(flat, &pairs.piece);
            let per_pair: Vec<f64> = pairs.point.iter().map(|&p| limit[owner[p]]).collect();
            let obstacle: Vec<usize> = pairs.piece.iter().map(|&p| self.obstacle[p]).collect();
            let found = smooth_of_pairs(
                &points,
                &pairs.point,
                &starts,
                &along,
                &pairs.inflation,
                &per_pair,
                &obstacle,
            );
            for (&i, &(first, last)) in layer.members.iter().zip(&bounds) {
                let low = found.point.partition_point(|&p| p < first);
                let high = found.point.partition_point(|&p| p < last);
                answers[i] = SmoothAnswer {
                    distance: found.distance[first..last].to_vec(),
                    gradient: found.gradient[first..last].to_vec(),
                    point: found.point[low..high].iter().map(|p| p - first).collect(),
                    edge: found.edge[low..high]
                        .iter()
                        .map(|&e| pairs.piece[e])
                        .collect(),
                    share: found.share[low..high].to_vec(),
                    direction: found.direction[low..high].to_vec(),
                };
            }
        }
        answers
    }

    /// Per member, the signed distance from each of its `queries` to the
    /// nearest piece it sees, as a chain field of the member's own pieces
    /// gives it, +inf where it sees none. `below` is the threshold the
    /// caller compares the distances against: below it they are exact,
    /// elsewhere at least `below`. `flat` is what the members show at the
    /// shapes this was built at, with the vias where the caller holds them.
    ///
    /// One search within the members' largest inflation plus `below`
    /// answers every point whose nearest piece lies within it.
    pub fn distances(&self, queries: &[&[V2]], flat: &[V2], below: f64) -> Vec<Vec<f64>> {
        let mut answers = vec![Vec::new(); self.within.len()];
        for layer in &self.layers {
            let (points, owner, bounds) = gathered(queries, &layer.members);
            let mut distance = vec![f64::INFINITY; points.len()];
            let widest = largest(layer.members.iter().map(|&i| self.within[i]), 0.0);
            let radius = widest + below;
            if !layer.edges.is_empty() && widest > 0.0 && radius >= 0.0 {
                let pairs = self.pairs(layer, &points, &owner, flat, radius);
                let (starts, along) = self.shapes(flat, &pairs.piece);
                distance = nearest_of_pairs(
                    &points,
                    &pairs.point,
                    &starts,
                    &along,
                    &pairs.inflation,
                    &vec![radius; pairs.len()],
                )
                .distance;
            }
            for (&i, &(first, last)) in layer.members.iter().zip(&bounds) {
                answers[i] = distance[first..last].to_vec();
            }
        }
        answers
    }

    /// Per member at `polylines`, whether each of its edges stays out of
    /// the pieces it sees at `flat`, as a chain field of the member's own
    /// pieces with `tolerance` gives it, and what each edge that does not
    /// enters.
    pub fn segments_clear(
        &self,
        polylines: &[&[V2]],
        flat: &[V2],
        tolerance: f64,
    ) -> Vec<EdgesClear> {
        let mut answers = vec![EdgesClear::default(); self.within.len()];
        let starts_of: Vec<&[V2]> = polylines
            .iter()
            .map(|p| &p[..p.len().saturating_sub(1)])
            .collect();
        let ends_of: Vec<&[V2]> = polylines
            .iter()
            .map(|p| p.get(1..).unwrap_or(&[]))
            .collect();
        for layer in &self.layers {
            let (a, owner, bounds) = gathered(&starts_of, &layer.members);
            let (b, _, _) = gathered(&ends_of, &layer.members);
            let mut crossing = vec![false; a.len()];
            let mut found = EdgesClear::default();
            let reach = largest(layer.members.iter().map(|&i| self.within[i]), 0.0) - tolerance;
            if !layer.edges.is_empty() && !a.is_empty() && reach > 0.0 {
                let longest = largest(a.iter().zip(&b).map(|(&p, &q)| (q - p).hypot()), 0.0);
                let radius = reach + longest / 2.0;
                let middles: Vec<V2> = a
                    .iter()
                    .zip(&b)
                    .map(|(&p, &q)| V2::new((p.x + q.x) / 2.0, (p.y + q.y) / 2.0))
                    .collect();
                let pairs = self.pairs(layer, &middles, &owner, flat, radius);
                let (starts, along) = self.shapes(flat, &pairs.piece);
                let segment_a: Vec<V2> = pairs.point.iter().map(|&s| a[s]).collect();
                let segment_ab: Vec<V2> = pairs.point.iter().map(|&s| b[s] - a[s]).collect();
                let inside = pairs_inside(
                    &segment_a,
                    &segment_ab,
                    &starts,
                    &along,
                    &pairs.inflation,
                    tolerance,
                );
                // The pairs come by segment, so those taken stay in order.
                let taken: Vec<usize> = (0..pairs.len()).filter(|&k| inside[k]).collect();
                let depth = pair_depths(
                    &taken.iter().map(|&k| segment_a[k]).collect::<Vec<_>>(),
                    &taken.iter().map(|&k| segment_ab[k]).collect::<Vec<_>>(),
                    &taken.iter().map(|&k| starts[k]).collect::<Vec<_>>(),
                    &taken.iter().map(|&k| along[k]).collect::<Vec<_>>(),
                    &taken
                        .iter()
                        .map(|&k| pairs.inflation[k])
                        .collect::<Vec<_>>(),
                );
                for &k in &taken {
                    let entered = pairs.piece[k];
                    found.edge.push(pairs.point[k]);
                    found.vertex.push(self.vertex[entered]);
                    found.obstacle.push(self.obstacle[entered]);
                    crossing[pairs.point[k]] = true;
                }
                found.depth = depth;
            }
            for (&i, &(first, last)) in layer.members.iter().zip(&bounds) {
                let low = found.edge.partition_point(|&s| s < first);
                let high = found.edge.partition_point(|&s| s < last);
                answers[i] = EdgesClear {
                    clear: crossing[first..last].iter().map(|c| !c).collect(),
                    edge: found.edge[low..high].iter().map(|s| s - first).collect(),
                    vertex: found.vertex[low..high].to_vec(),
                    obstacle: found.obstacle[low..high].to_vec(),
                    depth: found.depth[low..high].to_vec(),
                };
            }
        }
        answers
    }

    /// The (point, piece) pairs of layer n, with its members' quadrature
    /// points `points`, each of the member `owner` gives by its place on
    /// the layer, that `smooth_answers` with `reach` can answer a point
    /// with once the distance from each point to each piece has changed by
    /// at most `slack`, mm (`field::pairs::candidate_pairs`), and each
    /// pair's inflation, in the order `smooth_answers` takes them.
    pub(crate) fn candidates(
        &self,
        n: usize,
        points: &[V2],
        owner: &[usize],
        flat: &[V2],
        reach: f64,
        slack: f64,
    ) -> Pairs {
        let layer = &self.layers[n];
        let within = self.smooth_limit(layer, reach);
        let radius = largest(within.iter().copied(), f64::NEG_INFINITY) + slack;
        let pairs = self.pairs(layer, points, owner, flat, radius);
        let (starts, along) = self.shapes(flat, &pairs.piece);
        let per_pair: Vec<f64> = pairs.point.iter().map(|&p| within[owner[p]]).collect();
        let keep = candidate_pairs(
            points,
            &pairs.point,
            &starts,
            &along,
            &pairs.inflation,
            &per_pair,
            slack,
        );
        self.in_obstacle_order(&pairs.taken((0..pairs.len()).filter(|&k| keep[k])))
    }
}

/// Python's `max(a, b)`: `b` only if it is greater.
fn python_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two straight tracks on one layer, of nets 0 and 1 and one kind, each
    /// a member seeing the other at `apart`.
    fn two_tracks(apart: f64) -> (Seen, Vec<V2>) {
        let tracks = [
            Track {
                first: 0,
                vertices: 3,
                layer: "F".into(),
                net: 0,
                kind: 0,
            },
            Track {
                first: 3,
                vertices: 3,
                layer: "F".into(),
                net: 1,
                kind: 0,
            },
        ];
        let viewers = [
            Viewer {
                layer: "F".into(),
                net: 0,
                kind: 0,
                exempt: vec![],
            },
            Viewer {
                layer: "F".into(),
                net: 1,
                kind: 0,
                exempt: vec![],
            },
        ];
        let seen = Seen::new(
            &tracks,
            &viewers,
            vec![vec![0.3]],
            &[Pieces::default(), Pieces::default()],
        );
        let flat = vec![
            V2::new(0.0, 0.0),
            V2::new(1.0, 0.0),
            V2::new(2.0, 0.0),
            V2::new(0.0, apart),
            V2::new(1.0, apart),
            V2::new(2.0, apart),
        ];
        (seen, flat)
    }

    #[test]
    fn a_member_sees_the_other_net_and_not_its_own() {
        let (seen, flat) = two_tracks(1.0);
        let queries: [&[V2]; 2] = [&[V2::new(1.0, 0.0)], &[V2::new(1.0, 1.0)]];
        let distances = seen.distances(&queries, &flat, 10.0);
        assert_eq!(distances, vec![vec![0.7], vec![0.7]]);
    }

    #[test]
    fn copper_of_one_net_does_not_interact() {
        let tracks = [
            Track {
                first: 0,
                vertices: 2,
                layer: "F".into(),
                net: 0,
                kind: 0,
            },
            Track {
                first: 2,
                vertices: 2,
                layer: "F".into(),
                net: 0,
                kind: 0,
            },
        ];
        let viewer = Viewer {
            layer: "F".into(),
            net: 0,
            kind: 0,
            exempt: vec![],
        };
        let seen = Seen::new(
            &tracks,
            &[viewer.clone(), viewer],
            vec![vec![0.3]],
            &[Pieces::default(), Pieces::default()],
        );
        let flat = [
            V2::new(0.0, 0.0),
            V2::new(1.0, 0.0),
            V2::new(0.0, 0.1),
            V2::new(1.0, 0.1),
        ];
        let queries: [&[V2]; 2] = [&flat[..2], &flat[2..]];
        assert_eq!(
            seen.distances(&queries, &flat, 10.0),
            vec![vec![f64::INFINITY; 2]; 2]
        );
        let clear = seen.segments_clear(&queries, &flat, 0.0);
        assert!(clear.iter().all(|c| c.clear.iter().all(|&c| c)));
    }

    #[test]
    fn an_exempt_net_is_not_seen() {
        let (_, flat) = two_tracks(1.0);
        let seen = Seen::new(
            &[
                Track {
                    first: 0,
                    vertices: 3,
                    layer: "F".into(),
                    net: 0,
                    kind: 0,
                },
                Track {
                    first: 3,
                    vertices: 3,
                    layer: "F".into(),
                    net: 1,
                    kind: 0,
                },
            ],
            &[
                Viewer {
                    layer: "F".into(),
                    net: 0,
                    kind: 0,
                    exempt: vec![1],
                },
                Viewer {
                    layer: "F".into(),
                    net: 1,
                    kind: 0,
                    exempt: vec![],
                },
            ],
            vec![vec![0.3]],
            &[Pieces::default(), Pieces::default()],
        );
        let queries: [&[V2]; 2] = [&[V2::new(1.0, 0.0)], &[V2::new(1.0, 1.0)]];
        assert_eq!(
            seen.distances(&queries, &flat, 10.0),
            vec![vec![f64::INFINITY], vec![0.7]]
        );
    }

    #[test]
    fn a_crossing_edge_is_inside_by_the_whole_inflation() {
        let (seen, mut flat) = two_tracks(1.0);
        // The second track dips across the first.
        flat[4] = V2::new(1.0, -1.0);
        let polylines: [&[V2]; 2] = [&flat[..3], &flat[3..]];
        let clear = seen.segments_clear(&polylines, &flat, 0.0);
        assert_eq!(clear[1].clear, vec![false, false]);
        // Each edge crosses the track's edge below it, and keeps clear of
        // the other.
        assert_eq!(clear[1].edge, vec![0, 1]);
        assert_eq!(clear[1].vertex, vec![0, 1]);
        assert_eq!(clear[1].obstacle, vec![0; 2]);
        assert!(clear[1].depth.iter().any(|&d| d == -0.3));
    }

    #[test]
    fn a_via_exception_is_seen_with_its_own_inflation() {
        let tracks = [Track {
            first: 0,
            vertices: 2,
            layer: "F".into(),
            net: 0,
            kind: 0,
        }];
        let viewers = [Viewer {
            layer: "F".into(),
            net: 0,
            kind: 0,
            exempt: vec![],
        }];
        let ring = Pieces {
            vertex: vec![2],
            inflation: vec![0.4],
        };
        let seen = Seen::new(&tracks, &viewers, vec![vec![0.3]], &[ring]);
        let flat = [
            V2::new(0.0, 0.0),
            V2::new(1.0, 0.0),
            V2::new(0.5, 1.0),
            V2::new(0.5, 1.0),
        ];
        let queries: [&[V2]; 1] = [&flat[..2]];
        let answers = seen.smooth_answers(&queries, &flat, 1.0, None);
        let expected = (0.5f64.powi(2) + 1.0).sqrt() - 0.4;
        assert_eq!(answers[0].distance, vec![expected, expected]);
        // The member's own net's track is no edge it sees; the via is an
        // obstacle numbered after the one track, by its vertex.
        assert_eq!(seen.obstacle(), &[3, 3]);
    }
}
