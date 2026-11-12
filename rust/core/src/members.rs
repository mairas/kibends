//! The members of a problem, what they see of each other and their summed
//! energy.
//!
//! A `Member` is a chain or a coupled section as its centreline. The
//! others see a chain as itself and a section as its two tracks
//! (`Member::shown`), laid one after another, then each via's centre twice
//! (`Members::shown_flat`). `Members::seen` gives what each member sees of
//! the others on its layer as a `seen::Seen`: the rule by net and kind, and
//! as exceptions only the vias, so that it grows with the members, not
//! with pairs of them. `Members::energy` sums each member's energy in its
//! static field plus what it sees of the others, the distance its obstacle
//! term takes the smooth minimum over both
//! (`field::smooth::combine_fields`), each track and each via an obstacle,
//! and gives its gradient at every vertex and via centre: where a member's
//! obstacle term pulls on another's edge, each obstacle's share of the
//! gradient reaches the ends of its edge nearest the point, and through a
//! section's offset its centreline (`pair_blocks::PairOffset::
//! offset_pullback`). `pull_answers` takes a via's push back to what
//! answers it, by the same shares. `Members::stiffness` gives each
//! vertex's stiffness under its obstacle term, by which a burst scales its
//! variables.
//!
//! Copper of one net does not interact: a chain sees nothing of the
//! chains, tracks and vias of its net, nor of the tracks of the other net
//! of its diff pair, and a section nothing of its two nets' chains, which
//! ignore it too, so that its fan-outs leave it freely.

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

use crate::energy::EnergyParams;
use crate::energy::integrands::{Curvature, Integrand};
use crate::energy::polyline::{bend_radius_energy, shape_energy};
use crate::energy::quadrature::{edge_curvature, edge_integral, quadrature_points};
use crate::field::Field;
use crate::field::pairs::SmoothAnswer;
use crate::field::smooth::combine_fields;
use crate::geometry::V2;
use crate::neighbours::{NeighbourList, StaticList};
use crate::numeric::clip;
use crate::pair_blocks::PairOffset;
use crate::rules::track_inflation;
use crate::seen::{Pieces, Seen, Track, Viewer};
use crate::vias::sight::ViaSmooth;

/// A track's net, width and clearance, mm.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    pub net: String,
    pub width: f64,
    pub clearance: f64,
}

/// A coupled section's offset and the smallest bend radius its centreline
/// keeps, mm, through the bend-radius hinge; None leaves the hinge out.
#[derive(Clone, Debug)]
pub struct Section {
    pub offset: PairOffset,
    pub limit: Option<f64>,
}

/// A chain, or a coupled section as its centreline.
#[derive(Clone)]
pub struct Member {
    pub layer: String,
    /// What it keeps clear of the others as: a chain itself, a section the
    /// diff pair's outer extent.
    pub viewer: Trace,
    /// The nets whose tracks it ignores: a chain's own and the other net
    /// of its diff pair, a section's two.
    pub ignored: Vec<String>,
    /// What the others see of it: a chain itself, a section its two
    /// tracks.
    pub shown: Vec<Trace>,
    /// Its static field, without any selected chain.
    pub field: Arc<dyn Field>,
    /// The width its contact stops weigh by, mm.
    pub width: f64,
    /// The regions its vertices that keep a contact stay inside, held as
    /// the board edge is: positive inside.
    pub contacts: Vec<Arc<dyn Field>>,
    pub section: Option<Section>,
}

impl Member {
    /// Whether it sees nothing of the tracks of `net`: its own and those
    /// of `ignored`.
    pub fn ignores(&self, net: &str) -> bool {
        net == self.viewer.net || self.ignored.iter().any(|n| n == net)
    }

    /// How far its centreline keeps from `track`, one that `other` shows,
    /// mm (`track_inflation`): None where it sees nothing of it, on another
    /// layer or of a net it ignores. `Members::seen` applies the same rule
    /// by net and kind.
    pub fn keeps_from(&self, other: &Member, track: &Trace) -> Option<f64> {
        if other.layer != self.layer || self.ignores(&track.net) {
            return None;
        }
        let me = &self.viewer;
        Some(track_inflation(
            me.width,
            me.clearance,
            track.width,
            track.clearance,
        ))
    }

    /// How many vertices each track it shows at `points` has: a held
    /// section's tracks are its own, whatever its centreline.
    pub fn shown_lengths(&self, points: &[V2]) -> Vec<usize> {
        match &self.section {
            Some(section) if section.offset.held => {
                vec![section.offset.a.len(), section.offset.b.len()]
            }
            Some(_) => vec![points.len(); 2],
            None => vec![points.len()],
        }
    }

    /// What the others see of it at `points`: a chain itself, a section its
    /// two tracks.
    pub fn shown<'a>(&self, points: &'a [V2]) -> Vec<Cow<'a, [V2]>> {
        match &self.section {
            Some(section) => {
                let (a, b) = section.offset.offset(points);
                vec![Cow::Owned(a), Cow::Owned(b)]
            }
            None => vec![Cow::Borrowed(points)],
        }
    }
}

/// The summed energy of members, its gradient at every vertex of every
/// member, and at each via's centre.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Energy {
    pub value: f64,
    pub gradients: Vec<Vec<V2>>,
    pub via_gradients: Vec<V2>,
}

/// What every member's quadrature points see, one member after another.
struct Fields {
    /// Per member, its quadrature points.
    queries: Vec<Vec<V2>>,
    /// Per member, what its points see of the others.
    answers: Vec<SmoothAnswer>,
    /// Where each member's points start, and where the last one's end.
    bounds: Vec<usize>,
    /// The smooth minimum over the static field and the others, and its
    /// gradient.
    distance: Vec<f64>,
    direction: Vec<V2>,
    /// The others' share of it.
    seen_share: Vec<f64>,
}

/// One member's terms in `Members::energy`: its shape and obstacle terms,
/// its bend-radius hinge, their gradient at its vertices and the obstacle
/// integrand's slope at its quadrature points.
struct Terms {
    own: f64,
    bend: Option<f64>,
    gradient: Vec<V2>,
    slope: Vec<f64>,
}

/// The members of a problem, in order.
#[derive(Clone)]
pub struct Members {
    pub members: Vec<Member>,
}

/// Where along each segment from starts[i] to ends[i] its point nearest
/// points[i] lies, from 0 to 1.
pub fn parameter(point: V2, start: V2, end: V2) -> f64 {
    // numpy's `einsum` contracts two columns without fusing the products.
    let along = end - start;
    let squared = along.dot(along);
    let projection = (point - start).dot(along);
    clip(
        projection / if squared > 0.0 { squared } else { 1.0 },
        0.0,
        1.0,
    )
}

fn scaled(s: f64, v: V2) -> V2 {
    V2::new(s * v.x, s * v.y)
}

fn add_to(gradient: &mut [V2], other: &[V2]) {
    for (g, &o) in gradient.iter_mut().zip(other) {
        *g = *g + o;
    }
}

impl Members {
    pub fn new(members: Vec<Member>) -> Self {
        Self { members }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// What the others see of every member, polylines one after another,
    /// then each via's centre twice, an edge of no length.
    pub fn shown_flat(&self, polylines: &[&[V2]], positions: &[V2]) -> Vec<V2> {
        let mut flat = Vec::new();
        for (member, &points) in self.members.iter().zip(polylines) {
            for part in member.shown(points) {
                flat.extend_from_slice(&part);
            }
        }
        for &at in positions {
            flat.push(at);
            flat.push(at);
        }
        flat
    }

    /// For each member at its polyline, the others on its layer at theirs:
    /// a section sees as its centreline of the diff pair's outer extent,
    /// and is seen as its two tracks. Each via of `rings` is seen by the
    /// members it names, inflated by the distance it gives.
    ///
    /// The tracks of the nets a member does not ignore are seen whole, by
    /// the rule `Seen` applies; only the vias are listed, as its
    /// exceptions, so the work grows with the members and their edges, not
    /// with pairs of members.
    pub fn seen(&self, polylines: &[&[V2]], rings: &[Vec<(usize, f64)>]) -> Seen {
        let mut nets: Vec<String> = Vec::new();
        let mut kinds: Vec<(f64, f64)> = Vec::new();
        fn number<T: PartialEq + Clone>(values: &mut Vec<T>, value: &T) -> usize {
            values.iter().position(|v| v == value).unwrap_or_else(|| {
                values.push(value.clone());
                values.len() - 1
            })
        }
        let mut tracks = Vec::new();
        let mut first = 0;
        for (member, &points) in self.members.iter().zip(polylines) {
            for (trace, vertices) in member.shown.iter().zip(member.shown_lengths(points)) {
                tracks.push(Track {
                    first,
                    vertices,
                    layer: member.layer.clone(),
                    net: number(&mut nets, &trace.net),
                    kind: number(&mut kinds, &(trace.width, trace.clearance)),
                });
                first += vertices;
            }
        }
        let mut viewers = Vec::with_capacity(self.members.len());
        for member in &self.members {
            let me = &member.viewer;
            // A net no member shows has no tracks to ignore.
            let exempt: Vec<usize> = nets
                .iter()
                .enumerate()
                .filter(|(_, n)| **n != me.net && member.ignores(n))
                .map(|(k, _)| k)
                .collect();
            viewers.push(Viewer {
                layer: member.layer.clone(),
                net: number(&mut nets, &me.net),
                kind: number(&mut kinds, &(me.width, me.clearance)),
                exempt,
            });
        }
        let mut exceptions = vec![Pieces::default(); self.members.len()];
        for (k, ring) in rings.iter().enumerate() {
            for &(i, inflation) in ring {
                exceptions[i].vertex.push(first + 2 * k);
                exceptions[i].inflation.push(inflation);
            }
        }
        let table = kinds
            .iter()
            .map(|&(w, c)| {
                kinds
                    .iter()
                    .map(|&(ow, oc)| track_inflation(w, c, ow, oc))
                    .collect()
            })
            .collect();
        Seen::new(&tracks, &viewers, table, &exceptions)
    }

    /// The members' static fields as a `StaticList` with the energy's
    /// `reach`.
    pub fn static_list(&self, reach: f64, listed: bool) -> StaticList {
        StaticList::new(
            self.members.iter().map(|m| Some(m.field.clone())).collect(),
            reach,
            listed,
        )
    }

    /// The distances of every member's quadrature points at `polylines`,
    /// the others shown as `flat`, for `energy` and `stiffness`.
    fn fields(
        &self,
        polylines: &[&[V2]],
        params: &EnergyParams,
        seen: &Seen,
        flat: &[V2],
        listed: Option<&mut NeighbourList>,
        statics: Option<&mut StaticList>,
    ) -> Fields {
        let reach = params.obstacle_reach();
        let k = params.points_per_edge;
        let queries: Vec<Vec<V2>> = polylines
            .iter()
            .map(|points| quadrature_points(points, k))
            .collect();
        let refs: Vec<&[V2]> = queries.iter().map(Vec::as_slice).collect();
        let answers = seen.smooth_answers(&refs, flat, reach, listed);
        let static_answers = match statics {
            Some(statics) => statics.smooth(&refs),
            None => self.static_list(reach, false).smooth(&refs),
        };
        let mut bounds = vec![0];
        for q in &queries {
            bounds.push(bounds[bounds.len() - 1] + q.len());
        }
        let static_distance: Vec<f64> =
            static_answers.iter().flat_map(|(d, _)| d.clone()).collect();
        let seen_distance: Vec<f64> = answers.iter().flat_map(|a| a.distance.clone()).collect();
        let (distance, shares) = combine_fields(&[&static_distance, &seen_distance]);
        let static_gradient = static_answers.iter().flat_map(|(_, g)| g.iter().copied());
        let seen_gradient = answers.iter().flat_map(|a| a.gradient.iter().copied());
        let direction = static_gradient
            .zip(seen_gradient)
            .zip(shares[0].iter().zip(&shares[1]))
            .map(|((g, h), (&s, &t))| V2::new(s * g.x + t * h.x, s * g.y + t * h.y))
            .collect();
        let seen_share = shares.into_iter().nth(1).unwrap_or_default();
        Fields {
            queries,
            answers,
            bounds,
            distance,
            direction,
            seen_share,
        }
    }

    /// The summed energy of the members on any layers, each in its static
    /// field plus what it sees of the others on its own layer at
    /// `polylines` as `seen` shows them, with vias at `positions` among
    /// them, a section with its bend-radius hinge; its gradient at every
    /// vertex of every chain and centreline, with the vertices of `masks`
    /// zeroed; and the gradient at each via's centre. Each member's
    /// short-edge barrier starts at its own of `short_edges`. Err with the
    /// value if it is not finite.
    ///
    /// A section's centreline is what it keeps clear as, and its two
    /// tracks, offset from it, are what the others keep clear of: their
    /// part of the gradient is taken back through the offset.
    ///
    /// With `listed`, made for `seen`, each layer looks only at the pieces
    /// it lists; with `statics`, made by `static_list` with the energy's
    /// reach, the static fields answer from it, by default from one search
    /// each.
    #[allow(clippy::too_many_arguments)]
    pub fn energy(
        &self,
        polylines: &[&[V2]],
        masks: &[&[bool]],
        params: &EnergyParams,
        short_edges: &[f64],
        seen: &Seen,
        positions: &[V2],
        listed: Option<&mut NeighbourList>,
        statics: Option<&mut StaticList>,
    ) -> Result<Energy, f64> {
        let integrand = Integrand::obstacle(params);
        let k = params.points_per_edge;
        let flat = self.shown_flat(polylines, positions);
        let mut at_shown = vec![V2::default(); flat.len()];
        let fields = self.fields(polylines, params, seen, &flat, listed, statics);
        // Each member's terms on its own, summed in member order after.
        let terms: Vec<Terms> = self
            .members
            .par_iter()
            .zip(polylines)
            .enumerate()
            .map(|(i, (member, &points))| {
                let mine = fields.bounds[i]..fields.bounds[i + 1];
                let (obstacle, obstacle_gradient, slope) = edge_integral(
                    points,
                    &fields.distance[mine.clone()],
                    &fields.direction[mine],
                    &integrand,
                    k,
                );
                let (shape, shape_gradient) = shape_energy(points, masks[i], short_edges[i]);
                let mut gradient: Vec<V2> = shape_gradient
                    .iter()
                    .zip(&obstacle_gradient)
                    .map(|(&s, &o)| s + o)
                    .collect();
                let bend = member.section.as_ref().and_then(|s| s.limit).map(|limit| {
                    let (bend, bend_gradient) = bend_radius_energy(points, masks[i], limit, params);
                    add_to(&mut gradient, &bend_gradient);
                    bend
                });
                Terms {
                    own: shape + obstacle,
                    bend,
                    gradient,
                    slope,
                }
            })
            .collect();
        let mut slopes = Vec::with_capacity(fields.distance.len());
        let mut value = 0.0;
        let mut gradients = Vec::with_capacity(self.members.len());
        for Terms {
            own,
            bend,
            gradient,
            slope,
        } in terms
        {
            value += own;
            if let Some(bend) = bend {
                value += bend;
            }
            gradients.push(gradient);
            slopes.extend(slope);
        }
        // Where a member's term pulls on an edge another shows, each
        // answer's share goes to the ends of the edge, by where along it
        // the point is nearest.
        let pushes: Vec<Vec<(usize, V2, f64)>> = fields
            .answers
            .par_iter()
            .enumerate()
            .map(|(i, near)| {
                let mut pushes = Vec::new();
                for n in 0..near.point.len() {
                    let at = fields.bounds[i] + near.point[n];
                    let pull = (slopes[at] * fields.seen_share[at]) * near.share[n];
                    if pull != 0.0 {
                        let piece = near.edge[n];
                        let start = seen.vertex()[piece];
                        let a = flat[start];
                        let b = a + (flat[start + 1] - a);
                        let along = parameter(fields.queries[i][near.point[n]], a, b);
                        pushes.push((start, scaled(pull, near.direction[n]), along));
                    }
                }
                pushes
            })
            .collect();
        for &(start, push, along) in pushes.iter().flatten() {
            at_shown[start] = at_shown[start] + scaled(1.0 - along, -push);
        }
        for &(start, push, along) in pushes.iter().flatten() {
            at_shown[start + 1] = at_shown[start + 1] + scaled(along, -push);
        }
        let mut first = 0;
        for ((member, &points), gradient) in self.members.iter().zip(polylines).zip(&mut gradients)
        {
            let lengths = member.shown_lengths(points);
            match (&member.section, lengths.as_slice()) {
                (Some(section), &[a, b]) => {
                    let pulled = section.offset.offset_pullback(
                        points,
                        &at_shown[first..first + a],
                        &at_shown[first + a..first + a + b],
                    );
                    add_to(gradient, &pulled);
                }
                _ => add_to(gradient, &at_shown[first..first + points.len()]),
            }
            first += lengths.iter().sum::<usize>();
        }
        let via_gradients = (0..positions.len())
            .map(|k| at_shown[first + 2 * k] + at_shown[first + 2 * k + 1])
            .collect();
        let finite = value.is_finite()
            && gradients
                .iter()
                .all(|g| g.iter().all(|v| v.x.is_finite() && v.y.is_finite()));
        if !finite {
            return Err(value);
        }
        for (gradient, mask) in gradients.iter_mut().zip(masks) {
            for (g, &fixed) in gradient.iter_mut().zip(*mask) {
                if fixed {
                    *g = V2::default();
                }
            }
        }
        Ok(Energy {
            value,
            gradients,
            via_gradients,
        })
    }

    /// Per member, each vertex's stiffness under its obstacle term at
    /// `polylines` (`energy::quadrature::edge_curvature`), with what it
    /// sees of its static field and the others as `energy` finds it,
    /// `listed` and `statics` as there. Only the member's own quadrature
    /// points count: where its edges are what another member's term sees,
    /// the curvature of that pull is left out.
    pub fn stiffness(
        &self,
        polylines: &[&[V2]],
        params: &EnergyParams,
        seen: &Seen,
        positions: &[V2],
        listed: Option<&mut NeighbourList>,
        statics: Option<&mut StaticList>,
    ) -> Vec<Vec<f64>> {
        let flat = self.shown_flat(polylines, positions);
        let fields = self.fields(polylines, params, seen, &flat, listed, statics);
        let integrand = Integrand::obstacle(params);
        let curvature = Curvature::obstacle(params);
        polylines
            .par_iter()
            .enumerate()
            .map(|(i, points)| {
                let mine = fields.bounds[i]..fields.bounds[i + 1];
                edge_curvature(
                    points,
                    &fields.distance[mine.clone()],
                    &fields.direction[mine],
                    &integrand,
                    &curvature,
                    params.points_per_edge,
                )
            })
            .collect()
    }
}

/// Take from what `near` found answering each via its share of the via's
/// push, its unit gradient times the via's `slope`: from the other via, or
/// from the ends of a member's edge in proportion to where along it the
/// via's centre at `at` is nearest.
pub fn pull_answers(
    near: &ViaSmooth,
    slope: &[f64],
    at: &[V2],
    gradients: &mut [Vec<V2>],
    via_gradients: &mut [V2],
) {
    let push: Vec<V2> = (0..near.point.len())
        .map(|n| scaled(slope[near.point[n]] * near.share[n], near.direction[n]))
        .collect();
    let mut members: Vec<usize> = Vec::new();
    for (n, member) in near.member.iter().enumerate() {
        match member {
            None => {
                let v = near.vertex[n];
                via_gradients[v] = via_gradients[v] - push[n];
            }
            Some(j) => members.push(*j),
        }
    }
    members.sort_unstable();
    members.dedup();
    for j in members {
        let edges: Vec<(usize, f64)> = (0..near.point.len())
            .filter(|&n| near.member[n] == Some(j))
            .map(|n| {
                let t = parameter(at[near.point[n]], near.starts[n], near.ends[n]);
                (n, t)
            })
            .collect();
        let gradient = &mut gradients[j];
        for &(n, t) in &edges {
            let first = near.vertex[n];
            gradient[first] = gradient[first] - scaled(1.0 - t, push[n]);
        }
        for &(n, t) in &edges {
            let first = near.vertex[n];
            gradient[first + 1] = gradient[first + 1] - scaled(t, push[n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FreeSpace;

    fn chain(net: &str, layer: &str) -> Member {
        Member {
            layer: layer.into(),
            viewer: Trace {
                net: net.into(),
                width: 0.2,
                clearance: 0.2,
            },
            ignored: vec![net.into()],
            shown: vec![Trace {
                net: net.into(),
                width: 0.2,
                clearance: 0.2,
            }],
            field: Arc::new(FreeSpace),
            width: 0.2,
            contacts: vec![],
            section: None,
        }
    }

    fn params() -> EnergyParams {
        EnergyParams {
            repulsion: 0.5,
            falloff_range: 1.0,
            clearance_weight: 10.0,
            standoff: 0.05,
            points_per_edge: 3,
            attraction: 0.0,
            attraction_range: 3.0,
        }
    }

    fn line(y: f64) -> Vec<V2> {
        (0..5).map(|x| V2::new(f64::from(x) * 0.5, y)).collect()
    }

    #[test]
    fn a_member_keeps_from_tracks_of_other_nets_on_its_layer() {
        let mut a = chain("A", "F");
        a.ignored.push("A_N".into());
        let keeps = |other: &Member| a.keeps_from(other, &other.shown[0]);
        assert_eq!(keeps(&chain("B", "F")), Some(0.4));
        assert_eq!(keeps(&chain("A", "F")), None);
        assert_eq!(keeps(&chain("A_N", "F")), None);
        assert_eq!(keeps(&chain("B", "B")), None);
    }

    #[test]
    fn two_nets_on_one_layer_push_each_other_apart() {
        let members = Members::new(vec![chain("A", "F"), chain("B", "F")]);
        let (a, b) = (line(0.0), line(0.5));
        let polylines: [&[V2]; 2] = [&a, &b];
        let mask = [true, false, false, false, true];
        let masks: [&[bool]; 2] = [&mask, &mask];
        let seen = members.seen(&polylines, &[]);
        let energy = members
            .energy(
                &polylines,
                &masks,
                &params(),
                &[0.0; 2],
                &seen,
                &[],
                None,
                None,
            )
            .expect("finite");
        // A's free vertices are pushed down, away from B, and B's up.
        assert!(energy.gradients[0][2].y > 0.0);
        assert!(energy.gradients[1][2].y < 0.0);
        assert_eq!(energy.gradients[0][0], V2::default());
    }

    #[test]
    fn same_net_copper_does_not_interact() {
        let members = Members::new(vec![chain("A", "F"), chain("A", "F")]);
        let alone = Members::new(vec![chain("A", "F")]);
        let (a, b) = (line(0.0), line(0.1));
        let mask = [true, false, false, false, true];
        let polylines: [&[V2]; 2] = [&a, &b];
        let masks: [&[bool]; 2] = [&mask, &mask];
        let seen = members.seen(&polylines, &[]);
        let both = members
            .energy(
                &polylines,
                &masks,
                &params(),
                &[0.0; 2],
                &seen,
                &[],
                None,
                None,
            )
            .expect("finite");
        let one: [&[V2]; 1] = [&a];
        let own = alone
            .energy(
                &one,
                &masks[..1],
                &params(),
                &[0.0],
                &alone.seen(&one, &[]),
                &[],
                None,
                None,
            )
            .expect("finite");
        assert_eq!(both.gradients[0], own.gradients[0]);
    }

    #[test]
    fn a_member_on_another_layer_is_not_seen() {
        let members = Members::new(vec![chain("A", "F"), chain("B", "B")]);
        let (a, b) = (line(0.0), line(0.1));
        let polylines: [&[V2]; 2] = [&a, &b];
        let seen = members.seen(&polylines, &[]);
        let queries: [&[V2]; 2] = [&a, &b];
        let flat = members.shown_flat(&polylines, &[]);
        assert_eq!(
            seen.distances(&queries, &flat, 10.0),
            vec![vec![f64::INFINITY; 5]; 2]
        );
    }
}
