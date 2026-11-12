//! The energy of one optimization problem: every member's terms, what the
//! members see of each other, the moving vias' own terms, the pairwise via
//! term and the contact stops, as one function of the problem's variables
//! with its gradient.
//!
//! A moving via's own obstacle terms are its ring's, standing for a track
//! as long as it is wide, and there are two of them, each taking the
//! smooth minimum over what it covers. The first, the full integrand,
//! covers the via's static field over every layer it spans, the members of
//! other nets near it on those layers and the vias of other nets it does
//! not pair with (`repel`). The second, the clearance hinge alone, which
//! limits without repelling, covers what only stops the via: the outside
//! of its zones' regions and the fabrication rules that hold within its
//! net, and the moving vias of its net (`stop`). Each term's gradient also
//! reaches every edge or via that answers it, by its share
//! (`members::pull_answers`). The pairs of ground vias are the pairwise
//! term's (`vias::pairs::ViaPairs`).
//!
//! A burst's objective (`Objective`) takes the problem's variables, lays
//! them out into the members' polylines and the vias' centres (`Layout`),
//! and keeps what it searched between evaluations (`Lookups`). Each
//! evaluation counts in `stats`.

use std::sync::Arc;

use crate::contacts::contact_stops;
use crate::energy::EnergyParams;
use crate::energy::integrands::{Curvature, Integrand};
use crate::energy::quadrature::point_stiffness;
use crate::field::Field;
use crate::field::smooth::combine_fields;
use crate::geometry::V2;
use crate::members::{Energy, Members, pull_answers};
use crate::neighbours::{NeighbourList, StaticList};
use crate::numeric::{exact_sum, maximum};
use crate::seen::Seen;
use crate::stats;
use crate::vias::pairs::ViaPairs;
use crate::vias::sight::{ViaSight, ViaSmooth};

/// A moving via of a problem: its diameter, which weighs its terms, its
/// static field and what only stops it.
#[derive(Clone)]
pub struct ProblemVia {
    pub diameter: f64,
    pub field: Arc<dyn Field>,
    pub stops: Option<Arc<dyn Field>>,
}

/// What one objective's evaluations search through: the members' pairs
/// with each other, and the static fields of the members and of the
/// moving vias, those of each via term one after another.
#[derive(Clone)]
pub struct Lookups {
    pub neighbours: NeighbourList,
    pub members: StaticList,
    pub vias: StaticList,
}

/// What one via term sees of each moving via.
struct ViaField {
    /// What its sight finds.
    near: ViaSmooth,
    /// The smooth distance over that and the term's static field, and its
    /// gradient.
    distance: Vec<f64>,
    direction: Vec<V2>,
    /// The sight's share of it.
    seen_share: Vec<f64>,
}

/// The members, moving vias and terms of one problem.
pub struct Problem {
    pub members: Members,
    /// Per via, the members it sees and that see it, and how far their
    /// centrelines keep from its centre.
    pub rings: Vec<Vec<(usize, f64)>>,
    pub vias: Vec<ProblemVia>,
    /// What the vias' full terms see of the members and of each other.
    pub repel: ViaSight,
    /// What the vias' stops see of each other.
    pub stop: ViaSight,
    /// The pairwise term, None without a pair, so that the energy is the
    /// one without ground vias term for term.
    pub pairs: Option<Arc<ViaPairs>>,
    pub params: EnergyParams,
    pub short_edge: f64,
}

fn refs(polylines: &[Vec<V2>]) -> Vec<&[V2]> {
    polylines.iter().map(Vec::as_slice).collect()
}

impl Problem {
    /// What each member sees of the others and of the vias with the members
    /// at `polylines`.
    pub fn seen(&self, polylines: &[&[V2]]) -> Seen {
        self.members.seen(polylines, &self.rings)
    }

    /// The lists for the evaluations of one objective, made for `seen`.
    pub fn lookups(&self, seen: &Seen) -> Lookups {
        let reach = self.params.obstacle_reach();
        Lookups {
            neighbours: NeighbourList::new(seen, reach),
            members: self.members.static_list(reach, true),
            vias: self.via_static_list(true),
        }
    }

    /// The moving vias' static fields, those of each via term one after
    /// another.
    fn via_static_list(&self, listed: bool) -> StaticList {
        let fields = self
            .vias
            .iter()
            .map(|v| Some(v.field.clone()))
            .chain(self.vias.iter().map(|v| v.stops.clone()))
            .collect();
        StaticList::new(fields, self.params.obstacle_reach(), listed)
    }

    /// The two via terms: what each sees, its integrand and its curvature.
    fn via_parts(&self) -> [(&ViaSight, Integrand, Curvature); 2] {
        [
            (
                &self.repel,
                Integrand::obstacle(&self.params),
                Curvature::obstacle(&self.params),
            ),
            (
                &self.stop,
                Integrand::hinge(&self.params),
                Curvature::hinge(&self.params),
            ),
        ]
    }

    /// The problem's energy with the members at `polylines` and the vias at
    /// `positions`, what each member sees of the others as `seen` gives it,
    /// and its gradient at every vertex and every via's centre: the members'
    /// terms with the vertices of `masks` zeroed, then the vias' own terms,
    /// the pairwise term and each vertex's contact stop by `contact_of`,
    /// its index among its member's contacts. With `lookups`, made for
    /// `seen`, the searches come from them, by default each its own. Err
    /// with the members' value if it is not finite.
    pub fn total(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        masks: &[&[bool]],
        contact_of: &[&[Option<usize>]],
        seen: &Seen,
        lookups: Option<&mut Lookups>,
    ) -> Result<Energy, f64> {
        let (neighbours, members, vias) = match lookups {
            Some(l) => (
                Some(&mut l.neighbours),
                Some(&mut l.members),
                Some(&mut l.vias),
            ),
            None => (None, None, None),
        };
        let Energy {
            mut value,
            mut gradients,
            via_gradients,
        } = self.members.energy(
            polylines,
            masks,
            &self.params,
            self.short_edge,
            seen,
            positions,
            neighbours,
            members,
        )?;
        let own = self.via_energy(polylines, positions, vias);
        for (g, pull) in gradients.iter_mut().zip(&own.gradients) {
            for (a, &b) in g.iter_mut().zip(pull) {
                *a = *a + b;
            }
        }
        let mut own_value = own.value;
        let mut own_gradients = own.via_gradients;
        if let Some(pairs) = &self.pairs {
            let (paired, paired_gradients) = pairs.energy(positions);
            own_value += paired;
            for (a, b) in own_gradients.iter_mut().zip(paired_gradients) {
                *a = *a + b;
            }
        }
        let limit = Integrand::hinge(&self.params);
        for (i, member) in self.members.members.iter().enumerate() {
            if member.contacts.is_empty() {
                continue;
            }
            let within: Vec<&dyn Field> = member.contacts.iter().map(Arc::as_ref).collect();
            let (stop, push) =
                contact_stops(polylines[i], contact_of[i], &within, member.width, &limit);
            value += stop;
            for (a, b) in gradients[i].iter_mut().zip(push) {
                *a = *a + b;
            }
        }
        Ok(Energy {
            value: value + own_value,
            gradients,
            via_gradients: via_gradients
                .iter()
                .zip(&own_gradients)
                .map(|(&a, &b)| a + b)
                .collect(),
        })
    }

    /// The smooth distance and its gradient of each moving via to what the
    /// static field of each via term gives of it, from `statics`, made by
    /// `lookups`, by default from one search each.
    fn via_statics(
        &self,
        positions: &[V2],
        statics: Option<&mut StaticList>,
    ) -> Vec<(Vec<f64>, Vec<V2>)> {
        let mut own;
        let statics = match statics {
            Some(s) => s,
            None => {
                own = self.via_static_list(false);
                &mut own
            }
        };
        let rows: Vec<&[V2]> = (0..2).flat_map(|_| positions.chunks(1)).collect();
        let answers = statics.smooth(&rows);
        let count = positions.len();
        (0..2)
            .map(|part| {
                answers[part * count..(part + 1) * count]
                    .iter()
                    .map(|(d, g)| (d[0], g[0]))
                    .unzip()
            })
            .collect()
    }

    /// What each via term sees of each moving via.
    fn via_fields(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        statics: Option<&mut StaticList>,
    ) -> Vec<ViaField> {
        let reach = self.params.obstacle_reach();
        let statics = self.via_statics(positions, statics);
        self.via_parts()
            .iter()
            .zip(statics)
            .map(|((sight, _, _), (fixed, toward))| {
                let near = sight.smooth_answers(polylines, positions, reach);
                let (distance, shares) = combine_fields(&[&fixed, &near.distance]);
                let direction = toward
                    .iter()
                    .zip(&near.gradient)
                    .zip(shares[0].iter().zip(&shares[1]))
                    .map(|((g, h), (&s, &t))| V2::new(s * g.x + t * h.x, s * g.y + t * h.y))
                    .collect();
                let seen_share = shares.into_iter().nth(1).unwrap_or_default();
                ViaField {
                    near,
                    distance,
                    direction,
                    seen_share,
                }
            })
            .collect()
    }

    /// The moving vias' own obstacle terms, each via's ring standing for a
    /// track as long as it is wide, and their gradient at every vertex and
    /// at every via's centre. Where a member's edge or another via answers
    /// a via, the gradient also pulls on it. `statics` is as for
    /// `via_statics`.
    pub fn via_energy(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        statics: Option<&mut StaticList>,
    ) -> Energy {
        let mut gradients: Vec<Vec<V2>> = polylines
            .iter()
            .map(|p| vec![V2::default(); p.len()])
            .collect();
        let mut via_gradients = vec![V2::default(); self.vias.len()];
        if self.vias.is_empty() {
            return Energy {
                value: 0.0,
                gradients,
                via_gradients,
            };
        }
        let mut value = 0.0;
        for ((_, integrand, _), field) in self
            .via_parts()
            .iter()
            .zip(self.via_fields(polylines, positions, statics))
        {
            let mut terms = Vec::with_capacity(self.vias.len());
            let mut weighted = Vec::with_capacity(self.vias.len());
            for (via, &d) in self.vias.iter().zip(&field.distance) {
                let (term, slope) = integrand.at(d);
                terms.push(via.diameter * term);
                weighted.push(via.diameter * slope);
            }
            // numpy left this dot product to its BLAS, whose order no
            // platform shares; rounded once, it is the same everywhere.
            value += exact_sum(terms);
            for ((g, &w), &d) in via_gradients
                .iter_mut()
                .zip(&weighted)
                .zip(&field.direction)
            {
                *g = *g + V2::new(w * d.x, w * d.y);
            }
            let pull: Vec<f64> = weighted
                .iter()
                .zip(&field.seen_share)
                .map(|(&w, &s)| w * s)
                .collect();
            pull_answers(
                &field.near,
                &pull,
                positions,
                &mut gradients,
                &mut via_gradients,
            );
        }
        Energy {
            value,
            gradients,
            via_gradients,
        }
    }

    /// Each moving via's stiffness under its own obstacle terms: the larger
    /// over x and y of their `point_stiffness` summed. `statics` is as for
    /// `via_statics`.
    pub fn via_stiffness(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        statics: Option<&mut StaticList>,
    ) -> Vec<f64> {
        if self.vias.is_empty() {
            return Vec::new();
        }
        let mut stiffness = vec![V2::default(); self.vias.len()];
        for ((_, integrand, curvature), field) in self
            .via_parts()
            .iter()
            .zip(self.via_fields(polylines, positions, statics))
        {
            for (k, via) in self.vias.iter().enumerate() {
                let s =
                    point_stiffness(field.distance[k], field.direction[k], integrand, curvature);
                stiffness[k] = stiffness[k] + V2::new(via.diameter * s.x, via.diameter * s.y);
            }
        }
        stiffness.into_iter().map(|s| maximum(s.x, s.y)).collect()
    }

    /// Per member, each vertex's stiffness under its obstacle term
    /// (`members::Members::stiffness`), and each via's under its own terms
    /// (`via_stiffness`), at `polylines` and `positions`, searched through
    /// `lookups` as `total` searches.
    pub fn stiffness(
        &self,
        polylines: &[&[V2]],
        positions: &[V2],
        seen: &Seen,
        lookups: Option<&mut Lookups>,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let (neighbours, members, vias) = match lookups {
            Some(l) => (
                Some(&mut l.neighbours),
                Some(&mut l.members),
                Some(&mut l.vias),
            ),
            None => (None, None, None),
        };
        let per_member = self.members.stiffness(
            polylines,
            &self.params,
            seen,
            positions,
            neighbours,
            members,
        );
        (per_member, self.via_stiffness(polylines, positions, vias))
    }
}

/// How a burst's variables make the members' polylines and the vias'
/// centres: each via's centre is a variable, and an end it carries is that
/// variable plus the end's offset.
#[derive(Clone, Debug)]
pub struct Layout {
    pub starts: Vec<Vec<V2>>,
    /// Per member, each vertex's variable, None where it stays.
    pub index: Vec<Vec<Option<usize>>>,
    /// Per member, each vertex's offset from its variable.
    pub offsets: Vec<Vec<V2>>,
    /// Each via's variable.
    pub via_index: Vec<usize>,
    pub count: usize,
}

impl Layout {
    /// The members' polylines and the vias' centres at `x`, flat.
    pub fn unflatten(&self, x: &[f64]) -> (Vec<Vec<V2>>, Vec<V2>) {
        let at = |i: usize| V2::new(x[2 * i], x[2 * i + 1]);
        let polylines = self
            .starts
            .iter()
            .zip(&self.index)
            .zip(&self.offsets)
            .map(|((start, index), offset)| {
                start
                    .iter()
                    .zip(index)
                    .zip(offset)
                    .map(|((&p, i), &o)| i.map_or(p, |i| at(i) + o))
                    .collect()
            })
            .collect();
        let positions = self.via_index.iter().map(|&v| at(v)).collect();
        (polylines, positions)
    }

    /// The gradient by the variables, flat, from that at every vertex and
    /// at every via's centre.
    pub fn gradient(&self, gradients: &[Vec<V2>], via_gradients: &[V2]) -> Vec<f64> {
        let mut gradient = vec![V2::default(); self.count];
        for (g, index) in gradients.iter().zip(&self.index) {
            for (&g, i) in g.iter().zip(index) {
                if let Some(i) = *i {
                    gradient[i] = gradient[i] + g;
                }
            }
        }
        for (&v, &g) in self.via_index.iter().zip(via_gradients) {
            gradient[v] = gradient[v] + g;
        }
        gradient.into_iter().flat_map(|g| [g.x, g.y]).collect()
    }
}

/// A burst's energy as a function of its layout's variables, with what each
/// member sees of the others as `seen`, its energy masks and contacts.
pub struct Objective {
    pub problem: Arc<Problem>,
    pub seen: Arc<Seen>,
    pub layout: Layout,
    pub masks: Vec<Vec<bool>>,
    pub contact_of: Vec<Vec<Option<usize>>>,
}

impl Objective {
    /// The energy at `x` and its gradient by the variables, searching
    /// through `lookups`; Err with the members' value if it is not finite.
    pub fn evaluate(
        &self,
        x: &[f64],
        lookups: Option<&mut Lookups>,
    ) -> Result<(f64, Vec<f64>), f64> {
        stats::evaluated();
        let (polylines, positions) = self.layout.unflatten(x);
        let masks: Vec<&[bool]> = self.masks.iter().map(Vec::as_slice).collect();
        let contact_of: Vec<&[Option<usize>]> = self.contact_of.iter().map(Vec::as_slice).collect();
        let energy = self.problem.total(
            &refs(&polylines),
            &positions,
            &masks,
            &contact_of,
            &self.seen,
            lookups,
        )?;
        Ok((
            energy.value,
            self.layout
                .gradient(&energy.gradients, &energy.via_gradients),
        ))
    }
}
