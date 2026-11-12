//! The pairwise via term: what a ground via and a regular one, or ground
//! vias of two ground nets, give each other, summed over every pair in
//! range.
//!
//! A ground via is on one of the run's ground nets, a regular one on any
//! other net. Two vias interact here as `pair_kind` says, only where their
//! layer spans overlap, and each pair is counted from the side of each
//! moving via in it, weighted by that via's diameter, as the optimizer's
//! other via terms are; a via standing still has no side of its own.
//!
//! - A ground via and a regular one: the full obstacle term
//!   (`Integrand::obstacle`), repulsion and clearance hinge, at the
//!   distance by which their centres are farther apart than they keep,
//!   plus the attraction (`Attraction`), continued linearly below that
//!   distance. Together they give each pair a neutral distance.
//! - Ground vias of two ground nets: the full obstacle term alone.
//! - Ground vias of one ground net: the repulsion alone
//!   (`Integrand::repulsion`), beyond their rings and the netclass
//!   clearance, so that stitching vias drawn toward a signal via spread
//!   around it. It has no clearance hinge: their only limit stays
//!   hole-to-hole, which the optimizer's stops keep, so it pushes apart
//!   without becoming a wall that freezes or blocks them.
//!
//! Summed over pairs rather than taken for the nearest only: against
//! summed attraction, nearest-only repulsion has no neutral distance once
//! anything else is nearer. The pairs of other nets are therefore left out
//! of the optimizer's nearest-only via terms and out of a moving via's
//! static field, which then holds none of the vias standing still that it
//! pairs with (`field::build::via_obstacles`). Clearance judging is
//! unchanged: the pairs keep the distances `rules::via_via_inflation`
//! gives.
//!
//! The problem's vias are paired once, at setup, at the distances the
//! optimizer gives, and only the pairs whose centres are within reach of
//! each other give energy at an evaluation. The vias standing still sit in
//! a tree built once, at setup, which finds those near each moving via at
//! every evaluation, so the energy is one function of the positions however
//! far the vias move; the distance a moving via keeps from one is worked
//! out where they meet.
//!
//! The term's value is the correctly rounded sum of its pairs' weighted
//! terms (`numeric::exact_sum`); each via's gradient adds its pairs' pushes
//! in the order of the pairs, as its first via, then as its second, then
//! from the vias standing still in the order they stand.

use std::collections::{HashMap, HashSet};

use rstar::RTree;
use rstar::primitives::GeomWithData;

use crate::energy::EnergyParams;
use crate::energy::integrands::{Attraction, Integrand};
use crate::geometry::V2;
use crate::numeric::exact_sum;
use crate::rules::{HoleRules, Via, Zone, via_via_inflation};

/// How two vias interact in the pairwise term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairKind {
    /// A ground via and a regular one: attraction and the full obstacle
    /// term.
    Return = 0,
    /// Ground vias of two ground nets: the full obstacle term.
    Grounds = 1,
    /// Ground vias of one ground net: the repulsion alone, beyond their
    /// rings and clearance.
    Spread = 2,
}

/// How `via` and `other`, each a ground via or not, interact in the
/// pairwise term; None if they do not.
pub fn pair_kind(via: &Via, ground: bool, other: &Via, other_ground: bool) -> Option<PairKind> {
    if !via.shares_layer(other) {
        return None;
    }
    if via.net == other.net {
        return ground.then_some(PairKind::Spread);
    }
    match (ground, other_ground) {
        (true, true) => Some(PairKind::Grounds),
        (false, false) => None,
        _ => Some(PairKind::Return),
    }
}

/// A via of the pairwise term: whether it is a ground via, its netclass
/// clearance, and for a moving via the zones it stays inside.
#[derive(Clone, Debug)]
pub struct PairVia {
    pub via: Via,
    pub ground: bool,
    pub clearance: f64,
    pub zones: Vec<Zone>,
}

/// A pair of the problem's moving vias, from the side of its first.
struct Pair {
    first: usize,
    second: usize,
    /// The distance their centres keep, or for vias of one ground net the
    /// distance beyond which they repel.
    apart: f64,
    kind: PairKind,
}

/// The centre distance beyond which `given` and `other`, ground vias of
/// one net, repel each other: their rings and the larger of their
/// clearances.
fn spread(given: &PairVia, other: &PairVia) -> f64 {
    let rings = (given.via.diameter + other.via.diameter) / 2.0;
    rings + py_max(given.clearance, other.clearance)
}

/// Python's `max(a, b)`: `a` unless `b` is greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

type StaticPoint = GeomWithData<[f64; 2], usize>;

/// The pairwise term of a problem's moving vias, with each other and with
/// the vias standing still.
pub struct ViaPairs {
    vias: Vec<PairVia>,
    pairs: Vec<Pair>,
    reach: f64,
    paired: Vec<HashSet<usize>>,
    /// The vias standing still that pair with some moving via.
    statics: Vec<PairVia>,
    /// By moving via, then static via, how they pair.
    static_kinds: Vec<Vec<Option<PairKind>>>,
    tree: RTree<StaticPoint>,
    /// By moving via, how far from its centre a static via may give it
    /// energy.
    static_reach: Vec<f64>,
    holes: HoleRules,
    obstacle: Integrand,
    repulsion: Integrand,
    attraction: Attraction,
}

impl ViaPairs {
    /// The pairs of the problem's moving `vias`, which keep the distances
    /// `apart` gives from each other, and of them with the `statics`,
    /// which keep the distances `rules::via_via_inflation` gives with
    /// `holes`.
    pub fn new(
        vias: Vec<PairVia>,
        apart: &[HashMap<usize, f64>],
        statics: Vec<PairVia>,
        params: &EnergyParams,
        holes: HoleRules,
    ) -> Self {
        let mut pairs = Vec::new();
        for (k, given) in vias.iter().enumerate() {
            for (u, other) in vias.iter().enumerate() {
                let Some(kind) = pair_kind(&given.via, given.ground, &other.via, other.ground)
                else {
                    continue;
                };
                if u == k {
                    continue;
                }
                let apart = if kind == PairKind::Spread {
                    spread(given, other)
                } else {
                    apart[k].get(&u).copied().unwrap_or(f64::NAN)
                };
                pairs.push(Pair {
                    first: k,
                    second: u,
                    apart,
                    kind,
                });
            }
        }
        let reach = params.obstacle_reach();
        let cutoff = if params.attraction != 0.0 {
            params.attraction_range
        } else {
            0.0
        };
        let pair_reach = pairs
            .iter()
            .map(|p| {
                if p.kind == PairKind::Return {
                    py_max(cutoff, p.apart + reach)
                } else {
                    p.apart + reach
                }
            })
            .reduce(py_max)
            .unwrap_or(0.0);
        let mut paired = vec![HashSet::new(); vias.len()];
        for p in &pairs {
            paired[p.first].insert(p.second);
        }
        let all_kinds: Vec<Vec<Option<PairKind>>> = vias
            .iter()
            .map(|v| {
                statics
                    .iter()
                    .map(|s| pair_kind(&v.via, v.ground, &s.via, s.ground))
                    .collect()
            })
            .collect();
        let met: Vec<bool> = (0..statics.len())
            .map(|s| all_kinds.iter().any(|row| row[s].is_some()))
            .collect();
        let statics: Vec<PairVia> = statics
            .into_iter()
            .zip(&met)
            .filter_map(|(s, &m)| m.then_some(s))
            .collect();
        let static_kinds: Vec<Vec<Option<PairKind>>> = all_kinds
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(&met)
                    .filter_map(|(k, &m)| m.then_some(k))
                    .collect()
            })
            .collect();
        let tree = RTree::bulk_load(
            statics
                .iter()
                .enumerate()
                .map(|(s, v)| StaticPoint::new(v.via.position.array(), s))
                .collect(),
        );
        let mut found = Self {
            vias,
            pairs,
            reach: pair_reach,
            paired,
            statics,
            static_kinds,
            tree,
            static_reach: Vec::new(),
            holes,
            obstacle: Integrand::obstacle(params),
            repulsion: Integrand::repulsion(params),
            attraction: Attraction::new(params),
        };
        found.static_reach = (0..found.vias.len())
            .map(|k| py_max(cutoff, found.widest(k) + reach))
            .collect();
        found
    }

    /// The number of moving vias.
    pub fn via_count(&self) -> usize {
        self.vias.len()
    }

    /// Whether the term has no pair.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty() && self.statics.is_empty()
    }

    /// Per via, the problem's vias it pairs with.
    pub fn paired(&self) -> &[HashSet<usize>] {
        &self.paired
    }

    /// The term's value with the problem's vias at `positions`, and its
    /// gradient at each via's centre.
    pub fn energy(&self, positions: &[V2]) -> (f64, Vec<V2>) {
        let n = positions.len();
        let mut as_first = vec![V2::default(); n];
        let mut as_second = vec![V2::default(); n];
        let mut from_static = vec![V2::default(); n];
        let mut values = Vec::new();
        let squared_reach = self.reach * self.reach;
        for p in &self.pairs {
            let delta = positions[p.first] - positions[p.second];
            if delta.dot(delta) > squared_reach {
                continue;
            }
            let weight = self.vias[p.first].via.diameter;
            let (value, push) = self.term(delta, p.apart, p.kind, weight);
            values.push(value);
            as_first[p.first] = as_first[p.first] + push;
            as_second[p.second] = as_second[p.second] + -push;
        }
        if !self.statics.is_empty() {
            for (k, &at) in positions.iter().enumerate() {
                let reach = self.static_reach[k];
                let mut near: Vec<usize> = self
                    .tree
                    .locate_within_distance(at.array(), reach * reach)
                    .map(|s| s.data)
                    .filter(|&s| self.static_kinds[k][s].is_some())
                    .collect();
                near.sort_unstable();
                for s in near {
                    let Some(kind) = self.static_kinds[k][s] else {
                        continue;
                    };
                    let other = &self.statics[s];
                    let apart = if kind == PairKind::Spread {
                        spread(&self.vias[k], other)
                    } else {
                        self.kept(k, &other.via, other.clearance)
                            .unwrap_or(f64::NAN)
                    };
                    let delta = at - other.via.position;
                    let weight = self.vias[k].via.diameter;
                    let (value, push) = self.term(delta, apart, kind, weight);
                    values.push(value);
                    from_static[k] = from_static[k] + push;
                }
            }
        }
        let gradient = (0..n)
            .map(|k| (V2::default() + as_first[k]) + as_second[k] + from_static[k])
            .collect();
        (exact_sum(values), gradient)
    }

    /// One pair's weighted value and its gradient at its first via's
    /// centre, from the vector `delta` from its second via's centre to its
    /// first's, the distance `apart` their centres keep, its `kind` and the
    /// first via's `weight`.
    fn term(&self, delta: V2, apart: f64, kind: PairKind, weight: f64) -> (f64, V2) {
        let r = delta.norm();
        let (mut value, mut slope) = match kind {
            PairKind::Spread => self.repulsion.at(r - apart),
            _ => self.obstacle.at(r - apart),
        };
        if kind == PairKind::Return {
            let (pull, pull_slope) = self.attraction.at(r, apart);
            value += pull;
            slope += pull_slope;
        }
        let unit = if r > 0.0 {
            V2::new(delta.x / r, delta.y / r)
        } else {
            V2::default()
        };
        (weight * value, (weight * slope) * unit)
    }

    /// The distance moving via k's centre keeps from `other`, of netclass
    /// `clearance`, as its static field would count it: with its zones on
    /// the layers `other` spans. Paired vias share a layer, where vias of
    /// other nets keep their copper clearance; None if they share none.
    pub fn kept(&self, k: usize, other: &Via, clearance: f64) -> Option<f64> {
        let given = &self.vias[k];
        via_via_inflation(
            &given.via,
            given.clearance,
            other,
            clearance,
            self.holes,
            &given.zones,
            &[],
        )
    }

    /// The largest distance via k's centre keeps from a static via it
    /// pairs with, or beyond which one of its net repels it (`spread`),
    /// which is no larger: from one as wide, as widely drilled and of as
    /// wide a clearance as the widest, of a net other than its own.
    fn widest(&self, k: usize) -> f64 {
        if self.statics.is_empty() {
            return 0.0;
        }
        let given = &self.vias[k];
        let widest = Via {
            net: format!("{}\0other", given.via.net),
            diameter: self
                .statics
                .iter()
                .map(|s| s.via.diameter)
                .reduce(py_max)
                .unwrap_or(0.0),
            drill: self
                .statics
                .iter()
                .map(|s| s.via.drill)
                .reduce(py_max)
                .unwrap_or(0.0),
            ..given.via.clone()
        };
        let clearance = self
            .statics
            .iter()
            .map(|s| s.clearance)
            .fold(given.clearance, py_max);
        via_via_inflation(
            &given.via,
            clearance,
            &widest,
            clearance,
            self.holes,
            &given.zones,
            &[],
        )
        .unwrap_or(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::DEFAULT_ATTRACTION_RANGE;
    use crate::energy::tests::params;
    use crate::rules::{HOLE_CLEARANCE, HOLE_TO_HOLE, Span};

    const HOLES: HoleRules = HoleRules {
        hole_clearance: HOLE_CLEARANCE,
        hole_to_hole: HOLE_TO_HOLE,
    };

    fn via(id: &str, net: &str, ground: bool, x: f64) -> PairVia {
        PairVia {
            via: Via {
                id: id.into(),
                net: net.into(),
                position: V2::new(x, 0.0),
                diameter: 0.6,
                drill: 0.3,
                span: Span { low: 0, high: 2 },
            },
            ground,
            clearance: 0.2,
            zones: Vec::new(),
        }
    }

    fn attracting() -> EnergyParams {
        EnergyParams {
            attraction: 0.25,
            attraction_range: DEFAULT_ATTRACTION_RANGE,
            ..params()
        }
    }

    #[test]
    fn kinds_follow_ground_and_net() {
        let g = via("g", "GND", true, 0.0);
        let h = via("h", "GND", true, 1.0);
        let p = via("p", "PWR", true, 1.0);
        let s = via("s", "SIG", false, 1.0);
        let t = via("t", "SIG2", false, 1.0);
        assert_eq!(
            pair_kind(&g.via, true, &h.via, true),
            Some(PairKind::Spread)
        );
        assert_eq!(pair_kind(&s.via, false, &s.via, false), None);
        assert_eq!(
            pair_kind(&g.via, true, &p.via, true),
            Some(PairKind::Grounds)
        );
        assert_eq!(
            pair_kind(&g.via, true, &s.via, false),
            Some(PairKind::Return)
        );
        assert_eq!(pair_kind(&s.via, false, &t.via, false), None);
        let far = Via {
            span: Span { low: 3, high: 4 },
            ..t.via.clone()
        };
        assert_eq!(pair_kind(&g.via, true, &far, false), None);
    }

    fn pulled(at: f64) -> V2 {
        let ground = via("g", "GND", true, 0.0);
        let signal = via("s", "SIG", false, at);
        let pairs = ViaPairs::new(
            vec![ground],
            &[HashMap::new()],
            vec![signal],
            &attracting(),
            HOLES,
        );
        pairs.energy(&[V2::new(0.0, 0.0)]).1[0]
    }

    #[test]
    fn a_ground_via_and_a_regular_one_attract_within_range_only() {
        // The ground via at the origin is drawn toward the signal via at +x.
        assert!(pulled(2.0).x > 0.0);
        assert_eq!(pulled(DEFAULT_ATTRACTION_RANGE + 0.5), V2::default());
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let vias = vec![via("g", "GND", true, 0.0), via("h", "GND", true, 0.9)];
        let statics = vec![via("s", "SIG", false, 2.0), via("p", "PWR", true, -1.2)];
        let apart = [HashMap::new(), HashMap::new()];
        let pairs = ViaPairs::new(vias, &apart, statics, &attracting(), HOLES);
        let at = [V2::new(0.1, 0.2), V2::new(0.9, -0.1)];
        let (value, gradient) = pairs.energy(&at);
        assert!(value > 0.0);
        let step = 1e-7;
        for k in 0..2 {
            for axis in 0..2 {
                let mut moved = at;
                if axis == 0 {
                    moved[k].x += step;
                } else {
                    moved[k].y += step;
                }
                let difference = (pairs.energy(&moved).0 - value) / step;
                let want = if axis == 0 {
                    gradient[k].x
                } else {
                    gradient[k].y
                };
                assert!(
                    (difference - want).abs() < 1e-4 * want.abs().max(1.0),
                    "{k} {axis}"
                );
            }
        }
    }
}
