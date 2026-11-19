//! The pairwise via term: what a ground via and a regular one, or ground
//! vias of two ground nets, give each other, and what plated through-hole
//! pads standing still give the moving vias, summed over every pair in
//! range.
//!
//! A ground via is on one of the run's ground nets, a regular one on any
//! other net. Two vias interact here as `pair_kind` says, only where their
//! layer spans overlap, and each pair is counted from the side of each
//! moving via in it, weighted by that via's diameter, as the optimizer's
//! other via terms are; a via or pad standing still has no side of its
//! own.
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
//! A pad's barrel connects the planes as a via does, so a moving via and a
//! pad interact as `pad_kind` says, at the barrel distance: the centre
//! distance less the pad's drill radius.
//!
//! - A ground side and a regular one: the attraction alone, continued
//!   linearly below the closest barrel distance the hole rules allow
//!   (`barrel_limit`). There is no obstacle term: the pad stays in the
//!   moving via's static field, which keeps clearance from its copper, so
//!   a via drawn to a pad settles where the field's repulsion balances the
//!   pull, or against its clearance hinge.
//! - A ground via and a pad of its ground net: the repulsion alone, beyond
//!   the via's ring and the larger netclass clearance, as two vias of one
//!   ground net spread, with the barrel in place of the pad's ring.
//! - Anything else, two ground nets or two regular ones: nothing.
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
//! each other give energy at an evaluation. The vias and pads standing
//! still sit in two trees built once, at setup, which find those near each
//! moving via at every evaluation, so the energy is one function of the
//! positions however far the vias move; the distance a moving via keeps
//! from a static via is worked out where they meet. A pad's reach grows by
//! the widest pad drill's radius, as its terms run from the barrel.
//!
//! The term's value is the correctly rounded sum of its pairs' weighted
//! terms (`numeric::exact_sum`); each via's gradient adds its pairs' pushes
//! in the order of the pairs, as its first via, then as its second, then
//! from the vias standing still in the order they stand, then from the
//! pads in the order they stand.

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

/// A plated through-hole pad standing still, as the pairwise term takes
/// it: its centre, its drill's diameter, its net, whether that is a ground
/// net, and the net's netclass clearance. Its barrel spans every layer.
#[derive(Clone, Debug)]
pub struct PairPad {
    pub position: V2,
    pub drill: f64,
    pub net: String,
    pub ground: bool,
    pub clearance: f64,
}

/// How a moving via and a pad interact in the pairwise term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadKind {
    /// A ground side and a regular one: the attraction alone.
    Attract,
    /// A ground via and a pad of its net: the repulsion alone, beyond the
    /// via's ring and the larger netclass clearance.
    Spread,
}

/// How moving `given` and `pad` interact in the pairwise term; None if
/// they do not.
pub fn pad_kind(given: &PairVia, pad: &PairPad) -> Option<PadKind> {
    if given.via.net == pad.net {
        return given.ground.then_some(PadKind::Spread);
    }
    (given.ground != pad.ground).then_some(PadKind::Attract)
}

/// The closest a via's centre may come to another net's plated barrel by
/// the hole rules: its ring keeps the hole clearance from the hole's wall,
/// and its own hole keeps hole-to-hole. These are `Keeper::hole`'s two
/// terms for a hole of radius 0; the ring-to-copper clearance
/// `via_via_inflation` adds is left to the static field, which holds the
/// pad's copper. Positive, as a via's ring is.
fn barrel_limit(via: &Via, holes: HoleRules) -> f64 {
    py_max(
        via.diameter / 2.0 + holes.hole_clearance,
        via.drill / 2.0 + holes.hole_to_hole,
    )
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

/// By moving via, then standing item, how they pair.
type Kinds<K> = Vec<Vec<Option<K>>>;

/// The `items` standing still that pair with some moving via, and by
/// moving via how it pairs with each of them, from how it pairs with each
/// of `items` (`kinds`).
fn met_only<T, K>(items: Vec<T>, kinds: Kinds<K>) -> (Vec<T>, Kinds<K>) {
    let met: Vec<bool> = (0..items.len())
        .map(|s| kinds.iter().any(|row| row[s].is_some()))
        .collect();
    let items = items
        .into_iter()
        .zip(&met)
        .filter_map(|(s, &m)| m.then_some(s))
        .collect();
    let kinds = kinds
        .into_iter()
        .map(|row| {
            row.into_iter()
                .zip(&met)
                .filter_map(|(k, &m)| m.then_some(k))
                .collect()
        })
        .collect();
    (items, kinds)
}

type StaticPoint = GeomWithData<[f64; 2], usize>;

/// The items of `tree` whose centres lie within `reach` of `at` and that
/// pair with the moving via whose row of kinds is `row`, each with how it
/// pairs, in the order they stand.
fn near<K: Copy>(
    tree: &RTree<StaticPoint>,
    row: &[Option<K>],
    at: V2,
    reach: f64,
) -> Vec<(usize, K)> {
    let mut found: Vec<(usize, K)> = tree
        .locate_within_distance(at.array(), reach * reach)
        .filter_map(|s| row[s.data].map(|kind| (s.data, kind)))
        .collect();
    found.sort_unstable_by_key(|&(s, _)| s);
    found
}

/// A term of `value` and `slope` at distance r along `delta`, weighted by
/// `weight`, and its gradient at `delta`'s head.
fn weighted(weight: f64, value: f64, slope: f64, delta: V2, r: f64) -> (f64, V2) {
    let unit = if r > 0.0 {
        V2::new(delta.x / r, delta.y / r)
    } else {
        V2::default()
    };
    (weight * value, (weight * slope) * unit)
}

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
    static_kinds: Kinds<PairKind>,
    tree: RTree<StaticPoint>,
    /// By moving via, how far from its centre a static via may give it
    /// energy.
    static_reach: Vec<f64>,
    /// The pads that pair with some moving via.
    pads: Vec<PairPad>,
    /// By moving via, then pad, how they pair.
    pad_kinds: Kinds<PadKind>,
    pad_tree: RTree<StaticPoint>,
    /// By moving via, how far from its centre a pad's centre may give it
    /// energy.
    pad_reach: Vec<f64>,
    /// By moving via, the barrel distance below which a pad's attraction
    /// continues linearly (`barrel_limit`).
    pad_limits: Vec<f64>,
    holes: HoleRules,
    obstacle: Integrand,
    repulsion: Integrand,
    attraction: Attraction,
}

impl ViaPairs {
    /// The pairs of the problem's moving `vias`, which keep the distances
    /// `apart` gives from each other, and of them with the `statics`,
    /// which keep the distances `rules::via_via_inflation` gives with
    /// `holes`, and of them with the `pads`.
    pub fn new(
        vias: Vec<PairVia>,
        apart: &[HashMap<usize, f64>],
        statics: Vec<PairVia>,
        pads: Vec<PairPad>,
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
        let all_kinds: Kinds<PairKind> = vias
            .iter()
            .map(|v| {
                statics
                    .iter()
                    .map(|s| pair_kind(&v.via, v.ground, &s.via, s.ground))
                    .collect()
            })
            .collect();
        let (statics, static_kinds) = met_only(statics, all_kinds);
        let tree = RTree::bulk_load(
            statics
                .iter()
                .enumerate()
                .map(|(s, v)| StaticPoint::new(v.via.position.array(), s))
                .collect(),
        );
        let pad_kinds = vias
            .iter()
            .map(|v| pads.iter().map(|p| pad_kind(v, p)).collect())
            .collect();
        let (pads, pad_kinds) = met_only(pads, pad_kinds);
        let pad_tree = RTree::bulk_load(
            pads.iter()
                .enumerate()
                .map(|(s, p)| StaticPoint::new(p.position.array(), s))
                .collect(),
        );
        let widest_hole = pads.iter().map(|p| p.drill / 2.0).fold(0.0, py_max);
        let widest_clearance = pads.iter().map(|p| p.clearance).fold(0.0, py_max);
        let pad_reach = vias
            .iter()
            .map(|v| {
                let spread = v.via.diameter / 2.0 + py_max(v.clearance, widest_clearance);
                widest_hole + py_max(cutoff, spread + reach)
            })
            .collect();
        let pad_limits = vias.iter().map(|v| barrel_limit(&v.via, holes)).collect();
        let mut found = Self {
            vias,
            pairs,
            reach: pair_reach,
            paired,
            statics,
            static_kinds,
            tree,
            static_reach: Vec::new(),
            pads,
            pad_kinds,
            pad_tree,
            pad_reach,
            pad_limits,
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
        self.pairs.is_empty() && self.statics.is_empty() && self.pads.is_empty()
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
                let row = &self.static_kinds[k];
                for (s, kind) in near(&self.tree, row, at, self.static_reach[k]) {
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
        if !self.pads.is_empty() {
            for (k, &at) in positions.iter().enumerate() {
                let row = &self.pad_kinds[k];
                for (s, kind) in near(&self.pad_tree, row, at, self.pad_reach[k]) {
                    let (value, push) = self.pad_term(k, at, &self.pads[s], kind);
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
        weighted(weight, value, slope, delta, r)
    }

    /// Moving via k's weighted term with `pad` at `at`, as its `kind`
    /// says, and its gradient there. Both act at the barrel distance, the
    /// centre distance less the pad's drill radius: the attraction alone,
    /// or the repulsion alone beyond the via's ring and the larger
    /// netclass clearance, as `spread` gives two vias.
    fn pad_term(&self, k: usize, at: V2, pad: &PairPad, kind: PadKind) -> (f64, V2) {
        let given = &self.vias[k];
        let delta = at - pad.position;
        let r = delta.norm();
        let barrel = r - pad.drill / 2.0;
        let (value, slope) = match kind {
            PadKind::Spread => {
                let ring = given.via.diameter / 2.0;
                self.repulsion
                    .at(barrel - (ring + py_max(given.clearance, pad.clearance)))
            }
            PadKind::Attract => self.attraction.at(barrel, self.pad_limits[k]),
        };
        weighted(given.via.diameter, value, slope, delta, r)
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
            Vec::new(),
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
        let pairs = ViaPairs::new(vias, &apart, statics, Vec::new(), &attracting(), HOLES);
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

    /// A pad of `net` with its centre at (x, 0) and a hole of `drill` mm.
    fn pad(net: &str, ground: bool, x: f64, drill: f64) -> PairPad {
        PairPad {
            position: V2::new(x, 0.0),
            drill,
            net: net.into(),
            ground,
            clearance: 0.2,
        }
    }

    const DRILL: f64 = 1.0;

    /// The term of `given` alone with `pads`: its value and gradient with
    /// `given` at (x, 0).
    fn with_pads(given: PairVia, pads: Vec<PairPad>, params: &EnergyParams, x: f64) -> (f64, V2) {
        let pairs = ViaPairs::new(
            vec![given],
            &[HashMap::new()],
            Vec::new(),
            pads,
            params,
            HOLES,
        );
        let (value, gradient) = pairs.energy(&[V2::new(x, 0.0)]);
        (value, gradient[0])
    }

    /// `given` at the origin with a pad of `net` 1.5 mm from its barrel, on
    /// its +x side.
    fn beside(given: PairVia, net: &str, ground: bool) -> (f64, V2) {
        let p = pad(net, ground, 1.5 + DRILL / 2.0, DRILL);
        with_pads(given, vec![p], &attracting(), 0.0)
    }

    #[test]
    fn a_ground_pad_draws_a_signal_via_toward_it() {
        let (value, gradient) = beside(via("s", "SIG", false, 0.0), "GND", true);
        assert!(value < 0.0);
        assert!(gradient.x < 0.0, "{gradient:?}");
        assert_eq!(gradient.y, 0.0);
        let p = pad("GND", true, 1.5 + DRILL / 2.0, DRILL);
        let stepped = with_pads(
            via("s", "SIG", false, 0.0),
            vec![p],
            &attracting(),
            -0.01 * gradient.x,
        );
        assert!(stepped.0 < value);
    }

    #[test]
    fn a_signal_pad_draws_a_ground_via_toward_it() {
        let (value, gradient) = beside(via("g", "GND", true, 0.0), "SIG", false);
        assert!(value < 0.0);
        assert!(gradient.x < 0.0, "{gradient:?}");
    }

    #[test]
    fn a_ground_pad_of_its_net_pushes_a_ground_via_away_without_attraction() {
        let given = via("g", "GND", true, 0.0);
        let p = pad("GND", true, 1.0 + DRILL / 2.0, DRILL);
        let (value, gradient) = with_pads(given.clone(), vec![p.clone()], &attracting(), 0.0);
        assert!(value > 0.0);
        assert!(gradient.x > 0.0, "{gradient:?}");
        let repelled = with_pads(given, vec![p], &params(), 0.0);
        assert_eq!((value, gradient), repelled);
        // The barrel distance less the via's ring radius and the clearance.
        let (want, _) = Integrand::repulsion(&params()).at(1.0 - 0.3 - 0.2);
        assert!((value - 0.6 * want).abs() < 1e-12, "{value} {want}");
    }

    #[test]
    fn other_ground_nets_and_signal_pairs_give_nothing() {
        for (given, net, ground) in [
            (via("g", "GND", true, 0.0), "AGND", true),
            (via("s", "SIG", false, 0.0), "SIG2", false),
            (via("s", "SIG", false, 0.0), "SIG", false),
        ] {
            assert_eq!(beside(given, net, ground), (0.0, V2::default()));
        }
        let lone = |given: PairVia, p: PairPad| {
            ViaPairs::new(
                vec![given],
                &[HashMap::new()],
                Vec::new(),
                vec![p],
                &attracting(),
                HOLES,
            )
            .is_empty()
        };
        assert!(lone(
            via("g", "GND", true, 0.0),
            pad("AGND", true, 1.0, DRILL)
        ));
        assert!(!lone(
            via("s", "SIG", false, 0.0),
            pad("GND", true, 1.0, DRILL)
        ));
    }

    #[test]
    fn the_range_runs_from_the_barrel() {
        let at = |barrel: f64| {
            let p = pad("GND", true, barrel + DRILL / 2.0, DRILL);
            with_pads(via("s", "SIG", false, 0.0), vec![p], &attracting(), 0.0)
        };
        assert_eq!(at(DEFAULT_ATTRACTION_RANGE + 1e-3), (0.0, V2::default()));
        let (inside, slope) = at(DEFAULT_ATTRACTION_RANGE - 1e-2);
        assert!(inside < 0.0 && inside > -1e-4, "{inside}");
        assert!(slope.x < 0.0);
        // Beyond the range from the pad's centre, within it from its barrel.
        let centred = pad("GND", true, DEFAULT_ATTRACTION_RANGE + 0.1, DRILL);
        let (value, _) = with_pads(
            via("s", "SIG", false, 0.0),
            vec![centred],
            &attracting(),
            0.0,
        );
        assert!(value < 0.0);
    }

    #[test]
    fn a_wider_drill_pulls_harder() {
        let pulled = |drill: f64| {
            let p = pad("GND", true, 2.0, drill);
            with_pads(via("s", "SIG", false, 0.0), vec![p], &attracting(), 0.0)
        };
        let (narrow, narrow_slope) = pulled(0.6);
        let (wide, wide_slope) = pulled(1.2);
        assert!(wide < narrow);
        assert!(wide_slope.x < narrow_slope.x);
    }

    #[test]
    fn the_attraction_continues_linearly_below_the_hole_rules() {
        // Ring plus hole clearance, 0.3 + 0.25, exceeds drill radius plus
        // hole-to-hole, 0.15 + 0.25.
        let limit = 0.3 + HOLE_CLEARANCE;
        let at = |barrel: f64| {
            let p = pad("GND", true, barrel + DRILL / 2.0, DRILL);
            with_pads(via("s", "SIG", false, 0.0), vec![p], &attracting(), 0.0).1
        };
        assert_eq!(at(limit - 0.2), at(limit));
        assert_ne!(at(limit + 0.2), at(limit));
    }

    #[test]
    fn the_attraction_continues_linearly_below_the_hole_to_hole_limit() {
        // Drill radius plus hole-to-hole, 0.15 + 0.5, exceeds ring plus
        // hole clearance, 0.3 + 0.1.
        let holes = HoleRules {
            hole_clearance: 0.1,
            hole_to_hole: 0.5,
        };
        let limit = 0.15 + holes.hole_to_hole;
        let at = |barrel: f64| {
            let p = pad("GND", true, barrel + DRILL / 2.0, DRILL);
            let pairs = ViaPairs::new(
                vec![via("s", "SIG", false, 0.0)],
                &[HashMap::new()],
                Vec::new(),
                vec![p],
                &attracting(),
                holes,
            );
            pairs.energy(&[V2::default()]).1[0]
        };
        assert_eq!(at(limit - 0.2), at(limit));
        assert_ne!(at(limit + 0.2), at(limit));
    }

    /// The distances `mixed`'s moving signal via keeps from its ground
    /// vias, both ways.
    fn mixed_apart() -> [HashMap<usize, f64>; 3] {
        let kept = 0.8;
        [
            HashMap::from([(2, kept)]),
            HashMap::from([(2, kept)]),
            HashMap::from([(0, kept), (1, kept)]),
        ]
    }

    /// Three moving vias, two static vias and four pads, of every kind and
    /// one out of reach, with the moving vias' positions.
    fn mixed() -> (ViaPairs, [V2; 3]) {
        let vias = vec![
            via("g", "GND", true, 0.0),
            via("h", "GND", true, 0.9),
            via("t", "SIG", false, -0.5),
        ];
        let statics = vec![via("s", "SIG", false, 2.0), via("p", "PWR", true, -1.2)];
        let pads = vec![
            PairPad {
                position: V2::new(0.4, 1.5),
                ..pad("GND", true, 0.0, 1.0)
            },
            PairPad {
                position: V2::new(-1.0, -1.6),
                ..pad("SIG", false, 0.0, 0.8)
            },
            pad("AGND", true, 1.8, 1.1),
            pad("GND", true, 30.0, 1.0),
        ];
        let pairs = ViaPairs::new(vias, &mixed_apart(), statics, pads, &attracting(), HOLES);
        let at = [V2::new(0.1, 0.2), V2::new(0.9, -0.1), V2::new(-0.6, -0.4)];
        (pairs, at)
    }

    #[test]
    fn the_gradient_with_pads_matches_finite_differences() {
        let (pairs, at) = mixed();
        let (value, gradient) = pairs.energy(&at);
        let step = 1e-7;
        for k in 0..at.len() {
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
                    "{k} {axis}: {difference} {want}"
                );
            }
        }
    }

    #[test]
    fn pads_change_the_energy() {
        let (pairs, at) = mixed();
        let bare = ViaPairs::new(
            pairs.vias.clone(),
            &mixed_apart(),
            pairs.statics.clone(),
            Vec::new(),
            &attracting(),
            HOLES,
        );
        assert_ne!(pairs.energy(&at), bare.energy(&at));
    }

    #[test]
    fn without_pads_in_reach_the_energy_keeps_its_bits() {
        let vias = vec![via("g", "GND", true, 0.0), via("h", "GND", true, 0.9)];
        let statics = vec![via("s", "SIG", false, 2.0), via("p", "PWR", true, -1.2)];
        let apart = [HashMap::new(), HashMap::new()];
        let bits = |pads: Vec<PairPad>| {
            let pairs = ViaPairs::new(
                vias.clone(),
                &apart,
                statics.clone(),
                pads,
                &attracting(),
                HOLES,
            );
            let (value, gradient) = pairs.energy(&[V2::new(0.1, 0.2), V2::new(0.9, -0.1)]);
            (
                value.to_bits(),
                gradient
                    .iter()
                    .flat_map(|g| [g.x.to_bits(), g.y.to_bits()])
                    .collect::<Vec<u64>>(),
            )
        };
        let (bare_value, bare_gradient) = bits(Vec::new());
        let (value, gradient) = bits(vec![pad("SIG", false, 30.0, 1.0)]);
        assert_eq!(value, bare_value);
        assert_eq!(gradient, bare_gradient);
    }
}
