//! Moving vias in the optimizer: the chain ends they carry, the fields they
//! make and the distances they keep.
//!
//! The optimizer makes each moving via's centre a variable of its problem
//! and carries the chain ends it joins at their offsets from it: an end
//! joined off-centre stays where it overlaps the via's copper.
//!
//! A via and a chain are near when the chain is on a layer the via spans
//! and is of another net: each is then in the other's field. Copper of one
//! net does not interact, whether the chain ends at the via or not. On a
//! zone's layers a moving via keeps the zone's clearance from the chains
//! and vias moving with it.

use std::collections::HashMap;

use crate::field::build::{MovingVia, moving_via_obstacles};
use crate::field::chain::ChainField;
use crate::geometry::V2;
use crate::rules::{self, HoleRules, Via, Zone, via_via_inflation, zones_on};

/// A selected chain as the moving vias see it.
#[derive(Clone, Debug)]
pub struct Chain {
    pub net: String,
    /// The ordinal of its layer.
    pub layer: i32,
    pub width: f64,
    /// Its net's netclass clearance, mm.
    pub clearance: f64,
    pub first: V2,
    pub last: V2,
}

/// A moving via: its netclass clearance, the zones it stays inside, and
/// the chain ends it carries, each a chain's index and whether it is the
/// chain's last point.
#[derive(Clone, Debug)]
pub struct Moving {
    pub via: Via,
    pub clearance: f64,
    pub zones: Vec<Zone>,
    pub ends: Vec<(usize, bool)>,
}

/// A chain end that moves with a via, and its offset from the via's
/// centre, kept as the via moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct End {
    pub chain: usize,
    pub last: bool,
    pub offset: V2,
}

/// The moving vias of a run, and the chains they move among.
pub struct MovingVias {
    chains: Vec<Chain>,
    vias: Vec<Moving>,
    ends: Vec<Vec<End>>,
    clearances: HashMap<String, f64>,
    chains_near: Vec<Vec<usize>>,
    vias_near: Vec<Vec<usize>>,
    holes: HoleRules,
}

impl MovingVias {
    pub fn new(chains: Vec<Chain>, vias: Vec<Moving>, holes: HoleRules) -> Self {
        let ends = vias
            .iter()
            .map(|v| {
                v.ends
                    .iter()
                    .map(|&(chain, last)| {
                        let c = &chains[chain];
                        let at = if last { c.last } else { c.first };
                        End {
                            chain,
                            last,
                            offset: at - v.via.position,
                        }
                    })
                    .collect()
            })
            .collect();
        let mut clearances: HashMap<String, f64> = HashMap::new();
        for c in &chains {
            clearances.insert(c.net.clone(), c.clearance);
        }
        for v in &vias {
            clearances.insert(v.via.net.clone(), v.clearance);
        }
        let chains_near: Vec<Vec<usize>> = vias
            .iter()
            .map(|v| {
                (0..chains.len())
                    .filter(|&i| chains[i].net != v.via.net && v.via.reaches(chains[i].layer))
                    .collect()
            })
            .collect();
        let vias_near = (0..chains.len())
            .map(|i| {
                (0..vias.len())
                    .filter(|&v| chains_near[v].contains(&i))
                    .collect()
            })
            .collect();
        Self {
            chains,
            vias,
            ends,
            clearances,
            chains_near,
            vias_near,
            holes,
        }
    }

    pub fn via_count(&self) -> usize {
        self.vias.len()
    }

    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// The chain ends via v carries.
    pub fn ends(&self, v: usize) -> &[End] {
        &self.ends[v]
    }

    fn clearance_of(&self, net: &str) -> f64 {
        self.clearances.get(net).copied().unwrap_or(f64::NAN)
    }

    /// The vias near chain i, at `positions`, as obstacles to it; None
    /// when none is one.
    pub fn via_obstacles_for_chain(&self, i: usize, positions: &[V2]) -> Option<ChainField> {
        let chain = &self.chains[i];
        let placed: Vec<Via> = self.vias_near[i]
            .iter()
            .map(|&v| Via {
                position: positions[v],
                ..self.vias[v].via.clone()
            })
            .collect();
        let moving: Vec<MovingVia<'_>> = self.vias_near[i]
            .iter()
            .zip(&placed)
            .map(|(&v, via)| MovingVia {
                via,
                clearance: self.clearance_of(&via.net),
                zones: &self.vias[v].zones,
            })
            .collect();
        moving_via_obstacles(
            chain.layer,
            &chain.net,
            chain.width,
            chain.clearance,
            &moving,
            self.holes.hole_clearance,
        )
    }

    /// The distance via v's centre keeps from via u's; None if no rule
    /// applies. It is the same as u's from v's, the larger of both vias'
    /// zone distances.
    pub fn apart(&self, v: usize, u: usize) -> Option<f64> {
        let (a, b) = (&self.vias[v], &self.vias[u]);
        via_via_inflation(
            &a.via,
            self.clearance_of(&a.via.net),
            &b.via,
            self.clearance_of(&b.via.net),
            self.holes,
            &a.zones,
            &b.zones,
        )
    }

    /// The chains near via v, in what it sees and seeing it, and how far
    /// each keeps its centreline from v's centre, by v's zones on that
    /// chain's layer.
    pub fn rings(&self, v: usize) -> Vec<(usize, f64)> {
        let moving = &self.vias[v];
        self.chains_near[v]
            .iter()
            .filter_map(|&i| {
                let chain = &self.chains[i];
                rules::via_chain_inflation(
                    &moving.via,
                    self.clearance_of(&moving.via.net),
                    rules::Track {
                        net: &chain.net,
                        width: chain.width,
                        clearance: chain.clearance,
                    },
                    self.holes.hole_clearance,
                    &zones_on(&moving.zones, &[chain.layer]),
                )
                .map(|ring| (i, ring))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::rules::{HOLE_CLEARANCE, HOLE_TO_HOLE, Span};

    fn chain(net: &str, layer: i32) -> Chain {
        Chain {
            net: net.into(),
            layer,
            width: 0.2,
            clearance: 0.2,
            first: V2::new(0.0, 0.0),
            last: V2::new(5.0, 0.1),
        }
    }

    fn moving(net: &str, at: V2, ends: Vec<(usize, bool)>) -> Moving {
        Moving {
            via: Via {
                id: net.into(),
                net: net.into(),
                position: at,
                diameter: 0.6,
                drill: 0.3,
                span: Span { low: 0, high: 2 },
            },
            clearance: 0.2,
            zones: Vec::new(),
            ends,
        }
    }

    fn vias() -> MovingVias {
        MovingVias::new(
            vec![chain("SIG", 0), chain("GND", 0), chain("SIG", 31)],
            vec![
                moving("SIG", V2::new(5.0, 0.0), vec![(0, true)]),
                moving("GND", V2::new(2.0, 2.0), vec![]),
            ],
            HoleRules {
                hole_clearance: HOLE_CLEARANCE,
                hole_to_hole: HOLE_TO_HOLE,
            },
        )
    }

    #[test]
    fn ends_keep_their_offsets() {
        let vias = vias();
        let end = vias.ends(0)[0];
        assert_eq!((end.chain, end.last), (0, true));
        assert_eq!(end.offset, V2::new(5.0, 0.1) - V2::new(5.0, 0.0));
    }

    #[test]
    fn a_via_is_near_chains_of_other_nets_on_its_layers() {
        let vias = vias();
        assert_eq!(
            vias.rings(0).iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(vias.rings(1)[0], (0, 0.1 + 0.2 + 0.3));
        let at = [V2::new(5.0, 0.0), V2::new(1.0, 1.0)];
        let field = vias.via_obstacles_for_chain(0, &at).unwrap();
        let (d, _) = field.signed_distance(&[V2::new(1.0, 1.0)], f64::INFINITY);
        assert!((d[0] + 0.6).abs() < 1e-12);
        assert!(vias.via_obstacles_for_chain(2, &at).is_none());
        assert_eq!(vias.apart(0, 1), Some(0.3 + 0.2 + 0.3));
    }
}
