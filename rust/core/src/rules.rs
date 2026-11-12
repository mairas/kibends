//! The clearance rules: how far what keeps clear, a track's centreline or a
//! via's centre, keeps from the board's items, as the inflation of each
//! item's raw geometry.
//!
//! A `Keeper` is what keeps clear of the board's items: the copper disc
//! around each point of a track's centreline, or a via's ring and hole at
//! its centre. Copper keeps the larger of both items' netclass clearances
//! beyond both reaches; a via's hole also keeps the hole clearance from
//! copper, and hole-to-hole from other holes. Copper of one net does not
//! interact (decision `kibends-same-net-no-interaction`): a same-net
//! plated hole is part of the connection, so only hole-to-hole applies to
//! it.
//!
//! On a zone's layers a via also keeps the zone's clearance and margin
//! from other-net copper where that exceeds the netclass rule
//! (`zone_inflation`), so that the next refill reconnects it wherever the
//! other copper ends up. Two vias keep their distance, and a via's zones
//! count against another, only where their spans overlap
//! (`via_via_inflation`).
//!
//! The inflations become the fields' and so the optimizer's numbers, and
//! their order of operations keeps the optimizer's results on the recorded
//! scipy baseline (`tests/test_scipy_baseline.py`) and the same on every
//! platform. Changing it moves moving-via cases into other local minima,
//! so re-run the baseline test after changing it.

use crate::geometry::V2;

/// The span of a via's drill, as copper layer ordinals, low to high.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub low: i32,
    pub high: i32,
}

impl Span {
    /// Whether the layer of `ordinal` lies within the span.
    pub fn reaches(self, ordinal: i32) -> bool {
        self.low <= ordinal && ordinal <= self.high
    }

    /// Whether the two spans overlap.
    pub fn shares(self, other: Span) -> bool {
        self.low <= other.high && other.low <= self.high
    }
}

/// A via, with copper and a hole on the layers of its span.
#[derive(Clone, Debug, PartialEq)]
pub struct Via {
    pub id: String,
    pub net: String,
    pub position: V2,
    pub diameter: f64,
    pub drill: f64,
    pub span: Span,
}

impl Via {
    pub fn reaches(&self, ordinal: i32) -> bool {
        self.span.reaches(ordinal)
    }

    pub fn shares_layer(&self, other: &Via) -> bool {
        self.span.shares(other.span)
    }
}

/// A copper zone of a via's net that may connect to the via, as the rules
/// take it: its own clearance to other nets' copper, how far inside its
/// region the via's centre stays (`ViaZone.margin`), and the layers on
/// which it reaches the via's ring, as ordinals.
#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    pub clearance: f64,
    pub margin: f64,
    pub layers: Vec<i32>,
}

/// The zones among `zones` a via stays inside on any of `layers`.
pub fn zones_on(zones: &[Zone], layers: &[i32]) -> Vec<Zone> {
    zones
        .iter()
        .filter(|z| z.layers.iter().any(|l| layers.contains(l)))
        .cloned()
        .collect()
}

/// The zones among `zones` on a layer `via` spans. A zone's layers hold
/// only layers its own via spans, so these are the zones on the layers
/// both vias span, as `zones_on` gives them for those layers.
pub fn zones_reaching(zones: &[Zone], via: &Via) -> Vec<Zone> {
    zones
        .iter()
        .filter(|z| z.layers.iter().any(|&l| via.reaches(l)))
        .cloned()
        .collect()
}

/// The board's default copper-to-hole clearance, mm: the hole clearance
/// KiCad's own project templates set.
pub const HOLE_CLEARANCE: f64 = 0.25;
/// The board's default hole-to-hole minimum, mm, as KiCad's templates set
/// it.
pub const HOLE_TO_HOLE: f64 = 0.25;

/// What keeps clear of the board's items: the copper disc around each
/// point of a track's centreline, or a via's ring and hole at its centre.
///
/// Each method gives the inflation of another item's raw geometry, or None
/// where no rule applies between the two.
#[derive(Clone, Debug, PartialEq)]
pub struct Keeper {
    /// None for a pair's centreline, which keeps clear of every net.
    pub net: Option<String>,
    /// Half the track width, or the via's ring radius, mm.
    pub radius: f64,
    /// Its net's netclass clearance, mm.
    pub clearance: f64,
    pub hole_clearance: f64,
    /// A via's drill radius; None for a track.
    pub hole_radius: Option<f64>,
    pub hole_to_hole: f64,
    /// For a via, the zones it stays inside on the layers at hand.
    pub zones: Vec<Zone>,
}

impl Keeper {
    /// A track of `width` and netclass `clearance` of `net`, None for a
    /// pair's centreline.
    pub fn track(net: Option<&str>, width: f64, clearance: f64, hole_clearance: f64) -> Self {
        Self {
            net: net.map(str::to_owned),
            radius: width / 2.0,
            clearance,
            hole_clearance,
            hole_radius: None,
            hole_to_hole: HOLE_TO_HOLE,
            zones: Vec::new(),
        }
    }

    /// `via`, of netclass `clearance`, inside `zones` on the layers at hand.
    pub fn via(
        via: &Via,
        clearance: f64,
        hole_clearance: f64,
        hole_to_hole: f64,
        zones: Vec<Zone>,
    ) -> Self {
        Self {
            net: Some(via.net.clone()),
            radius: via.diameter / 2.0,
            clearance,
            hole_clearance,
            hole_radius: Some(via.drill / 2.0),
            hole_to_hole,
            zones,
        }
    }

    /// The keeper inside `zones` instead.
    pub fn with_zones(&self, zones: Vec<Zone>) -> Self {
        Self {
            zones,
            ..self.clone()
        }
    }

    /// For copper of `net` reaching `extent` mm beyond its geometry, with
    /// `clearance` its own clearance; None for copper of its own net.
    pub fn copper(&self, net: &str, extent: f64, clearance: f64) -> Option<f64> {
        if self.net.as_deref() == Some(net) {
            return None;
        }
        let inflation = py_max(
            extent + py_max(self.clearance, clearance) + self.radius,
            zone_inflation(&self.zones, extent, &[self.clearance, clearance]),
        );
        Some(match self.hole_radius {
            None => inflation,
            Some(hole) => py_max(inflation, extent + hole + self.hole_clearance),
        })
    }

    /// For a hole of `radius` mm of `net`. A same-net plated hole is part
    /// of the connection, so only hole-to-hole applies to it.
    pub fn hole(&self, net: &str, radius: f64, plated: bool) -> Option<f64> {
        let mut inflation: Option<f64> = None;
        if self.net.as_deref() != Some(net) || !plated {
            inflation = Some(radius + self.hole_clearance + self.radius);
        }
        if let Some(hole) = self.hole_radius {
            let apart = radius + hole + self.hole_to_hole;
            inflation = Some(inflation.map_or(apart, |i| py_max(i, apart)));
        }
        inflation
    }
}

/// Python's `max(a, b)`: `a` unless `b` is greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// How far a via's centre keeps from other-net copper reaching `extent` mm
/// beyond its geometry for the next refill of `zones` to connect it: the
/// refill keeps the larger of the zone's clearance and `clearances` from
/// that copper, and the fill reaches the via's centre by the zone's
/// margin. -inf without zones.
pub fn zone_inflation(zones: &[Zone], extent: f64, clearances: &[f64]) -> f64 {
    zones
        .iter()
        .map(|z| {
            let widest = clearances
                .iter()
                .copied()
                .chain([z.clearance])
                .reduce(py_max)
                .unwrap_or(z.clearance);
            extent + widest + z.margin
        })
        .fold(f64::NEG_INFINITY, py_max)
}

/// The distance a track of `width` and `clearance` keeps its centreline
/// from that of a track of another net of `other_width` and
/// `other_clearance`, mm: both half widths and the larger of both
/// clearances.
pub fn track_inflation(width: f64, clearance: f64, other_width: f64, other_clearance: f64) -> f64 {
    // A keeper of no net keeps clear of every net; tracks have no holes, so
    // its hole clearance is never asked.
    Keeper::track(None, width, clearance, HOLE_CLEARANCE)
        .copper("", other_width / 2.0, other_clearance)
        .unwrap_or(f64::NAN)
}

/// How far `keeper` keeps from `via`'s centre, by its ring and hole, and
/// by `zones`, the zones `via` stays inside where the keeper is, as the
/// via keeps from it (`Keeper::copper`). `clearance` is the via's net's
/// netclass clearance.
pub fn via_inflation(keeper: &Keeper, via: &Via, clearance: f64, zones: &[Zone]) -> Option<f64> {
    let ring = keeper
        .copper(&via.net, via.diameter / 2.0, clearance)
        .map(|ring| {
            py_max(
                ring,
                zone_inflation(zones, keeper.radius, &[keeper.clearance, clearance]),
            )
        });
    let hole = keeper.hole(&via.net, via.drill / 2.0, true);
    match (ring, hole) {
        (Some(r), Some(h)) => Some(py_max(r, h)),
        (r, h) => r.or(h),
    }
}

/// The holes' clearances a via keeps: copper-to-hole and hole-to-hole, mm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoleRules {
    pub hole_clearance: f64,
    pub hole_to_hole: f64,
}

/// The distance `via`'s centre keeps from `other`'s, mm, as a moving via's
/// field counts a via on the board; None if no rule applies, as for two
/// vias whose spans do not overlap. `clearance` and `other_clearance` are
/// their nets' netclass clearances.
///
/// Rings and holes of another net keep the copper and hole clearances,
/// holes of any net keep hole-to-hole, and rings of another net the
/// clearance and margin of `zones`, the zones `via` stays inside, and of
/// `other_zones`, those `other` stays inside, each on the layers both vias
/// span: given both, the distance is the same both ways.
#[allow(clippy::too_many_arguments)]
pub fn via_via_inflation(
    via: &Via,
    clearance: f64,
    other: &Via,
    other_clearance: f64,
    holes: HoleRules,
    zones: &[Zone],
    other_zones: &[Zone],
) -> Option<f64> {
    if !via.shares_layer(other) {
        return None;
    }
    let keeper = Keeper::via(
        via,
        clearance,
        holes.hole_clearance,
        holes.hole_to_hole,
        zones_reaching(zones, other),
    );
    via_inflation(
        &keeper,
        other,
        other_clearance,
        &zones_reaching(other_zones, via),
    )
}

/// A chain as the rules take it: its net, width and netclass clearance.
#[derive(Clone, Copy, Debug)]
pub struct Track<'a> {
    pub net: &'a str,
    pub width: f64,
    pub clearance: f64,
}

/// The distance `via`'s centre keeps from `chain`'s centreline, mm: the
/// larger of the copper clearance from the via's ring and the hole
/// clearance from its hole, and the clearance and margin of `zones`, the
/// zones the via stays inside on the chain's layer; None for a chain of
/// its net, which is no obstacle to it. `clearance` is the via's net's.
pub fn via_chain_inflation(
    via: &Via,
    clearance: f64,
    chain: Track<'_>,
    hole_clearance: f64,
    zones: &[Zone],
) -> Option<f64> {
    if chain.net == via.net {
        return None;
    }
    // Chains have no holes, so the keeper's hole-to-hole is never asked.
    let keeper = Keeper::via(via, clearance, hole_clearance, HOLE_TO_HOLE, zones.to_vec());
    keeper.copper(chain.net, chain.width / 2.0, chain.clearance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn via(net: &str, low: i32, high: i32) -> Via {
        Via {
            id: format!("{net}{low}{high}"),
            net: net.into(),
            position: V2::new(0.0, 0.0),
            diameter: 0.6,
            drill: 0.3,
            span: Span { low, high },
        }
    }

    const HOLES: HoleRules = HoleRules {
        hole_clearance: HOLE_CLEARANCE,
        hole_to_hole: HOLE_TO_HOLE,
    };

    #[test]
    fn same_net_copper_does_not_interact() {
        let keeper = Keeper::track(Some("GND"), 0.25, 0.2, HOLE_CLEARANCE);
        assert_eq!(keeper.copper("GND", 0.1, 0.2), None);
        assert_eq!(keeper.copper("SIG", 0.1, 0.2), Some(0.1 + 0.2 + 0.125));
    }

    #[test]
    fn a_same_net_plated_hole_keeps_only_hole_to_hole() {
        let keeper = Keeper::via(&via("GND", 0, 2), 0.2, HOLE_CLEARANCE, HOLE_TO_HOLE, vec![]);
        assert_eq!(
            keeper.hole("GND", 0.15, true),
            Some(0.15 + 0.15 + HOLE_TO_HOLE)
        );
        let unplated = keeper.hole("GND", 0.15, false).unwrap();
        assert_eq!(
            unplated,
            (0.15 + HOLE_CLEARANCE + 0.3).max(0.15 + 0.15 + HOLE_TO_HOLE)
        );
        let track = Keeper::track(Some("GND"), 0.25, 0.2, HOLE_CLEARANCE);
        assert_eq!(track.hole("GND", 0.15, true), None);
    }

    #[test]
    fn zones_keep_their_clearance_and_margin_and_none_is_minus_infinity() {
        assert_eq!(zone_inflation(&[], 0.3, &[0.2]), f64::NEG_INFINITY);
        let zone = Zone {
            clearance: 0.5,
            margin: 0.1,
            layers: vec![0],
        };
        assert_eq!(zone_inflation(&[zone], 0.3, &[0.2, 0.1]), 0.3 + 0.5 + 0.1);
    }

    #[test]
    fn a_via_with_zones_keeps_the_zone_s_clearance_plus_margin() {
        let zone = Zone {
            clearance: 0.5,
            margin: 0.1,
            layers: vec![0],
        };
        let v = via("GND", 0, 2);
        let track = Track {
            net: "SIG",
            width: 0.2,
            clearance: 0.2,
        };
        let plain = via_chain_inflation(&v, 0.2, track, HOLE_CLEARANCE, &[]).unwrap();
        let zoned = via_chain_inflation(&v, 0.2, track, HOLE_CLEARANCE, &[zone]).unwrap();
        assert_eq!(plain, 0.1 + 0.2 + 0.3);
        assert_eq!(zoned, 0.1 + 0.5 + 0.1);
    }

    #[test]
    fn vias_apart_only_where_their_spans_overlap_and_alike_both_ways() {
        let a = via("A", 0, 1);
        let b = via("B", 2, 3);
        assert_eq!(via_via_inflation(&a, 0.2, &b, 0.2, HOLES, &[], &[]), None);
        let zone = Zone {
            clearance: 0.6,
            margin: 0.1,
            layers: vec![1],
        };
        let c = via("C", 1, 2);
        let ab = via_via_inflation(&a, 0.2, &c, 0.3, HOLES, std::slice::from_ref(&zone), &[]);
        let ba = via_via_inflation(&c, 0.3, &a, 0.2, HOLES, &[], &[zone]);
        assert_eq!(ab, ba);
        assert_eq!(ab, Some(0.3 + 0.6 + 0.1));
    }

    #[test]
    fn tracks_keep_both_half_widths_and_the_larger_clearance() {
        assert_eq!(track_inflation(0.2, 0.1, 0.4, 0.3), 0.2 + 0.3 + 0.1);
    }
}
