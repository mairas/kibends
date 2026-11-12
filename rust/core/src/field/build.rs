//! Building the static fields from the board's items: a chain's field per
//! (layer, net, width), a coupled section's, a moving via's, and what only
//! stops a via.
//!
//! Python reads the board's snapshot of each layer into a `Layer`, its
//! copper, holes, keepouts and board outline as plain shapes; this module
//! picks the items each field holds and inflates them by the clearance
//! rules (`rules`). Each field comes with the source of each of its items
//! (`Picked`), so that Python can name them.
//!
//! A chain's static field is one per (layer, net, width), from other nets'
//! copper, holes, keepouts and the board edge (`chain_obstacles`); no
//! copper of the chain's net is an obstacle, and its holes are not either,
//! except unplated ones. A coupled section of a differential pair keeps
//! clear as one track as wide as the pair (`pair_obstacles`). A moving
//! via's field spans every layer it spans, including rule areas keeping
//! out vias (`via_obstacles`). What only limits a via without repelling it
//! is held apart, as the board edge is: the fabrication rules within its
//! net, hole-to-hole and pad overlap (`own_net_stops`). On a zone's layers
//! the via also keeps the zone's clearance and margin from other-net copper
//! where that exceeds the netclass rule (`rules::zone_inflation`). The
//! vias standing still that a moving via pairs with leave its field and are
//! built apart, each as the field would count it.

use std::collections::{HashMap, HashSet};

use super::TOUCH_TOLERANCE;
use super::chain::ChainField;
use super::obstacles::{Item, Obstacles};
use crate::geometry::{Part, Shape, V2};
use crate::rules::{HoleRules, Keeper, Via, Zone, via_inflation, zones_on};

/// shapely's type id of a Point.
const POINT: i32 = 0;

/// A pad: its copper on the layer, if it has any there, and its hole.
#[derive(Clone, Debug)]
pub struct Pad {
    pub net: String,
    pub position: V2,
    pub drill: f64,
    /// Whether its hole is plated; an NPTH pad's is not.
    pub plated: bool,
    pub shape: Option<Shape>,
    /// Its own clearance, replacing its netclass's.
    pub clearance_override: Option<f64>,
}

/// A track that is not being replaced.
#[derive(Clone, Debug)]
pub struct Track {
    pub net: String,
    pub width: f64,
    pub centreline: Shape,
}

/// Copper graphics, text and shapes, of a net.
#[derive(Clone, Debug)]
pub struct Graphic {
    pub net: String,
    pub shape: Shape,
}

/// The copper of one layer, without the tracks being replaced, as the
/// snapshot reads it: pads whatever their layers, since their holes cross
/// every layer, vias only on the layers they span; no copper zones, which
/// reflow after an edit.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    pub name: String,
    pub ordinal: i32,
    pub pads: Vec<Pad>,
    pub tracks: Vec<Track>,
    pub vias: Vec<Via>,
    pub graphics: Vec<Graphic>,
    /// Rule areas on the layer that keep out tracks.
    pub track_keepouts: Vec<Shape>,
    /// Rule areas on the layer that keep out vias.
    pub via_keepouts: Vec<Shape>,
    /// The board area, None if Edge.Cuts is not closed.
    pub outline: Option<Shape>,
    /// Netclass clearance per net name, mm.
    pub clearances: HashMap<String, f64>,
    pub default_clearance: f64,
}

impl Layer {
    /// The net's netclass clearance; the Default one for no net.
    pub fn clearance(&self, net: &str) -> f64 {
        self.clearances
            .get(net)
            .copied()
            .unwrap_or(self.default_clearance)
    }
}

/// What a field's item is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A pad's copper.
    Pad = 0,
    /// A track's centreline.
    Track = 1,
    /// A via's ring, at its centre.
    ViaRing = 2,
    Graphic = 3,
    TrackKeepout = 4,
    ViaKeepout = 5,
    /// A pad's hole, at its position.
    PadHole = 6,
    /// A via's hole, at its centre.
    ViaHole = 7,
    /// A pad of a via's net, which the via may touch but not overlap.
    PadStop = 8,
}

/// Where a field's item comes from: its kind, the layer whose record it
/// is and its index there, and the layer of its copper, None for what is
/// not copper of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Picked {
    pub source: Source,
    pub layer: usize,
    pub index: usize,
    pub copper: Option<usize>,
}

/// A field's items as they are picked, with their sources.
#[derive(Default)]
pub struct Built {
    pub items: Vec<Item>,
    pub picked: Vec<Picked>,
}

/// A static field and the sources of its items in order.
pub struct BuiltField {
    pub field: Obstacles,
    pub picked: Vec<Picked>,
}

/// A via standing still, by its layer and index there.
#[derive(Clone, Copy, Debug)]
struct ViaRef {
    layer: usize,
    index: usize,
}

fn point(p: V2) -> Shape {
    Shape {
        parts: vec![Part::Line(vec![p])],
        type_id: POINT,
    }
}

impl Built {
    fn push(&mut self, item: Item, picked: Picked) {
        self.items.push(item);
        self.picked.push(picked);
    }

    /// Copper of `net` on layer `l`, reaching `extent` mm beyond `shape`,
    /// with `clearance` its own, as an obstacle to what `keeper` keeps;
    /// nothing for copper of its own net.
    #[allow(clippy::too_many_arguments)]
    fn kept(
        &mut self,
        keeper: &Keeper,
        layers: &[&Layer],
        l: usize,
        shape: &Shape,
        net: &str,
        extent: f64,
        clearance: f64,
        picked: Picked,
    ) {
        if let Some(inflation) = keeper.copper(net, extent, clearance) {
            self.push(
                Item {
                    shape: shape.clone(),
                    inflation,
                    net: Some(net.to_owned()),
                    extent,
                    layer: Some(layers[l].name.clone()),
                },
                picked,
            );
        }
    }

    /// An item that is not copper of one layer.
    fn plain(&mut self, shape: Shape, inflation: f64, picked: Picked) {
        self.push(
            Item {
                shape,
                inflation,
                net: None,
                extent: 0.0,
                layer: None,
            },
            picked,
        );
    }

    /// Layer l's copper and track keepouts, with `vias` in place of the
    /// layer's vias, holes left out; `via_keeper` gives the keeper of each
    /// via's ring. `keep` says which of the layer's items are there.
    fn copper(
        &mut self,
        layers: &[&Layer],
        l: usize,
        keeper: &Keeper,
        vias: &[ViaRef],
        via_keeper: &dyn Fn(&Via) -> Keeper,
        keep: &Keep,
    ) {
        let layer = &layers[l];
        let at = |source, index| Picked {
            source,
            layer: l,
            index,
            copper: Some(l),
        };
        for (k, pad) in layer.pads.iter().enumerate() {
            if let Some(shape) = &pad.shape
                && keep.pads.as_ref().is_none_or(|p| p[k])
            {
                let clearance = pad
                    .clearance_override
                    .unwrap_or_else(|| layer.clearance(&pad.net));
                self.kept(
                    keeper,
                    layers,
                    l,
                    shape,
                    &pad.net,
                    0.0,
                    clearance,
                    at(Source::Pad, k),
                );
            }
        }
        for (k, track) in layer.tracks.iter().enumerate() {
            if keep.tracks.as_ref().is_none_or(|t| t[k]) {
                self.kept(
                    keeper,
                    layers,
                    l,
                    &track.centreline,
                    &track.net,
                    track.width / 2.0,
                    layer.clearance(&track.net),
                    at(Source::Track, k),
                );
            }
        }
        for v in vias {
            let via = &layers[v.layer].vias[v.index];
            self.kept(
                &via_keeper(via),
                layers,
                l,
                &point(via.position),
                &via.net,
                via.diameter / 2.0,
                layer.clearance(&via.net),
                Picked {
                    source: Source::ViaRing,
                    layer: v.layer,
                    index: v.index,
                    copper: Some(l),
                },
            );
        }
        for (k, graphic) in layer.graphics.iter().enumerate() {
            if keep.graphics.as_ref().is_none_or(|g| g[k]) {
                self.kept(
                    keeper,
                    layers,
                    l,
                    &graphic.shape,
                    &graphic.net,
                    0.0,
                    layer.clearance(&graphic.net),
                    at(Source::Graphic, k),
                );
            }
        }
        for (k, keepout) in layer.track_keepouts.iter().enumerate() {
            self.plain(
                keepout.clone(),
                keeper.radius,
                Picked {
                    source: Source::TrackKeepout,
                    layer: l,
                    index: k,
                    copper: None,
                },
            );
        }
    }

    /// The holes of layer l's pads `pads` keeps and of `vias`, as
    /// obstacles to `keeper`.
    fn holes(
        &mut self,
        layers: &[&Layer],
        l: usize,
        pads: &dyn Fn(usize, &Pad) -> bool,
        vias: &[ViaRef],
        keeper: &Keeper,
    ) {
        for (k, pad) in layers[l].pads.iter().enumerate() {
            if pad.drill > 0.0
                && pads(k, pad)
                && let Some(inflation) = keeper.hole(&pad.net, pad.drill / 2.0, pad.plated)
            {
                self.plain(
                    point(pad.position),
                    inflation,
                    Picked {
                        source: Source::PadHole,
                        layer: l,
                        index: k,
                        copper: None,
                    },
                );
            }
        }
        for v in vias {
            let via = &layers[v.layer].vias[v.index];
            if let Some(inflation) = keeper.hole(&via.net, via.drill / 2.0, true) {
                self.plain(
                    point(via.position),
                    inflation,
                    Picked {
                        source: Source::ViaHole,
                        layer: v.layer,
                        index: v.index,
                        copper: None,
                    },
                );
            }
        }
    }

    fn field(self, outline: Option<Shape>, edge_inflation: f64) -> BuiltField {
        BuiltField {
            field: Obstacles::new(self.items, outline, edge_inflation),
            picked: self.picked,
        }
    }
}

/// Which of a layer's pads, tracks and graphics a field holds; None for
/// all.
#[derive(Default)]
struct Keep {
    pads: Option<Vec<bool>>,
    tracks: Option<Vec<bool>>,
    graphics: Option<Vec<bool>>,
}

/// Layer l's items, those `keep` keeps, as obstacles to `keeper`, and
/// the board edge `edge_clearance` beyond its half width.
fn layer_obstacles(
    layers: &[&Layer],
    l: usize,
    keeper: &Keeper,
    edge_clearance: f64,
    keep: &Keep,
    vias: &[ViaRef],
) -> BuiltField {
    let mut built = Built::default();
    built.copper(layers, l, keeper, vias, &|_| keeper.clone(), keep);
    let pads = |k: usize, _: &Pad| keep.pads.as_ref().is_none_or(|p| p[k]);
    built.holes(layers, l, &pads, vias, keeper);
    let layer = &layers[l];
    match &layer.outline {
        None => built.field(None, 0.0),
        Some(outline) => built.field(Some(outline.clone()), edge_clearance + keeper.radius),
    }
}

fn every_via(layers: &[&Layer], l: usize) -> Vec<ViaRef> {
    (0..layers[l].vias.len())
        .map(|index| ViaRef { layer: l, index })
        .collect()
}

/// Obstacles for a track of `net` and `width` mm on the layer.
///
/// Copper clearance is the larger of both items' netclass clearances, with
/// a pad's own override replacing its netclass. Copper of `net` is never
/// an obstacle; its holes are not either, except unplated ones.
pub fn chain_obstacles(
    layer: &Layer,
    net: &str,
    width: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> BuiltField {
    let keeper = Keeper::track(Some(net), width, layer.clearance(net), hole_clearance);
    let layers = &[layer];
    layer_obstacles(
        layers,
        0,
        &keeper,
        edge_clearance,
        &Keep::default(),
        &every_via(layers, 0),
    )
}

/// One track of a coupled section: its net and centreline.
pub struct PairTrack<'a> {
    pub net: &'a str,
    pub points: &'a [V2],
}

/// Obstacles for the centreline of a coupled section of a differential
/// pair, its tracks `a` and `b` of `width` `gap` mm apart, on the layer.
///
/// The centreline keeps clear as one track as wide as the pair would:
/// half its outer extent, a track width and half the gap, plus the larger
/// of the two nets' clearances and the obstacle's own. That is exact beside
/// the pair and errs on the safe side ahead of it.
///
/// Copper of the pair's own nets is an obstacle like any other's, since
/// the other track must keep clear of it and the centreline cannot tell its
/// two sides apart. Copper that overlaps the section's track of its own
/// net is left out: what the section connects to, at its pinned ends or
/// between them, where the centreline cannot keep clear of it. Copper that
/// a fan-out beyond the section connects to stays, as the other track keeps
/// clear of it in its own fan-out's place.
pub fn pair_obstacles(
    layer: &Layer,
    a: PairTrack<'_>,
    b: PairTrack<'_>,
    width: f64,
    gap: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> BuiltField {
    // A pair of one net keeps the second track's line, as a mapping by net
    // would.
    let mut connected: HashMap<&str, Shape> = HashMap::new();
    for track in [&a, &b] {
        connected.insert(track.net, Shape::line(track.points.to_vec()));
    }
    let free = |net: &str, shape: &Shape, extent: f64| {
        connected
            .get(net)
            .is_none_or(|line| shape.shape_distance(line) > extent + width / 2.0 + TOUCH_TOLERANCE)
    };
    let keep = Keep {
        pads: Some(
            layer
                .pads
                .iter()
                .map(|p| p.shape.as_ref().is_none_or(|s| free(&p.net, s, 0.0)))
                .collect(),
        ),
        tracks: Some(
            layer
                .tracks
                .iter()
                .map(|t| free(&t.net, &t.centreline, t.width / 2.0))
                .collect(),
        ),
        graphics: Some(
            layer
                .graphics
                .iter()
                .map(|g| free(&g.net, &g.shape, 0.0))
                .collect(),
        ),
    };
    let vias: Vec<ViaRef> = layer
        .vias
        .iter()
        .enumerate()
        .filter(|(_, v)| free(&v.net, &point(v.position), v.diameter / 2.0))
        .map(|(index, _)| ViaRef { layer: 0, index })
        .collect();
    let clearance = py_max(layer.clearance(a.net), layer.clearance(b.net));
    let keeper = Keeper {
        net: None,
        radius: width + gap / 2.0,
        clearance,
        hole_clearance,
        hole_radius: None,
        hole_to_hole: crate::rules::HOLE_TO_HOLE,
        zones: Vec::new(),
    };
    layer_obstacles(&[layer], 0, &keeper, edge_clearance, &keep, &vias)
}

/// Python's `max(a, b)`: `a` unless `b` is greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Why a via's field cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoLayer(pub String);

impl std::fmt::Display for NoLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no layer that via {} spans was given", self.0)
    }
}

impl std::error::Error for NoLayer {}

/// The layers a moving via spans, its net's clearance and its keeper
/// there, and the vias there that stand still, of other nets and of its
/// own.
struct Surroundings {
    spanned: Vec<usize>,
    clearance: f64,
    keeper: Keeper,
    others: Vec<ViaRef>,
    own: Vec<ViaRef>,
}

impl Surroundings {
    fn of(
        layers: &[&Layer],
        via: &Via,
        moving: &HashSet<&str>,
        holes: HoleRules,
    ) -> Result<Self, NoLayer> {
        let spanned: Vec<usize> = (0..layers.len())
            .filter(|&l| via.reaches(layers[l].ordinal))
            .collect();
        let Some(&first) = spanned.first() else {
            return Err(NoLayer(via.id.clone()));
        };
        let clearance = layers[first].clearance(&via.net);
        let keeper = Keeper::via(
            via,
            clearance,
            holes.hole_clearance,
            holes.hole_to_hole,
            Vec::new(),
        );
        // By id, in the order first met, the last record met, as a Python
        // dict keeps them.
        let mut order: Vec<&str> = Vec::new();
        let mut standing: HashMap<&str, ViaRef> = HashMap::new();
        for &l in &spanned {
            for (index, v) in layers[l].vias.iter().enumerate() {
                if v.id == via.id || moving.contains(v.id.as_str()) {
                    continue;
                }
                if standing.insert(&v.id, ViaRef { layer: l, index }).is_none() {
                    order.push(&v.id);
                }
            }
        }
        let (own, others) = order
            .iter()
            .map(|id| standing[id])
            .partition(|r| layers[r.layer].vias[r.index].net == via.net);
        Ok(Self {
            spanned,
            clearance,
            keeper,
            others,
            own,
        })
    }

    /// The keeper of another via's ring: with `zones` on the layers both
    /// span.
    fn via_keeper(&self, layers: &[&Layer], zones: &[Zone], other: &Via) -> Keeper {
        let on: Vec<i32> = self
            .spanned
            .iter()
            .map(|&l| layers[l].ordinal)
            .filter(|&o| other.reaches(o))
            .collect();
        self.keeper.with_zones(zones_on(zones, &on))
    }
}

/// Whether `pad` has a plated hole of `via`'s net, which only stops the
/// via (`own_net_stops`) and is left out of its field.
fn plated_of(pad: &Pad, via: &Via) -> bool {
    pad.net == via.net && pad.plated
}

/// A moving via's field, and the vias it pairs with apart.
pub struct ViaFields {
    pub field: BuiltField,
    /// The vias of `split`, as obstacles to the via, None if none is one.
    pub split: Option<BuiltField>,
}

/// The fabrication rules and clearances a via keeps.
#[derive(Clone, Copy, Debug)]
pub struct ViaRules {
    pub edge_clearance: f64,
    pub holes: HoleRules,
}

/// Obstacles for `via` at any position, over every layer of `layers` it
/// spans. The outside of the regions of its zones is not among them, nor
/// is anything of its own net: those only limit where the via may go and
/// do not repel it (`own_net_stops`).
///
/// Rule areas keeping out vias on those layers keep its ring out, as track
/// keepouts keep out a track's copper.
///
/// Its ring keeps the copper clearance, as a track of its diameter would,
/// and its hole keeps the hole clearance, to other nets' copper on each of
/// those layers. On a layer of one of `zones`, its centre also keeps the
/// zone's clearance and margin from that copper, so that the refill
/// reaches it. Its hole keeps hole-to-hole from every other hole of
/// another net the layers hold, and an unplated hole of its own net keeps
/// the hole clearance. One field over all the layers' items is their
/// minimum. The via itself and the vias in `moving`, which the caller adds
/// at their current positions, are left out; so is the copper of zones,
/// which KiCad refills.
///
/// The vias of other nets whose ids are in `split` are left out of the
/// field and come back apart, as obstacles to `via`, each as the field
/// would count it.
pub fn via_obstacles(
    layers: &[&Layer],
    via: &Via,
    split: &HashSet<&str>,
    moving: &HashSet<&str>,
    rules: ViaRules,
    zones: &[Zone],
) -> Result<ViaFields, NoLayer> {
    let found = Surroundings::of(layers, via, moving, rules.holes)?;
    let is_split = |r: &ViaRef| split.contains(layers[r.layer].vias[r.index].id.as_str());
    let (picked, kept): (Vec<ViaRef>, Vec<ViaRef>) =
        found.others.iter().copied().partition(|r| is_split(r));
    let field = via_field(layers, &found, via, &kept, rules.edge_clearance, zones);
    if picked.is_empty() {
        return Ok(ViaFields { field, split: None });
    }
    let first = found.spanned[0];
    let mut built = Built::default();
    for r in &picked {
        let other = &layers[r.layer].vias[r.index];
        built.kept(
            &found.via_keeper(layers, zones, other),
            layers,
            first,
            &point(other.position),
            &other.net,
            other.diameter / 2.0,
            layers[first].clearance(&other.net),
            Picked {
                source: Source::ViaRing,
                layer: r.layer,
                index: r.index,
                copper: Some(first),
            },
        );
    }
    built.holes(layers, first, &|_, _| false, &picked, &found.keeper);
    Ok(ViaFields {
        field,
        split: Some(built.field(None, 0.0)),
    })
}

fn via_field(
    layers: &[&Layer],
    found: &Surroundings,
    via: &Via,
    others: &[ViaRef],
    edge_clearance: f64,
    zones: &[Zone],
) -> BuiltField {
    let keeper = &found.keeper;
    let on = |l: usize| keeper.with_zones(zones_on(zones, &[layers[l].ordinal]));
    let first = found.spanned[0];
    let mut built = Built::default();
    // The via spans every layer here, so another via's ring counts once, as
    // do pads' holes, which are the same on every layer. The copper items
    // include track keepouts, which hold the via back as well, since the
    // chain ends it joins sit at its centre.
    let via_keeper = |other: &Via| found.via_keeper(layers, zones, other);
    built.copper(
        layers,
        first,
        &on(first),
        others,
        &via_keeper,
        &Keep::default(),
    );
    for &l in &found.spanned[1..] {
        let keeper = on(l);
        built.copper(
            layers,
            l,
            &keeper,
            &[],
            &|_| keeper.clone(),
            &Keep::default(),
        );
    }
    built.holes(layers, first, &|_, p| !plated_of(p, via), others, keeper);
    for &l in &found.spanned {
        for (k, keepout) in layers[l].via_keepouts.iter().enumerate() {
            built.plain(
                keepout.clone(),
                keeper.radius,
                Picked {
                    source: Source::ViaKeepout,
                    layer: l,
                    index: k,
                    copper: None,
                },
            );
        }
    }
    let outline = layers[first].outline.clone();
    built.field(outline, edge_clearance + keeper.radius)
}

/// What of a via's net limits it: its pads and holes, and its vias.
pub struct OwnNetStops {
    pub items: Option<BuiltField>,
    pub vias: Option<ChainField>,
}

/// What of `via`'s net on the layers of `layers` it spans limits it, as
/// obstacles that stop it without repelling it.
///
/// Copper of one net does not interact, but two drilling rules hold: the
/// holes of its net's vias, but those in `moving`, and of its net's plated
/// pads keep hole-to-hole from its hole, as two moving vias of one net
/// keep (`rules::via_via_inflation`); and its ring may touch its net's pads
/// but not overlap them, since a via in a pad needs the fabricator to fill
/// and cap it, unless `allow_via_in_pad`, which KiCad's DRC does not check
/// either.
pub fn own_net_stops(
    layers: &[&Layer],
    via: &Via,
    moving: &HashSet<&str>,
    holes: HoleRules,
    allow_via_in_pad: bool,
) -> Result<OwnNetStops, NoLayer> {
    let found = Surroundings::of(layers, via, moving, holes)?;
    let first = found.spanned[0];
    let mut built = Built::default();
    built.holes(layers, first, &|_, p| plated_of(p, via), &[], &found.keeper);
    if !allow_via_in_pad {
        for &l in &found.spanned {
            for (k, pad) in layers[l].pads.iter().enumerate() {
                if pad.net == via.net
                    && let Some(shape) = &pad.shape
                {
                    built.plain(
                        shape.clone(),
                        found.keeper.radius,
                        Picked {
                            source: Source::PadStop,
                            layer: l,
                            index: k,
                            copper: None,
                        },
                    );
                }
            }
        }
    }
    let items = (!built.items.is_empty()).then(|| built.field(None, 0.0));
    let own = found
        .own
        .iter()
        .map(|r| (&layers[r.layer].vias[r.index], found.clearance, Vec::new()));
    let vias = vias_field(&found.keeper, own);
    Ok(OwnNetStops { items, vias })
}

/// A via at its position as an obstacle to a chain: its netclass
/// clearance and the zones it stays inside.
#[derive(Clone, Copy, Debug)]
pub struct MovingVia<'a> {
    pub via: &'a Via,
    pub clearance: f64,
    pub zones: &'a [Zone],
}

/// `vias`, at their positions, as obstacles to a track of `net`, `width`
/// and netclass `clearance` (mm) on the layer of `ordinal`, or None if
/// none is one.
///
/// Each counts as `chain_obstacles` counts a via on the board: its ring and
/// hole, kept at the clearances of both nets. A via of `net` is no
/// obstacle, nor is one not spanning the layer. A via of another net inside
/// zones keeps their clearance and margin on the layer from the track, as
/// `rules::via_chain_inflation` keeps it.
pub fn moving_via_obstacles(
    ordinal: i32,
    net: &str,
    width: f64,
    clearance: f64,
    vias: &[MovingVia<'_>],
    hole_clearance: f64,
) -> Option<ChainField> {
    let keeper = Keeper::track(Some(net), width, clearance, hole_clearance);
    let reaching = vias
        .iter()
        .filter(|v| v.via.reaches(ordinal))
        .map(|v| (v.via, v.clearance, zones_on(v.zones, &[ordinal])));
    vias_field(&keeper, reaching)
}

/// The vias' rings and holes as obstacles to `keeper`, or None if none is
/// one; each with its netclass clearance and the zones it stays inside
/// where the keeper is.
fn vias_field<'a>(
    keeper: &Keeper,
    vias: impl IntoIterator<Item = (&'a Via, f64, Vec<Zone>)>,
) -> Option<ChainField> {
    let mut centres = Vec::new();
    let mut inflations = Vec::new();
    for (v, clearance, zones) in vias {
        if let Some(inflation) = via_inflation(keeper, v, clearance, &zones) {
            centres.push(v.position);
            inflations.push(inflation);
        }
    }
    if centres.is_empty() {
        return None;
    }
    // Zero-length edges: a chain field measures to a point exactly as to an
    // edge. Each via is an obstacle of its own.
    let obstacles = (0..centres.len()).collect();
    Some(ChainField::new(&centres, &centres, inflations, obstacles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::rules::{HOLE_CLEARANCE, HOLE_TO_HOLE, Span};

    const HOLES: HoleRules = HoleRules {
        hole_clearance: HOLE_CLEARANCE,
        hole_to_hole: HOLE_TO_HOLE,
    };

    fn via(id: &str, net: &str, at: (f64, f64)) -> Via {
        Via {
            id: id.into(),
            net: net.into(),
            position: V2::new(at.0, at.1),
            diameter: 0.6,
            drill: 0.3,
            span: Span { low: 0, high: 2 },
        }
    }

    fn pad(net: &str, at: (f64, f64), plated: bool) -> Pad {
        let (x, y) = at;
        Pad {
            net: net.into(),
            position: V2::new(x, y),
            drill: 0.8,
            plated,
            shape: Some(Shape::polygon(vec![vec![
                V2::new(x - 1.0, y - 1.0),
                V2::new(x + 1.0, y - 1.0),
                V2::new(x + 1.0, y + 1.0),
                V2::new(x - 1.0, y + 1.0),
                V2::new(x - 1.0, y - 1.0),
            ]])),
            clearance_override: None,
        }
    }

    fn layer(name: &str, ordinal: i32) -> Layer {
        Layer {
            name: name.into(),
            ordinal,
            default_clearance: 0.2,
            ..Layer::default()
        }
    }

    #[test]
    fn same_net_copper_is_no_obstacle_but_an_unplated_hole_is() {
        let mut top = layer("F.Cu", 0);
        top.pads = vec![pad("SIG", (0.0, 0.0), true), pad("SIG", (5.0, 0.0), false)];
        top.tracks = vec![Track {
            net: "SIG".into(),
            width: 0.2,
            centreline: Shape::line(vec![V2::new(0.0, 3.0), V2::new(5.0, 3.0)]),
        }];
        let built = chain_obstacles(&top, "SIG", 0.2, 0.3, HOLE_CLEARANCE);
        // The unplated pad's copper is of the net too; only its hole stays.
        assert_eq!(
            built.picked.iter().map(|p| p.source).collect::<Vec<_>>(),
            vec![Source::PadHole]
        );
        assert_eq!(built.picked[0].index, 1);
        let other = chain_obstacles(&top, "GND", 0.2, 0.3, HOLE_CLEARANCE);
        assert_eq!(other.picked.len(), 5);
    }

    #[test]
    fn a_moving_via_s_split_vias_leave_its_field() {
        let mut top = layer("F.Cu", 0);
        top.vias = vec![
            via("me", "SIG", (0.0, 0.0)),
            via("ground", "GND", (2.0, 0.0)),
            via("other", "X", (4.0, 0.0)),
            via("own", "SIG", (6.0, 0.0)),
        ];
        let rules = ViaRules {
            edge_clearance: 0.3,
            holes: HOLES,
        };
        let me = top.vias[0].clone();
        let split = HashSet::from(["ground"]);
        let fields = via_obstacles(&[&top], &me, &split, &HashSet::new(), rules, &[]).unwrap();
        let rings: Vec<usize> = fields
            .field
            .picked
            .iter()
            .filter(|p| p.source == Source::ViaRing)
            .map(|p| p.index)
            .collect();
        assert_eq!(rings, vec![2]);
        let apart = fields.split.unwrap();
        assert_eq!(
            apart.picked.iter().map(|p| p.source).collect::<Vec<_>>(),
            vec![Source::ViaRing, Source::ViaHole]
        );
        let stops = own_net_stops(&[&top], &me, &HashSet::new(), HOLES, false).unwrap();
        assert!(stops.items.is_none());
        let vias = stops.vias.unwrap();
        // Hole to hole from the via of its net: 0.15 + 0.15 + 0.25.
        let (d, _) = vias.signed_distance(&[V2::new(6.0, 0.0)], f64::INFINITY);
        assert!((d[0] + 0.55).abs() < 1e-12);
    }

    #[test]
    fn a_via_of_no_spanned_layer_is_an_error() {
        let me = Via {
            span: Span { low: 3, high: 4 },
            ..via("me", "SIG", (0.0, 0.0))
        };
        let result = own_net_stops(&[&layer("F.Cu", 0)], &me, &HashSet::new(), HOLES, false);
        assert_eq!(result.err(), Some(NoLayer("me".into())));
    }

    #[test]
    fn moving_vias_of_the_track_s_net_or_another_layer_are_no_obstacles() {
        let a = via("a", "SIG", (0.0, 0.0));
        let b = via("b", "GND", (1.0, 0.0));
        let c = Via {
            span: Span { low: 3, high: 4 },
            ..via("c", "GND", (2.0, 0.0))
        };
        let moving = |via| MovingVia {
            via,
            clearance: 0.2,
            zones: &[],
        };
        let none = moving_via_obstacles(0, "SIG", 0.2, 0.2, &[moving(&a)], 0.25);
        assert!(none.is_none());
        let all = [moving(&a), moving(&b), moving(&c)];
        let field = moving_via_obstacles(0, "SIG", 0.2, 0.2, &all, 0.25).unwrap();
        assert_eq!(field.len(), 1);
    }
}
