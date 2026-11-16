//! Building the static fields from the board's items: a chain's field per
//! (layer, net, width), a coupled section's, a moving via's, and what only
//! stops a via.
//!
//! Python reads the board's snapshot of each layer into a `Layer`, its
//! copper, holes, keepouts and board outline as plain shapes. A run's
//! fields share one `BoardItems`: each layer's items, less the moving
//! vias, in one item set (`items::ItemSet`) whose items keep their records
//! (`items::Record`), so that Python can name them. A builder picks the
//! items each field holds by walking the layers' records and inflates them
//! by the clearance rules (`rules`); the field is a view of the sets that
//! holds which items it has, in its order, and their inflations
//! (`edges::StaticEdges`). A layer's set orders its items as a chain's
//! field takes them: pads' copper, tracks, vias' rings, graphics, track
//! keepouts, pads' holes, vias' holes, via keepouts, the board outline.
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
//!
//! A moving via's field is one view over the sets of every layer it
//! spans, in the order its builder takes them: the first layer's copper,
//! with the other vias' rings, then each other layer's copper, the holes,
//! the via keepouts and the outline. A ring of another via lies in the set
//! of the layer where it was last met, and counts as copper of the
//! field's first layer (`edges::Override::Layer`); a pad that only stops
//! a via is its copper held as no copper (`edges::Override::Stop`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::chain::ChainField;
use super::edges::{Member, Override, StaticEdges, number};
use super::items::Source;
use super::items::{Copper, Interner, ItemIds, ItemSet, Record};
use super::obstacles::{Obstacles, edge_obstacle, obstacle_ids};
use super::{Kind, TOUCH_TOLERANCE};
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

fn point(p: V2) -> Shape {
    Shape {
        parts: vec![Part::Line(vec![p])],
        type_id: POINT,
    }
}

/// How a field measures to `shape`: to its area, or to its geometry.
fn kind_of(shape: &Shape) -> Kind {
    if shape.is_area() {
        Kind::Area
    } else {
        Kind::Line
    }
}

/// Each of a layer's records' items in its set, None for a record that
/// has none there: a pad without copper on the layer or without a drill,
/// a moving via.
#[derive(Default)]
struct Records {
    pad_copper: Vec<Option<u32>>,
    pad_holes: Vec<Option<u32>>,
    tracks: Vec<u32>,
    via_rings: Vec<Option<u32>>,
    via_holes: Vec<Option<u32>>,
    graphics: Vec<u32>,
    track_keepouts: Vec<u32>,
    via_keepouts: Vec<u32>,
    outline: Option<u32>,
}

/// A layer's set as it is gathered.
struct Gathered {
    layer: u32,
    shapes: Vec<Shape>,
    kinds: Vec<Kind>,
    copper: Copper,
    records: Vec<Record>,
}

impl Gathered {
    /// Adds `shape` of `kind`, from record `index` of `source`, as copper
    /// of the net numbered `net` reaching `extent` mm beyond it, or None
    /// for what is not copper; and gives its index.
    fn add(
        &mut self,
        shape: Shape,
        kind: Kind,
        copper: Option<(u32, f64)>,
        source: Source,
        index: usize,
    ) -> u32 {
        let item = number(self.shapes.len());
        self.shapes.push(shape);
        self.kinds.push(kind);
        self.copper.net.push(copper.map(|(net, _)| net));
        self.copper.layer.push(copper.map(|_| self.layer));
        self.copper
            .extent
            .push(copper.map_or(0.0, |(_, extent)| extent));
        self.records.push(Record {
            source,
            layer: self.layer,
            index: number(index),
        });
        item
    }
}

/// The board's layers' items as a run's fields share them: one set per
/// layer, without the vias of `moving`, which the optimizer adds at their
/// current positions. A layer's copper is on the layer of its index.
pub struct BoardItems {
    layers: Vec<Arc<Layer>>,
    sets: Vec<Arc<ItemSet>>,
    records: Vec<Records>,
    moving: HashSet<String>,
}

impl BoardItems {
    pub fn new(layers: Vec<Arc<Layer>>, moving: HashSet<String>) -> Self {
        let mut interner = Interner::new();
        let mut sets = Vec::with_capacity(layers.len());
        let mut records = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let (set, found) = gather(layer, number(l), &moving, &mut interner);
            sets.push(Arc::new(set));
            records.push(found);
        }
        Self {
            layers,
            sets,
            records,
            moving,
        }
    }

    pub fn layers(&self) -> &[Arc<Layer>] {
        &self.layers
    }

    fn layer(&self, l: usize) -> &Layer {
        &self.layers[l]
    }

    fn is_moving(&self, via: &Via) -> bool {
        self.moving.contains(&via.id)
    }

    /// Layer l's vias but the moving ones.
    fn still_vias(&self, l: usize) -> Vec<ViaRef> {
        self.layer(l)
            .vias
            .iter()
            .enumerate()
            .filter(|(_, v)| !self.is_moving(v))
            .map(|(index, _)| ViaRef { layer: l, index })
            .collect()
    }
}

/// Layer `layer`'s items, in the order a chain's field takes them, and
/// each record's.
fn gather(
    from: &Layer,
    layer: u32,
    moving: &HashSet<String>,
    interner: &mut Interner,
) -> (ItemSet, Records) {
    let mut set = Gathered {
        layer,
        shapes: Vec::new(),
        kinds: Vec::new(),
        copper: Copper::default(),
        records: Vec::new(),
    };
    let mut found = Records::default();
    let mut net = |name: &str, extent: f64| Some((interner.name(name), extent));
    for (k, pad) in from.pads.iter().enumerate() {
        let item = pad.shape.as_ref().map(|shape| {
            set.add(
                shape.clone(),
                kind_of(shape),
                net(&pad.net, 0.0),
                Source::Pad,
                k,
            )
        });
        found.pad_copper.push(item);
    }
    for (k, track) in from.tracks.iter().enumerate() {
        let shape = &track.centreline;
        let copper = net(&track.net, track.width / 2.0);
        let item = set.add(shape.clone(), kind_of(shape), copper, Source::Track, k);
        found.tracks.push(item);
    }
    for (k, via) in from.vias.iter().enumerate() {
        let item = (!moving.contains(&via.id)).then(|| {
            let copper = net(&via.net, via.diameter / 2.0);
            set.add(point(via.position), Kind::Line, copper, Source::ViaRing, k)
        });
        found.via_rings.push(item);
    }
    for (k, graphic) in from.graphics.iter().enumerate() {
        let shape = &graphic.shape;
        let copper = net(&graphic.net, 0.0);
        let item = set.add(shape.clone(), kind_of(shape), copper, Source::Graphic, k);
        found.graphics.push(item);
    }
    for (k, shape) in from.track_keepouts.iter().enumerate() {
        let item = set.add(shape.clone(), kind_of(shape), None, Source::TrackKeepout, k);
        found.track_keepouts.push(item);
    }
    for (k, pad) in from.pads.iter().enumerate() {
        let item = (pad.drill > 0.0)
            .then(|| set.add(point(pad.position), Kind::Line, None, Source::PadHole, k));
        found.pad_holes.push(item);
    }
    for (k, via) in from.vias.iter().enumerate() {
        let item = (!moving.contains(&via.id))
            .then(|| set.add(point(via.position), Kind::Line, None, Source::ViaHole, k));
        found.via_holes.push(item);
    }
    for (k, shape) in from.via_keepouts.iter().enumerate() {
        let item = set.add(shape.clone(), kind_of(shape), None, Source::ViaKeepout, k);
        found.via_keepouts.push(item);
    }
    found.outline = from
        .outline
        .as_ref()
        .map(|outline| set.add(outline.clone(), Kind::Outside, None, Source::Outline, 0));
    let items = ItemSet::new(set.shapes, set.kinds, Some(set.copper), Some(interner))
        .with_records(set.records);
    (items, found)
}

/// A via standing still, by its layer and index there.
#[derive(Clone, Copy, Debug)]
struct ViaRef {
    layer: usize,
    index: usize,
}

/// A field's items as they are picked from the board's sets: each by its
/// layer and index there, with its inflation, and the items taken
/// otherwise than their sets hold them.
struct Built<'b> {
    board: &'b BoardItems,
    /// The layer of each part.
    parts: Vec<usize>,
    members: Vec<Member>,
    inflations: Vec<f64>,
    overrides: Vec<(u32, Override)>,
}

impl<'b> Built<'b> {
    fn new(board: &'b BoardItems) -> Self {
        Self {
            board,
            parts: Vec::new(),
            members: Vec::new(),
            inflations: Vec::new(),
            overrides: Vec::new(),
        }
    }

    /// Item `item` of layer l's set, inflated by `inflation`.
    fn push(&mut self, l: usize, item: u32, inflation: f64, how: Option<Override>) {
        let part = match self.parts.iter().position(|&p| p == l) {
            Some(part) => part,
            None => {
                self.parts.push(l);
                self.parts.len() - 1
            }
        };
        if let Some(how) = how {
            self.overrides.push((number(self.members.len()), how));
        }
        self.members.push(Member {
            part: number(part),
            index: item,
        });
        self.inflations.push(inflation);
    }

    /// Copper of `net` on layer l, item `item` of layer r's set, reaching
    /// `extent` mm beyond its geometry, with `clearance` its own, as an
    /// obstacle to what `keeper` keeps; nothing for copper of its own net.
    #[allow(clippy::too_many_arguments)]
    fn kept(
        &mut self,
        keeper: &Keeper,
        l: usize,
        (r, item): (usize, u32),
        net: &str,
        extent: f64,
        clearance: f64,
    ) {
        if let Some(inflation) = keeper.copper(net, extent, clearance) {
            let how = (r != l).then_some(Override::Layer(number(l)));
            self.push(r, item, inflation, how);
        }
    }

    /// Layer l's copper and track keepouts, with `vias` in place of the
    /// layer's vias, holes left out; `via_keeper` gives the keeper of each
    /// via's ring. `keep` says which of the layer's items are there.
    fn copper(
        &mut self,
        l: usize,
        keeper: &Keeper,
        vias: &[ViaRef],
        via_keeper: &dyn Fn(&Via) -> Keeper,
        keep: &Keep,
    ) {
        let board = self.board;
        let layer = board.layer(l);
        let records = &board.records[l];
        for (k, pad) in layer.pads.iter().enumerate() {
            if let Some(item) = records.pad_copper[k]
                && keep.pads.as_ref().is_none_or(|p| p[k])
            {
                let clearance = pad
                    .clearance_override
                    .unwrap_or_else(|| layer.clearance(&pad.net));
                self.kept(keeper, l, (l, item), &pad.net, 0.0, clearance);
            }
        }
        for (k, track) in layer.tracks.iter().enumerate() {
            if keep.tracks.as_ref().is_none_or(|t| t[k]) {
                self.kept(
                    keeper,
                    l,
                    (l, records.tracks[k]),
                    &track.net,
                    track.width / 2.0,
                    layer.clearance(&track.net),
                );
            }
        }
        for v in vias {
            let via = &board.layer(v.layer).vias[v.index];
            let ring = board.records[v.layer].via_rings[v.index]
                .expect("a via standing still is in its layer's set");
            self.kept(
                &via_keeper(via),
                l,
                (v.layer, ring),
                &via.net,
                via.diameter / 2.0,
                layer.clearance(&via.net),
            );
        }
        for (k, graphic) in layer.graphics.iter().enumerate() {
            if keep.graphics.as_ref().is_none_or(|g| g[k]) {
                self.kept(
                    keeper,
                    l,
                    (l, records.graphics[k]),
                    &graphic.net,
                    0.0,
                    layer.clearance(&graphic.net),
                );
            }
        }
        for &item in &records.track_keepouts {
            self.push(l, item, keeper.radius, None);
        }
    }

    /// The holes of layer l's pads `pads` keeps and of `vias`, as
    /// obstacles to `keeper`.
    fn holes(
        &mut self,
        l: usize,
        pads: &dyn Fn(usize, &Pad) -> bool,
        vias: &[ViaRef],
        keeper: &Keeper,
    ) {
        let board = self.board;
        for (k, pad) in board.layer(l).pads.iter().enumerate() {
            if pad.drill > 0.0
                && pads(k, pad)
                && let Some(inflation) = keeper.hole(&pad.net, pad.drill / 2.0, pad.plated)
            {
                let hole = board.records[l].pad_holes[k].expect("a drilled pad has a hole");
                self.push(l, hole, inflation, None);
            }
        }
        for v in vias {
            let via = &board.layer(v.layer).vias[v.index];
            if let Some(inflation) = keeper.hole(&via.net, via.drill / 2.0, true) {
                let hole = board.records[v.layer].via_holes[v.index]
                    .expect("a via standing still is in its layer's set");
                self.push(v.layer, hole, inflation, None);
            }
        }
    }

    /// The items' obstacles, then the board edge, layer l's outline
    /// inflated by `edge_inflation`, if it has one.
    fn with_edge(
        &mut self,
        outline: Option<usize>,
        edge_inflation: f64,
    ) -> (Vec<u32>, Option<f64>) {
        let keys: Vec<(ItemIds, f64)> = self
            .members
            .iter()
            .zip(&self.inflations)
            .map(|(m, &inflation)| {
                let set = &self.board.sets[self.parts[m.part as usize]];
                let ids = set.ids(m.index as usize).expect("a board's sets have ids");
                (ids, inflation)
            })
            .collect();
        let mut ids = obstacle_ids(&keys);
        match outline.and_then(|l| self.board.records[l].outline.map(|item| (l, item))) {
            None => (ids, None),
            Some((l, item)) => {
                ids.push(edge_obstacle(&ids));
                self.push(l, item, edge_inflation, None);
                (ids, Some(edge_inflation))
            }
        }
    }

    /// The field of the items, and the board edge as `with_edge` takes it.
    fn field(mut self, outline: Option<usize>, edge_inflation: f64) -> Obstacles {
        let (ids, edge_inflation) = self.with_edge(outline, edge_inflation);
        let parts = self
            .parts
            .iter()
            .map(|&l| self.board.sets[l].clone())
            .collect();
        let edges = StaticEdges::view(parts, self.members, self.inflations, ids, self.overrides);
        Obstacles::of(edges, edge_inflation)
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
    board: &BoardItems,
    l: usize,
    keeper: &Keeper,
    edge_clearance: f64,
    keep: &Keep,
    vias: &[ViaRef],
) -> Obstacles {
    let mut built = Built::new(board);
    built.copper(l, keeper, vias, &|_| keeper.clone(), keep);
    let pads = |k: usize, _: &Pad| keep.pads.as_ref().is_none_or(|p| p[k]);
    built.holes(l, &pads, vias, keeper);
    built.field(Some(l), edge_clearance + keeper.radius)
}

/// Obstacles for a track of `net` and `width` mm on the board's layer l.
///
/// Copper clearance is the larger of both items' netclass clearances, with
/// a pad's own override replacing its netclass. Copper of `net` is never
/// an obstacle; its holes are not either, except unplated ones. The moving
/// vias, which the optimizer adds at their current positions, are left
/// out.
pub fn chain_obstacles(
    board: &BoardItems,
    l: usize,
    net: &str,
    width: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> Obstacles {
    let layer = board.layer(l);
    let keeper = Keeper::track(Some(net), width, layer.clearance(net), hole_clearance);
    layer_obstacles(
        board,
        l,
        &keeper,
        edge_clearance,
        &Keep::default(),
        &board.still_vias(l),
    )
}

/// One track of a coupled section: its net and centreline.
pub struct PairTrack<'a> {
    pub net: &'a str,
    pub points: &'a [V2],
}

/// Obstacles for the centreline of a coupled section of a differential
/// pair, its tracks `a` and `b` of `width` `gap` mm apart, on the board's
/// layer l.
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
/// clear of it in its own fan-out's place. The moving vias are left out, as
/// for a chain (`chain_obstacles`).
#[allow(clippy::too_many_arguments)]
pub fn pair_obstacles(
    board: &BoardItems,
    l: usize,
    a: PairTrack<'_>,
    b: PairTrack<'_>,
    width: f64,
    gap: f64,
    edge_clearance: f64,
    hole_clearance: f64,
) -> Obstacles {
    let layer = board.layer(l);
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
    let vias: Vec<ViaRef> = board
        .still_vias(l)
        .into_iter()
        .filter(|r| {
            let v = &layer.vias[r.index];
            free(&v.net, &point(v.position), v.diameter / 2.0)
        })
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
    layer_obstacles(board, l, &keeper, edge_clearance, &keep, &vias)
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
    fn of(board: &BoardItems, via: &Via, holes: HoleRules) -> Result<Self, NoLayer> {
        let layers = board.layers();
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
                if v.id == via.id || board.is_moving(v) {
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
    fn via_keeper(&self, board: &BoardItems, zones: &[Zone], other: &Via) -> Keeper {
        let on: Vec<i32> = self
            .spanned
            .iter()
            .map(|&l| board.layer(l).ordinal)
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
    pub field: Obstacles,
    /// The vias of `split`, as obstacles to the via, None if none is one.
    pub split: Option<Obstacles>,
}

/// The fabrication rules and clearances a via keeps.
#[derive(Clone, Copy, Debug)]
pub struct ViaRules {
    pub edge_clearance: f64,
    pub holes: HoleRules,
}

/// Obstacles for `via` at any position, over every layer of the board it
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
/// minimum. The via itself and the moving vias, which the caller adds at
/// their current positions, are left out; so is the copper of zones, which
/// KiCad refills.
///
/// The vias of other nets whose ids are in `split` are left out of the
/// field and come back apart, as obstacles to `via`, each as the field
/// would count it.
pub fn via_obstacles(
    board: &BoardItems,
    via: &Via,
    split: &HashSet<&str>,
    rules: ViaRules,
    zones: &[Zone],
) -> Result<ViaFields, NoLayer> {
    let found = Surroundings::of(board, via, rules.holes)?;
    let is_split = |r: &ViaRef| split.contains(board.layer(r.layer).vias[r.index].id.as_str());
    let (picked, kept): (Vec<ViaRef>, Vec<ViaRef>) =
        found.others.iter().copied().partition(|r| is_split(r));
    let field = via_field(board, &found, via, &kept, rules.edge_clearance, zones);
    if picked.is_empty() {
        return Ok(ViaFields { field, split: None });
    }
    let first = found.spanned[0];
    let mut built = Built::new(board);
    for r in &picked {
        let other = &board.layer(r.layer).vias[r.index];
        let ring = board.records[r.layer].via_rings[r.index]
            .expect("a via standing still is in its layer's set");
        built.kept(
            &found.via_keeper(board, zones, other),
            first,
            (r.layer, ring),
            &other.net,
            other.diameter / 2.0,
            board.layer(first).clearance(&other.net),
        );
    }
    built.holes(first, &|_, _| false, &picked, &found.keeper);
    Ok(ViaFields {
        field,
        split: Some(built.field(None, 0.0)),
    })
}

fn via_field(
    board: &BoardItems,
    found: &Surroundings,
    via: &Via,
    others: &[ViaRef],
    edge_clearance: f64,
    zones: &[Zone],
) -> Obstacles {
    let keeper = &found.keeper;
    let on = |l: usize| keeper.with_zones(zones_on(zones, &[board.layer(l).ordinal]));
    let first = found.spanned[0];
    let mut built = Built::new(board);
    // The via spans every layer here, so another via's ring counts once, as
    // do pads' holes, which are the same on every layer. The copper items
    // include track keepouts, which hold the via back as well, since the
    // chain ends it joins sit at its centre.
    let via_keeper = |other: &Via| found.via_keeper(board, zones, other);
    built.copper(first, &on(first), others, &via_keeper, &Keep::default());
    for &l in &found.spanned[1..] {
        let keeper = on(l);
        built.copper(l, &keeper, &[], &|_| keeper.clone(), &Keep::default());
    }
    built.holes(first, &|_, p| !plated_of(p, via), others, keeper);
    for &l in &found.spanned {
        for &item in &board.records[l].via_keepouts {
            built.push(l, item, keeper.radius, None);
        }
    }
    built.field(Some(first), edge_clearance + keeper.radius)
}

/// What of a via's net limits it: its pads and holes, and its vias.
pub struct OwnNetStops {
    pub items: Option<Obstacles>,
    pub vias: Option<ChainField>,
}

/// What of `via`'s net on the layers of the board it spans limits it, as
/// obstacles that stop it without repelling it.
///
/// Copper of one net does not interact, but two drilling rules hold: the
/// holes of its net's vias, but the moving ones, and of its net's plated
/// pads keep hole-to-hole from its hole, as two moving vias of one net
/// keep (`rules::via_via_inflation`); and its ring may touch its net's pads
/// but not overlap them, since a via in a pad needs the fabricator to fill
/// and cap it, unless `allow_via_in_pad`, which KiCad's DRC does not check
/// either.
pub fn own_net_stops(
    board: &BoardItems,
    via: &Via,
    holes: HoleRules,
    allow_via_in_pad: bool,
) -> Result<OwnNetStops, NoLayer> {
    let found = Surroundings::of(board, via, holes)?;
    let first = found.spanned[0];
    let mut built = Built::new(board);
    built.holes(first, &|_, p| plated_of(p, via), &[], &found.keeper);
    if !allow_via_in_pad {
        for &l in &found.spanned {
            for (k, pad) in board.layer(l).pads.iter().enumerate() {
                if pad.net == via.net
                    && let Some(item) = board.records[l].pad_copper[k]
                {
                    built.push(l, item, found.keeper.radius, Some(Override::Stop));
                }
            }
        }
    }
    let items = (!built.members.is_empty()).then(|| built.field(None, 0.0));
    let own = found.own.iter().map(|r| {
        (
            &board.layer(r.layer).vias[r.index],
            found.clearance,
            Vec::new(),
        )
    });
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
    use crate::field::items::Picked;
    use crate::field::obstacles::Item;
    use crate::rules::{HOLE_CLEARANCE, HOLE_TO_HOLE, Span};
    use crate::testing::Random;

    const HOLES: HoleRules = HoleRules {
        hole_clearance: HOLE_CLEARANCE,
        hole_to_hole: HOLE_TO_HOLE,
    };
    const RULES: ViaRules = ViaRules {
        edge_clearance: 0.3,
        holes: HOLES,
    };
    /// Points a field is asked at, over the boards below.
    const POINTS: usize = 300;
    const SIDE: f64 = 12.0;
    const REACH: f64 = 0.5;

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

    fn square(x: f64, y: f64, half: f64) -> Shape {
        Shape::polygon(vec![vec![
            V2::new(x - half, y - half),
            V2::new(x + half, y - half),
            V2::new(x + half, y + half),
            V2::new(x - half, y + half),
            V2::new(x - half, y - half),
        ]])
    }

    fn pad(net: &str, at: (f64, f64), plated: bool) -> Pad {
        Pad {
            net: net.into(),
            position: V2::new(at.0, at.1),
            drill: 0.8,
            plated,
            shape: Some(square(at.0, at.1, 1.0)),
            clearance_override: None,
        }
    }

    fn track(net: &str, from: (f64, f64), to: (f64, f64), width: f64) -> Track {
        Track {
            net: net.into(),
            width,
            centreline: Shape::line(vec![V2::new(from.0, from.1), V2::new(to.0, to.1)]),
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

    fn board(layers: Vec<Layer>, moving: &[&str]) -> BoardItems {
        BoardItems::new(
            layers.into_iter().map(Arc::new).collect(),
            moving.iter().map(|&id| id.to_owned()).collect(),
        )
    }

    /// A layer of copper of several nets: pads of two shapes, tracks
    /// meeting end to end, at a width change and at a T-junction, vias,
    /// a keepout and the board outline.
    fn busy(name: &str, ordinal: i32) -> Layer {
        let mut busy = layer(name, ordinal);
        busy.pads = vec![
            pad("SIG", (2.0, 2.0), true),
            pad("SIG", (5.0, 2.0), false),
            pad("GND", (8.0, 2.0), true),
            Pad {
                clearance_override: Some(0.4),
                ..pad("HV", (2.0, 9.0), true)
            },
        ];
        busy.tracks = vec![
            track("GND", (1.0, 5.0), (4.0, 5.0), 0.25),
            track("GND", (4.0, 5.0), (6.0, 5.0), 0.25),
            track("GND", (6.0, 5.0), (8.0, 5.0), 0.4),
            track("GND", (5.0, 5.0), (5.0, 7.0), 0.25),
            track("SIG", (1.0, 4.0), (10.0, 4.0), 0.25),
            track("HV", (3.0, 10.0), (10.0, 10.0), 0.25),
        ];
        busy.vias = vec![
            via("moving", "GND", (7.0, 7.0)),
            via("still", "GND", (9.0, 7.0)),
            via("signal", "SIG", (3.0, 7.0)),
        ];
        busy.graphics = vec![Graphic {
            net: String::new(),
            shape: square(10.0, 2.0, 0.5),
        }];
        busy.track_keepouts = vec![square(10.0, 8.0, 0.5)];
        busy.via_keepouts = vec![square(1.0, 1.0, 0.3)];
        busy.outline = Some(square(SIDE / 2.0, SIDE / 2.0, SIDE / 2.0));
        busy.clearances = HashMap::from([("HV".to_owned(), 0.4)]);
        busy
    }

    fn sources(field: &Obstacles) -> Vec<(Source, usize)> {
        field
            .picked()
            .unwrap()
            .iter()
            .map(|p| (p.source, p.index))
            .collect()
    }

    /// `field` built again from copies of its items, as `Obstacles::new`
    /// builds a field.
    fn owned(board: &BoardItems, field: &Obstacles) -> Obstacles {
        let edges = field.edges();
        let items = field
            .picked()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let layer = board.layer(p.layer);
                let (net, extent) = match p.source {
                    Source::Pad => (layer.pads[p.index].net.clone(), 0.0),
                    Source::Track => {
                        let t = &layer.tracks[p.index];
                        (t.net.clone(), t.width / 2.0)
                    }
                    Source::ViaRing => {
                        let v = &layer.vias[p.index];
                        (v.net.clone(), v.diameter / 2.0)
                    }
                    Source::Graphic => (layer.graphics[p.index].net.clone(), 0.0),
                    _ => (String::new(), 0.0),
                };
                let copper = p.copper.map(|l| board.layer(l).name.clone());
                Item {
                    shape: edges.shape(k).clone(),
                    inflation: edges.inflations()[k],
                    net: copper.is_some().then_some(net),
                    extent: if copper.is_some() { extent } else { 0.0 },
                    layer: copper,
                }
            })
            .collect();
        let outline = field
            .edge_inflation()
            .map(|_| edges.shape(edges.len() - 1).clone());
        Obstacles::new(items, outline, field.edge_inflation().unwrap_or(0.0))
    }

    /// Asserts that `field` answers as one built from copies of its items.
    fn answers_as_owned(board: &BoardItems, field: &Obstacles) {
        let again = owned(board, field);
        let (a, b) = (field.edges(), again.edges());
        assert_eq!(a.obstacles(), b.obstacles());
        assert_eq!(a.clearance_obstacles(), b.clearance_obstacles());
        let mut random = Random(195);
        let points: Vec<V2> = (0..POINTS).map(|_| random.point(SIDE)).collect();
        let ends: Vec<V2> = (0..POINTS).map(|_| random.point(SIDE)).collect();
        for reach in [REACH, f64::INFINITY] {
            assert_eq!(
                field.signed_distance(&points, reach),
                again.signed_distance(&points, reach)
            );
            assert_eq!(
                field.smooth_distance(&points, reach),
                again.smooth_distance(&points, reach)
            );
            assert_eq!(a.distances(&points, reach), b.distances(&points, reach));
            assert_eq!(
                a.search(&points, reach, REACH),
                b.search(&points, reach, REACH)
            );
        }
        assert_eq!(
            field.segments_clear(&points, &ends, 0.0),
            again.segments_clear(&points, &ends, 0.0)
        );
    }

    #[test]
    fn same_net_copper_is_no_obstacle_but_an_unplated_hole_is() {
        let mut top = layer("F.Cu", 0);
        top.pads = vec![pad("SIG", (0.0, 0.0), true), pad("SIG", (5.0, 0.0), false)];
        top.tracks = vec![track("SIG", (0.0, 3.0), (5.0, 3.0), 0.2)];
        let board = board(vec![top], &[]);
        let built = chain_obstacles(&board, 0, "SIG", 0.2, 0.3, HOLE_CLEARANCE);
        // The unplated pad's copper is of the net too; only its hole stays.
        assert_eq!(sources(&built), vec![(Source::PadHole, 1)]);
        answers_as_owned(&board, &built);
        let other = chain_obstacles(&board, 0, "GND", 0.2, 0.3, HOLE_CLEARANCE);
        assert_eq!(other.picked().unwrap().len(), 5);
        answers_as_owned(&board, &other);
    }

    #[test]
    fn a_chain_s_field_answers_as_one_of_its_own_items() {
        let board = board(vec![busy("F.Cu", 0), busy("B.Cu", 2)], &["moving"]);
        for l in 0..2 {
            for net in ["SIG", "GND", "HV", ""] {
                for width in [0.25, 0.4] {
                    let field = chain_obstacles(&board, l, net, width, 0.3, HOLE_CLEARANCE);
                    assert!(field.picked().unwrap().iter().all(|p| p.layer == l));
                    answers_as_owned(&board, &field);
                }
            }
        }
    }

    #[test]
    fn a_section_s_field_leaves_out_what_overlaps_its_own_track() {
        let board = board(vec![busy("F.Cu", 0)], &[]);
        let along = [V2::new(1.0, 4.0), V2::new(10.0, 4.0)];
        let beside = [V2::new(1.0, 4.4), V2::new(10.0, 4.4)];
        let field = pair_obstacles(
            &board,
            0,
            PairTrack {
                net: "SIG",
                points: &along,
            },
            PairTrack {
                net: "N",
                points: &beside,
            },
            0.25,
            0.15,
            0.3,
            HOLE_CLEARANCE,
        );
        let tracks: Vec<usize> = sources(&field)
            .into_iter()
            .filter(|&(source, _)| source == Source::Track)
            .map(|(_, k)| k)
            .collect();
        assert_eq!(tracks, vec![0, 1, 2, 3, 5]);
        answers_as_owned(&board, &field);
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
        let me = top.vias[0].clone();
        let board = board(vec![top], &["me"]);
        let split = HashSet::from(["ground"]);
        let fields = via_obstacles(&board, &me, &split, RULES, &[]).unwrap();
        let rings: Vec<usize> = sources(&fields.field)
            .into_iter()
            .filter(|&(source, _)| source == Source::ViaRing)
            .map(|(_, k)| k)
            .collect();
        assert_eq!(rings, vec![2]);
        let apart = fields.split.unwrap();
        assert_eq!(
            sources(&apart),
            vec![(Source::ViaRing, 1), (Source::ViaHole, 1)]
        );
        let stops = own_net_stops(&board, &me, HOLES, false).unwrap();
        assert!(stops.items.is_none());
        let vias = stops.vias.unwrap();
        // Hole to hole from the via of its net: 0.15 + 0.15 + 0.25.
        let (d, _) = vias.signed_distance(&[V2::new(6.0, 0.0)], f64::INFINITY);
        assert!((d[0] + 0.55).abs() < 1e-12);
    }

    #[test]
    fn moving_vias_leave_a_chain_s_field_keeping_the_others_indices() {
        let mut top = layer("F.Cu", 0);
        top.vias = vec![
            via("moving", "GND", (0.0, 0.0)),
            via("still", "GND", (2.0, 0.0)),
        ];
        let board = board(vec![top], &["moving"]);
        let built = chain_obstacles(&board, 0, "SIG", 0.2, 0.3, HOLE_CLEARANCE);
        assert_eq!(
            sources(&built),
            vec![(Source::ViaRing, 1), (Source::ViaHole, 1)]
        );
        answers_as_owned(&board, &built);
    }

    /// Three busy layers, the through via "moving" moving, and a blind
    /// via standing on the inner and bottom layers only.
    fn stack() -> (BoardItems, Via) {
        let top = busy("F.Cu", 0);
        let mut inner = busy("In1.Cu", 1);
        let mut bottom = busy("B.Cu", 2);
        let blind = Via {
            span: Span { low: 1, high: 2 },
            ..via("blind", "HV", (6.0, 8.0))
        };
        inner.vias.push(blind.clone());
        bottom.vias.push(blind);
        let me = top.vias[0].clone();
        (board(vec![top, inner, bottom], &["moving"]), me)
    }

    /// A zone on the inner and bottom layers whose clearance exceeds the
    /// netclass rule.
    fn plane() -> Vec<Zone> {
        vec![Zone {
            clearance: 0.6,
            margin: 0.2,
            layers: vec![1, 2],
        }]
    }

    #[test]
    fn a_via_s_fields_answer_as_ones_of_their_own_items() {
        let (board, me) = stack();
        for zones in [Vec::new(), plane()] {
            for split in [HashSet::new(), HashSet::from(["signal", "blind"])] {
                let fields = via_obstacles(&board, &me, &split, RULES, &zones).unwrap();
                answers_as_owned(&board, &fields.field);
                if let Some(apart) = &fields.split {
                    answers_as_owned(&board, apart);
                }
            }
        }
        for allow in [false, true] {
            let stops = own_net_stops(&board, &me, HOLES, allow).unwrap();
            answers_as_owned(&board, &stops.items.unwrap());
        }
    }

    #[test]
    fn a_blind_via_s_ring_counts_on_the_field_s_first_layer() {
        let (board, me) = stack();
        let fields = via_obstacles(&board, &me, &HashSet::new(), RULES, &[]).unwrap();
        let ring = fields
            .field
            .picked()
            .unwrap()
            .into_iter()
            .find(|p| {
                p.source == Source::ViaRing && board.layer(p.layer).vias[p.index].id == "blind"
            })
            .unwrap();
        // Last met on the bottom layer, its copper is the top layer's.
        assert_eq!((ring.layer, ring.copper), (2, Some(0)));
    }

    #[test]
    fn a_pad_on_every_layer_is_one_obstacle_to_a_via() {
        let (board, me) = stack();
        let fields = via_obstacles(&board, &me, &HashSet::new(), RULES, &[]).unwrap();
        let obstacles = fields.field.edges().obstacles();
        let pads: Vec<(usize, u32)> = fields
            .field
            .picked()
            .unwrap()
            .iter()
            .zip(obstacles)
            .filter(|(p, _)| p.source == Source::Pad && p.index == 0)
            .map(|(p, &o)| (p.layer, o))
            .collect();
        assert_eq!(pads.len(), 3);
        assert!(pads.iter().all(|&(_, o)| o == pads[0].1));
    }

    #[test]
    fn a_zoned_via_s_field_holds_the_plain_one_s_items_inflated_apart() {
        let (board, me) = stack();
        let plain = via_obstacles(&board, &me, &HashSet::new(), RULES, &[]).unwrap();
        let zoned = via_obstacles(&board, &me, &HashSet::new(), RULES, &plane()).unwrap();
        assert_eq!(plain.field.picked(), zoned.field.picked());
        let (a, b) = (plain.field.edges(), zoned.field.edges());
        assert!(
            a.inflations()
                .iter()
                .zip(b.inflations())
                .any(|(x, y)| x < y)
        );
        // The inner layer's width change and T-junction keep their
        // obstacles apart in both, their inflations growing alike.
        assert_eq!(a.obstacle_count(), b.obstacle_count());
    }

    #[test]
    fn a_via_s_own_pads_stop_it_as_no_copper() {
        let (board, me) = stack();
        let stops = own_net_stops(&board, &me, HOLES, false).unwrap();
        let picked = stops.items.unwrap().picked().unwrap();
        let pads: Vec<&Picked> = picked
            .iter()
            .filter(|p| p.source == Source::PadStop)
            .collect();
        // The ground pad on each of the three layers.
        assert_eq!(pads.len(), 3);
        assert!(pads.iter().all(|p| p.copper.is_none() && p.index == 2));
    }

    #[test]
    fn a_via_s_overlapping_own_pads_stay_apart_to_its_clearance_checks() {
        let mut top = layer("F.Cu", 0);
        top.vias = vec![via("me", "GND", (0.0, 0.0))];
        // Two pads of its net whose copper overlaps.
        top.pads = vec![pad("GND", (3.0, 0.0), true), pad("GND", (4.5, 0.0), true)];
        let me = top.vias[0].clone();
        let board = board(vec![top], &["me"]);
        let stops = own_net_stops(&board, &me, HOLES, false)
            .unwrap()
            .items
            .unwrap();
        let edges = stops.edges();
        let stop: Vec<u32> = stops
            .picked()
            .unwrap()
            .iter()
            .zip(edges.clearance_obstacles())
            .filter(|(p, _)| p.source == Source::PadStop)
            .map(|(_, &o)| o)
            .collect();
        assert_eq!(stop.len(), 2);
        // Held as no copper, they are not joined as touching copper is.
        assert_ne!(stop[0], stop[1]);
        let between = edges.distances(&[V2::new(3.75, 0.0)], f64::INFINITY);
        assert!(between.obstacle.contains(&(stop[0] as usize)));
        assert!(between.obstacle.contains(&(stop[1] as usize)));
        answers_as_owned(&board, &stops);
    }

    #[test]
    fn a_via_of_no_spanned_layer_is_an_error() {
        let me = Via {
            span: Span { low: 3, high: 4 },
            ..via("me", "SIG", (0.0, 0.0))
        };
        let board = board(vec![layer("F.Cu", 0)], &["me"]);
        let result = own_net_stops(&board, &me, HOLES, false);
        assert_eq!(result.err().map(|e| e.0), Some("me".to_owned()));
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
