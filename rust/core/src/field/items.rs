//! The items static fields are made of, held once and shared by every
//! field that takes them (`edges::StaticEdges`).
//!
//! An item set holds what an item is whatever field holds it: its
//! geometry, its kind (`Kind`), its edges, numbered together item by item,
//! its bounding box, the pieces of its edges a search near a point takes
//! (`ItemSet::pieces`, built when first asked), the tree over the boxes of
//! the items but the `Outside` ones, and for copper its net, layer and
//! extent. What differs between fields, the inflation and the obstacle of
//! each item and which items are there, is the field's own.
//!
//! The ids under which items are one obstacle (`ItemIds`) are interned
//! (`Interner`): equal exactly when the geometry, or the end of a
//! centreline, is equal to the bit, so that a field numbers its obstacles
//! from them as from the geometry itself (`obstacles::obstacle_ids`). Ids
//! of two interners are not comparable, so a field takes the sets of one
//! interner only.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::pairs::{BoxTree, PointTree};
use super::{Kind, MAX_EDGE_LENGTH};
use crate::geometry::{LINE_STRING, Part, Shape, V2};

/// What a clearance check joins of the items: copper of one net on one
/// layer that touches is one obstacle to it. Nets are numbered by the
/// set's interner; layers by the index of their layer in the board for a
/// board's sets (`build::BoardItems`), by the interner for a set of a
/// field's own items (`obstacles::Obstacles::new`). None for an item that
/// is not copper of one net on one layer.
#[derive(Clone, Debug, Default)]
pub struct Copper {
    pub net: Vec<Option<u32>>,
    pub layer: Vec<Option<u32>>,
    /// How far each item's copper reaches beyond its geometry, mm.
    pub extent: Vec<f64>,
}

/// What an item is made of.
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
    /// The board outline, which a field names apart from its items.
    Outline = 9,
}

/// Where an item comes from: its kind, the layer whose record it is and
/// its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub source: Source,
    pub layer: u32,
    pub index: u32,
}

/// Where a field's item comes from, as `Record`, and the layer of its
/// copper in the field, None for what is not copper of one layer there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Picked {
    pub source: Source,
    pub layer: usize,
    pub index: usize,
    pub copper: Option<usize>,
}

/// The ids of an item's geometry and, for a line string, of its two ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemIds {
    pub geometry: u32,
    pub ends: Option<[u32; 2]>,
}

static INTERNERS: AtomicU64 = AtomicU64::new(1);

/// Numbers geometries, points and names, each equal exactly when its bits
/// are.
pub struct Interner {
    token: u64,
    geometries: HashMap<Vec<u64>, u32>,
    points: HashMap<[u64; 2], u32>,
    names: HashMap<String, u32>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

fn next_id<K: std::hash::Hash + Eq>(map: &mut HashMap<K, u32>, key: K) -> u32 {
    let next = u32::try_from(map.len()).expect("fewer than 2³² keys");
    *map.entry(key).or_insert(next)
}

impl Interner {
    pub fn new() -> Self {
        Self {
            token: INTERNERS.fetch_add(1, Ordering::Relaxed),
            geometries: HashMap::new(),
            points: HashMap::new(),
            names: HashMap::new(),
        }
    }

    /// The ids of `shape`: its geometry by its type and coordinates, and
    /// for a line string its first and last points.
    pub fn ids(&mut self, shape: &Shape) -> ItemIds {
        let geometry = next_id(&mut self.geometries, geometry_key(shape));
        let mut ends = None;
        if shape.type_id == LINE_STRING
            && let Some(Part::Line(line)) = shape.parts.first()
            && let (Some(a), Some(b)) = (line.first(), line.last())
        {
            ends = Some([self.point(*a), self.point(*b)]);
        }
        ItemIds { geometry, ends }
    }

    fn point(&mut self, p: V2) -> u32 {
        next_id(&mut self.points, [p.x.to_bits(), p.y.to_bits()])
    }

    /// The number of a net's or a layer's name.
    pub fn name(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        next_id(&mut self.names, name.to_owned())
    }
}

fn geometry_key(shape: &Shape) -> Vec<u64> {
    let mut key = vec![
        u64::from(shape.type_id.unsigned_abs()),
        shape.parts.len() as u64,
    ];
    let points = |points: &[V2], key: &mut Vec<u64>| {
        key.push(points.len() as u64);
        for p in points {
            key.push(p.x.to_bits());
            key.push(p.y.to_bits());
        }
    };
    for part in &shape.parts {
        match part {
            Part::Polygon(rings) => {
                key.extend([0, rings.len() as u64]);
                for ring in rings {
                    points(ring, &mut key);
                }
            }
            Part::Line(line) => {
                key.push(1);
                points(line, &mut key);
            }
        }
    }
    key
}

/// An item's edges split into pieces of at most MAX_EDGE_LENGTH: a tree
/// over their middles, each piece's edge, numbered from the item's first,
/// and half the longest piece.
pub struct Pieces {
    pub tree: PointTree,
    pub edge_of: Vec<usize>,
    pub half_piece: f64,
}

/// Items, each a geometry of a kind, as several fields share them.
pub struct ItemSet {
    shapes: Vec<Shape>,
    kinds: Vec<Kind>,
    first: Vec<usize>,
    count: Vec<usize>,
    starts: Vec<V2>,
    ends: Vec<V2>,
    bounds: Vec<Option<[f64; 4]>>,
    pieces: Vec<OnceLock<Pieces>>,
    tree: BoxTree,
    copper: Option<Copper>,
    records: Option<Vec<Record>>,
    ids: Option<Vec<ItemIds>>,
    /// The interner of `ids` and of `copper`'s numbers, 0 for none.
    token: u64,
}

impl ItemSet {
    /// Items of `shapes`, each of its kind; with `copper`, what each is as
    /// copper; with `interner`, their ids, which a field numbers its
    /// obstacles from.
    pub fn new(
        shapes: Vec<Shape>,
        kinds: Vec<Kind>,
        copper: Option<Copper>,
        interner: Option<&mut Interner>,
    ) -> Self {
        let mut first = Vec::with_capacity(shapes.len());
        let mut count = Vec::with_capacity(shapes.len());
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for shape in &shapes {
            let edges = shape.edges();
            first.push(starts.len());
            count.push(edges.len());
            for (a, b) in edges {
                starts.push(a);
                ends.push(b);
            }
        }
        let bounds: Vec<Option<[f64; 4]>> = shapes.iter().map(Shape::bounds).collect();
        let tree = BoxTree::new(
            (0..shapes.len())
                .filter(|&k| kinds[k] != Kind::Outside)
                .filter_map(|k| {
                    bounds[k].map(|[x0, y0, x1, y1]| (k, V2::new(x0, y0), V2::new(x1, y1)))
                }),
        );
        let (ids, token) = match interner {
            None => (None, 0),
            Some(interner) => (
                Some(shapes.iter().map(|s| interner.ids(s)).collect()),
                interner.token,
            ),
        };
        Self {
            pieces: (0..shapes.len()).map(|_| OnceLock::new()).collect(),
            shapes,
            kinds,
            first,
            count,
            starts,
            ends,
            bounds,
            tree,
            copper,
            records: None,
            ids,
            token,
        }
    }

    /// The same set, each item from its record.
    pub fn with_records(mut self, records: Vec<Record>) -> Self {
        assert_eq!(records.len(), self.len(), "one record per item");
        self.records = Some(records);
        self
    }

    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    pub fn shape(&self, item: usize) -> &Shape {
        &self.shapes[item]
    }

    pub fn kind(&self, item: usize) -> Kind {
        self.kinds[item]
    }

    pub fn bounds(&self, item: usize) -> Option<[f64; 4]> {
        self.bounds[item]
    }

    /// The item's edges, numbered across the set's items.
    pub fn edge_range(&self, item: usize) -> Range<usize> {
        self.first[item]..self.first[item] + self.count[item]
    }

    pub fn edge_count(&self, item: usize) -> usize {
        self.count[item]
    }

    pub fn start(&self, edge: usize) -> V2 {
        self.starts[edge]
    }

    pub fn end(&self, edge: usize) -> V2 {
        self.ends[edge]
    }

    /// The tree over the boxes of the items but the `Outside` ones.
    pub fn tree(&self) -> &BoxTree {
        &self.tree
    }

    pub fn copper(&self) -> Option<&Copper> {
        self.copper.as_ref()
    }

    /// The item's record; None for a set without them.
    pub fn record(&self, item: usize) -> Option<Record> {
        self.records.as_ref().map(|records| records[item])
    }

    /// The item's ids; None for a set without them.
    pub fn ids(&self, item: usize) -> Option<ItemIds> {
        self.ids.as_ref().map(|ids| ids[item])
    }

    /// The interner of the set's ids, 0 for none.
    pub fn token(&self) -> u64 {
        self.token
    }

    /// The item's pieces, built when first asked.
    pub fn pieces(&self, item: usize) -> &Pieces {
        self.pieces[item].get_or_init(|| {
            let mut middles = Vec::new();
            let mut edge_of = Vec::new();
            let mut longest: f64 = 0.0;
            for (local, e) in self.edge_range(item).enumerate() {
                let a = self.starts[e];
                let along = self.ends[e] - a;
                let length = along.hypot();
                let parts = (length / MAX_EDGE_LENGTH).ceil().max(1.0);
                longest = longest.max(length / parts);
                // An f64 count of pieces of a finite edge is a small whole
                // number.
                let pieces = parts as usize;
                for part in 0..pieces {
                    middles.push(a + ((part as f64 + 0.5) / parts) * along);
                    edge_of.push(local);
                }
            }
            Pieces {
                tree: PointTree::new(&middles),
                edge_of,
                half_piece: longest / 2.0,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ends(interner: &mut Interner, a: (f64, f64), b: (f64, f64)) -> [u32; 2] {
        let line = Shape::line(vec![V2::new(a.0, a.1), V2::new(b.0, b.1)]);
        interner.ids(&line).ends.unwrap()
    }

    #[test]
    fn points_share_an_id_exactly_when_their_bits_are_equal() {
        let mut interner = Interner::new();
        let [origin, right] = ends(&mut interner, (0.0, 0.0), (1.0, 0.0));
        let [again, up] = ends(&mut interner, (0.0, 0.0), (0.0, 1.0));
        let [_, near] = ends(&mut interner, (0.0, 0.0), (1.0, 1e-300));
        assert_eq!(origin, again);
        assert_ne!(right, up);
        assert_ne!(right, near);
        assert_ne!(origin, right);
        // -0.0 equals 0.0 but has other bits.
        let [negative, _] = ends(&mut interner, (-0.0, 0.0), (1.0, 0.0));
        assert_ne!(negative, origin);
    }

    #[test]
    fn geometries_share_an_id_exactly_when_their_coordinates_are_equal() {
        let mut interner = Interner::new();
        let line = |y: f64| Shape::line(vec![V2::new(0.0, 0.0), V2::new(1.0, y)]);
        let a = interner.ids(&line(0.0)).geometry;
        assert_eq!(a, interner.ids(&line(0.0)).geometry);
        assert_ne!(a, interner.ids(&line(1e-300)).geometry);
    }
}
