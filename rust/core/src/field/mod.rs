//! Inflated obstacles on the chain's layer, or on every layer a moving via
//! spans, answering distance queries.
//!
//! Each obstacle is raw geometry (a copper polygon, a track centreline, a
//! via or hole centre) plus an inflation: the distance from that geometry
//! at which the chain's centreline, or the via's centre, reaches the
//! required clearance. Distances are measured to the raw geometry and the
//! inflation subtracted, which is the exact signed distance to the inflated
//! shape without buffering it. The board edge is the outside of the
//! Edge.Cuts outline, inflated the same way.
//!
//! Every field answers two questions. `signed_distance` is the exact
//! distance to the nearest inflated obstacle, which every clearance
//! judgement takes. `smooth_distance` is what the energy takes: the smooth
//! minimum over the field's obstacles (`smooth::smooth_minimum`), each
//! obstacle's own distance the exact one to the nearest of its parts. What
//! one obstacle is depends on the field: in `Obstacles`, the items of one
//! geometry, every connected run of centrelines of one inflation that meet
//! end to end, and the board edge (`obstacles::obstacle_ids`); in a chain
//! field, the edges its caller groups, a chain or a via.
//!
//! `Obstacles` answers `signed_distance` and every segment check item by
//! item, and `smooth_distance` from its items as edges (`StaticEdges`): a
//! search with the field's tree finds the items near the points and the
//! edges of them that can answer, and those are measured by GEOS's
//! point-to-segment formula (`search::static_smooth`). A moving via's
//! clearance checks take each obstacle's exact distance from the same edges
//! (`search::static_distances`), so that they judge static copper obstacle
//! by obstacle; to them an obstacle of `Obstacles` is its connected copper
//! of one net on one layer (`StaticEdges::clearance_obstacles`), since the
//! energy's split at a width change or a T-junction is no gap in the
//! copper. Every distance is measured by GEOS's formulas (`geometry`), so an
//! `Obstacles` field's distances are shapely's to round-off. The energy
//! keeps each member's and moving via's search between evaluations
//! (`neighbours::StaticList`), so that static copper costs no tree query
//! per evaluation.

pub mod build;
pub mod chain;
pub mod composite;
pub mod edges;
pub mod obstacles;
pub mod pairs;
pub mod search;
pub mod smooth;
mod union_find;

use std::sync::Arc;

use crate::geometry::V2;
use edges::StaticEdges;

/// Signed distance to the nearest inflated obstacle boundary, and segment
/// checks.
pub trait Field: Send + Sync {
    /// The distance at each point and its unit gradient. The distance is
    /// positive in free space and negative inside an inflated obstacle; the
    /// gradient points toward increasing distance. Distances beyond `reach`
    /// may be +inf with a zero gradient.
    fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>);

    /// `signed_distance` as the energy takes it: the smooth minimum over the
    /// field's obstacles (`smooth::smooth_minimum`) of each one's exact
    /// distance, and its gradient, the obstacles' unit gradients weighted
    /// by their shares. Obstacles farther than `reach` plus SMOOTH_REACH may
    /// be left out, and a point with none gets +inf.
    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>);

    /// Whether each segment from starts[i] to ends[i] stays out of every
    /// inflated obstacle: it does not enter the region where the signed
    /// distance is below -`tolerance`; touching the boundary is clear.
    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool>;

    /// The field as `StaticEdges`, one per field it joins, for a search
    /// that answers several points of an asker at once and is kept between
    /// evaluations (`neighbours::StaticList`); None for a field that gives
    /// none, which answers by its own `smooth_distance`.
    fn static_parts(&self) -> Option<Vec<Arc<StaticEdges>>> {
        None
    }
}

/// Closer than this to the nearest feature, the direction to it is noise;
/// the boundary's normal is used instead. Far below KiCad's 1 nm resolution.
pub const DEGENERATE_DISTANCE: f64 = 1e-9;
/// Copper this much farther from a chain than both half widths still
/// touches it: 1 nm, KiCad's coordinate resolution.
pub const TOUCH_TOLERANCE: f64 = 1e-6;
/// A segment that touches an inflated chain may measure this far inside
/// it, from round-off in splitting its edges and in the distance; it still
/// touches.
pub const ROUND_OFF: f64 = 1e-9;
/// Edges of an item's index are split to at most this length, so that a
/// point near an edge is also near a piece's middle, where the index finds
/// it.
pub const MAX_EDGE_LENGTH: f64 = 0.5;
/// An item with more edges than this, such as copper text, a logo or a
/// board outline, finds its edges near a point with a tree over them rather
/// than measuring every one; twice a default buffered circle's.
pub const INDEXED_EDGES: usize = 128;

/// What a `StaticEdges` item is measured to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Its own geometry: a centreline or a centre.
    Line = 0,
    /// The boundary of its area, negative inside it: copper, a keepout.
    Area = 1,
    /// The boundary of its area, negative outside it: the board edge, the
    /// region a via's zones connect it in.
    Outside = 2,
}
