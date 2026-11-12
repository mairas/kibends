//! What a problem of the run returns of each of its members and vias, as
//! plain data: the binding wraps it into the result types the Python side
//! reports (`kibends.results`).

use crate::geometry::V2;
use crate::optimize::state::State;

/// A member's shape, its best or as given.
#[derive(Clone, Debug, PartialEq)]
pub struct MemberState {
    pub polyline: Vec<V2>,
    /// Its pinned vertices.
    pub mask: Vec<bool>,
    /// Each vertex's contact among the member's, None for none.
    pub contact_of: Vec<Option<usize>>,
    /// Edges whose burst moves were undone because they crossed an
    /// obstacle.
    pub rejected_edges: usize,
}

/// A moving via's centre, its best or as given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViaState {
    pub position: V2,
    /// End edges of its chains whose crossing an obstacle undid its move.
    pub rejected_edges: usize,
}

impl State {
    /// Member k's best state, with `best`, else its input.
    pub fn member(&self, k: usize, best: bool) -> MemberState {
        let shared = self.shared();
        let given = &shared.setup.members[k];
        let (polyline, mask, contact_of) = if best {
            (
                self.best.points[k].clone(),
                self.best.masks[k].clone(),
                self.best.contact_of[k].clone(),
            )
        } else {
            (
                given.original.clone(),
                given.anchors.clone(),
                shared.given_contact_of[k].clone(),
            )
        };
        MemberState {
            polyline,
            mask,
            contact_of,
            rejected_edges: self.rejected_edges[k],
        }
    }

    /// Via k's best centre, with `best`, else its input.
    pub fn via(&self, k: usize, best: bool) -> ViaState {
        ViaState {
            position: if best {
                self.best.positions[k]
            } else {
                self.shared().setup.vias[k].start
            },
            rejected_edges: self.via_rejected[k],
        }
    }
}
