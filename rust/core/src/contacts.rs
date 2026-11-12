//! The terms that keep a chain on its contacts.
//!
//! A vertex that keeps a pad or a via that stays is free, held inside the
//! region where it keeps contact by a stop, the clearance hinge at its
//! distance to the region's edge, which limits without repelling
//! (`contact_stops`). After a burst, the optimizer moves back a vertex that
//! newly ends outside (`contacts_left`). Each vertex's contact is an index
//! into the chain's regions, None for a vertex that keeps none.

use crate::energy::TermValue;
use crate::energy::integrands::Integrand;
use crate::field::Field;
use crate::geometry::V2;

/// The distance from `p` to the edge of `region`, negative outside it.
fn depth(region: &dyn Field, p: V2) -> (f64, V2) {
    let (distance, direction) = region.signed_distance(&[p], f64::INFINITY);
    (distance[0], direction[0])
}

/// The stops that keep each vertex of `points` that keeps a contact inside
/// its region, and their gradient at every vertex: `limit`, the clearance
/// hinge, at its distance to the edge of `within[contact_of[v]]`, weighted
/// by the chain's `width`, as a via's ring stands for a track as long as it
/// is wide. They limit without repelling: the hinge rises only within its
/// standoff of the edge.
pub fn contact_stops(
    points: &[V2],
    contact_of: &[Option<usize>],
    within: &[&dyn Field],
    width: f64,
    limit: &Integrand,
) -> TermValue {
    let mut gradient = vec![V2::default(); points.len()];
    let mut value = 0.0;
    for (v, contact) in contact_of.iter().enumerate() {
        let Some(k) = *contact else { continue };
        let (distance, direction) = depth(within[k], points[v]);
        let (term, slope) = limit.at(distance);
        value += width * term;
        gradient[v] = gradient[v] + (width * slope) * direction;
    }
    (value, gradient)
}

/// The vertices of `points` that keep a contact and lie outside its
/// region by more than `tolerance`.
pub fn contacts_left(
    points: &[V2],
    contact_of: &[Option<usize>],
    within: &[&dyn Field],
    tolerance: f64,
) -> Vec<bool> {
    points
        .iter()
        .zip(contact_of)
        .map(|(&p, contact)| contact.is_some_and(|k| depth(within[k], p).0 < -tolerance))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::tests::params;
    use crate::testing::Disc;

    #[test]
    fn a_vertex_near_its_region_s_edge_is_stopped_and_one_outside_is_left() {
        // The region is the outside of the disc, where its distance is
        // positive.
        let region = Disc {
            cx: 0.0,
            cy: 0.0,
            radius: 1.0,
        };
        let within: [&dyn Field; 1] = [&region];
        let points = [V2::new(0.0, -0.99), V2::new(3.0, 3.0), V2::new(1.5, 0.0)];
        let contact_of = [Some(0), None, Some(0)];
        let limit = Integrand::hinge(&params());
        let (value, gradient) = contact_stops(&points, &contact_of, &within, 0.2, &limit);
        assert!(value > 0.0);
        assert_eq!(gradient[1], V2::default());
        let left = contacts_left(&points, &contact_of, &within, 1e-7);
        assert_eq!(left, vec![true, false, false]);
    }
}
