//! Obstacle-aware merging of a polyline's segments.
//!
//! A chord around a convex obstacle always deviates toward it, so merging
//! by chord tolerance alone would cut into the clearance at every wrap;
//! each merged segment must also pass the exact edge check of the field
//! (`Field::segments_clear`). Merging runs from the polyline's first
//! vertex: each chord grows while every vertex it merges lies within its
//! own tolerance of it and the chord stays clear, and stops at a fixed
//! vertex (`kept_vertices`).

use crate::field::Field;
use crate::geometry::V2;
use crate::numeric::clip;

/// The vertices of `polyline` that merging keeps, in order, the first and
/// the last always among them: no vertex in `fixed` is merged, and each
/// merged vertex lies within its `tolerances` of the chord that replaces
/// it, which is clear in `field`.
pub fn kept_vertices(
    polyline: &[V2],
    fixed: &[bool],
    field: &dyn Field,
    tolerances: &[f64],
) -> Vec<usize> {
    let mut kept = vec![0];
    let Some(last) = polyline.len().checked_sub(1) else {
        return kept;
    };
    let mut start = 0;
    while start < last {
        let mut end = start + 1;
        while end < last && !fixed[end] && mergeable(polyline, start, end + 1, field, tolerances) {
            end += 1;
        }
        kept.push(end);
        start = end;
    }
    kept
}

/// Whether the chord from vertex `start` to vertex `end` may replace the
/// vertices between them.
fn mergeable(
    polyline: &[V2],
    start: usize,
    end: usize,
    field: &dyn Field,
    tolerances: &[f64],
) -> bool {
    let a = polyline[start];
    let b = polyline[end];
    let chord = b - a;
    let squared = chord.dot(chord).max(f64::MIN_POSITIVE);
    let near = polyline[start + 1..end]
        .iter()
        .zip(&tolerances[start + 1..end])
        .all(|(&p, &tolerance)| {
            let t = clip((p - a).dot(chord) / squared, 0.0, 1.0);
            let off = p - (a + t * chord);
            // A NaN deviation is no deviation beyond the tolerance.
            off.dot(off).sqrt().partial_cmp(&tolerance) != Some(std::cmp::Ordering::Greater)
        });
    near && field.segments_clear(&[a], &[b], 0.0)[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Disc, FreeSpace};

    #[test]
    fn a_straight_route_merges_into_one_segment_and_fixed_vertices_stay() {
        let line: Vec<V2> = (0..6).map(|k| V2::new(f64::from(k), 0.0)).collect();
        let tolerances = vec![0.01; line.len()];
        let free = vec![false; line.len()];
        assert_eq!(
            kept_vertices(&line, &free, &FreeSpace, &tolerances),
            vec![0, 5]
        );
        let mut fixed = free.clone();
        fixed[2] = true;
        assert_eq!(
            kept_vertices(&line, &fixed, &FreeSpace, &tolerances),
            vec![0, 2, 5]
        );
    }

    #[test]
    fn a_vertex_whose_merge_would_break_clearance_is_kept() {
        // A route around a disc: its chord deviates by less than the
        // tolerance but cuts into the disc.
        let route = [V2::new(-2.0, 1.05), V2::new(0.0, 1.1), V2::new(2.0, 1.05)];
        let tolerances = [1.0; 3];
        let free = [false; 3];
        let disc = Disc {
            cx: 0.0,
            cy: 0.0,
            radius: 1.06,
        };
        assert_eq!(
            kept_vertices(&route, &free, &disc, &tolerances),
            vec![0, 1, 2]
        );
        assert_eq!(
            kept_vertices(&route, &free, &FreeSpace, &tolerances),
            vec![0, 2]
        );
    }
}
