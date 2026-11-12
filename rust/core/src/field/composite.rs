//! The union of several fields.

use std::sync::Arc;

use super::Field;
use super::smooth::combine_fields;
use crate::geometry::V2;

/// The union of several fields: the nearest of them answers each point,
/// and the smooth distance is over all their obstacles.
///
/// A point beyond `reach` of every field is +inf with a zero gradient. A
/// segment is clear only if it is clear in every field.
pub struct CompositeField {
    fields: Vec<Arc<dyn Field>>,
}

impl CompositeField {
    pub fn new(fields: Vec<Arc<dyn Field>>) -> Self {
        Self { fields }
    }

    pub fn fields(&self) -> &[Arc<dyn Field>] {
        &self.fields
    }
}

impl Field for CompositeField {
    fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let Some((first, rest)) = self.fields.split_first() else {
            return (
                vec![f64::INFINITY; points.len()],
                vec![V2::default(); points.len()],
            );
        };
        let (mut distance, mut gradient) = first.signed_distance(points, reach);
        for field in rest {
            let (d, g) = field.signed_distance(points, reach);
            for i in 0..points.len() {
                if d[i] < distance[i] {
                    distance[i] = d[i];
                    gradient[i] = g[i];
                }
            }
        }
        (distance, gradient)
    }

    /// The smooth minimum of the fields' smooth distances
    /// (`smooth::combine_fields`). It is the one over all their obstacles
    /// but for those SMOOTH_REACH leaves out within each field: up to
    /// τ·e⁻¹⁰ per obstacle apart from the one a single search over every
    /// field's edges gives.
    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let mut gradient = vec![V2::default(); points.len()];
        if self.fields.is_empty() {
            return (vec![f64::INFINITY; points.len()], gradient);
        }
        let answers: Vec<(Vec<f64>, Vec<V2>)> = self
            .fields
            .iter()
            .map(|f| f.smooth_distance(points, reach))
            .collect();
        let distances: Vec<&[f64]> = answers.iter().map(|(d, _)| d.as_slice()).collect();
        let (smooth, shares) = combine_fields(&distances);
        for (share, (_, g)) in shares.iter().zip(&answers) {
            for ((sum, &s), &d) in gradient.iter_mut().zip(share).zip(g) {
                *sum = *sum + V2::new(s * d.x, s * d.y);
            }
        }
        (smooth, gradient)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        let mut clear = vec![true; starts.len()];
        for field in &self.fields {
            for (c, ok) in clear
                .iter_mut()
                .zip(field.segments_clear(starts, ends, tolerance))
            {
                *c &= ok;
            }
        }
        clear
    }
}
