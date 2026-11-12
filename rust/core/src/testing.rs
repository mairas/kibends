//! Analytic fields for tests: free space, a disc, a box, two walls, walls
//! whose gradient misleads, the union of several, another field ignoring
//! the reach, and a field of NaN.
//!
//! Each shape is one obstacle, but `Walls`, two, so their smooth distance
//! is their exact one, and that of `Walls` and `Union` the smooth minimum
//! over theirs. They ignore the reach: every distance is exact.

use std::sync::Arc;

use crate::field::Field;
use crate::field::smooth::combine_fields;
use crate::geometry::V2;
use crate::numeric::{clip, maximum, minimum};

/// The distance `FreeSpace` reports, beyond any falloff range used in
/// tests, mm.
pub const FAR: f64 = 1e3;

/// Nothing anywhere: every point at `FAR`.
pub struct FreeSpace;

impl Field for FreeSpace {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        (vec![FAR; points.len()], vec![V2::default(); points.len()])
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.signed_distance(points, reach)
    }

    fn segments_clear(&self, starts: &[V2], _ends: &[V2], _tolerance: f64) -> Vec<bool> {
        vec![true; starts.len()]
    }
}

/// A disc of `radius` at (`cx`, `cy`).
pub struct Disc {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
}

impl Field for Disc {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        points
            .iter()
            .map(|&p| {
                let offset = p - V2::new(self.cx, self.cy);
                let norm = offset.norm();
                let direction = if norm > 0.0 {
                    V2::new(offset.x / norm, offset.y / norm)
                } else {
                    V2::new(1.0, 0.0)
                };
                (norm - self.radius, direction)
            })
            .unzip()
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.signed_distance(points, reach)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        let centre = V2::new(self.cx, self.cy);
        starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| point_segment_distance(centre, a, b) >= self.radius - tolerance)
            .collect()
    }
}

/// An axis-aligned box given by its centre and half-sizes. A corner radius
/// rounds the corners, as inflating a sharp box by that radius would; the
/// half-sizes include the rounding.
pub struct Box {
    pub cx: f64,
    pub cy: f64,
    pub half_x: f64,
    pub half_y: f64,
    pub corner_radius: f64,
}

impl Box {
    /// The half-sizes of the box the rounding inflates.
    fn core(&self) -> V2 {
        V2::new(
            self.half_x - self.corner_radius,
            self.half_y - self.corner_radius,
        )
    }

    fn outside_core_distance(&self, p: V2) -> f64 {
        let core = self.core();
        let offset = V2::new(
            (p.x - self.cx).abs() - core.x,
            (p.y - self.cy).abs() - core.y,
        );
        V2::new(maximum(offset.x, 0.0), maximum(offset.y, 0.0)).norm()
    }
}

impl Field for Box {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        let core = self.core();
        points
            .iter()
            .map(|&p| {
                let offset = p - V2::new(self.cx, self.cy);
                let sign = |v: f64| if v >= 0.0 { 1.0 } else { -1.0 };
                let s = V2::new(sign(offset.x), sign(offset.y));
                let q = V2::new(offset.x.abs() - core.x, offset.y.abs() - core.y);
                let outside = V2::new(maximum(q.x, 0.0), maximum(q.y, 0.0));
                let outside_norm = outside.norm();
                let (distance, gradient) = if outside_norm > 0.0 {
                    (
                        outside_norm,
                        V2::new(
                            s.x * outside.x / outside_norm,
                            s.y * outside.y / outside_norm,
                        ),
                    )
                } else if q.x >= q.y || q.x.is_nan() {
                    (maximum(q.x, q.y), V2::new(s.x, 0.0))
                } else {
                    (maximum(q.x, q.y), V2::new(0.0, s.y))
                };
                (distance - self.corner_radius, gradient)
            })
            .unzip()
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.signed_distance(points, reach)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        // A sharp box's core, or a rounded box's corner radius, shrinks by
        // the tolerance.
        let sharp = self.corner_radius == 0.0;
        let shrink = if sharp { tolerance } else { 0.0 };
        let core = self.core();
        let core = V2::new(core.x - shrink, core.y - shrink);
        let radius = self.corner_radius - tolerance;
        let low = V2::new(self.cx - core.x, self.cy - core.y);
        let high = V2::new(self.cx + core.x, self.cy + core.y);
        starts
            .iter()
            .zip(ends)
            .map(|(&a, &b)| {
                let (enter, leave) = clip_to_rectangle(a, b, low, high);
                let meets_core = enter <= leave;
                if sharp {
                    let t = if meets_core {
                        (enter + leave) / 2.0
                    } else {
                        0.0
                    };
                    let middle = a + t * (b - a);
                    let strictly_inside = middle.x > low.x
                        && middle.x < high.x
                        && middle.y > low.y
                        && middle.y < high.y;
                    return !(meets_core && strictly_inside);
                }
                let corners = [low, V2::new(low.x, high.y), high, V2::new(high.x, low.y)];
                let corner_distance = corners
                    .iter()
                    .map(|&c| point_segment_distance(c, a, b))
                    .reduce(minimum)
                    .unwrap_or(f64::INFINITY);
                let endpoint_distance =
                    minimum(self.outside_core_distance(a), self.outside_core_distance(b));
                !meets_core && minimum(corner_distance, endpoint_distance) >= radius
            })
            .collect()
    }
}

/// Liang-Barsky: the parameter interval of the segment from `a` to `b`
/// inside the closed box from `low` to `high`; it misses the box where the
/// returned enter exceeds leave.
fn clip_to_rectangle(a: V2, b: V2, low: V2, high: V2) -> (f64, f64) {
    let delta = b - a;
    let mut enter: f64 = 0.0;
    let mut leave: f64 = 1.0;
    for (d, start, low, high) in [(delta.x, a.x, low.x, high.x), (delta.y, a.y, low.y, high.y)] {
        for (p, q) in [(-d, start - low), (d, high - start)] {
            let parallel = p == 0.0;
            let t = q / if parallel { 1.0 } else { p };
            if !parallel && p < 0.0 {
                enter = maximum(enter, t);
            }
            if !parallel && p > 0.0 {
                leave = minimum(leave, t);
            }
            if parallel && q < 0.0 {
                leave = f64::NEG_INFINITY;
            }
        }
    }
    (enter, leave)
}

fn point_segment_distance(p: V2, a: V2, b: V2) -> f64 {
    let edge = b - a;
    let squared = edge.x * edge.x + edge.y * edge.y;
    let relative = p - a;
    let t = (relative.x * edge.x + relative.y * edge.y) / if squared > 0.0 { squared } else { 1.0 };
    let closest = a + clip(t, 0.0, 1.0) * edge;
    (p - closest).norm()
}

/// Two walls parallel to the x axis at y = ±`half_gap`, each an obstacle,
/// with free space between; `misleading`, their gradient points the wrong
/// way.
pub struct Walls {
    pub half_gap: f64,
    pub misleading: bool,
}

impl Walls {
    fn sign(&self) -> f64 {
        if self.misleading { -1.0 } else { 1.0 }
    }
}

impl Field for Walls {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        points
            .iter()
            .map(|p| {
                let toward_centre = if p.y >= 0.0 { -1.0 } else { 1.0 };
                (
                    self.half_gap - p.y.abs(),
                    V2::new(self.sign() * 0.0, self.sign() * toward_centre),
                )
            })
            .unzip()
    }

    /// The smooth minimum over the two walls.
    fn smooth_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        let below: Vec<f64> = points.iter().map(|p| self.half_gap + p.y).collect();
        let above: Vec<f64> = points.iter().map(|p| self.half_gap - p.y).collect();
        let (smooth, shares) = combine_fields(&[&below, &above]);
        let gradient = shares[0]
            .iter()
            .zip(&shares[1])
            .map(|(b, a)| V2::new(self.sign() * 0.0, self.sign() * (b - a)))
            .collect();
        (smooth, gradient)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        starts
            .iter()
            .zip(ends)
            .map(|(a, b)| {
                a.y.abs() <= self.half_gap + tolerance && b.y.abs() <= self.half_gap + tolerance
            })
            .collect()
    }
}

/// The nearest of several fields at each point, exact in free space; the
/// smooth distance over theirs.
pub struct Union {
    pub shapes: Vec<Arc<dyn Field>>,
}

impl Field for Union {
    fn signed_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let mut distance = vec![f64::INFINITY; points.len()];
        let mut gradient = vec![V2::default(); points.len()];
        let mut seen = vec![false; points.len()];
        for shape in &self.shapes {
            let (d, g) = shape.signed_distance(points, reach);
            for i in 0..points.len() {
                // numpy's argmin: the first of the least, the first NaN.
                let take =
                    !seen[i] || (!distance[i].is_nan() && (d[i] < distance[i] || d[i].is_nan()));
                if take {
                    distance[i] = d[i];
                    gradient[i] = g[i];
                    seen[i] = true;
                }
            }
        }
        (distance, gradient)
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        let answers: Vec<(Vec<f64>, Vec<V2>)> = self
            .shapes
            .iter()
            .map(|s| s.smooth_distance(points, reach))
            .collect();
        let distances: Vec<&[f64]> = answers.iter().map(|(d, _)| d.as_slice()).collect();
        let (smooth, shares) = combine_fields(&distances);
        let mut gradient = vec![V2::default(); points.len()];
        for (share, (_, g)) in shares.iter().zip(&answers) {
            for ((sum, &s), &d) in gradient.iter_mut().zip(share).zip(g) {
                *sum = *sum + V2::new(s * d.x, s * d.y);
            }
        }
        (smooth, gradient)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        let mut clear = vec![true; starts.len()];
        for shape in &self.shapes {
            for (c, ok) in clear
                .iter_mut()
                .zip(shape.segments_clear(starts, ends, tolerance))
            {
                *c &= ok;
            }
        }
        clear
    }
}

/// Another field answering every query exactly, whatever the reach, and
/// giving no static edges, so that the energy asks it for each point.
pub struct ReachIgnored {
    pub field: Arc<dyn Field>,
}

impl Field for ReachIgnored {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.field.signed_distance(points, f64::INFINITY)
    }

    fn smooth_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.field.smooth_distance(points, f64::INFINITY)
    }

    fn segments_clear(&self, starts: &[V2], ends: &[V2], tolerance: f64) -> Vec<bool> {
        self.field.segments_clear(starts, ends, tolerance)
    }
}

/// A field that answers NaN everywhere, for the energy's check that it is
/// finite.
pub struct NanField;

impl Field for NanField {
    fn signed_distance(&self, points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
        (
            vec![f64::NAN; points.len()],
            vec![V2::default(); points.len()],
        )
    }

    fn smooth_distance(&self, points: &[V2], reach: f64) -> (Vec<f64>, Vec<V2>) {
        self.signed_distance(points, reach)
    }

    /// NaN is below no tolerance, so every segment is clear.
    fn segments_clear(&self, starts: &[V2], _ends: &[V2], _tolerance: f64) -> Vec<bool> {
        vec![true; starts.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f64 = f64::INFINITY;

    #[test]
    fn a_disc_is_measured_from_its_rim() {
        let disc = Disc {
            cx: 1.0,
            cy: 0.0,
            radius: 0.5,
        };
        let (d, g) = disc.signed_distance(&[V2::new(3.0, 0.0), V2::new(1.0, 0.0)], INF);
        assert_eq!(d, vec![1.5, -0.5]);
        assert_eq!(g, vec![V2::new(1.0, 0.0), V2::new(1.0, 0.0)]);
        assert_eq!(
            disc.segments_clear(
                &[V2::new(0.0, 0.4), V2::new(0.0, 0.6)],
                &[V2::new(2.0, 0.4), V2::new(2.0, 0.6)],
                0.0
            ),
            vec![false, true]
        );
    }

    #[test]
    fn a_rounded_box_is_its_core_inflated() {
        let rounded = Box {
            cx: 0.0,
            cy: 0.0,
            half_x: 1.0,
            half_y: 1.0,
            corner_radius: 0.5,
        };
        let (d, g) = rounded.signed_distance(&[V2::new(2.0, 0.0), V2::new(0.0, -0.2)], INF);
        assert_eq!(d, vec![1.0, -0.8]);
        assert_eq!(g, vec![V2::new(1.0, 0.0), V2::new(0.0, -1.0)]);
        let corner = 0.5 + 0.5 * 2f64.sqrt();
        let near = corner + 0.01;
        assert_eq!(
            rounded.segments_clear(&[V2::new(near, 0.0)], &[V2::new(0.0, near)], 0.0),
            vec![false]
        );
    }

    #[test]
    fn a_sharp_box_lets_segments_along_its_side_pass() {
        let sharp = Box {
            cx: 0.0,
            cy: 0.0,
            half_x: 1.0,
            half_y: 1.0,
            corner_radius: 0.0,
        };
        let clear = sharp.segments_clear(
            &[V2::new(-2.0, 1.0), V2::new(-2.0, 0.0)],
            &[V2::new(2.0, 1.0), V2::new(2.0, 0.0)],
            0.0,
        );
        assert_eq!(clear, vec![true, false]);
    }

    #[test]
    fn walls_share_the_middle_and_mislead_on_request() {
        let walls = Walls {
            half_gap: 1.0,
            misleading: false,
        };
        let (d, g) = walls.smooth_distance(&[V2::new(0.0, 0.0)], INF);
        assert!((d[0] - (1.0 - 0.005 * 2f64.ln())).abs() < 1e-15);
        assert_eq!(g[0].y, 0.0);
        let misleading = Walls {
            half_gap: 1.0,
            misleading: true,
        };
        let at = [V2::new(0.0, 0.5)];
        assert_eq!(
            misleading.signed_distance(&at, INF).1[0].y,
            -walls.signed_distance(&at, INF).1[0].y
        );
    }

    #[test]
    fn a_union_answers_with_its_nearest_shape() {
        let union = Union {
            shapes: vec![
                Arc::new(Disc {
                    cx: 0.0,
                    cy: 0.0,
                    radius: 1.0,
                }),
                Arc::new(Disc {
                    cx: 5.0,
                    cy: 0.0,
                    radius: 1.0,
                }),
            ],
        };
        let (d, g) = union.signed_distance(&[V2::new(4.0, 0.0)], INF);
        assert_eq!(d, vec![0.0]);
        assert_eq!(g, vec![V2::new(-1.0, 0.0)]);
    }

    #[test]
    fn the_nan_field_answers_nan() {
        assert!(NanField.smooth_distance(&[V2::default()], INF).0[0].is_nan());
    }
}
