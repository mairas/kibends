//! A coupled section of a differential pair as the optimizer moves it: its
//! centreline, and its two tracks as the others see them.
//!
//! A coupled section moves as its centreline. Both tracks are the
//! centreline offset by half the pitch at the section's routed gap all
//! along (`offset_pair`), through their own ends and pins: where an end's
//! own gap differs from it, the track's end edge takes up the difference
//! (`PairOffset::offset`). The other members see the two tracks at the
//! centreline's current shape, so where an obstacle term pulls on a
//! track's edge, the gradient goes back through the offset to the
//! centreline's vertices (`PairOffset::offset_pullback`). A held section's
//! tracks are its own throughout.
//!
//! Python builds each section's block once per run (`kibends.pair_blocks`),
//! its pins and direction pins included, and hands over what the offset
//! needs.

use crate::energy::MAX_SMOOTH_TURN;
use crate::geometry::V2;
use crate::numeric::maximum;

/// The miter's least scale, 1 + cos(MAX_SMOOTH_TURN): a vertex turning
/// farther shifts as much as one turning by MAX_SMOOTH_TURN.
fn miter_floor() -> f64 {
    1.0 + MAX_SMOOTH_TURN.cos()
}

/// The unit normals of each edge of `polyline`, a quarter turn of its
/// direction, and the edges' directions and lengths.
fn normals(polyline: &[V2]) -> (Vec<V2>, Vec<V2>, Vec<f64>) {
    let mut normal = Vec::with_capacity(polyline.len().saturating_sub(1));
    let mut along = Vec::with_capacity(normal.capacity());
    let mut lengths = Vec::with_capacity(normal.capacity());
    for w in polyline.windows(2) {
        let delta = w[1] - w[0];
        let length = delta.x.hypot(delta.y);
        let direction = V2::new(delta.x / length, delta.y / length);
        normal.push(V2::new(-direction.y, direction.x));
        along.push(direction);
        lengths.push(length);
    }
    (normal, along, lengths)
}

/// The normals before and after each vertex: an end's own edge's twice.
fn around(normal: &[V2], vertex: usize) -> (V2, V2) {
    let last = normal.len() - 1;
    (
        normal[vertex.saturating_sub(1).min(last)],
        normal[vertex.min(last)],
    )
}

/// Per vertex, the shift that moves both adjacent segments one unit to the
/// right of travel (y down): the bisector, scaled by 1/cos(turn/2), at most
/// as much as at MAX_SMOOTH_TURN.
fn miters(polyline: &[V2]) -> Vec<V2> {
    let (normal, _, _) = normals(polyline);
    if normal.is_empty() {
        return Vec::new();
    }
    (0..polyline.len())
        .map(|v| {
            let (before, after) = around(&normal, v);
            let scale = maximum(
                1.0 + (before.x * after.x + before.y * after.y),
                miter_floor(),
            );
            V2::new((before.x + after.x) / scale, (before.y + after.y) / scale)
        })
        .collect()
}

/// The two tracks of a pair at ±(gap + width)/2 from `centreline`.
///
/// The first track lies to the right of the direction of travel as seen
/// with y down, the second to the left. Vertices are mitred, so every
/// offset segment is parallel to its centreline segment at exactly half
/// the pitch. The centreline must have no repeated points, and neither
/// track folds while `pairs::check_bend_radius` passes.
pub fn offset_pair(centreline: &[V2], gap: f64, width: f64) -> (Vec<V2>, Vec<V2>) {
    let half_pitch = (gap + width) / 2.0;
    miters(centreline)
        .into_iter()
        .zip(centreline)
        .map(|(m, &c)| {
            let shift = V2::new(half_pitch * m.x, half_pitch * m.y);
            (c + shift, c - shift)
        })
        .unzip()
}

/// The gradient of a function of `offset_pair`'s two tracks, `right` and
/// `left` at each of their vertices, taken back to the centreline's
/// vertices.
pub fn offset_pair_pullback(
    centreline: &[V2],
    gap: f64,
    width: f64,
    right: &[V2],
    left: &[V2],
) -> Vec<V2> {
    let n = centreline.len();
    let mut gradient: Vec<V2> = right.iter().zip(left).map(|(&r, &l)| r + l).collect();
    if n < 2 {
        return gradient;
    }
    let half_pitch = (gap + width) / 2.0;
    let floor = miter_floor();
    let (normal, along, lengths) = normals(centreline);
    let mut at_before = Vec::with_capacity(n);
    let mut at_after = Vec::with_capacity(n);
    for v in 0..n {
        let (before, after) = around(&normal, v);
        let total = before + after;
        let one_plus_cos = 1.0 + (before.x * after.x + before.y * after.y);
        let scale = maximum(one_plus_cos, floor);
        let difference = right[v] - left[v];
        let at_miter = V2::new(half_pitch * difference.x, half_pitch * difference.y);
        // The miter (b + a) / max(1 + b·a, MITER_FLOOR) of the normals b
        // before and a after a vertex, differentiated in each.
        let weight = if one_plus_cos > floor {
            (total.x * at_miter.x + total.y * at_miter.y) / (scale * scale)
        } else {
            0.0
        };
        let scaled = V2::new(at_miter.x / scale, at_miter.y / scale);
        at_before.push(scaled - V2::new(after.x * weight, after.y * weight));
        at_after.push(scaled - V2::new(before.x * weight, before.y * weight));
    }
    let edges = n - 1;
    let mut at_normal = vec![V2::default(); edges];
    at_normal[0] = at_normal[0] + at_before[0];
    for e in 0..edges {
        at_normal[e] = at_normal[e] + at_before[e + 1];
    }
    for e in 0..edges {
        at_normal[e] = at_normal[e] + at_after[e];
    }
    at_normal[edges - 1] = at_normal[edges - 1] + at_after[n - 1];
    // normal = R·along with R a quarter turn; along = delta / |delta|.
    let at_delta: Vec<V2> = (0..edges)
        .map(|e| {
            let a = at_normal[e];
            let turned = V2::new(a.y, -a.x);
            let across = normal[e].x * a.x + normal[e].y * a.y;
            let u = along[e];
            V2::new(
                (turned.x - u.x * across) / lengths[e],
                (turned.y - u.y * across) / lengths[e],
            )
        })
        .collect();
    for e in 0..edges {
        gradient[e + 1] = gradient[e + 1] + at_delta[e];
    }
    for e in 0..edges {
        gradient[e] = gradient[e] - at_delta[e];
    }
    gradient
}

/// The points a section's centreline holds of its tracks, its ends and the
/// pins between them: each where the centreline keeps a vertex, `*_at`,
/// and the track's own point there.
#[derive(Clone, Debug, PartialEq)]
pub struct Pins {
    pub a_at: Vec<V2>,
    pub a: Vec<V2>,
    pub b_at: Vec<V2>,
    pub b: Vec<V2>,
}

/// Which of `points` are each of `at`, exactly: fixed vertices stay where
/// they started.
fn held_at(points: &[V2], at: &[V2]) -> Vec<bool> {
    points.iter().map(|p| at.contains(p)).collect()
}

/// A section's tracks as the offset of its centreline: its gap and track
/// width, which side track a lies on, its pins, and its tracks as given,
/// which a held section keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct PairOffset {
    pub gap: f64,
    pub width: f64,
    pub a_on_right: bool,
    pub held: bool,
    pub pins: Pins,
    pub a: Vec<V2>,
    pub b: Vec<V2>,
}

impl PairOffset {
    /// Tracks a and b along the centreline at `points`, through the
    /// section's own ends and pins.
    pub fn offset(&self, points: &[V2]) -> (Vec<V2>, Vec<V2>) {
        if self.held {
            return (self.a.clone(), self.b.clone());
        }
        let (right, left) = offset_pair(points, self.gap, self.width);
        let (mut a, mut b) = if self.a_on_right {
            (right, left)
        } else {
            (left, right)
        };
        // The offsets miss the points the centreline holds by round-off.
        for (at, own, track) in [
            (&self.pins.a_at, &self.pins.a, &mut a),
            (&self.pins.b_at, &self.pins.b, &mut b),
        ] {
            for (v, p) in points.iter().enumerate() {
                for (pin, held) in at.iter().enumerate() {
                    if p == held {
                        track[v] = own[pin];
                    }
                }
            }
        }
        (a, b)
    }

    /// The gradient of a function of `offset(points)`, `at_a` and `at_b`
    /// at the vertices of tracks a and b, taken back to the centreline's
    /// vertices. The tracks' ends and pins are the section's own, and a
    /// held section's tracks are its own throughout, so neither depends on
    /// the centreline.
    pub fn offset_pullback(&self, points: &[V2], at_a: &[V2], at_b: &[V2]) -> Vec<V2> {
        if self.held {
            return vec![V2::default(); points.len()];
        }
        let unpinned = |at: &[V2], pins: &[V2]| -> Vec<V2> {
            held_at(points, pins)
                .into_iter()
                .zip(at)
                .map(|(held, &g)| if held { V2::default() } else { g })
                .collect()
        };
        let (at_a, at_b) = (
            unpinned(at_a, &self.pins.a_at),
            unpinned(at_b, &self.pins.b_at),
        );
        let (right, left) = if self.a_on_right {
            (at_a, at_b)
        } else {
            (at_b, at_a)
        };
        offset_pair_pullback(points, self.gap, self.width, &right, &left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner() -> Vec<V2> {
        vec![
            V2::new(0.0, 0.0),
            V2::new(1.0, 0.0),
            V2::new(2.0, 0.5),
            V2::new(2.5, 1.5),
        ]
    }

    #[test]
    fn the_tracks_are_parallel_at_half_the_pitch() {
        let line = corner();
        let (right, left) = offset_pair(&line, 0.2, 0.2);
        for e in 0..line.len() - 1 {
            let u = line[e + 1] - line[e];
            let unit = V2::new(u.x / u.norm(), u.y / u.norm());
            for (track, side) in [(&right, 1.0), (&left, -1.0)] {
                let offset = track[e] - line[e];
                assert!((unit.cross(offset) - side * 0.2).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_pullback_matches_finite_differences() {
        let line = corner();
        let (gap, width) = (0.2, 0.15);
        // A linear function of the tracks: its gradient is these weights.
        let weights_right: Vec<V2> = (0..4).map(|v| V2::new(0.3 * v as f64, -0.2)).collect();
        let weights_left: Vec<V2> = (0..4).map(|v| V2::new(0.1, 0.4 - 0.1 * v as f64)).collect();
        let function = |points: &[V2]| -> f64 {
            let (right, left) = offset_pair(points, gap, width);
            right
                .iter()
                .zip(&weights_right)
                .chain(left.iter().zip(&weights_left))
                .map(|(p, w)| p.dot(*w))
                .sum()
        };
        let gradient = offset_pair_pullback(&line, gap, width, &weights_right, &weights_left);
        let step = 1e-6;
        for v in 0..line.len() {
            for axis in 0..2 {
                let mut up = line.clone();
                let mut down = line.clone();
                let delta = if axis == 0 {
                    V2::new(step, 0.0)
                } else {
                    V2::new(0.0, step)
                };
                up[v] = up[v] + delta;
                down[v] = down[v] - delta;
                let numeric = (function(&up) - function(&down)) / (2.0 * step);
                let got = if axis == 0 {
                    gradient[v].x
                } else {
                    gradient[v].y
                };
                assert!((got - numeric).abs() < 1e-6, "{v} {axis}: {got} {numeric}");
            }
        }
    }

    #[test]
    fn a_pinned_vertex_keeps_the_track_s_own_point() {
        let line = corner();
        let own = V2::new(0.0, 0.123);
        let block = PairOffset {
            gap: 0.2,
            width: 0.2,
            a_on_right: true,
            held: false,
            pins: Pins {
                a_at: vec![line[0]],
                a: vec![own],
                b_at: vec![],
                b: vec![],
            },
            a: vec![],
            b: vec![],
        };
        let (a, _) = block.offset(&line);
        assert_eq!(a[0], own);
        let pulled = block.offset_pullback(&line, &[V2::new(1.0, 1.0); 4], &[V2::default(); 4]);
        let free = offset_pair_pullback(
            &line,
            0.2,
            0.2,
            &[
                V2::default(),
                V2::new(1.0, 1.0),
                V2::new(1.0, 1.0),
                V2::new(1.0, 1.0),
            ],
            &[V2::default(); 4],
        );
        assert_eq!(pulled, free);
    }
}
