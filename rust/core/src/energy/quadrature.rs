//! The integral of an integrand of the distance along a polyline, by the
//! trapezoid rule on each edge, and the stiffness it gives each vertex.

use super::MIN_EDGE_LENGTH;
use super::integrands::{Curvature, Integrand};
use super::polyline::{Edges, scatter_edge_gradient};
use crate::field::smooth::SMOOTH_WIDTH;
use crate::geometry::V2;
use crate::numeric::{exact_sum, fused_dot, maximum};

/// Where along an edge each of its `points_per_edge` interior quadrature
/// points lies, from 0 to 1.
fn interior_t(points_per_edge: usize) -> impl Iterator<Item = f64> {
    let spans = (points_per_edge + 1) as f64;
    (1..=points_per_edge).map(move |i| i as f64 / spans)
}

/// The points at which `edge_integral` takes distances: the vertices, then
/// each edge's interior points in order.
pub fn quadrature_points(points: &[V2], points_per_edge: usize) -> Vec<V2> {
    let mut at = points.to_vec();
    for w in points.windows(2) {
        for t in interior_t(points_per_edge) {
            let s = 1.0 - t;
            at.push(V2::new(w[0].x * s + w[1].x * t, w[0].y * s + w[1].y * t));
        }
    }
    at
}

/// The quadrature points of each edge, its ends included, as
/// `edge_integral` weights them, from the distances at
/// `quadrature_points` of a polyline of `n` vertices and their gradients.
struct EdgeSamples<'a> {
    n: usize,
    points_per_edge: usize,
    distance: &'a [f64],
    direction: &'a [V2],
    /// Where along its edge each point lies, from 0 to 1.
    t: Vec<f64>,
    /// The trapezoid rule's weight of each point, summing to 1.
    weights: Vec<f64>,
}

impl<'a> EdgeSamples<'a> {
    fn new(n: usize, distance: &'a [f64], direction: &'a [V2], points_per_edge: usize) -> Self {
        let mut t = vec![0.0];
        t.extend(interior_t(points_per_edge));
        t.push(1.0);
        let mut weights = vec![1.0 / (points_per_edge + 1) as f64; points_per_edge + 2];
        weights[0] /= 2.0;
        weights[points_per_edge + 1] /= 2.0;
        Self {
            n,
            points_per_edge,
            distance,
            direction,
            t,
            weights,
        }
    }

    /// The index in `distance` of edge `e`'s point `k`, its start at 0.
    fn index(&self, e: usize, k: usize) -> usize {
        if k == 0 {
            e
        } else if k == self.points_per_edge + 1 {
            e + 1
        } else {
            self.n + e * self.points_per_edge + k - 1
        }
    }

    fn per_edge(&self) -> usize {
        self.points_per_edge + 2
    }
}

/// Σ w[k]·v[k] over an edge's samples, summed in order: Rust neither
/// reassociates nor fuses these operations, so the mean is the same on
/// every platform.
fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(weights)
        .fold(0.0, |sum, (&v, &w)| sum + v * w)
}

/// The diagonal second derivative of `integrand` at a point by x and by y,
/// from its smooth distance and gradient, as the terms of it that cannot be
/// negative: f''(d)·g², with g the gradient, and |f'(d)|·(1 - |g|²)/τ on
/// both axes, the smooth minimum's own curvature where obstacles in
/// different directions are about as near, the spread of their unit
/// gradients measured by the shortfall of |g| from 1. It leaves out each
/// obstacle's own curvature.
pub fn point_stiffness(
    distance: f64,
    direction: V2,
    integrand: &Integrand,
    curvature: &Curvature,
) -> V2 {
    let (_, slope) = integrand.at(distance);
    let squared = V2::new(direction.x * direction.x, direction.y * direction.y);
    let spread = maximum(1.0 - (squared.x + squared.y), 0.0);
    let smoothing = slope.abs() * spread / SMOOTH_WIDTH;
    let c = curvature.at(distance);
    V2::new(c * squared.x + smoothing, c * squared.y + smoothing)
}

/// ∫ f(d(s)) ds along the polyline by the trapezoid rule on each edge,
/// from the distances and their unit gradients at `quadrature_points`.
///
/// Each quadrature point's gradient goes to the edge's endpoints by its
/// linear interpolation weights; the edge-length factor contributes f's
/// mean times the edge direction. Returns the value, its gradient at every
/// vertex, and its derivative by the distance at each quadrature point,
/// for a caller whose distances also depend on other coordinates.
pub fn edge_integral(
    points: &[V2],
    distance: &[f64],
    direction: &[V2],
    integrand: &Integrand,
    points_per_edge: usize,
) -> (f64, Vec<V2>, Vec<f64>) {
    let n = points.len();
    let samples = EdgeSamples::new(n, distance, direction, points_per_edge);
    let Edges { lengths, units, .. } = Edges::of(points);
    let per_edge = samples.per_edge();
    let mut slopes = vec![0.0; distance.len()];
    let mut start_gradient = Vec::with_capacity(lengths.len());
    let mut end_gradient = Vec::with_capacity(lengths.len());
    let mut by_length = Vec::with_capacity(lengths.len());
    let mut values = vec![0.0; per_edge];
    let mut point_gradient = vec![V2::default(); per_edge];
    for (e, (&length, &unit)) in lengths.iter().zip(&units).enumerate() {
        for k in 0..per_edge {
            let at = samples.index(e, k);
            let (v, slope) = integrand.at(samples.distance[at]);
            values[k] = v;
            let weighted_slope = samples.weights[k] * slope;
            let g = samples.direction[at];
            point_gradient[k] = V2::new(weighted_slope * g.x, weighted_slope * g.y);
            let by_edge = length * weighted_slope;
            if k == 0 || k == per_edge - 1 {
                slopes[at] += by_edge;
            } else {
                slopes[at] = by_edge;
            }
        }
        let edge_mean = weighted_mean(&values, &samples.weights);
        by_length.push(length * edge_mean);
        let toward_start = samples.t.iter().map(|&t| 1.0 - t);
        let start = V2::new(
            length * fused_dot(toward_start.clone(), point_gradient.iter().map(|g| g.x)),
            length * fused_dot(toward_start, point_gradient.iter().map(|g| g.y)),
        );
        let end = V2::new(
            length
                * fused_dot(
                    samples.t.iter().copied(),
                    point_gradient.iter().map(|g| g.x),
                ),
            length
                * fused_dot(
                    samples.t.iter().copied(),
                    point_gradient.iter().map(|g| g.y),
                ),
        );
        let measure = edge_mean * unit;
        start_gradient.push(start - measure);
        end_gradient.push(end + measure);
    }
    let gradient = scatter_edge_gradient(n, &start_gradient, &end_gradient);
    (exact_sum(by_length), gradient, slopes)
}

/// Each vertex's stiffness under `edge_integral` of `integrand`: the
/// larger over x and y of its diagonal second derivative, from the
/// distances and their gradients at `quadrature_points`.
///
/// Each quadrature point adds its `point_stiffness` times lₑ·wₖ and the
/// square of the vertex's interpolation weight, and each edge its mean f̄
/// times the curvature of its length, f̄·(1 - u²)/lₑ across its direction
/// u. It leaves out the product of the length's and the distance's slopes,
/// which can be negative, so that it is exact for edges parallel to flat
/// obstacles.
pub fn edge_curvature(
    points: &[V2],
    distance: &[f64],
    direction: &[V2],
    integrand: &Integrand,
    curvature: &Curvature,
    points_per_edge: usize,
) -> Vec<f64> {
    let n = points.len();
    let samples = EdgeSamples::new(n, distance, direction, points_per_edge);
    let Edges { lengths, units, .. } = Edges::of(points);
    let per_edge = samples.per_edge();
    let start_square: Vec<f64> = samples.t.iter().map(|&t| (1.0 - t) * (1.0 - t)).collect();
    let end_square: Vec<f64> = samples.t.iter().map(|&t| t * t).collect();
    let mut starts = Vec::with_capacity(lengths.len());
    let mut ends = Vec::with_capacity(lengths.len());
    let mut values = vec![0.0; per_edge];
    let mut per_point = vec![V2::default(); per_edge];
    for (e, (&length, &unit)) in lengths.iter().zip(&units).enumerate() {
        for k in 0..per_edge {
            let at = samples.index(e, k);
            let d = samples.distance[at];
            values[k] = integrand.at(d).0;
            let measure = length * samples.weights[k];
            let stiffness = point_stiffness(d, samples.direction[at], integrand, curvature);
            per_point[k] = V2::new(measure * stiffness.x, measure * stiffness.y);
        }
        let edge_mean = weighted_mean(&values, &samples.weights);
        let scale = edge_mean / maximum(length, MIN_EDGE_LENGTH);
        let across = V2::new(
            scale * (1.0 - unit.x * unit.x),
            scale * (1.0 - unit.y * unit.y),
        );
        let start = V2::new(
            fused_dot(start_square.iter().copied(), per_point.iter().map(|p| p.x)),
            fused_dot(start_square.iter().copied(), per_point.iter().map(|p| p.y)),
        );
        let end = V2::new(
            fused_dot(end_square.iter().copied(), per_point.iter().map(|p| p.x)),
            fused_dot(end_square.iter().copied(), per_point.iter().map(|p| p.y)),
        );
        starts.push(start + across);
        ends.push(end + across);
    }
    scatter_edge_gradient(n, &starts, &ends)
        .into_iter()
        .map(|g| maximum(g.x, g.y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::EnergyParams;
    use crate::energy::tests::{assert_gradient_matches, params, wiggly};
    use crate::field::smooth::combine_fields;

    /// Walls along y = ±`half_gap`: the smooth distance over the two and
    /// its gradient at each point.
    fn walls(half_gap: f64, at: &[V2]) -> (Vec<f64>, Vec<V2>) {
        let below: Vec<f64> = at.iter().map(|p| half_gap + p.y).collect();
        let above: Vec<f64> = at.iter().map(|p| half_gap - p.y).collect();
        let (smooth, shares) = combine_fields(&[&below, &above]);
        let gradient = shares[0]
            .iter()
            .zip(&shares[1])
            .map(|(b, a)| V2::new(0.0, b - a))
            .collect();
        (smooth, gradient)
    }

    /// A disc of radius 1 at the origin.
    fn disc(at: &[V2]) -> (Vec<f64>, Vec<V2>) {
        at.iter()
            .map(|p| {
                let r = p.norm();
                (r - 1.0, V2::new(p.x / r, p.y / r))
            })
            .unzip()
    }

    fn obstacle_energy(points: &[V2], p: &EnergyParams) -> (f64, Vec<V2>) {
        let at = quadrature_points(points, p.points_per_edge);
        let (distance, direction) = disc(&at);
        let (value, gradient, _) = edge_integral(
            points,
            &distance,
            &direction,
            &Integrand::obstacle(p),
            p.points_per_edge,
        );
        (value, gradient)
    }

    #[test]
    fn quadrature_points_are_the_vertices_then_each_edge_s_interior() {
        let at = quadrature_points(&[V2::new(0.0, 0.0), V2::new(4.0, 0.0)], 3);
        let x: Vec<f64> = at.iter().map(|p| p.x).collect();
        assert_eq!(x, vec![0.0, 4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_obstacle_integral_s_gradient_matches_finite_differences() {
        for seed in 0..3 {
            let points = wiggly(seed);
            assert_gradient_matches(|q, _| obstacle_energy(q, &params()), &points);
        }
    }

    #[test]
    fn an_edge_at_constant_distance_integrates_the_falloff() {
        let p = EnergyParams {
            repulsion: 1.0,
            clearance_weight: 0.0,
            ..params()
        };
        let points = [V2::new(-1.0, 0.0), V2::new(1.0, 0.0)];
        let at = quadrature_points(&points, p.points_per_edge);
        let distance = vec![p.falloff_range / 2.0; at.len()];
        let direction = vec![V2::new(0.0, 1.0); at.len()];
        let (value, _, _) = edge_integral(
            &points,
            &distance,
            &direction,
            &Integrand::obstacle(&p),
            p.points_per_edge,
        );
        assert!((value - 2.0 * 0.5f64.powi(3)).abs() < 1e-12);
    }

    #[test]
    fn the_slopes_are_the_value_s_derivatives_by_each_distance() {
        let p = params();
        let points = wiggly(0);
        let at = quadrature_points(&points, p.points_per_edge);
        let (distance, direction) = disc(&at);
        let integrand = Integrand::obstacle(&p);
        let (_, _, slopes) = edge_integral(&points, &distance, &direction, &integrand, 3);
        for k in [0, 3, points.len() - 1, points.len() + 5] {
            let shifted = |by: f64| {
                let mut d = distance.clone();
                d[k] += by;
                edge_integral(&points, &d, &direction, &integrand, 3).0
            };
            let numeric = (shifted(1e-6) - shifted(-1e-6)) / 2e-6;
            assert!((numeric - slopes[k]).abs() <= 1e-5 * slopes[k].abs().max(1.0));
        }
    }

    /// Between flat walls each wall's distance is linear and moves across
    /// them do not change edge lengths to first order, so the diagonal
    /// curvature across them is exactly the terms `edge_curvature` keeps,
    /// the smooth minimum's between the walls among them.
    #[test]
    fn the_edge_curvature_is_the_diagonal_curvature_between_flat_walls() {
        const STEP: f64 = 1e-5;
        let p = EnergyParams {
            repulsion: 0.7,
            ..params()
        };
        let integrand = Integrand::obstacle(&p);
        let gradient_y = |points: &[V2], half_gap: f64, i: usize| {
            let at = quadrature_points(points, 3);
            let (distance, direction) = walls(half_gap, &at);
            edge_integral(points, &distance, &direction, &integrand, 3).1[i].y
        };
        for (half_gap, y) in [
            (5.0, -4.9),
            (5.0, -4.3),
            (5.0, -3.0),
            (0.1, 0.0),
            (0.1, 0.004),
        ] {
            let points: Vec<V2> = [0.0, 0.3, 0.5, 0.9, 1.0]
                .into_iter()
                .map(|x| V2::new(x, y))
                .collect();
            let at = quadrature_points(&points, 3);
            let (distance, direction) = walls(half_gap, &at);
            let stiffness = edge_curvature(
                &points,
                &distance,
                &direction,
                &integrand,
                &Curvature::obstacle(&p),
                3,
            );
            for i in 0..points.len() {
                let shifted = |by: f64| {
                    let mut moved = points.clone();
                    moved[i].y += by;
                    gradient_y(&moved, half_gap, i)
                };
                let numeric = (shifted(STEP) - shifted(-STEP)) / (2.0 * STEP);
                assert!(
                    (numeric - stiffness[i]).abs() <= 1e-3 * numeric.abs() + 1e-6,
                    "gap {half_gap} at {y}, vertex {i}: {} against {numeric}",
                    stiffness[i]
                );
            }
        }
    }
}
