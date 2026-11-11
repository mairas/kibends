//! Plane geometry the static field measures with: shapes of polygons,
//! lines and points, where a point lies against them, and the distances
//! GEOS takes, by GEOS's own formulas, so that the field's distances are
//! shapely's to round-off.
//!
//! Python builds every shape with shapely and hands it over as plain
//! coordinates (`Shape::from_plain`); nothing here buffers, unions or
//! polygonizes.

use std::ops::{Add, Mul, Neg, Sub};

/// A point or a vector in the board's plane, mm.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}

impl V2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// The length as numpy's `abs` of a complex number takes it.
    pub fn hypot(self) -> f64 {
        self.x.hypot(self.y)
    }

    /// The length as `numpy.linalg.norm` takes it.
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    pub fn array(self) -> [f64; 2] {
        [self.x, self.y]
    }
}

impl Add for V2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub for V2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<V2> for f64 {
    type Output = V2;
    fn mul(self, v: V2) -> V2 {
        V2::new(self * v.x, self * v.y)
    }
}

impl Neg for V2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

/// One part of a shape: a polygon, its exterior ring then its holes, each
/// closed, or a line, a point being a line of one coordinate.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Polygon(Vec<Vec<V2>>),
    Line(Vec<V2>),
}

/// Where a point lies against an area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Interior,
    Boundary,
    Exterior,
}

/// A shapely geometry as plain parts, with shapely's type id, which tells
/// geometries of equal coordinates apart as their WKB does.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub parts: Vec<Part>,
    pub type_id: i32,
}

/// shapely's type id of a LineString.
pub const LINE_STRING: i32 = 1;

/// Geometries as plain arrays, as Python hands them over: coordinates, and
/// where each ring, part and shape ends, each counted in the level below.
pub struct PlainShapes<'a> {
    pub coords: &'a [[f64; 2]],
    pub ring_ends: &'a [usize],
    pub part_ends: &'a [usize],
    pub part_polygon: &'a [bool],
    pub shape_ends: &'a [usize],
    pub type_ids: &'a [i32],
}

impl Shape {
    /// The shapes of `plain`, in order.
    pub fn from_plain(plain: &PlainShapes<'_>) -> Vec<Shape> {
        let ring = |r: usize| -> Vec<V2> {
            let start = if r == 0 { 0 } else { plain.ring_ends[r - 1] };
            plain.coords[start..plain.ring_ends[r]]
                .iter()
                .map(|c| V2::new(c[0], c[1]))
                .collect()
        };
        let mut shapes = Vec::with_capacity(plain.shape_ends.len());
        let mut part = 0;
        for (s, &end) in plain.shape_ends.iter().enumerate() {
            let mut parts = Vec::with_capacity(end.saturating_sub(part));
            while part < end {
                let first = if part == 0 {
                    0
                } else {
                    plain.part_ends[part - 1]
                };
                let rings = first..plain.part_ends[part];
                if plain.part_polygon[part] {
                    parts.push(Part::Polygon(rings.map(ring).collect()));
                } else {
                    parts.push(Part::Line(rings.flat_map(ring).collect()));
                }
                part += 1;
            }
            shapes.push(Shape {
                parts,
                type_id: plain.type_ids[s],
            });
        }
        shapes
    }

    /// A line through `points`, or a point of one.
    pub fn line(points: Vec<V2>) -> Shape {
        Shape {
            parts: vec![Part::Line(points)],
            type_id: LINE_STRING,
        }
    }

    /// A polygon of `rings`, the exterior first, each closed.
    pub fn polygon(rings: Vec<Vec<V2>>) -> Shape {
        Shape {
            parts: vec![Part::Polygon(rings)],
            type_id: 3,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| match p {
            Part::Polygon(rings) => rings.first().is_none_or(Vec::is_empty),
            Part::Line(points) => points.is_empty(),
        })
    }

    /// Whether it has area, as shapely's dimension 2.
    pub fn is_area(&self) -> bool {
        self.parts.iter().any(|p| matches!(p, Part::Polygon(_)))
    }

    /// The lines it is measured to: the rings of its polygons, its
    /// boundary, and its lines, a point a line of one coordinate.
    pub fn targets(&self) -> impl Iterator<Item = &[V2]> {
        self.parts.iter().flat_map(|part| match part {
            Part::Polygon(rings) => rings.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            Part::Line(points) => vec![points.as_slice()],
        })
    }

    /// The edges of its targets in order, each target's consecutive
    /// coordinates, a point an edge of no length.
    pub fn edges(&self) -> Vec<(V2, V2)> {
        let mut edges = Vec::new();
        for target in self.targets() {
            match target {
                [] => {}
                [single] => edges.push((*single, *single)),
                _ => edges.extend(target.windows(2).map(|w| (w[0], w[1]))),
            }
        }
        edges
    }

    /// Its bounding box, min x, min y, max x, max y; None if empty.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let mut bounds: Option<[f64; 4]> = None;
        for target in self.targets() {
            for p in target {
                bounds = Some(match bounds {
                    None => [p.x, p.y, p.x, p.y],
                    Some([a, b, c, d]) => [a.min(p.x), b.min(p.y), c.max(p.x), d.max(p.y)],
                });
            }
        }
        bounds
    }

    /// Where `p` lies against the polygons among its parts.
    pub fn locate(&self, p: V2) -> Location {
        let mut found = Location::Exterior;
        for part in &self.parts {
            if let Part::Polygon(rings) = part {
                match locate_in_polygon(rings, p) {
                    Location::Interior => return Location::Interior,
                    Location::Boundary => found = Location::Boundary,
                    Location::Exterior => {}
                }
            }
        }
        found
    }

    /// Whether `p` lies in the interior of its area, as shapely's
    /// `contains_xy`.
    pub fn contains(&self, p: V2) -> bool {
        self.locate(p) == Location::Interior
    }

    /// Whether `p` lies in its area or on its boundary, as shapely's
    /// `covers` of a point.
    pub fn covers(&self, p: V2) -> bool {
        self.locate(p) != Location::Exterior
    }

    /// The distance from `p` to the shape, zero inside its area, as
    /// shapely's `distance`.
    pub fn distance(&self, p: V2) -> f64 {
        if self.is_area() && self.covers(p) {
            return 0.0;
        }
        nearest_edge(&self.edges(), p).map_or(f64::INFINITY, |(_, d)| d)
    }

    /// The distance from the segment from `a` to `b` to the shape, as
    /// GEOS's `DistanceOp` takes it: zero if the segment starts in its
    /// area, else the least distance to any of its edges.
    pub fn segment_distance(&self, a: V2, b: V2) -> f64 {
        if self.is_area() && self.covers(a) {
            return 0.0;
        }
        self.edges()
            .iter()
            .map(|&(c, d)| segment_to_segment(a, b, c, d))
            .fold(f64::INFINITY, f64::min)
    }

    /// The distance between two shapes, as shapely's `distance`.
    pub fn shape_distance(&self, other: &Shape) -> f64 {
        if self.is_area() && other.first_points().any(|p| self.covers(p))
            || other.is_area() && self.first_points().any(|p| other.covers(p))
        {
            return 0.0;
        }
        let mine = self.edges();
        let theirs = other.edges();
        let mut least = f64::INFINITY;
        for &(a, b) in &mine {
            for &(c, d) in &theirs {
                least = least.min(segment_to_segment(a, b, c, d));
            }
        }
        least
    }

    /// The first coordinate of each part, which GEOS tests for containment
    /// before it measures edges.
    fn first_points(&self) -> impl Iterator<Item = V2> + '_ {
        self.parts.iter().filter_map(|part| match part {
            Part::Polygon(rings) => rings.first().and_then(|r| r.first()).copied(),
            Part::Line(points) => points.first().copied(),
        })
    }

    /// Whether the segment from `a` to `b` has no point outside the area,
    /// as shapely's `covered_by`.
    /// `edges` are the shape's edges the segment may touch, all of them
    /// or a superset of those.
    pub fn covers_segment(&self, a: V2, b: V2, edges: &[(V2, V2)]) -> bool {
        if !self.covers(a) || !self.covers(b) {
            return false;
        }
        if a == b {
            return true;
        }
        // Between where the segment meets the boundary it lies wholly on
        // one side of it, so its middle there tells the side.
        let along = b - a;
        let squared = along.dot(along);
        let mut cuts = vec![0.0, 1.0];
        for &(c, d) in edges {
            let oc = orientation(a, b, c);
            let od = orientation(a, b, d);
            if oc == 0 && od == 0 {
                cuts.extend([(c - a).dot(along) / squared, (d - a).dot(along) / squared]);
            } else if oc != od && orientation(c, d, a) != orientation(c, d, b) {
                let cd = d - c;
                let denominator = along.cross(cd);
                if denominator != 0.0 {
                    cuts.push((c - a).cross(cd) / denominator);
                }
            }
        }
        let mut cuts: Vec<f64> = cuts.into_iter().filter(|t| (0.0..=1.0).contains(t)).collect();
        cuts.sort_by(f64::total_cmp);
        cuts.windows(2)
            .filter(|w| w[1] > w[0])
            .all(|w| self.covers(a + (0.5 * (w[0] + w[1])) * along))
    }
}

/// The sign of the turn from `a` to `b` to `c`: 1 left, -1 right, 0
/// collinear, exactly.
pub fn orientation(a: V2, b: V2, c: V2) -> i32 {
    let turn = robust::orient2d(
        robust::Coord { x: a.x, y: a.y },
        robust::Coord { x: b.x, y: b.y },
        robust::Coord { x: c.x, y: c.y },
    );
    if turn > 0.0 {
        1
    } else if turn < 0.0 {
        -1
    } else {
        0
    }
}

/// Where `p` lies against a closed ring, by GEOS's ray-crossing count.
pub fn locate_in_ring(ring: &[V2], p: V2) -> Location {
    let mut crossings = 0usize;
    for w in ring.windows(2) {
        let (p1, p2) = (w[0], w[1]);
        if p1.x < p.x && p2.x < p.x {
            continue;
        }
        if p == p2 {
            return Location::Boundary;
        }
        if p1.y == p.y && p2.y == p.y {
            if p1.x.min(p2.x) <= p.x && p.x <= p1.x.max(p2.x) {
                return Location::Boundary;
            }
            continue;
        }
        if (p1.y > p.y && p2.y <= p.y) || (p2.y > p.y && p1.y <= p.y) {
            let mut turn = orientation(p1, p2, p);
            if turn == 0 {
                return Location::Boundary;
            }
            if p2.y < p1.y {
                turn = -turn;
            }
            if turn > 0 {
                crossings += 1;
            }
        }
    }
    if crossings % 2 == 1 {
        Location::Interior
    } else {
        Location::Exterior
    }
}

/// Where `p` lies against a polygon, its exterior ring then its holes.
pub fn locate_in_polygon(rings: &[Vec<V2>], p: V2) -> Location {
    let Some((shell, holes)) = rings.split_first() else {
        return Location::Exterior;
    };
    match locate_in_ring(shell, p) {
        Location::Interior => {}
        other => return other,
    }
    for hole in holes {
        match locate_in_ring(hole, p) {
            Location::Interior => return Location::Exterior,
            Location::Boundary => return Location::Boundary,
            Location::Exterior => {}
        }
    }
    Location::Interior
}

/// The distance from `p` to the segment from `a` to `b`, by GEOS's
/// `pointToSegment`, which chooses a geometry's nearest segment.
pub fn segment_distance(p: V2, a: V2, b: V2) -> f64 {
    let d = b - a;
    let squared = d.x * d.x + d.y * d.y;
    let single = squared == 0.0;
    let scale = if single { 1.0 } else { squared };
    let r = ((p.x - a.x) * d.x + (p.y - a.y) * d.y) / scale;
    if single || r <= 0.0 {
        return ((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y)).sqrt();
    }
    if r >= 1.0 {
        return ((p.x - b.x) * (p.x - b.x) + (p.y - b.y) * (p.y - b.y)).sqrt();
    }
    let across = ((a.y - p.y) * d.x - (a.x - p.x) * d.y) / scale;
    across.abs() * scale.sqrt()
}

/// The point of the segment from `a` to `b` nearest `p`, as GEOS's
/// `closestPoint` finds it.
pub fn closest_point(p: V2, a: V2, b: V2) -> V2 {
    let d = b - a;
    let squared = d.x * d.x + d.y * d.y;
    let on_start = p == a;
    let on_end = p == b;
    let r = ((p.x - a.x) * d.x + (p.y - a.y) * d.y) / if squared == 0.0 { 1.0 } else { squared };
    let factor = if on_start || squared == 0.0 {
        0.0
    } else if on_end {
        1.0
    } else {
        r
    };
    if factor > 0.0 && factor < 1.0 {
        return V2::new(a.x + factor * d.x, a.y + factor * d.y);
    }
    let to_start = ((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y)).sqrt();
    let to_end = ((p.x - b.x) * (p.x - b.x) + (p.y - b.y) * (p.y - b.y)).sqrt();
    if to_start < to_end { a } else { b }
}

/// The first of the edges nearest `p` by `segment_distance`, and that
/// distance, as GEOS chooses a geometry's nearest segment.
pub fn nearest_edge(edges: &[(V2, V2)], p: V2) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (k, &(a, b)) in edges.iter().enumerate() {
        let d = segment_distance(p, a, b);
        if best.is_none_or(|(_, least)| d < least) {
            best = Some((k, d));
        }
    }
    best
}

/// The vector to `p` from the nearest point of the segment from `start`
/// along `along`.
pub fn offset(p: V2, start: V2, along: V2) -> V2 {
    let relative = p - start;
    let projection = relative.x * along.x + relative.y * along.y;
    let squared = along.x * along.x + along.y * along.y;
    let t = (projection / if squared > 0.0 { squared } else { 1.0 }).clamp(0.0, 1.0);
    relative - t * along
}

/// The distance between the segment from `a` along `ab` and the one from
/// `c` along `cd`: zero where they cross, else the nearest of their ends
/// to the other.
pub fn segment_gap(a: V2, ab: V2, c: V2, cd: V2) -> f64 {
    let b = a + ab;
    let d = c + cd;
    let crossing =
        ab.cross(c - a) * ab.cross(d - a) < 0.0 && cd.cross(a - c) * cd.cross(b - c) < 0.0;
    if crossing {
        return 0.0;
    }
    offset(c, a, ab)
        .hypot()
        .min(offset(d, a, ab).hypot())
        .min(offset(a, c, cd).hypot())
        .min(offset(b, c, cd).hypot())
}

/// The distance between the segments from `a` to `b` and from `c` to `d`,
/// by GEOS's `segmentToSegment`.
pub fn segment_to_segment(a: V2, b: V2, c: V2, d: V2) -> f64 {
    if a == b {
        return segment_distance(a, c, d);
    }
    if c == d {
        return segment_distance(d, a, b);
    }
    let apart = a.x.max(b.x) < c.x.min(d.x)
        || c.x.max(d.x) < a.x.min(b.x)
        || a.y.max(b.y) < c.y.min(d.y)
        || c.y.max(d.y) < a.y.min(b.y);
    let mut crossing = false;
    if !apart {
        let denominator = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
        if denominator != 0.0 {
            let r = ((a.y - c.y) * (d.x - c.x) - (a.x - c.x) * (d.y - c.y)) / denominator;
            let s = ((a.y - c.y) * (b.x - a.x) - (a.x - c.x) * (b.y - a.y)) / denominator;
            crossing = (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&s);
        }
    }
    if crossing {
        return 0.0;
    }
    segment_distance(a, c, d)
        .min(segment_distance(b, c, d))
        .min(segment_distance(c, a, b))
        .min(segment_distance(d, a, b))
}

/// How far `p` lies inside `area`, negative outside it, -inf for an empty
/// area: the distance to its boundary, signed by whether the area covers
/// the point. A via's zone depth (`vias._depth`) and the containment field
/// both judge by it.
pub fn zone_depth(area: &Shape, p: V2) -> f64 {
    if area.is_empty() {
        return f64::NEG_INFINITY;
    }
    let to_edge = nearest_edge(&area.edges(), p).map_or(f64::INFINITY, |(_, d)| d);
    if area.covers(p) { to_edge } else { -to_edge }
}

/// Lines measured along, as GEOS's linear referencing measures a line or
/// several: each target in turn, one after another.
struct Measured<'a> {
    lines: Vec<&'a [V2]>,
}

impl Measured<'_> {
    /// How far along the lines their point nearest `p` lies, by GEOS's
    /// `project`.
    fn project(&self, p: V2) -> f64 {
        let mut least = f64::INFINITY;
        let mut measure = 0.0;
        let mut start = 0.0;
        for line in &self.lines {
            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let length = (b - a).norm();
                let distance = segment_distance(p, a, b);
                let factor = projection_factor(p, a, b);
                let at = if factor <= 0.0 {
                    start
                } else if factor <= 1.0 {
                    start + factor * length
                } else {
                    start + length
                };
                if distance < least {
                    least = distance;
                    measure = at;
                }
                start += length;
            }
        }
        measure
    }

    /// The point `length` along the lines, by GEOS's `interpolate`.
    fn interpolate(&self, length: f64) -> V2 {
        let first = self.lines.first().and_then(|l| l.first()).copied();
        if length <= 0.0 {
            return first.unwrap_or_default();
        }
        let mut total = 0.0;
        for line in &self.lines {
            for w in line.windows(2) {
                let (a, b) = (w[0], w[1]);
                let segment = (b - a).norm();
                if total + segment > length {
                    let fraction = (length - total) / segment;
                    return along_by_fraction(a, b, fraction);
                }
                total += segment;
            }
            if total == length
                && let Some(&end) = line.last()
            {
                return end;
            }
        }
        self.lines
            .last()
            .and_then(|l| l.last())
            .copied()
            .unwrap_or_default()
    }
}

fn projection_factor(p: V2, a: V2, b: V2) -> f64 {
    if p == a {
        return 0.0;
    }
    if p == b {
        return 1.0;
    }
    let d = b - a;
    let squared = d.x * d.x + d.y * d.y;
    if squared <= 0.0 {
        return f64::NAN;
    }
    ((p.x - a.x) * d.x + (p.y - a.y) * d.y) / squared
}

fn along_by_fraction(a: V2, b: V2, fraction: f64) -> V2 {
    if fraction <= 0.0 {
        return a;
    }
    if fraction >= 1.0 {
        return b;
    }
    V2::new((b.x - a.x) * fraction + a.x, (b.y - a.y) * fraction + a.y)
}

/// Step along a boundary to find its tangent, and off it to tell inside
/// from out.
pub const NORMAL_PROBE: f64 = 1e-4;
/// The direction taken where a target has no tangent.
pub const FALLBACK_DIRECTION: V2 = V2::new(1.0, 0.0);

/// The unit normal of the shape's targets at a point lying on them,
/// outward for an area: a quarter turn of the tangent across NORMAL_PROBE
/// either side of the point, turned out of the area if it points in.
pub fn outward_normal(shape: &Shape, p: V2, area: bool) -> V2 {
    let lines: Vec<&[V2]> = shape.targets().filter(|l| l.len() > 1).collect();
    if lines.is_empty() {
        return FALLBACK_DIRECTION;
    }
    let measured = Measured { lines };
    let along = measured.project(p);
    let before = measured.interpolate((along - NORMAL_PROBE).max(0.0));
    let after = measured.interpolate(along + NORMAL_PROBE);
    let t = after - before;
    let norm = t.x.hypot(t.y);
    if norm == 0.0 {
        return FALLBACK_DIRECTION;
    }
    let n = V2::new(-t.y / norm, t.x / norm);
    if area && shape.contains(p + NORMAL_PROBE * n) {
        return -n;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, size: f64) -> Vec<V2> {
        vec![
            V2::new(x0, y0),
            V2::new(x0 + size, y0),
            V2::new(x0 + size, y0 + size),
            V2::new(x0, y0 + size),
            V2::new(x0, y0),
        ]
    }

    fn holed() -> Shape {
        Shape::polygon(vec![square(0.0, 0.0, 4.0), square(1.0, 1.0, 2.0)])
    }

    #[test]
    fn locates_points_against_a_polygon_with_a_hole() {
        let shape = holed();
        assert_eq!(shape.locate(V2::new(0.5, 0.5)), Location::Interior);
        assert_eq!(shape.locate(V2::new(2.0, 2.0)), Location::Exterior);
        assert_eq!(shape.locate(V2::new(1.0, 2.0)), Location::Boundary);
        assert_eq!(shape.locate(V2::new(0.0, 0.0)), Location::Boundary);
        assert_eq!(shape.locate(V2::new(4.0, 2.0)), Location::Boundary);
        assert_eq!(shape.locate(V2::new(5.0, 2.0)), Location::Exterior);
    }

    #[test]
    fn a_hole_is_outside_and_measured_to_its_ring() {
        let shape = holed();
        assert_eq!(shape.distance(V2::new(2.0, 2.0)), 1.0);
        assert!((zone_depth(&shape, V2::new(2.0, 2.0)) + 1.0).abs() < 1e-15);
        assert_eq!(zone_depth(&shape, V2::new(0.5, 2.0)), 0.5);
        assert_eq!(zone_depth(&shape, V2::new(0.0, 2.0)), 0.0);
    }

    #[test]
    fn an_empty_area_is_infinitely_far_inside_out() {
        let empty = Shape {
            parts: vec![],
            type_id: 3,
        };
        assert_eq!(zone_depth(&empty, V2::new(0.0, 0.0)), f64::NEG_INFINITY);
    }

    #[test]
    fn segments_measure_as_geos_measures_them() {
        let a = V2::new(0.0, 0.0);
        let b = V2::new(2.0, 0.0);
        assert_eq!(segment_to_segment(a, b, V2::new(1.0, -1.0), V2::new(1.0, 1.0)), 0.0);
        assert_eq!(segment_to_segment(a, b, V2::new(3.0, 0.0), V2::new(4.0, 0.0)), 1.0);
        assert_eq!(segment_to_segment(a, a, V2::new(0.0, 2.0), V2::new(0.0, 2.0)), 2.0);
    }

    #[test]
    fn a_segment_across_a_notch_is_not_covered() {
        // A U shape: the segment across its opening leaves the area.
        let u = Shape::polygon(vec![vec![
            V2::new(0.0, 0.0),
            V2::new(3.0, 0.0),
            V2::new(3.0, 3.0),
            V2::new(2.0, 3.0),
            V2::new(2.0, 1.0),
            V2::new(1.0, 1.0),
            V2::new(1.0, 3.0),
            V2::new(0.0, 3.0),
            V2::new(0.0, 0.0),
        ]]);
        assert!(!u.covers_segment(V2::new(0.5, 2.0), V2::new(2.5, 2.0), &u.edges()));
        assert!(u.covers_segment(V2::new(0.5, 0.5), V2::new(2.5, 0.5), &u.edges()));
        // Along the boundary and through the reflex corner, still inside.
        assert!(u.covers_segment(V2::new(0.0, 0.0), V2::new(3.0, 0.0), &u.edges()));
        assert!(!u.covers_segment(V2::new(0.5, 0.5), V2::new(1.5, 1.5), &u.edges()));
    }

    #[test]
    fn the_normal_on_an_edge_points_out_of_the_area() {
        let shape = Shape::polygon(vec![square(0.0, 0.0, 2.0)]);
        let n = outward_normal(&shape, V2::new(1.0, 0.0), true);
        assert!((n.x).abs() < 1e-12 && (n.y + 1.0).abs() < 1e-12);
        let n = outward_normal(&shape, V2::new(2.0, 1.0), true);
        assert!((n.x - 1.0).abs() < 1e-12 && n.y.abs() < 1e-12);
    }

    #[test]
    fn a_point_has_no_tangent() {
        let point = Shape::line(vec![V2::new(1.0, 1.0)]);
        assert_eq!(outward_normal(&point, V2::new(1.0, 1.0), false), FALLBACK_DIRECTION);
    }
}
