//! What the energy keeps between evaluations: the pairs of the members'
//! quadrature points and the pieces they see of each other
//! (`NeighbourList`), and each asker's search of its static field
//! (`StaticList`). Either only spares searches: the energy is the one a
//! search over everything gives.

use std::sync::Arc;

use crate::field::Field;
use crate::field::edges::{Found, StaticEdges};
use crate::field::search::{StaticItems, search_parts, static_smooth};
use crate::field::smooth::smooth_reach;
use crate::geometry::V2;
use crate::numeric::maximum;
use crate::seen::{Pairs, Seen};
use crate::stats;

/// The skin each layer of a `NeighbourList` starts with, mm.
pub const SKIN_LEAST: f64 = 0.05;
/// The widest skin a layer of a `NeighbourList` takes, mm.
pub const SKIN_MOST: f64 = 0.15;
/// The skin of every asker of a `StaticList`, mm.
pub const STATIC_SKIN: f64 = SKIN_LEAST;

/// The largest distance from start[i] to points[i], mm, NaN if any is.
fn moved(points: &[V2], start: &[V2]) -> f64 {
    // numpy's `einsum` contracts two columns without fusing the products.
    let squared = points.iter().zip(start).map(|(&p, &s)| {
        let delta = p - s;
        delta.dot(delta)
    });
    squared.reduce(maximum).map_or(0.0, f64::sqrt)
}

#[derive(Clone, Debug)]
struct ListFound {
    points: Vec<V2>,
    ends: Vec<V2>,
    pairs: Pairs,
    skin: f64,
}

/// A Verlet list: per layer, the pairs of its members' quadrature points
/// and the pieces they see of the others (`Seen`) that can answer each
/// other while every one of those points and every vertex of the edges
/// they see stays within the layer's skin of where it was when the pairs
/// were found. Once any has moved farther, the layer finds its pairs again
/// where they are. Either way the energy is the one a search over every
/// piece gives, whatever the skin: a point's distance to a piece changes by
/// at most its own move plus the larger move of the piece's ends, so the
/// pairs are found with twice the skin to spare. The list only spares the
/// search.
///
/// A layer's skin starts at SKIN_LEAST and, each time the layer finds its
/// pairs again, becomes how far its points and pieces had moved since it
/// last found them, at most SKIN_MOST. Pieces that moved that far are
/// likely to move as far again, so a list lasts about as many evaluations
/// whether they move far or barely, and one made for a burst that hardly
/// moves stays small. A wider skin costs at every evaluation: a pair is
/// listed while its edge may come within SMOOTH_REACH of the point's
/// nearest, so the band listed behind the nearest is SMOOTH_REACH plus four
/// skins wide. A search costs about six evaluations' answers.
///
/// Measured with `./run bench --only bus`, a fixed skin of 0.2 mm listed
/// more than twice the pairs of SKIN_LEAST and ran no faster than one of
/// 0.1 mm. SKIN_MOST ran fastest on the Z-shaped buses (`z_bus`), whose
/// tracks move far. It ran a tenth slower than a fixed 0.1 mm on 262 short
/// chains at 0.6 mm pitch (`short_chains`), where the band reaches a
/// second track behind the nearest. On a real board of 100 chains and 96
/// moving vias, a 60 s run took 81 searches in 166 evaluations instead of
/// 170, and its objective 58.3 s instead of 70.8 s.
///
/// Every search counts in `stats`.
#[derive(Clone, Debug)]
pub struct NeighbourList {
    reach: f64,
    found: Vec<Option<ListFound>>,
}

impl NeighbourList {
    /// A list for the layers of `seen`, for an energy whose terms end at
    /// `reach`.
    pub fn new(seen: &Seen, reach: f64) -> Self {
        Self {
            reach,
            found: vec![None; seen.layers.len()],
        }
    }

    /// Layer n's (point, piece) pairs with its members' quadrature points
    /// at `points`, each of the member `owner` gives, and what they see of
    /// the others at `flat`, with each pair's inflation.
    pub fn pairs(
        &mut self,
        seen: &Seen,
        n: usize,
        points: &[V2],
        owner: &[usize],
        flat: &[V2],
    ) -> &Pairs {
        let edges = &seen.layers[n].edges;
        let ends: Vec<V2> = edges
            .iter()
            .map(|&v| flat[v])
            .chain(edges.iter().map(|&v| flat[v + 1]))
            .collect();
        let mut skin = SKIN_LEAST;
        let mut keep = false;
        if let Some(found) = &self.found[n]
            && found.points.len() == points.len()
        {
            let (of_points, of_ends) = (moved(points, &found.points), moved(&ends, &found.ends));
            let farthest = if of_ends > of_points {
                of_ends
            } else {
                of_points
            };
            keep = farthest <= found.skin;
            skin = if SKIN_MOST < farthest {
                SKIN_MOST
            } else {
                farthest
            };
        }
        if !keep {
            // A point's distance to a piece changes by at most its own
            // move plus the larger move of the piece's ends.
            stats::searched();
            let pairs = seen.candidates(n, points, owner, flat, self.reach, 2.0 * skin);
            self.found[n] = Some(ListFound {
                points: points.to_vec(),
                ends,
                pairs,
                skin,
            });
        }
        self.found[n]
            .as_ref()
            .map_or(&NO_PAIRS, |found| &found.pairs)
    }
}

static NO_PAIRS: Pairs = Pairs {
    point: Vec::new(),
    piece: Vec::new(),
    inflation: Vec::new(),
};

#[derive(Clone)]
struct Asker {
    field: Option<Arc<dyn Field>>,
    /// Its field's static edges and their items, None for a field that
    /// gives none.
    parts: Option<(Vec<Arc<StaticEdges>>, StaticItems)>,
    found: Option<(Vec<V2>, Found)>,
}

/// The static fields of several askers, members or moving vias, each at
/// its own points, answered from their edges (`field::edges::StaticEdges`)
/// with one search per asker, kept between evaluations as `NeighbourList`
/// keeps a layer's pairs.
///
/// Static edges never move, so a point's distance to one changes by at most
/// the point's own move: an asker searches again once one of its points
/// has moved farther than STATIC_SKIN since its last search, with that skin
/// to spare. Either way the answers are those of a search at the points;
/// the list only spares the search.
///
/// Unlike a layer's in `NeighbourList`, the skin does not follow how far
/// the points moved: a wider skin lists more edges of each item and more
/// items behind the nearest at every evaluation. Measured with
/// `./run bench --only bus` on the pad rows, a skin following the moves up
/// to SKIN_MOST listed 1.6 times the pairs and took a tenth longer.
///
/// A field that gives no edges (`Field::static_parts`) answers by its own
/// `smooth_distance`, and an asker with no field sees nothing. Not
/// `listed`, every call searches without a skin. The points each field is
/// asked about count in `stats`.
#[derive(Clone)]
pub struct StaticList {
    askers: Vec<Asker>,
    reach: f64,
    listed: bool,
}

impl StaticList {
    pub fn new(fields: Vec<Option<Arc<dyn Field>>>, reach: f64, listed: bool) -> Self {
        let askers = fields
            .into_iter()
            .map(|field| {
                let parts = field.as_ref().and_then(|f| f.static_parts()).map(|parts| {
                    let items = StaticItems::new(parts.clone());
                    (parts, items)
                });
                Asker {
                    field,
                    parts,
                    found: None,
                }
            })
            .collect();
        Self {
            askers,
            reach,
            listed,
        }
    }

    /// Each asker's smooth distance at its `queries` and its gradient, as
    /// its field's `smooth_distance` with the list's reach gives them.
    pub fn smooth(&mut self, queries: &[&[V2]]) -> Vec<(Vec<f64>, Vec<V2>)> {
        let reach = smooth_reach(self.reach);
        let listed = self.listed;
        self.askers
            .iter_mut()
            .zip(queries)
            .map(|(asker, &points)| {
                let Some(field) = &asker.field else {
                    return (
                        vec![f64::INFINITY; points.len()],
                        vec![V2::default(); points.len()],
                    );
                };
                stats::queried(field, points.len());
                let Some((parts, items)) = &asker.parts else {
                    return field.smooth_distance(points, self.reach);
                };
                let found = match &mut asker.found {
                    Some((last, found))
                        if listed
                            && last.len() == points.len()
                            && moved(points, last) <= STATIC_SKIN =>
                    {
                        &*found
                    }
                    slot => {
                        let numbered: Vec<(&StaticEdges, usize)> = parts
                            .iter()
                            .zip(items.starts())
                            .map(|(part, &first)| (part.as_ref(), first))
                            .collect();
                        let slack = if listed { STATIC_SKIN } else { 0.0 };
                        let found = search_parts(&numbered, points, reach, slack);
                        &slot.insert((points.to_vec(), found)).1
                    }
                };
                static_smooth(points, found, items)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::chain::ChainField;
    use crate::seen::{Pieces, Track, Viewer};

    fn level(first: usize, net: usize) -> Track {
        Track {
            first,
            vertices: 5,
            layer: "F".into(),
            net,
            kind: 0,
        }
    }

    fn viewer(net: usize) -> Viewer {
        Viewer {
            layer: "F".into(),
            net,
            kind: 0,
            exempt: vec![],
        }
    }

    fn levels(heights: &[f64]) -> Vec<V2> {
        heights
            .iter()
            .flat_map(|&y| (0..5).map(move |x| V2::new(f64::from(x), y)))
            .collect()
    }

    const REACH: f64 = 1.0;
    const INFLATION: f64 = 0.3;

    /// Two levels of nets 0 and 1 seeing each other.
    fn two() -> Seen {
        Seen::new(
            &[level(0, 0), level(5, 1)],
            &[viewer(0), viewer(1)],
            vec![vec![INFLATION]],
            &[Pieces::default(), Pieces::default()],
        )
    }

    fn assert_listed_is_searched(seen: &Seen, list: &mut NeighbourList, flat: &[V2]) {
        let queries: Vec<&[V2]> = vec![&flat[..5], &flat[5..]];
        let listed = seen.smooth_answers(&queries, flat, REACH, Some(list));
        let searched = seen.smooth_answers(&queries, flat, REACH, None);
        assert_eq!(listed, searched);
    }

    fn searches() -> u64 {
        stats::statistics().list_searches
    }

    #[test]
    fn a_list_searches_again_only_past_its_skin() {
        let seen = two();
        let mut list = NeighbourList::new(&seen, REACH);
        let before = searches();
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 0.8]));
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 0.8 - 0.9 * SKIN_LEAST]));
        // Other tests in the process count too, so only a lower bound holds.
        assert!(searches() > before);
        let found = list.found[0].clone().expect("searched");
        assert_eq!(found.skin, SKIN_LEAST);
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 0.8 - 1.1 * SKIN_LEAST]));
        let again = list.found[0].clone().expect("searched");
        assert!(again.skin > SKIN_LEAST);
    }

    #[test]
    fn an_edge_listed_beyond_reach_answers_once_it_comes_within_it() {
        let seen = two();
        let mut list = NeighbourList::new(&seen, REACH);
        let within = REACH + crate::field::smooth::SMOOTH_REACH + INFLATION;
        let far = within + 1.8 * SKIN_LEAST;
        let approach = 0.95 * SKIN_LEAST;
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, far]));
        let first = list.found[0].clone().expect("searched").points;
        assert_listed_is_searched(&seen, &mut list, &levels(&[approach, far - approach]));
        assert_eq!(list.found[0].clone().expect("kept").points, first);
        assert!(far - 2.0 * approach < within);
    }

    #[test]
    fn a_list_widens_its_skin_to_skin_most_at_most() {
        let seen = two();
        let mut list = NeighbourList::new(&seen, REACH);
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 0.6]));
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 1.6]));
        assert_eq!(list.found[0].as_ref().expect("searched").skin, SKIN_MOST);
        let kept = list.found[0].clone().expect("searched").points;
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 1.6 + 0.95 * SKIN_MOST]));
        assert_eq!(list.found[0].clone().expect("kept").points, kept);
        assert_listed_is_searched(&seen, &mut list, &levels(&[0.0, 1.6 + 1.05 * SKIN_MOST]));
        assert_ne!(list.found[0].clone().expect("searched").points, kept);
    }

    #[test]
    fn a_static_list_answers_as_a_search_and_searches_again_only_past_its_skin() {
        let field: Arc<dyn Field> = Arc::new(ChainField::new(
            &[V2::new(0.0, 0.0)],
            &[V2::new(4.0, 0.0)],
            vec![INFLATION],
            vec![0],
        ));
        let mut list = StaticList::new(vec![Some(field.clone()), None], REACH, true);
        let mut unlisted = StaticList::new(vec![Some(field), None], REACH, false);
        let at = |y: f64| vec![V2::new(1.0, y), V2::new(2.0, y)];
        for y in [0.8, 0.8 - 0.9 * STATIC_SKIN, 0.5] {
            let points = at(y);
            let queries: [&[V2]; 2] = [&points, &points[..1]];
            let listed = list.smooth(&queries);
            assert_eq!(listed, unlisted.smooth(&queries));
            assert_eq!(listed[1].0, vec![f64::INFINITY]);
        }
        let (last, _) = list.askers[0].found.clone().expect("searched");
        assert_eq!(last, at(0.5));
    }

    use crate::field::composite::CompositeField;
    use crate::field::obstacles::{Item, Obstacles};
    use crate::field::search::parts_smooth_distance;
    use crate::geometry::Shape;
    use crate::testing::FreeSpace;

    fn ring(corners: &[(f64, f64)]) -> Vec<V2> {
        let mut ring: Vec<V2> = corners.iter().map(|&(x, y)| V2::new(x, y)).collect();
        ring.push(ring[0]);
        ring
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<V2> {
        ring(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    fn item(shape: Shape, inflation: f64) -> Item {
        Item {
            shape,
            inflation,
            net: None,
            extent: 0.0,
            layer: None,
        }
    }

    const PAD_ROW_REACH: f64 = 0.3;
    const PAD_INFLATION: f64 = 0.3;

    /// Four pads in a row and a track, and a board edge.
    fn pad_row() -> Arc<Obstacles> {
        let mut items: Vec<Item> = (0..4)
            .map(|k| {
                let x = 1.2 * f64::from(k);
                item(
                    Shape::polygon(vec![rect(x, 0.0, x + 1.0, 0.6)]),
                    PAD_INFLATION,
                )
            })
            .collect();
        items.push(item(
            Shape::line(vec![V2::new(0.0, 1.6), V2::new(5.0, 1.9)]),
            0.35,
        ));
        let outline = Shape::polygon(vec![rect(-1.0, -1.0, 6.0, 2.5)]);
        Arc::new(Obstacles::new(items, Some(outline), 0.4))
    }

    /// Points beside and between the pad row's pads, the last 0.015 mm right
    /// of a pad's side, which a move left by more than that takes inside it.
    fn pad_row_points() -> Vec<V2> {
        let mut points = Vec::new();
        for j in 0..15 {
            for i in 0..40 {
                points.push(V2::new(
                    -0.6 + 6.2 * f64::from(i) / 39.0,
                    -0.6 + 2.8 * f64::from(j) / 14.0,
                ));
            }
        }
        points.push(V2::new(1.015, 0.3));
        points
    }

    fn parts(field: &dyn Field) -> Vec<Arc<StaticEdges>> {
        field.static_parts().unwrap_or_default()
    }

    #[test]
    fn a_static_list_on_a_pad_row_searches_again_only_past_its_skin() {
        let field = pad_row();
        let edges = parts(field.as_ref());
        let asker: Arc<dyn Field> = field;
        let mut list = StaticList::new(vec![Some(asker)], PAD_ROW_REACH, true);
        let points = pad_row_points();
        let mut last: Option<Vec<V2>> = None;
        for (shift, searches) in [
            (0.0, true),
            (0.8 * STATIC_SKIN, false),
            (2.0 * STATIC_SKIN, true),
        ] {
            let at: Vec<V2> = points.iter().map(|&p| p - V2::new(shift, 0.0)).collect();
            let got = list.smooth(&[&at]);
            let searched = list.askers[0].found.as_ref().map(|(p, _)| p.clone());
            assert_eq!(searched != last, searches);
            last = searched;
            assert_eq!(got[0], parts_smooth_distance(&edges, &at, PAD_ROW_REACH));
            let inside = got[0].0[at.len() - 1] < -PAD_INFLATION;
            assert_eq!(inside, shift > 0.0);
        }
    }

    #[test]
    fn a_static_list_answers_each_asker_as_its_own_field() {
        let row = pad_row();
        let centres = Arc::new(ChainField::new(
            &[V2::new(2.2, 1.0), V2::new(4.0, 1.0)],
            &[V2::new(2.2, 1.0), V2::new(4.0, 1.0)],
            vec![0.5, 0.5],
            vec![0, 1],
        ));
        let joined: Arc<dyn Field> = Arc::new(CompositeField::new(vec![row.clone(), centres]));
        let analytic: Arc<dyn Field> = Arc::new(FreeSpace);
        let points = pad_row_points();
        let mut list = StaticList::new(
            vec![
                Some(row.clone()),
                Some(joined.clone()),
                Some(analytic.clone()),
                None,
            ],
            PAD_ROW_REACH,
            true,
        );
        let got = list.smooth(&[&points, &points, &points, &points]);
        assert_eq!(got[0], row.smooth_distance(&points, PAD_ROW_REACH));
        assert_eq!(
            got[1],
            parts_smooth_distance(&parts(joined.as_ref()), &points, PAD_ROW_REACH)
        );
        assert_eq!(got[2], analytic.smooth_distance(&points, PAD_ROW_REACH));
        assert!(got[3].0.iter().all(|&d| d == f64::INFINITY));
        assert!(got[3].1.iter().all(|&g| g == V2::default()));
    }

    /// A xorshift generator, so that random copper is the same every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn uniform(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.next()
        }

        fn below(&mut self, count: usize) -> usize {
            ((self.next() * count as f64) as usize).min(count - 1)
        }
    }

    fn disc(c: V2, radius: f64, segments: u32) -> Vec<V2> {
        let corners: Vec<(f64, f64)> = (0..segments)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / f64::from(segments);
                (c.x + radius * libm::cos(a), c.y + radius * libm::sin(a))
            })
            .collect();
        ring(&corners)
    }

    /// Items of every kind at random in a 6 mm square, one geometry twice.
    fn random_copper(random: &mut Random) -> Vec<Item> {
        let count = 3 + random.below(12);
        let mut items: Vec<Item> = Vec::new();
        for _ in 0..count {
            let c = V2::new(random.uniform(0.0, 6.0), random.uniform(0.0, 6.0));
            let inflation = [0.0, 0.1, 0.25, 0.4, 0.6][random.below(5)];
            let shape = match random.below(5) {
                0 => {
                    let (w, h) = (random.uniform(0.05, 1.0), random.uniform(0.05, 1.0));
                    Shape::polygon(vec![rect(c.x, c.y, c.x + w, c.y + h)])
                }
                1 => Shape::polygon(vec![disc(c, random.uniform(0.05, 0.6), 32)]),
                2 => {
                    let mut turns = vec![c];
                    for _ in 0..2 {
                        let step = V2::new(random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0));
                        turns.push(c + step);
                    }
                    Shape::line(turns)
                }
                3 => Shape::line(vec![c]),
                _ => Shape::polygon(vec![
                    rect(c.x, c.y, c.x + 1.5, c.y + 1.5),
                    rect(c.x + 0.4, c.y + 0.4, c.x + 1.1, c.y + 1.1),
                ]),
            };
            items.push(item(shape, inflation));
        }
        let again = item(items[0].shape.clone(), items[0].inflation);
        items.push(again);
        items
    }

    const EDGE_INFLATION: f64 = 0.3;

    /// A static field of random copper, with or without a board edge, with a
    /// cutout in the board, or joined with centres and a zone's region.
    fn random_field(random: &mut Random, trial: usize) -> Arc<dyn Field> {
        let board = rect(-0.5, -0.5, 6.5, 6.5);
        let outline = if !trial.is_multiple_of(2) {
            Shape::polygon(vec![board, rect(2.8, 2.8, 3.2, 3.2)])
        } else {
            Shape::polygon(vec![board])
        };
        let field = Arc::new(Obstacles::new(
            random_copper(random),
            (!trial.is_multiple_of(3)).then_some(outline),
            EDGE_INFLATION,
        ));
        if !trial.is_multiple_of(4) {
            return field;
        }
        let starts: Vec<V2> = (0..4)
            .map(|_| V2::new(random.uniform(0.0, 6.0), random.uniform(0.0, 6.0)))
            .collect();
        let ends: Vec<V2> = starts
            .iter()
            .map(|&s| {
                if random.next() > 0.5 {
                    s + V2::new(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                } else {
                    s
                }
            })
            .collect();
        let inflations = (0..4).map(|_| random.uniform(0.1, 0.5)).collect();
        let centres = Arc::new(ChainField::new(
            &starts,
            &ends,
            inflations,
            vec![0, 0, 1, 2],
        ));
        let zone = Arc::new(Obstacles::new(
            Vec::new(),
            Some(Shape::polygon(vec![disc(V2::new(3.0, 3.0), 2.5, 64)])),
            0.0,
        ));
        Arc::new(CompositeField::new(vec![field, centres, zone]))
    }

    /// How much deeper than the board edge's reach the points beside it
    /// start, mm: just past where a search finds the edge.
    const BESIDE_DEEP: f64 = 0.01;

    #[test]
    fn a_static_list_answers_as_a_search_on_random_copper() {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        for trial in 0..30 {
            let field = random_field(&mut random, trial);
            let edges = parts(field.as_ref());
            let reach = [0.2, 0.5, f64::INFINITY][trial / 2 % 3];
            // Just outside where the board edge is skipped as deep inside,
            // moving toward the left edge; with no finite reach, none is.
            let skipped = EDGE_INFLATION + smooth_reach(reach) + BESIDE_DEEP;
            let beside = if reach.is_finite() { 40 } else { 0 };
            let mut points: Vec<V2> = (0..200)
                .map(|_| V2::new(random.uniform(-1.0, 7.0), random.uniform(-1.0, 7.0)))
                .collect();
            for _ in 0..beside {
                let depth = skipped + random.uniform(0.0, 0.03);
                points.push(V2::new(-0.5 + depth, random.uniform(0.5, 5.5)));
            }
            let mut list = StaticList::new(vec![Some(field.clone())], reach, true);
            list.smooth(&[&points]);
            for _ in 0..4 {
                // Within the skin of where the list searched, so that it
                // does not search again.
                let at: Vec<V2> = points
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| {
                        let way = if k >= points.len() - beside {
                            V2::new(-1.0, 0.0)
                        } else {
                            let angle = random.uniform(0.0, std::f64::consts::TAU);
                            V2::new(libm::cos(angle), libm::sin(angle))
                        };
                        let step = random.uniform(0.0, 0.999 * STATIC_SKIN);
                        p + V2::new(step * way.x, step * way.y)
                    })
                    .collect();
                let got = list.smooth(&[&at]);
                let searched = list.askers[0].found.as_ref().map(|(p, _)| p);
                assert_eq!(searched, Some(&points));
                assert_eq!(
                    got[0],
                    parts_smooth_distance(&edges, &at, reach),
                    "trial {trial}"
                );
            }
        }
    }
}
