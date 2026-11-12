//! A burst's variables and their scale.
//!
//! Each free vertex of a member is a variable, the loose ends of a
//! junction's branches share one, and each moving via's centre is one; an
//! end a via carries is the via's variable plus the end's offset
//! (`problem::Layout`).
//!
//! A burst scales each variable by 1/sqrt of its stiffness at the burst's
//! start, relative to that of a vertex between two edges of its level's
//! spacing in free space, and never above 1 (`scale`). L-BFGS builds one curvature
//! model for the whole problem: unscaled, it fits the stiffest members,
//! such as tracks pressed together at clearance, and a far chain that could
//! settle in a few bursts moves 3 to 10 times less per burst
//! (https://github.com/mairas/kibends/issues/170). The stiffness is the
//! diagonal second derivative of the energy, the larger over x and y, from
//! the length term (`length_stiffness`), each member's obstacle term at its
//! own quadrature points (`members::Members::stiffness`) and the moving
//! vias' own terms, without the distance field's own curvature
//! (`energy::edge_curvature`). It leaves out the pull of a member's
//! obstacle term on another member's edges, the pairwise via term, the
//! contact stops and a section's bend-radius hinge.

use crate::energy::MIN_EDGE_LENGTH;
use crate::geometry::V2;
use crate::numeric::maximum;

use super::End;
use super::problem::Layout;

/// A variable with this many edges of the spacing at its vertex, and
/// nothing else, scales by 1: a vertex inside a resampled chain in free
/// space.
pub const REFERENCE_EDGES: f64 = 2.0;

/// Per member, each vertex's variable, None where it is fixed, with the
/// ends of each of `groups` sharing one; and how many variables there are.
pub fn variables(
    masks: &[Vec<bool>],
    groups: &[Vec<(usize, End)>],
) -> (Vec<Vec<Option<usize>>>, usize) {
    let mut index: Vec<Vec<Option<usize>>> = masks.iter().map(|m| vec![None; m.len()]).collect();
    let mut count = 0;
    for group in groups {
        for &(i, end) in group {
            let at = end.of(index[i].len());
            index[i][at] = Some(count);
        }
        count += 1;
    }
    for (row, mask) in index.iter_mut().zip(masks) {
        for (i, &fixed) in row.iter_mut().zip(mask) {
            if !fixed && i.is_none() {
                *i = Some(count);
                count += 1;
            }
        }
    }
    (index, count)
}

/// The layout of a burst from `starts`, with the vias at `positions`
/// carrying the `carried` ends, per via the member, its end and the end's
/// offset from the via's centre.
pub fn layout(
    starts: Vec<Vec<V2>>,
    masks: &[Vec<bool>],
    groups: &[Vec<(usize, End)>],
    carried: &[Vec<(usize, End, V2)>],
) -> Layout {
    let (mut index, mut count) = variables(masks, groups);
    let mut offsets: Vec<Vec<V2>> = starts
        .iter()
        .map(|s| vec![V2::default(); s.len()])
        .collect();
    let via_index: Vec<usize> = (count..count + carried.len()).collect();
    for (&k, ends) in via_index.iter().zip(carried) {
        for &(i, end, offset) in ends {
            let at = end.of(index[i].len());
            index[i][at] = Some(k);
            offsets[i][at] = offset;
        }
    }
    count += carried.len();
    Layout {
        starts,
        index,
        offsets,
        via_index,
        count,
    }
}

/// The flat start of `layout`'s variables with the vias at `positions`:
/// each vertex's start less its offset, then each via's centre.
pub fn start_of(layout: &Layout, positions: &[V2]) -> Vec<f64> {
    let mut x = vec![V2::default(); layout.count];
    for ((start, index), offset) in layout.starts.iter().zip(&layout.index).zip(&layout.offsets) {
        for ((&p, i), &o) in start.iter().zip(index).zip(offset) {
            if let Some(i) = *i {
                x[i] = p - o;
            }
        }
    }
    for (&v, &at) in layout.via_index.iter().zip(positions) {
        x[v] = at;
    }
    x.into_iter().flat_map(|p| [p.x, p.y]).collect()
}

/// Each of `layout`'s variables' stiffness under the length term at its
/// start: Σ 1/l over the edges at the vertices it moves.
pub fn length_stiffness(layout: &Layout) -> Vec<f64> {
    let mut stiffness = vec![0.0; layout.count];
    for (points, index) in layout.starts.iter().zip(&layout.index) {
        let inverse: Vec<f64> = points
            .windows(2)
            .map(|w| 1.0 / maximum((w[1] - w[0]).norm(), MIN_EDGE_LENGTH))
            .collect();
        for side in 0..2 {
            for (k, &inv) in inverse.iter().enumerate() {
                if let Some(i) = index[k + side] {
                    stiffness[i] += inv;
                }
            }
        }
    }
    stiffness
}

/// Each flat coordinate's scale for a burst: 1/sqrt of its variable's
/// `stiffness` relative to a free chain vertex's between edges of the
/// `spacing`, so that L-BFGS sees the variables about equally stiff. No
/// variable scales above 1: a via carrying no ends has no length
/// stiffness, and its curvature in free space, near 0, would scale it by
/// tens and throw it across obstacles.
pub fn scale(stiffness: &[f64], spacing: f64) -> Vec<f64> {
    let reference = REFERENCE_EDGES / spacing;
    stiffness
        .iter()
        .flat_map(|&s| {
            let k = (reference / maximum(s, reference)).sqrt();
            [k, k]
        })
        .collect()
}

/// For each of a problem's `count` members, then each of its vias, the
/// unit it goes back with: a junction's branches share their loose end,
/// and a via carries the ends of its chains. A unit is known by one of its
/// own.
pub fn units(
    count: usize,
    groups: &[Vec<(usize, End)>],
    carried: &[Vec<(usize, End, V2)>],
) -> Vec<usize> {
    let mut root: Vec<usize> = (0..count + carried.len()).collect();
    fn find(root: &[usize], mut n: usize) -> usize {
        while root[n] != n {
            n = root[n];
        }
        n
    }
    fn join(root: &mut [usize], a: usize, b: usize) {
        let (a, b) = (find(root, a), find(root, b));
        root[a] = b;
    }
    for group in groups {
        for pair in group.windows(2) {
            join(&mut root, pair[0].0, pair[1].0);
        }
    }
    for (k, ends) in carried.iter().enumerate() {
        for &(i, _, _) in ends {
            join(&mut root, count + k, i);
        }
    }
    (0..root.len()).map(|n| find(&root, n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_junction_s_ends_share_a_variable_and_a_via_carries_its_ends() {
        let masks = vec![vec![true, false, false], vec![false, false, true]];
        let groups = vec![vec![(0, End::Last), (1, End::First)]];
        let carried = vec![vec![(0, End::First, V2::new(0.1, 0.0))]];
        let starts = vec![
            vec![V2::new(0.0, 0.0), V2::new(1.0, 0.0), V2::new(2.0, 0.0)],
            vec![V2::new(2.0, 0.0), V2::new(3.0, 0.0), V2::new(4.0, 0.0)],
        ];
        let layout = layout(starts, &masks, &groups, &carried);
        assert_eq!(layout.index[0], vec![Some(3), Some(1), Some(0)]);
        assert_eq!(layout.index[1], vec![Some(0), Some(2), None]);
        assert_eq!(layout.count, 4);
        let x = start_of(&layout, &[V2::new(-0.1, 0.0)]);
        assert_eq!(x, vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0, -0.1, 0.0]);
        let (polylines, positions) = layout.unflatten(&x);
        assert_eq!(polylines[0][0], V2::new(0.0, 0.0));
        assert_eq!(positions, vec![V2::new(-0.1, 0.0)]);
    }

    #[test]
    fn a_free_vertex_between_edges_of_the_spacing_scales_by_one() {
        let starts = vec![vec![
            V2::new(0.0, 0.0),
            V2::new(0.1, 0.0),
            V2::new(0.2, 0.0),
        ]];
        let layout = layout(starts, &[vec![true, false, true]], &[], &[]);
        let reference = REFERENCE_EDGES / 0.1;
        assert_eq!(length_stiffness(&layout), vec![reference]);
        // Stiffer scales down, and nothing scales above 1.
        assert_eq!(
            scale(&[0.0, reference, 4.0 * reference], 0.1),
            vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
        );
    }

    #[test]
    fn a_via_and_the_branches_it_joins_go_back_as_one() {
        let groups = vec![vec![(0, End::Last), (1, End::First)]];
        let carried = vec![vec![(2, End::First, V2::default())]];
        let unit = units(4, &groups, &carried);
        assert_eq!(unit[0], unit[1]);
        assert_eq!(unit[2], unit[4]);
        assert_ne!(unit[0], unit[2]);
        assert_eq!(unit[3], 3);
    }
}
