//! The differential pair helpers the optimizer calls: which nets pair, and
//! how tightly a pair's centreline bends.
//!
//! A net is of a diff pair by KiCad's router's naming rule
//! (`complementary_net`). A coupled section's centreline may bend no
//! tighter than the radius at which the inner track's inner edge still has
//! a radius of at least zero (`bend_radius_limit`); its bend radius at
//! each edge is the largest offset at which the edge's mitred offset still
//! has positive length (`energy::polyline::edge_bend_radii`), 0 beside a
//! fold and for an edge of no length (`bend_radii`).

use crate::energy::polyline::edge_bend_radii;
use crate::geometry::V2;
use crate::numeric::minimum;

/// Relative: a dense polyline drawn at the radius limit measures a hair
/// below it, R·cos(Δ/2) for a turn Δ per vertex, and must still pass.
pub const BEND_RADIUS_TOLERANCE: f64 = 1e-3;
/// 1 + cos(turn) at or below which a vertex is a fold: off the axes,
/// rounding leaves an exact fold's 1 + cos a hair above 0, where
/// sin / (1 + cos) is noise. A turn this close to a fold has a bend radius
/// of well under a nanometre either way.
pub const FOLD_SLACK: f64 = 1e-12;

/// Characters after a diff pair suffix that KiCad's router skips.
const IGNORED_TRAILING: &str = "0123456789_";

/// The name the other net of a diff pair would have, matching KiCad's
/// router (`MatchDpSuffix`): trailing digits and underscores are skipped,
/// and the character before them must be `+`, `-`, `P` or `N` (upper case
/// only). A net is of a diff pair only when that net exists.
pub fn complementary_net(name: &str) -> Option<String> {
    let stem = name.trim_end_matches(|c| IGNORED_TRAILING.contains(c));
    let last = stem.chars().next_back()?;
    let complement = match last {
        '+' => '-',
        '-' => '+',
        'P' => 'N',
        'N' => 'P',
        _ => return None,
    };
    let head = &stem[..stem.len() - last.len_utf8()];
    Some(format!("{head}{complement}{}", &name[stem.len()..]))
}

/// `net` and the other net of its diff pair, if it has one. Callers match
/// it only against nets of selected chains, so a complement that is not on
/// the board matches nothing.
pub fn diff_pair_nets(net: &str) -> Vec<String> {
    let mut nets = vec![net.to_owned()];
    nets.extend(complementary_net(net));
    nets
}

/// The smallest centreline bend radius at which the inner track's inner
/// edge still has a radius of at least zero.
pub fn bend_radius_limit(gap: f64, width: f64) -> f64 {
    (gap + width) / 2.0 + width / 2.0
}

/// Whether a bend radius is below `limit`, beyond round-off.
pub fn too_tight(radius: f64, limit: f64) -> bool {
    radius < limit * (1.0 - BEND_RADIUS_TOLERANCE)
}

/// Each edge's bend radius, as `edge_bend_radii` defines it, 0 beside a
/// fold and for an edge of no length, where the centreline turns on the
/// spot. For a dense arc of radius R with turn Δ per vertex it is
/// R·cos(Δ/2). Only the turns and folds at the vertices `turning` count,
/// every vertex when None.
pub fn bend_radii(centreline: &[V2], turning: Option<&[bool]>) -> Vec<f64> {
    let delta: Vec<V2> = centreline.windows(2).map(|w| w[1] - w[0]).collect();
    let lengths: Vec<f64> = delta.iter().map(|d| d.x.hypot(d.y)).collect();
    let direction: Vec<V2> = delta
        .iter()
        .zip(&lengths)
        .map(|(&d, &l)| {
            if l == 0.0 {
                V2::default()
            } else {
                V2::new(d.x / l, d.y / l)
            }
        })
        .collect();
    let mut folds = Vec::with_capacity(direction.len().saturating_sub(1));
    let mut half_turns = Vec::with_capacity(folds.capacity());
    for (k, w) in direction.windows(2).enumerate() {
        let (u, v) = (w[0], w[1]);
        let sine = u.x * v.y - u.y * v.x;
        let one_plus_cos = 1.0 + (u.x * v.x + u.y * v.y);
        let mut fold = one_plus_cos <= FOLD_SLACK;
        let mut half_turn = if fold { 0.0 } else { sine / one_plus_cos };
        if let Some(turning) = turning
            && !turning[k + 1]
        {
            fold = false;
            half_turn = 0.0;
        }
        folds.push(fold);
        half_turns.push(half_turn);
    }
    let (mut radii, _) = edge_bend_radii(&lengths, &half_turns);
    for (k, &fold) in folds.iter().enumerate() {
        if fold {
            radii[k] = 0.0;
            radii[k + 1] = 0.0;
        }
    }
    for (radius, &length) in radii.iter_mut().zip(&lengths) {
        if length == 0.0 {
            *radius = 0.0;
        }
    }
    radii
}

/// The centreline's smallest bend radius, counting the turns at the
/// vertices `turning` (`bend_radii`), and whether it is below the limit for
/// tracks of `width` `gap` apart; None for a centreline of no edge.
pub fn check_bend_radius(
    centreline: &[V2],
    gap: f64,
    width: f64,
    turning: Option<&[bool]>,
) -> Option<(f64, bool)> {
    let radius = bend_radii(centreline, turning)
        .into_iter()
        .reduce(minimum)?;
    Some((radius, too_tight(radius, bend_radius_limit(gap, width))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nets_pair_by_kicad_s_suffixes() {
        assert_eq!(complementary_net("USB_D+").as_deref(), Some("USB_D-"));
        assert_eq!(complementary_net("CLK_P_1").as_deref(), Some("CLK_N_1"));
        assert_eq!(complementary_net("LVDS3N").as_deref(), Some("LVDS3P"));
        assert_eq!(complementary_net("GND"), None);
        assert_eq!(complementary_net("12"), None);
        assert_eq!(complementary_net("clk_p"), None);
        assert_eq!(diff_pair_nets("A+"), vec!["A+".to_owned(), "A-".to_owned()]);
        assert_eq!(diff_pair_nets("GND"), vec!["GND".to_owned()]);
    }

    fn arc(radius: f64, steps: usize) -> Vec<V2> {
        (0..=steps)
            .map(|k| {
                let a = std::f64::consts::FRAC_PI_2 * k as f64 / steps as f64;
                V2::new(radius * a.cos(), radius * a.sin())
            })
            .collect()
    }

    #[test]
    fn a_dense_arc_s_radius_is_r_cos_half_its_turn() {
        let steps = 90;
        let turn = std::f64::consts::FRAC_PI_2 / steps as f64;
        let radii = bend_radii(&arc(2.0, steps), None);
        for r in &radii[1..radii.len() - 1] {
            assert!((r - 2.0 * (turn / 2.0).cos()).abs() < 1e-9, "{r}");
        }
    }

    #[test]
    fn a_fold_and_an_empty_edge_bend_on_the_spot() {
        let folded = [V2::new(0.0, 0.0), V2::new(1.0, 0.0), V2::new(0.0, 0.0)];
        assert_eq!(bend_radii(&folded, None), vec![0.0, 0.0]);
        let unturned = bend_radii(&folded, Some(&[true, false, true]));
        assert_eq!(unturned, vec![f64::INFINITY, f64::INFINITY]);
        let repeated = [V2::new(0.0, 0.0), V2::new(0.0, 0.0), V2::new(1.0, 0.0)];
        assert_eq!(bend_radii(&repeated, None)[0], 0.0);
    }

    #[test]
    fn a_bend_below_the_limit_is_too_tight() {
        let limit = bend_radius_limit(0.2, 0.2);
        let (radius, tight) = check_bend_radius(&arc(0.9 * limit, 90), 0.2, 0.2, None).unwrap();
        assert!(tight && radius < limit);
        let (_, tight) = check_bend_radius(&arc(1.1 * limit, 90), 0.2, 0.2, None).unwrap();
        assert!(!tight);
        assert!(!too_tight(
            limit * (1.0 - BEND_RADIUS_TOLERANCE / 2.0),
            limit
        ));
        assert_eq!(
            check_bend_radius(&[V2::new(0.0, 0.0)], 0.2, 0.2, None),
            None
        );
    }
}
