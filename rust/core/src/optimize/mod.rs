//! The optimizer's run loop: every chain, coupled section and moving via of
//! a problem as one L-BFGS problem, given bursts until it settles.
//!
//! The energy of a problem (`problem`) is the sum of each chain's energy in
//! its static field plus the other chains on its layer, taken as one
//! function of every free vertex. The distance the obstacle terms take is
//! the smooth minimum over the obstacles near a point
//! (`field::smooth::smooth_minimum`), so where a quadrature point of one
//! chain sees another chain, the obstacle term's gradient also reaches, by
//! that chain's share, the two vertices of its edge nearest the point. The
//! checks take exact distances. Without moving vias the layers do not
//! interact, so each layer is a problem of its own, and an outer iteration
//! (`advance`) gives every problem that has not settled one burst
//! (`state::State::step`): resample each chain to the spacing between its
//! fixed vertices and those pressed into the others' clearance, which
//! resampling keeps where they are (`state::State::pressed`), run L-BFGS
//! over every free vertex without bounds (`Bursts`, `burst`), scaled by
//! each variable's stiffness (`layout`), and move back what the checks
//! reject (`checks`).
//!
//! Copper of one net does not interact: a chain sees nothing of the chains,
//! tracks and vias of its net (`members::Members::seen`), and its static
//! field holds none of its net's copper, so chains of one net may come
//! close, overlap or cross. Nor do the tracks of a diff pair's two nets:
//! those may come close, overlap or cross too, and KiCad's DRC reports it
//! if they do. A selection that branches is split at each junction into
//! chains that end there, loose: those ends are one shared variable, and
//! the branches, of one net, do not see each other.
//!
//! A coupled section of a differential pair is a member of its layer's
//! problem as its centreline: its free vertices are variables, its energy
//! has the bend-radius hinge, and it keeps clear of the others as the diff
//! pair's outer extent, but for the chains of its two nets, which it
//! ignores and which ignore it, so that its fan-outs leave it freely. The
//! others see its two tracks, offset from the centreline at each
//! evaluation, so where the obstacle term pulls on a track's edge, the
//! gradient goes back through the offset to the centreline's vertices
//! (`pair_blocks::PairOffset::offset_pullback`). Besides the edge check, a
//! burst moves back what `polylines::SelfCheck` finds: a centreline newly
//! bent below its limit. A held member, such as a held section, is fully
//! pinned and not resampled. A section the run would end bent too tightly
//! is held so within the run: its problem is built again with the section
//! as drawn and goes on from its best state, with what would come too close
//! to the section put back where it started (`state::State::go_on_from`).
//!
//! A moving via is two more variables, its centre. The ends of the chains
//! it joins are not variables of their own: each is the via's centre plus
//! its offset at the start, at every evaluation, so they keep their offsets
//! exactly and their gradient goes to the via. Its own obstacle terms are
//! described in `problem`. The layers that any moving via spans make one
//! problem, every member on them included, whether or not the vias' spans
//! overlap; a via sees only what lies on its layers, and two vias keep
//! their distance only where their spans overlap.
//!
//! When it stops
//! -------------
//! A problem settles when its burst moves nothing by `settled_change` and
//! moves nothing back, when its lowest energy and its tracks' length stall
//! over `stall_window` bursts, the energy by a tolerance for each moving
//! member and via and the length by one (`stalled`), or after
//! MAX_ABNORMAL_BURSTS abnormal bursts in a row. At a coarse level any of
//! these hands over to the next level instead (`state`), so only the final
//! level gives up. The settle threshold and
//! the energy's tolerance hold per member, so that a problem and far-apart
//! copies of it stall together. Members that have settled still add their
//! tolerance, so one still gaining energy while its length grows may stop
//! sooner beside them. The run converges when every problem has settled,
//! none of them on abnormal bursts, and otherwise stops at the iteration
//! cap or at the time limit, which ends the running burst at its next
//! L-BFGS iteration; that burst's state goes through the checks and the
//! guard as any, and it settles nothing. Each problem keeps its best state
//! either way. A run where every problem settled without moving anything
//! had nothing to do.

pub mod checks;
pub mod layout;
pub mod problem;
pub mod state;

use crate::burst;
use crate::numeric::python_max;
use checks::{Check, Round};
use problem::{Lookups, Objective};
use state::State;

/// After this many abnormal L-BFGS ends in a row a problem counts as
/// settled and keeps what its bursts reached.
pub const MAX_ABNORMAL_BURSTS: u32 = 3;

/// A run whose steps change no shape by more than this, KiCad's 1 nm grid,
/// mm, has nothing to do, however its stop rules are set.
pub const NOTHING_MOVED: f64 = 1e-6;

/// How deep inside the other members' and vias' clearance, mm, a vertex
/// lies when resampling keeps it where it is (`state::State::pressed`), so
/// that one that only grazes a neighbour's clearance resamples as any
/// other.
pub const PRESSED_DEPTH: f64 = 0.02;

/// Depth inside the clearance, mm, that the violation guard ignores as
/// float noise: a track routed at exactly the clearance resamples to
/// vertices about 1e-15 mm inside. Far below KiCad's 1 nm coordinate step.
pub const INSIDE_TOLERANCE: f64 = 1e-7;

/// How much deeper than at the start something already inside may measure
/// without going deeper, mm: round-off in measuring a shape that moved
/// along what it is inside, such as an edge sliding inside a clearance, up
/// to about 1e-16 mm. A thousandth of KiCad's 1 nm coordinate step.
pub const DEEPER_ROUND_OFF: f64 = 1e-9;

/// A chain end: its first point or its last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    First,
    Last,
}

impl End {
    /// The end's index in a polyline of `len` vertices.
    pub fn of(self, len: usize) -> usize {
        match self {
            End::First => 0,
            End::Last => len - 1,
        }
    }
}

/// The most coarse spacings a run's schedule may hold before its final
/// spacing.
pub const MAX_COARSE_LEVELS: usize = 7;

/// The coarse spacings a problem runs at before its final spacing, mm,
/// coarsest first; none by default. Held in a fixed array so that
/// `Settings` stays `Copy`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoarseSpacings {
    values: [f64; MAX_COARSE_LEVELS],
    count: usize,
}

impl CoarseSpacings {
    /// `spacings`, or None if there are more than `MAX_COARSE_LEVELS`.
    pub fn new(spacings: &[f64]) -> Option<Self> {
        let count = spacings.len();
        if count > MAX_COARSE_LEVELS {
            return None;
        }
        let mut values = [0.0; MAX_COARSE_LEVELS];
        values[..count].copy_from_slice(spacings);
        Some(Self { values, count })
    }

    /// The spacings, coarsest first.
    pub fn as_slice(&self) -> &[f64] {
        &self.values[..self.count]
    }
}

/// What the run loop and a problem's bursts take of the optimizer's
/// settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Target distance between resampled vertices at the final level, mm.
    pub spacing: f64,
    /// The levels a problem runs at before the final spacing.
    pub coarse: CoarseSpacings,
    /// A problem whose burst moves no shape and no via's centre by this
    /// much, mm, settles.
    pub settled_change: f64,
    pub stall_window: usize,
    pub stall_tolerance: f64,
    pub burst: burst::Settings,
}

/// The energy of a problem was not finite: its members' value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotFinite(pub f64);

/// Whether a problem of `members` moving members and `vias` moving vias
/// has stalled: over `stall_window` bursts its lowest energy fell by no
/// more than `stall_tolerance` for each member and via, and its tracks'
/// summed `lengths` by no more than one tolerance.
///
/// The energy side counts per member and via, so that a problem and two
/// far-apart copies of it stall together on it. Moving vias that repel
/// each other, such as a selected grid of stitching vias, settle on a
/// shallow energy: once they have made way for the routes they creep on by
/// tens of micrometres a burst while the energy keeps falling by about a
/// micrometre a burst, and their share of the tolerance trades that small
/// gain away. The length side keeps that share from stopping routes that
/// still get shorter as the vias spread, and holds for the whole problem:
/// per member, it let a stitched plane's routes end 17 µm longer. So two
/// copies of a problem may run longer than one, never stop sooner.
/// Measured in https://github.com/mairas/kibends/issues/138 and
/// https://github.com/mairas/kibends/issues/114.
pub fn stalled(
    lowest: &[f64],
    lengths: &[f64],
    settings: &Settings,
    members: usize,
    vias: usize,
) -> bool {
    let window = settings.stall_window;
    if lowest.len() <= window {
        return false;
    }
    let gain = lowest[lowest.len() - 1 - window] - lowest[lowest.len() - 1];
    let shortened = lengths[lengths.len() - 1 - window] - lengths[lengths.len() - 1];
    gain <= settings.stall_tolerance * (members + vias) as f64
        && shortened <= settings.stall_tolerance
}

/// What one burst cost and moved, for the run's caller to log.
#[derive(Clone, Debug, PartialEq)]
pub struct BurstReport {
    /// The problem, by its place among those the run advanced.
    pub problem: usize,
    /// The level of the schedule it ran at (`state::Shared::spacing_at`).
    pub level: usize,
    pub iterations: u64,
    pub evaluations: u64,
    /// Seconds the evaluations took.
    pub evaluating: f64,
    /// Seconds the rest of the step took: L-BFGS's own work, resampling and
    /// the checks.
    pub rest: f64,
    /// What L-BFGS says ended it; empty for a burst that did not run it.
    pub message: String,
    /// The lowest energy the problem's bursts have reached, and its
    /// tracks' summed length after this one, as the stall rule takes them.
    pub lowest: f64,
    pub length: f64,
    /// The problem's free members, and each one's length and shape change
    /// in the burst.
    pub members: Vec<usize>,
    pub lengths: Vec<f64>,
    pub changes: Vec<f64>,
}

/// What the run tells its caller as it goes.
#[derive(Clone, Debug, PartialEq)]
pub enum Notice {
    /// L-BFGS ended a burst of the problem abnormally, `streak` in a row.
    Abnormal {
        problem: usize,
        streak: u32,
    },
    Burst(BurstReport),
    /// The problem enters `level` of a schedule with coarse levels: each
    /// coarse level, and the final one after a coarser one hands over. Not
    /// for the probe at the final spacing that runs before them.
    Stage {
        problem: usize,
        level: usize,
    },
    /// After an outer iteration: its number, its largest change, and each
    /// problem's length and energy.
    Progress {
        iteration: u64,
        change: f64,
        lengths: Vec<f64>,
        energies: Vec<f64>,
    },
}

/// A burst as its runner reports it, its point after the check moved back
/// what crossed.
#[derive(Clone, Debug, PartialEq)]
pub struct Ran {
    pub point: Vec<f64>,
    pub abnormal: bool,
    pub iterations: u64,
    pub evaluations: u64,
    pub message: String,
    pub cut: bool,
    pub evaluating: f64,
}

/// What runs a problem's bursts: L-BFGS over `objective` from `start` with
/// the per-coordinate `scale`, its evaluations searching through
/// `lookups`, then `check`'s revert on the point it reached. Returns the
/// burst and the first round the check found, if it ran.
pub trait Bursts<E> {
    fn run(
        &mut self,
        objective: Objective,
        lookups: Lookups,
        start: Vec<f64>,
        scale: Vec<f64>,
        check: Check,
    ) -> Result<(Ran, Option<Round>), E>;
}

/// How an `advance` ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every problem settled, none on abnormal bursts, and none moved
    /// anything.
    NothingToDo,
    /// Every problem settled, none on abnormal bursts.
    Converged,
    /// Every problem settled, some on abnormal bursts, or one given up on
    /// before.
    GaveUp,
    IterationCap,
    /// `expired` answered true after a burst.
    TimeLimit,
}

/// Outer iterations of `problems` from `iteration` until `cap`, each giving
/// every problem that has not settled one burst, until every problem has
/// settled or `expired`, asked after each burst, answers true. `gave_up`
/// says some problem was given up on before. After each outer iteration it
/// reports its progress.
pub fn advance<E: From<NotFinite>>(
    problems: &mut [&mut State],
    iteration: &mut u64,
    cap: u64,
    gave_up: bool,
    mut expired: Option<&mut dyn FnMut() -> Result<bool, E>>,
    bursts: &mut dyn Bursts<E>,
    report: &mut dyn FnMut(Notice) -> Result<(), E>,
) -> Result<Outcome, E> {
    while *iteration < cap {
        *iteration += 1;
        let mut out_of_time = false;
        let mut change = 0.0;
        for (n, problem) in problems.iter_mut().enumerate() {
            if problem.settled {
                continue;
            }
            change = python_max(change, problem.step(n, bursts, report)?);
            if let Some(expired) = expired.as_mut()
                && expired()?
            {
                out_of_time = true;
                break;
            }
        }
        report(Notice::Progress {
            iteration: *iteration,
            change,
            lengths: problems.iter().map(|p| p.length()).collect(),
            energies: problems.iter().map(|p| p.energy).collect(),
        })?;
        if problems.iter().all(|p| p.settled) {
            return Ok(if gave_up || problems.iter().any(|p| p.gave_up()) {
                Outcome::GaveUp
            } else if !problems.iter().any(|p| p.moved) {
                Outcome::NothingToDo
            } else {
                Outcome::Converged
            });
        }
        if out_of_time {
            return Ok(Outcome::TimeLimit);
        }
    }
    Ok(Outcome::IterationCap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Settings {
        Settings {
            spacing: 0.1,
            coarse: CoarseSpacings::default(),
            settled_change: 0.005,
            stall_window: 3,
            stall_tolerance: 1e-3,
            burst: burst::Settings {
                max_iterations: 50,
                ftol: 1e-9,
                gtol: 1e-10,
            },
        }
    }

    #[test]
    fn coarse_spacings_hold_up_to_the_most_levels_in_order() {
        let most = [6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0.15];
        assert_eq!(CoarseSpacings::new(&most).unwrap().as_slice(), most);
        assert_eq!(CoarseSpacings::new(&[]).unwrap().as_slice(), &[] as &[f64]);
        assert_eq!(CoarseSpacings::new(&[]), Some(CoarseSpacings::default()));
        let too_many = [12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0.15];
        assert_eq!(CoarseSpacings::new(&too_many), None);
    }

    #[test]
    fn a_problem_stalls_when_energy_and_length_stop_falling_over_the_window() {
        let s = settings();
        let flat = [10.0, 9.9998, 9.9996, 9.9995];
        assert!(!stalled(&flat[..3], &flat[..3], &s, 1, 0));
        assert!(stalled(&flat, &flat, &s, 1, 0));
        let falling = [10.0, 9.0, 8.0, 7.0];
        assert!(!stalled(&falling, &flat, &s, 1, 0));
        // The energy's tolerance grows with the members and vias, the
        // length's does not.
        let slow = [10.0, 9.9992, 9.9985, 9.998];
        assert!(!stalled(&slow, &flat, &s, 1, 0));
        assert!(stalled(&slow, &flat, &s, 2, 1));
        assert!(!stalled(&flat, &slow, &s, 2, 1));
    }
}
