//! One L-BFGS burst: argmin's `LBFGS` driven by a thin wrapper solver that
//! ends it by scipy's L-BFGS-B rules, which the optimizer's settings and
//! iteration budgets were measured with.
//!
//! The wrapper delegates every iteration to `LBFGS` (memory `MEMORY`, a
//! More-Thuente line search with scipy's Wolfe constants, `LBFGS`'s own
//! tolerances 0) and decides in `Solver::terminate` when the burst ends:
//!
//! - before the first iteration, when the gradient's largest component is
//!   at most `gtol`;
//! - after an iteration, at the iteration cap, when `stop` answers true,
//!   when the gradient's largest component is at most `gtol`, or when the
//!   energy fell by at most `ftol` relative to the larger of the two
//!   energies and 1, in that order. scipy asks its callback and checks its
//!   cap as an iteration ends and tests convergence only when it is called
//!   for the next one, so the cap and `stop` come first, and the cap before
//!   `stop`, which is asked after every iteration, the last included.
//!
//! An iteration is accepted when its line search ended at an energy no
//! higher than the one it started from. The wrapper keeps the last accepted
//! point, its energy and gradient, and the counts of accepted iterations
//! and of energy evaluations after every iteration, so that whatever ends
//! the burst it returns them. A line search fails when argmin reports it
//! failed (an energy that is not a number, a direction that does not
//! descend), which leaves no point behind, when it ends above its start,
//! as More-Thuente may when it cannot meet the Wolfe conditions, or after
//! `MAX_LINE_SEARCH` evaluations, scipy's `maxls`.
//!
//! A failed search whose lowest trial lies below the start by the
//! sufficient decrease takes that trial as the iteration, without a
//! correction pair, as L-BFGS skips a step that misses the curvature
//! condition. Against a steep wall along the line, such as a via's
//! clearance, only a narrow band meets the curvature condition, and the
//! search creeps toward it from below while each bisection lands beyond;
//! scipy then ends the burst abnormally where the next trial might have
//! found the band. Otherwise, as in scipy, a failure while L-BFGS has
//! correction pairs drops them and retries the iteration from the last
//! accepted point along the gradient, and a failure without pairs ends the
//! burst abnormally at that point, as scipy's status 2 did.
//!
//! L-BFGS keeps a correction pair only when it curves upward, s·y > ε·(−g·s)
//! with s the step, y the change of the gradient along it and g the
//! gradient at its start, as scipy requires. A More-Thuente search may end
//! on its width tolerance without the curvature condition and argmin's
//! L-BFGS keeps every pair, where s·y ≤ 0 would turn the next direction
//! uphill or make it 0/0. An accepted iteration whose pair does not curve
//! upward is kept without its pair, as scipy skips it, and so is a lowest
//! trial taken after a search that ended above its start, which argmin
//! paired with the search's end point. argmin's pairs are private, so the
//! wrapper keeps a copy of `LBFGS` from before each step and takes it back
//! to drop the step's pair.
//!
//! A burst given `entered` keeps every point it takes clear: a trial of a
//! line search that `entered` rejects counts as an energy of +inf, which
//! the objective is not asked for. argmin's More-Thuente cannot take an
//! infinite energy, so the objective's answer there is an error
//! (`Rejected`) that the search catches (`ScipyStart`) and starts again
//! from its start. What `entered` still rejects at PROBE_FRACTION of the
//! step crosses at first order: copper resting against a clearance that
//! the direction pushes into, which no step along it can take. The search
//! holds at the start the coordinates `entered` names there and at the
//! rejected trial, and tries the rest of the step again at its length, so
//! that the copper beside moves on as if nothing rested against a wall:
//! among tracks pressed together at clearance, holding only what rests
//! took a round of rejections for each track. Otherwise, or where holding
//! leaves no descent, it bisects: it runs along its direction bounded to
//! STEP_BACK of the rejected step and starting there, and a search ended
//! at its bound still descending takes the bound, short of what it
//! rejected. Rejected trials count toward MAX_LINE_SEARCH, so a search
//! whose every trial is rejected fails as one that runs out of
//! evaluations does; a bound below More-Thuente's smallest step fails it
//! at once. Either way the failure is handled as any other below, its
//! lowest trial admitted, but that a failure without correction pairs
//! whose last rejected trial would have descended, as one more evaluation
//! there tells, ends the burst normally where it is (`Ending::Blocked`, a
//! defensive path no known run reaches): a clearance blocks every descent
//! there, and counting it abnormal would give up on a problem that only
//! rests against its clearances.
//!
//! The line search's width tolerance is scipy's too, since argmin's default
//! keeps narrowing an interval that round-off already decides, and so is
//! its first step: the burst's first search, along the gradient, starts
//! at a step of unit length, every later one at 1 (`ScipyStart`): against
//! a wall, where the first trial lands decides whether a search finds the
//! band.
//!
//! An error of the objective or of `stop` ends the burst and is returned as
//! it is. argmin's Ctrl-C handling is off: the caller's `stop` checks for
//! signals.
//!
//! The solver works on `Vec<f64>`: argmin-math's newest ndarray backend is
//! older than the one rust-numpy uses. With a scale s, the burst minimises
//! over y with x = s ⊙ y, so that one curvature model fits stiff and soft
//! coordinates alike; the gradient rules apply to the gradient by y, and
//! the point returned is x.
//!
//! The objective gives the energy and its gradient together, but argmin
//! asks for them apart, the gradient right after the energy at the same
//! point, so the last evaluation is cached and counted once, as scipy
//! counts its evaluations.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

use argmin::core::{
    CostFunction, Error, Executor, Gradient, IterState, KV, LineSearch, Problem, Solver, State,
    TerminationReason, TerminationStatus,
};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

/// The correction pairs L-BFGS keeps, scipy's default `maxcor`.
pub const MEMORY: usize = 10;
/// The line search's sufficient decrease constant, as scipy's L-BFGS-B
/// sets it.
pub const SUFFICIENT_DECREASE: f64 = 1e-3;
/// The line search's curvature constant, as scipy's L-BFGS-B sets it.
pub const CURVATURE: f64 = 0.9;
/// The relative width of the line search's interval at which it ends, as
/// scipy's L-BFGS-B sets it.
pub const WIDTH_TOLERANCE: f64 = 0.1;
/// Energy evaluations after which a line search fails, scipy's default
/// `maxls`; a rejected trial counts as one.
pub const MAX_LINE_SEARCH: u64 = 20;
/// The fraction of a rejected step that the line search starts again
/// bounded to: bisection toward the start.
pub const STEP_BACK: f64 = 0.5;
/// The fraction of a rejected step at which the check is asked again:
/// what it still rejects there crosses at first order, such as copper
/// resting against a clearance that the direction pushes into. Then what
/// either rejection names is held for the rest of the search, and the
/// rest of the step is tried again whole. Small enough that a step only
/// a long step takes across is bisected instead.
pub const PROBE_FRACTION: f64 = 1e-3;
/// The smallest step of a line search, More-Thuente's default.
const MIN_STEP: f64 = 1.490_116_119_384_765_6e-8;

/// When a burst ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// The iteration cap.
    pub max_iterations: u64,
    /// The relative energy decrease of an iteration at or below which the
    /// burst converges.
    pub ftol: f64,
    /// The gradient's largest component at or below which the burst
    /// converges.
    pub gtol: f64,
}

/// What ended a burst.
#[derive(Clone, Debug, PartialEq)]
pub enum Ending {
    /// The gradient's largest component fell to `gtol`.
    Gradient,
    /// An iteration lowered the energy by at most `ftol`, relatively.
    Decrease,
    /// The burst ran its iteration cap.
    MaxIterations,
    /// `stop` answered true before the cap.
    Stopped,
    /// A line search without correction pairs failed, none of its admitted
    /// trials below the start by the sufficient decrease, and its last
    /// rejected trial (`ClearanceEntered`) lies below it so: a clearance
    /// blocks the descent from the last accepted point along the gradient.
    /// A search whose rejected trials would not have descended either ends
    /// abnormally. A defensive path: no run of the test scenarios, of the
    /// benchmark's cases or from a result resting at its clearances reaches
    /// it, since copper resting at a clearance with its edges clear has no
    /// descent that only enters it, and a search that fails after its first
    /// iteration has correction pairs; this module's tests reach it with
    /// hand-made objectives.
    Blocked,
    /// An iteration was not accepted, for the reason given.
    Abnormal(String),
}

impl Ending {
    /// The ending as scipy's L-BFGS-B words it.
    pub fn message(&self) -> String {
        match self {
            Ending::Gradient => "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL".into(),
            Ending::Decrease => "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH".into(),
            Ending::MaxIterations => "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT".into(),
            Ending::Stopped => "STOP: CALLBACK REQUESTED HALT".into(),
            Ending::Blocked => "STOP: A CLEARANCE BLOCKS THE LINE SEARCH".into(),
            Ending::Abnormal(reason) => format!("ABNORMAL: {reason}"),
        }
    }
}

/// A burst's result.
#[derive(Clone, Debug, PartialEq)]
pub struct Burst {
    /// The last accepted point, x.
    pub point: Vec<f64>,
    /// The energy there.
    pub energy: f64,
    /// The accepted iterations.
    pub iterations: u64,
    /// The objective's evaluations.
    pub evaluations: u64,
    /// The line searches' trials `entered` rejected.
    pub rejected: u64,
    /// The line searches that failed, whether a lowest trial, a retry
    /// without correction pairs or an abnormal end followed.
    pub failed_searches: u64,
    /// The time the evaluations took.
    pub evaluating: Duration,
    pub ending: Ending,
}

impl Burst {
    pub fn abnormal(&self) -> bool {
        matches!(self.ending, Ending::Abnormal(_))
    }

    /// Whether `stop` ended it before its iteration cap.
    pub fn cut(&self) -> bool {
        self.ending == Ending::Stopped
    }
}

/// A line search's trial: its y, and the energy and gradient there.
type Trial = (Vec<f64>, Evaluation);

/// The energy and its gradient at x.
pub type Evaluation = (f64, Vec<f64>);

/// The energy and its gradient as a function of x, or the error that ends
/// the burst.
pub type Objective<'a, E> = dyn FnMut(&[f64]) -> Result<Evaluation, E> + 'a;

/// Where x enters a clearance: None where it enters none and a line search
/// may take it; else the coordinates to hold at the search's start for it
/// to enter none, which may be none.
pub type ClearanceEntered<'a> = dyn FnMut(&[f64]) -> Option<Vec<bool>> + 'a;

/// Minimises `objective` from `start` over y with x = `scale` ⊙ y, or over
/// x without a scale, asking `stop` after each iteration and taking only
/// trials that enter no clearance by `entered`, if given.
pub fn run<E>(
    objective: &mut Objective<'_, E>,
    start: &[f64],
    scale: Option<&[f64]>,
    settings: &Settings,
    stop: &mut dyn FnMut() -> Result<bool, E>,
    entered: Option<&mut ClearanceEntered<'_>>,
) -> Result<Burst, E> {
    if let Some(s) = scale {
        assert_eq!(s.len(), start.len(), "one scale per coordinate");
    }
    let scaled = Scaled {
        objective: RefCell::new(objective),
        scale,
        entered: RefCell::new(entered.map(|a| a as &mut ClearanceEntered<'_>)),
        last: RefCell::new(None),
        failure: RefCell::new(None),
        evaluations: Cell::new(0),
        rejected: Cell::new(0),
        last_rejected: RefCell::new(None),
        search_start: RefCell::new(None),
        failed_searches: Cell::new(0),
        searching: Cell::new(None),
        lowest: RefCell::new(None),
        evaluating: Cell::new(Duration::ZERO),
    };
    let y0 = match scale {
        Some(s) => start.iter().zip(s).map(|(x, s)| x / s).collect(),
        None => start.to_vec(),
    };
    let mut record = Record {
        point: y0.clone(),
        energy: f64::INFINITY,
        gradient: Vec::new(),
        largest: f64::INFINITY,
        decrease: f64::INFINITY,
        iterations: 0,
        ending: None,
        failure: None,
    };
    let wrapper = Wrapper {
        lbfgs: lbfgs(true),
        pairs: 0,
        settings: *settings,
        record: &mut record,
        stop,
    };
    let run = Executor::new(Operator(&scaled), wrapper)
        .configure(|state| state.param(y0))
        .ctrlc(false)
        .run()
        .map(drop);
    if let Some(e) = scaled.failure.take().or(record.failure.take()) {
        return Err(e);
    }
    let ending = match run {
        Ok(_) => record
            .ending
            .unwrap_or_else(|| Ending::Abnormal("the solver ended without a reason".into())),
        Err(e) => Ending::Abnormal(e.to_string()),
    };
    let point = match scale {
        Some(s) => record.point.iter().zip(s).map(|(y, s)| s * y).collect(),
        None => record.point,
    };
    crate::stats::burst_ended(
        matches!(ending, Ending::Abnormal(_)),
        scaled.failed_searches.get(),
        scaled.rejected.get(),
    );
    Ok(Burst {
        point,
        energy: record.energy,
        iterations: record.iterations,
        evaluations: scaled.evaluations.get(),
        rejected: scaled.rejected.get(),
        failed_searches: scaled.failed_searches.get(),
        evaluating: scaled.evaluating.get(),
        ending,
    })
}

type Lbfgs = LBFGS<ScipyStart, Vec<f64>, Vec<f64>, f64>;
type Iterate = IterState<Vec<f64>, Vec<f64>, (), (), (), f64>;
type MoreThuente = MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>;

/// L-BFGS whose first line search, if `first`, starts as scipy's first
/// one does.
fn lbfgs(first: bool) -> Lbfgs {
    let search = ScipyStart {
        search: more_thuente(),
        first,
        direction: Vec::new(),
        held: Vec::new(),
        start: None,
    };
    LBFGS::new(search, MEMORY)
        .with_tolerance_grad(0.0)
        .and_then(|l| l.with_tolerance_cost(0.0))
        .expect("zero tolerances are valid")
}

/// A line search's trial that `entered` rejected, at y, and the
/// coordinates it named to hold.
#[derive(Debug)]
struct Rejected {
    at: Vec<f64>,
    hold: Vec<bool>,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a trial was not admitted")
    }
}

impl std::error::Error for Rejected {}

/// More-Thuente starting a burst's first line search, along the gradient,
/// at a step of unit length, 1/|d|, as scipy's L-BFGS-B does, and every
/// later one at 1. L-BFGS sets the direction on the line search it keeps
/// and searches with a copy, so the first direction set is the first
/// search's. A rejected trial starts the search again (`step_back`).
#[derive(Clone)]
struct ScipyStart {
    search: MoreThuente,
    first: bool,
    /// The direction the search runs along, held coordinates 0.
    direction: Vec<f64>,
    /// The coordinates the running search holds at its start.
    held: Vec<bool>,
    /// The search's start: y, the gradient there and the energy.
    start: Option<(Vec<f64>, Vec<f64>, f64)>,
}

/// More-Thuente with scipy's constants.
fn more_thuente() -> MoreThuente {
    MoreThuente::new()
        .with_c(SUFFICIENT_DECREASE, CURVATURE)
        .and_then(|l| l.with_width_tolerance(WIDTH_TOLERANCE))
        .expect("scipy's line search constants are valid")
}

impl ScipyStart {
    /// The search started again from its start after a trial `rejected`.
    /// Where that trial names coordinates to hold that the search does not
    /// hold yet, and the direction with them held still descends, along
    /// that direction from the rejected step, unbounded; else along its
    /// direction bounded to STEP_BACK of the rejected step, its first
    /// trial there, which fails when the bound falls below MIN_STEP.
    fn step_back<O>(
        &mut self,
        problem: &mut Problem<O>,
        state: Iterate,
        rejected: &Rejected,
    ) -> Result<(Iterate, Option<KV>), Error>
    where
        O: CostFunction<Param = Vec<f64>, Output = f64>
            + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
    {
        let (y0, g0, f0) = self.start.clone().expect("the search keeps its start");
        let along: f64 = rejected
            .at
            .iter()
            .zip(&y0)
            .zip(&self.direction)
            .map(|((y, y0), d)| (y - y0) * d)
            .sum();
        let step = along / self.direction.iter().map(|d| d * d).sum::<f64>();
        if self.held.len() != y0.len() {
            self.held = vec![false; y0.len()];
        }
        let mut grew = false;
        for (h, &r) in self.held.iter_mut().zip(&rejected.hold) {
            grew |= r && !*h;
            *h |= r;
        }
        let holding: Vec<f64> = self
            .direction
            .iter()
            .zip(&self.held)
            .map(|(&d, &h)| if h { 0.0 } else { d })
            .collect();
        let slope: f64 = holding.iter().zip(&g0).map(|(d, g)| d * g).sum();
        let mut search = if grew && slope < 0.0 {
            self.direction = holding;
            let mut search = more_thuente();
            search.initial_step_length(step)?;
            search
        } else {
            let bound = STEP_BACK * step;
            let mut search = more_thuente().with_bounds(MIN_STEP, bound)?;
            search.initial_step_length(bound)?;
            search
        };
        search.search_direction(self.direction.clone());
        search.init(problem, IterState::new().param(y0).gradient(g0).cost(f0))?;
        self.search = search;
        Ok((state, None))
    }
}

impl LineSearch<Vec<f64>, f64> for ScipyStart {
    fn search_direction(&mut self, direction: Vec<f64>) {
        let length = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
        let step = if self.first && length.is_normal() {
            1.0 / length
        } else {
            1.0
        };
        self.first = false;
        self.search
            .initial_step_length(step)
            .expect("a positive step is valid");
        self.direction.clone_from(&direction);
        self.search.search_direction(direction);
    }

    fn initial_step_length(&mut self, step: f64) -> Result<(), Error> {
        self.search.initial_step_length(step)
    }
}

impl<O> Solver<O, Iterate> for ScipyStart
where
    O: CostFunction<Param = Vec<f64>, Output = f64>
        + Gradient<Param = Vec<f64>, Gradient = Vec<f64>>,
{
    fn name(&self) -> &str {
        "More-Thuente starting as scipy does"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: Iterate,
    ) -> Result<(Iterate, Option<KV>), Error> {
        if let (Some(y), Some(g)) = (state.get_param(), state.get_gradient()) {
            self.start = Some((y.clone(), g.clone(), state.get_cost()));
        }
        self.held.clear();
        self.search.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: Iterate,
    ) -> Result<(Iterate, Option<KV>), Error> {
        let kept = state.clone();
        match self.search.next_iter(problem, state) {
            Err(e) => match e.downcast::<Rejected>() {
                Ok(rejected) => self.step_back(problem, kept, &rejected),
                Err(e) => Err(e),
            },
            stepped => stepped,
        }
    }
}

/// The objective over y, caching its last evaluation.
struct Scaled<'a, E> {
    objective: RefCell<&'a mut Objective<'a, E>>,
    scale: Option<&'a [f64]>,
    /// What judges a line search's trials, by x.
    entered: RefCell<Option<&'a mut ClearanceEntered<'a>>>,
    /// The last y evaluated, the energy there and its gradient by y.
    last: RefCell<Option<(Vec<f64>, Evaluation)>>,
    /// The objective's error, which ends the burst.
    failure: RefCell<Option<E>>,
    evaluations: Cell<u64>,
    rejected: Cell<u64>,
    /// The running line search's last rejected trial, y.
    last_rejected: RefCell<Option<Vec<f64>>>,
    /// Where the running line search starts, y.
    search_start: RefCell<Option<Vec<f64>>>,
    failed_searches: Cell<u64>,
    /// The evaluations of the running line search; None outside one.
    searching: Cell<Option<u64>>,
    /// The running line search's lowest trial: its y, energy and gradient.
    lowest: RefCell<Option<(Vec<f64>, Evaluation)>>,
    evaluating: Cell<Duration>,
}

impl<E> Scaled<'_, E> {
    fn x_of(&self, y: &[f64]) -> Vec<f64> {
        match self.scale {
            Some(s) => y.iter().zip(s).map(|(y, s)| s * y).collect(),
            None => y.to_vec(),
        }
    }

    fn evaluate(&self, y: &[f64]) -> Result<Evaluation, Error> {
        if let Some((at, evaluation)) = self.last.borrow().as_ref()
            && at.as_slice() == y
        {
            return Ok(evaluation.clone());
        }
        if self.failure.borrow().is_some() {
            return Err(Error::msg("the objective failed"));
        }
        if let Some(n) = self.searching.get() {
            if n >= MAX_LINE_SEARCH {
                return Err(Error::msg(format!(
                    "the line search took more than {MAX_LINE_SEARCH} evaluations"
                )));
            }
            self.searching.set(Some(n + 1));
        }
        let x = self.x_of(y);
        if self.searching.get().is_some()
            && let Some(entered) = self.entered.borrow_mut().as_mut()
            && let Some(named) = entered(&x)
        {
            self.rejected.set(self.rejected.get() + 1);
            *self.last_rejected.borrow_mut() = Some(y.to_vec());
            let hold = match self.search_start.borrow().as_ref() {
                Some(y0) => {
                    let near: Vec<f64> = y0
                        .iter()
                        .zip(y)
                        .map(|(a, b)| a + PROBE_FRACTION * (b - a))
                        .collect();
                    match entered(&self.x_of(&near)) {
                        Some(resting) => {
                            resting.iter().zip(&named).map(|(&r, &n)| r || n).collect()
                        }
                        None => Vec::new(),
                    }
                }
                None => Vec::new(),
            };
            return Err(Error::new(Rejected {
                at: y.to_vec(),
                hold,
            }));
        }
        let began = Instant::now();
        let answer = (self.objective.borrow_mut())(&x);
        self.evaluating.set(self.evaluating.get() + began.elapsed());
        self.evaluations.set(self.evaluations.get() + 1);
        let (value, mut gradient) = match answer {
            Ok(evaluation) => evaluation,
            Err(e) => {
                *self.failure.borrow_mut() = Some(e);
                return Err(Error::msg("the objective failed"));
            }
        };
        if let Some(s) = self.scale {
            for (g, s) in gradient.iter_mut().zip(s) {
                *g *= s;
            }
        }
        let evaluation = (value, gradient);
        *self.last.borrow_mut() = Some((y.to_vec(), evaluation.clone()));
        if self.searching.get().is_some() {
            let mut lowest = self.lowest.borrow_mut();
            if lowest.as_ref().is_none_or(|(_, (f, _))| value < *f) {
                *lowest = Some((y.to_vec(), evaluation.clone()));
            }
        }
        Ok(evaluation)
    }

    /// Counts the evaluations of a line search from now on.
    fn begin_search(&self, from: &[f64]) {
        *self.search_start.borrow_mut() = Some(from.to_vec());
        self.searching.set(Some(0));
        *self.last_rejected.borrow_mut() = None;
        *self.lowest.borrow_mut() = None;
    }

    /// Ends the line search; returns its lowest trial, and its last
    /// rejected trial.
    fn end_search(&self) -> (Option<Trial>, Option<Vec<f64>>) {
        self.searching.set(None);
        (
            self.lowest.borrow_mut().take(),
            self.last_rejected.borrow_mut().take(),
        )
    }
}

/// `Scaled` as argmin's problem.
struct Operator<'s, 'a, E>(&'s Scaled<'a, E>);

impl<E> CostFunction for Operator<'_, '_, E> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, y: &Vec<f64>) -> Result<f64, Error> {
        Ok(self.0.evaluate(y)?.0)
    }
}

impl<E> Gradient for Operator<'_, '_, E> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, y: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(self.0.evaluate(y)?.1)
    }
}

/// What the burst has reached, kept outside the executor, which drops its
/// state when it ends with an error.
struct Record<E> {
    /// The last accepted y.
    point: Vec<f64>,
    energy: f64,
    /// The gradient there, by y.
    gradient: Vec<f64>,
    /// Its largest component.
    largest: f64,
    /// The last accepted iteration's relative energy decrease.
    decrease: f64,
    iterations: u64,
    ending: Option<Ending>,
    /// `stop`'s error, which ends the burst.
    failure: Option<E>,
}

impl<E> Record<E> {
    fn accept(&mut self, state: &Iterate) {
        if let Some(param) = state.get_param() {
            self.point.clone_from(param);
        }
        if let Some(gradient) = state.get_gradient() {
            self.gradient.clone_from(gradient);
        }
        self.largest = self.gradient.iter().fold(0.0, |m: f64, g| m.max(g.abs()));
        self.energy = state.get_cost();
    }

    /// `state` back at the last accepted point, to retry an iteration.
    fn restore(&self, state: Iterate) -> Iterate {
        let gradient = self.gradient.clone();
        at(state, self.point.clone(), (self.energy, gradient))
    }

    /// Whether the step from the last accepted point to `state`'s gives a
    /// correction pair that L-BFGS may keep: s·y > ε·(−g·s), with s the
    /// step, y the change of the gradient along it and g the gradient at
    /// its start, as scipy's L-BFGS-B requires.
    fn curves_up(&self, state: &Iterate) -> bool {
        let (Some(param), Some(gradient)) = (state.get_param(), state.get_gradient()) else {
            return false;
        };
        let (mut sy, mut gs) = (0.0, 0.0);
        for (((y, y0), g), g0) in param
            .iter()
            .zip(&self.point)
            .zip(gradient)
            .zip(&self.gradient)
        {
            let s = y - y0;
            sy += s * (g - g0);
            gs += s * g0;
        }
        sy > f64::EPSILON * -gs
    }

    /// Whether energy `f` at `y` lies below the last accepted point's by
    /// the line search's sufficient decrease along the step to it.
    fn sufficiently_below(&self, y: &[f64], f: f64) -> bool {
        let slope: f64 = y
            .iter()
            .zip(&self.point)
            .zip(&self.gradient)
            .map(|((y, y0), g)| g * (y - y0))
            .sum();
        f < self.energy && f <= self.energy + SUFFICIENT_DECREASE * slope
    }
}

/// `state`, running, at `y` with its energy and gradient.
fn at(mut state: Iterate, y: Vec<f64>, (f, gradient): Evaluation) -> Iterate {
    state.termination_status = TerminationStatus::NotTerminated;
    state.param = Some(y);
    state.grad = Some(gradient);
    state.cost = f;
    state
}

/// argmin's `LBFGS` with scipy's ending rules.
struct Wrapper<'r, E> {
    lbfgs: Lbfgs,
    /// The correction pairs `lbfgs` took since it last started without any.
    pairs: u64,
    settings: Settings,
    record: &'r mut Record<E>,
    stop: &'r mut dyn FnMut() -> Result<bool, E>,
}

impl<E> Wrapper<'_, E> {
    /// Accepts `next`, an iteration from energy `before`.
    fn accept(&mut self, next: Iterate, before: f64) -> Iterate {
        self.record.accept(&next);
        self.record.decrease = (before - next.cost) / before.abs().max(next.cost.abs()).max(1.0);
        self.record.iterations += 1;
        next
    }

    /// Starts `lbfgs` again without correction pairs.
    fn restart(&mut self) {
        self.lbfgs = lbfgs(false);
        self.pairs = 0;
    }

    /// Drops the pair the last step gave `lbfgs` by taking back `kept`,
    /// `lbfgs` before that step. Without pairs it starts again instead:
    /// `kept` may still start its search as the burst's first does.
    fn unpair(&mut self, kept: Lbfgs) {
        if self.pairs == 0 {
            self.restart();
        } else {
            self.lbfgs = kept;
        }
    }

    fn end(&mut self, ending: Ending) -> TerminationStatus {
        let reason = match &ending {
            Ending::Abnormal(r) => TerminationReason::SolverExit(r.clone()),
            Ending::MaxIterations => TerminationReason::MaxItersReached,
            Ending::Stopped => TerminationReason::SolverExit(ending.message()),
            Ending::Gradient | Ending::Decrease | Ending::Blocked => {
                TerminationReason::SolverConverged
            }
        };
        self.record.ending = Some(ending);
        TerminationStatus::Terminated(reason)
    }
}

impl<'s, 'a, E> Solver<Operator<'s, 'a, E>, Iterate> for Wrapper<'_, E> {
    fn name(&self) -> &str {
        "L-BFGS ending by scipy's rules"
    }

    fn init(
        &mut self,
        problem: &mut Problem<Operator<'s, 'a, E>>,
        state: Iterate,
    ) -> Result<(Iterate, Option<KV>), Error> {
        let (state, kv) = self.lbfgs.init(problem, state)?;
        self.record.accept(&state);
        Ok((state, kv))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<Operator<'s, 'a, E>>,
        mut state: Iterate,
    ) -> Result<(Iterate, Option<KV>), Error> {
        let before = self.record.energy;
        let iteration = state.iter;
        let scaled = problem.problem.as_ref().map(|o| o.0);
        loop {
            // LBFGS drops its oldest pair once the iteration count reaches
            // its memory, so it counts from its own start.
            state.iter = self.pairs;
            if let Some(s) = scaled {
                s.begin_search(&self.record.point);
            }
            let kept = self.lbfgs.clone();
            let stepped = self.lbfgs.next_iter(problem, state);
            let (lowest, rejected) = scaled.map_or((None, None), |s| {
                // A failed line search drops the problem it took.
                problem.problem.get_or_insert(Operator(s));
                s.end_search()
            });
            let (mut next, kv) = stepped?;
            next.iter = iteration;
            let failure = match next.get_termination_reason() {
                Some(TerminationReason::SolverExit(reason)) => Some(reason.clone()),
                _ if matches!(
                    next.get_cost().partial_cmp(&before),
                    Some(Ordering::Greater) | None
                ) =>
                {
                    Some("the line search ended above its start".to_string())
                }
                _ => None,
            };
            let Some(reason) = failure else {
                if self.record.curves_up(&next) {
                    self.pairs += 1;
                } else {
                    self.unpair(kept);
                }
                return Ok((self.accept(next, before), kv));
            };
            if let Some(s) = scaled {
                s.failed_searches.set(s.failed_searches.get() + 1);
            }
            if let Some((y, evaluation)) = lowest
                && self.record.sufficiently_below(&y, evaluation.0)
            {
                // Without a pair: the step did not meet the curvature
                // condition. A search that ended above its start left
                // LBFGS a pair for its end point.
                self.unpair(kept);
                return Ok((self.accept(at(next, y, evaluation), before), kv));
            }
            if self.pairs == 0 {
                // What failed the search is the check where its last
                // rejected trial would have descended, as the energy there,
                // asked outside the search, tells.
                let blocked = rejected.zip(scaled).is_some_and(|(y, s)| {
                    s.evaluate(&y)
                        .is_ok_and(|(f, _)| self.record.sufficiently_below(&y, f))
                });
                let ended = self.record.restore(next);
                let status = if blocked {
                    self.end(Ending::Blocked)
                } else {
                    self.end(Ending::Abnormal(reason))
                };
                let TerminationStatus::Terminated(why) = status else {
                    unreachable!("`end` terminates");
                };
                return Ok((ended.terminate_with(why), kv));
            }
            self.restart();
            state = self.record.restore(next);
        }
    }

    fn terminate(&mut self, _state: &Iterate) -> TerminationStatus {
        let settings = self.settings;
        let iterated = self.record.iterations > 0;
        if iterated {
            let stopped = match (self.stop)() {
                Ok(stopped) => stopped,
                Err(e) => {
                    self.record.failure = Some(e);
                    return self.end(Ending::Abnormal("stop failed".into()));
                }
            };
            if self.record.iterations >= settings.max_iterations {
                return self.end(Ending::MaxIterations);
            }
            if stopped {
                return self.end(Ending::Stopped);
            }
        }
        if self.record.largest <= settings.gtol {
            return self.end(Ending::Gradient);
        }
        if iterated && self.record.decrease <= settings.ftol {
            return self.end(Ending::Decrease);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    const SETTINGS: Settings = Settings {
        max_iterations: 50,
        ftol: 1e-9,
        gtol: 1e-10,
    };

    fn quadratic(curvatures: &[f64]) -> impl FnMut(&[f64]) -> Result<Evaluation, ()> + '_ {
        move |x| Ok(testing::quadratic(curvatures, false, x))
    }

    fn rosenbrock(x: &[f64]) -> Result<Evaluation, ()> {
        let (a, b) = (x[0], x[1]);
        let value = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
        let gradient = vec![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ];
        Ok((value, gradient))
    }

    fn never() -> impl FnMut() -> Result<bool, ()> {
        || Ok(false)
    }

    fn burst(
        objective: &mut dyn FnMut(&[f64]) -> Result<Evaluation, ()>,
        start: &[f64],
        settings: &Settings,
    ) -> Burst {
        run(objective, start, None, settings, &mut never(), None).unwrap()
    }

    #[test]
    fn a_round_quadratic_converges_on_its_gradient() {
        let b = burst(&mut quadratic(&[1.0, 1.0]), &[1.0, 2.0], &SETTINGS);

        assert_eq!(b.ending, Ending::Gradient);
        assert!(b.point.iter().all(|x| x.abs() < 1e-10));
        assert!(b.iterations >= 1);
        assert!(b.evaluations >= b.iterations);
    }

    #[test]
    fn a_start_at_the_minimum_converges_before_iterating() {
        let b = burst(&mut quadratic(&[1.0, 1.0]), &[0.0, 0.0], &SETTINGS);

        assert_eq!(b.ending, Ending::Gradient);
        assert_eq!((b.iterations, b.evaluations), (0, 1));
        assert_eq!(b.point, vec![0.0, 0.0]);
    }

    #[test]
    fn rosenbrock_reaches_its_minimum() {
        let b = burst(&mut rosenbrock, &[-1.2, 1.0], &SETTINGS);

        assert!(!b.abnormal(), "{:?}", b.ending);
        assert!(matches!(b.ending, Ending::Gradient | Ending::Decrease));
        assert!((b.point[0] - 1.0).abs() < 1e-3 && (b.point[1] - 1.0).abs() < 1e-3);
        assert!(b.iterations < SETTINGS.max_iterations);
    }

    #[test]
    fn a_small_decrease_converges() {
        // Every iteration decreases by far less than a relative tolerance of 1.
        let loose = Settings {
            ftol: 1.0,
            ..SETTINGS
        };

        let b = burst(&mut rosenbrock, &[-1.2, 1.0], &loose);

        assert_eq!(b.ending, Ending::Decrease);
        assert_eq!(b.iterations, 1);
    }

    #[test]
    fn a_burst_stops_at_its_iteration_cap_exactly() {
        let capped = Settings {
            max_iterations: 3,
            ..SETTINGS
        };

        let b = burst(&mut rosenbrock, &[-1.2, 1.0], &capped);

        assert_eq!(b.ending, Ending::MaxIterations);
        assert_eq!(b.iterations, 3);
        assert!(!b.cut());
    }

    #[test]
    fn stop_ends_the_burst_after_the_iteration_it_answers_true() {
        let mut asked = 0;
        let mut stop = || {
            asked += 1;
            Ok::<_, ()>(asked >= 2)
        };

        let b = run(
            &mut rosenbrock,
            &[-1.2, 1.0],
            None,
            &SETTINGS,
            &mut stop,
            None,
        )
        .unwrap();

        assert_eq!(b.ending, Ending::Stopped);
        assert_eq!(b.iterations, 2);
        assert!(b.cut());
    }

    #[test]
    fn stop_answering_true_at_the_cap_ends_at_the_cap() {
        let capped = Settings {
            max_iterations: 2,
            ..SETTINGS
        };
        let mut asked = 0;
        let mut stop = || {
            asked += 1;
            Ok::<_, ()>(asked >= 2)
        };

        let b = run(
            &mut rosenbrock,
            &[-1.2, 1.0],
            None,
            &capped,
            &mut stop,
            None,
        )
        .unwrap();

        assert_eq!(b.ending, Ending::MaxIterations);
        assert_eq!(b.iterations, 2);
        assert!(!b.cut());
    }

    fn always() -> impl FnMut() -> Result<bool, ()> {
        || Ok(true)
    }

    #[test]
    fn stop_answering_true_on_the_iteration_that_meets_ftol_cuts_the_burst() {
        let loose = Settings {
            ftol: 1.0,
            ..SETTINGS
        };

        let b = run(
            &mut rosenbrock,
            &[-1.2, 1.0],
            None,
            &loose,
            &mut always(),
            None,
        )
        .unwrap();

        assert_eq!(b.ending, Ending::Stopped);
        assert_eq!(b.iterations, 1);
        assert!(b.cut());
    }

    #[test]
    fn stop_answering_true_on_the_iteration_that_meets_gtol_cuts_the_burst() {
        let loose = Settings {
            gtol: 1.0,
            ..SETTINGS
        };
        let start = [1.0, 2.0];
        let unstopped = burst(&mut quadratic(&[1.0, 1.0]), &start, &loose);
        assert_eq!(unstopped.ending, Ending::Gradient);
        let mut asked = 0;
        let mut stop = || {
            asked += 1;
            Ok::<_, ()>(asked >= unstopped.iterations)
        };

        let b = run(
            &mut quadratic(&[1.0, 1.0]),
            &start,
            None,
            &loose,
            &mut stop,
            None,
        )
        .unwrap();

        assert_eq!(b.ending, Ending::Stopped);
        assert_eq!(b.iterations, unstopped.iterations);
        assert!(b.cut());
    }

    #[test]
    fn ftol_met_at_the_cap_ends_at_the_cap() {
        let capped = Settings {
            max_iterations: 1,
            ftol: 1.0,
            ..SETTINGS
        };

        let b = burst(&mut rosenbrock, &[-1.2, 1.0], &capped);

        assert_eq!(b.ending, Ending::MaxIterations);
        assert_eq!(b.iterations, 1);
        assert!(!b.cut());
    }

    #[test]
    fn a_gradient_below_gtol_ends_the_burst_before_stop_is_asked() {
        let mut asked = 0;
        let mut stop = || {
            asked += 1;
            Ok::<_, ()>(true)
        };

        let b = run(
            &mut quadratic(&[1.0, 1.0]),
            &[0.0, 0.0],
            None,
            &SETTINGS,
            &mut stop,
            None,
        )
        .unwrap();

        assert_eq!(b.ending, Ending::Gradient);
        assert_eq!(asked, 0);
    }

    #[test]
    fn an_error_of_stop_ends_the_burst_with_it() {
        let mut stop = || Err::<bool, _>("interrupted");
        let mut objective = |x: &[f64]| rosenbrock(x).map_err(|()| "");

        let answer = run(
            &mut objective,
            &[-1.2, 1.0],
            None,
            &SETTINGS,
            &mut stop,
            None,
        );

        assert_eq!(answer, Err("interrupted"));
    }

    #[test]
    fn an_error_of_the_objective_ends_the_burst_with_it() {
        let mut stop = || Ok::<_, &str>(false);
        let mut objective = |x: &[f64]| {
            if x[0].abs() < 0.5 {
                Err("not finite")
            } else {
                Ok((x[0] * x[0], vec![2.0 * x[0]]))
            }
        };

        let answer = run(&mut objective, &[1.0], None, &SETTINGS, &mut stop, None);

        assert_eq!(answer, Err("not finite"));
    }

    #[test]
    fn a_failed_line_search_ends_abnormally_at_the_last_accepted_point() {
        // Every step along the misleading gradient raises the energy.
        let mut uphill = |x: &[f64]| Ok(testing::quadratic(&[1.0], true, x));

        let b = burst(&mut uphill, &[1.0], &SETTINGS);

        assert!(b.abnormal(), "{:?}", b.ending);
        assert_eq!(b.point, vec![1.0]);
        assert_eq!(b.energy, 1.0);
        assert_eq!(b.iterations, 0);
        assert!(b.evaluations > 1);
        assert!(b.ending.message().starts_with("ABNORMAL: "));
    }

    #[test]
    fn a_failed_line_search_with_pairs_retries_without_them_before_ending() {
        // A valley one iteration does not minimise, whose gradient turns
        // uphill after the first iteration: every later line search runs
        // to its cap.
        let valley = [1.0, 100.0];
        let start = [1.0, 2.0];
        let once = Settings {
            max_iterations: 1,
            ..SETTINGS
        };
        let first = burst(&mut quadratic(&valley), &start, &once);
        let mut calls = 0;
        let mut turning = |x: &[f64]| {
            calls += 1;
            Ok(testing::quadratic(&valley, calls > first.evaluations, x))
        };

        let b = burst(&mut turning, &start, &SETTINGS);

        assert!(b.abnormal(), "{:?}", b.ending);
        assert!(b.iterations >= 1);
        // The failed search with pairs and its retry without them.
        let searches = b.evaluations - first.evaluations;
        assert!(searches >= 2 * MAX_LINE_SEARCH, "{searches}");
    }

    /// Falling at slope 1 up to a wall at 1 that rises as 10⁴·d²: only
    /// steps into a band 1e-4 wide meet the curvature condition.
    fn wall(x: &[f64]) -> Result<Evaluation, ()> {
        let inside = (x[0] - 1.0).max(0.0);
        Ok((
            -x[0] + STEEPNESS * inside * inside,
            vec![-1.0 + 2.0 * STEEPNESS * inside],
        ))
    }

    const STEEPNESS: f64 = 1e4;
    const WALL_MINIMUM: f64 = 1.0 + 0.5 / STEEPNESS;

    #[test]
    fn a_search_against_a_wall_takes_its_lowest_sufficient_trial() {
        let b = burst(&mut wall, &[0.5], &SETTINGS);

        assert!(!b.abnormal(), "{:?}", b.ending);
        assert!((b.point[0] - WALL_MINIMUM).abs() < 1e-9, "{:?}", b.point);
    }

    #[test]
    fn a_step_along_a_straight_stretch_keeps_every_trial_finite() {
        // The first search ends on the straight stretch, where the gradient
        // is the start's: its pair has s·y = 0, which would scale the next
        // direction by 0/0.
        let mut nonfinite = 0;
        let mut counted = |x: &[f64]| {
            if !x.iter().all(|x| x.is_finite()) {
                nonfinite += 1;
            }
            wall(x)
        };

        let b = burst(&mut counted, &[0.5], &SETTINGS);

        assert_eq!(nonfinite, 0);
        assert!(!b.abnormal(), "{:?}", b.ending);
    }

    #[test]
    fn a_search_after_a_dropped_first_pair_starts_at_a_step_of_1() {
        // Falling at slope 4 up to `wall`'s wall: the first search ends on
        // the straight stretch, where its pair is dropped. The next search,
        // along the gradient, starts at a step of 1, 4 beyond, not at one
        // of unit length as the first did.
        const SLOPE: f64 = 4.0;
        let mut steep = |x: &[f64]| -> Result<Evaluation, ()> {
            let inside = (x[0] - 1.0).max(0.0);
            Ok((
                -SLOPE * x[0] + STEEPNESS * inside * inside,
                vec![-SLOPE + 2.0 * STEEPNESS * inside],
            ))
        };
        let start = [0.5];
        let once = Settings {
            max_iterations: 1,
            ..SETTINGS
        };
        let first = burst(&mut steep, &start, &once);
        assert!(first.point[0] < 1.0, "{:?}", first.point);
        let mut trials = Vec::new();
        let mut recorded = |x: &[f64]| {
            trials.push(x[0]);
            steep(x)
        };

        burst(&mut recorded, &start, &SETTINGS);

        let next = trials[usize::try_from(first.evaluations).unwrap()];
        assert!((next - first.point[0] - SLOPE).abs() < 1e-12, "{next}");
    }

    /// Falling as -x², ever steeper, up to a wall at 1 that rises as
    /// 10⁵·d², beside a valley across it of curvature 10.
    fn concave_valley(x: &[f64]) -> Result<Evaluation, ()> {
        let inside = (x[0] - 1.0).max(0.0);
        Ok((
            -x[0] * x[0] + VALLEY_WALL * inside * inside + 10.0 * x[1] * x[1],
            vec![-2.0 * x[0] + 2.0 * VALLEY_WALL * inside, 20.0 * x[1]],
        ))
    }

    const VALLEY_WALL: f64 = 1e5;

    #[test]
    fn a_pair_that_does_not_curve_upward_is_skipped_and_the_others_kept() {
        // Keeping the pair, or dropping every pair with it, leaves the
        // burst creeping toward the wall to its cap.
        let minimum = VALLEY_WALL / (VALLEY_WALL - 1.0);

        let b = burst(&mut concave_valley, &[0.3, 0.5], &SETTINGS);

        assert!(
            matches!(b.ending, Ending::Gradient | Ending::Decrease),
            "{:?}",
            b.ending
        );
        assert!((b.point[0] - minimum).abs() < 1e-9, "{:?}", b.point);
        assert!(b.point[1].abs() < 1e-9, "{:?}", b.point);
    }

    #[test]
    fn an_energy_that_is_not_a_number_ends_abnormally() {
        let mut objective = |x: &[f64]| -> Result<Evaluation, ()> {
            let value = if x[0] < 0.5 { f64::NAN } else { x[0] * x[0] };
            Ok((value, vec![2.0 * x[0]]))
        };

        let b = burst(&mut objective, &[1.0], &SETTINGS);

        assert!(b.abnormal(), "{:?}", b.ending);
        assert!(b.point[0] >= 0.5);
    }

    #[test]
    fn a_scaled_burst_minimises_over_y_and_returns_x() {
        // Scaled by 1/sqrt of its curvatures, the valley is round: the
        // first line search, along the gradient, points at the minimum, and
        // the second lands on it.
        let curvatures = [1.0, 100.0];
        let scale = [1.0, 0.1];
        let capped = Settings {
            max_iterations: 2,
            ..SETTINGS
        };

        let plain = burst(&mut quadratic(&curvatures), &[1.0, 2.0], &capped);
        let scaled = run(
            &mut quadratic(&curvatures),
            &[1.0, 2.0],
            Some(&scale),
            &capped,
            &mut never(),
            None,
        )
        .unwrap();

        assert!(plain.point.iter().any(|x| x.abs() > 1e-6));
        assert!(
            scaled.point.iter().all(|x| x.abs() < 1e-6),
            "{:?}",
            scaled.point
        );
    }

    #[test]
    fn evaluations_count_the_objective_s_calls() {
        let mut calls = 0;
        let mut objective = |x: &[f64]| {
            calls += 1;
            rosenbrock(x)
        };

        let b = burst(&mut objective, &[-1.2, 1.0], &SETTINGS);

        assert_eq!(b.evaluations, calls);
    }

    /// Enters a clearance beyond x[0] = WALL, holding coordinate 0 there.
    fn below_wall(x: &[f64]) -> Option<Vec<bool>> {
        (x[0] > WALL).then(|| {
            let mut hold = vec![false; x.len()];
            hold[0] = true;
            hold
        })
    }

    const WALL: f64 = 1.0;

    /// Falling along x[0] without end, and a bowl of curvature 1 in x[1]
    /// about 1.
    fn falling(x: &[f64]) -> Result<Evaluation, ()> {
        let value = -x[0] + 0.5 * (x[1] - 1.0) * (x[1] - 1.0);
        Ok((value, vec![-1.0, x[1] - 1.0]))
    }

    fn admitted_burst(
        objective: &mut dyn FnMut(&[f64]) -> Result<Evaluation, ()>,
        start: &[f64],
    ) -> Burst {
        let mut entered = below_wall;
        run(
            objective,
            start,
            None,
            &SETTINGS,
            &mut never(),
            Some(&mut entered),
        )
        .unwrap()
    }

    #[test]
    fn a_burst_takes_no_trial_its_check_rejects_and_stops_short_of_it() {
        let mut evaluated = Vec::new();
        let mut recorded = |x: &[f64]| {
            evaluated.push(x[0]);
            falling(x)
        };

        let b = admitted_burst(&mut recorded, &[0.0, 0.0]);

        assert!(!b.abnormal(), "{:?}", b.ending);
        assert!(b.rejected > 0);
        assert!(evaluated.iter().all(|&x| x <= WALL), "{evaluated:?}");
        assert!(
            b.point[0] <= WALL && b.point[0] > 0.9 * WALL,
            "{:?}",
            b.point
        );
        assert!((b.point[1] - 1.0).abs() < 1e-6, "{:?}", b.point);
    }

    #[test]
    fn a_coordinate_against_the_wall_is_held_and_the_rest_move_on() {
        // x[0] starts on the wall, which every step along the gradient
        // crosses at once; x[1] reaches its bowl's minimum all the same.
        let b = admitted_burst(&mut falling, &[WALL, 0.0]);

        assert!(!b.abnormal(), "{:?}", b.ending);
        assert_eq!(b.point[0], WALL);
        assert!((b.point[1] - 1.0).abs() < 1e-6, "{:?}", b.point);
    }

    #[test]
    fn a_burst_whose_every_descent_the_wall_blocks_ends_normally_where_it_is() {
        let mut along = |x: &[f64]| -> Result<Evaluation, ()> { Ok((-x[0], vec![-1.0])) };

        let b = admitted_burst(&mut along, &[WALL]);

        assert_eq!(b.ending, Ending::Blocked);
        assert!(!b.abnormal());
        assert_eq!(b.point, vec![WALL]);
        assert_eq!(b.iterations, 0);
    }

    /// Falling as -SLOPE·x from 0, which every step leaves; `rejects` holds
    /// nothing.
    const SLOPE: f64 = 1e6;

    #[test]
    fn a_bound_below_the_smallest_step_ends_the_search_on_its_rejection() {
        // The first step, of unit length, is 1/SLOPE: halving it falls below
        // More-Thuente's smallest step long before MAX_LINE_SEARCH trials.
        let mut steep = |x: &[f64]| -> Result<Evaluation, ()> { Ok((-SLOPE * x[0], vec![-SLOPE])) };
        let mut rejects = |x: &[f64]| (x[0] > 0.0).then(|| vec![false]);

        let b = run(
            &mut steep,
            &[0.0],
            None,
            &SETTINGS,
            &mut never(),
            Some(&mut rejects),
        )
        .unwrap();

        assert_eq!(b.ending, Ending::Blocked);
        assert!(b.rejected < MAX_LINE_SEARCH, "{}", b.rejected);
        assert_eq!(b.point, vec![0.0]);
    }

    #[test]
    fn holding_what_leaves_no_descent_bisects_instead() {
        // Coordinate 0 is the only one that descends; holding it leaves a
        // direction of slope 0, so the search bisects along the gradient,
        // rejecting again.
        let mut along = |x: &[f64]| -> Result<Evaluation, ()> { Ok((-x[0], vec![-1.0, 0.0])) };

        let b = admitted_burst(&mut along, &[WALL, 0.0]);

        assert_eq!(b.ending, Ending::Blocked);
        assert!(b.rejected > 1, "{}", b.rejected);
        assert_eq!(b.point, vec![WALL, 0.0]);
    }

    #[test]
    fn a_search_that_rejected_a_trial_but_failed_otherwise_ends_abnormally() {
        // Every step along the misleading gradient raises the energy; the
        // first trial lies beyond the wall and is rejected, the search then
        // fails on admitted trials above its start, and the rejected one
        // would not have descended either.
        let mut uphill = |x: &[f64]| Ok(testing::quadratic(&[1.0], true, x));
        let mut rejects = |x: &[f64]| (x[0] > 1.5).then(|| vec![false]);

        let b = run(
            &mut uphill,
            &[1.0],
            None,
            &SETTINGS,
            &mut never(),
            Some(&mut rejects),
        )
        .unwrap();

        assert!(b.abnormal(), "{:?}", b.ending);
        assert!(b.rejected > 0);
    }
}
