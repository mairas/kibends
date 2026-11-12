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
//! - after an iteration, when the gradient's largest component is at most
//!   `gtol`, when the energy fell by at most `ftol` relative to the larger
//!   of the two energies and 1, when `stop` answers true, or at the
//!   iteration cap, in that order, as scipy checks them. `stop` is asked
//!   after every iteration that did not converge, the last included.
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
/// `maxls`.
pub const MAX_LINE_SEARCH: u64 = 20;

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

/// The energy and its gradient at x.
pub type Evaluation = (f64, Vec<f64>);

/// The energy and its gradient as a function of x, or the error that ends
/// the burst.
pub type Objective<'a, E> = dyn FnMut(&[f64]) -> Result<Evaluation, E> + 'a;

/// Minimises `objective` from `start` over y with x = `scale` ⊙ y, or over
/// x without a scale, asking `stop` after each iteration that did not
/// converge.
pub fn run<E>(
    objective: &mut Objective<'_, E>,
    start: &[f64],
    scale: Option<&[f64]>,
    settings: &Settings,
    stop: &mut dyn FnMut() -> Result<bool, E>,
) -> Result<Burst, E> {
    if let Some(s) = scale {
        assert_eq!(s.len(), start.len(), "one scale per coordinate");
    }
    let scaled = Scaled {
        objective: RefCell::new(objective),
        scale,
        last: RefCell::new(None),
        failure: RefCell::new(None),
        evaluations: Cell::new(0),
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
    Ok(Burst {
        point,
        energy: record.energy,
        iterations: record.iterations,
        evaluations: scaled.evaluations.get(),
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
    let search = MoreThuente::new()
        .with_c(SUFFICIENT_DECREASE, CURVATURE)
        .and_then(|l| l.with_width_tolerance(WIDTH_TOLERANCE))
        .expect("scipy's line search constants are valid");
    LBFGS::new(ScipyStart { search, first }, MEMORY)
        .with_tolerance_grad(0.0)
        .and_then(|l| l.with_tolerance_cost(0.0))
        .expect("zero tolerances are valid")
}

/// More-Thuente starting a burst's first line search, along the gradient,
/// at a step of unit length, 1/|d|, as scipy's L-BFGS-B does, and every
/// later one at 1. L-BFGS sets the direction on the line search it keeps
/// and searches with a copy, so the first direction set is the first
/// search's.
#[derive(Clone)]
struct ScipyStart {
    search: MoreThuente,
    first: bool,
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
        self.search.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: Iterate,
    ) -> Result<(Iterate, Option<KV>), Error> {
        self.search.next_iter(problem, state)
    }
}

/// The objective over y, caching its last evaluation.
struct Scaled<'a, E> {
    objective: RefCell<&'a mut Objective<'a, E>>,
    scale: Option<&'a [f64]>,
    /// The last y evaluated, the energy there and its gradient by y.
    last: RefCell<Option<(Vec<f64>, Evaluation)>>,
    /// The objective's error, which ends the burst.
    failure: RefCell<Option<E>>,
    evaluations: Cell<u64>,
    /// The evaluations of the running line search; None outside one.
    searching: Cell<Option<u64>>,
    /// The running line search's lowest trial: its y, energy and gradient.
    lowest: RefCell<Option<(Vec<f64>, Evaluation)>>,
    evaluating: Cell<Duration>,
}

impl<E> Scaled<'_, E> {
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
        let x: Vec<f64> = match self.scale {
            Some(s) => y.iter().zip(s).map(|(y, s)| s * y).collect(),
            None => y.to_vec(),
        };
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
    fn begin_search(&self) {
        self.searching.set(Some(0));
        *self.lowest.borrow_mut() = None;
    }

    /// Ends the line search; returns its lowest trial.
    fn end_search(&self) -> Option<(Vec<f64>, Evaluation)> {
        self.searching.set(None);
        self.lowest.borrow_mut().take()
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
    /// The iterations since `lbfgs` last started without correction pairs.
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

    fn end(&mut self, ending: Ending) -> TerminationStatus {
        let reason = match &ending {
            Ending::Abnormal(r) => TerminationReason::SolverExit(r.clone()),
            Ending::MaxIterations => TerminationReason::MaxItersReached,
            Ending::Stopped => TerminationReason::SolverExit(ending.message()),
            Ending::Gradient | Ending::Decrease => TerminationReason::SolverConverged,
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
                s.begin_search();
            }
            let stepped = self.lbfgs.next_iter(problem, state);
            let lowest = scaled.and_then(|s| {
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
                self.pairs += 1;
                return Ok((self.accept(next, before), kv));
            };
            if let Some((y, evaluation)) = lowest
                && self.record.sufficiently_below(&y, evaluation.0)
            {
                // Without a pair: the step did not meet the curvature
                // condition.
                return Ok((self.accept(at(next, y, evaluation), before), kv));
            }
            if self.pairs == 0 {
                self.record.ending = Some(Ending::Abnormal(reason.clone()));
                let ended = self.record.restore(next);
                return Ok((
                    ended.terminate_with(TerminationReason::SolverExit(reason)),
                    kv,
                ));
            }
            self.lbfgs = lbfgs(false);
            self.pairs = 0;
            state = self.record.restore(next);
        }
    }

    fn terminate(&mut self, _state: &Iterate) -> TerminationStatus {
        let settings = self.settings;
        if self.record.largest <= settings.gtol {
            return self.end(Ending::Gradient);
        }
        if self.record.iterations == 0 {
            return TerminationStatus::NotTerminated;
        }
        if self.record.decrease <= settings.ftol {
            return self.end(Ending::Decrease);
        }
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
        run(objective, start, None, settings, &mut never()).unwrap()
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

        let b = run(&mut rosenbrock, &[-1.2, 1.0], None, &SETTINGS, &mut stop).unwrap();

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

        let b = run(&mut rosenbrock, &[-1.2, 1.0], None, &capped, &mut stop).unwrap();

        assert_eq!(b.ending, Ending::MaxIterations);
        assert_eq!(b.iterations, 2);
        assert!(!b.cut());
    }

    #[test]
    fn an_error_of_stop_ends_the_burst_with_it() {
        let mut stop = || Err::<bool, _>("interrupted");
        let mut objective = |x: &[f64]| rosenbrock(x).map_err(|()| "");

        let answer = run(&mut objective, &[-1.2, 1.0], None, &SETTINGS, &mut stop);

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

        let answer = run(&mut objective, &[1.0], None, &SETTINGS, &mut stop);

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
}
