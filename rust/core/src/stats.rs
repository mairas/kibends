//! What the optimizer's energy costs, counted for the whole process: the
//! energy evaluations a burst's objective makes (`problem::Objective`), the
//! times a layer's neighbour list searched for its pairs
//! (`neighbours::NeighbourList`), and the points each static field was
//! asked about (`neighbours::StaticList`), by the field.
//!
//! The counts only grow until `reset`. A field is known by its address, so
//! a field freed and another made at the same address share a count; a
//! caller comparing fields resets first and keeps them alive.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::field::Field;

static EVALUATIONS: AtomicU64 = AtomicU64::new(0);
static LIST_SEARCHES: AtomicU64 = AtomicU64::new(0);
static QUERIES: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// The counts at one moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    /// Energy evaluations of the bursts' objectives.
    pub evaluations: u64,
    /// Searches of a layer's neighbour list for its pairs.
    pub list_searches: u64,
}

/// The counts so far.
pub fn statistics() -> Statistics {
    Statistics {
        evaluations: EVALUATIONS.load(Ordering::Relaxed),
        list_searches: LIST_SEARCHES.load(Ordering::Relaxed),
    }
}

/// Every count back to zero.
pub fn reset() {
    EVALUATIONS.store(0, Ordering::Relaxed);
    LIST_SEARCHES.store(0, Ordering::Relaxed);
    *QUERIES.lock().unwrap_or_else(PoisonError::into_inner) = None;
}

pub(crate) fn evaluated() {
    EVALUATIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn searched() {
    LIST_SEARCHES.fetch_add(1, Ordering::Relaxed);
}

fn address(field: &Arc<dyn Field>) -> usize {
    Arc::as_ptr(field).cast::<()>() as usize
}

/// Counts `points` asked of `field`.
pub(crate) fn queried(field: &Arc<dyn Field>, points: usize) {
    let mut queries = QUERIES.lock().unwrap_or_else(PoisonError::into_inner);
    *queries
        .get_or_insert_with(HashMap::new)
        .entry(address(field))
        .or_default() += points as u64;
}

/// The points the energy asked `field` about since the last `reset`.
pub fn queries(field: &Arc<dyn Field>) -> u64 {
    QUERIES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .and_then(|q| q.get(&address(field)).copied())
        .unwrap_or(0)
}
