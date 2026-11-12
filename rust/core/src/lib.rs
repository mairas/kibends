//! kibends' optimizer core in plain Rust, with no Python types: the
//! `kibends-py` crate wraps it as the extension module `kibends._core`.

pub mod burst;
pub mod contacts;
pub mod energy;
pub mod field;
pub mod geometry;
pub mod members;
pub mod neighbours;
pub mod numeric;
pub mod optimize;
pub mod pair_blocks;
pub mod pairs;
pub mod parallel;
pub mod polylines;
pub mod results;
pub mod rules;
pub mod seen;
pub mod simplify;
pub mod stats;
pub mod testing;
pub mod vias;

/// The crate's version, which is kibends' version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
