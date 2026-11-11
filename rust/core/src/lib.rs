//! kibends' optimizer core in plain Rust, with no Python types: the
//! `kibends-py` crate wraps it as the extension module `kibends._core`.

pub mod field;
pub mod geometry;

/// The crate's version, which is kibends' version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
