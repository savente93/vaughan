//! Bringing the SciPy stack to rust and Polars.
//!
//! `vaughan` is a library that builds an analog to SciPy on top of Polars.
//! The goal is to implement many commonly used scientific algorithms and
//! calculations for general usage.
//!
//! Polars is a very fast, lazy Dataframe library for rust with a nice,
//! spark like API. This library attempts to make it more useful
//! and a bit more applicable by implementing common calculations
//! so users dont' have to rely in NumPy or SciPy
//!
//! It uses the Polars API, and does calculations lazily whenever possible
//! to improve speed and memory usage.
//!
//! ## Usage
//! For now the usage of vaughn is still relatively simple
//! but after having implemented some procedures I plan to take another good
//! look at the API to make sure it plays together nicely with everything.
//! For now you can use vaughan like this:
//!
//! ```
//! use polars::prelude::*;
//! use vaughan::error_metrics::binary_classification::*;
//!
//! # fn example() -> PolarsResult<()> {
//!
//!    let lazy_units = df!("x" => [1,0], "y"=>[0,1])?.lazy();
//!    let euclidian = euclidian(lazy_units.clone(), "x", "y")?;
//!    assert!((euclidian - std::f64::consts::SQRT_2).abs() < 0.000001);
//!    let hamming = hamming(lazy_units.clone(), "x", "y")?;
//!    assert_eq!(hamming, 2.0);
//! # Ok(())
//! # }
//! ```
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(missing_docs)]
#![warn(absolute_paths_not_starting_with_crate)]
#![warn(missing_debug_implementations)]
#![warn(non_local_definitions)]
#![warn(rust_2021_incompatible_or_patterns)]
#![warn(single_use_lifetimes)]
#![warn(unused_lifetimes)]
#![warn(unreachable_pub)]
#![warn(unused_results)]

pub mod distance;
pub mod error_metrics;
pub mod information;
mod utils;

#[cfg(test)]
pub mod testing;

use distance::euclidian;
use distance::hamming;
use polars::prelude::*;

fn example() -> PolarsResult<()> {
    Ok(())
}
