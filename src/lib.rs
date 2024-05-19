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
