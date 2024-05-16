pub mod testing;

use polars::error::{ErrString, PolarsResult as Result};
use polars::prelude::*;
pub fn extract_numeric(a: &AnyValue) -> Result<f64> {
    match a {
        AnyValue::Int8(i) => Ok(*i as f64),
        AnyValue::Int16(i) => Ok(*i as f64),
        AnyValue::Int32(i) => Ok(*i as f64),
        AnyValue::Int64(i) => Ok(*i as f64),
        AnyValue::UInt8(i) => Ok(*i as f64),
        AnyValue::UInt16(i) => Ok(*i as f64),
        AnyValue::UInt32(i) => Ok(*i as f64),
        AnyValue::UInt64(i) => Ok(*i as f64),
        AnyValue::Float32(i) => Ok(*i as f64),
        AnyValue::Float64(i) => Ok(*i as f64),
        _ => Err(PolarsError::SchemaMismatch(ErrString::from(format!(
            "Value was not numeric"
        )))),
    }
}
