use polars::error::{ErrString, PolarsResult as Result};
use polars::prelude::*;

/// Extract a numeric value from a Polars::prelude::AnyValue
/// Useful to pull results out of polars data frames when necessary.
/// ```
///  # use polars::prelude::*;
///  # use vaughan::utils::extract_numeric;
///  # fn main() -> PolarsResult<()> {
///    let df = df!("x" => [0])?;
///    let any_value = df.get(0).unwrap()[0].clone();
///    assert_eq!(extract_numeric(&any_value)?, 0.0);
/// # Ok(())
/// # }
/// ```
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
        AnyValue::Float64(i) => Ok(*i),
        _ => Err(PolarsError::SchemaMismatch(ErrString::from(
            "Value was not numeric".to_string(),
        ))),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_any_value_variats_extract_numeric() -> Result<()> {
        assert!(extract_numeric(&AnyValue::Int8(2)).is_ok());
        assert!(extract_numeric(&AnyValue::Int16(2)).is_ok());
        assert!(extract_numeric(&AnyValue::Int32(2)).is_ok());
        assert!(extract_numeric(&AnyValue::Int64(2)).is_ok());
        assert!(extract_numeric(&AnyValue::UInt8(2)).is_ok());
        assert!(extract_numeric(&AnyValue::UInt16(2)).is_ok());
        assert!(extract_numeric(&AnyValue::UInt32(2)).is_ok());
        assert!(extract_numeric(&AnyValue::UInt64(2)).is_ok());
        assert!(extract_numeric(&AnyValue::Float32(2.0)).is_ok());
        assert!(extract_numeric(&AnyValue::Float64(2.0)).is_ok());

        assert!(extract_numeric(&AnyValue::Null).is_err());
        assert!(extract_numeric(&AnyValue::Boolean(true)).is_err());

        Ok(())
    }
}
