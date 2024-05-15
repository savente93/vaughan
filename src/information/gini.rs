use polars::chunked_array::cast;
use polars::error::PolarsResult as Result;
use polars::prelude::*;

macro_rules! assert_eq_fl {
    ($left:expr, $right:expr) => {
        let d = ($left as f64 - $right as f64).abs();
        assert!(
            d < 0.00001,
            "Expressions {} and {} differ by {}",
            $left,
            $right,
            d
        );
    };
    ($left:expr, $right:expr, $tol:expr) => {
        let d = ($left as f64 - $right as f64).abs();
        assert!(
            d < $tol,
            "Expressions {} and {} differ by {}",
            $left,
            $right,
            d
        );
    };
}

#[macro_export]
macro_rules! assert_sr_close {
    ($left:expr, $right:expr) => {
        assert_eq!($left.len(), $right.len());
        let diffs = ($left - $right).abs()?;
        let uneq = diffs.gt(0.0001)?;
        assert!(
            &uneq.filter(&uneq)?.len() == &0,
            "Expressions {:?} and {:?} differ by {:?} ",
            $left.filter(&uneq),
            $right.filter(&uneq),
            diffs.filter(&uneq)
        );
    };
    ($left:expr, $right:expr, $tol:expr) => {
        assert_eq!($left.len(), $right.len());
        let diffs = ($left - $right).abs()?;
        let uneq = diffs.gt($tol)?;
        assert!(
            &uneq.filter(&uneq)?.len() == &0,
            "Expressions {:?} and {:?} differ by {:?} ",
            $left.filter(&uneq),
            $right.filter(&uneq),
            diffs.filter(&uneq)
        );
    };
}

pub fn gini_impurity(data: LazyFrame, target_name: &str) -> Result<f64> {
    let col_name = data.schema()?.iter_names().next().unwrap().clone();

    let lazy_counts = data
        .group_by([target_name])
        .agg([col(&col_name).count().alias("counts")]);

    println!("{:?}", lazy_counts.clone().collect()?);
    let lazy_probs = lazy_counts
        .with_column(col("counts").alias("total").sum())
        .with_column((cast(col("counts"), DataType::Float64) / col("total")).alias("prob"))
        .with_column(col("prob").alias("prob_sq") * col("prob"));
    println!("{:?}", lazy_probs.clone().collect()?);
    let computed_probs = lazy_probs.select([col("prob_sq")]).sum()?.collect()?;

    println!("{}", &computed_probs);
    let val = computed_probs.get(0).unwrap().get(0).unwrap().clone();

    let almost_ans = match val {
        AnyValue::Float32(f) => f as f64,
        AnyValue::Float64(f) => f,
        AnyValue::UInt8(f) => f as f64,
        AnyValue::UInt16(f) => f as f64,
        AnyValue::UInt32(f) => f as f64,
        AnyValue::UInt64(f) => f as f64,
        AnyValue::Int8(f) => f as f64,
        AnyValue::Int16(f) => f as f64,
        AnyValue::Int32(f) => f as f64,
        AnyValue::Int64(f) => f as f64,
        _ => unreachable!(),
    };
    Ok(1.0 - almost_ans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_data_gini_impurity() -> Result<()> {
        let data = df!(
            "day" => [1,2,3,4,5,6,7], 
            "weather" => [1,0,1,0,0,1,0],
            "ate" => [1,1,0,0,0,1,0],
            "late" => [0,1,1,0,0,0,1],
            "running" => [1,0,1,0,1,1,0])?;

        let impurity = gini_impurity(data.lazy(), "running")?;

        assert_eq_fl!(impurity, 0.489796);

        Ok(())
    }
    #[test]
    fn test_pure_set_has_minimal_gini() -> Result<()> {
        let data = df!(
        "index" => [1,2,3,4,5,6,7],
        "class" => [1,1,1,1,1,1,1],
        )?;

        let impurity = gini_impurity(data.lazy(), "class")?;

        assert_eq_fl!(impurity, 0.0);

        Ok(())
    }

    #[test]
    fn test_totally_impure_set_has_maximal_gini() -> Result<()> {
        let data = df!(
        "index" => [1,2,3,4,5,6,7],
        "class" => [1,2,3,4,5,6,7],
        )?;

        let impurity = gini_impurity(data.lazy(), "class")?;

        assert_eq_fl!(impurity, 1.0 - 1.0 / 7.0);

        Ok(())
    }
}
