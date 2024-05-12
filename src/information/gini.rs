use polars::error::PolarsResult as Result;
use polars::prelude::*;

pub fn gini_impurity(data: LazyFrame, target_name: &str) -> Result<f64> {
    let lazy_counts = data
        .group_by([target_name])
        .agg([col("*").count().alias("counts")]);
    let lazy_probs = lazy_counts
        .with_column(col("counts").sum().alias("total"))
        .with_column(col("counts") / col("total").alias("prob"))
        .with_column(col("prob") * col("prob").alias("prob_sq"));
    let computed_probs = lazy_probs.select([col("prob_sq")]).sum()?.collect()?;
    let val = computed_probs.get(0).unwrap().get(0).unwrap().clone();

    let almost_ans = match val {
        AnyValue::Float32(f) => f as f64,
        AnyValue::Float64(f) => f,
        _ => unreachable!(),
    };
    Ok(1.0 - almost_ans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        todo!();
    }
}