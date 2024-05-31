use polars::error::PolarsResult as Result;
use polars::prelude::*;

use crate::utils::extract_numeric;

/// Calulate the Gini Impurity which is defined as
/// $$Gini(\mathcal{X}) = 1 - \sum_{x\in\mathcal{X}} \mathbb{P}(x)^2$$
/// \cite{gini}
pub fn gini_impurity(data: LazyFrame, target_name: &str) -> Result<f64> {
    let col_name = data.schema()?.iter_names().next().unwrap().clone();

    let lazy_counts = data
        .group_by([target_name])
        .agg([col(&col_name).count().alias("counts")]);

    let lazy_probs = lazy_counts
        .with_column(col("counts").alias("total").sum())
        .with_column((cast(col("counts"), DataType::Float64) / col("total")).alias("prob"))
        .with_column(col("prob").alias("prob_sq") * col("prob"));
    let computed_probs = lazy_probs.select([col("prob_sq")]).sum()?.collect()?;

    let val = computed_probs.get(0).unwrap().first().unwrap().clone();

    let almost_ans = extract_numeric(&val)?;
    Ok(1.0 - almost_ans)
}

#[cfg(test)]
mod tests {
    use crate::assert_eq_fl;

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
