use polars::error::PolarsResult as Result;
use polars::prelude::*;

use crate::utils::extract_numeric;

pub fn entropy(data: LazyFrame, target_name: &str) -> Result<f64> {
    let col_name = data.schema()?.iter_names().next().unwrap().clone();

    // no internal log yet, so no lazy for us for now :(
    let lazy_probs = data
        .group_by([target_name])
        .agg([cast(
            col(&col_name).count().alias("counts"),
            DataType::Float64,
        )])
        .with_column((col("counts") / col("counts").sum()).alias("prob"))
        .with_column(col("prob").log(2.0).alias("prob_log"))
        .with_column(col("prob").alias("marginal_entropy") * col("prob_log"))
        .select(&[col("marginal_entropy").sum()]);

    let row = lazy_probs.collect()?;
    let almost_neg_entropy = row.get(0).unwrap();

    let neg_entropy = extract_numeric(&almost_neg_entropy[0])?;

    Ok(-1.0 * neg_entropy)
}

#[cfg(test)]
mod tests {
    use crate::assert_eq_fl;

    use super::*;

    // #[test]
    // fn test_dummy_data_entropy() -> Result<()> {
    //     let data = df!(
    //         "day" => [1,2,3,4,5,6,7],
    //         "weather" => [1,0,1,0,0,1,0],
    //         "ate" => [1,1,0,0,0,1,0],
    //         "late" => [0,1,1,0,0,0,1],
    //         "running" => [1,0,1,0,1,1,0])?;

    //     let entorpy = entropy(data.lazy(), "running")?;

    //     assert_eq_fl!(entropy, 0.489796);

    //     Ok(())
    // }
    #[test]
    fn test_pure_set_has_minimal_entropy() -> Result<()> {
        let data = df!(
        "index" => [1,2,3,4,5,6,7],
        "class" => [1,1,1,1,1,1,1],
        )?;

        let entropy = entropy(data.lazy(), "class")?;

        assert_eq_fl!(entropy, 0.0);

        Ok(())
    }

    #[test]
    fn test_arange_7_has_entropy_log_7() -> Result<()> {
        let data = df!(
        "index" => [1,2,3,4,5,6,7],
        "class" => [1,2,3,4,5,6,7],
        )?;

        let entropy = entropy(data.lazy(), "class")?;

        assert_eq_fl!(entropy, 7.0_f64.log(2.0));

        Ok(())
    }
}
