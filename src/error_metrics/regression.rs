use polars::error::PolarsResult as Result;
use polars::prelude::*;

pub fn rsme(prediction: Series, truth: Series) -> Result<f64> {
    let df = prediction
        .with_name("prediction")
        .into_frame()
        .with_column(truth.with_name("truth"))?;
    let err = df
        .select([(col("prediction") - col("truth")).alias("err")])?
        .select([(col("err") * col("err")).sqrt().alias("err_sqrt")])?;
    df([col("err_sqrt").mean()])?.get(0)?.first()
}
pub fn mae(prediction: Series, truth: Series) -> Result<f64> {}
