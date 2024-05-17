use polars::error::PolarsResult as Result;
use polars::prelude::*;

use crate::utils::extract_numeric;

fn compute_error(data: LazyFrame, prediction_column: &str, truth_column: &str) -> LazyFrame {
    data.select(&[(col(prediction_column) - col(truth_column)).alias("_err")])
}

pub fn mean_absolute_error(data: LazyFrame) -> Result<f64> {
    let d = data.select(&[col("_err").abs().mean()]).collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}

pub fn mean_squared_error(data: LazyFrame) -> Result<f64> {
    let d = data
        .select(&[(col("_err") * col("_err")).mean()])
        .collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}
pub fn root_mean_squared_error(data: LazyFrame) -> Result<f64> {
    let d = data
        .select(&[(col("_err") * col("_err")).mean()])
        .collect()?;
    Ok(extract_numeric(&d.get(0).unwrap()[0])?.sqrt())
}

// pub fn mae(prediction: Series, truth: Series) -> Result<f64> {
//     todo!()
// }

// // TODO implement
// pub fn explained_variance(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn max_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_absolute_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_squared_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn root_mean_squared_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_squared_log_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn root_mean_squared_log_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn median_absolute_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn r2(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_poisson_deviance(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_gamma_deviance(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_absolute_percentage_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn d2_absolute_error_score(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn d2_pinball_score(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn d2_tweedie_score(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
