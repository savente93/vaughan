// use polars::error::PolarsResult as Result;
// use polars::prelude::*;

// // pub fn rsme(prediction: Series, truth: Series) -> Result<f64> {
// //     let df = prediction
// //         .with_name("prediction")
// //         .into_frame()
// //         .with_column(truth.with_name("truth"))?;
// //     let err = df
// //         .select(&[(col("prediction") - col("truth")).alias("err")])?
// //         .select(&[(col("err") * col("err")).sqrt().alias("err_sqrt")])?;
// //     df([col("err_sqrt").mean()])?.get(0)?.first()
// // }

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
