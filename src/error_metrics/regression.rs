use polars::error::PolarsResult as Result;
use polars::lazy::dsl::when;
use polars::prelude::*;

use crate::utils::extract_numeric;

pub fn compute_error(data: LazyFrame, prediction_column: &str, truth_column: &str) -> LazyFrame {
    data.select(&[
        col("*"),
        (col(truth_column) - col(prediction_column)).alias("_err"),
    ])
}

pub fn mean_absolute_error(data: LazyFrame) -> Result<f64> {
    let d = data.select(&[col("_err").abs().mean()]).collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}
pub fn median_absolute_error(data: LazyFrame) -> Result<f64> {
    let d = data.select(&[col("_err").abs().median()]).collect()?;
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

pub fn max_absolute_error(data: LazyFrame) -> Result<f64> {
    let d = data.select(&[col("_err").abs().max()]).collect()?;
    Ok(extract_numeric(&d.get(0).unwrap()[0])?)
}

pub fn mean_absolute_percentage_error(
    data: LazyFrame,
    prediction_column: &str,
    truth_column: &str,
) -> Result<f64> {
    let d = data
        .select(&[when(col(truth_column).eq(lit(0)))
            .then(col(prediction_column))
            .otherwise(col("_err") / col(truth_column))
            .abs()
            .alias("_perc_err")
            .mean()])
        .collect()?;
    Ok(extract_numeric(&d.get(0).unwrap()[0])?)
}

pub fn r2(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let d = data
        .select(&[
            col("_err").alias("_err_sq") * col("_err"),
            col(prediction_column).alias("_res") - col(truth_column).mean(),
        ])
        .with_column(col("_res").alias("_res_sq") * col("_res"))
        .select([col("_err_sq").sum(), col("_res_sq").sum()])
        .collect()?;
    let row = d.get(0).unwrap();
    let sum_err_sq = extract_numeric(&row[0])?;
    let sum_res_sq = extract_numeric(&row[1])?;
    Ok(1.0 - (sum_err_sq / sum_res_sq))
}
pub fn explained_variance(data: LazyFrame, truth_column: &str) -> Result<f64> {
    let d = data
        .select(&[col("_err").var(0), col(truth_column).var(0)])
        .collect()?;
    let row = d.get(0).unwrap();
    let var_err = extract_numeric(&row[0])?;
    let var_truth = extract_numeric(&row[1])?;
    Ok(1.0 - (var_err / var_truth))
}

#[cfg(test)]
mod test {
    use crate::assert_eq_fl;

    use super::*;

    fn get_skl_test_predictions() -> DataFrame {
        df!("truth"=> [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
        ])
            .unwrap().lazy()
            .select([
                col("truth"),
                (col("truth") + lit(1)).alias("pred1") ,
                (col("truth") - lit(1)).alias("pred2") ,
            ]).collect().unwrap()
    }

    #[test]
    fn test_skl_mean_squared_error() -> Result<()> {
        assert_eq_fl!(
            mean_squared_error(compute_error(
                get_skl_test_predictions().lazy(),
                "truth",
                "pred1"
            ))?,
            1.0
        );
        Ok(())
    }

    #[test]
    fn test_skl_mean_absolute_error() -> Result<()> {
        assert_eq_fl!(
            mean_absolute_error(compute_error(
                get_skl_test_predictions().lazy(),
                "truth",
                "pred1"
            ))?,
            1.0
        );
        Ok(())
    }

    #[test]
    fn test_skl_median_absolute_error() -> Result<()> {
        assert_eq_fl!(
            median_absolute_error(compute_error(
                get_skl_test_predictions().lazy(),
                "truth",
                "pred1"
            ))?,
            1.0
        );
        Ok(())
    }

    #[test]
    fn test_skl_max_error() -> Result<()> {
        assert_eq_fl!(
            max_absolute_error(compute_error(
                get_skl_test_predictions().lazy(),
                "truth",
                "pred1"
            ))?,
            1.0
        );
        Ok(())
    }

    #[test]
    fn test_skl_r2() -> Result<()> {
        assert_eq_fl!(
            r2(
                compute_error(get_skl_test_predictions().lazy(), "truth", "pred1"),
                "truth",
                "pred1"
            )?,
            0.995221027479092
        );
        Ok(())
    }
    #[test]
    fn test_skl_explained_variance() -> Result<()> {
        assert_eq_fl!(
            explained_variance(
                compute_error(get_skl_test_predictions().lazy(), "truth", "pred1"),
                "truth"
            )?,
            1.0
        );
        Ok(())
    }
}
