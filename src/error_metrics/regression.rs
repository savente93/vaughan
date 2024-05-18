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

pub fn max_error(data: LazyFrame) -> Result<f64> {
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
            max_error(compute_error(
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
    // assert_almost_equal(
    //     mean_squared_log_error(y_true, y_pred),
    //     mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)),
    // )
    // assert_almost_equal(mean_pinball_loss(y_true, y_pred), 0.5)
    // assert_almost_equal(mean_pinball_loss(y_true, y_pred_2), 0.5)
    // assert_almost_equal(mean_pinball_loss(y_true, y_pred, alpha=0.4), 0.6)
    // assert_almost_equal(mean_pinball_loss(y_true, y_pred_2, alpha=0.4), 0.4)
    // mape = mean_absolute_percentage_error(y_true, y_pred)
    // assert np.isfinite(mape)
    // assert mape > 1e6
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=0),
    //     mean_squared_error(y_true, y_pred),
    // )
    // assert_almost_equal(
    //     d2_tweedie_score(y_true, y_pred, power=0), r2_score(y_true, y_pred)
    // )
    // dev_median = np.abs(y_true - np.median(y_true)).sum()
    // assert_array_almost_equal(
    //     d2_absolute_error_score(y_true, y_pred),
    //     1 - np.abs(y_true - y_pred).sum() / dev_median,
    // )
    // alpha = 0.2
    // pinball_loss = lambda y_true, y_pred, alpha: alpha * np.maximum(
    //     y_true - y_pred, 0
    // ) + (1 - alpha) * np.maximum(y_pred - y_true, 0)
    // y_quantile = np.percentile(y_true, q=alpha * 100)
    // assert_almost_equal(
    //     d2_pinball_score(y_true, y_pred, alpha=alpha),
    //     1
    //     - pinball_loss(y_true, y_pred, alpha).sum()
    //     / pinball_loss(y_true, y_quantile, alpha).sum(),
    // )
    // assert_almost_equal(
    //     d2_absolute_error_score(y_true, y_pred),
    //     d2_pinball_score(y_true, y_pred, alpha=0.5),
    // )

    // # Tweedie deviance needs positive y_pred, except for p=0,
    // # p>=2 needs positive y_true
    // # results evaluated by sympy
    // y_true = np.arange(1, 1 + n_samples)
    // y_pred = 2 * y_true
    // n = n_samples
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=-1),
    //     5 / 12 * n * (n**2 + 2 * n + 1),
    // )
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=1), (n + 1) * (1 - np.log(2))
    // )
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=2), 2 * np.log(2) - 1
    // )
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=3 / 2),
    //     ((6 * np.sqrt(2) - 8) / n) * np.sqrt(y_true).sum(),
    // )
    // assert_almost_equal(
    //     mean_tweedie_deviance(y_true, y_pred, power=3), np.sum(1 / y_true) / (4 * n)
    // )

    // dev_mean = 2 * np.mean(xlogy(y_true, 2 * y_true / (n + 1)))
    // assert_almost_equal(
    //     d2_tweedie_score(y_true, y_pred, power=1),
    //     1 - (n + 1) * (1 - np.log(2)) / dev_mean,
    // )

    // dev_mean = 2 * np.log((n + 1) / 2) - 2 / n * np.log(factorial(n))
    // assert_almost_equal(
    //     d2_tweedie_score(y_true, y_pred, power=2), 1 - (2 * np.log(2) - 1) / dev_mean
    // )
}
// // TODO implement
// pub fn mean_squared_log_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn root_mean_squared_log_error(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_poisson_deviance(predition: Series, truth: Series) -> Result<Series> {
//     todo!()
// }
// pub fn mean_gamma_deviance(predition: Series, truth: Series) -> Result<Series> {
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
