use polars::error::PolarsResult as Result;
use polars::prelude::*;

use crate::utils::extract_numeric;

fn compare(data: LazyFrame, prediction_column: &str, truth_column: &str) -> LazyFrame {
    data.with_columns([cast(
        col(&prediction_column)
            .eq(col(&truth_column))
            .alias("_correct"),
        DataType::Boolean,
    )])
    .with_columns(&[
        when(
            cast(col(&prediction_column), DataType::Boolean)
                .not()
                .and(cast(col(&truth_column), DataType::Boolean).not()),
        )
        .then(lit(1))
        .otherwise(lit(0))
        .alias("_tn"),
        when(
            cast(col(&prediction_column), DataType::Boolean)
                .and(cast(col(&truth_column), DataType::Boolean).not()),
        )
        .then(lit(1))
        .otherwise(lit(0))
        .alias("_fp"),
        when(
            cast(col(&prediction_column), DataType::Boolean)
                .not()
                .and(cast(col(&truth_column), DataType::Boolean)),
        )
        .then(lit(1))
        .otherwise(lit(0))
        .alias("_fn"),
        when(
            cast(col(&prediction_column), DataType::Boolean)
                .and(cast(col(&truth_column), DataType::Boolean)),
        )
        .then(lit(1))
        .otherwise(lit(0))
        .alias("_tp"),
    ])
}

pub fn matthews_correlation_coeficient(
    data: LazyFrame,
    prediction_column: &str,
    truth_column: &str,
) -> Result<f64> {
    let compared = if !data.schema()?.get_names().contains(&"_correct") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = compared
        .select(&[
            col("_tp").sum(),
            col("_fp").sum(),
            col("_fn").sum(),
            col("_tn").sum(),
        ])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    let _fn = extract_numeric(&col[2])?;
    let _tn = extract_numeric(&col[3])?;
    Ok((_tp * _tn - _fp * _fn) / ((_tp + _fp) * (_tp + _fn) * (_tn + _fp) * (_tn + _fn)).sqrt())
}

pub fn accuracy(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = compare(data, prediction_column, truth_column);
    println!("compared: {}", compared.clone().collect()?);
    let num = compared
        .select_seq(&[cast(col("_correct"), DataType::Float32)])
        .collect()?[0]
        .clone();
    Ok(num.mean().unwrap())
}

pub fn f1(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = if !data.schema()?.get_names().contains(&"_correct") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = compared
        .select(&[col("_tp").sum(), col("_fp").sum(), col("_fn").sum()])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    let _fn = extract_numeric(&col[2])?;
    Ok((2.0 * _tp) / ((2.0 * _tp) + _fp + _fn))
}
pub fn jaccard(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = if !data.schema()?.get_names().contains(&"_correct") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = compared
        .select(&[
            col("_tp").sum(),
            col("_fp").sum(),
            col("_fn").sum(),
            col("_tn").sum(),
        ])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    let _fn = extract_numeric(&col[2])?;
    let _tn = extract_numeric(&col[3])?;
    Ok((_tp + _tn) / (_tp + _tn + _fp + _fn))
}
pub fn precision(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = if !data.schema()?.get_names().contains(&"_correct") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = compared
        .select(&[col("_tp").sum(), col("_fp").sum()])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    Ok(_tp / (_tp + _fp))
}

pub fn recall(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = if !data.schema()?.get_names().contains(&"_correct") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = compared
        .select(&[col("_tp").sum(), col("_fn").sum()])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fn = extract_numeric(&col[1])?;
    Ok(_tp / (_tp + _fn))
}

#[cfg(test)]
mod test {
    use crate::{assert_eq_fl, utils::testing::iris_skl_predictions_binary};

    use super::*;

    // kindly provided by sklearn

    #[test]
    fn test_compare() -> Result<()> {
        let test = df!(
            "predictions" => [0,1,0,1],
            "truth" =>       [0,0,1,1],

        )?;
        let expected = df!(
            "predictions" => [0,1,0,1],
            "truth" =>       [0,0,1,1],
            "_correct" => [true,false,false,true],
            "_tn" => [1,0,0,0],
            "_fp" => [0,1,0,0],
            "_fn" => [0,0,1,0],
            "_tp" => [0,0,0,1],

        )?;

        let actual = compare(test.lazy(), "predictions", "truth").collect()?;
        assert!(
            &actual.equals(&expected),
            "expected: {}\nactual: {}",
            &expected,
            &actual,
        );
        Ok(())
    }
    #[test]
    fn test_skl_compare_iris_binary() -> Result<()> {
        let test = iris_skl_predictions_binary();
        let expected = df!(
            "prediction" => [0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,0],
            "truth" =>      [0,0,1,1,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0],
            "_correct" =>   [true,true,true,true,false,true,true,false,false,true,true,false,false,true,true,true,true,true,true,true,true,true,false,true,true,true,true,false,true,true,true,true,true,false,true,true,true,false,true,true,false,true,true,true,true,true,true,true,false,true],
            "_tn" =>        [1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,1],
            "_fp" =>        [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "_fn" =>        [0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0],
            "_tp" =>        [0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,0],

        )?;

        let actual = compare(test.lazy(), "prediction", "truth").collect()?;
        assert!(
            &actual.equals(&expected),
            "expected: {}\nactual: {}",
            &expected,
            &actual
        );
        Ok(())
    }
    #[test]
    fn test_skl_simple_binary_accuracy() -> Result<()> {
        let test = df!(
            "predictions" => [0,1,1,1,0,1],
            "truth" =>       [1,0,0,1,0,1]

        )?;
        assert_eq_fl!(accuracy(test.clone().lazy(), "predictions", "truth")?, 0.5);
        assert_eq_fl!(
            accuracy(test.clone().lazy(), "predictions", "predictions")?,
            1.0
        );
        assert_eq_fl!(accuracy(test.clone().lazy(), "truth", "truth")?, 1.0);
        assert_eq_fl!(
            accuracy(
                test.clone()
                    .lazy()
                    .select(&[col("predictions"), col("predictions").not().alias("truth")]),
                "predictions",
                "truth"
            )?,
            0.0
        );

        Ok(())
    }

    #[test]
    fn test_skl_iris_precision_score_binary() -> Result<()> {
        let data = iris_skl_predictions_binary().lazy();

        assert_eq_fl!(precision(data, "prediction", "truth")?, 0.85);
        Ok(())
    }

    #[test]
    fn test_skl_iris_f1_score_binary() -> Result<()> {
        let data = iris_skl_predictions_binary();
        assert_eq_fl!(f1(data.lazy(), "prediction", "truth")?, 0.7555555555555555);
        Ok(())
    }

    #[test]
    fn test_skl_iris_recall_score_binary() -> Result<()> {
        let data = iris_skl_predictions_binary();
        assert_eq_fl!(recall(data.lazy(), "prediction", "truth")?, 0.68);
        Ok(())
    }

    #[test]
    fn test_skl_precision_recall_f_binary_positive_single_class() -> Result<()> {
        let df = df!(
             "prediction" => [1,1,1,1],
             "truth" => [1,1,1,1],
        )?;
        assert_eq_fl!(precision(df.clone().lazy(), "prediction", "truth")?, 1.0);
        assert_eq_fl!(recall(df.clone().lazy(), "prediction", "truth")?, 1.0);
        assert_eq_fl!(f1(df.clone().lazy(), "prediction", "truth")?, 1.0);
        Ok(())
    }

    #[test]
    fn test_skl_precision_recall_f_binary_negative_single_class() -> Result<()> {
        let df = df!(
             "prediction" => [-1,-1,-1,-1],
             "truth" => [-1,-1,-1,-1],
        )?;
        assert_eq_fl!(precision(df.clone().lazy(), "prediction", "truth")?, 1.0);
        assert_eq_fl!(recall(df.clone().lazy(), "prediction", "truth")?, 1.0);
        assert_eq_fl!(f1(df.clone().lazy(), "prediction", "truth")?, 1.0);
        Ok(())
    }

    #[test]
    fn test_skl_mcc_binary() -> Result<()> {
        assert_eq_fl!(
            matthews_correlation_coeficient(
                iris_skl_predictions_binary().lazy(),
                "prediction",
                "truth"
            )?,
            0.5715476066494083
        );
        Ok(())
    }

    #[test]
    fn test_skl_jaccard_score_validation_simple() -> Result<()> {
        let test = df!(
            "prediction" => [1, 1, 1, 1, 0, 1],
            "truth" =>      [0, 1, 1, 1, 0, 0]
        )?;
        let expected = 4.0 / 6.0;
        assert_eq_fl!(jaccard(test.lazy(), "prediction", "truth")?, expected);
        Ok(())
    }
    #[test]
    fn test_skl_jaccard_score_validation_perfect() -> Result<()> {
        let test = df!(
            "prediction" => [1, 1, 1, 1, 0, 1],
            "truth" =>      [1, 1, 1, 1, 0, 1]
        )?;
        let expected = 1.0;
        assert_eq_fl!(jaccard(test.lazy(), "prediction", "truth")?, expected);
        Ok(())
    }
    #[test]
    fn test_skl_jaccard_score_validation_terrible() -> Result<()> {
        let test = df!(
            "prediction" => [1, 1, 1, 1, 0, 1],
            "truth" =>      [0, 0, 0, 0,1, 0]
        )?;
        let expected = 0.0;
        assert_eq_fl!(jaccard(test.lazy(), "prediction", "truth")?, expected);
        Ok(())
    }
}
