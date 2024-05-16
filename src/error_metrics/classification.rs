use polars::error::PolarsResult as Result;
use polars::prelude::*;

use crate::utils::extract_numeric;

pub fn compare(data: LazyFrame, prediction_column: &str, truth_column: &str) -> LazyFrame {
    data.with_columns([col(&prediction_column).eq(col(&truth_column)).alias("_cmp")])
        .with_columns(&[
            when(col("_cmp").and(col(&truth_column)))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("_tp"),
            when(col("_cmp").and(col(&truth_column).not()))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("_tn"),
            when(col("_cmp").not().and(col(&truth_column)))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("_fp"),
            when(col("_cmp").not().and(col(&truth_column).not()))
                .then(lit(1))
                .otherwise(lit(0))
                .alias("_fn"),
        ])
}

pub fn accuracy(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let compared = compare(data, prediction_column, truth_column);
    let num = compared
        .select_seq(&[cast(col("_cmp"), DataType::Float32)])
        .collect()?[0]
        .clone();
    Ok(num.mean().unwrap())
}

pub fn f1(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let data = if !data.schema()?.get_names().contains(&"_cmp") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = data
        .select(&[col("_tp").sum(), col("_fp").sum(), col("_fn")])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    let _fn = extract_numeric(&col[2])?;
    Ok(_tp / (_tp + _fp))
}
pub fn precisions(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let data = if !data.schema()?.get_names().contains(&"_cmp") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = data
        .select(&[col("_tp").sum(), col("_fp").sum()])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fp = extract_numeric(&col[1])?;
    Ok(_tp / (_tp + _fp))
}

pub fn recall(data: LazyFrame, prediction_column: &str, truth_column: &str) -> Result<f64> {
    let data = if !data.schema()?.get_names().contains(&"_cmp") {
        compare(data, prediction_column, truth_column)
    } else {
        data
    };
    let summed = data
        .select(&[col("_tp").sum(), col("_fn").sum()])
        .collect()?;
    let col = summed.get(0).unwrap();
    let _tp = extract_numeric(&col[0])?;
    let _fn = extract_numeric(&col[1])?;
    Ok(_tp / (_tp + _fn))
}

#[cfg(test)]
mod test {
    use crate::assert_eq_fl;

    use super::*;

    // kindly provided by sklearn

    #[test]
    fn test_multilabel_accuracy_score_subset_accuracy() -> Result<()> {
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
    fn test_precision_recall_f1_score_binary() -> Result<()> {
        Ok(())
        // # Test Precision Recall and F1 Score for binary classification task
        // y_true, y_pred, _ = make_prediction(binary=True)

        // # detailed measures for each class
        // p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        // assert_array_almost_equal(p, [0.73, 0.85], 2)
        // assert_array_almost_equal(r, [0.88, 0.68], 2)
        // assert_array_almost_equal(f, [0.80, 0.76], 2)
        // assert_array_equal(s, [25, 25])

        // # individual scoring function that can be used for grid search: in the
        // # binary class case the score is the value of the measure for the positive
        // # class (e.g. label == 1). This is deprecated for average != 'binary'.
        // for kwargs, my_assert in [
        //     ({}, assert_no_warnings),
        //     ({"average": "binary"}, assert_no_warnings),
        // ]:
        //     ps = my_assert(precision_score, y_true, y_pred, **kwargs)
        //     assert_array_almost_equal(ps, 0.85, 2)

        //     rs = my_assert(recall_score, y_true, y_pred, **kwargs)
        //     assert_array_almost_equal(rs, 0.68, 2)

        //     fs = my_assert(f1_score, y_true, y_pred, **kwargs)
        //     assert_array_almost_equal(fs, 0.76, 2)

        //     assert_almost_equal(
        //         my_assert(fbeta_score, y_true, y_pred, beta=2, **kwargs),
        //         (1 + 2**2) * ps * rs / (2**2 * ps + rs),
        //         2,
        //     )
    }

    #[test]
    fn test_precision_recall_f_binary_single_class() -> Result<()> {
        Ok(())
        // # Test precision, recall and F-scores behave with a single positive or
        // # negative class
        // # Such a case may occur with non-stratified cross-validation
        // assert 1.0 == precision_score([1, 1], [1, 1])
        // assert 1.0 == recall_score([1, 1], [1, 1])
        // assert 1.0 == f1_score([1, 1], [1, 1])
        // assert 1.0 == fbeta_score([1, 1], [1, 1], beta=0)

        // assert 0.0 == precision_score([-1, -1], [-1, -1])
        // assert 0.0 == recall_score([-1, -1], [-1, -1])
        // assert 0.0 == f1_score([-1, -1], [-1, -1])
        // assert 0.0 == fbeta_score([-1, -1], [-1, -1], beta=float("inf"))
        // assert fbeta_score([-1, -1], [-1, -1], beta=float("inf")) == pytest.approx(
        //     fbeta_score([-1, -1], [-1, -1], beta=1e5)
        // )
    }

    #[test]
    fn test_precision_recall_f_extra_labels() -> Result<()> {
        Ok(())
        // # Test handling of explicit additional (not in input) labels to PRF
        // y_true = [1, 3, 3, 2]
        // y_pred = [1, 1, 3, 2]
        // y_true_bin = label_binarize(y_true, classes=np.arange(5))
        // y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
        // data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

        // for i, (y_true, y_pred) in enumerate(data):
        //     # No average: zeros in array
        //     actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
        //     assert_array_almost_equal([0.0, 1.0, 1.0, 0.5, 0.0], actual)

        //     # Macro average is changed
        //     actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro")
        //     assert_array_almost_equal(np.mean([0.0, 1.0, 1.0, 0.5, 0.0]), actual)

        //     # No effect otherwise
        //     for average in ["micro", "weighted", "samples"]:
        //         if average == "samples" and i == 0:
        //             continue
        //         assert_almost_equal(
        //             recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=average),
        //             recall_score(y_true, y_pred, labels=None, average=average),
        //         )

        // # Error when introducing invalid label in multilabel case
        // # (although it would only affect performance if average='macro'/None)
        // for average in [None, "macro", "micro", "samples"]:
        //     with pytest.raises(ValueError):
        //         recall_score(y_true_bin, y_pred_bin, labels=np.arange(6), average=average)
        //     with pytest.raises(ValueError):
        //         recall_score(
        //             y_true_bin, y_pred_bin, labels=np.arange(-1, 4), average=average
        //         )

        // # tests non-regression on issue #10307
        // y_true = np.array([[0, 1, 1], [1, 0, 0]])
        // y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        // p, r, f, _ = precision_recall_fscore_support(
        //     y_true, y_pred, average="samples", labels=[0, 1]
        // )
        // assert_almost_equal(np.array([p, r, f]), np.array([3 / 4, 1, 5 / 6]))
    }
    #[test]
    fn test_precision_recall_f_ignored_labels() -> Result<()> {
        Ok(())
        // # Test a subset of labels may be requested for PRF
        // y_true = [1, 1, 2, 3]
        // y_pred = [1, 3, 3, 3]
        // y_true_bin = label_binarize(y_true, classes=np.arange(5))
        // y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
        // data = [(y_true, y_pred), (y_true_bin, y_pred_bin)]

        // for i, (y_true, y_pred) in enumerate(data):
        //     recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        //     recall_all = partial(recall_score, y_true, y_pred, labels=None)

        //     assert_array_almost_equal([0.5, 1.0], recall_13(average=None))
        //     assert_almost_equal((0.5 + 1.0) / 2, recall_13(average="macro"))
        //     assert_almost_equal((0.5 * 2 + 1.0 * 1) / 3, recall_13(average="weighted"))
        //     assert_almost_equal(2.0 / 3, recall_13(average="micro"))

        //     # ensure the above were meaningful tests:
        //     for average in ["macro", "weighted", "micro"]:
        //         assert recall_13(average=average) != recall_all(average=average)
    }

    #[test]
    fn test_average_precision_score_non_binary_class() -> Result<()> {
        Ok(())
        // """Test multiclass-multiouptut for `average_precision_score`."""
        // y_true = np.array(
        //     [
        //         [2, 2, 1],
        //         [1, 2, 0],
        //         [0, 1, 2],
        //         [1, 2, 1],
        //         [2, 0, 1],
        //         [1, 2, 1],
        //     ]
        // )
        // y_score = np.array(
        //     [
        //         [0.7, 0.2, 0.1],
        //         [0.4, 0.3, 0.3],
        //         [0.1, 0.8, 0.1],
        //         [0.2, 0.3, 0.5],
        //         [0.4, 0.4, 0.2],
        //         [0.1, 0.2, 0.7],
        //     ]
        // )
        // err_msg = "multiclass-multioutput format is not supported"
        // with pytest.raises(ValueError, match=err_msg):
        //     average_precision_score(y_true, y_score, pos_label=2)
    }

    #[test]
    fn test_average_precision_score_duplicate_values() -> Result<()> {
        Ok(())
        // @pytest.mark.parametrize(
        //     "y_true, y_score",
        //     [
        //         (
        //             [0, 0, 1, 2],
        //             np.array(
        //                 [
        //                     [0.7, 0.2, 0.1],
        //                     [0.4, 0.3, 0.3],
        //                     [0.1, 0.8, 0.1],
        //                     [0.2, 0.3, 0.5],
        //                 ]
        //             ),
        //         ),
        //         (
        //             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        //             [0, 0.1, 0.1, 0.4, 0.5, 0.6, 0.6, 0.9, 0.9, 1, 1],
        //         ),
        //     ],
        // )
        //     """
        //     Duplicate values with precision-recall require a different
        //     processing than when computing the AUC of a ROC, because the
        //     precision-recall curve is a decreasing curve
        //     The following situation corresponds to a perfect
        //     test statistic, the average_precision_score should be 1.
        //     """
        //     assert average_precision_score(y_true, y_score) == 1
    }

    #[test]
    fn test_average_precision_score_tied_values() -> Result<()> {
        Ok(())
        // @pytest.mark.parametrize(
        //     "y_true, y_score",
        //     [
        //         (
        //             [2, 2, 1, 1, 0],
        //             np.array(
        //                 [
        //                     [0.2, 0.3, 0.5],
        //                     [0.2, 0.3, 0.5],
        //                     [0.4, 0.5, 0.3],
        //                     [0.4, 0.5, 0.3],
        //                     [0.8, 0.5, 0.3],
        //                 ]
        //             ),
        //         ),
        //         (
        //             [0, 1, 1],
        //             [0.5, 0.5, 0.6],
        //         ),
        //     ],
        // )
        //     # Here if we go from left to right in y_true, the 0 values are
        //     # separated from the 1 values, so it appears that we've
        //     # correctly sorted our classifications. But in fact the first two
        //     # values have the same score (0.5) and so the first two values
        //     # could be swapped around, creating an imperfect sorting. This
        //     # imperfection should come through in the end score, making it less
        //     # than one.
        //     assert average_precision_score(y_true, y_score) != 1.0
    }

    #[test]
    fn test_precision_recall_f_unused_pos_label() -> Result<()> {
        Ok(())
        // # Check warning that pos_label unused when set to non-default value
        // # but average != 'binary'; even if data is binary.

        // msg = (
        //     r"Note that pos_label \(set to 2\) is "
        //     r"ignored when average != 'binary' \(got 'macro'\). You "
        //     r"may use labels=\[pos_label\] to specify a single "
        //     "positive class."
        // )
        // with pytest.warns(UserWarning, match=msg):
        //     precision_recall_fscore_support(
        //         [1, 2, 1], [1, 2, 2], pos_label=2, average="macro"
        //     )
    }

    #[test]
    fn test_confusion_matrix_binary() -> Result<()> {
        Ok(())
        // # Test confusion matrix - binary classification case
        // y_true, y_pred, _ = make_prediction(binary=True)

        // def test(y_true, y_pred):
        //     cm = confusion_matrix(y_true, y_pred)
        //     assert_array_equal(cm, [[22, 3], [8, 17]])

        //     tp, fp, fn, tn = cm.flatten()
        //     num = tp * tn - fp * fn
        //     den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        //     true_mcc = 0 if den == 0 else num / den
        //     mcc = matthews_corrcoef(y_true, y_pred)
        //     assert_array_almost_equal(mcc, true_mcc, decimal=2)
        //     assert_array_almost_equal(mcc, 0.57, decimal=2)

        // test(y_true, y_pred)
        // test([str(y) for y in y_true], [str(y) for y in y_pred])
    }

    #[test]
    fn test_multilabel_confusion_matrix_binary() -> Result<()> {
        Ok(())
        // # Test multilabel confusion matrix - binary classification case
        // y_true, y_pred, _ = make_prediction(binary=True)

        // def test(y_true, y_pred):
        //     cm = multilabel_confusion_matrix(y_true, y_pred)
        //     assert_array_equal(cm, [[[17, 8], [3, 22]], [[22, 3], [8, 17]]])

        // test(y_true, y_pred)
        // test([str(y) for y in y_true], [str(y) for y in y_pred])
    }

    #[test]
    fn test_multilabel_confusion_matrix_multiclass() -> Result<()> {
        Ok(())
        // # Test multilabel confusion matrix - multi-class case
        // y_true, y_pred, _ = make_prediction(binary=False)

        // def test(y_true, y_pred, string_type=False):
        //     # compute confusion matrix with default labels introspection
        //     cm = multilabel_confusion_matrix(y_true, y_pred)
        //     assert_array_equal(
        //         cm, [[[47, 4], [5, 19]], [[38, 6], [28, 3]], [[30, 25], [2, 18]]]
        //     )

        //     # compute confusion matrix with explicit label ordering
        //     labels = ["0", "2", "1"] if string_type else [0, 2, 1]
        //     cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        //     assert_array_equal(
        //         cm, [[[47, 4], [5, 19]], [[30, 25], [2, 18]], [[38, 6], [28, 3]]]
        //     )

        //     # compute confusion matrix with super set of present labels
        //     labels = ["0", "2", "1", "3"] if string_type else [0, 2, 1, 3]
        //     cm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        //     assert_array_equal(
        //         cm,
        //         [
        //             [[47, 4], [5, 19]],
        //             [[30, 25], [2, 18]],
        //             [[38, 6], [28, 3]],
        //             [[75, 0], [0, 0]],
        //         ],
        //     )

        // test(y_true, y_pred)
        // test([str(y) for y in y_true], [str(y) for y in y_pred], string_type=True)
    }

    #[test]
    fn test_multilabel_confusion_matrix_multilabel() -> Result<()> {
        Ok(())
        // # Test multilabel confusion matrix - multilabel-indicator case

        // y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        // y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        // y_true_csr = csr_container(y_true)
        // y_pred_csr = csr_container(y_pred)
        // y_true_csc = csc_container(y_true)
        // y_pred_csc = csc_container(y_pred)

        // # cross test different types
        // sample_weight = np.array([2, 1, 3])
        // real_cm = [[[1, 0], [1, 1]], [[1, 0], [1, 1]], [[0, 2], [1, 0]]]
        // trues = [y_true, y_true_csr, y_true_csc]
        // preds = [y_pred, y_pred_csr, y_pred_csc]

        // for y_true_tmp in trues:
        //     for y_pred_tmp in preds:
        //         cm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
        //         assert_array_equal(cm, real_cm)

        // # test support for samplewise
        // cm = multilabel_confusion_matrix(y_true, y_pred, samplewise=True)
        // assert_array_equal(cm, [[[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [2, 0]]])

        // # test support for labels
        // cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0])
        // assert_array_equal(cm, [[[0, 2], [1, 0]], [[1, 0], [1, 1]]])

        // # test support for labels with samplewise
        // cm = multilabel_confusion_matrix(y_true, y_pred, labels=[2, 0], samplewise=True)
        // assert_array_equal(cm, [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 1], [1, 0]]])

        // # test support for sample_weight with sample_wise
        // cm = multilabel_confusion_matrix(
        //     y_true, y_pred, sample_weight=sample_weight, samplewise=True
        // )
        // assert_array_equal(cm, [[[2, 0], [2, 2]], [[1, 1], [0, 1]], [[0, 3], [6, 0]]])
    }

    #[test]
    fn test_multilabel_confusion_matrix_errors() -> Result<()> {
        Ok(())
        // y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        // y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

        // # Bad sample_weight
        // with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        //     multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
        // with pytest.raises(ValueError, match="should be a 1d array"):
        //     multilabel_confusion_matrix(
        //         y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        //     )

        // # Bad labels
        // err_msg = r"All labels must be in \[0, n labels\)"
        // with pytest.raises(ValueError, match=err_msg):
        //     multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
        // err_msg = r"All labels must be in \[0, n labels\)"
        // with pytest.raises(ValueError, match=err_msg):
        //     multilabel_confusion_matrix(y_true, y_pred, labels=[3])

        // # Using samplewise outside multilabel
        // with pytest.raises(ValueError, match="Samplewise metrics"):
        //     multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)

        // # Bad y_type
        // err_msg = "multiclass-multioutput is not supported"
        // with pytest.raises(ValueError, match=err_msg):
        //     multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])
    }

    #[test]
    fn test_confusion_matrix_single_label() -> Result<()> {
        Ok(())
        // """Test `confusion_matrix` warns when only one label found."""
        // y_test = [0, 0, 0, 0]
        // y_pred = [0, 0, 0, 0]

        // with pytest.warns(UserWarning, match="A single label was found in"):
        //     confusion_matrix(y_pred, y_test)
    }

    #[test]
    fn test_precision_recall_f1_score_multiclass() -> Result<()> {
        Ok(())
        //     # Test Precision Recall and F1 Score for multiclass classification task
        //     y_true, y_pred, _ = make_prediction(binary=False)

        //     # compute scores with default labels introspection
        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        //     assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
        //     assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
        //     assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
        //     assert_array_equal(s, [24, 31, 20])

        //     # averaging tests
        //     ps = precision_score(y_true, y_pred, pos_label=1, average="micro")
        //     assert_array_almost_equal(ps, 0.53, 2)

        //     rs = recall_score(y_true, y_pred, average="micro")
        //     assert_array_almost_equal(rs, 0.53, 2)

        //     fs = f1_score(y_true, y_pred, average="micro")
        //     assert_array_almost_equal(fs, 0.53, 2)

        //     ps = precision_score(y_true, y_pred, average="macro")
        //     assert_array_almost_equal(ps, 0.53, 2)

        //     rs = recall_score(y_true, y_pred, average="macro")
        //     assert_array_almost_equal(rs, 0.60, 2)

        //     fs = f1_score(y_true, y_pred, average="macro")
        //     assert_array_almost_equal(fs, 0.51, 2)

        //     ps = precision_score(y_true, y_pred, average="weighted")
        //     assert_array_almost_equal(ps, 0.51, 2)

        //     rs = recall_score(y_true, y_pred, average="weighted")
        //     assert_array_almost_equal(rs, 0.53, 2)

        //     fs = f1_score(y_true, y_pred, average="weighted")
        //     assert_array_almost_equal(fs, 0.47, 2)

        //     with pytest.raises(ValueError):
        //         precision_score(y_true, y_pred, average="samples")
        //     with pytest.raises(ValueError):
        //         recall_score(y_true, y_pred, average="samples")
        //     with pytest.raises(ValueError):
        //         f1_score(y_true, y_pred, average="samples")
        //     with pytest.raises(ValueError):
        //         fbeta_score(y_true, y_pred, average="samples", beta=0.5)

        //     # same prediction but with and explicit label ordering
        //     p, r, f, s = precision_recall_fscore_support(
        //         y_true, y_pred, labels=[0, 2, 1], average=None
        //     )
        //     assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
        //     assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
        //     assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
        //     assert_array_equal(s, [24, 20, 31])
    }

    #[test]
    fn test_precision_refcall_f1_score_multilabel_unordered_labels() -> Result<()> {
        Ok(())
        // @pytest.mark.parametrize("average", ["samples", "micro", "macro", "weighted", None])
        //     # test that labels need not be sorted in the multilabel case
        //     y_true = np.array([[1, 1, 0, 0]])
        //     y_pred = np.array([[0, 0, 1, 1]])
        //     p, r, f, s = precision_recall_fscore_support(
        //         y_true, y_pred, labels=[3, 0, 1, 2], warn_for=[], average=average
        //     )
        //     assert_array_equal(p, 0)
        //     assert_array_equal(r, 0)
        //     assert_array_equal(f, 0)
        //     if average is None:
        //         assert_array_equal(s, [0, 1, 1, 0])
    }

    #[test]
    fn test_precision_recall_f1_score_binary_averaged() -> Result<()> {
        Ok(())
        //     y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        //     y_pred = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1])

        //     # compute scores with default labels introspection
        //     ps, rs, fs, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        //     p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
        //     assert p == np.mean(ps)
        //     assert r == np.mean(rs)
        //     assert f == np.mean(fs)
        //     p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        //     support = np.bincount(y_true)
        //     assert p == np.average(ps, weights=support)
        //     assert r == np.average(rs, weights=support)
        //     assert f == np.average(fs, weights=support)
    }

    #[test]
    fn test_zero_precision_recall() -> Result<()> {
        Ok(())
        //     # Check that pathological cases do not bring NaNs

        //     old_error_settings = np.seterr(all="raise")

        //     try:
        //         y_true = np.array([0, 1, 2, 0, 1, 2])
        //         y_pred = np.array([2, 0, 1, 1, 2, 0])

        //         assert_almost_equal(precision_score(y_true, y_pred, average="macro"), 0.0, 2)
        //         assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.0, 2)
        //         assert_almost_equal(f1_score(y_true, y_pred, average="macro"), 0.0, 2)

        //     finally:
        //         np.seterr(**old_error_settings)
    }

    #[test]
    fn test_confusion_matrix_multiclass_subset_labels() -> Result<()> {
        Ok(())
        //     # Test confusion matrix - multi-class case with subset of labels
        //     y_true, y_pred, _ = make_prediction(binary=False)

        //     # compute confusion matrix with only first two labels considered
        //     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        //     assert_array_equal(cm, [[19, 4], [4, 3]])

        //     # compute confusion matrix with explicit label ordering for only subset
        //     # of labels
        //     cm = confusion_matrix(y_true, y_pred, labels=[2, 1])
        //     assert_array_equal(cm, [[18, 2], [24, 3]])

        //     # a label not in y_true should result in zeros for that row/column
        //     extra_label = np.max(y_true) + 1
        //     cm = confusion_matrix(y_true, y_pred, labels=[2, extra_label])
        //     assert_array_equal(cm, [[18, 0], [0, 0]])
    }

    #[test]
    fn test_multilabel_zero_one_loss_subset() -> Result<()> {
        Ok(())
        //     # Dense label indicator matrix format
        //     y1 = np.array([[0, 1, 1], [1, 0, 1]])
        //     y2 = np.array([[0, 0, 1], [1, 0, 1]])

        //     assert zero_one_loss(y1, y2) == 0.5
        //     assert zero_one_loss(y1, y1) == 0
        //     assert zero_one_loss(y2, y2) == 0
        //     assert zero_one_loss(y2, np.logical_not(y2)) == 1
        //     assert zero_one_loss(y1, np.logical_not(y1)) == 1
        //     assert zero_one_loss(y1, np.zeros(y1.shape)) == 1
        //     assert zero_one_loss(y2, np.zeros(y1.shape)) == 1
    }
    #[test]
    fn test_multilabel_hamming_loss() -> Result<()> {
        Ok(())
        //     # Dense label indicator matrix format
        //     y1 = np.array([[0, 1, 1], [1, 0, 1]])
        //     y2 = np.array([[0, 0, 1], [1, 0, 1]])
        //     w = np.array([1, 3])

        //     assert hamming_loss(y1, y2) == 1 / 6
        //     assert hamming_loss(y1, y1) == 0
        //     assert hamming_loss(y2, y2) == 0
        //     assert hamming_loss(y2, 1 - y2) == 1
        //     assert hamming_loss(y1, 1 - y1) == 1
        //     assert hamming_loss(y1, np.zeros(y1.shape)) == 4 / 6
        //     assert hamming_loss(y2, np.zeros(y1.shape)) == 0.5
        //     assert hamming_loss(y1, y2, sample_weight=w) == 1.0 / 12
        //     assert hamming_loss(y1, 1 - y2, sample_weight=w) == 11.0 / 12
        //     assert hamming_loss(y1, np.zeros_like(y1), sample_weight=w) == 2.0 / 3
        //     # sp_hamming only works with 1-D arrays
        //     assert hamming_loss(y1[0], y2[0]) == sp_hamming(y1[0], y2[0])
    }

    #[test]
    fn test_jaccard_score_validation() -> Result<()> {
        Ok(())
        //     y_true = np.array([0, 1, 0, 1, 1])
        //     y_pred = np.array([0, 1, 0, 1, 1])
        //     err_msg = r"pos_label=2 is not a valid label. It should be one of \[0, 1\]"
        //     with pytest.raises(ValueError, match=err_msg):
        //         jaccard_score(y_true, y_pred, average="binary", pos_label=2)

        //     y_true = np.array([[0, 1, 1], [1, 0, 0]])
        //     y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        //     msg1 = (
        //         r"Target is multilabel-indicator but average='binary'. "
        //         r"Please choose another average setting, one of \[None, "
        //         r"'micro', 'macro', 'weighted', 'samples'\]."
        //     )
        //     with pytest.raises(ValueError, match=msg1):
        //         jaccard_score(y_true, y_pred, average="binary", pos_label=-1)

        //     y_true = np.array([0, 1, 1, 0, 2])
        //     y_pred = np.array([1, 1, 1, 1, 0])
        //     msg2 = (
        //         r"Target is multiclass but average='binary'. Please choose "
        //         r"another average setting, one of \[None, 'micro', 'macro', "
        //         r"'weighted'\]."
        //     )
        //     with pytest.raises(ValueError, match=msg2):
        //         jaccard_score(y_true, y_pred, average="binary")
        //     msg3 = "Samplewise metrics are not available outside of multilabel classification."
        //     with pytest.raises(ValueError, match=msg3):
        //         jaccard_score(y_true, y_pred, average="samples")

        //     msg = (
        //         r"Note that pos_label \(set to 3\) is ignored when "
        //         r"average != 'binary' \(got 'micro'\). You may use "
        //         r"labels=\[pos_label\] to specify a single positive "
        //         "class."
        //     )
        //     with pytest.warns(UserWarning, match=msg):
        //         jaccard_score(y_true, y_pred, average="micro", pos_label=3)
    }

    #[test]
    fn test_multilabel_jaccard_score() -> Result<()> {
        Ok(())
        //     # Dense label indicator matrix format
        //     y1 = np.array([[0, 1, 1], [1, 0, 1]])
        //     y2 = np.array([[0, 0, 1], [1, 0, 1]])

        //     # size(y1 \inter y2) = [1, 2]
        //     # size(y1 \union y2) = [2, 2]

        //     assert jaccard_score(y1, y2, average="samples") == 0.75
        //     assert jaccard_score(y1, y1, average="samples") == 1
        //     assert jaccard_score(y2, y2, average="samples") == 1
        //     assert jaccard_score(y2, np.logical_not(y2), average="samples") == 0
        //     assert jaccard_score(y1, np.logical_not(y1), average="samples") == 0
        //     assert jaccard_score(y1, np.zeros(y1.shape), average="samples") == 0
        //     assert jaccard_score(y2, np.zeros(y1.shape), average="samples") == 0

        //     y_true = np.array([[0, 1, 1], [1, 0, 0]])
        //     y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        //     # average='macro'
        //     assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 2.0 / 3)
        //     # average='micro'
        //     assert_almost_equal(jaccard_score(y_true, y_pred, average="micro"), 3.0 / 5)
        //     # average='samples'
        //     assert_almost_equal(jaccard_score(y_true, y_pred, average="samples"), 7.0 / 12)
        //     assert_almost_equal(
        //         jaccard_score(y_true, y_pred, average="samples", labels=[0, 2]), 1.0 / 2
        //     )
        //     assert_almost_equal(
        //         jaccard_score(y_true, y_pred, average="samples", labels=[1, 2]), 1.0 / 2
        //     )
        //     # average=None
        //     assert_array_equal(
        //         jaccard_score(y_true, y_pred, average=None), np.array([1.0 / 2, 1.0, 1.0 / 2])
        //     )

        //     y_true = np.array([[0, 1, 1], [1, 0, 1]])
        //     y_pred = np.array([[1, 1, 1], [1, 0, 1]])
        //     assert_almost_equal(jaccard_score(y_true, y_pred, average="macro"), 5.0 / 6)
        //     # average='weighted'
        //     assert_almost_equal(jaccard_score(y_true, y_pred, average="weighted"), 7.0 / 8)

        //     msg2 = "Got 4 > 2"
        //     with pytest.raises(ValueError, match=msg2):
        //         jaccard_score(y_true, y_pred, labels=[4], average="macro")
        //     msg3 = "Got -1 < 0"
        //     with pytest.raises(ValueError, match=msg3):
        //         jaccard_score(y_true, y_pred, labels=[-1], average="macro")

        //     msg = (
        //         "Jaccard is ill-defined and being set to 0.0 in labels "
        //         "with no true or predicted samples."
        //     )

        //     with pytest.warns(UndefinedMetricWarning, match=msg):
        //         assert (
        //             jaccard_score(np.array([[0, 1]]), np.array([[0, 1]]), average="macro")
        //             == 0.5
        //         )

        //     msg = (
        //         "Jaccard is ill-defined and being set to 0.0 in samples "
        //         "with no true or predicted labels."
        //     )

        //     with pytest.warns(UndefinedMetricWarning, match=msg):
        //         assert (
        //             jaccard_score(
        //                 np.array([[0, 0], [1, 1]]),
        //                 np.array([[0, 0], [1, 1]]),
        //                 average="samples",
        //             )
        //             == 0.5
        //         )

        //     assert not list(recwarn)
    }

    #[test]
    fn test_multiclass_jaccard_score() -> Result<()> {
        Ok(())
        //     y_true = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
        //     y_pred = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "cat"]
        //     labels = ["ant", "bird", "cat"]
        //     lb = LabelBinarizer()
        //     lb.fit(labels)
        //     y_true_bin = lb.transform(y_true)
        //     y_pred_bin = lb.transform(y_pred)
        //     multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
        //     bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
        //     multi_labels_list = [
        //         ["ant", "bird"],
        //         ["ant", "cat"],
        //         ["cat", "bird"],
        //         ["ant"],
        //         ["bird"],
        //         ["cat"],
        //         None,
        //     ]
        //     bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]

        //     # other than average='samples'/'none-samples', test everything else here
        //     for average in ("macro", "weighted", "micro", None):
        //         for m_label, b_label in zip(multi_labels_list, bin_labels_list):
        //             assert_almost_equal(
        //                 multi_jaccard_score(average=average, labels=m_label),
        //                 bin_jaccard_score(average=average, labels=b_label),
        //             )

        //     y_true = np.array([[0, 0], [0, 0], [0, 0]])
        //     y_pred = np.array([[0, 0], [0, 0], [0, 0]])
        //     with ignore_warnings():
        //         assert jaccard_score(y_true, y_pred, average="weighted") == 0

        //     assert not list(recwarn)
    }

    // def test_average_binary_jaccard_score(recwarn):
    //     # tp=0, fp=0, fn=1, tn=0
    //     assert jaccard_score([1], [0], average="binary") == 0.0
    //     # tp=0, fp=0, fn=0, tn=1
    //     msg = (
    //         "Jaccard is ill-defined and being set to 0.0 due to "
    //         "no true or predicted samples"
    //     )
    //     with pytest.warns(UndefinedMetricWarning, match=msg):
    //         assert jaccard_score([0, 0], [0, 0], average="binary") == 0.0

    //     # tp=1, fp=0, fn=0, tn=0 (pos_label=0)
    //     assert jaccard_score([0], [0], pos_label=0, average="binary") == 1.0
    //     y_true = np.array([1, 0, 1, 1, 0])
    //     y_pred = np.array([1, 0, 1, 1, 1])
    //     assert_almost_equal(jaccard_score(y_true, y_pred, average="binary"), 3.0 / 4)
    //     assert_almost_equal(
    //         jaccard_score(y_true, y_pred, average="binary", pos_label=0), 1.0 / 2
    //     )

    //     assert not list(recwarn)

    #[test]
    fn test_precision_recall_f1_score_multilabel_1() -> Result<()> {
        Ok(())
        //     # Test precision_recall_f1_score on a crafted multilabel example
        //     # First crafted example

        //     y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        //     y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]])

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

        //     # tp = [0, 1, 1, 0]
        //     # fn = [1, 0, 0, 1]
        //     # fp = [1, 1, 0, 0]
        //     # Check per class

        //     assert_array_almost_equal(p, [0.0, 0.5, 1.0, 0.0], 2)
        //     assert_array_almost_equal(r, [0.0, 1.0, 1.0, 0.0], 2)
        //     assert_array_almost_equal(f, [0.0, 1 / 1.5, 1, 0.0], 2)
        //     assert_array_almost_equal(s, [1, 1, 1, 1], 2)

        //     f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
        //     support = s
        //     assert_array_almost_equal(f2, [0, 0.83, 1, 0], 2)

        //     # Check macro
        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
        //     assert_almost_equal(p, 1.5 / 4)
        //     assert_almost_equal(r, 0.5)
        //     assert_almost_equal(f, 2.5 / 1.5 * 0.25)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
        //     )

        //     # Check micro
        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
        //     assert_almost_equal(p, 0.5)
        //     assert_almost_equal(r, 0.5)
        //     assert_almost_equal(f, 0.5)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="micro"),
        //         (1 + 4) * p * r / (4 * p + r),
        //     )

        //     # Check weighted
        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        //     assert_almost_equal(p, 1.5 / 4)
        //     assert_almost_equal(r, 0.5)
        //     assert_almost_equal(f, 2.5 / 1.5 * 0.25)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        //         np.average(f2, weights=support),
        //     )
        //     # Check samples
        //     # |h(x_i) inter y_i | = [0, 1, 1]
        //     # |y_i| = [1, 1, 2]
        //     # |h(x_i)| = [1, 1, 2]
        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
        //     assert_almost_equal(p, 0.5)
        //     assert_almost_equal(r, 0.5)
        //     assert_almost_equal(f, 0.5)
        //     assert s is None
        //     assert_almost_equal(fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.5)
    }

    #[test]
    fn test_precision_recall_f1_score_multilabel_2() -> Result<()> {
        Ok(())
        //     # Test precision_recall_f1_score on a crafted multilabel example 2
        //     # Second crafted example
        //     y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]])
        //     y_pred = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])

        //     # tp = [ 0.  1.  0.  0.]
        //     # fp = [ 1.  0.  0.  2.]
        //     # fn = [ 1.  1.  1.  0.]

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        //     assert_array_almost_equal(p, [0.0, 1.0, 0.0, 0.0], 2)
        //     assert_array_almost_equal(r, [0.0, 0.5, 0.0, 0.0], 2)
        //     assert_array_almost_equal(f, [0.0, 0.66, 0.0, 0.0], 2)
        //     assert_array_almost_equal(s, [1, 2, 1, 0], 2)

        //     f2 = fbeta_score(y_true, y_pred, beta=2, average=None)
        //     support = s
        //     assert_array_almost_equal(f2, [0, 0.55, 0, 0], 2)

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="micro")
        //     assert_almost_equal(p, 0.25)
        //     assert_almost_equal(r, 0.25)
        //     assert_almost_equal(f, 2 * 0.25 * 0.25 / 0.5)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="micro"),
        //         (1 + 4) * p * r / (4 * p + r),
        //     )

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="macro")
        //     assert_almost_equal(p, 0.25)
        //     assert_almost_equal(r, 0.125)
        //     assert_almost_equal(f, 2 / 12)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="macro"), np.mean(f2)
        //     )

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        //     assert_almost_equal(p, 2 / 4)
        //     assert_almost_equal(r, 1 / 4)
        //     assert_almost_equal(f, 2 / 3 * 2 / 4)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        //         np.average(f2, weights=support),
        //     )

        //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average="samples")
        //     # Check samples
        //     # |h(x_i) inter y_i | = [0, 0, 1]
        //     # |y_i| = [1, 1, 2]
        //     # |h(x_i)| = [1, 1, 2]

        //     assert_almost_equal(p, 1 / 6)
        //     assert_almost_equal(r, 1 / 6)
        //     assert_almost_equal(f, 2 / 4 * 1 / 3)
        //     assert s is None
        //     assert_almost_equal(
        //         fbeta_score(y_true, y_pred, beta=2, average="samples"), 0.1666, 2
        //     )
    }

    #[test]
    fn test_precision_recall_f1_no_labels() -> Result<()> {
        Ok(())
        //     y_true = np.zeros((20, 3))
        //     y_pred = np.zeros_like(y_true)

        //     p, r, f, s = assert_no_warnings(
        //         precision_recall_fscore_support,
        //         y_true,
        //         y_pred,
        //         average=average,
        //         beta=beta,
        //         zero_division=zero_division,
        //     )
        //     fbeta = assert_no_warnings(
        //         fbeta_score,
        //         y_true,
        //         y_pred,
        //         beta=beta,
        //         average=average,
        //         zero_division=zero_division,
        //     )
        //     assert s is None

        //     # if zero_division = nan, check that all metrics are nan and exit
        //     if np.isnan(zero_division):
        //         for metric in [p, r, f, fbeta]:
        //             assert np.isnan(metric)
        //         return

        //     zero_division = float(zero_division)
        //     assert_almost_equal(p, zero_division)
        //     assert_almost_equal(r, zero_division)
        //     assert_almost_equal(f, zero_division)

        //     assert_almost_equal(fbeta, float(zero_division))
    }

    #[test]
    fn test_hinge_loss_binary() -> Result<()> {
        Ok(())
        //     y_true = np.array([-1, 1, 1, -1])
        //     pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
        //     assert hinge_loss(y_true, pred_decision) == 1.2 / 4

        //     y_true = np.array([0, 2, 2, 0])
        //     pred_decision = np.array([-8.5, 0.5, 1.5, -0.3])
        //     assert hinge_loss(y_true, pred_decision) == 1.2 / 4
    }

    #[test]
    fn test_hinge_loss_multiclass() -> Result<()> {
        Ok(())
        //     pred_decision = np.array(
        //         [
        //             [+0.36, -0.17, -0.58, -0.99],
        //             [-0.54, -0.37, -0.48, -0.58],
        //             [-1.45, -0.58, -0.38, -0.17],
        //             [-0.54, -0.38, -0.48, -0.58],
        //             [-2.36, -0.79, -0.27, +0.24],
        //             [-1.45, -0.58, -0.38, -0.17],
        //         ]
        //     )
        //     y_true = np.array([0, 1, 2, 1, 3, 2])
        //     dummy_losses = np.array(
        //         [
        //             1 - pred_decision[0][0] + pred_decision[0][1],
        //             1 - pred_decision[1][1] + pred_decision[1][2],
        //             1 - pred_decision[2][2] + pred_decision[2][3],
        //             1 - pred_decision[3][1] + pred_decision[3][2],
        //             1 - pred_decision[4][3] + pred_decision[4][2],
        //             1 - pred_decision[5][2] + pred_decision[5][3],
        //         ]
        //     )
        //     np.clip(dummy_losses, 0, None, out=dummy_losses)
        //     dummy_hinge_loss = np.mean(dummy_losses)
        //     assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss
    }

    #[test]
    fn test_hinge_loss_multiclass_missing_labels_with_labels_none() -> Result<()> {
        Ok(())
        //     y_true = np.array([0, 1, 2, 2])
        //     pred_decision = np.array(
        //         [
        //             [+1.27, 0.034, -0.68, -1.40],
        //             [-1.45, -0.58, -0.38, -0.17],
        //             [-2.36, -0.79, -0.27, +0.24],
        //             [-2.36, -0.79, -0.27, +0.24],
        //         ]
        //     )
        //     error_message = (
        //         "Please include all labels in y_true or pass labels as third argument"
        //     )
        //     with pytest.raises(ValueError, match=error_message):
        //         hinge_loss(y_true, pred_decision)
    }
}
