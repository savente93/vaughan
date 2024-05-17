// #[test]
// fn test_hinge_loss_multiclass() -> Result<()> {
//     Ok(())
//     //     pred_decision = np.array(
//     //         [
//     //             [+0.36, -0.17, -0.58, -0.99],
//     //             [-0.54, -0.37, -0.48, -0.58],
//     //             [-1.45, -0.58, -0.38, -0.17],
//     //             [-0.54, -0.38, -0.48, -0.58],
//     //             [-2.36, -0.79, -0.27, +0.24],
//     //             [-1.45, -0.58, -0.38, -0.17],
//     //         ]
//     //     )
//     //     y_true = np.array([0, 1, 2, 1, 3, 2])
//     //     dummy_losses = np.array(
//     //         [
//     //             1 - pred_decision[0][0] + pred_decision[0][1],
//     //             1 - pred_decision[1][1] + pred_decision[1][2],
//     //             1 - pred_decision[2][2] + pred_decision[2][3],
//     //             1 - pred_decision[3][1] + pred_decision[3][2],
//     //             1 - pred_decision[4][3] + pred_decision[4][2],
//     //             1 - pred_decision[5][2] + pred_decision[5][3],
//     //         ]
//     //     )
//     //     np.clip(dummy_losses, 0, None, out=dummy_losses)
//     //     dummy_hinge_loss = np.mean(dummy_losses)
//     //     assert hinge_loss(y_true, pred_decision) == dummy_hinge_loss

// #[test]
// fn test_precision_recall_f1_score_multiclass() -> Result<()> {
//     Ok(())
//     //     # Test Precision Recall and F1 Score for multiclass classification task
//     //     y_true, y_pred, _ = make_prediction(binary=False)

//     //     # compute scores with default labels introspection
//     //     p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
//     //     assert_array_almost_equal(p, [0.83, 0.33, 0.42], 2)
//     //     assert_array_almost_equal(r, [0.79, 0.09, 0.90], 2)
//     //     assert_array_almost_equal(f, [0.81, 0.15, 0.57], 2)
//     //     assert_array_equal(s, [24, 31, 20])

//     //     # averaging tests
//     //     ps = precision_score(y_true, y_pred, pos_label=1, average="micro")
//     //     assert_array_almost_equal(ps, 0.53, 2)

//     //     rs = recall_score(y_true, y_pred, average="micro")
//     //     assert_array_almost_equal(rs, 0.53, 2)

//     //     fs = f1_score(y_true, y_pred, average="micro")
//     //     assert_array_almost_equal(fs, 0.53, 2)

//     //     ps = precision_score(y_true, y_pred, average="macro")
//     //     assert_array_almost_equal(ps, 0.53, 2)

//     //     rs = recall_score(y_true, y_pred, average="macro")
//     //     assert_array_almost_equal(rs, 0.60, 2)

//     //     fs = f1_score(y_true, y_pred, average="macro")
//     //     assert_array_almost_equal(fs, 0.51, 2)

//     //     ps = precision_score(y_true, y_pred, average="weighted")
//     //     assert_array_almost_equal(ps, 0.51, 2)

//     //     rs = recall_score(y_true, y_pred, average="weighted")
//     //     assert_array_almost_equal(rs, 0.53, 2)

//     //     fs = f1_score(y_true, y_pred, average="weighted")
//     //     assert_array_almost_equal(fs, 0.47, 2)

//     //     with pytest.raises(ValueError):
//     //         precision_score(y_true, y_pred, average="samples")
//     //     with pytest.raises(ValueError):
//     //         recall_score(y_true, y_pred, average="samples")
//     //     with pytest.raises(ValueError):
//     //         f1_score(y_true, y_pred, average="samples")
//     //     with pytest.raises(ValueError):
//     //         fbeta_score(y_true, y_pred, average="samples", beta=0.5)

//     //     # same prediction but with and explicit label ordering
//     //     p, r, f, s = precision_recall_fscore_support(
//     //         y_true, y_pred, labels=[0, 2, 1], average=None
//     //     )
//     //     assert_array_almost_equal(p, [0.83, 0.41, 0.33], 2)
//     //     assert_array_almost_equal(r, [0.79, 0.90, 0.10], 2)
//     //     assert_array_almost_equal(f, [0.81, 0.57, 0.15], 2)
//     //     assert_array_equal(s, [24, 20, 31])
// }
// #[test]
// fn test_multiclass_jaccard_score() -> Result<()> {
//     Ok(())
//     //     y_true = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "bird"]
//     //     y_pred = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "cat"]
//     //     labels = ["ant", "bird", "cat"]
//     //     lb = LabelBinarizer()
//     //     lb.fit(labels)
//     //     y_true_bin = lb.transform(y_true)
//     //     y_pred_bin = lb.transform(y_pred)
//     //     multi_jaccard_score = partial(jaccard_score, y_true, y_pred)
//     //     bin_jaccard_score = partial(jaccard_score, y_true_bin, y_pred_bin)
//     //     multi_labels_list = [
//     //         ["ant", "bird"],
//     //         ["ant", "cat"],
//     //         ["cat", "bird"],
//     //         ["ant"],
//     //         ["bird"],
//     //         ["cat"],
//     //         None,
//     //     ]
//     //     bin_labels_list = [[0, 1], [0, 2], [2, 1], [0], [1], [2], None]

//     //     # other than average='samples'/'none-samples', test everything else here
//     //     for average in ("macro", "weighted", "micro", None):
//     //         for m_label, b_label in zip(multi_labels_list, bin_labels_list):
//     //             assert_almost_equal(
//     //                 multi_jaccard_score(average=average, labels=m_label),
//     //                 bin_jaccard_score(average=average, labels=b_label),
//     //             )

//     //     y_true = np.array([[0, 0], [0, 0], [0, 0]])
//     //     y_pred = np.array([[0, 0], [0, 0], [0, 0]])
//     //     with ignore_warnings():
//     //         assert jaccard_score(y_true, y_pred, average="weighted") == 0

//     //     assert not list(recwarn)
// }
