# tests/layers/train: test_binary_cross_entropy_objective.cpp, test_multi_label_objective.cpp, test_mean_squared_error_objective.cpp, test_objective.cpp


The leaf and role each of the binary, multi-label and squared-error objectives declares, what each refuses, their feeds and their metrics. Needs an engine.

Suites: `Unit_DenseObjectiveGraphs`, `Unit_DenseObjectives`.

Tests `Train::BinaryCrossEntropyObjective`, `Train::MultiLabelObjective` and
`Train::MeanSquaredErrorObjective`: the leaf and the role each declares, what `make` and the
factories refuse, what `feed` writes and what `measure` reports. The graph cases trace a
Linear, ReLU, Linear model of 6 features over a batch of 4 and need no device. The
`Unit_DenseObjectives` fixture needs an engine, and its `bind_targets` helper binds a tensor of
7s to the `targets` leaf before a feed.

## BinaryDeclaresAColumnLeafAndRealLabels

The objective is named "binary_cross_entropy" and its threshold is 0.5. In the built training
graph the `targets` leaf has the shape [4, 1] and is `spec.targets`, and the loss has one
element. The one required target has the role `K_ROLE_LABELS`, an empty sample shape, the type
float32 and 0 classes.

## MultiLabelDeclaresOneLabelPerClass

`MultiLabelObjective::make(4, 3)` is named "multi_label" and reports 3 classes. The `targets`
leaf has the shape [4, 3] and the loss has one element. The one required target has the role
`K_ROLE_LABELS`, the sample shape [3] and the type float32.

## MeanSquaredErrorDeclaresTheValuesRole

`MeanSquaredErrorObjective::make(4, 2)` is named "mean_squared_error" and reports the width 2.
The `targets` leaf has the shape [4, 2] and the loss has one element. The one required target
has the role `K_ROLE_VALUES`, the sample shape [2] and the type float32.

## MakeRefusesWhatNoLeafCanHold

Six calls fail with `InvalidDescription`. `BinaryCrossEntropyObjective::make` refuses a batch of
0 ("at least 1"), a threshold of 1 ("above 0 and below 1") and int32 ("float32", "int32").
`MultiLabelObjective::make` refuses 0 classes and a threshold of 0, and
`MeanSquaredErrorObjective::make` refuses a width of 0.

## TheFactoriesReadTheOneOutput

`make_binary_cross_entropy` over a [4, 1] output and `{"threshold": 0.9}` gives the threshold
0.9 and the batch 4. `make_multi_label` over a [4, 5] output and `{}` gives 5 classes and the
threshold 0.5. `make_mean_squared_error` over a [4, 2] output gives the width 2.

## TheFactoriesRefuseAnotherOutputAndAStrayKey

Each failure is `InvalidDescription` and names "objective". The binary factory refuses a [4, 2]
output ("one logit per sample") and a model that publishes two outputs ("publishes 2"). The
multi-label factory refuses a [4, 3, 5] output and the regression factory an [8, 2] output,
both with "[4, columns]". The key `huber` is refused by the regression factory and the key
`weights` by the multi-label factory, which also names `'threshold'`.

## BinaryFeedCopiesLabelsOfZeroAndOne

A batch whose labels are 1, 0.25, 0, 1 fails with `Incompatible`, "element 1" and "0 or 1",
and the leaf still holds four 7s. A batch of 1, 0, 0, 1 feeds ok and the leaf holds those
values.

## MeanSquaredErrorFeedReadsTheValuesRole

A batch that carries only the labels role fails with `InvalidDescription` and a message naming
`'values'`. A batch with the values 1.5, -2, 0.25, 8 under `K_ROLE_VALUES` feeds ok and the
leaf holds them.

## BinaryMeasuresAccuracyPrecisionAndRecall

Over five rows with the logits 2, 1, -1, -3, 4 and the labels 1, 0, 1, 0, 1, `measure` adds
three metrics in the order accuracy, precision, recall, with the values 3/5, 2/3 and 2/3. An
objective made with the threshold 0.95 reports the precision 1 and the recall 1/3.

## MultiLabelMeasuresTheMeanF1OfTheClasses

Over two rows of three classes `measure` adds one metric named "f1" that is a fraction and
equals 1/3.

## MeanSquaredErrorMeasuresTheMeanAbsoluteError

Over the predictions 1, 2, 3, 4 and the values 1.5, 2, 2, 6, `measure` adds one metric named
"mean_absolute_error" that is not a fraction and equals 3.5/4.

## AMetricRefusesAnOutputOfAnotherCount

A multi-label objective of 2 rows and 4 classes given a [2, 3] output fails with `Incompatible`
and a message containing "the logits", "8 values" and "holds 6". No metric is added.
