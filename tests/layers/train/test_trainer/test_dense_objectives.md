# tests/layers/train/test_trainer/test_dense_objectives.cpp


Suites: `Medium_DenseObjectives`.

A `Linear` over three features is trained through `Train::Trainer` on a batch of four with plain
SGD. The inputs and the weights follow fixed integer patterns. The value cases use a rate of 0, so
the logits the step computed can be read back and the loss recomputed on the host from them. Each
case states the loss alone: the metrics of these objectives are pinned beside their objectives, in
`test_binary_cross_entropy_objective.cpp`, `test_multi_label_objective.cpp` and
`test_mean_squared_error_objective.cpp`. The trainer trains on a device, so the suite stays
`Medium_`.

A loss is compared within a count of rounded operations of float: one per element of the sum, and
the roundings of one term (6 for the logistic term, 2 for the squared one). The logistic and the
squared terms are not negative, so the relative bound is the bound of the sum.

## TheBinaryLossEqualsTheHostComputation

`BinaryCrossEntropyObjective` over one output. The loss equals the mean of `max(x, 0) - x y + log(1
+ exp(-|x|))` on the host.

## ALargeLogitGivesAFiniteBinaryLoss

The inputs are scaled by 400, so the largest logit exceeds 100 in magnitude and the exponential of
it overflows float32. The loss is finite and equals the host value.

## TheMultiLabelLossEqualsTheHostComputation

`MultiLabelObjective` over three classes. The loss over the twelve logits equals the host value.

## TheSquaredErrorEqualsTheHostComputation

`MeanSquaredErrorObjective` over two outputs. The loss equals the sum of squared differences over 8.

## FitLowersTheSquaredError

`fit` runs 40 epochs over a dataset of one batch at a rate of 0.05. The report holds 40 steps and
the last loss is below the first. That the step compiles once is stated in `test_compile_once.cpp`.

## TheFitReportCarriesTheRegressionMetric

After the same fit, `mean_absolute_error` is among the last metrics and in the report's text.

The refusal of a dataset whose `labels` role has another type is held by
`test_role_match.cpp`; that the trainer takes no step after it is an add recorded in the report.
