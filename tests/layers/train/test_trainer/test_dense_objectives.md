# tests/layers/train/test_trainer/test_dense_objectives.cpp


Suites: `Integration_DenseObjectives`.

A `Linear` over three features is trained through `Train::Trainer` on a batch of four with plain
SGD. The inputs and the weights follow fixed integer patterns. The value cases use a rate of 0, so
the logits the step computed can be read back and the loss recomputed on the host from them. A
float32 loss of order one is compared within 1e-5 and a ratio of counts within 1e-12.

## TheBinaryLossAndMetricsEqualTheHostComputation

`BinaryCrossEntropyObjective` over one output. The loss equals the mean of `max(x, 0) - x y + log(1
+ exp(-|x|))` on the host. The step reports three metrics, and `accuracy`, `precision` and `recall`
equal the ratios counted from the signs of the logits. Both signs occur among the logits.

## ALargeLogitGivesAFiniteBinaryLoss

The inputs are scaled by 400, so the largest logit exceeds 100 in magnitude. The loss is finite and
equals the host value within 1e-3.

## TheMultiLabelLossEqualsTheHostComputation

`MultiLabelObjective` over three classes. The loss over the twelve logits equals the host value. The
one metric, `f1`, equals the mean over the classes of `2 hits / (2 hits + wrong)`, with zero for a
class without a hit.

## TheSquaredErrorEqualsTheHostComputation

`MeanSquaredErrorObjective` over two outputs. The loss equals the sum of squared differences over 8,
and the one metric, `mean_absolute_error`, the sum of absolute differences over 8, both within 1e-5.

## FitLowersTheSquaredError

`fit` runs 40 epochs over a dataset of one batch at a rate of 0.05. The report holds 40 steps and
one compile, the last loss is below the first, and `mean_absolute_error` is among the last metrics
and in the report's text.

## FitRefusesLabelsOfAnotherType

A binary objective is fitted to a dataset whose `labels` role is int32 class indices. `fit` is
refused as `Incompatible`, the message names 'classes', 'labels', int32 and float32, and the trainer
has taken zero steps.
