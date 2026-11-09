# tests/layers/train/test_dense_losses.cpp


Suites: `Small_DenseLosses`, `Small_DenseLossesLogistic`, `Medium_DenseLossesDeathTest`.

Tests the values `Train::emit_mean_of_elements`, `Train::emit_logistic_loss` and
`Train::emit_squared_error` add to a graph. The mean and the squared error run on
`Graph::HostEvaluator`, which has a rule for each operation they emit, RELU, ABS and SOFTPLUS
included. Each count of rounded operations is derived from the graph: the host sum rounds
n - 1 times over n elements and the scale once, so the mean of n elements is n operations.

## LogisticRow

`LogisticRow` holds a shape, the logits and the targets of one row of the logistic loss.

## TheMeanIsOneColumnSummedAndScaled

Over a [4, 3] input of 1 to 12 the mean has one element and equals 6.5, within 12 rounded
operations of double: the 11 additions and the scale.

## TheLogisticLossReadsTheLogitInItsStableForm

The loss equals the closed form, the mean of max(x, 0) - x * y + log(1 + exp(-|x|)), within n + 3
rounded operations of double for n elements: three for the longest path of one element (the
exponential, the logarithm and the final sum, or the product, the difference and the final sum),
then the mean. The rows are four moderate logits against targets in [0, 1], and
the logits 100 and -100 against 0 and 1, where a form that took the exponential of the logit itself
would overflow float32 and this one is 100.

## TheSquaredErrorSquaresTheDifference

Over [4, 2] outputs 1 to 8 and targets 0, 2, 5, 4, 5, 6, 7, 6 the differences are 1, 0, -2, 0, 0, 0,
0, 2, so the loss has one element and equals 9/8, within 10 rounded operations of double: the
difference, the product, 7 additions and the scale.

## OperandsOfTwoShapesAreFatal

`emit_squared_error` over a [4, 2] and a [4, 3] value stops the process with a report
containing "[4, 2]" and "[4, 3]". It is a death test, so it starts a child process and runs as
`Medium_`.
