# tests/layers/train/test_dense_losses.cpp


Suites: `Small_DenseLosses`, `Medium_DenseLossesLogistic`, `Medium_DenseLossesDeathTest`.

Tests the values `Train::emit_mean_of_elements`, `Train::emit_logistic_loss` and
`Train::emit_squared_error` add to a graph. The mean and the squared error run on
`Graph::HostEvaluator`, which has a rule for each operation they emit. The logistic loss uses
RELU, ABS and SOFTPLUS, which the evaluator has no rule for, so its rows compile the graph and run
it on the engine.

## LogisticRow

`LogisticRow` holds a shape, the logits and the targets of one row of the logistic loss.

## TheMeanIsOneColumnSummedAndScaled

Over a [4, 3] input of 1 to 12 the mean has one element and equals 6.5, within 14 rounded
operations of double.

## TheLogisticLossReadsTheLogitInItsStableForm

The loss equals the closed form, the mean of max(x, 0) - x * y + log(1 + exp(-|x|)), within 16
rounded operations of float32. The rows are four moderate logits against targets in [0, 1], and
the logits 100 and -100 against 0 and 1, where a form that took the exponential of the logit itself
would overflow float32 and this one is 100.

## TheSquaredErrorSquaresTheDifference

Over [4, 2] outputs 1 to 8 and targets 0, 2, 5, 4, 5, 6, 7, 6 the differences are 1, 0, -2, 0, 0, 0,
0, 2, so the loss has one element and equals 9/8, within 10 rounded operations of double.

## OperandsOfTwoShapesAreFatal

`emit_squared_error` over a [4, 2] and a [4, 3] value stops the process with a report
containing "[4, 2]" and "[4, 3]". It is a death test, so it starts a child process and runs as
`Medium_`.
