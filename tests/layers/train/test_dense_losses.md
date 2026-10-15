# tests/layers/train/test_dense_losses.cpp


Suites: `Unit_DenseLosses`, `Unit_DenseLossesDeathTest`.

Tests the nodes `Train::emit_mean_of_elements`, `Train::emit_logistic_loss` and
`Train::emit_squared_error` add to a graph. Each case reads the operations of the whole graph in
emission order.

## TheMeanIsOneColumnSummedAndScaled

Over a [4, 3] input the mean is RESHAPE, SUM_ROWS, SCALE. The result has one element and the
reshaped value has the shape [12, 1].

## TheLogisticLossReadsTheLogitInItsStableForm

Over [4, 3] logits and targets the graph is RELU, MUL, ABS, SCALE, SOFTPLUS, SUB, ADD, RESHAPE,
SUM_ROWS, SCALE, and the loss is one float32 element. SOFTPLUS reads a SCALE whose operand is
the ABS node, and ABS reads the logits.

## TheSquaredErrorSquaresTheDifference

Over [4, 2] outputs and targets the graph is SUB, MUL, RESHAPE, SUM_ROWS, SCALE and the loss
has one element. MUL reads one value twice, and SUB reads the outputs and then the targets.

## OperandsOfTwoShapesAreFatal

`emit_squared_error` over a [4, 2] and a [4, 3] value stops the process with a report
containing "[4, 2]" and "[4, 3]".
