# tests/core/autograd/rules/test_linalg/test_gradients.cpp


Suites: `Medium_Autograd`.

Tests the backward rules of matmul, bias add, sum rows, gather, scatter, take along and put along against central differences in FLOAT64, through the `GradientProbe` of `tests/support/gradient_probe.hpp`. The method is described in [`tests/core/autograd/README.md`](../../README.md).

## MatmulWithoutTransposes

The variable is a `[4, 5]` left operand multiplied by a `[5, 3]` input with neither side transposed.
Its gradient is checked against central differences.

## MatmulWithATransposedRightOperand

A `[4, 5]` variable against the transpose of a `[3, 5]` input, the form a `Linear` layer emits. The
gradient is checked against central differences.

## MatmulWithATransposedLeftOperand

The transpose of a `[5, 4]` variable against a `[5, 3]` input. The gradient is checked against
central differences.

## MatmulWithBothOperandsTransposed

The transpose of a `[5, 4]` variable against the transpose of a `[3, 5]` input. The gradient is
checked against central differences.

## MatmulGradientOfTheRightOperand

The same node is differentiated from the other side. The variable is the weight, so the case
covers the dB half of the rule where the earlier matmul cases cover dA.

## BiasAddPassesTheMatrixGradientThrough

The variable is the `[4, 3]` matrix of a `bias_add` whose bias of three elements is an input. The
gradient is checked against central differences.

## BiasAddSumsOverTheBroadcastAxis

The same node is differentiated at the bias of three elements, with the `[4, 3]` matrix as an input.

## SumRowsBroadcastsItsGradientBack

`sum_rows` reduces a `[4, 3]` variable to three sums, and a `bias_add` onto a `[1, 3]` value of
zeros turns them into the one-row batch the loss reads. The gradient at the variable is checked
against central differences.

## GatherRowsScatterAddsARepeatedIndexIntoItsTable

The variable is a `[5, 3]` table gathered with the index `{3, 1, 3, 3, 0, 1}`. Row 3 is read three
times and row 1 twice, and rows 2 and 4 are read by no index.

## GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape

A `[4, 3]` table is gathered with a `[2, 3]` index into `[2, 3, 3]` and reshaped to `[6, 3]` for the
loss. The gradient at the table is checked against central differences.

## ScatterAddRowsGathersARepeatedIndexBackOntoItsRows

The variable is the `[6, 3]` rows scattered into five table rows with the index `{3, 1, 3, 3, 0,
1}`. Row 3 is written three times and row 1 twice, and rows 2 and 4 are written by no index.

## ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape

`[2, 3, 3]` rows over a `[2, 3]` index sum into a `[4, 3]` table. The gradient at the rows is
checked against central differences.

## WholeTwoLayerModelUnderRandomDirections

This is the end-to-end check. A full Jacobian over every weight would take 2n device runs.
Random unit directions take 2k runs whatever n is, and the chance that a wrong gradient agrees
along all sixteen of them is negligible.

## TakeAlongPutsItsGradientAtThePickedColumn

A [5, 4] value goes through `take_along` with the index 0, 3, 3, 1, 2, so two rows pick column 3.
The gradient lands at the picked column of each row and nowhere else.

## PutAlongTakesItsGradientFromThePutColumn

A [5] value goes through `put_along` into four columns with the same index. The gradient is the
gradient of the result at each row's put column.
