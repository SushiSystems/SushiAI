# tests/core/graph/lowering/test_elementwise.cpp


Suites: `Small_ElementwiseLowerings`, `Medium_ElementwiseLowerings`.

Tests `K_ELEMENTWISE_LOWERINGS`, DYNAMIC_SCALE and CAST. The other elementwise rows are compared
with host values in `tests/core/graph/test_lowering/test_hand_values.cpp` and
`tests/core/graph/test_lowering/test_host_values.cpp`. The medium cases run on an engine; scaled
values are compared within 1e-6 and converted values exactly, since float32 holds them without
rounding.

## EachRowPairsItsOperationWithItsLowering

The table has seventeen rows, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## DynamicScaleReadsItsFactorOnEveryRun

One compiled step scales a [4] input by a one-element factor. The factor is written as 2 and the
step runs, then it is written as -0.5 and the step runs again. Each run writes the input times the
factor written before it, so the row reads the factor when the step runs and not when it is
recorded.

## CastConvertsEachElementToTheResultType

`cast` of a float32 [3] input to float64 gives a float64 tensor holding the same three values.
