# tests/core/autograd/rules/test_linalg.cpp


Suites: `Small_CausalMaskBackward`.

Tests the nodes `Autograd::differentiate` appends for CAUSAL_MASK. The two graph cases mask
[4, 4] scores with the fill -1e9 and take the `nll_loss` of the `log_softmax` of the masked
value against a `targets` input.

## TheMaskHasARule

`Ops::K_CAUSAL_MASK` is differentiable, does not block the gradient, and
`Autograd::has_backward_rule` is true for its id.

## TheScoreGradientIsTheGradientMaskedWithZero

With the scores as a parameter, the graph holds two CAUSAL_MASK nodes after differentiation.
The first has the fill -1e9 in attribute slot 0 and the second has 0. The one output of the
second is the returned gradient, whose shape is [4, 4].

## ScoresThatNeedNoGradientAddNoNode

With the scores as an input, the returned gradient is `K_NO_VALUE` and the graph holds one
CAUSAL_MASK node.

Suites: `Small_GatherBackward`.

Tests the nodes `Autograd::differentiate` appends for GATHER_ROWS and SCATTER_ADD_ROWS. The
table has 7 rows of length 3, the index is int32, and the loss is the `nll_loss` of the
`log_softmax` of the rows against a `targets` input.

## TheGatherAndTheScatterAddEachHaveARule

`Ops::K_GATHER_ROWS` and `Ops::K_SCATTER_ADD_ROWS` are differentiable and
`Autograd::has_backward_rule` is true for both.

## TheScatteredRowsGradientIsOneGatherOverTheForwardIndex

With [5, 3] rows as a parameter scattered into 7 rows, the graph holds one GATHER_ROWS node
after differentiation. It reads a [7, 3] value and the forward index and writes the returned
gradient, whose shape is [5, 3]. The graph still holds one SCATTER_ADD_ROWS node.

## TheScatterIndexReceivesNoGradient

Asked for the rows and the index of a scatter-add, `differentiate` returns a value for the rows
and `K_NO_VALUE` for the index.

## FrozenScatteredRowsAppendNoGather

With the rows as an input and a `weight` parameter multiplied after the scatter-add,
differentiating at the weight adds no GATHER_ROWS node.

## TheTableGradientIsOneScatterAddOverTheForwardIndex

With the table as a parameter gathered by a [5] index, the graph holds one SCATTER_ADD_ROWS
node. It reads a [5, 3] value and the forward index and writes the returned gradient, a
float32 value of shape [7, 3] with the lifetime `Gradient`.

## TheIndexReceivesNoGradient

Asked for the table and the index of a gather, `differentiate` returns a value for the table
and `K_NO_VALUE` for the index, and the index does not require a gradient.

## ARankTwoIndexScattersTheGradientOfTheGathersShape

With a [2, 4] index and the gathered value reshaped to [8, 3], the one SCATTER_ADD_ROWS node
reads a [2, 4, 3] value and the forward index, and the gradient has the shape [7, 3].

## AFrozenTableAppendsNoScatterAdd

With the table as a buffer and a `weight` parameter multiplied after the gather, the weight
gets a gradient and the graph holds no SCATTER_ADD_ROWS node.

## TwoGathersOfOneTableAccumulateTheirScatterAdds

With two gathers of one table added together, the graph holds two SCATTER_ADD_ROWS nodes and
the table's gradient is written by an ACCUMULATE node.
