# tests/core/graph/test_builder.cpp


Suites: `Small_BatchedMatmul`.

Tests the IR that `GraphBuilder::matmul` emits for rank-2 and rank-3 operands and that
`GraphBuilder::softmax` emits over the last axis. Every refusal case also expects the graph's
node count to be what it was before the refused calls.

## Rank2ProductKeepsItsShapeAndFlags

`matmul` over [3, 4] and [5, 4] with the second operand transposed yields [3, 5]. The node's
operation is `Ops::K_MATMUL`, its flag 0 is false and its flag 1 is true.

## FourFlagCombinationsYieldBatchedProducts

For each of the four transpose pairs, with operands shaped to fit it, a rank-3 product yields
[2, 3, 5]. The node is a MATMUL that reads the two operands in order, writes one value and
carries the two flags it was given.

## BatchedProductTakesTheSharedBatchCount

`matmul` over [7, 2, 3] and [7, 3, 6] yields [7, 2, 6] with the first operand's layout and the
element type `Float32`.

## RefusesMixedRanksAndRanksOutsideTwoAndThree

`matmul` fails with `InvalidDescription` for a rank-2 operand with a rank-3 one in either order, for a rank-1
operand with a rank-3 one, and for two rank-4 operands.

## RefusesUnequalBatchesAndInnerExtents

Against a [2, 3, 4] operand, `matmul` fails with `InvalidDescription` for a [3, 4, 5] operand, whose batch
differs, and for a [2, 5, 3] operand, whose inner extent differs.

## SoftmaxEmitsReshapeLogSoftmaxExpReshape

`softmax` over a [2, 3, 3] input adds four nodes. Read back from the result they are RESHAPE,
EXP, LOG_SOFTMAX and RESHAPE. The first RESHAPE reads the input and yields [6, 3], LOG_SOFTMAX
yields [6, 3], and the result has the shape [2, 3, 3].

## SoftmaxOverRank2KeepsItsRowsAndLastAxis

`softmax` over a [4, 5] input yields [4, 5], and the EXP node behind the last RESHAPE reads a
[4, 5] value.

## SoftmaxRefusesRankBelowTwo

`softmax` over a [5] input fails with `InvalidDescription` and adds no node.

Suites: `Small_CausalMaskOps`.

Tests `GraphBuilder::causal_mask`: the node it emits, the shapes it accepts and the operands it
refuses. Every case uses the fill -1e9.

## TheMaskWritesItsIdentityAndKeepsTheShape

Over [2, 5, 5] float64 scores the call is ok and emits one CAUSAL_MASK node that reads the
scores, writes the returned value and holds the fill in attribute slot 0. The result has the
shape [2, 5, 5], the type float64 and the lifetime `Activation`, and the graph has no refusal.

## TheMaskAcceptsAnyNumberOfLeadingAxes

A [3, 3] matrix and a [2, 4, 3, 3] value both return a value of their own shape.

## TheMaskRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: a [5] vector ("[..., T, T] scores"), a [2, 4, 5]
value ("[2, 4, 5]"), int32 scores ("half, float32 or float64 scores") and a fill of minus
infinity ("finite fill"). After each the node count is unchanged and `has_refusal()` is true.

Suites: `Small_DepthwiseOps`.

Tests the three depthwise emitters of `GraphBuilder`: `depthwise_conv2d`,
`depthwise_conv2d_backward_input` and `depthwise_conv2d_backward_weight`. Most cases share one
geometry, a 3x3 window with unit stride and one pixel of padding, over a [2, 5, 5, 3] input and
[3, 3, 3] taps.

## DepthwiseConv2dWritesItsIdentityOverInputAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D` node that reads the input and then the
taps and writes that one value.

## DepthwiseConv2dYieldsTheWindowGridOverThePaddedInput

The shared geometry yields [2, 5, 5, 3]. The same window with stride 2 on both axes yields
[2, 3, 3, 3].

## DepthwiseConv2dKeepsLayoutDTypeAndLifetime

Over a `Float64` input and `Float64` taps the result is tagged NHWC, has the element type
`Float64` and the lifetime `Activation`.

## DepthwiseConv2dStoresItsGeometryInTheGeometrySlots

For a 3x5 window with padding 1 on top and 2 on the left over a [2, 7, 9, 3] input, the node's
first eight attribute values are 3, 5, 1, 1, 1, 1, 1 and 2.

## DepthwiseConv2dRefusesARankFourWeight

Taps of the shape [3, 3, 3, 1] make the emitter fail with `InvalidDescription`, and the node count does not
change.

## DepthwiseConv2dRefusesTapsOfAnotherWindowOrChannelCount

Taps of the shapes [5, 3, 3], [3, 5, 3] and [3, 3, 4] each make the emitter fail with `InvalidDescription`
against the 3x3 geometry and three channels, and the node count does not change.

## DepthwiseConv2dRefusesAnNCHWInput

An input converted to NCHW with `layout_convert` makes the emitter fail with `InvalidDescription`.

## DepthwiseConv2dRefusesTapsOfAnotherDType

A `Float32` input with `Float64` taps makes the emitter fail with `InvalidDescription`.

## DepthwiseConv2dRefusesAWindowThatDoesNotFit

A 3x3 window without padding over a [2, 2, 2, 3] input makes the emitter fail with `InvalidDescription`.

## BackwardInputWritesItsIdentityOverGradientAndTapsInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_INPUT` node that reads the
gradient and then the taps.

## BackwardInputRecoversTheForwardInputShape

From a [2, 3, 3, 3] gradient, stride 2 and the input shape [2, 5, 5, 3], the result has the
shape [2, 5, 5, 3], the layout NHWC and the element type `Float32`.

## BackwardInputRefusesAGradientOnAnotherGrid

For the input shape [2, 5, 5, 3], gradients of the shapes [2, 4, 5, 3], [2, 5, 5, 4] and
[2, 5, 3] each make the emitter fail with `InvalidDescription`, and the node count does not change.

## BackwardWeightWritesItsIdentityOverGradientAndInputInOrder

The result's producer is one `Ops::K_DEPTHWISE_CONV2D_BACKWARD_WEIGHT` node that reads the
gradient and then the forward input.

## BackwardWeightYieldsTheTapShapeInTheOperandsDType

Over `Float64` operands the result has the shape [3, 3, 3], the layout NHWC, the element type
`Float64` and the lifetime `Activation`.

## BackwardWeightRefusesMismatchedBatchChannelsOrDType

Against a [2, 5, 5, 3] gradient, an input of another batch, of another channel count or of
another element type makes the emitter fail with `InvalidDescription`, as does a rank-3 gradient. The node count
does not change.

Suites: `Small_ElementwiseOps`.

Tests the nodes the builder emits for the elementwise, comparison, selection and constant
operations.

This file needs no device. Every case in `Small_ElementwiseOps` calls an emitter of
`modules/core/graph/include/SushiAI/graph/builder.hpp` on a host-side graph and reads the node and the value it
wrote. The emitters are `div`, `minimum`, `maximum`, `less_equal`, `abs`, `atan`, `softplus`,
`silu`, `select`, `activation_backward`, `constant` and `constant_like`. The cases that concern
several emitters loop over a table and label each iteration with `SCOPED_TRACE`, so a failure
names the emitter. `test_backward.cpp` covers the backward rules of the same operations.

## BinaryEmitter and UnaryEmitter

`BinaryEmitter` holds a label, the identity an emitter must write and a pointer to a
two-operand member of `GraphBuilder`. `UnaryEmitter` is the same for one operand.

## K_BINARY_EMITTERS

`K_BINARY_EMITTERS` lists `div`, `minimum`, `maximum` and `less_equal` with K_DIV, K_MINIMUM,
K_MAXIMUM and K_LESS_EQUAL.

## K_UNARY_EMITTERS

`K_UNARY_EMITTERS` lists `abs`, `atan`, `softplus` and `silu` with K_ABS, K_ATAN, K_SOFTPLUS and K_SILU.

## K_INPUT_READING_GRADIENTS

`K_INPUT_READING_GRADIENTS` lists K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, the two activation
gradients that read the forward input.

## nchw_input

`nchw_input` declares a [2, 2, 2, 2] input and converts it to NCHW. Every extent is 2, so the
permuted shape equals the original and two such values still agree in shape.

## expect_produced_by

`expect_produced_by` expects a value to be the single result of a node of a given operation
that reads a given operand list in order.

## expect_described_like

`expect_described_like` expects a value to have the shape, element type and layout of another
value and the lifetime ACTIVATION.

## BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder

Each two-operand emitter writes one node of its own identity whose inputs are `a` then `b`.

## BinaryEmittersKeepShapeDTypeAndLayout

With two FLOAT64 NCHW operands, the result of each two-operand emitter has the first operand's
shape and is FLOAT64 and NCHW.

## BinaryEmittersRejectUnequalShapesWithoutBroadcasting

Each two-operand emitter fails with `InvalidDescription` for a [4, 3] operand with a [4, 1] one, in both orders, and
with a [3] one. These are the pairs a broadcasting library accepts. The node count is unchanged
after the three calls.

## BinaryEmittersRejectUnequalDtypes

Each two-operand emitter fails with `InvalidDescription` for a FLOAT32 operand with a FLOAT64 one, in both orders.

## LessEqualYieldsAMaskInTheOperandsOwnDType

The result of `less_equal` is FLOAT32 for FLOAT32 operands and FLOAT64 for FLOAT64 ones. The
mask has no element type of its own, which is what lets `mul` and `select` read it beside a
value of the same type.

## UnaryEmittersWriteTheirIdentityOverTheirOperand

Each single-operand emitter returns a value other than its operand, written by one node of its
own identity that reads the operand.

## UnaryEmittersKeepShapeDTypeAndLayout

With a FLOAT64 NCHW operand, the result of each single-operand emitter has the operand's shape
and is FLOAT64 and NCHW.

## SelectReadsItsConditionThenBothBranches

`select(condition, a, b)` writes one K_SELECT node whose inputs are the condition, `a` and `b`,
in that order.

## SelectKeepsShapeDTypeAndLayout

With three FLOAT64 NCHW operands the result has the shape, element type and layout of `a`.

## SelectRejectsAConditionOfAnotherShape

A [4, 1] condition over [4, 3] branches fails with `InvalidDescription`, and the node count is unchanged.

## SelectRejectsAConditionOfAnotherDType

An INT64 condition over FLOAT32 branches fails with `InvalidDescription`.

## SelectRejectsBranchesOfUnequalShape

A [4, 3] branch with a [4, 1] branch fails with `InvalidDescription` in both orders.

## SelectRejectsBranchesOfUnequalDType

A FLOAT32 branch with a FLOAT64 branch fails with `InvalidDescription` in both orders.

## ActivationBackwardEmitsTheInputReadingGradients

For K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, `activation_backward(gradient, dy, x)` writes one node
of that identity whose inputs are `dy` then `x`, and the result is described like `dy`.

## InputReadingGradientsRejectAGradientOfAnotherShape

A [4, 1] `dy` with a [4, 3] `x` fails with `InvalidDescription` for both gradients.

## InputReadingGradientsRejectAGradientOfAnotherDType

A FLOAT32 value with a FLOAT64 one fails with `InvalidDescription` for both gradients, in both orders.

## ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient

`activation_backward` stops the process, reporting "is not an activation gradient", when the identity is K_SILU, K_SOFTPLUS or K_DIV. The first two are
forward identities of `Ops::K_ACTIVATIONS` and the third is not in the table.

## FinaliseAcceptsAGraphUsingEveryEmitter

Two NCHW inputs are chained through all four two-operand emitters and all four single-operand
ones, and the chain ends in a `select` on a `less_equal` mask. `Graph::finalise()` does not
throw.

## FinaliseRejectsOperandsInTwoLayouts

For each two-operand emitter, a node that reads one NCHW operand and one NHWC operand is
emitted, and `finalise()` returns `InvalidDescription`. The emitter itself accepts the pair because the shapes are
equal.

## FinaliseRejectsASelectConditionInAnotherLayout

A `select` with an NHWC condition over NCHW branches is emitted, and `finalise()` returns
`InvalidDescription`.

## ResultsOfAParameterRequireAGradient

After `finalise()`, `silu(div(weight, constant))` requires a gradient and
`atan(maximum(constant, constant))` over an input does not.

## AComparisonOfAParameterRequiresNoGradient

After `finalise()`, the mask `less_equal(weight, bound)` requires no gradient, and neither does
a `select` that reads it as its condition over two inputs.

## ConstantWritesAnOperandFreeNodeHoldingItsValue

`constant([4, 3], 1e-7, FLOAT64)` writes one K_CONSTANT node with no input. Attribute slot 0
holds 1e-7 within 1e-20, and the value is [4, 3], FLOAT64, NHWC and ACTIVATION. `K_VALUE` and
`K_VALUE_TOLERANCE` are constants local to the case.

## ConstantDefaultsToFloat32

`constant([2], 1.0)` with no element type is FLOAT32.

## ConstantRejectsAnElementTypeThatIsNotRealFloating

`constant` fails with `InvalidDescription` for INT64 and for COMPLEX32, and the node count and the value count are
unchanged.

## ConstantLikeCopiesShapeDTypeAndLayoutWithoutReadingItsOperand

`constant_like(x, 0.5)` over a FLOAT64 NCHW value writes a K_CONSTANT node with no input whose
result is described like `x`. The value it names is not an operand of the node.

## ConstantLikeRejectsAValueThatIsNotRealFloating

`constant_like` over an INT64 value fails with `InvalidDescription`, and the node count and the value count are
unchanged.

## AConstantLikeAValueIsAnOperandThatFinaliseAccepts

`add(x, constant_like(x, 1e-7))` over an NCHW input passes `finalise()`, and the sum requires
no gradient.

## ZerosStillWritesTheZeroIdentity

`zeros` and `zeros_like` each write a K_ZERO node with no input. K_ZERO and K_CONSTANT are two
identities.

Suites: `Small_GatherOps`.

Tests `GraphBuilder::gather_rows` and `GraphBuilder::scatter_add_rows`: the node each emits,
the result shape and the operands each refuses. The table has 7 rows of length 3. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

## GatherRowsWritesItsIdentityOverTheTableAndTheIndex

Over a [7, 3] table and a [5] int32 index the call is ok and the result is the one output of a
GATHER_ROWS node that reads the table and then the index. The result has the shape [5, 3], the
type float32 and the lifetime `Activation`.

## GatherRowsAppendsTheRowLengthToAnIndexOfAnyRank

Over a float64 table, a [2, 4] index gives a float64 [2, 4, 3] value and a [2, 1, 3, 1, 2]
index gives a [2, 1, 3, 1, 2, 3] value.

## GatherRowsRefusesATableThatIsNotRankTwo

A [7] table is refused with "rank-2 [V, D] table" and a [7, 2, 3] table with "[7, 2, 3]".

## GatherRowsRefusesAnIndexThatIsNotInt32

A float32 index and an int64 index are each refused with "INT32 index".

## GatherRowsRefusesATableThatIsNotFloating

An int32 table is refused with "half, float32 or float64 table".

## GatherRowsRefusesAnIndexWhoseResultWouldPassTheHighestRank

An index of rank `K_MAX_RANK`, shaped [1, 1, 1, 1, 1, 1], is refused with "rank 1 to 5".

## ScatterAddRowsWritesItsIdentityAndTheTableShape

Over [2, 4, 3] float64 rows and a [2, 4] index, `scatter_add_rows` with 7 table rows is ok. The
result is the one output of a SCATTER_ADD_ROWS node that reads the rows and then the index, and
it has the shape [7, 3], the type float64 and the lifetime `Activation`.

## ScatterAddRowsRefusesOperandsThatDoNotFit

Five calls are refused: rank-1 rows ("rank 2 or more"), int32 rows ("half, float32 or float64
rows"), a float32 index ("INT32 index"), a [4] index for five rows ("one index per row") and a
row count of 0 ("a table of 0 rows").

## AnInt32InputLeafRequiresNoGradient

After `graph.finalise()` the int32 index has the lifetime `Input` and does not require a
gradient, and the gathered value does.

## TheCastPassLeavesAnInt32LeafAndItsGatherAsDeclared

After `Graph::insert_casts` under a `MixedPrecisionPolicy`, the index is int32, the table and
the gathered value are float32 and the `matmul` that reads the gathered value is `Half`. The
gather still reads the table and the index. No reported cast has the index as its source or
GATHER_ROWS as its op, and `find_dtype_mismatches` is empty.

Suites: `Small_LayerNormOps`.

Tests `GraphBuilder::layer_norm` and `GraphBuilder::layer_norm_backward`. The leaves are a
[2, 3, 4] input, a [4] gain and a [4] bias, and the epsilon is 1e-5. A refusal is
`InvalidDescription`, leaves the node count unchanged and sets `has_refusal()`.

## LayerNormWritesItsValueAndTheTwoRowStatistics

Over float64 leaves the call is ok and emits one LAYER_NORM node that reads x, the gain and the
bias and writes the output, the saved mean and the saved inverse deviation, with the epsilon in
attribute slot 0. The output has the shape [2, 3, 4] and both saves the shape [6]. The output
and the saved mean are float64, the output has the lifetime `Activation` and the graph has no
refusal.

## ARankOneInputIsOneRow

Over a [4] input the output has the shape [4] and the saved mean the shape [1].

## LayerNormRefusesOperandsThatDoNotFit

Four calls are refused: an int32 input ("half, float32 or float64 input"), a [3] gain ("gamma
as [4]"), a float64 bias under a float32 input ("beta as [4] of float32") and an epsilon of 0
("epsilon must be positive").

## LayerNormBackwardWritesTheThreeGradients

`layer_norm_backward` is ok and emits one LAYER_NORM_BACKWARD node that reads dy, x, the gain,
the saved mean and the saved inverse deviation and writes dx, dgamma and dbeta, of the shapes
[2, 3, 4], [4] and [4].

## LayerNormBackwardRefusesOperandsThatDoNotFit

A [6, 4] gradient is refused with "operand of [2, 3, 4]" and a [4] saved mean with "operand of
[6]".

Suites: `Small_LogSoftmaxOps`.

Tests `GraphBuilder::log_softmax_backward`.

## TheGradientWritesItsIdentityAndKeepsTheShape

Over a [4, 3] float64 gradient and the `log_softmax` of a [4, 3] float64 input, the call is ok
and emits one LOG_SOFTMAX_BACKWARD node that reads the gradient and then the log-probabilities.
The result has the shape [4, 3], the type float64 and the lifetime `Activation`, and the graph
has no refusal.

## TheGradientRefusesOperandsThatDoNotFit

Four calls fail with `InvalidDescription`: rank-3 operands ("rank-2 log-probabilities"), int32
operands ("half, float32 or float64 log-probabilities"), a [4, 5] gradient ("[4, 5]") and a
float64 gradient under float32 log-probabilities ("a gradient like them"). After each the node
count is unchanged and `has_refusal()` is true.

Suites: `Small_RefusalFamilies`.

Tests the refusal of each emitter family that the cases above do not reach. Each call fails with
`InvalidDescription` and the named part of its message, leaves the node count as it was and sets
`has_refusal()`, through `expect_refused`.

## RefusalBench, window_2x2 and window_9x9

`RefusalBench` holds a graph and a builder over it and declares an NHWC [2, 4, 4, 3] image.
`window_2x2` is a 2x2 window of unit stride, and no 4x4 input holds the 9x9 window `window_9x9`.

## WindowHelpersReturnTheirRefusal

`window_extents` refuses a zero stride ("positive stride") and a window the input does not hold
("fit its padded input"). `upsampled_extents` refuses a zero scale and a product beyond 64 bits.
A 2x2 window over a 4x4 input yields 3x3.

## AFailedOperandPassesThroughWithoutANode

A refused `reshape` passed to `matmul` and that result to `relu` comes back as the same failure
block, with no node added and `has_refusal()` true.

## ScaleActivationCastAndLayoutEmittersRefuse

`dynamic_scale` with a two-element factor, `leaky_relu` with slope 1, `cast` of an INT32 value,
`cast_into` across two shapes, `layout_convert` of a rank-2 value and `reshape` to another
element count.

## WindowEmittersRefuse

`im2col`, `im2col_with_extents`, `col2im`, `col2im_with_extents`, `conv2d`, `maxpool2d`,
`maxpool2d_with_extents`, `avgpool2d`, `upsample2d`, `spatial_mean` and the gradients of the
three pools, the upsampling and the spatial mean, each over one operand or window that does not
fit.

## ChannelAndBatchNormEmittersRefuse

`concat_channels` with no operand and with two that disagree on H, `slice_channels` past the
last channel, `channel_affine` with a [4] beta over three channels, `sum_channels` of a rank-2
value, `batchnorm2d` with a zero epsilon and with a float64 beta, `batchnorm2d_backward` with a
rank-2 gradient, and `batchnorm_fold` with a [4] beta.
