# Unit tests

The unit tier runs without a device. Every file in this folder builds a graph, a config document
or a byte stream on the host and inspects the result, so no case needs an Engine, a TensorPool or
SYCL.

This page holds the reasoning behind the cases. A `//` line in a test states the invariant the
case relies on and cites this page. The section under the test's name says why the invariant
matters, which number it was measured at, or which failure it once let through. Where a test has
several such lines, each paragraph opens with the assertion or the statement it explains.
Sections follow the files in path order, and inside a file the headings follow the order of the
tests.

## Files

| File | What it tests |
| --- | --- |
| [`test_avg_pool2d.cpp`](test_avg_pool2d.cpp) | The shapes the AvgPool2d layer traces and reports, and the inputs it refuses. |
| [`test_backward.cpp`](test_backward.cpp) | The graph that differentiation emits and the coverage of the backward rule table. |
| [`test_checkpoint_format.cpp`](test_checkpoint_format.cpp) | The checkpoint format's round trip and its refusal of each malformed byte pattern. |
| [`test_config.cpp`](test_config.cpp) | How the config layer parses its JSON documents and which ones it refuses. |
| [`test_conv2d.cpp`](test_conv2d.cpp) | The Conv2d layer's trace, parameters, shape inference and rejections. |
| [`test_diagnostics.cpp`](test_diagnostics.cpp) | That each host-lane precondition raises a message carrying the values that disagree. |
| [`test_flatten.cpp`](test_flatten.cpp) | The Flatten layer's trace, shape rule, rejections and parameter walk. |
| [`test_fusion.cpp`](test_fusion.cpp) | The matches and the refusals of the static fusion pass on synthetic graphs. |
| [`test_global_avg_pool.cpp`](test_global_avg_pool.cpp) | The shapes GlobalAvgPool traces and reports, and the inputs it refuses. |
| [`test_gradcheck.cpp`](test_gradcheck.cpp) | The gradient checker against closed-form functions and a wrong gradient. |
| [`test_ir.cpp`](test_ir.cpp) | The IR's single-assignment and ordering rules and the builder's shape inference. |
| [`test_layer_registry.cpp`](test_layer_registry.cpp) | Each layer registry row's factory, the keys it reads and the model it builds. |
| [`test_layer_stack.cpp`](test_layer_stack.cpp) | The runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread. |
| [`test_layout.cpp`](test_layout.cpp) | That every value carries a tensor layout and that the rebuilding passes keep it. |
| [`test_leaky_relu.cpp`](test_leaky_relu.cpp) | The LeakyReLU layer's trace, shape inference and slope range. |
| [`test_loss_scaling.cpp`](test_loss_scaling.cpp) | The dynamic loss-scaling rule on LossScaleState, with no device involved. |
| [`test_max_pool2d.cpp`](test_max_pool2d.cpp) | The MaxPool2d layer's trace, shape inference and rejections. |
| [`test_memory_plan.cpp`](test_memory_plan.cpp) | That the memory planner never shares a byte between two values live together. |
| [`test_module.cpp`](test_module.cpp) | The parameter walk over layers, containers and a layer enumerated at run time. |
| [`test_objective.cpp`](test_objective.cpp) | The objective seam: the leaves it declares, the loss it emits and the resulting IR. |
| [`test_ops.cpp`](test_ops.cpp) | That the operation table is complete and that the reverse lookup resolves names. |
| [`test_precision.cpp`](test_precision.cpp) | The mixed-precision policy and where the cast insertion pass places its casts. |
| [`test_provenance.cpp`](test_provenance.cpp) | The forward origin that differentiation records for every node it appends. |
| [`test_region_signature.cpp`](test_region_signature.cpp) | The signature that `signature_of` computes for one region of a partition. |
| [`test_rematerialise.cpp`](test_rematerialise.cpp) | The structure of the graph that the IM2COL rematerialisation pass rewrites. |
| [`test_rule_table.cpp`](test_rule_table.cpp) | That every domain's backward rules reach the combined table exactly once. |
| [`test_scope_partitioner.cpp`](test_scope_partitioner.cpp) | The region keys ScopePartitioner gives the nodes of a differentiated graph. |
| [`test_scope_trace.cpp`](test_scope_trace.cpp) | How ScopeTrace cuts a trace into segments and which calls it rejects. |
| [`test_shape.cpp`](test_shape.cpp) | Shape's extents, indexing, validation, equality and hash. |
| [`test_spatial_shapes.cpp`](test_spatial_shapes.cpp) | The shapes the builder infers for spatial operations and the layouts it refuses. |
| [`test_unflatten.cpp`](test_unflatten.cpp) | The trace, the shape rule and the rejected inputs of the Unflatten layer. |
| [`test_upsample2d.cpp`](test_upsample2d.cpp) | The shape, layout and rejections of the Upsample2d layer. |
| [`test_version.cpp`](test_version.cpp) | That the runtime version agrees with the compile-time macros. |

## test_avg_pool2d.cpp

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `include/SushiAI/nn/avg_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter, so a case either traces the layer into a host-side
graph or calls `output_shape()` on a per-sample shape.

### window

`window` returns a geometry with the given kernel and stride on each axis, dilation 1 and no
padding. A case that needs padding or dilation sets the field afterwards.

### traced_shape

`traced_shape` traces a layer over a batch of two images of the given per-sample shape and
returns the shape of the result, batch axis included.

### per_sample

`per_sample` drops the batch axis of a rank-4 shape.

### SatisfiesTheModuleTrait

Four `static_assert` lines hold the layer's place in the layer library: `AvgPool2d` is a
`Module`, `const AvgPool2d` is one too, which requires a const `forward()`, it is not a
`ModuleContainer`, and it has no parameter walk of its own, so the walk reaches it through
`fields()`.

### ForwardTracesThePooledShapeInNhwc

A 3x3 window with stride 3 over a [2, 6, 6, 3] input traces a [2, 2, 2, 3] value tagged NHWC.

### OutputShapeAgreesWithTheTracedShapeForASquareWindow

`output_shape([6, 6, 3])` is [2, 2, 3], and it equals the traced shape with the batch axis
dropped. The literal is asserted beside the comparison so the two cannot agree on a wrong
shape. The next two cases have the same form.

### OutputShapeAgreesWithTheTracedShapeForAPaddedUnevenWindow

A 3x2 window with stride 2 by 1 and one pixel of padding on each axis maps [7, 5, 4] to
[4, 6, 4].

### OutputShapeAgreesWithTheTracedShapeForADilatedWindow

A 3x3 unit-stride window dilated by 2 on the height axis maps [9, 8, 1] to [5, 6, 1]. The two
axes have different dilations, so an implementation that applied one axis's dilation to both
gives another shape.

### OutputShapeRejectsAFeatureVector

`output_shape([8])` throws, because a per-sample shape must be rank 3.

### OutputShapeRejectsABatchedShape

`output_shape([2, 4, 4, 3])` throws. A shape that still carries the batch axis is rank 4.

### OutputShapeRejectsAWindowLargerThanTheImage

A 5x5 window over a [4, 4, 3] sample throws.

### OutputShapeRejectsAZeroStride

A geometry with stride 0 on both axes throws.

### OutputShapeNamesAZeroStride

The message for the same geometry contains "AvgPool2d" and "stride 0,0". The case holds the
message to the stride, where a check ordered differently would report a window that does not
fit.

### OutputShapeRejectsANegativePadding

A top padding of -1 throws with a message containing "AvgPool2d" and "padding -1,0".

### ForwardRejectsARankTwoInput

`forward()` throws on a [2, 8] input.

### ForwardRejectsAnNchwInput

A [1, 6, 6, 3] input is converted with `layout_convert` to NCHW, and `forward()` throws on the
converted value.

### ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with stride 0 on both axes throws with a message containing "AvgPool2d" and
"stride 0,0".

### ForwardRejectsANegativePadding

`forward()` with a top padding of -1 throws with a message containing "AvgPool2d" and
"padding -1,0".

### ForwardRejectsAWindowLargerThanTheImage

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ATracedLayerHasNoParameter

After a trace the parameter walk visits nothing and `parameter_count` is 0. The graph holds
two values, the input and the pooled result.

## test_backward.cpp

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

### EveryImplementedOperationHasARule

`has_backward_rule(BATCHNORM_UPDATE_STATS)` is false, and that is decision CV-7, not an
omission. A running mean is a buffer the training step writes beside the graph. No gradient
flows through it.

`has_backward_rule` is false for LEAF and ZERO because both produce values from nothing. There
is nothing upstream to hand a gradient to, so there is no rule to write.

`has_backward_rule(CAST)` is false because of the order of the pipeline. Casts are inserted
after differentiation (`include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

### EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `src/autograd/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/functional/integration/test_autograd_numeric.cpp` and all run in
FLOAT64, for the reason `include/SushiAI/autograd/gradcheck.hpp` derives.

The LOG_SOFTMAX entry names a message test. LOG_SOFTMAX's rule exists only to produce a good
error, so there is no gradient to verify and the message test is its proof.

### TwoLayerMlpProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

### TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another MATMUL. It is skipped, not
computed and discarded.

### FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered ACCUMULATE form.

### ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

### ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

### AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

### LogSoftmaxOutsideTheFusedPairSaysSoClearly

A RELU consumes the log-softmax result, so the cross-entropy fusion does not apply and the
standalone rule has to refuse.

## test_checkpoint_format.cpp

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `include/SushiAI/io/checkpoint_format.hpp` out of
`include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

### Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

### DistinguishesTwoShapesWithTheSameElementCount

This case covers the defect the format exists to close: an element count alone lets a [2, 3]
weight load into a [3, 2] one. The shape is recorded, so the two records are not
interchangeable.

## test_config.cpp

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`tests/common/test_throws.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document, and the layer registry it selects through
- the train and eval documents

### KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

### AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

### AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

### WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

### AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

### ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

### ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

### AnUnknownLayerTypeListsEveryTypeThisBuildHas

The model names a layer type "hologram", which no row of `NN::LAYER_REGISTRY` carries. The
message contains "hologram", "layer type", "'linear'" and "'gelu'". The unknown type has to
be a name the registry does not hold, and "conv2d" is a row of the table.

### AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

### Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

### LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

### AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

## test_conv2d.cpp

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

### make_conv

`make_conv` builds a layer with a square window and one stride and one padding for both axes.
The stride defaults to 1 and the padding to 0. A case that needs an anisotropic window or a
dilation sets the geometry field afterwards.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### traced_shape

`traced_shape` traces a copy of a layer over a batch of the given per-sample shape and returns
the result's shape. It takes the layer by value, so a const layer can be traced although
`forward()` declares parameters.

### walk

`walk` returns the parameter walk's output as "name:extents" strings.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents

A 3x3 window with padding 1, one input channel and eight filters traces a [4, 28, 28, 1]
input to a [4, 28, 28, 8] value tagged NHWC.

### TheParametersAndTheResultTakeTheInputElementType

With a FLOAT64 input the weight, the bias and the result are FLOAT64.

### TheParameterWalkNamesTheWeightAndTheBias

The walk is empty before the trace. After it the walk yields "weight:[8, 3, 3, 1]" then
"bias:[8]", the names `Linear` uses, and `parameter_elements` is 8 * 3 * 3 * 1 + 8.

### TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn

A 2x4 window with three input channels and five filters gives a weight of shape [5, 2, 4, 3].
The four extents differ, so a weight with two axes swapped fails the comparison.

### ALayerWithABiasTracesOneBiasAdd

The traced graph holds exactly one BIAS_ADD node.

### ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd

With `use_bias` false the walk yields the weight alone, the bias is not declared, the graph
holds no BIAS_ADD node, and the result shape is still [4, 28, 28, 8].

### TheWeightAndTheBiasRecordTheirFans

For a 3x5 window with four input channels and six filters, both parameters record a fan-in of
3 * 5 * 4 and a fan-out of 3 * 5 * 6. The bias carries the same pair as the weight.

### TheParametersRecordTheirRoles

The weight has role WEIGHT and the bias has role BIAS.

### TracingTwiceDoesNotDuplicateParameters

After a second trace into the same graph the weight and the bias keep their value ids and
`parameter_count` is 2.

### OutputShapeAgreesWithForwardForAStridedWindow

A 3x3 stride-2 unpadded window with sixteen filters maps [9, 12, 3] to [4, 5, 16], and the
traced shape of a batch of two equals that shape with the batch extent in front. The four
cases from here assert the literal and the agreement together.

### OutputShapeAgreesWithForwardForAPaddedWindow

A 5x5 window with padding 2 maps [7, 10, 2] to [7, 10, 4].

### OutputShapeAgreesWithForwardForADilatedStridedWindow

A 3x3 window with stride 2, padding 1 and dilations 2 by 3 maps [11, 13, 2] to [5, 5, 3].

### OutputShapeAgreesWithForwardForAPointwiseWindow

A 1x1 window with six filters maps [5, 7, 4] to [5, 7, 6]. This is the geometry for which
`GraphBuilder::conv2d` emits no IM2COL.

### OutputShapeRejectsANonPositiveInputChannelCount

A layer with `in_channels` 0 throws with a message containing "Conv2d", "in=0" and "out=8".

### OutputShapeRejectsANonPositiveOutputChannelCount

A layer with `out_channels` -1 throws with a message containing "Conv2d", "in=3" and "out=-1".

### OutputShapeRejectsAWrongChannelCount

A three-channel layer given [8, 8, 4] throws. The message contains "Conv2d(3, 8)", the shape
"[8, 8, 4]" and the expected form "[H, W, 3]".

### OutputShapeRejectsAFeatureVector

`output_shape([3])` throws with a message containing "Conv2d(3, 8)", "[3]" and "[H, W, 3]".

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 3, 3])` throws with a message containing the shape and "[H, W, 3]". The
third extent equals `in_channels`, so only the rank check can refuse this shape.

### OutputShapeRejectsAWindowThatDoesNotFit

A 5x5 window over [4, 4, 3] throws with a message containing "Conv2d(3, 8)", "[4, 4, 3]" and
"5x5".

### OutputShapeRejectsAZeroStride

A stride of 0 throws with a message containing "Conv2d(3, 8)" and "stride 0,0".

### OutputShapeRejectsANegativePadding

A padding of -1 throws with a message containing "Conv2d(3, 8)" and "padding -1,-1".

### ForwardRejectsAWrongChannelCount

`forward()` on a [2, 8, 8, 4] input throws with a message containing "Conv2d(3, 8)",
"[8, 8, 4]" and "[H, W, 3]", the three strings OutputShapeRejectsAWrongChannelCount expects.
The weight is not declared afterwards.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` on a [2, 3] input throws with a message containing "Conv2d(3, 8)", "rank-4" and
"[2, 3]", and the weight is not declared afterwards.

### ForwardRejectsNonPositiveChannelCounts

`forward()` on a layer with `out_channels` 0 throws with a message containing "Conv2d",
"in=3" and "out=0".

### ForwardRejectsAWindowThatDoesNotFit

`forward()` with a 5x5 window over a [2, 4, 4, 3] input throws with a message containing
"Conv2d(3, 8)", "[4, 4, 3]" and "5x5".

### ForwardRejectsAnNchwInput

The input is [2, 3, 3, 3] converted to NCHW. Every extent is 3, so the permuted shape passes
every check but the one on the layout tag. The message contains "Conv2d" and "NHWC". The node
count and the value count are unchanged and neither parameter is declared, so the layer
refuses before it writes anything to the graph.

### DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients, neither is `NO_VALUE`, and each has the shape
of its parameter. The case checks that a gradient exists and has the right shape; it computes
no number.

## test_diagnostics.cpp

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`tests/common/test_throws.hpp` defines SA_EXPECT_ERROR.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### ALossThatNoRuleCanSeedIsRejectedRatherThanReturningNothing

A 1-element MATMUL result passes the "is it a scalar" check. Every rule then declines to act,
because the root's own gradient is implicit and only NLL_LOSS knows to start from it. Before
this diagnostic existed the caller got a full set of NO_VALUE gradients and no error. Then
`bind_parameters` reported a misleading "parameter receives no gradient", which blamed the model
and not the loss.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### LogSoftmaxOutsideTheFusedPairExplainsTheRestriction

Naming the way out matters more than naming the failure. The caller has to be told to route
through NLL_LOSS, and not merely that log-softmax said no.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.

## test_flatten.cpp

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

### traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

### SatisfiesTheModuleTrait

Two `static_assert` lines require `Flatten` to be a `Module` and not a `ModuleContainer`.

### ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a RESHAPE
with the input as its one operand and the result as its one output.

### ARank3InputFoldsItsTwoInnerAxes

A [2, 3, 4] input traces to [2, 12].

### ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

### AnNhwcInputYieldsAnNhwcResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

### ForwardRejectsAnNchwInput

The layout tag of a [2, 3, 4, 5] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeAgreesWithTheTracedShape

`output_shape()` maps [4, 5, 3] to [60], [3, 4] to [12] and [7] to [7]. For each, the traced
shape of a batch equals the batch extent followed by that one extent.

### ForwardRejectsARank1Input

`forward()` throws on a [6] input, which has a batch axis and no per-sample axis.

### OutputShapeRejectsARank0Shape

`output_shape()` throws on an empty shape.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## test_fusion.cpp

Tests the matches and the refusals of the static fusion pass on synthetic graphs.

This file needs no device. The static fusion pass is a pure function of the IR, as the memory
planner is, so every case is a synthetic graph and a table of expectations with no engine
anywhere.

The cases pin the set of refusals more than the matches. A pass that fuses too little is slow.
A pass that fuses too much is silently wrong: the graph still runs, the loss still falls, and
the only symptom is a model that converges somewhere slightly different. So most cases are
graphs the pass must decline, one per rule:

- an intermediate with a second consumer (fusing it means losing it),
- an intermediate the caller reads by id (the same loss, seen from outside),
- a spilled pre-activation read by a node that runs before the fused one.

The activation whose gradient differentiates at its input used to be a fourth entry on that
list. It no longer is. GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal became a condition: an INPUT-gradient activation may be folded only by an
entry that writes the tensor its gradient reads, and only if it writes the right one. That
condition has a compile-time half in `src/graph/fusion.cpp`, where an entry breaking either half
fails a static_assert. A test cannot cover that half, because a test that fails to compile is
not a test. This file checks the pair of properties the assertion reads and, above all, that the
fold leaves the pre-activation written and readable by the backward node.

The cases fall into four groups, in file order:

- the catalog is well formed
- the matches
- the refusals, which are the half that matters
- the pass keeps the graph usable

### EveryFormRoundTripsThroughItsOperationIdentity

The lowering dispatches on `fused_form_op` and `fused_form_of`. A form that does not round-trip
would lower as "no recorder for this operation", which is loud, but only at run time and only on
a graph that happens to contain it.

### AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts

`activation_survives_fusion(RELU)` is the property the static_assert in `src/graph/fusion.cpp`
reads. If it ever flips for an activation that has a catalog entry, the static_assert catches
it, but only while the property stays true of the table the static_assert consults. This test
pins that.

`activation_survives_fusion(GELU)` is false because SushiBLAS's `gelu_backward` takes the
pre-activation. An entry that wrote only its result would differentiate at a tensor that never
reached memory. That has not changed and must not, because it is the premise the static_assert
reasons from.

`catalog_folds_activation(GELU)` is true because the conclusion drawn from that premise
changed. The one entry that spills folds GELU. "Survives fusion" and "the catalog folds it" are
now two questions with two answers, and a caller who reads the first as the second gets the
wrong answer for exactly one activation.

### AMatmulAndABiasAddBecomeOneNode

`report.fused().size()` is 1, and the fused node reads A, B and the bias and still writes the
value the caller named. That is the whole reason ids are preserved.

The `flag(0)` and `flag(1)` checks hold the transposition flags. They have to survive, because
the fused recorder hands them straight to `execute_gemm`, and losing one gives a wrong answer of
exactly the right shape.

### TheLongestMatchWinsOverTheShorterOneInsideIt

Both gemm+bias and gemm+bias+relu match this graph. Rooted at the bias_add, the shorter one
would win and the relu would be left outside for good, which is why the matcher walks roots in
reverse.

### ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

### AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

### AnIntermediateTheCallerReadsByIdIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
bias_add, whose own intermediate is free.

### AParameterIsProtectedWithoutBeingListed

A persistent value's buffer outlives the step whatever the graph says, so the caller must not
have to remember to list it. In this graph the "intermediate" is a parameter. That is an odd
graph, but it is the shape a caller would get wrong.

The assignment `lifetime = OPTIMIZER_STATE` forces the exp's result into a persistent class, as
a checkpointed activation would be.

### AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills

The graph is the trap the fusion design is arranged around. The differentiation pass has
already emitted `gelu_backward` reading the pre-activation, so that value has two consumers.
Every other entry would stop at that point under rule 1. This entry does not, for the same
reason it is allowed to exist: the fused node keeps writing that value, as a second output,
under the same id.

The check on the fused node's second output is the assertion that carries the test, and a plain
GEMM_BIAS_GELU with no spill would fail it. The fused node still writes the pre-activation as
its second output, and it is the same value, so `gelu_backward`'s operand needed no rewriting
and the backward rule needed no change.

`fused()[0].eliminated` holds only the MATMUL's result. A report that listed the spilled value
as eliminated would read like the report of the silently wrong version of this pass.

### ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the GELU's position, so a reader
of the pre-activation sitting between the bias_add and the GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

### ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

### TheGeluFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

### DisablingTheGeluFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::DISABLED`, is the only one.

### ARecoverableActivationNextToAFusedGemmIsFoldedNotReported

This case mirrors DisablingTheGeluFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

### ValueIdsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

### TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

## test_global_avg_pool.cpp

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

### traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

### SatisfiesTheModuleTrait

Three `static_assert` lines require `GlobalAvgPool` to be a `Module`, not to be a
`ModuleContainer`, and to have a `forward()` that is callable on a const object and returns
`Graph::ValueId`.

### ForwardReducesTheSpatialAxes

A [4, 7, 5, 16] input traces to [4, 16].

### ForwardEmitsOneSpatialMeanNode

The trace adds exactly one node, a SPATIAL_MEAN with the input as its one operand and the
result as its one output.

### ForwardKeepsTheNhwcLayoutTag

The result of an NHWC input is tagged NHWC, and `finalise()` accepts the graph.

### OutputShapeKeepsOnlyTheChannels

`output_shape([7, 5, 16])` is [16].

### OutputShapeAgreesWithForward

For a square [7, 7, 512] sample and an oblong [3, 9, 2] one, `output_shape()` is rank 1 and
the traced shape is the batch extent followed by its one extent.

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsARankTwoShape

`output_shape([7, 16])` throws.

### OutputShapeRejectsAShapeWithTheBatchAxis

`output_shape([4, 7, 5, 16])` throws.

### ForwardRejectsAFeatureMatrix

`forward()` throws on a [4, 16] input.

### ForwardRejectsAnInputWithoutTheBatchAxis

`forward()` throws on a [7, 5, 16] input.

### ForwardRejectsAnNchwInput

A [1, 4, 4, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace, and `parameter_elements` is 0 after it.

## test_gradcheck.cpp

Tests the gradient checker against closed-form functions and a wrong gradient.

This file needs no device. The gradient checker is the instrument the integration tier measures
with, so this file calibrates it against closed-form functions whose right answer is known. One
of them is a deliberately wrong gradient, which proves the check can fail.

### RejectsAGradientThatIsWrongByATenthOfAPercent

This case is the reason for insisting on FLOAT64. A 1e-3 relative error is far above the noise
floor of about 1e-9 in FLOAT64. It would sit inside the floor of about 2.4e-5 that an fp32
difference quotient bottoms out at, and would pass unnoticed.

### AReluFamilyFunctionNeedsToBeSampledOffTheKink

`options.min_step` is large enough that the difference survives the cancellation of summing it
against the O(1) terms. A step of 1e-4 keeps the pushed coordinates at 1e-3, where a central
difference of x^2 is still exact to 1e-9.

The `unshifted` point straddles the kink. The quotient taken across it reports the average of
the two one-sided derivatives, which no correct rule can match.

## test_ir.cpp

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

### NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

### InsertNodeRejectsAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

`node_count()` is still 1 after the throw. The insert was rejected, not half-applied: nothing
was inserted and no producer was claimed.

The final `EXPECT_NO_THROW` inserts the same node ahead of that reader and succeeds. That makes
the rejection a statement about position and not about the node.

### AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
`src/graph/lowering.cpp` reads slots with `get()` and passes the values straight to SushiBLAS.

## test_layer_registry.cpp

Tests the layer registry: each row's factory, the keys it reads and the model it builds.

This file needs no device. `NN::LAYER_REGISTRY` in `include/SushiAI/nn/layer_registry.hpp` is
a constant table of thirteen rows, and a factory turns a per-sample input shape and a parsed
JSON object into an `NN::ILayer`. A case therefore needs a `Shape`, a string of JSON and, for
the model cases, a host-side graph.

The cases fall into three groups, in file order:

- every row of the table: that it has a factory, builds its own layer type and refuses an
  unknown key
- the keys of each factory, their defaults and the values they refuse
- the convolutional classifier of `K_CONV_MODEL`, built, traced, named and differentiated

### K_CONV_MODEL

`K_CONV_MODEL` is a model file as a string: 784 input features, then `unflatten` to
28 x 28 x 1, `conv2d` with 8 filters, `relu`, `maxpool2d`, `conv2d` with 16 filters, `relu`,
`maxpool2d`, `flatten` and `linear` with 10 units. Both convolutions have kernel 3 and
padding 1, and both pools have kernel 2. The layers of
`examples/mnist_cnn/model.json` are the same nine, with the optional keys spelled out.

### K_CONV_MODEL_PARAMETERS

`K_CONV_MODEL_PARAMETERS` lists the six parameters that model declares, as "name:extents":
a weight and a bias under the indices 1, 4 and 8. The two tests that compare names read the
expected list from here.

### MinimalCase

`MinimalCase` holds one registry key, a layer object without its "type", the per-sample shape
that enters the layer and the per-sample shape the layer must report.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` holds one `MinimalCase` per registry row, in registry order. Each object
carries the required keys of its row and no other.

### attributes_from

`attributes_from` parses a string as the attributes of one layer.

### make_layer

`make_layer` looks a type up in `NN::LAYER_REGISTRY` with `Config::find_entry` and calls the
row's factory with the context "layer 0 '<type>'". The cases that assert a message expect that
context in it.

### wrapped

`wrapped<L>` returns the layer of type `L` inside an `NN::LayerAdapter<L>`, or null when the
`ILayer` wraps another type. The `dynamic_cast` is how a case tells which layer type a factory
built and reads the fields the factory set.

### square_window

`square_window` returns a geometry with a square kernel and one stride and one padding for
both axes.

### make_conv

`make_conv` returns a `Conv2d` with the given channel counts and geometry, its parameters
undeclared.

### expect_pool_window_spellings_agree

`expect_pool_window_spellings_agree<Pool>` builds a pool from `"kernel": 3` and from
`"kernel": [3, 3]`. It expects both to wrap a `Pool`, the first to hold a 3x3 kernel, and the
two to hold the same kernel and the same stride.

### make_conv_stack

`make_conv_stack` parses `K_CONV_MODEL` and returns the `LayerStack` that `build_layer_stack`
makes of it.

### traced_names

`traced_names` traces a model over a [4, 784] input and returns its parameters as
"name:extents" strings. It is a template, so it takes a `LayerStack` and a `Sequential`.

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `NN::LAYER_REGISTRY`, and entry `i` names the type of
row `i`. A row added to the registry without a case here fails this test, so the two tests
that loop over the cases cover every factory.

### EachFactoryBuildsFromAMinimalObject

For every case the factory returns a layer whose `output_shape()` is the shape the case lists.

### EachKeyBuildsItsOwnLayerType

Each of the thirteen keys builds an adapter over the layer type it names: `wrapped<L>` is not
null for the expected `L`. Two rows that pointed at one factory would fail for one of them.

### EachFactoryRejectsAnUnknownKey

For every case, an object holding only `"no_such_key"` throws with a message containing the
layer's context and "'no_such_key'".

### Conv2dTakesItsInputChannelsFromTheInputShape

A `conv2d` built over a [8, 8, 7] input has `in_channels` 7 and `out_channels` 5. An object
that sets `"in_channels"` is refused by name, so a model file cannot state a channel count
that disagrees with the layer before it.

### Conv2dRefusesAnInputThatIsNotAnImage

A `conv2d` given the per-sample shape [784] throws. The message contains the context, "[784]",
"[H, W, C]" and "'unflatten'", the layer that gives a feature vector its image shape.

### Conv2dDefaultsItsOptionalKeys

With only "filters" and "kernel" set, the layer has stride 1, padding 0 and dilation 1 on both
axes, and `use_bias` is true.

### Conv2dReadsEveryKeyItAccepts

The object sets all six keys, four of them as [h, w] pairs with unequal entries. The layer
holds kernel 3 by 5, stride 2 by 1, padding 1 by 2, dilation 1 by 2 and `use_bias` false, and
over a [16, 16, 2] input it reports [8, 12, 5].

### AKernelAsAnIntegerAndAsAPairAgree

For `conv2d`, `maxpool2d` and `avgpool2d`, `"kernel": 3` and `"kernel": [3, 3]` give the same
output shape. The case then reads the stored window: the two convolutions both hold a 3x3
kernel, and `expect_pool_window_spellings_agree` checks the two pools. Equal output shapes
alone would also hold for two layers that both stored a wrong window.

### AWindowSettingOfTheWrongFormIsRefused

A one-entry array, a three-entry array, a string and the real number 1.5 are each refused.
Every message names the layer and the key, as in "layer 0 'maxpool2d' 'kernel'", and contains
"two-entry [h, w]". The two array messages also give the entry count, "1 entries" and
"3 entries".

### APoolStrideDefaultsToItsKernel

A `maxpool2d` with `"kernel": [2, 3]` and no stride holds stride 2 by 3 and padding 0, and
maps [12, 12, 2] to [6, 4, 2]. An `avgpool2d` with kernel 3, stride 1 and padding 1 holds
those values and keeps [12, 12, 2].

### Conv2dRejectsANonPositiveExtent

Eight objects are refused: a filter count of 0 and of -4, a kernel of 0, of [3, 0] and of -3,
a stride of 0 and of [1, -1], and a dilation of 0. Each message contains the context, "Conv2d"
and "positive".

### ThePoolsRejectANonPositiveExtent

For both pools, a kernel of 0, of [2, 0] and of -2 and a stride of 0 and of [-1, 2] are
refused with a message containing the context and "positive".

### AWindowedLayerRejectsANegativePadding

For `conv2d` and both pools, a padding of -1, of [0, -1] and of [-1, 0] is refused with a
message containing the context and "padding". The two pairs put the negative entry on each
axis in turn.

### AFactoryRejectsAMissingRequiredKey

A `conv2d` without "filters", a `conv2d` without "kernel", each pool without "kernel" and a
`linear` without "units" are refused. Each message contains the context and "has no" followed
by the quoted key.

### LeakyReluRejectsASlopeOutsideItsRange

Slopes of 0, 1 and -0.1 are refused when the layer is built, with a message containing the
context and "slope".

### UpsampleRejectsAScaleBelowOne

A scale of 0 is refused when the layer is built, with a message containing the context,
"Upsample2d" and "scale".

### LeakyReluAndUpsampleReadTheirOptionalKey

A `leaky_relu` built from an empty object has slope 0.01, and one built with `"slope": 0.25`
has 0.25. An `upsample` built from an empty object maps [4, 4, 2] to [8, 8, 2], and one built
with `"scale": 3` maps it to [12, 12, 2].

### UnflattenRefusesAnInputOfAnotherSize

A 28 x 28 x 1 `unflatten` given [783] throws with a message containing the context, "[783]"
and "[784]". An object without "channels" throws with a message containing the context and
"channels".

### TheConvModelBuildsAndReportsItsOutputShape

The stack has nine layers and its output shape is [10]. The shapes after layers 1, 3, 6 and 7
are [28, 28, 8], [14, 14, 8], [7, 7, 16] and [784], so each factory received the shape its
predecessor reported.

### TheConvModelTracesToTheShapeItReports

The stack traces a [4, 784] input to a [4, 10] value.

### TheConvModelNamesItsParametersByLayerPosition

The traced stack's parameters are `K_CONV_MODEL_PARAMETERS`: indices 1, 4 and 8, with gaps at
the layers that own no parameter.

### ASequentialOfTheSameLayersYieldsTheSameNames

An `NN::Sequential` of the same nine layer types is traced over the same input. Its names
equal `K_CONV_MODEL_PARAMETERS`, and the stack's names equal the `Sequential`'s. A checkpoint
written from one model therefore names its parameters as the other reads them.

### CrossEntropyDifferentiatesToEveryParameterOfTheConvModel

The stack's output goes into `NN::CrossEntropyLoss`, and the loss is differentiated against
the six parameters the walk yields. Every gradient is a value other than `NO_VALUE` with the
shape of its parameter. The case checks that the graph differentiates; it computes no number.

## test_layer_stack.cpp

Tests the runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread.

This file needs no device. A LayerStack is host-lane, and the first property this file pins,
that it names its parameters exactly as Sequential does, is a property of the walk and not of
any memory. The second is the per-sample `Shape` that `NN::ILayer::output_shape()` reports
and `NN::build_layer_stack` hands from each layer to the next factory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

### NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

### KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

### TracingTwiceDoesNotDuplicateParameters

`Parameter::declare` is idempotent, and the run-time path must inherit that and not reintroduce
the duplicate-on-retrace bug at this layer. Both traces go into one graph, because a
parameter's id names a value in the graph that declared it. Retracing into a second graph would
reuse an index that means something else there.

### AnAdapterBuiltFromAShapeReportsIt

A `LayerAdapter<GELU>` constructed with the shape [4, 5, 3] returns that shape from
`output_shape()`.

### AnAdapterBuiltFromAWidthReportsARankOneShape

A `LayerAdapter<Linear>` constructed with the width 8 reports the shape [8] and
`output_features()` 8. `make_stack` builds its layers through this constructor.

### OutputFeaturesEqualsTheElementCountOfTheOutputShape

Through an `ILayer` reference, `output_features()` of an adapter of shape [4, 5, 3] is 60 and
equals `output_shape().num_elements()`. `output_features()` is a non-virtual function of
`ILayer` in `include/SushiAI/nn/ilayer.hpp`, derived from the one virtual.

### ReportsTheLastLayersOutputShape

The stack of `make_stack` reports [4]. With a `GELU` adapter of shape [2, 2, 1] pushed after
it, the stack reports [2, 2, 1] and `output_features()` 4.

### RefusesAnOutputShapeWhenEmpty

`output_shape()` of an empty stack throws with a message containing "no layers" and "output
shape".

### BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer

A `Config::ModelSpec` of 6 input features with `linear` 8, `gelu` and `linear` 4 builds three
layers that report [8], [8] and [4]. The last layer is cast to `LayerAdapter<Linear>` and its
`in_features` is 8, the width the activation before it reported. The file never states that
number, so it can only come from the shape `build_layer_stack` hands on.

### AnActivationFactoryKeepsARankThreeInputShape

`NN::make_gelu` given the per-sample shape [4, 4, 3] returns a layer that reports [4, 4, 3]
and 48 features. A factory that reduced its input to a width would report [48].

### ALinearFactoryRefusesARankThreeInputAndNamesFlatten

`NN::make_linear` given [4, 4, 3] throws with a message containing the context, "[4, 4, 3]"
and "flatten". Given [48], the same object builds.

## test_layout.cpp

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### EveryValueIsNhwcUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

### FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data.

### LayoutConvertIsAllowedToChangeIt

LAYOUT_CONVERT is the one exemption, and the reason the check runs per node and not as a single
sweep over every value.

### DeclareLikeInheritsTheSourceValuesLayout

Fourteen tasks still to come each emit a value and carry its layout forward by hand.
`declare_like` is the one place that does it, so a forgotten carry cannot happen.

### Crossed

Both rebuilding passes copy the value table entry by entry, so that copy is what the `Crossed`
graph tests, and not the pass's own subject. The chain is deliberately plain for the same
reason: a fusion candidate or a precision decision in it would only obscure which field went
missing.

### FusionCarriesTheLayoutTagAcrossItsRebuild

`fuse()` rewrites the node list and copies the value table. A field it forgets to copy is reset
to `add_value`'s default, which is NHWC. A dropped tag is therefore a wrong layout, not a
missing one, and every shape check downstream still agrees with it.

The cost of a dropped tag shows in `Lowering::record_layout_convert`, which decides whether to
relabel a side into [N,H,W,C] by reading these two tags. A genuine NCHW -> NHWC convert that
reads as NHWC -> NHWC applies no transpose and copies transposed data straight through.

### CastInsertionCarriesTheLayoutTagAcrossItsRebuild

Cast insertion is the other rebuilding pass and could make the same omission. Trainer runs both
passes unconditionally, so either one is enough to lose the tag on the default path.

### ARebuiltGraphStillProvesUniform

The consequence is worth stating separately. With every tag reset to one value, `finalise()`
passes for the wrong reason. A graph that is uniform because it was flattened cannot be told
from one that was uniform to begin with, so the proof has to be re-run against tags that
survived.

### TheThreeRulesThatEmitZerosCannotBeReachedWithAnNchwOperand

The test checks why the zeros exposure is latent and not live, and does not leave that to a
comment. It is the premise the `zeros_like()` change rests on, and it is a premise about other
emitters' guards, which is the kind that stops being true without anyone editing this file.

The two `EXPECT_THROW` lines on `slice_channels` and `channel_affine` cover the rank-4 sites,
`slice_channels_backward` and `channel_affine_backward`. CV-3 makes their forward emitters
refuse an NCHW operand outright, so their dy can never carry that tag.

The `reshape` to `rows` covers the rank-2 site, `sum_rows_backward`. Reshape carries an NCHW
tag into rank-2 land with no guard at all, so unlike the two rank-4 sites the emitters let this
chain be built.

The last `EXPECT_THROW(graph.finalise(), Error)` shows what blocks that chain: the far end.
`differentiate()` only seeds a loss produced by NLL_LOSS, whose one-hot operand is a leaf and so
always NHWC, and nothing but LAYOUT_CONVERT may cross back, which is rank-4 only. `finalise()`
rejects that node before differentiation ever runs, so the rule is never reached with an NCHW
gradient. The guard is the pre-existing proof in `finalise()`, not a check in any emitter, and
that is why it is worth a test and not a sentence.

### ZerosLikeCarriesTheTagAndPlainZerosCannot

The three backward rules now depend on this contract. ZERO reads nothing, so plain `zeros()` has
no operand to inherit from and its NHWC tag is a default, not an answer. That is harmless for
the ZERO node itself, which the per-node proof skips for having no inputs. The consumer that
reads those zeros beside a real operand is the node that would be seen crossing layouts.

### FinaliseStillPropagatesRequiresGrad

The rename must not quietly drop the behaviour the old name described. The assertion is the
same one `test_ir.cpp` made.

## test_leaky_relu.cpp

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReluDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

### is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

### is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
the trace throws `Error`. It uses `Test::capture_error` from `tests/common/test_throws.hpp`,
so an exception of another type does not count as a rejection.

### TheDefaultSlopeIsOneHundredth

A default-constructed layer has slope 0.01.

### ForwardEmitsOneLeakyReluNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

A [2, 4] input traces to a [2, 4] value of the input's element type.

### ForwardKeepsTheShapeAndNhwcTagOfAnImageBatch

A [2, 8, 6, 3] input traces to a value of the same shape tagged NHWC.

### ForwardCarriesTheLayoutOfItsInput

The input is converted to NCHW first. The result is tagged NCHW and has the converted value's
shape. The activation is elementwise and reads no axis, so it accepts either layout and
passes the tag on, where the spatial layers refuse NCHW.

### OutputShapeAgreesWithForwardForAFeatureVector

`output_shape([7])` is [7], and the traced shape of a [5, 7] input is a batch of it.

### OutputShapeAgreesWithForwardForAnImage

`output_shape([8, 6, 4])` is [8, 6, 4], and the traced shape of a [3, 8, 6, 4] input is a
batch of it.

### ForwardRejectsASlopeOfZero

A slope of exactly 0 is rejected. The range is open at this end.

### ForwardRejectsASlopeOfOne

A slope of exactly 1 is rejected. The range is open at this end too.

### ForwardRejectsANegativeSlope

A slope of -0.01 is rejected.

### ForwardRejectsASlopeAboveOne

A slope of 1.5 is rejected.

### ForwardRejectsANanSlope

A NaN slope is rejected. NaN compares false against both bounds, so a check written as "throw
when the slope is at or below 0 or at or above 1" lets it through.

### ForwardAcceptsSlopesJustInsideTheOpenRange

The smallest positive normal float32 and 1 minus the float32 epsilon are both accepted. Each
lies inside the range and still differs from its bound after narrowing to float32.

### OutputShapeRejectsASlopeOutsideTheOpenRange

For the five slopes the forward cases reject, `output_shape([4])` throws with a message
containing "LeakyReLU" and "slope". A factory calls `output_shape()` and not `forward()`, so
this is the check that refuses a slope when a model file is read.

### ForwardNamesTheLayerAndTheSlopeItRefuses

`forward()` with slope 1.5 throws with a message containing "LeakyReLU" and "slope".

### ARejectedSlopeEmitsNothing

After `forward()` throws for a slope of 1, the node count and the value count are unchanged.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## test_loss_scaling.cpp

Tests the dynamic loss-scaling rule on LossScaleState, with no device involved.

This file tests the dynamic-scaling rule with no device in the measurement.

LossScaleState is split out of LossScaler so that this file can exist. "The scale doubles after
exactly N clean steps" is a property of the rule, not of a kernel. Asserting it through a
training run would make every one of these cases cost a compile and a replay while proving less.

The control matters as much as the case. A growth test that only ran N clean steps and saw a
doubling would pass against an implementation that doubled after every step, so each growth
assertion in this file is paired with the N-1 run that must not double.

### OneStepShortOfTheIntervalTheScaleDoesNotMove

This case is the control for TheScaleDoublesAfterExactlyTheGrowthInterval. Without it, "the
scale doubled after four clean steps" is equally true of an implementation that doubles after
one.

### AnOverflowRestartsTheRunOfCleanSteps

Three clean steps since the overflow is one short of the interval, so nothing has grown back.
The scale is still the backed-off one.

### BackoffStopsAtTheFloorButStillSkipsTheStep

At the floor the scale cannot go lower. The step is still garbage and must still be thrown
away, or an overflowed gradient reaches the weights the moment the backoff bottoms out.

### ANonDyadicConstantIsRefusedByName

Every constant is a power of two so that scaling and unscaling change only the exponent field.
A scale of 3 would work and would quietly make every gradient in the model carry a rounding
error, so it is refused.

### EveryReachableScaleHasAnExactReciprocal

The loss-scaling design is arranged around this property, and the test asserts it on the rule
itself. Whatever sequence of backoffs and growths the machine takes, the scale is a power of
two, so the reciprocal is exact and the round trip is lossless.

## test_max_pool2d.cpp

Tests the 2-D max pooling layer's trace, shape inference and rejections.

This file needs no device. `NN::MaxPool2d` in `include/SushiAI/nn/max_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter. `GraphBuilder::maxpool2d` returns the pooled value and
an INT64 argmax together; the layer returns the first.

Two `static_assert` lines at file scope require `MaxPool2d` to be a `Module` and not a
`ModuleContainer`.

### make_pool

`make_pool` builds a layer from a kernel and a stride per axis and an optional top and left
padding, which default to 0.

### traced_shape

`traced_shape` traces a layer over a batch of the given per-sample shape and returns the
result's shape.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### ForwardHalvesTheSpatialExtentsOfAnEvenInput

A 2x2 stride-2 pool traces a [2, 8, 6, 3] input to [2, 4, 3, 3].

### ForwardTagsItsResultNhwc

The traced result is rank 4 and tagged NHWC.

### ForwardReturnsThePooledValueNotTheArgmax

The result has the input's element type, and that type is not INT64. A layer that returned
the second result of the node would hand the next layer the index buffer.

### OutputShapeOfAnUnpaddedSquareWindow

`output_shape([8, 6, 3])` of a 2x2 stride-2 pool is [4, 3, 3].

### OutputShapeAgreesWithForwardForASquareWindow

For the same pool and sample, the traced shape of a batch of two equals `output_shape()` with
the batch extent in front.

### OutputShapeAgreesWithForwardForAPaddedAnisotropicWindow

A 3x2 window with stride 2 by 1 and a top padding of 1 maps [5, 7, 4] to [3, 6, 4], and the
traced shape agrees.

### OutputShapeAgreesWithForwardWhenTheStrideLeavesARemainder

A 3x3 stride-2 window maps [8, 10, 1] to [3, 4, 1], and the traced shape agrees. Neither
extent divides evenly, so the case covers the rounding down.

### OutputShapeAgreesWithForwardForADilatedAnisotropicWindow

A 2x2 unit-stride window with dilations 2 by 3 maps [7, 9, 2] to [5, 6, 2], and the traced
shape agrees.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` throws on a [2, 16] input.

### ForwardRejectsAWindowThatDoesNotFit

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ForwardRejectsAnNchwInput

A [2, 8, 8, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with a vertical stride of 0 throws with a message containing "MaxPool2d" and
"stride 0,2".

### ForwardRejectsANegativePadding

`forward()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAZeroStride

`output_shape()` throws when the vertical stride is 0.

### OutputShapeNamesAZeroStride

The message for the same layer contains "MaxPool2d" and "stride 0,2", so it reports the
stride and not a window that does not fit.

### OutputShapeRejectsANegativePadding

`output_shape()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 8, 3])` throws.

### OutputShapeRejectsAWindowTallerThanTheInput

A 5x2 window over [4, 8, 3] throws. The window fits the width, so the height check refuses.

### OutputShapeRejectsAWindowWiderThanTheInput

A 2x5 window over [8, 4, 3] throws. The window fits the height, so the width check refuses.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace.

## test_memory_plan.cpp

Tests that the memory planner never shares a byte between two values live together.

This file needs no device: no engine, no allocator, no SYCL. The memory planner is a pure
function of the IR, which is the reason the IR is a host-lane structure, and every case is a
synthetic graph and a table of offsets.

The contract is one sentence: two values whose live ranges intersect never share a byte, and
two whose live ranges do not are free to. The first half is checked exhaustively over every
pair. The liveness it is checked against is computed in the test file by a breadth-first search,
`descendants()`, and not by asking the planner what it thinks. A planner checked against its own
reachability code would agree with itself whatever it did.

The individual cases are for the second half, and the diamond in
TwoUnorderedBranchesNeverShareBytes is the one that matters. It is the shape where a planner
that overlaps disjoint [first_use, last_use] intervals on a linearised schedule gets a wrong
answer that no test written from the linear model would catch.

The cases fall into six groups, in file order:

- degenerate graphs
- the two shapes that decide whether the pass is right
- lifetimes
- aliasing
- mixed dtypes
- knobs, refusals and malformed input

### AChainPingPongsBetweenTwoBuffers

The graph is x -> a -> b -> c -> d. Nothing branches, so `a` is finished before `c` starts and
`b` before `d`. Four activations need exactly two buffers, which is the most any planner can do
with a chain.

### TwoUnorderedBranchesNeverShareBytes

This is the case the planner is judged by. Linearised, the two branches have disjoint
[first, last] intervals: a1 dies at node 2 and b1 is born at node 3. An interval planner
overlaps them and is wrong. Nothing in the graph orders node 2 before node 3 and the runtime may
run the branches concurrently, so the sharing would corrupt the result and also serialise the
parallelism it was meant to leave free.

### OnlyTheNonPersistentClassesArePacked

The graph holds all six lifetimes. The four persistent ones live for the whole run, so packing
them would save nothing and would merge the dependency-tracker shards that matter most. Only
ACTIVATION and SCRATCH reach an arena, and they reach different ones.

### AValueNothingWritesKeepsItsOwnAllocation

An ACTIVATION with no producer is written from outside the step and never rewritten, so its
bytes are live for the whole run whatever its class says. Handing them away would survive
exactly one replay.

### AnAccumulateChainIsOneBufferWithOneLiveRange

ACCUMULATE's result is its accumulator's storage. Planning the two separately would be silently
wrong and not merely wasteful: the planner would think the accumulator died at the ACCUMULATE
node and hand its bytes to `sink`, which is produced by the node still reading them.

### EveryStartIsAWholeNumberOfItsOwnElements

Alignment is turned down to 1 so that the element-width floor is the only thing left doing the
work. A 20-byte float32 buffer at offset 0 leaves the next free byte at 20, which is not a whole
number of float64 elements. The float64 buffer therefore has to start at 24, or its
storage_offset would not be representable at all.

### TooManyNodesDeclinesOverlapRatherThanApproximatingIt

The refusal is the point. A planner that quietly fell back to overlapping linear intervals
would produce a plan that is wrong in the way TwoUnorderedBranchesNeverShareBytes describes, and
would report a saving for it.

### AGraphThatIsNotInTopologicalOrderThrows

GraphBuilder cannot produce this graph, because it always emits a producer before anything can
read it. The whole pass rests on the property all the same, so the test checks it. A closure
built over a back edge would silently miss ordering and the plan would overlap things it must
not.

### ThePlanIsDeterministicForTheSameGraph

The offsets are byte-identical across two builds. A plan that shuffled would move every tensor
and take the library's bit-for-bit replay guarantee with it.

### TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes

`FusedForm::GEMM_BIAS_GELU` writes two values from one node: its result and the pre-activation
its epilogue spills. They have the same shape, they are born at the same instant, and the
planner packs by liveness. "One died before the other was written" is the mistake available to
it, and taking it would have the GELU computed from its own output with nothing to show for it.

Nothing special prevents it, which is why it is worth pinning. `precedes()` asks whether every
use of the earlier buffer strictly reaches the later one's first write, and a node does not
reach itself. The planner therefore refuses both directions and the two values conflict. That is
an inference from an existing rule and not a rule of its own, so a future rewrite of
`precedes()` could take it away silently.

## test_module.cpp

Tests the parameter walk over layers, containers and a layer enumerated at run time.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds four groups: the walk over the library's own layers and containers, the
tutorial's custom layer, the run-time arm of the walk, and the per-sample output shape of
`Linear` and the four activations of `include/SushiAI/nn/layers.hpp`.

### AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

### SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

### TracingTwiceDoesNotDuplicateParameters

The second trace emits nodes but declares no new parameters, so the value count grows by
exactly the two intermediates.

### The tutorial's custom layer

`MyOffset` is the custom layer of `docs/getting_started/TUTORIAL.md` §6, compiled.

A tutorial whose code sample does not compile is worse than no tutorial, and prose cannot be
made to fail a build. So the sample lives in this file as well, and the two must be kept
identical: if you edit one, edit the other.

TheTutorialsCustomLayerTracesAndIsWalkable and TheTutorialIsRightThatMulWouldNotBroadcast also
pin the two claims §6 makes about the sample: that bias_add is required and is not a stylistic
choice, and that the two ways of getting `fields()` wrong fail differently.

### TheTutorialsCustomLayerTracesAndIsWalkable

The test checks the role the tutorial gives the parameter, because the role decides the
initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.

### TheTutorialIsRightThatMulWouldNotBroadcast

§6 says bias_add is required and not merely preferred, because `mul` goes through
`require_same` and this builder has no implicit broadcast anywhere. That is the whole reason the
sample reads the way it does, so the test asserts it and does not leave it as a remark.

### The run-time arm of the walk

`WalkableStub` exercises the third arm of the walk: a layer whose parameters are enumerated at
run time.

The stub is deliberately not a LayerStack. This file tests the walk, and the walk's contract is
`walk_parameters(prefix, ParameterVisitor)`. Anything that satisfies it is enumerated through
it, whether or not it is one of the library's own types. Testing the walk against LayerStack
would tie its test to a type that arrived two commits later.

### AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

### TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
restates it so that the property is visible in the test that covers the walk, and so that it
names the detector and not the arm's consequences.

### Per-sample output shapes

Every layer type has `Shape output_shape(const Shape&) const`, which maps the per-sample shape
that enters the layer to the one that leaves it. A layer factory asks the layer for that shape
and does not restate the rule. The six cases below hold `Linear`, `ReLU`, `GELU`, `Tanh` and
`Sigmoid` to it, and compare it with what `forward()` traces.

### traced_shape

The file-local `traced_shape` traces a layer over a [2, features] input and returns the
result's shape.

### LinearReportsItsOutFeaturesAsThePerSampleOutputShape

`Linear{4, 3}` maps [4] to [3].

### LinearsOutputShapeAgreesWithItsTrace

The traced shape of `Linear{4, 3}` over a batch of two is [2, n], where n is the one extent
of `output_shape([4])`.

### LinearRefusesAPerSampleShapeOfTheWrongRank

`output_shape([4, 2, 1])` throws with a message containing "Linear(4, 3)", "[4, 2, 1]" and
"[4]". The first extent equals `in_features`, so only the rank check can refuse this shape.

### LinearRefusesAPerSampleShapeOfTheWrongWidth

`output_shape([7])` throws with a message containing "Linear(4, 3)", "[7]" and "[4]".

### AnActivationReportsItsInputShapeAsItsOutputShape

For the shapes [5] and [4, 4, 3], each of the four activations returns the shape it was given.
The rank-3 shape is the one a convolution hands on.

### AnActivationsOutputShapeAgreesWithItsTrace

The traced shape of each activation over a [2, 5] input is [2, n], where n is the one extent
of its `output_shape([5])`.

## test_objective.cpp

Tests the objective seam: the leaves it declares, the loss it emits and the resulting IR.

This file needs no device. Declaring leaves and emitting a loss are IR-to-IR, so the whole
objective seam is testable without an Engine. Only `score()` needs a tensor, and the integration
tier covers that through the trainer.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIr. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

### TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

### TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

### CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

## test_ops.cpp

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`include/SushiAI/core/ops.hpp`. These tests cover what that assert cannot: that the table is
complete, and that the reverse lookup behaves.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is IM2COL + MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReluDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

## test_precision.cpp

Tests the mixed-precision policy and where the cast insertion pass places its casts.

This file needs no device. The precision policy is a table of judgements and the cast pass is a
pure function of the IR, as the fusion pass and the memory planner are.

The cases pin placement, not survival. A mixed-precision graph that runs proves almost nothing:
a missing cast throws in SushiBLAS, so "it ran" only rules out the loud half of the failure
space. The quiet half, a cast in the wrong place, leaves a graph that trains and learns worse.
So every case is one of these:

- the whole-graph invariant (every node's operands agree with the type it computes in), asserted
  by walking the graph and not by spot checks,
- a control (the same graph under the fp32 policy must get zero casts, computed and not
  short-circuited), or
- one of the two places the master-weight arrangement can silently break: a parameter retyped in
  place, or a gradient that reaches its fp32 buffer without being widened.

The cases fall into five groups, in file order:

- the policy itself
- the control: the fp32 policy changes nothing
- the invariant
- master weights, and the direction that goes silently wrong
- what the pass promises about ids, reuse and repetition

### Traced

The `Traced` graph is built by hand and not through NN::Sequential, so that the test can name
every value it wants to assert about. It is also the smallest graph that still contains all four
interesting shapes: a parameter read by both passes, a reduction, a loss, and a gradient buffer.

### OnlyTheProductAndWhatAGemmFoldsIntoItNarrow

The `narrows` list states the rule set as a table, so that adding an operation to the narrowing
set is a visible edit in this test and not a silent behaviour change. Everything absent from it
stays in the graph's own precision, which is the safe answer for an operation nobody has thought
about yet.

The LEAKY_RELU entries are in `narrows` because LEAKY_RELU joined `Ops::ACTIVATIONS` with
`GradientSource::OUTPUT`. It narrows by the same substitution argument as RELU.

The `stays` list opens with the reductions and anything that sums many terms. fp16 carries about
11 bits of significand and SushiBLAS's own reduction header documents a 2048 ceiling for HALF,
so these are the operations that go quietly wrong.

DYNAMIC_SCALE is in `stays` because it carries the gradient loss scale, the one number in the
graph that exists to keep values out of fp16's subnormals. Narrowing the node that applies it
would overflow the very quantity it protects; see `include/SushiAI/optim/loss_scaler.hpp`.

GELU, GELU_BACKWARD and GEMM_BIAS_GELU, the one fused form that folds GELU, are in `stays`. The
form is a GEMM and rule 1 would otherwise take it. Narrowing it puts the tanh approximation's
`x*x*x` into fp16, which overflows at |x| ~ 40.3. Rule 5 in
`include/SushiAI/graph/precision.hpp` says so, and also explains why that leaves rules 2 and 5
consistent and not merely quiet.

The twenty spatial and CV identities are in `stays` because none of them is a GEMM epilogue or
an activation this policy folds. Each gets the same safe default as everything else nobody has
special-cased: it stays at the graph's own precision.

The closing `EXPECT_EQ` on the two list sizes holds because every identity is accounted for by
`narrows` or by `stays`. A new operation cannot join the enum without someone deciding which
list it belongs to.

### GeluStaysWideEvenThoughACatalogEntryNowFoldsIt

Rule 5 used to rest on two arguments, and one of them is gone. `FusedForm::GEMM_BIAS_GELU`
folds GELU by spilling the pre-activation, so "nothing folds it, therefore nothing forces its
precision" is no longer a reason for anything. The surviving argument is the arithmetic: the
tanh approximation computes `x*x*x`, which overflows fp16 at |x| about 40.3. A unit test cannot
assert that without a device.

The first three assertions check what a unit test can: that the conclusion did not quietly
follow the dead premise. The catalog can fold GELU and the policy still keeps it wide, which is
the pair of facts rule 5 in `include/SushiAI/graph/precision.hpp` now asserts.

`narrows_under_mixed_precision(GEMM_BIAS_GELU)` is false because the fused form has to go with
GELU. Otherwise the cast pass would narrow a node whose epilogue is the very expression rule 5
refuses to put in fp16.

### ChoosingIsIdempotentForEveryOperation

`include/SushiAI/graph/precision.hpp` states this contract, and it is what makes the invariant
checkable after the pass has run and not only while it runs. Without it,
`find_dtype_mismatches` would report a graph the pass had just finished making correct.

### AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType

The checks on the fp64 and fp16 graphs say that an fp64 graph keeps fp64 masters and still gets
fp16 products, and that an fp16 graph is already at the floor and is left alone, not "narrowed"
to something wider.

The `to_single` policy narrows to fp32. That is a legal thing to ask for, and it is what an fp64
model would use.

### TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Unit_CastInsertion cases measures nothing.

### EveryNodesOperandsAgreeWithTheTypeItComputesIn

This assertion is worth more than a dozen spot checks. It walks the whole graph and states the
rule, so it keeps holding as operations are added.

### TheInvariantSurvivesTheFusionPassThatRunsAfterIt

Fusion rewrites nodes and re-pairs operands, so it is where a correctly cast graph could stop
being one, and where a master weight could get swallowed.
`include/SushiAI/graph/cast_insertion.hpp` argues the ordering, and this test is the assertion
behind the argument.

### ADeliberatelyBrokenGraphIsReportedRatherThanAccepted

`find_dtype_mismatches` is the test file's own instrument, so it has to be shown to fail on a
graph that is wrong. Otherwise every "no mismatches" in the earlier cases could come from a
checker that never looks.

`mismatches.size()` is 3 because the graph is an uncast fp32 matmul under a mixed policy: two
operands and one result, all three float32 where the policy says half.

### AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy

The fp16 value the GEMM reads is a different value from the parameter. A cast produces it, and
it lives with the step's scratch and not for the whole run.

### AGradientIsWidenedBeforeItReachesItsBuffer

The test covers the backward direction. A matmul computing in fp16 produces an fp16 gradient,
and if that reached the fp32 gradient buffer uncast, SushiBLAS would throw. If the buffer had
been retyped to fp16 instead, nothing would throw and the optimizer would accumulate in fp16,
which is the failure master weights exist to prevent.

The loop over `gradients[0]` and `gradients[2]` covers the weights only. A backward MATMUL,
which computes low, produces the gradients of w1 and w2. SUM_ROWS, which does not, produces
those of b1 and b2, so only the weights need widening.

The loop over `gradients[1]` and `gradients[3]` covers the biases. Their gradients come out of
a reduction that stayed wide, so nothing was widened into them and nothing needed to be.

### AccumulatingGradientContributionsHappensWide

A parameter read twice gets two gradient contributions and an ACCUMULATE to sum them. Two things
have to hold at once. The sum runs in the wide type, because it is a sum. Its result and its
first operand agree, because the lowering makes the result alias the accumulator's buffer, so a
disagreement there is a wrong-width write and not merely a wrong number.

### AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot

This case covers the half of the pinning rule that a lifetime class does not. The logits are an
ordinary activation. The training loop reads them back by id for accuracy, and a caller who
forgets to say so gets an fp16 buffer where their host-side argmax expects floats.

### AValueReadTwiceIsConvertedOnce

w2 is read by the forward matmul and again by the backward one. Two conversions would be two
kernels and two buffers for one number, and the reuse is safe because the IR is
single-assignment.

### AGraphThatIsNotInTopologicalOrderIsRefused

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first.

### Pooled

MAXPOOL2D is the first operation in this IR whose results are not all one element type: it
writes the pooled tensor and, beside it, the INT64 argmax its backward reads. The cast pass
derives one precision per node from `outputs[0]`, which makes the argmax look like a value in
the wrong type and not like a value with no precision to choose.

`x` is a parameter and not an input for the reason
`tests/functional/regression/test_argmax_lifetime.cpp` gives: MAXPOOL2D_BACKWARD is only emitted
when something upstream of the pool needs a gradient.

### AnIndexBufferHasNoPrecisionToChoose

The pass forced every operand and every result of a node to one dtype, derived from the node's
first result. For MAXPOOL2D that is FLOAT32 from the pooled tensor, so the INT64 argmax read as
a mismatch and the pass tried to convert it. `GraphBuilder::cast` refuses anything but
real-float to real-float by design, so the whole build threw.

The defect does not depend on the policy. `wanted` comes from `outputs[0]`, not from the
policy's `choose()`, so the uniform policy reaches it too.

### APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer

This is the loud half of the same defect. An unpinned argmax is retyped in place and the
mismatch disappears silently. A pinned one keeps its id and its INT64 buffer, so the pass
reaches for the widening cast, and CAST is real-float to real-float by design.

## test_provenance.cpp

Tests the forward origin that differentiation records for every node it appends.

This file needs no device. `Autograd::differentiate()` fills a `Provenance` while it appends
backward nodes, so every case differentiates a host-side graph and reads the record back.
`include/SushiAI/eager/README.md`, under `autograd/provenance.hpp`, states the origin rule and
what the partitioner needs it for.

### TwoLinearGraph

`TwoLinearGraph` is a two-layer perceptron with a cross-entropy loss: an [8, 6] input, a 6-to-4
linear layer, a RELU, a 4-to-3 linear layer, then LOG_SOFTMAX and NLL_LOSS. It keeps the ids the
cases name: the four parameters, the result of the first MATMUL and the loss.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. It passes no seed factor and hands the `Provenance` pointer through unchanged,
which lets one case pass null.

### nodes_of

`nodes_of` returns the ids of the nodes that run one operation, starting at a given node index.
A case passes `forward_node_count` as the start, so the search covers the appended nodes only.

### EveryBackwardNodeHasAForwardOrigin

`forward_node_count` equals the node count taken before differentiation, `origin` holds one
entry per appended node, and every entry is below `forward_node_count`. An origin at or above
that count would name a backward node, which no segment of the forward trace holds.

### ARuleEmitsUnderTheNodeItDifferentiates

The node that produces the first weight's gradient is an appended node, and its origin is the
MATMUL that produces `first_product`. The case asserts the operation and the node id, so an
origin that named another MATMUL of the graph fails it.

### AFanOutAccumulateBelongsToTheProducer

`shared` is a RELU result that a TANH and a SIGMOID both read, so its gradient has two
contributors. The backward nodes hold exactly one ACCUMULATE, and its origin is the RELU that
produces `shared`. `x` is read once, so no closing sum is emitted and the count stays at one.

### AParameterReadTwiceAccumulatesUnderItsLeaf

The parameter `x` is read by a RELU and a TANH. The node that produces its gradient is an
ACCUMULATE, and the origin of that node is the LEAF of `x`.

`targets` is declared before `x` so that the LEAF of `x` is not node 0. The reverse walk ends at
node 0, so with `x` there an origin taken from the walk's last position would equal the LEAF's
id and the `EXPECT_NE(origin, 0u)` line could not tell the two apart.

### AReusedProvenanceHoldsOnlyTheLatestRun

One `Provenance` is filled by a first differentiation and then passed to a second one over a
`TwoLinearGraph`. Its `forward_node_count` and `origin` equal those of a fresh `Provenance`
filled from an identical graph. A `differentiate()` that appended to `origin` and did not reset
it would leave the first run's entries in front.

### ARequestedValueWithoutAProducerIsRefused

`x` is added with `add_value` and so has no LEAF node, and it is read twice, so the closing sum
of its gradient needs an origin that does not exist. The call throws, and the message contains
"provenance" and "producer node".

### PassingNoProvenanceChangesNothing

Two identical graphs are differentiated, one with a `Provenance` and one with null. The returned
gradient ids are equal, the node counts are equal, and every node has the same operation,
inputs and outputs. Recording is therefore an observation of the transform and not a branch in
it.

## test_region_signature.cpp

Tests the signature that `signature_of` computes for one region of a partition.

This file needs no device. `Eager::signature_of` reads a graph, a partition and one identity per
value, all of which a case builds on the host. `include/SushiAI/eager/README.md`, under
`region_signature.hpp`, lists what the hash covers and gives the reason for each operand rule.

Most cases build two graphs that differ in one property and compare the two signatures. Where
the operation list is the same in both, the case asserts that first, so the difference it then
asserts is a difference in the hash alone.

### Region keys and identities

`K_BODY` is key 1, the one region of a hand-built partition, and `K_READER` is key 2, the region
that reads from it in a two-region partition. `K_PARAMETER_IDENTITY` and `K_EXTERNAL_IDENTITY`
are the default identities of the parameter and of the value `K_READER` reads from outside.
They are stated once so that a case names the one it changes.

### Traced

`Traced` holds a graph, its partition and one identity per value id, which are the three things
`Eager::signature_of` reads.

### signature_of

The file-local `signature_of` has three overloads. The first takes a `Traced` and a region key
and calls `Eager::signature_of` with the identity vector as a span. The other two take a
`Recipe` or an `ExternalRecipe`, build its graph and return the signature of `K_BODY` or of
`K_READER`.

### Recipe

`Recipe` describes `scale(combine(relu(x), tanh(x)), alpha)` over a parameter `x`, and each of
its fields is one property a case varies: the shape, the element type, the SCALE factor, SUB in
place of ADD, the operand order of the combine, and the identity of `x`. `build(Recipe)` puts
the four operations under `K_BODY` and the LEAF under no region, and gives every value but `x`
identity zero.

### node_of

`node_of` returns the first node of a graph that runs a given operation. The attribute and
layout cases use it to reach the node they edit after `build` returns.

### assign_regions

`assign_regions` gives a `Traced` its partition from one key per non-LEAF node, in node order,
and a region list. A LEAF gets `K_NO_REGION`. It also resets every identity to zero, so a
builder that calls it sets afterwards only the identities its case is about.

### ExternalRecipe

`ExternalRecipe` describes a graph whose `K_READER` region holds one SCALE that reads a value
the region does not produce. Its fields choose whether that value is the parameter `x` or the
RELU result of `K_BODY`, whether `K_BODY` appears in the region list, and which identity the
value has.

### ProbeOperands

`ProbeOperands` names four operand orders for a hand-built three-operand ADD in `K_READER`.
`build(ProbeOperands)` leaves every identity at zero, and the one in-region operand is result 0
of the region's node 0. Both orders of a pair therefore feed the hash the same words, and the
two signatures can differ only if the kind of each operand enters the hash.

### build_reader_of_result

`build_reader_of_result` adds a RELU node by hand with two results and a TANH that reads the
result at the given index, both under `K_BODY`. The RELU is added through `Graph::add_node`
because `GraphBuilder::relu` returns one result.

### with_identity

`with_identity` records an identity for one value and returns the value id, so that
`ScopedPerceptron` can assign an identity in the expression that declares a leaf.

### info_of

`info_of` returns the `RegionInfo` of a key, or null when the partition does not list the key.

### role_identity

`role_identity` hashes a region's name, its direction, how many nodes of the region follow the
producing node, and the index of the value among that node's results. Counting from the
region's end keeps the identity of a region's last result the same when nodes are added before
it.

### identify_produced_values

`identify_produced_values` gives every value that a region's node writes its `role_identity`.
In the tests it stands in for the caller of `signature_of`, which owns the storage a boundary
value is bound to; `include/SushiAI/eager/README.md` says why only the caller can supply that
identity.

### ScopedPerceptron

`ScopedPerceptron` traces three scopes, "stem", "body" and "head", then differentiates and
partitions the graph with `ScopePartitioner`. Its argument is the number of linear layers in
"stem". Leaves get fixed identities and produced values get role identities. `key_of` returns
the key of a region by name and direction, and `stem_output` is the value "body" reads from
"stem".

### TheSameTraceGivesTheSameSignature

Two graphs built from the default `Recipe` have equal hashes and compare equal. The operation
list is RELU, TANH, ADD, SCALE, which is the region's nodes in order with the LEAF left out.

### AChangedAttributeChangesTheHash

A SCALE factor of 3.0 in place of 2.0 leaves the operation list equal and makes the hash
differ.

### AChangedLastAttributeSlotChangesTheHash

The case writes 1.0 into slot `Attributes::CAPACITY - 1` of the SCALE node and expects another
hash. It covers the last slot because a hash that read a fixed number of leading slots would
agree with the other attribute case and miss this one.

### AChangedLayoutChangesTheHash

The RELU result is retagged NCHW in one of two otherwise equal graphs, and the hashes differ.
Shape and element type are the same in both, so the layout is the only input that moved.

### AChangedShapeChangesTheHash

A shape of [4, 5] in place of [4, 3] leaves the operation list equal and makes the hash differ.

### AChangedDtypeChangesTheHash

FLOAT64 in place of FLOAT32 leaves the operation list equal and makes the hash differ.

### AChangedOpChangesTheHash

SUB in place of ADD makes both the operation list and the hash differ. It is the one
`Recipe` case in which the two lists are unequal.

### ADifferentParameterChangesTheHash

The same structure over a parameter with identity `K_PARAMETER_IDENTITY + 1` has the same
operation list and another hash. The parameter is external to the region, so its identity is
the only thing the hash holds about it.

### AnEarlierRegionGrowingDoesNotChangeALaterRegion

`ScopedPerceptron(1)` and `ScopedPerceptron(2)` differ by one linear layer in "stem". The three
`ASSERT` lines at the top establish that the case tests what it claims: the value "body" reads
has a different id in the two graphs, the same identity in both, and the forward and backward
regions of "body" have different keys.

For each direction, the signature of "body#0" has a non-empty operation list and is equal in
the two graphs, by hash and by `operator==`. The forward signature of "stem#0" differs between
them, which shows that the two graphs are not equal throughout.

### ARewiredOperandChangesTheHash

`add(relu(x), tanh(x))` and `add(tanh(x), relu(x))` have the same operation list and different
hashes. Both operands are produced inside the region, so the case covers the in-region operand
position.

### AReadOfAnotherResultChangesTheHash

A TANH that reads result 1 of a two-result node hashes apart from one that reads result 0. The
producer's index in the region is the same in both, so the result index is what separates them.

### ABoundaryValueOfAnotherIdentityChangesTheHash

The SCALE in `K_READER` reads the RELU result of `K_BODY`. With identity
`K_EXTERNAL_IDENTITY + 1` for that value, the operation list is equal and the hash differs.

### ALeafAndABoundaryValueOfOneIdentityHashAlike

The reader's signature is equal, by hash and by `operator==`, whether its operand is the
parameter `x` or the result of another region, as long as both carry `K_EXTERNAL_IDENTITY`.
Where an external operand comes from in the graph does not enter the hash.

### AnOperandFromAnUnlistedRegionHashesAsExternal

The same graph is signed with `K_BODY` present in the region list and with it absent, and the
two signatures of `K_READER` are equal. An operand whose producer has a key other than the
region's is external whether or not the partition lists that key.

### AnOperandInsideTheRegionHashesApartFromOneOutside

The probe node reads one result of `K_READER` and the result of `K_BODY` twice, in the two
orders `InsideThenOutside` and `OutsideThenInside`. The operation lists are equal and the
hashes differ.

### ALeafOperandHashesApartFromOneInsideTheRegion

The same pair of orders with the leaf `x` in place of the `K_BODY` result,
`InsideThenLeaves` and `LeavesThenInside`, also gives equal operation lists and different
hashes.

### EqualHashesWithDifferentOpsCompareUnequal

Three signatures are written by hand with hash 42. The one listing ADD and the one listing SUB
compare unequal under `==` and `!=`, and two that both list ADD compare equal. A real collision
cannot be produced on demand, so the case constructs one.

### ARegionThePartitionDoesNotListThrows

`signature_of` throws for key 6 and for `K_NO_REGION`. Each message names the key, "Region 6"
and "Region 0", and contains "not listed".

### APartitionOfAnotherNodeCountThrows

One entry is removed from `region_of_node`, which leaves four entries for a five-node graph.
The message contains "assigns 4 nodes" and "holds 5".

### AnExternalValueWithoutAnIdentityThrows

The identity vector is cleared, so the id of `x` lies beyond its end. The message contains
"External value 0" and "no identity". The call throws and does not read past the span.

## test_rematerialise.cpp

Tests the structure of the graph that the IM2COL rematerialisation pass rewrites.

This file needs no device. The IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Integration_Rematerialise in `tests/functional/integration/test_rematerialise.cpp`
does that by running the same graph twice.

The brief's three cases are all in this file, with two deviations that the library forced:

1. `Autograd::differentiate()` refuses any root that is not a scalar produced by NLL_LOSS
   (`src/autograd/differentiator.cpp` explains why every other root would silently return empty
   gradients). The brief's graphs stopped at a MATMUL, so each is carried on to a loss. The
   shapes, geometry and expectations are the brief's.
2. A "backward consumer" cannot be identified by node index, because the pass runs when the
   backward nodes are already in the graph and no marker separates the phases. The rule the pass
   applies is "every consumer after the first", which is the same set for every graph
   differentiation produces and is well-defined for one it did not produce.

### TheColumnBufferStopsBeingAnActivation

This case checks the number CV-10 is about. ResNet-18's first block at batch 128 holds
`128*32*32*64*9*4` bytes, about 302 MB, in this one buffer, and there are twenty convolutions.
Cloning the gather next to its backward consumer drops the lifetime to SCRATCH, so one buffer is
live at a time and not twenty.

### TheCloneComputesTheSameThingAsTheOriginal

The operand and attribute checks guard against a clone that read a different operand or carried
different attributes. Such a clone would produce a plausible column matrix and a wrong weight
gradient. Both are checked because the geometry lives in the attributes and is easy to drop when
copying a node.

The check on `copy` expects the same description and the same layout. The layout is a
graph-wide invariant, and a clone that dropped it would fail `finalise()` on a graph that is
numerically correct.

### TheCloneIsInsertedAheadOfEveryConsumerItServes

The failure this case catches is not a wrong number. It is a graph whose insertion order stopped
being a topological order, with a recomputation appended after the GEMM that reads it.
`MemoryPlan::build()` rests on that order and says so by name, and a lowering that got past it
would record a GEMM against a buffer nothing had written yet.

### AnIm2colWithNoBackwardConsumerIsLeftAlone

This is the inference case. Nothing was differentiated, so there is no second consumer and
cloning would be a free slowdown.

### AForwardOnlyConvolutionIsLeftAlone

This is the other half of the inference case, and the one that would break if the pass cloned
for every consumer and not for every consumer after the first. The column matrix has exactly one
reader, the GEMM.

### RunningItTwiceAddsNothingTheSecondTime

Once the rewiring is done the clone has one consumer of its own, so the pass has no reason to
rematerialise its own output. If it did, the pass would not terminate on a graph with three
consumers.

## test_rule_table.cpp

Tests that every domain's backward rules reach the combined table exactly once.

### EveryDomainsRulesReachTheCombinedTable

`include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

### NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

### TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

## test_scope_partitioner.cpp

Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.

This file needs no device. `ScopePartitioner::assign` reads a graph, the segments of its forward
trace and the provenance of its backward nodes, and returns a `Partition`.
`include/SushiAI/eager/README.md`, under `scope_partitioner.hpp` and `partition.hpp`, states the
key rule the cases check: with `m` segments, key `k` forward, key `2m - k` backward, and `2m` for
the optimizer.

The cases fall into three groups, in file order:

- keys and region lists of traced and differentiated steps
- graphs built by hand to reach a LEAF origin or a graph with no backward node
- inputs that `assign` refuses

### Step

`Step` holds one traced, differentiated and partitioned training step: the graph, the segments,
the provenance, the parameters, one gradient per parameter and the partition.

### as_span

`as_span` returns a segment vector as the span `assign` takes.

### finish

`finish` seals the trace, differentiates the loss against `step.parameters` with a `Provenance`
and partitions the result. Every step builder ends with it, so the four steps differ only in
what they trace.

### linear

`linear` emits a MATMUL and a BIAS_ADD over a new weight and bias and appends both to the
parameter list. The parameter order is therefore weight, bias per layer, which is how a case
indexes `gradients`.

### unscoped_step

`unscoped_step` builds two linear layers and a loss with no scope open, which gives one segment.

### three_segment_step

`three_segment_step` builds scopes "first" and "second" with one linear layer and an activation
each, and a scope "loss". It has three segments, six leaves and seven other forward nodes.

### shared_weight_step

`shared_weight_step` declares one weight in a scope "weights", multiplies by it in "first" and
again in "second", and closes with "loss". It has four segments, and the first holds only a
LEAF.

### repeated_block_step

`repeated_block_step` opens a scope "block" twice in a loop, with one linear layer each time,
then "loss". The input is declared before the first scope, so the trace opens with an unnamed
segment that holds only a LEAF.

### TwoReluGraph

`TwoReluGraph` holds a graph built without differentiation: a parameter, a RELU in a segment
"first" and a RELU of that in a segment "second". `build_two_relu_graph` fills it, seals the
trace and sets `forward_node_count`, and leaves `origin` empty. A case then appends backward
nodes and writes their origins by hand, which reaches arrangements `differentiate()` does not
emit.

### is_leaf

`is_leaf` reports whether a node of a graph is a LEAF.

### segment_number

`segment_number` returns the number, counted from 1, of the segment that holds a forward node,
or 0 when none does. It is computed from the segment ranges and not from the partition, so a
forward key is compared against something the partitioner did not write.

### backward_keys_of

`backward_keys_of` returns the keys of the backward nodes whose origin is a non-LEAF node of
one segment. It skips LEAF origins because those take their key from their operands.

### AnUnscopedStepIsOneRegion

With one segment, every non-LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

### ForwardKeysFollowTraceOrder

Every non-LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the loop skips nodes it should compare.

### BackwardKeysMirrorForwardKeys

With three segments the optimizer key is 6, and the backward nodes whose origins lie in segments
1, 2 and 3 have keys 5, 4 and 3. Each segment must have at least one such node, so an empty
result cannot satisfy the loop.

The region list has five entries with keys 1 to 5, named "first#0", "second#0", "loss#0",
"second#0", "first#0". The first three are forward and the last two backward.

### TheLastSegmentSharesItsRegionWithItsBackward

The NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

### EveryOperandComesFromALowerOrEqualKey

For each of the four steps, every node's key is at least the key of the node that produces each
of its operands. Operands with no producer, or whose producer has no region, are skipped. The
check runs over every node of the graph and not over chosen ones, and each step must contribute
at least one comparison. `include/SushiAI/eager/README.md` gives this as the condition for a
valid partition under `RegionKey`.

### AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that no longer contains the case, and the message on the last assertion says so.

### ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an ACCUMULATE whose origin is a LEAF. Its two
operands are produced under keys 5 and 6, the backward keys of "second" and "first" among four
segments, and the ACCUMULATE has key 6. Key 7, the backward key of the segment that holds the
weight's LEAF, is not used for it.

### ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder

Two ADD nodes are appended to a `TwoReluGraph` with the LEAF as their origin. One reads the
key 2 value first and the key 1 value second, the other the reverse, and both get key 2. A rule
that took the key of the first or of the last operand would give one of them key 1.

### RegionsListOnlyKeysThatHoldANode

`shared_weight_step` has four segments and optimizer key 8. Its region list has five entries
with keys 2 to 6. Key 1 belongs to the LEAF-only segment "weights" and key 7 to its backward
mirror, no node carries either, and neither is listed.

### ARepeatedScopeIsNamedByItsOccurrence

`repeated_block_step` has four segments and optimizer key 8. The region list has keys 2 to 6,
named "block#0", "block#1", "loss#0", "block#1", "block#0", so the backward regions carry the
names of the forward regions they mirror.

### AGraphWithNoBackwardNodeHasForwardRegionsOnly

A `TwoReluGraph` with an empty `origin` gives its two RELU nodes keys 1 and 2 and optimizer
key 4. The region list holds "first#0" and "second#0", both forward. The optimizer key is `2m`
whether or not any backward node exists.

### LeavesBelongToNoRegion

In `three_segment_step` every LEAF has `K_NO_REGION` and every other node has a key. The count
of six leaves, two inputs and four parameters, shows that the loop reached the LEAF branch.

### SegmentsThatDoNotCoverTheForwardGraphThrow

Four malformed segment lists are refused, each with its own message: a list missing its last
segment ("The segments cover nodes 0 to", "forward graph has"), a list with a gap before segment
1 ("Segment 1 ('second')", "contiguous"), a list with an empty segment inserted ("Segment 1
('empty')", "holds no node"), and an empty list ("at least one segment").

### AProvenanceOfAnotherGraphThrows

A provenance with one origin too few is refused with a message containing "The provenance
describes" and "backward nodes". One whose last origin equals `forward_node_count`, the first
id that is not a forward node, is refused with "has origin" and "forward nodes".

### ABackwardNodeThatReadsAHigherKeyThrows

Two RELU nodes are appended to a `TwoReluGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
throws, and the message contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

### ALeafOriginNodeThatReadsOnlyLeavesThrows

An ADD of two parameters is appended with a LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The message contains "as its origin" and "reads only
leaves".

## test_scope_trace.cpp

Tests how ScopeTrace cuts a trace into segments and which calls it rejects.

This file needs no device. A `ScopeTrace` reads the node count of a graph, so every case
appends nodes to a host-side graph, opens and closes scopes around them and inspects what
`seal()` returns. `include/SushiAI/eager/README.md`, under `segment.hpp` and `scope_trace.hpp`,
defines a segment and the names a scope may have.

The cases fall into two groups, in file order: the segments a trace yields, and the calls
`open()`, `close()` and `seal()` reject.

### trace_nodes

`trace_nodes` appends a given number of ADD nodes to a graph, each adding one input value to
itself. The input is added as a value with no LEAF node, so the node count rises by exactly the
number asked for and a case can state every range as a literal.

### segment_is

`segment_is` compares one segment with an expected name, occurrence, begin and end. On a
mismatch it prints all four fields of both, so a failure shows which field differs.

### NodesOutsideAnyScopeFormOneUnnamedSegment

Three nodes traced with no scope open give one segment: empty name, occurrence 0, nodes 0 to 3.

### AScopeSplitsTheTraceIntoThreeSegments

One node, then two in a scope "body", then three give three segments. The unnamed run after the
scope has occurrence 1, because the unnamed run before it was occurrence 0.

### ANestedScopeIsNamedByItsPath

Inside "outer" and "inner" the depth is 2 and the segment is named "outer/inner". The nodes of
"outer" before and after it form two segments named "outer", with occurrences 0 and 1.

### AScopeOpenedTwiceGetsRisingOccurrences

A scope "cell" opened three times in a loop, with two nodes each time, gives three segments
named "cell" with occurrences 0, 1 and 2 and consecutive ranges.

### AnEmptyScopeLeavesNoSegment

A scope "cell" that closes with no node traced leaves no segment. The next "cell", which holds
one node, has occurrence 0, so the dropped scope did not use up an occurrence.

### SealWithAScopeStillOpenThrows

`seal()` throws while "outer" and "inner" are open, and the message contains "seal()", "still
open" and the path "outer/inner". The depth is still 2 afterwards.

The case then traces one more node, closes both scopes and seals. The result is one segment
"outer/inner" over nodes 0 to 2, which shows the failed `seal()` did not cut the running
segment.

### CloseWithNoScopeOpenThrows

`close()` throws on a new trace, and again after one matched `open()` and `close()`. Both
messages contain "close()" and "no scope open".

### SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

### OpenRejectsAnEmptyName

`open("")` throws with a message containing "open()" and "empty scope name", and the depth
stays 0.

### OpenRejectsANameWithASlash

`open("outer/inner")` throws. The message contains the name and "may not contain", and the
depth stays 0.

### OpenRejectsANameWithAHash

`open("cell#0")` throws. The message contains the name and "may not contain", and the depth
stays 0.

### ARejectedNameDoesNotSplitTheRunningSegment

One node is traced, `open("a/b")` throws, and one more node is traced. `seal()` returns one
unnamed segment over nodes 0 to 2. A trace that ended the running segment before it validated
the name would return two.

### ARejectedCloseDoesNotSplitTheRunningSegment

The same arrangement with a `close()` that throws for having no scope open also returns one
unnamed segment over nodes 0 to 2.

### ASecondSealThrows

A second `seal()` throws with a message containing "seal()" and "sealed trace".

### OpenAndCloseAfterSealThrow

After `seal()`, `open("late")` and `close()` both throw with "sealed trace" in the message, and
the depth stays 0.

## test_shape.cpp

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
RuntimeContext, which is what keeps the unit tier fast.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

## test_spatial_shapes.cpp

Tests the shapes the builder infers for spatial operations and the layouts it refuses.

### Im2ColProducesTheColumnMatrixAGemmExpects

[N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5 windows per image. Each row is
one window's kernel_h * kernel_w * C taps with channels innermost, so a sub-group reads a
contiguous run (CV-3).

### Im2ColHonoursStrideAndDilation

A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows, and stride 2 over the same
input yields 2. Both numbers are wrong under the naive (H - K)/S + 1 that forgets dilation.

### Im2ColRejectsAWindowLargerThanThePaddedInput

The oversized window produces a zero extent, which Shape refuses anyway. The message a caller
gets should still name the geometry, not the Shape invariant it happened to trip.

### Col2ImReturnsToTheImageItWasAskedFor

col2im cannot derive its output extents. ConvGeometry carries pad_top and pad_left but no
pad_bottom or pad_right, so an asymmetric padding would be silently mis-shaped. The caller
states the image shape instead. It always has it, because it held the value im2col consumed.

### Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward

A 3x3 stride-1 window with padding only on the bottom and right of a 13x13 input still yields
13x13 windows, which is not what the symmetric formula (13-3)/1+1 = 11 would say. The forward
pass (`im2col_with_extents`) states 13x13 outright. The backward rule must be able to state the
same numbers back, because it is the one holding them, and must not re-derive them from
ConvGeometry alone.

The geometry leaves pad_top and pad_left at 0. Padding was applied on the bottom and right
only, which ConvGeometry cannot express and does not need to.

### Col2ImDerivingStillRejectsThatSameAsymmetricPair

This case is the point of having `col2im_with_extents` as a twin. The deriving col2im assumes
symmetric padding and is not a substitute for it. Feeding it the exact cols/image pair that
`col2im_with_extents` accepted in Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward must
still throw, which proves the twin is needed.

### Im2ColRejectsAnNchwInput

im2col reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it would read channels
as height and produce a plausible but silently wrong shape, where it should throw.

### MaxPoolProducesAValueAndItsArgmaxTogether

The indices are a second result of the node, not a second output tensor the caller allocates.
The backward rule needs exactly the indices this forward produced, and a pair returned together
cannot be mismatched the way two separate calls could.

### MaxPoolHandlesYoloTinysAsymmetricThirteenToThirteen

YOLOv3-tiny's last pool is 2x2 stride 1 over 13x13, padded only on the bottom and right, staying
13x13. ConvGeometry cannot express that padding, so the caller states the extents. This case
proves the explicit overload is needed.

### AvgPoolProducesOnlyAValue

avgpool2d records no indices: its backward divides by a constant window area and needs nothing
the forward recorded (CV-6).

### MaxPoolRejectsAnNchwInput

maxpool2d reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it must throw and
must not pool the channel axis.

### SpatialMeanReducesToNAndC

The result is the [N,C] that the numeric Probe's loss needs, which is why every convolutional
gradient test in this plan ends with this call.

### BatchNorm2dKeepsItsShapeAndDeclaresTwoPerChannelStatistics

A non-positive epsilon is a division by the standard deviation itself, which is exactly zero
for a constant channel.

### BatchNorm2dRejectsAnNchwInput

The layout check runs before any extent is read, so the operand that is wrong is the one the
message names.

### ReshapeRefusesToChangeTheElementCount

A reshape has one failure mode. It moves no data, so a mismatched count is not a resize; it is
a read past the end.

### Convolution, window extents and upsampled extents

The cases from here cover three additions to `include/SushiAI/graph/builder.hpp`.
`GraphBuilder::conv2d` emits a convolution as a node sequence: a RESHAPE of the rank-4 weight
to [C_out, taps], an IM2COL of the input, a MATMUL of the two with its second operand
transposed, an optional BIAS_ADD, and a RESHAPE of the rows to [N, out_h, out_w, C_out].
`Graph::window_extents` computes the output extents of a window and validates its geometry,
and `conv2d`, `im2col`, `maxpool2d` and `avgpool2d` all derive their extents through it.
`Graph::upsampled_extents` does the same for UPSAMPLE2D.

### op_names_from

`op_names_from` returns the operation names of a graph's nodes from a given index on. A case
records the node count after it has declared its operands and passes that count, so the list
holds only what the emitter under test added.

### square_window

`square_window` returns a unit-stride geometry with a square kernel and the same padding on
both axes.

### window

`window` returns a unit-stride unpadded geometry with the given kernel height and width.

### Conv2dReturnsAnNhwcValueWithOneChannelPerFilter

A [2, 5, 5, 3] input, an [8, 3, 3, 3] weight and an [8] bias under a padded 3x3 window give a
[2, 5, 5, 8] value tagged NHWC.

### Conv2dDividesTheOutputExtentsByItsStride

A padded 3x3 window with stride 2 over a [2, 8, 6, 3] input gives [2, 4, 3, 4]. The call
passes `NO_VALUE` as the bias.

### Conv2dWithABiasEmitsIm2colMatmulAndBiasAddBetweenTwoReshapes

The nodes the call adds are, in order, "sushiai.reshape", "sushiai.im2col", "sushiai.matmul",
"sushiai.bias_add" and "sushiai.reshape". The sequence holds no operation of its own, which is
decision CV-1: `test_ops.cpp` asserts under ThereIsNoConv2dIdentity that no CONV2D identity
exists.

### Conv2dWithoutABiasEmitsNoBiasAdd

With `NO_VALUE` as the bias the same call adds four nodes: reshape, im2col, matmul, reshape.

### Conv2dContractsTheColumnMatrixAgainstTheTransposedFilterMatrix

The third node added is a MATMUL. Its first operand has shape [2 * 5 * 5, 3 * 3 * 3], one row
per window, and its second has shape [8, 3 * 3 * 3], one row per filter. Flag 0 is false and
flag 1 is true, so the filter matrix is the transposed operand and the product has one column
per filter.

### Conv2dPointwiseEmitsNoIm2col

A 1x1 unit-stride unpadded window adds reshape, reshape, matmul, reshape: the input is
reshaped to one row per pixel and no IM2COL is emitted. The result is [2, 5, 5, 8].

### Conv2dStridedOneByOneKernelStillEmitsIm2col

A 1x1 window with stride 2 over a [2, 6, 6, 3] input adds reshape, im2col, matmul, reshape and
gives [2, 3, 3, 8]. A stride above one selects pixels, which a reshape cannot do, so the
shortcut of the previous case must not apply.

### Conv2dTakesANonSquareKernelInHeightThenWidthOrder

An [8, 1, 3, 3] weight under a 1x3 window over a [2, 5, 5, 3] input gives [2, 5, 3, 8]. Axis
1 of the weight is the kernel height and axis 2 the kernel width.

### Conv2dRejectsAWeightWhoseKernelExtentsDifferFromTheWindow

An [8, 3, 1, 3] weight under the same 1x3 window throws. It holds as many elements as
[8, 1, 3, 3], so the reshape that follows accepts it and only the check in `conv2d` refuses
the transposed kernel.

### Conv2dRejectsAWeightWhoseInputChannelsDifferFromTheImage

An [8, 3, 3, 4] weight over a three-channel input throws with a message containing "conv2d".
The reshape rejects the same weight by element count in a message that does not name the
convolution, so the case requires the name.

### Conv2dRejectsAFlattenedWeight

An [8, 27] weight throws. The weight enters `conv2d` as rank 4 and is flattened inside it.

### Conv2dRejectsARankThreeInput

A [5, 5, 3] input, which lacks the batch axis, throws.

### Conv2dPointwiseRejectsAnNchwInput

A [2, 5, 5, 3] input is converted to NCHW and paired with an [8, 1, 1, 5] weight, whose
input-channel extent matches the last axis of the converted shape. The pointwise path skips
IM2COL, which is the emitter that checks the layout, so `conv2d` has to check it itself. The
call throws.

### Conv2dRejectsABiasWhoseExtentDiffersFromTheFilterCount

A [4] bias with eight filters throws.

### Conv2dRejectsAWindowLargerThanThePaddedInput

A 5x5 unpadded window over a [1, 2, 2, 3] input throws.

### Conv2dEmitsNothingWhenItRejectsItsOperands

Three calls throw: an oversized window, a weight with four input channels over a
three-channel input, and a [4] bias with eight filters. The node count afterwards equals the
count taken before the calls. `conv2d` validates every operand before it emits its first
reshape, so a refused call leaves no orphan node in the graph.

### Conv2dBuildsAGraphThatFinaliseAccepts

A biased 3x3 convolution feeds a pointwise one, and `Graph::finalise()` does not throw. Both
paths therefore leave every value with a layout tag the per-node proof accepts.

### Conv2dDifferentiatesToAWeightAndABiasGradient

The convolution's result goes through `spatial_mean`, `log_softmax` and `nll_loss`.
`Autograd::differentiate` returns two gradients other than `NO_VALUE`, of shapes [8, 3, 3, 3]
and [8]. The weight gradient has the rank-4 shape of the parameter and not the [8, 27] shape
of the reshaped operand the MATMUL read.

### WindowExtentsPadBothSidesOfAnAxisByTheLeadingPad

A 3x5 window with a top pad of 1 and a left pad of 2 over a 7 by 9 input yields 7 by 9. The
extents are unchanged only if each pad is applied to both sides of its axis.

### WindowExtentsHonourStrideAndDilationPerAxis

A 3x3 window with stride 2 by 3 and a height dilation of 2 over an 11 by 16 input yields 4 by
5. The two axes differ in stride and in dilation, so a value taken from the wrong axis changes
one of the two numbers.

### WindowExtentsOfAWindowThatExactlyFitsAreOneByOne

A 5x5 window with padding 1 over a 3 by 3 input yields 1 by 1. The padded input is exactly as
large as the window.

### WindowExtentsRejectANonPositiveKernel

A kernel of 0 by 3 and one of 3 by -2 throw. Each message contains "positive kernel" and the
kernel as given, "kernel 0x3" and "kernel 3x-2".

### WindowExtentsRejectANonPositiveStrideAndNameIt

A vertical stride of 0 and a horizontal stride of -1 throw with "positive stride" and the
pair, "stride 0,1" and "stride 1,-1". A zero stride is a division by zero in the extent
formula, and the message names the stride and not a window that does not fit.

### WindowExtentsRejectANonPositiveDilation

A horizontal dilation of 0 throws with "positive dilation" and "dilation 1,0".

### WindowExtentsRejectANegativePadding

A padding of -1 on both axes throws with "padding -1,-1", and a left pad of -1 alone throws
with "padding 0,-1".

### WindowExtentsRejectAWindowThatDoesNotFit

A 5x5 window over a 4 by 6 input throws with a message containing "5x5 window" and "4x6
input". The window fits the width and not the height.

### UpsampledExtentsScaleEachAxisByItsOwnFactor

A 3 by 5 input with scales 2 and 4 yields 6 by 20.

### UpsampledExtentsRejectAScaleBelowOne

Scales of 0 by 2 and of 2 by -1 throw with "at least one" and the pair, "0x2" and "2x-1".

### UpsampledExtentsRejectAProductBeyondSixtyFourBits

With a scale of 2^62, an extent of 5 on the height axis throws with "fit in 64 bits", and so
does an extent of 5 on the width axis. Each call overflows one axis and leaves the other at
1, so both products are checked.

### UpsampledExtentsAcceptTheLargestProductThatFits

Half of the largest int64, rounded down, scaled by 2 yields the largest int64 minus 1. The
overflow check therefore refuses nothing that 64 bits hold.

### UpsampleRejectsAScaledExtentBeyondSixtyFourBits

`GraphBuilder::upsample2d` with a horizontal scale of 2^62 over a width of 5 throws with "fit
in 64 bits", and the node count is unchanged.

### TheDerivingEmittersRejectANegativePadding

`im2col`, `conv2d`, `maxpool2d` and `avgpool2d` are each called with a padding of -1, and each
message contains "padding". These are the four emitters that derive their extents from the
geometry.

### Col2ImRejectsANegativePadding

`col2im` with a padding of -1 throws with a message containing "padding".

### Col2ImRejectsAnImageShapeThatIsNotRankFour

`col2im` asked for the image shape [5, 5, 3] throws with a message containing "col2im" and
"rank-4".

### TheDerivingEmittersNameTheLayoutBeforeTheWindow

A [1, 8, 8, 2] input is converted to NCHW, which leaves 2 on the axis an NHWC reader takes
for the height, and a 3x3 window does not fit that. `im2col`, `maxpool2d` and `avgpool2d` each
throw a message containing their own name and "NHWC". A caller who passed the wrong layout is
told so, and is not told that the window is too large.

## test_unflatten.cpp

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `include/SushiAI/nn/unflatten.hpp` holds a
height, a width and a channel count and no parameter. It reshapes an
[N, height * width * channels] value to [N, height, width, channels]. Images enter a model
flat, as [N, features], so this layer is what gives a convolution its rank-4 input.

### per_sample

`per_sample` returns the shape of a graph value with the batch axis removed.

### SatisfiesTheModuleContract

Three `static_assert` lines require `Unflatten` to be a `Module`, not to be a
`ModuleContainer`, and to have no parameter walk of its own, so the walk reaches it through
`fields()`.

### TracesTheImageShape

`Unflatten{4, 5, 3}` traces a FLOAT64 [2, 60] input to a FLOAT64 [2, 4, 5, 3] value.

### TagsTheResultNhwc

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

### EmitsOneReshapeNode

The trace adds exactly one node. The producer of the result is a RESHAPE whose one operand is
the input.

### OutputShapeAgreesWithTheTraceForAnRgbImage

`output_shape()` of the input's per-sample shape equals the per-sample shape of the traced
result, and `output_shape([60])` is [4, 5, 3].

### OutputShapeAgreesWithTheTraceForAGreyImage

The same two assertions hold for `Unflatten{28, 28, 1}` over a [7, 784] input, the MNIST
geometry: `output_shape([784])` is [28, 28, 1].

### ForwardRejectsAZeroExtent

`Unflatten{4, 0, 3}` throws with a message containing "Unflatten needs positive extents". The
element-count check would also throw for this layer, in a message that does not name the
extents, so the case requires the layer's own message.

### ForwardRejectsANegativeExtent

`Unflatten{-4, -5, 3}` throws the same message. The two negative extents multiply to a
positive count of 60, which matches the input, so no later check refuses the layer by count;
`Shape`'s constructor would throw, in a message that does not name the layer.

### ForwardRejectsAnInputThatIsNotRankTwo

A [2, 60, 1, 1] input throws. Its second extent is 60 and it holds 120 elements, so only the
rank check can refuse it.

### ForwardRejectsAMismatchedElementCount

A [3, 40] input to `Unflatten{4, 5, 3}` throws with a message containing "Unflatten(4, 5, 3)"
and "inner extent is 60". `reshape` would also reject the input, in a message that does not
name the layer, so the case requires the layer's own.

### ForwardRejectsAnNchwInput

The layout tag of a [2, 60] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeRejectsAZeroExtent

`Unflatten{4, 5, 0}.output_shape([60])` throws with a message containing "Unflatten needs
positive extents".

### OutputShapeRejectsANegativeExtent

`Unflatten{-4, -5, 3}.output_shape([60])` throws the same message.

### OutputShapeRejectsAShapeThatIsNotRankOne

`output_shape([60, 1])` throws.

### OutputShapeRejectsAMismatchedElementCount

`output_shape([59])` throws.

### HasNoParameters

`parameter_count` is 0 before and after a trace and `parameter_elements` is 0. The trace
raises the value count by one, the reshaped value, so it declares no parameter.

## test_upsample2d.cpp

Tests the shape, layout and rejections of the Upsample2d layer.

This file needs no device. `NN::Upsample2d` in `include/SushiAI/nn/upsample2d.hpp` holds one
integer scale, 2 by default, and no parameter. It repeats every pixel of an NHWC image over a
scale-by-scale block.

Two `static_assert` lines at file scope require `Upsample2d` to be a `Module` and not a
`ModuleContainer`.

### traced_per_sample_shape

`traced_per_sample_shape` traces a layer over a new input of the given batched shape and
returns the result's shape with the batch axis dropped. The forward rejection cases call it
too, since it throws whatever `forward()` throws.

### K_OVERFLOWING_SCALE

`K_OVERFLOWING_SCALE` is 2^62. Its product with an extent of 5 does not fit in 64 bits.

### TheDefaultScaleDoublesHeightAndWidth

A default layer traces a [2, 3, 5, 7] input to [2, 6, 10, 7].

### TheScaleFieldSetsTheFactorOnBothSpatialAxes

`Upsample2d{3}` traces a [1, 4, 2, 5] input to [1, 12, 6, 5].

### TheResultIsTaggedNhwc

The traced result is rank 4 and tagged NHWC.

### ForwardEmitsOneUpsampleNodeCarryingTheScale

The trace of `Upsample2d{3}` adds exactly one node. The producer of the result is an
UPSAMPLE2D whose one operand is the input, and attribute slots 0 and 1 both hold 3.

### OutputShapeScalesHeightAndWidthOnly

A default layer maps [3, 5, 7] to [6, 10, 7], and `Upsample2d{4}` maps [2, 1, 9] to [8, 4, 9].
The channel extent is unchanged in both.

### OutputShapeAgreesWithTheTracedShape

For scales 2, 3 and 1, `output_shape()` equals the traced shape without its batch axis. A
scale of 1 is the smallest the layer accepts and leaves the shape as it is.

### OutputShapeRejectsAZeroScale

`Upsample2d{0}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=0".

### OutputShapeRejectsANegativeScale

`Upsample2d{-2}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=-2".

### OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws with a message containing "Upsample2d", the expected form
"[H, W, C]" and "[16]".

### OutputShapeRejectsAScaledHeightThatOverflows

With `K_OVERFLOWING_SCALE`, `output_shape([5, 1, 7])` throws with a message containing
"Upsample2d" and "fit in 64 bits". The width is 1, so the height is the product that
overflows.

### OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws. The shape still carries its batch axis.

### ForwardRejectsAZeroScale

`forward()` of `Upsample2d{0}` over a [1, 2, 2, 3] input throws with a message containing
"Upsample2d" and "[2, 2, 3] with scale=0".

### ForwardRejectsARankTwoInput

`forward()` over a [2, 12] input throws with a message containing "Upsample2d", "rank-4" and
"[2, 12]".

### ForwardRejectsAScaledWidthThatOverflows

With `K_OVERFLOWING_SCALE`, `forward()` over a [1, 1, 5, 3] input throws with a message
containing "Upsample2d" and "fit in 64 bits". The height is 1, so the width is the product
that overflows.

### ForwardRejectsAnNchwInput

A [1, 3, 2, 2] input is converted to NCHW, and `forward()` throws with a message containing
"upsample2d" and "NHWC". The lower-case name is the builder's: `GraphBuilder::upsample2d`
raises this message.

### TheParameterWalkVisitsNothing

After a trace the walk visits nothing and `parameter_count` is 0.

## test_version.cpp

Tests that the runtime version agrees with the compile-time macros.
