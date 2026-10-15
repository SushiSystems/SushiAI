# tests/layers/nn

The tests of the module `layers/nn`, one file per header of `modules/layers/nn/include/`. The suite prefix of a case gives its ctest label.

## test_layer_stack.cpp

Suites: `Unit_LayerStack`.

Tests the runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread.

This file needs no device. A LayerStack is host-lane, and the first property this file pins,
that it names its parameters exactly as Sequential does, is a property of the walk and not of
any memory. The second is the per-sample `Shape` that `NN::Layer::output_shape()` reports
and `NN::build_layer_stack` hands from each layer to the next factory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

### chain_row

`chain_row` returns one `Config::LayerSpec` of a given type and attributes that reads one given
source row.

### hand_built_chain

`hand_built_chain` fills a `Config::ModelSpec` field by field, with no parser involved: the
input shape [6] and the rows `linear` 8, `gelu` and `linear` 4. Row 0 reads
`Config::K_MODEL_INPUT` and each later row reads the row above it.

### NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

### KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

### ReportsTheLastLayersOutputWidth

The stack of `make_stack` has a size of 3, is not empty and reports 4 output features.

### TracesTheLayersInOrder

`forward` of that stack over a [4, 6] input gives a [4, 4] value.

### RefusesToTraceWhenEmpty

`forward` of an empty stack raises an error holding "no layers" and "identity".

### RefusesAnOutputWidthWhenEmpty

`output_features` of an empty stack raises an error holding "no layers".

### RefusesANullLayer

`push(nullptr)` raises an error holding "null layer" and "index 0".

### ReportsAnOutOfRangeLayerIndex

On a stack of three layers `layer(2)` does not throw, and `layer(3)` raises an error holding
"layer 3" and "holds 3".

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

Through an `Layer` reference, `output_features()` of an adapter of shape [4, 5, 3] is 60 and
equals `output_shape().num_elements()`. `output_features()` is a non-virtual function of
`Layer` in `modules/layers/nn/include/SushiAI/nn/layer.hpp`, derived from the one virtual.

### ReportsTheLastLayersOutputShape

The stack of `make_stack` reports [4]. With a `GELU` adapter of shape [2, 2, 1] pushed after
it, the stack reports [2, 2, 1] and `output_features()` 4.

### RefusesAnOutputShapeWhenEmpty

`output_shape()` of an empty stack throws with a message containing "no layers" and "output
shape".

### BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer

The spec of `hand_built_chain` builds three layers that report [8], [8] and [4]. The last
layer is cast to `LayerAdapter<Linear>` and its `in_features` is 8, the width the activation
before it reported. The spec never states that number, so it can only come from the shape
`build_layer_stack` hands on.

### BuildLayerStackSeedsTheChainFromTheInputShape

The spec of `hand_built_chain` has its `input_shape` replaced by [5]. The first layer's
`in_features` is 5, so `build_layer_stack` seeds the chain from the shape.

### BuildLayerStackRefusesAHandBuiltRowWithoutASource

With the `from` list of row 1 cleared, `build_layer_stack` throws with a message containing
"layer 1 'gelu'" and "the row before it".

### OutputShapeOfAnInputIsTheLastLayersOutputShape

`LayerStack::output_shape([6])` of the built chain is [4] and equals the argument-free
`output_shape()`.

### RefusesAnOutputShapeOfAnInputWhenEmpty

`output_shape([6])` of an empty stack throws with a message containing "no layers" and "output
shape".

### AnActivationFactoryKeepsARankThreeInputShape

`NN::make_gelu` given the per-sample shape [4, 4, 3] returns a layer that reports [4, 4, 3]
and 48 features. A factory that reduced its input to a width would report [48].

### ALinearFactoryRefusesARankThreeInputAndNamesFlatten

`NN::make_linear` given [4, 4, 3] throws with a message containing the context, "[4, 4, 3]"
and "flatten". Given [48], the same object builds.

## test_leaky_relu.cpp

Suites: `Unit_LeakyReLU`.

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `modules/layers/nn/include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReLUDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

### is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

### is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
the trace throws `Error`. It uses `Test::capture_error` from `modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp`,
so an exception of another type does not count as a rejection.

### TheDefaultSlopeIsOneHundredth

A default-constructed layer has slope 0.01.

### ForwardEmitsOneLeakyReLUNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a K_LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

A [2, 4] input traces to a [2, 4] value of the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

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

### ForwardRejectsANaNSlope

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

## test_max_pool2d.cpp

Suites: `Unit_MaxPool2d`.

Tests the 2-D max pooling layer's trace, shape inference and rejections.

This file needs no device. `NN::MaxPool2d` in `modules/layers/nn/include/SushiAI/nn/max_pool2d.hpp` holds a
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

### ForwardTagsItsResultNHWC

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

### ForwardRejectsAnNCHWInput

A [2, 8, 8, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

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

### TrailingRowKeepsTheExtentOfAStrideOnePool

A 2x2 pool of stride 1 over [13, 13, 4] reports [12, 12, 4]. With a trailing pad of 1 on both
axes it reports [13, 13, 4], and a trace over a batch of 2 has that shape behind the batch.

### TrailingPadAtStrideTwoGivesSevenFromThirteen

A 2x2 pool of stride 2 with a trailing pad of 1 on both axes reports [7, 7, 4] over
[13, 13, 4], and its trace over a batch of 1 is [1, 7, 7, 4].

### TrailingPadGivesTwoHundredEightFromFourHundredSixteen

The same pool reports [208, 208, 16] over [416, 416, 16].

### TrailingPadAppliesToOneAxisOnly

A 2x2 pool of stride 1 with `trailing_pad_h` 1 alone reports [3, 2, 2] over [3, 3, 2].

### TrailingPadLetsAWindowFitAShortInput

A 2x2 pool of stride 1 throws `Error` from `output_shape` over [1, 4, 3]. With a trailing pad
of 1 on both axes it reports [1, 4, 3] and traces to [1, 1, 4, 3].

### OutputShapeRejectsANegativeTrailingPad

With `trailing_pad_w` -1, `output_shape` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "0,-1".

### ForwardRejectsANegativeTrailingPad

With `trailing_pad_h` -2, `forward` raises an error holding "MaxPool2d",
"non-negative trailing padding" and "-2,0", and the graph's node count does not change.

### OutputShapeRejectsATrailingPadThatOverflows

With `trailing_pad_h` at the largest `int64_t`, `output_shape` raises an error holding
"MaxPool2d" and "within 64 bits".

### OutputShapeRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool with `trailing_pad_h` 2 raises from `output_shape` an error holding "MaxPool2d",
"every window a tap inside the input" and "2,0".

### ForwardRejectsATrailingPadThatLeavesAWindowNoTap

A 2x2 pool of stride 2 with `trailing_pad_w` 3 raises from `forward` an error holding
"MaxPool2d", "every window a tap inside the input" and "0,3", and the graph's node count does
not change.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace.

## test_mish.cpp

Suites: `Unit_Mish`.

Tests `NN::Mish` in `modules/layers/nn/include/SushiAI/nn/mish.hpp`, which has no parameter and
no attribute.

Three `static_assert` lines at the top of the file hold that `Mish` is a `Module`, is not a
`ModuleContainer`, and that `Mish::fields()` is an empty tuple.

### ForwardEmitsSoftplusTanhThenMulReadingTheInput

`forward` adds three nodes in the order SOFTPLUS, TANH, MUL. SOFTPLUS reads the input, TANH
reads the SOFTPLUS result, and MUL reads the input and then the TANH result and writes the
value `forward` returns.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::NHWC`.

### ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## test_model_graph.cpp

Suites: `Unit_ModelGraph`, `Unit_ModelGraphDeathTest`.

Tests ModelGraph: what add and output refuse, what a trace emits, and the state names.

This file needs no device. `NN::ModelGraph` in `modules/layers/nn/include/SushiAI/nn/model_graph.hpp` holds
named nodes in the order they were added. `add` takes a name, a layer or an `NN::Node` and the
`NodeRef`s the node reads, and returns the `NodeRef` of the new node. `output` publishes a ref
under a name. `trace` returns the published values as an `NN::ModelOutputs`, and `forward`
returns the one published value. `add`, `output`, `port`, `output_ports` and `trace` return a
`Status` or a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelGraph` is a `TraceModule`,
is not a `Module`, and is detected by `Detail::HasStateWalk`.

### OpList

`OpList` is a vector of the operation of every node of a graph, in order.

### TwoPortNode

`TwoPortNode` is an `NN::Node` with one input and two ports. `forward` appends the ReLU and
then the GELU of its input. The one-argument constructor names the ports "low" and "high", and
the three-argument one takes the two names.

### BranchingModel

`BranchingModel` holds a `ModelGraph` and the refs `stem`, `up` and `mix` of three of its
nodes.

### build_branching_model

`build_branching_model` builds five nodes over an [8, 8, 3] sample and publishes nothing:
"stem", a same-padded `Conv2d(3, 4)` of kernel 3 and stride 2 on the input; "down", a
`Conv2d(4, 8)` of the same window on the stem; "up", an `Upsample2d{2}` on that; "merge", a
`Concat` of up and stem; and "mix", a pointwise `Conv2d(12, 4)` on the merge.

### ops_of

`ops_of` returns the `OpList` of a graph.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields under a prefix as
"name:extents" strings.

### add_relu_declared_as

`add_relu_declared_as` adds a `LayerNode` over a `LayerAdapter<ReLU>` constructed with a given
shape. The node declares that shape on its port whatever its input is.

### RefsNameTheInputAndTheNodesInTheOrderAdded

`model.input()` reports `is_input()` and port 0. The refs of stem, up and mix report the nodes
0, 2 and 4, mix reports port 0, stem and up compare unequal, and `size()` is 5.

### ShapeOfReturnsTheDeclaredPerSampleShapes

`shape_of` is [8, 8, 3] for the input, [4, 4, 4] for stem, [4, 4, 8] for up and [4, 4, 4] for
mix. Node 3, the merge, declares [4, 4, 12] on its port.

### BranchAndMergeTracesTheNodeSequenceOfTheHandWrittenTrace

The model's `forward` under an evaluation trace is compared with the same five layers traced
by hand through a `GraphBuilder` in a second graph. The two graphs hold the same operations in
the same order, and both results are [2, 4, 4, 4]. The sequence holds one K_UPSAMPLE2D and one
K_CONCAT_CHANNELS.

### MergeEmitsOneConcatChannelsReadingBothBranches

With stem, up and mix published, the traced graph holds one K_CONCAT_CHANNELS node. Its inputs
are the published value of up and then that of stem, and its output is [2, 4, 4, 12].

### TraceReturnsEveryPublishedOutputByName

With "features" and "skip" published in that order, `trace` returns two entries in that order.
`at` returns a different id for each name, and both values are [2, 4, 4, 4].

### ForwardReturnsTheSinglePublishedOutput

A model of one ReLU node whose result is published traces one K_RELU node. That node reads the
input, and its output is the id `forward` returned.

### ForwardFailsWhenTwoOutputsArePublished

With two outputs published, `forward` returns its input id and records `InvalidDescription` on
the trace with a message containing "exactly one model output", "features" and "skip". The
graph is marked refused.

### OutputShapeIsTheShapeOfTheSinglePublishedOutput

`output_shape([8, 8, 3])` is an empty `Shape` while nothing is published, [4, 4, 8] once up is
the one published output, and an empty `Shape` again once a second output is published.

### AGraphIsANodeOfAnotherGraph

A `ModelGraph` holding a `Linear{6, 8}` named "dense" is added to an outer graph as the node
"inner", with a ReLU after it. `shape_of` of the nested node is [8], the one output is
[4, 8], and the outer walk yields "inner.dense.weight:[8, 6]" and "inner.dense.bias:[8]".

### ANestedGraphThatPublishesTwoOutputsFailsTheOuterTrace

An inner graph that publishes "first" and "second" is added as a node. The outer `trace` fails
with `InvalidDescription` and a message containing "node 'inner'", "exactly one model output",
"first" and "second".

### ASequentialAndALayerStackAreNodesOfAGraph

A `Sequential{Linear{6, 8}, ReLU}` is added as "head" and a `LayerStack` of a `Linear{8, 4}`
and a GELU as "tail". `shape_of` is [8] and [4], the output is [4, 4], and the walk yields
"head.0.weight:[8, 6]", "head.0.bias:[8]", "tail.0.weight:[4, 8]" and "tail.0.bias:[4]".

### TheInputCanBePublishedAsAnOutput

A model with no node that publishes its input returns the input id from `forward` and emits no
node.

### WalkNamesStateByNodeNameInTheOrderAdded

After a trace of the branching model, the walk under "model" yields "model.stem.weight:[4, 3,
3, 3]", "model.down.weight:[8, 3, 3, 4]" and "model.mix.weight:[4, 1, 1, 12]", and
`NN::buffer_count` is 0. The upsample and the concat contribute no name.

### ATrainingTraceReachesAModalNodeAndTheWalkNamesItsBuffers

A model of one `BatchNorm2d(3)` node named "bn", traced under a training trace, emits one
K_BATCHNORM2D node and leaves two updates in the list. The parameter walk under "model" yields
"model.bn.weight:[3]" and "model.bn.bias:[3]", and `for_each_buffer` yields
"model.bn.running_mean" and "model.bn.running_var".

### AChainNamedByIndexYieldsTheNamesOfALayerStack

A model whose nodes are named "0", "1" and "2" holds a `Linear{6, 8}`, a GELU and a
`Linear{8, 4}`. A `LayerStack` holds the same three layers. After each is traced into its own
graph, the two parameter walks are equal, the two graphs hold the same operation sequence, and
the names are "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]" and "2.bias:[4]".

### NestsInASequentialUnderItsIndex

A `Sequential<ModelGraph, ReLU>` traces to a [4, 8] value whose last node is a K_RELU. Its walk
yields "0.dense.weight:[8, 6]" and "0.dense.bias:[8]".

### AddRejectsADuplicateName

A second node named "act" fails with `InvalidDescription` and a message containing "already
has a node named 'act'" and "index 0". `size()` stays 1.

### AddRejectsAnEmptyName

The name "" fails with "needs a name" and `size()` stays 0.

### AddRejectsANameWithADot

The name "a.conv" fails with "'a.conv'" and "contains '.'" through the layer overload, and
"b.act" fails the same way through the `NN::Node` overload. `size()` stays 0.

### AddReportsARefusedNameBeforeTheLayerSeesItsSource

A `Conv2d` added under a taken name, or under an empty one, on a [6] source fails with the
message about the name. The convolution would refuse a [6] source, and that failure is not the
one returned. `size()` stays 1.

### AddReturnsTheFailureOfALayerThatRefusesItsSource

`add("stem", Conv2d::same_padded(3, 4, 3, 2), input)` on a [6] sample fails with a message
containing "node 'stem'", "Conv2d(3, 4)" and "[6]", and `size()` stays 0.

### AddRejectsAnUnknownSource

A ref to node 1 of another graph, used as the source in a graph of one node, fails with a
message containing "node 'late'", "source node 1" and "holds 1 nodes". The `NN::Node` overload
fails with "node 'later'" and "source node 1". `size()` stays 1.

### AddRejectsASourcePortTheNodeDoesNotHave

A ref to port 1 of a two-port node of another graph, used on a graph whose node 0 has one
port, fails with a message containing "node 'late'", "'only' has 1 ports" and "port 1".

### PortSelectsANamedPortThatALaterNodeReads

`port(pair, "high")` returns a ref to node 0, port 1, and `port(high, "low")` returns a ref
equal to `pair`. A ReLU added on `high` and published traces three nodes. Node 1 is the GELU,
and node 2 reads node 1's output.

### PortRejectsAnUnknownNameAndTheInput

`port(pair, "middle")` fails with a message containing "node 'pair'", "'middle'" and "low,
high". `port(input, "low")` fails with "the model input has no port named 'low'".

### OutputPortsPublishesEveryPortUnderItsOwnName

`output_ports(pair)` publishes two outputs. `at("low")` is the output of graph node 0 and
`at("high")` that of graph node 1.

### OutputPortsPublishesNothingWhenAPortNameIsTaken

With "high" already published, `output_ports(pair)` fails with "port 'high' of node 'pair'",
and `output_ports(input)` fails with "the model input has no ports to publish". A trace then
returns one output, so the failed call did not publish "low" either.

### OutputRejectsATakenNameAnEmptyNameAndAnUnknownSource

A second `output("echo", ...)` fails with "already publishes an output named 'echo'". The name
"" fails with "needs a name". A ref to a node of another graph fails with "output 'far'",
"source node 0" and "holds 0 nodes".

### TraceRejectsADeclaredShapeThatDisagreesWithTheTracedOne

A node that declares [5] and traces a ReLU of a [4, 6] value fails the trace with
`InvalidDescription` and a message containing "node 'narrow'", "port 'output'", "[5]" and
"[4, 6]".

### TraceRejectsAnInputOfAnotherSampleShape

A model over a [6] sample traced with a [4, 7] value fails with "the model input", "[6]" and
"[4, 7]". Traced with a [6] value, which has no batch axis, it fails with "the model input"
and "[6]". The graph gains no node.

### TraceRejectsAModelThatPublishesNoOutput

A model with one node and no published output fails the trace with "publishes no output", and
the graph gains no node.

### AddRejectsASourceCountTheNodeDoesNotRead

A `Concat` built over two shapes and added with one source fails with a message containing
"node 'merge'", "reads 2 values" and "given is 1". A `TwoPortNode` added with two sources fails
with "node 'split'", "reads 1 values" and "given is 2". `size()` stays 0.

### AMovedGraphKeepsItsRefsAndLeavesAnEmptyUsableGraph

A graph move-constructed from the branching model has `size()` 5, resolves the ref `up` to
[4, 4, 8] and accepts `output("features", mix)`. The graph moved from has `size()` 0. A ReLU
added to it and published traces one node whose output is the published value.

### AccessorsAreFatalOutOfRange

The case is in the suite `Unit_ModelGraphDeathTest`. On an empty graph, `shape_of` of a ref to
another graph's node stops the process with "shape_of" and "source node 0", and `node(0)`
stops it with "ModelGraph::node" and "node 0".

### AddIsFatalOnANullNode

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a null `unique_ptr<NN::Node>`
stops the process with a report containing "null node" and "'ghost'".

### AddIsFatalOnANodeWithTwoPortsOfOneName

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a `TwoPortNode` whose ports
are both named "low" stops the process with a report containing "'pair'" and "two ports named
'low'".

### TraceIncrementalHandsEachLayerThePerSampleShapeItReads

A model of the sample shape [5, 6] holds one `SliceProbe`. `trace_incremental` in a carrying
trace over a [1, 1, 6] slice hands the probe the shape [5, 6] and adds one GELU node, and the
output "y" has the shape [1, 1, 6].

### TraceStillChecksThePerSampleShapeTheIncrementalWalkSkips

`trace` of the same model over the [1, 1, 6] slice fails with `InvalidDescription`, "[5, 6]"
and "[1, 1, 6]".

### TraceIncrementalNamesTheNodeThatHasNoIncrementalForm

With a `ReLU` named "act" after the probe, `trace_incremental` fails with `Unsupported`,
"node 'act'" and "no incremental form".

### TraceIncrementalNeedsACarryingTrace

`trace_incremental` in an evaluation trace fails with `InvalidDescription` and
"keeps nothing between runs", and adds no node.

### ANestedGraphTracesItsIncrementalForm

With the probe's graph added as one node of an outer graph, `trace_incremental` of the outer
graph hands the probe [5, 6] and adds one GELU node.

## test_model_outputs.cpp

Suites: `Unit_ModelOutputs`, `Unit_ModelOutputsDeathTest`.

Tests ModelOutputs: publishing named values and reading them back.

This file needs no device. `NN::ModelOutputs` in `modules/layers/nn/include/SushiAI/nn/model_outputs.hpp` is
the set of named `ValueID`s one trace of a model publishes. `add()` returns a `Status`, and
`at()` and `only()` return a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelOutputs` is final,
copy-constructible and nothrow move-constructible.

### boxes_and_scores

`boxes_and_scores` returns a set holding "boxes" as value 4 and "scores" as value 9, added in
that order.

### add_without_value

`add_without_value` adds "boxes" with `Graph::K_NO_VALUE`. It is the statement of the death
test.

### StartsEmpty

A new set's `all()` is empty.

### AllKeepsTheOrderOfAdd

After adding "scores", "boxes" and "features", `all()` holds those three entries in that order
with their ids 9, 4 and 0. The order is the order of the calls, not of the names or the ids.

### AddRejectsADuplicateName

A second `add("boxes", ...)` returns `InvalidDescription` with a message containing "'boxes'"
and "boxes, scores". The set still holds two entries and `at("boxes")` is still 4.

### AddRejectsAnEmptyName

`add("", ...)` returns `InvalidDescription` with "needs a name" and adds nothing.

### AddStopsOnNoValue

The case is in the suite `Unit_ModelOutputsDeathTest`. `add("boxes", K_NO_VALUE)` stops the
process with a report containing "'boxes'" and "no value".

### AtReturnsTheNamedValue

`at("boxes")` is 4 and `at("scores")` is 9.

### AtDoesNotMatchAPrefix

`at("box")` and `at("boxes.")` both fail with `InvalidDescription` and the name that was asked
for.

### AtOnAMissingNameListsTheNames

`at("masks")` fails with a message containing "'masks'" and "boxes, scores".

### AtOnAnEmptySetSaysNone

`at("boxes")` on an empty set fails with a message containing "'boxes'" and "none".

### OnlyReturnsTheSingleValue

`only()` on a set of one entry returns that entry's id, 12.

### OnlyFailsOnZero

`only()` on an empty set fails with a message containing "exactly one", "has 0" and "none".

### OnlyFailsOnTwo

`only()` on a set of two fails with a message containing "exactly one", "has 2" and "boxes,
scores".

## test_module.cpp

Suites: `Unit_Module`.

Tests Parameter across graphs and the state walk over layers, containers and fields.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds six groups: the walk over the library's own layers and containers, the
tutorial's custom layer, the run-time arm of the walk, the per-sample output shape of `Linear`
and the four activations of `modules/layers/nn/include/SushiAI/nn/layers.hpp`, a parameter traced into more than
one graph, and the state walk over parameters, buffers and children.

### AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

### ALinearLayerNamesItsWeightAndBias

After a trace, the walk of `Linear{4, 3}` lists `weight` [3, 4] and `bias` [3], and
`parameter_elements` is 15.

### ABiasFreeLayerSimplyHasNoBiasEntry

The walk of `Linear{4, 3, false}` lists `weight` [3, 4] alone.

### SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

### NestedContainersCompoundTheirIndices

The walk of a `Sequential` holding a `Sequential` of a `Linear{4, 4}` and a `ReLU`, then a
`Linear{4, 2}`, lists `0.0.weight` [4, 4], `0.0.bias` [4], `1.weight` [2, 4] and `1.bias`
[2].

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

### ParametersAreDeclaredAsGradientRequiringLeaves

After a trace of `Linear{4, 3}`, its weight and bias require a gradient, the weight has the
`Parameter` lifetime, and the input does not require a gradient.

### RolesAndFansAreRecordedForTheInitializer

The weight of `Linear{4, 3}` has the role `Weight`, a fan-in of 4 and a fan-out of 3, and the
bias has the role `Bias`.

### LinearRejectsAnInputWhoseInnerExtentIsWrong

`Linear{4, 3}` over a [2, 7] input throws `Error`.

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

### AWalkableLeafIsEnumeratedByItsOwnWalk

The walk of `WalkableStub` lists `weight` [2, 3] and `bias` [2], with a parameter count of 2
and 8 elements.

### AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

### TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`modules/layers/nn/include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
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

### A parameter across graphs

The cases from AParameterTracedIntoTwoGraphsHasALeafInEach to ARoleDeclareFillsByItsRole cover
`NN::Parameter` in `modules/layers/nn/include/SushiAI/nn/module.hpp`. A parameter holds a `Graph::StateID` and
no graph. `declare()` adds a leaf to the graph of the builder it is given and binds the leaf to
the state there, so one parameter has one leaf in each graph it was traced into. `id_in(graph)`
returns that graph's leaf.

The file-local `K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is
stored and read back unchanged.

### declare_weight

`declare_weight` declares a [2, 3] parameter of role `WEIGHT` named "weight" through a given
builder and returns its leaf.

### AParameterTracedIntoTwoGraphsHasALeafInEach

The first graph holds a padding input, so the parameter's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, `state_leaf` of the
parameter's state returns the same in each graph, and the second leaf has the lifetime
`PARAMETER`.

### AParameterHasNoLeafInAGraphItWasNotTracedInto

Before any `declare()`, `id_in` is `K_NO_VALUE` and `state()` is `K_NO_STATE`. After one
`declare()` the state is set, and `id_in` of a graph that was never traced into is still
`K_NO_VALUE`.

### ASecondDeclareInOneGraphReturnsTheFirstLeaf

A second `declare()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ADeclareWithAnotherShapeIsRefused

After a [2, 3] declaration, a `declare()` with [4, 3] throws with a message containing
"'weight'", "[2, 3]" and "[4, 3]", through the same builder and through a builder over a second
graph. The second graph holds no value and the parameter's shape stays [2, 3].

### ADeclareWithAnotherElementTypeIsRefused

After a FLOAT32 declaration, a `declare()` with FLOAT64 throws with a message containing
"'weight'" and both type names, in the same graph and in a second one. The second graph holds
no value and the parameter's type stays FLOAT32.

### ALaterDeclareKeepsTheFirstDescription

A parameter declared as a `SCALE` with a constant fill of 1.5 and fans of 4 is declared into a
second graph as a `WEIGHT` with a fan-scaled fill and fans of 9. Its fill, role and both fans
are still the first ones, and it has a leaf in the second graph.

### ACopiedLayerHasANewStateID

A copy of a traced `Linear` has a weight whose state is `K_NO_STATE` and whose `id_in(graph)`
is `K_NO_VALUE`, and `declared()` is true. Once the copy is traced, its weight has a leaf in the
graph and a state that differs from the original's, and its weight and bias leaves differ from
the original's.

### AMovedLayerKeepsItsStateID

A `Linear` move-constructed from a traced one has the same weight state and the same weight
leaf in the graph. Tracing it again adds two values, the two activations, and no parameter
leaf.

### ARoleDeclareRecordsTheFillOfItsRole

A `declare()` that states a role and no fill records a fan-scaled fill for a `WEIGHT` and a
constant fill of 0.0 for a `BIAS`.

### AnInitDeclareRecordsTheStatedFillAndTheStatedRole

A `declare()` that states both keeps both: a `BIAS` with a constant fill of 1.5 and a fan-in
of 4, and a `WEIGHT` with a fan-scaled fill. The bias's leaf requires a gradient.

### ANormalFillKeepsItsDeviation

`ParameterInit::normal(0.02)` has the kind `Normal`, the deviation 0.02 and the constant 0.
`fan_scaled()` has the kind `FanScaled` and the deviation 0. `constant_value(1.5)` has the
kind `Constant`, the constant 1.5 and the deviation 0.

### AnEmbeddingRoleWithoutAFillIsFanScaled

A parameter declared with the role `Embedding` and no fill reports that role and an init of
the kind `FanScaled`.

### ALinearDeclaresTheFillItStates

After a trace, the weight of a default `Linear{3, 4}` has an init of the kind `FanScaled`.
With `weight_init` set to `normal(0.02)` the weight's init is `Normal` with the deviation
0.02, its role is still `Weight`, and the bias's init is `Constant`.

### ARoleDeclareFillsByItsRole

The three roles give three fills: fan-scaled for `WEIGHT`, a constant 0.0 for `BIAS` and a
constant 1.0 for `SCALE`.

### The state walk

The cases from WalkStateYieldsTheThreeFieldKindsInDeclarationOrder on cover `NN::walk_state`
and the two walks derived from it. A `fields()` list holds three kinds of entry: `NN::field`
for a `Parameter`, `NN::buffer` for a `Buffer` and `NN::child` for a member that is itself
walked. `walk_state` hands each parameter and each defined buffer to an `NN::StateVisitor`.
`for_each_parameter` and `for_each_buffer` each yield one kind.

### StatePair

`StatePair` lists a parameter "scale" and a buffer "mean". `describe()` declares the parameter
as a [4] `SCALE` and defines the buffer as [4] zeros.

### MixedState

`MixedState` lists, in this order, a buffer "count", a `StatePair` child "pair", a parameter
"offset" and a buffer "unset". `describe()` defines "count" as a [1] INT64 buffer, describes
the pair and declares "offset" as a [4] `BIAS`. It leaves "unset" undefined.

### walk_all

`walk_all` collects what `walk_state` yields under a prefix as "parameter name:extents" and
"buffer name:extents" strings.

### buffer_names

`buffer_names` collects the names `NN::for_each_buffer` yields.

### WalkStateYieldsTheThreeFieldKindsInDeclarationOrder

The walk of a described `MixedState` yields "buffer count:[1]", "parameter pair.scale:[4]",
"buffer pair.mean:[4]" and "parameter offset:[4]". Parameters and buffers interleave in the
order `fields()` lists them.

### WalkStateJoinsItsPrefixToEveryName

Under the prefix "model.3" the same four entries are named "model.3.count",
"model.3.pair.scale", "model.3.pair.mean" and "model.3.offset".

### ForEachParameterNeverYieldsABuffer

`for_each_parameter` yields "pair.scale:[4]" and "offset:[4]". `parameter_count` is 2 and
`parameter_elements` is 8.

### ForEachBufferNeverYieldsAParameter

`for_each_buffer` yields "count" and "pair.mean", and `buffer_count` is 2.

### AnUndefinedBufferIsSkipped

Before `describe()` no buffer is yielded. After it two are. Once `define()` is called on
"unset", the walk yields "count", "pair.mean" and "unset".

### ADefinedBufferIsWalkedWithoutALeaf

The buffer "count" has no leaf in the graph, `id_in` is `K_NO_VALUE`, and it is one of the two
buffers counted.

### SelfWalkingStub

`SelfWalkingStub` has a member `walk_state(prefix, visitor)` and no `fields()`. It yields its
buffer "level" before its parameter "gain".

### StubPair

`StubPair` lists a `SelfWalkingStub` as the child "own" and a `WalkableStub` as the child
"legacy".

### ATypesOwnWalkStateIsTheWalk

`Detail::HasStateWalk` is true for `SelfWalkingStub` and false for `Linear`. The stub's walk
under "head" yields "buffer head.level:[3]" and then "parameter head.gain:[3]", the order its
member function states. `for_each_parameter` yields "gain:[3]" and `for_each_buffer` yields
"level".

### AChildFieldNamesEveryKindOfWalkableChild

The walk of a `StubPair` yields "buffer own.level:[3]", "parameter own.gain:[3]", "parameter
legacy.weight:[2, 3]" and "parameter legacy.bias:[2]". A `child` entry prefixes a type with its
own `walk_state` and a type with `walk_parameters` alike.

### AWalkParametersTypeYieldsNoBuffer

The walk of a `WalkableStub` yields its weight and its bias as parameters, and `buffer_count`
is 0.

### ALayerAdapterYieldsItsLayersParametersUnderThePrefix

A traced `LayerAdapter<Linear>` walked under the prefix "2" yields "parameter 2.weight:[3, 4]"
and "parameter 2.bias:[3]", and `buffer_count` is 0.

### TwoFieldsWithOneNameAreDetected

`Detail::has_unique_names` is checked in four `static_assert` lines. It is true for
`MixedState::fields()` and for an empty tuple. It is false for two `field` entries both named
"weight", and for a `buffer` and a `child` both named "pair". At run time it is true for
`Linear::fields()`. `walk_state` asserts the same predicate.

### ASequentialOfModulesKeepsItsBuilderForward

Three `static_assert` lines hold that `Sequential<Linear, ReLU>` is a `Module`, that it stays
one when nested in another `Sequential`, and that its `forward` over a `GraphBuilder` returns
a `Graph::ValueID`. A nested model of `Linear{4, 8}`, `ReLU` and `Linear{8, 3}` traces to
[2, 3] through the builder and walks as "0.0.weight:[8, 4]", "0.0.bias:[8]", "1.weight:[3, 8]"
and "1.bias:[3]".

## test_module_list.cpp

Suites: `Unit_ModuleList`, `Unit_ModuleListDeathTest`.

Tests ModuleList: its storage, its index names and its place in the state walk.

This file needs no device. `NN::ModuleList<M>` in `modules/layers/nn/include/SushiAI/nn/module_list.hpp` holds
children of one type whose count is known at run time.

Three `static_assert` lines at the top of the file hold that `ModuleList<Linear>` is a
`ModuleContainer` and that neither `Detail::HasStateWalk` nor `Detail::HasParameterWalk`
detects it, so the walk reaches it through `visit_children`.

### Stage

`Stage` lists a `ModuleList<Linear>` under the child name "blocks" and a `Linear{4, 2}` under
"head".

### Counter

`Counter` holds one `NN::Buffer` listed as "count", so a list of it carries state that is not
a parameter.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields as "name:extents" strings.

### read_past_the_end

`read_past_the_end` builds a list of two layers and reads `list[2]`. It is the statement of
the death test.

### ANewListIsEmpty

A new list has size 0, `visit_children` calls its visitor zero times, and `begin()` equals
`end()`.

### PushBackAppendsInOrder

After two `push_back` calls the size is 2, element 0 has 8 output features and element 1
has 3. Indexing a const reference to the list reads element 1's 8 input features.

### IterationFollowsIndexOrder

A range-for over three layers reads the output widths 8, 3 and 1.

### VisitChildrenPairsEachChildWithItsIndex

`visit_children` over two layers passes the indices 0 and 1 with the addresses of `list[0]` and
`list[1]`.

### TheWalkNamesChildrenByIndex

Two traced layers, the second without a bias, walk as "0.weight:[8, 4]", "0.bias:[8]" and
"1.weight:[3, 8]", and `parameter_count` is 3.

### AListUnderAChildFieldIsNamedByFieldThenIndex

A traced `Stage` with two blocks walks as "blocks.0.weight", "blocks.0.bias",
"blocks.1.weight", "blocks.1.bias", "head.weight" and "head.bias", each with its extents.

### TheWalkNamesBuffersByIndex

Of three `Counter` children, 0 and 2 have their buffer defined. `for_each_buffer` yields
"0.count" and "2.count": child 1 yields nothing and still keeps its index. `parameter_count`
is 0.

### GrowingTheListKeepsATracedChildsStateIdentity

The first child is traced, then 64 more layers are appended, which moves a one-element
vector's storage at least once. The first child's weight keeps its state and its leaf in the
graph, and its bias keeps its leaf in the graph.

### AnIndexPastTheEndIsFatal

The case is in the suite `Unit_ModuleListDeathTest`. `list[2]` on a list of two stops the
process with a report containing "index 2" and "holds 2 children".

## test_node.cpp

Suites: `Unit_Node`, `Unit_NodeDeathTest`.

Tests ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time.

This file needs no device. `NN::Node` in `modules/layers/nn/include/SushiAI/nn/node.hpp` is the seam a
`ModelGraph` holds its rows through: a node declares its ports, reads a span of values and
appends its results. `NN::ModuleNode<L>` in `modules/layers/nn/include/SushiAI/nn/module_node.hpp` wraps a
layer of any of the three forms, and `NN::LayerNode` in `modules/layers/nn/include/SushiAI/nn/layer_node.hpp`
wraps an `NN::Layer`. Both return a `Status` from `forward`, and an `Error` a layer throws
comes back as a failure.

### BufferedStub

`BufferedStub` lists one `Buffer` as "running" and defines it as [4] zeros in its builder
`forward`.

### BufferedParentStub

`BufferedParentStub` lists a `BufferedStub` as the child "inner".

### ModalStub

`ModalStub` has the modal pair, counts the calls of each form and returns its input.

### TraceStub

`TraceStub` has a trace `forward` that records the mode of each trace it is given.

### ShapelessStub

`ShapelessStub` has `fields()` and a builder `forward` and no `output_shape`.

### RefusingStub

`RefusingStub` records an `InvalidDescription` failure on the trace and returns its input.

### RefusingThenThrowingStub

`RefusingThenThrowingStub` records an `OutOfMemory` failure on the trace and then throws an
`Error` with a different message.

### node_of

`node_of` calls `ModuleNode<L>::create` over a layer and an input shape, expects success and
returns the node.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### state_names

`state_names` collects the name of every parameter and buffer a node's `walk_state` yields
under a prefix.

### make_test_conv2d

`make_test_conv2d` calls `NN::make_conv2d` with 5 filters and a 3x3 kernel, the layer a
`conv2d` row of a model file builds.

### The two predicates at compile time

`LayerAdapter` asserts that `Detail::HoldsBufferField` is false for its layer. The predicate is
true for `BufferedStub`, for `BufferedParentStub`, for `Sequential<Linear, BufferedStub>` and
for `ModuleList<BufferedStub>`, and false for `Linear` and `Sequential<Linear, ReLU>`.

`ModuleNode` asserts that `Detail::HasOutputShapeOfInput` is true for its layer. The predicate
is true for `Linear` and `TraceStub` and false for `ShapelessStub`.

### ModuleNodeDeclaresOnePortWithTheLayersOutputShape

The node of `Linear{6, 8}` over [6] has one port named "output" of shape [8].

### ALayerNodeOfEitherKindReadsOneValue

`input_count()` is 1 for a `ModuleNode<Linear>` and for a `LayerNode`.

### ModuleNodeCreateReturnsTheFailureOfAShapeTheLayerRefuses

`ModuleNode<Linear>::create(Linear{6, 8}, [5])` returns `InvalidDescription` with a message
containing "Linear(6, 8)", "[5]" and "[6]". `Linear::output_shape` throws for that shape, and
`create` returns the throw as a failure.

### ModuleNodeForwardReturnsTheFailureOfALayerThatThrows

`forward` over a [2, 5] value returns `InvalidDescription` with "Linear(6, 8)" and "[2, 5]",
and `outputs` stays empty.

### ModuleNodeForwardReturnsTheFailureALayerRecorded

The node of `RefusingStub` returns the failure the stub recorded on the trace, and `outputs`
stays empty although the stub returned an id.

### ModuleNodeForwardReturnsTheRecordedFailureBeforeALaterThrow

The node of `RefusingThenThrowingStub` returns the `OutOfMemory` failure with "the stub's
recorded cause". The `Error` thrown after it is not what the caller sees.

### ModuleNodeForwardAppendsTheLayersResult

`outputs` holds one id before the call and two after it. The first is untouched and the second
is a value of shape [2, 8].

### ModuleNodeTracesAModalLayerInTheModeOfTheTrace

Under a training trace the stub counts one training call and no evaluation call. After an
evaluation trace it counts one of each.

### ModuleNodeHandsTheTraceToATraceModule

A training trace and then an evaluation trace leave `TraceStub::modes` as Training,
Evaluation, and `outputs` as the input id twice.

### ModuleNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("stem", ...)` of the node of `Linear{6, 8}` yields "stem.weight" and
"stem.bias".

### ModuleNodeWalksABufferOfTheLayer

After a trace, `walk_state("bn", ...)` of the node of `BufferedStub` yields "bn.running".

### ModuleNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeReportsTheShapeOfTheLayerItWraps

A `LayerNode` over the conv2d layer built for [8, 8, 2] has one port named "output" whose
shape is the layer's `output_shape()`, [6, 6, 5].

### LayerNodeForwardAppendsAValueShapedLikeItsPort

`forward` over a [2, 8, 8, 2] value appends one id of shape [2, 6, 6, 5].

### LayerNodeForwardReturnsTheFailureOfALayerThatThrows

A `LayerNode` over `LayerAdapter<Linear>` given a [2, 5] value returns `InvalidDescription`
with "Linear(6, 8)" and "[2, 5]", and `outputs` stays empty.

### LayerNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("0", ...)` yields "0.weight" and "0.bias".

### LayerNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeRejectsNoInput

`forward` given an empty span returns `InvalidDescription` with "input count is 1" and "given
is 0".

### LayerNodeOverANullLayerIsFatal

The case is in the suite `Unit_NodeDeathTest`. Constructing a `LayerNode` from a null pointer
stops the process with a report containing "LayerNode" and "null layer".

## test_node_registry.cpp

Suites: `Unit_NodeRegistry`.

Tests `NN::NodeRegistry`, `NN::register_layers`, `NN::make_node` and `NN::build_model` over
the rows `register_layers` adds. The file needs no device. `make_node` looks a type up among
the registry's node rows and then among its layer rows, and wraps a layer in an
`NN::LayerNode`. `build_model` turns a `Config::ModelSpec` into an `NN::ModelGraph`. Each
returns a `Result`. `MinimalCase`, `TraceSummary`, `span_of`, `make_minimal` and
`trace_summary` come from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`.

### K_NORM

`K_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and the momentum the defaults of
`K_BRANCHING_MODEL` state.

### MinimalCase

`MinimalCase` holds a node registry key, a row's attributes as JSON, the per-sample shapes the
row reads and the per-sample shape of the node's one port.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` has one `MinimalCase` per node row of `registered()`, in row order:
batchnorm2d, concat, add, embedding, layernorm and dropout.

### LayerCase

`LayerCase` holds a layer registry key, a row's attributes as JSON and the per-sample shape
the row reads.

### K_LAYER_CASES

`K_LAYER_CASES` has one `LayerCase` per layer row of `registered()`, in row order.

### K_INPUT_JOINING_MODEL

`K_INPUT_JOINING_MODEL` is a version 2 model over [8, 8, 3] with three rows: a `conv2d` of 4
filters and kernel 1, a `silu`, and a `concat` whose `from` is `["input", -1]`.

### K_BRANCHING_MODEL

`K_BRANCHING_MODEL` is a version 2 model over [16, 16, 3] with a `defaults` block that gives
every `batchnorm2d` row `bn_epsilon` 0.001 and `bn_momentum` 0.03. Its seven rows are a
`conv2d` of 8 filters, kernel 3 and stride 2; a `conv2d` of 16 filters with the same window;
an `upsample` of scale 2; a `concat` whose `from` is `[-1, 0]`; a `conv2d` of 8 filters and
kernel 1; a `batchnorm2d`; and a `silu`.

### TraceSummary

`TraceSummary` holds the operation list of one traced graph, each published output as "name
[extents]" and each piece of state as "name [extents] parameter" or "name [extents] buffer".

### span_of

`span_of` views a vector of shapes as the `NN::ShapeSpan` of one row.

### make_minimal

`make_minimal` calls `NN::make_node` for a `MinimalCase` with given attributes and the context
"layer 0 '<type>'".

### trace_summary

`trace_summary` traces a model under a training trace over a batch of 2 samples and returns
its `TraceSummary`.

### registered

`registered` returns a registry that `register_layers` filled.

### build_branching_model_by_hand

`build_branching_model_by_hand` builds the graph `K_BRANCHING_MODEL` describes through
`ModelGraph::add`, with the nodes named "0" to "6", and publishes the last node through
`output_ports`.

### named_output

`named_output` returns an `OutputSpec` that publishes the port `NN::K_SINGLE_PORT_NAME` of a
row under a name.

### RegisterLayersAddsTheRowsInOrder

`register_layers` gives the layer rows linear, gelu, relu, tanh, sigmoid, leaky_relu,
unflatten, conv2d, maxpool2d, avgpool2d, global_avgpool, flatten, upsample, silu, mish and
slice_channels, and the node rows batchnorm2d, concat, add, embedding, layernorm and dropout,
each list in that order.

### ATypeAddedTwiceIsRefused

Adding a node row named "concat", and one named "relu", each fail with `InvalidDescription`,
"the type '<name>'" and "already". The node and layer row counts do not change.

### AnEmptyRegistryKnowsNoType

`make_node` of "silu" over an empty registry fails with `InvalidDescription`, "layer 0",
"the type 'silu'" and "does not have", and its message ends with
"the node types are: ; the layer types are: ".

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `registered()` has node rows, and entry `i` names
the type of row `i`.

### EachRowBuildsFromAMinimalObject

For every case, `make_node` returns a node with one port named `NN::K_SINGLE_PORT_NAME` whose
shape is the one the case lists.

### EachRowRejectsAnUnknownKey

For every case, the attributes `{"no_such_key": 1}` fail with `InvalidDescription` and a
message containing the row's context and "'no_such_key'".

### AConcatRowNamesItselfWhenItsInputsDisagree

A `concat` row over [4, 4, 2] and [8, 4, 2] fails with a message containing "layer 7
'concat'", "[8, 4, 2]" and "H and W".

### ANodeTypeBuildsItsOwnNode

`make_node("concat", ...)` over [4, 4, 2] and [4, 4, 3] returns an `NN::Concat` and not a
`LayerNode`.

### ALayerTypeIsWrappedInALayerNode

`make_node("upsample", ...)` with scale 2 over [4, 4, 2] returns a `LayerNode` with one port
named `K_SINGLE_PORT_NAME` of shape [8, 8, 2]. The case then walks `K_LAYER_CASES`, which has
one entry per layer row of the registry in order. Every type builds a `LayerNode`, and every
type given `{"no_such_key": 1}` fails with "'no_such_key'", which shows the layer's own
factory ran.

### ALayerTypeRefusesTwoInputs

A `relu` row over two shapes fails with a message containing "layer 3 'relu'", "reads 2
values" and "reads one".

### ALayerFactorysRefusalComesBackAsAStatus

A `linear` row with no attribute fails with a message containing "layer 4 'linear'" and
"'units'". A `linear` row over [8, 8, 2] fails with the context, "[8, 8, 2]" and "'flatten'".
The layer factories throw, and `make_node` returns each throw as an `InvalidDescription`
failure.

### AMaxPoolRowReadsItsTrailingPadding

Over [13, 13, 2] with `kernel` 2 and `stride` 1, the maxpool2d row gives a port of
[13, 13, 2] for `trailing_padding` 1, [13, 12, 2] for [1, 0] and [12, 12, 2] without the key.

### AnAvgPoolRowRefusesTrailingPadding

The avgpool2d row with `trailing_padding` fails with `InvalidDescription`,
"layer 0 'avgpool2d'" and "'trailing_padding'".

### AMaxPoolRowRefusesANegativeTrailingPadding

The maxpool2d row with `trailing_padding` -1 fails with `InvalidDescription`,
"layer 0 'maxpool2d'", "non-negative trailing padding" and "-1,-1".

### ASliceChannelsRowReadsItsOffsetAndCount

Over [4, 4, 64], the slice_channels row with `offset` 32 and `count` 32 gives a port of
[4, 4, 32]. With `offset` 48 and `count` 32 it fails with `InvalidDescription`,
"layer 0 'slice_channels'", "offset=48" and "count=32", and without `count` with "count".

### AnUnknownTypeListsBothTables

`make_node("conv3d", ...)` fails with a message containing "layer 5 'conv3d'", "the type
'conv3d'", "does not have", "node types" and "layer types". The message also contains every
type of the registry's node rows and of its layer rows in quotes.

### AFileAndAHandBuiltGraphAgree

`build_model` of `K_BRANCHING_MODEL` and `build_branching_model_by_hand` both hold seven
nodes. Their training traces have equal operation lists, equal outputs and equal state
records. The one output is "output [2, 8, 8, 8]". The state holds 7 records, from
"model.0.weight [8, 3, 3, 3] parameter" to "model.5.running_var [8] buffer".

### ARowReadsItsSourcesInFileOrder

In the model built from `K_BRANCHING_MODEL`, node 3 declares [8, 8, 24]: row 2's 16 channels
and then row 0's 8. Node 2 declares [8, 8, 16] and node 1 [4, 4, 16].

### OutputsArePublishedUnderTheirStatedNames

With the outputs set to row 4 as "mixed" and row 0 as "stem", the trace publishes "mixed
[2, 8, 8, 8]" and "stem [2, 8, 8, 8]" in that order.

### AnOutputWithoutANameKeepsThePortsName

An output of row 1 with an empty name is published as "output [2, 4, 4, 16]".

### ASourceMayNameAPort

A source that names the port `K_SINGLE_PORT_NAME` of the row above builds. A source that names
"no_such_port" fails with "layer 1 'conv2d'" and "'no_such_port'".

### ALaterRowMayReadTheModelInput

Row 2 of `K_INPUT_JOINING_MODEL` parses with `Config::K_MODEL_INPUT` as its first source. The
built node 2 declares [8, 8, 7]: the input's 3 channels and then row 1's 4.

### ASourceCannotNameAPortOfTheModelInput

A source that is the model input and names a port fails with a message containing "layer 2
'concat'", "the model input has no port" and "'output'".

### ARefusedRowNamesItsIndexAndType

Row 4 with `filters` 0 fails with a message containing "layer 4 'conv2d'",
"positive channel counts" and "out=0".

### ARowThatReadsARowNotAboveItIsRefused

The specs here are changed after parsing, so they hold what the parser would have refused. A
source of row 3 on row 1 fails with "layer 1 'conv2d'", "row 3" and "not above". A source of -2
on row 2 fails with "layer 2 'upsample'" and "row -2". A row with no source fails with "layer 2
'upsample'" and "no source".

### AnEmptyOutputListPublishesTheLastRowWhole

A spec whose `outputs` were cleared builds a model whose `output_shape` is [8, 8, 8], the same
as that of the model built from the parsed spec.

### AWrongOutputListIsRefused

A spec with no layer fails with "no layers". An output of row 7 fails with "output 0", "row 7"
and "rows 0 to 6". A named output with an empty port fails with "output 0", "'mixed'" and
"without a port". An output of the port "no_such_port" fails with "output 0" and
"'no_such_port'". Two outputs named "mixed" fail with "output 1" and "'mixed'".

## test_silu.cpp

Suites: `Unit_SiLU`.

Tests the SiLU layer's trace, shape inference and empty state.

This file needs no device. `NN::SiLU` in `modules/layers/nn/include/SushiAI/nn/silu.hpp` has no parameter and no
attribute.

Three `static_assert` lines at the top of the file hold that `SiLU` is a `Module`, is not a
`ModuleContainer`, and that `SiLU::fields()` is an empty tuple.

### ForwardEmitsOneSiLUNodeReadingTheInput

`forward` adds one node. Its operation is `Ops::K_SILU`, its one input is the value passed in
and its one output is the value returned.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

### ForwardKeepsTheShapeAndNHWCTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::NHWC`.

### ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## test_throwing_bridge.cpp

Suites: `Unit_ThrowingBridge`, `Unit_ThrowingBridgeDeathTest`.

Tests bridge_throwing: a value passes through and a thrown Error becomes a Status.

This file needs no device. `NN::Detail::bridge_throwing` in
`modules/layers/nn/include/SushiAI/nn/throwing_bridge.hpp` runs a callable that may throw and returns its value
as a `Result`.

### refuse_a_shape

`refuse_a_shape` throws an `Error` through `SA_THROW_IF` with the message "Linear(6, 8) was
given [5]".

### refuse_an_allocation

`refuse_an_allocation` throws `std::bad_alloc`.

### throw_a_logic_error

`throw_a_logic_error` throws `std::logic_error("an index was out of range")`, which is neither
an `Error` nor `std::bad_alloc`.

### bridge_a_logic_error

`bridge_a_logic_error` passes `throw_a_logic_error` to `bridge_throwing`, as the statement of
a death test.

### AValuePassesThrough

A callable that returns 7 gives an ok `Result<int>` holding 7.

### AMoveOnlyValuePassesThrough

A callable that returns `std::make_unique<int>(7)` gives an ok result whose pointer is not
null and owns 7.

### AnErrorBecomesInvalidDescriptionWithItsMessage

Bridging `refuse_a_shape` gives an `InvalidDescription` failure whose message contains
"Linear(6, 8)" and "[5]".

### ARefusedAllocationBecomesOutOfMemory

Bridging `refuse_an_allocation` gives an `OutOfMemory` failure whose message contains
"allocation was refused".

### AnotherExceptionIsFatal

The case is in the suite `Unit_ThrowingBridgeDeathTest`. `bridge_a_logic_error` stops the
process with a report containing "other than Error" and "an index was out of range".

## test_trace.cpp

Suites: `Unit_Trace`, `Unit_TraceDeathTest`.

Tests Trace, the buffer-update sinks, the three layer forms and trace_layer.

This file needs no device. `NN::Trace` in `modules/layers/nn/include/SushiAI/nn/trace.hpp` carries a
`Graph::GraphBuilder`, a `TraceMode` and the sink that receives buffer updates. A layer has one
of three forms: `Module`, whose `forward` takes a builder; `TraceModule`, whose `forward` takes
a `Trace`; and `ModalModule`, which states `trace_training` and `trace_evaluation` and no
`forward`. `NN::trace_layer` calls whichever form a layer has.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### ModalStub

`ModalStub` has the modal pair and an empty `fields()`. `trace_training` emits one ReLU and
`trace_evaluation` returns its input, and each counts its calls.

### CompositeStub

`CompositeStub` lists a `ModalStub` as the child "inner". Its `forward` takes a `Trace` and
passes it to the child through `NN::trace_layer`.

### ByValueTraceStub

`ByValueTraceStub` has a `forward` that takes its `Trace` by value.

### TrainingOnlyStub

`TrainingOnlyStub` states `trace_training` and no `trace_evaluation`.

### ModalWithForwardStub

`ModalWithForwardStub` states the modal pair and a builder `forward` together.

### BothForwardsStub

`BothForwardsStub` has a builder `forward` and a trace `forward` and counts the calls of each.

### FieldlessStub

`FieldlessStub` has the modal pair and no `fields()`.

### RefusingStub

`RefusingStub` has a trace `forward` that records an `InvalidDescription` failure on the trace
through `Trace::fail` and returns its input.

### failure_saying

`failure_saying` returns an `InvalidDescription` failure with a given message.

### defined_buffer

`defined_buffer` returns a `Buffer` defined as [3] with the starting value 0.0 that no graph
has read.

### update_undefined_buffer

`update_undefined_buffer` requests an update of a buffer on which `define()` was never called.
It is the statement of a death test.

### update_from

`update_from` requests an update whose source is a given id, in a graph that holds one value.
It is the statement of two death tests.

### parameter_names

`parameter_names` collects the names `NN::for_each_parameter` yields, in walk order.

### The layer forms at compile time

The `static_assert` lines after the helpers hold which form each type has. `Linear` is a
`Module` only, `ModalStub` a `ModalModule` only and `CompositeStub` a `TraceModule` only, and
all three are `Traceable`. A `Sequential` of `Module`s is both a `Module` and a `TraceModule`.
A `Sequential` that holds a modal layer or a trace-only composite, at any depth, is a
`TraceModule` and not a `Module`, and is still a `ModuleContainer`. `BothForwardsStub` is a
`Module` and a `TraceModule`.

`Detail::HasHalfModalPair` detects `TrainingOnlyStub` and `Detail::HasModalMemberAndForward`
detects `ModalWithForwardStub`; neither detects `ModalStub` or `Linear`. `TrainingOnlyStub`,
`FieldlessStub`, `ByValueTraceStub` and `int` are not `Traceable`.

`Trace` is final. It is not copy-constructible, move-constructible or default-constructible,
and a `GraphBuilder` does not convert to it. `NN::BufferUpdateSink` has a virtual destructor.

### AFactoryStatesItsModeAndKeepsItsBuilder

`Trace::training(builder, list)` reports `TraceMode::Training` and `Trace::evaluation(builder)`
reports `TraceMode::Evaluation`. `builder()` of each returns the address of the builder it was
made from.

### UpdateUnderAnEvaluationTraceFails

`update()` on an evaluation trace returns `InvalidDescription` with a message containing
"evaluation trace".

### ANewListIsEmpty

A new `BufferUpdateList` has no update and no source.

### EachUpdateAddsOneUpdateAndOneSource

Two `update()` calls on a training trace leave two entries in `updates()` and two in
`sources()`. Entry 0 holds the first buffer's state, the source x, the momentum 0.1 and the
source scale 1.0. Entry 1 holds the second buffer's state, the source y, 0.25 and 1.5.
`sources()` is x then y.

### AnUpdateAddsNothingToTheGraph

An `update()` leaves the graph's value count unchanged and the buffer without a leaf in it.

### ARecordedTargetSurvivesAMoveOfItsBuffer

After the buffer of a recorded update is move-constructed into another, the update's `target`
equals the moved buffer's state and is not `K_NO_STATE`.

### TheDiscardingSinkAcceptsAnUpdateAndIsOneObject

A trace made with `NN::discard_updates()` is in Training mode, its `update()` returns ok and
the graph gains no value. Two calls of `discard_updates()` return the same address.

### AnUpdateOfAnUndefinedBufferIsFatal

The case is in the suite `Unit_TraceDeathTest`. `update_undefined_buffer` stops the process
with "define() has not given a shape".

### AnUpdateOfABufferThatWasNeverReadIsRecorded

An update of a defined buffer that no graph has read is recorded with the buffer's state as
its target. The graph gains no value and the buffer has no leaf in it.

### ANewTraceHoldsNoFailure

A new trace's `status()` is ok and the graph has no refusal.

### FailKeepsTheFirstFailureAndMarksTheGraphRefused

After `fail` with "the first reason" and then "the second reason", `status()` carries the
first message and does not contain "second". `graph.has_refusal()` is true.

### ValueOrFailReturnsAHeldValue

`value_or_fail(Result<ValueID>(y), x)` returns y and leaves `status()` ok.

### ValueOrFailRecordsAFailureAndReturnsTheFallback

`value_or_fail` over a failed result returns the fallback x, and `status()` then carries that
failure.

### FailGivenASuccessIsFatal

The case is in the suite `Unit_TraceDeathTest`. `trace.fail(Status())` stops the process with
"Trace::fail was given a success".

### SequentialThrowsTheFailureAChildRecorded

`Sequential{RefusingStub, ModalStub}::forward` over a trace throws with a message containing
"RefusingStub refuses". The second child's `trace_evaluation` is never called.

### SequentialFoldsAnInputShapeThroughItsChildren

`output_shape([4])` of `Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` is [3].
`output_shape([5])` throws with a message containing "Linear(4, 8)" and "[5]".

### AnUpdateWithoutASourceIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is `K_NO_VALUE` stops the
process with "a buffer update reads value".

### AnUpdateFromAValueOutsideTheGraphIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is value 7, in a graph
of one value, stops the process with a report containing "reads value 7" and "holds 1 values".

### TraceLayerCallsTheTrainingFormInTrainingMode

Under a training trace, `trace_layer` calls `ModalStub::trace_training` once and
`trace_evaluation` never. The result differs from the input, because the training form emits a
ReLU.

### TraceLayerCallsTheEvaluationFormInEvaluationMode

Under an evaluation trace, `trace_layer` calls `trace_evaluation` once and `trace_training`
never, and returns its input.

### TraceLayerTracesAModuleThroughTheBuilder

`trace_layer` on a `Linear{4, 3}` and `Linear::forward` on a second one, each in its own graph,
return the same id and leave the same value count. The traced result has the shape [2, 3].

### TraceLayerPassesTheTraceToATraceModule

`CompositeStub` is traced once under an evaluation trace and twice under a training trace. Its
child counts one evaluation call and two training calls.

### TraceLayerPrefersTheTraceForwardOverTheBuilderForward

On `BothForwardsStub`, `trace_layer` calls the trace `forward` once and the builder `forward`
never.

### ANestedSequentialTracesItsModalChildInTheTracesMode

A `ModalStub` inside a `Sequential` inside a `Sequential` counts one evaluation call after an
evaluation trace of the outer container, and one training call after a training trace of it.

### SequentialTracesAModalChildInTheTracesMode

`Sequential{Linear{4, 3}, ModalStub}` traced under each mode calls the stub's matching form
once and yields a [2, 3] value both times. Its parameter walk yields "0.weight" and "0.bias".

### SequentialOfModulesTracesTheSameThroughBothForwards

`Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` traced through the trace `forward` in one graph
and through the builder `forward` in another returns the same id, leaves the same value count
and walks the same parameter names.

### AnEvaluationTraceRefusesAFedLeafAndACarriedValue

An evaluation trace is not carrying. `feed` fails with `InvalidDescription`, "'position'" and
"carrying trace"; `carried_leaf` with "'kept'" and "carrying trace"; `carry` with
"carrying trace". No node is added.

### ATrainingTraceRefusesAFedLeaf

A training trace is not carrying, and `feed` fails with `InvalidDescription` and
"'position'".

### ACarryingTraceIsAnEvaluationTraceThatRefusesABufferUpdate

A carrying trace reports `is_carrying` and the mode `TraceMode::Evaluation`. `update` fails
with `InvalidDescription` and "evaluation trace", and no carried value is recorded.

### AFedLeafIsDeclaredOncePerName

Two `feed` calls for "position" return the same value and add one node between them; `feed`
for "mask" returns another. The value is an `Int32` input leaf. The fed list holds "position"
then "mask", `find("mask")` returns the mask's value and `find("absent")` returns
`K_NO_VALUE`.

### AFedLeafDeclaredAgainAsAnotherLeafIsRefused

After "mask" is fed as [1, 4] `Float32`, feeding it as [1, 5] fails with
`InvalidDescription`, "'mask'", "[1, 4]" and "[1, 5]", and as `Float64` with "'mask'". The
fed list still holds one leaf.

### CarryRecordsTheLeafItsNextValueAndItsInitialValue

`carried_leaf` declares an input leaf that is not in the fed list. After
`carry(leaf, next, 0.5)` the carried list holds one entry with that leaf, that next value and
the initial value 0.5, and `sources` holds the next value.

### CarryRefusesANextValueOfAnotherShape

`carry` of a [4, 3] leaf with a [3, 4] next value fails with `InvalidDescription`, "[4, 3]"
and "[3, 4]", and records nothing.

### CarryingAValueThatIsNoCarriedLeafIsFatal

`carry` of a value that `carried_leaf` did not declare is fatal with "did not declare".

## test_unflatten.cpp

Suites: `Unit_Unflatten`.

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `modules/layers/nn/include/SushiAI/nn/unflatten.hpp` holds a
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

### TagsTheResultNHWC

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

### EmitsOneReshapeNode

The trace adds exactly one node. The producer of the result is a K_RESHAPE whose one operand is
the input.

### OutputShapeAgreesWithTheTraceForAnRGBImage

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

### ForwardRejectsAnNCHWInput

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

Suites: `Unit_Upsample2d`.

Tests the shape, layout and rejections of the Upsample2d layer.

This file needs no device. `NN::Upsample2d` in `modules/layers/nn/include/SushiAI/nn/upsample2d.hpp` holds one
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

### TheResultIsTaggedNHWC

The traced result is rank 4 and tagged NHWC.

### ForwardEmitsOneUpsampleNodeCarryingTheScale

The trace of `Upsample2d{3}` adds exactly one node. The producer of the result is an
K_UPSAMPLE2D whose one operand is the input, and attribute slots 0 and 1 both hold 3.

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

### ForwardRejectsAnNCHWInput

A [1, 3, 2, 2] input is converted to NCHW, and `forward()` throws with a message containing
"upsample2d" and "NHWC". The lower-case name is the builder's: `GraphBuilder::upsample2d`
raises this message.

### TheParameterWalkVisitsNothing

After a trace the walk visits nothing and `parameter_count` is 0.

## test_built_rows.cpp

Suites: `Unit_LinkedRows`, `Unit_LinkedRowsDeathTest`.

Tests `NN::Detail::read_row_above`, `NN::LinkedNodeEntry` rows of `NN::NodeRegistry` and
`NN::BuiltRows`. The file registers a linked type `follower` whose factory reads a row number
under the key `names`, fails unless that row holds an embedding and returns an `Identity`
node.

### ARowReferenceTakesThreeForms

With three rows above, `{"names": 0}` reads as row 0, `{"names": -1}` as row 2 and
`{"names": {"node": 1}}` as row 1.

### ARowReferenceThatDoesNotFitIsRefused

Seven references fail with `InvalidDescription`: row 3 of three ("layer 3", "'names'", "row 3",
"not above"), -2 ("row -2"), -1 with no row above ("0 rows are above"), a string ("row number",
"string"), an object with a `port` key ("'port'"), an empty object ("'node'") and an absent key
("'names'", "absent").

### ALinkedTypeHasOneRowAmongAllKinds

`find_linked("follower")` returns a row, `find_node("follower")` is null and `linked()` holds
that one row. Adding a linked row named `concat` or `relu`, or a `NodeEntry` named `follower`,
fails with `InvalidDescription`, the name and "already".

### BuildModelHandsALinkedRowTheRowsAbove

`NN::build_model` over an embedding row, a `layernorm` row and a `follower` row naming row 0
returns a model of three nodes. `NN::find_embedding` returns a layer for node 0 and null for
node 1. The same model with `"names": -1` fails with `InvalidDescription`, "layer 2
'follower'", "row 1" and "no embedding".

### ALinkedRowBuiltAloneSeesNoRow

`NN::make_node` for `follower` with `{"names": 0}` fails with `InvalidDescription`, "layer 0
'follower'" and "not above". For the unknown type `missing` it fails with a message containing
`'missing'` and `'follower'`.

### BuiltRowsCountsTheNodesOfItsGraph

A `BuiltRows` over a `ModelGraph` of one node has the size 1 and `node(0)` is the graph's node
0. A default `BuiltRows` has the size 0.

### ARowOutsideTheViewIsFatal

`node(0)` on a default `BuiltRows` stops the process with a report containing
"BuiltRows::node" and "row 0".

## test_conv2d.cpp

Suites: `Unit_Conv2d`.

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `modules/layers/nn/include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

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

The traced graph holds exactly one K_BIAS_ADD node.

### ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd

With `use_bias` false the walk yields the weight alone, the bias is not declared, the graph
holds no K_BIAS_ADD node, and the result shape is still [4, 28, 28, 8].

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
`GraphBuilder::conv2d` emits no K_IM2COL.

### SamePaddedPadsByHalfTheKernelAndDropsTheBias

`Conv2d::same_padded(3, 16, 5, 2)` has 3 input and 16 output channels, a 5x5 kernel, stride 2
on both axes, a pad of 2 on top and left, dilation 1 and `use_bias` false.

### SamePaddedAtStrideOneKeepsTheSpatialExtents

`same_padded(4, 4, 3, 1)` maps [7, 10, 4] to [7, 10, 4] in `output_shape()` and traces
[2, 7, 10, 4] over a batch of two.

### SamePaddedAtStrideOneKeepsTheExtentsForAPointwiseAndAFiveByFiveKernel

Over [9, 6, 2], `same_padded(2, 8, 1, 1)` and `same_padded(2, 8, 5, 1)` both report [9, 6, 8].

### SamePaddedAtStrideTwoHalvesAnEvenExtent

`same_padded(3, 16, 3, 2)` maps [640, 320, 3] to [320, 160, 16] in `output_shape()` and in the
trace.

### SamePaddedTracesNoBias

A traced `same_padded(2, 8, 3, 1)` layer walks as "weight:[8, 3, 3, 2]" alone. Its bias is not
declared and the graph holds no K_BIAS_ADD node.

### ANonZeroBiasInitReachesTheBiasParameter

A layer with `bias_init` set to -4.5 traces a bias whose `init()` is a constant of -4.5 and
whose role is `ParameterRole::Bias`. -4.5 is exact in binary.

### TheDefaultBiasInitIsTheConstantZero

A layer left at the default traces a bias whose `init()` is a constant of 0.0.

### BiasInitLeavesTheWeightFanScaled

With `bias_init` set to 1.25 the weight's `init()` is still `ParameterInit::Kind::FanScaled`
and its role is `WEIGHT`.

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

### ForwardRejectsAnNCHWInput

The input is [2, 3, 3, 3] converted to NCHW. Every extent is 3, so the permuted shape passes
every check but the one on the layout tag. The message contains "Conv2d" and "NHWC". The node
count and the value count are unchanged and neither parameter is declared, so the layer
refuses before it writes anything to the graph.

### DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients, neither is `K_NO_VALUE`, and each has the shape
of its parameter. The case checks that a gradient exists and has the right shape; it computes
no number.

## test_layer_registry.cpp

Suites: `Unit_LayerRegistry`.

Tests the layer registry: each row's factory, the keys it reads and the model it builds.

This file needs no device. `NN::register_layers` in `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp`
adds sixteen layer rows to an `NN::NodeRegistry`, and a factory turns a per-sample input shape and a parsed
JSON object into an `NN::Layer`. A case therefore needs a `Shape`, a string of JSON and, for
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
`applications/examples/mnist_cnn/model.json` are the same nine, with the optional keys spelled out.

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

`make_layer` looks a type up with `find_layer` in the registry `make_layer_registry` returns and calls the
row's factory with the context "layer 0 '<type>'". The cases that assert a message expect that
context in it.

### wrapped

`wrapped<L>` returns the layer of type `L` inside an `NN::LayerAdapter<L>`, or null when the
`Layer` wraps another type. The `dynamic_cast` is how a case tells which layer type a factory
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

`K_MINIMAL_CASES` has as many entries as the registry has layer rows, and entry `i` names the type of
row `i`. A row added to the registry without a case here fails this test, so the two tests
that loop over the cases cover every factory.

### TheTypeStringsAreTheOnesModelFilesSpell

The `type` fields of the layer rows are, in order, "linear", "gelu", "relu", "tanh",
"sigmoid", "leaky_relu", "unflatten", "conv2d", "maxpool2d", "avgpool2d", "global_avgpool",
"flatten", "upsample", "silu", "mish" and "slice_channels". A model file names a layer by one of these strings, so a
renamed row fails here.

### EachFactoryBuildsFromAMinimalObject

For every case the factory returns a layer whose `output_shape()` is the shape the case lists.

### EachKeyBuildsItsOwnLayerType

Each of the fourteen keys builds an adapter over the layer type it names: `wrapped<L>` is not
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

### LeakyReLURejectsASlopeOutsideItsRange

Slopes of 0, 1 and -0.1 are refused when the layer is built, with a message containing the
context and "slope".

### UpsampleRejectsAScaleBelowOne

A scale of 0 is refused when the layer is built, with a message containing the context,
"Upsample2d" and "scale".

### LeakyReLUAndUpsampleReadTheirOptionalKey

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
the six parameters the walk yields. Every gradient is a value other than `K_NO_VALUE` with the
shape of its parameter. The case checks that the graph differentiates; it computes no number.

### ASiLURowTracesOneSiLUNode

The layer of a `silu` row over [8, 8, 2], traced over [4, 8, 8, 2], adds one node. Its
operation is `Ops::K_SILU` and the result keeps the shape [4, 8, 8, 2].

### AVersionTwoChainStartsFromTheInputShape

A version 2 document over the input shape [8, 8, 2] holds a `conv2d` of 5 filters and kernel
3, a `silu` and a `flatten`. `NN::build_layer_stack` builds three layers. Layer 0 reports
[6, 6, 5], which requires the factory to have received the rank-3 shape and not the count 128,
and the stack reports [180].

### AChainRefusesARowThatReadsTwoRows

Row 2 with `"from": [-1, 0]` makes `build_layer_stack` throw with a message containing "layer
2 'relu'", "the row before it" and "build_model".

### AChainRefusesARowThatSkipsBack

Row 2 with `"from": 0` throws with "layer 2 'relu'" and "the row before it".

### AChainRefusesALaterRowThatReadsTheInput

Row 1 with `"from": "input"` throws with "layer 1 'tanh'" and "the row before it".

### AChainRefusesOutputsThatPublishAnEarlierRow

A two-row document whose `outputs` is `[{"node": 0}]` throws with a message containing
"'outputs'", "last row" and "build_model".

### AChainAcceptsOutputsThatPublishTheLastRow

The same document with `[{"node": 1}]` builds a stack of two layers.

### AChainRefusesARowThatReadsANamedPort

Row 1 with `"from": {"node": 0, "port": "output"}` throws with "layer 1 'tanh'" and "the row
before it". The row it names is the row above; the port is what the chain refuses.

## test_embedding.cpp

Suites: `Integration_GatherRows`.

The file runs GATHER_ROWS, SCATTER_ADD_ROWS and the `Embedding` layer on an engine. The tables hold
the dyadic values 0.25, 0.5, 0.75 and so on, whose copies and sums are exact, so results are
compared with a host loop at a tolerance of 0. Indices are written into INT32 leaves.

### GatherCopiesTheRowsItsIndexNames

A `[2, 3]` index over a `[5, 3]` table, with row 2 named three times, gives the rows a host loop
copies.

### AReplayReadsTheIndexOfItsOwnRun

One compiled step over a FLOAT64 `[6, 2]` table is run with two different indices of four. Each run
returns the rows of the index written before it.

### ScatterAddSumsTheRowsOfARepeatedIndex

Six FLOAT64 rows are scattered into five table rows with the index `{3, 1, 3, 3, 0, 1}`, and the
step is run twice. The result equals the host sums, and rows 2 and 4, which no index names, are
zero.

### TheTableGradientIsBitIdenticalAcrossReplays

A FLOAT32 `[13, 8]` table is gathered with 256 indices, so each row is named about twenty times, and
the loss is differentiated at the table. The bytes of the gradient are the same after each of four
runs.

### AnEmbeddingYieldsTheRowsOfItsTable

An `Embedding` of 7 rows and width 4 over a `[2, 3]` batch of ids has the output shape `[2, 3, 4]`,
and its output holds the table rows the ids name.

## test_init.cpp

Suites: `Integration_InitState`.

The cases check the fill that `NN::initialize()` writes over a `NN::ModelState`: the constants of a
batch norm, the zeroed final layer, and the determinism of the default seed.

### AScaleIsOneAShiftIsZeroAndABufferHoldsItsInitialValue

A training trace of a batch norm named `bn` is initialised. Afterwards `bn.weight` holds only ones,
`bn.bias` and `bn.running_mean` hold only zeros, and `bn.running_var` holds only ones.

### ZeroFinalLayerZeroesOnlyTheLastFanScaledEntry

A convolution named `conv`, followed by a batch norm named `bn`, is initialised with the default
policy. Every element of `conv.weight` is zero, and `bn.weight` is still one.

### WithoutZeroFinalLayerTheLastWeightIsNotZero

An MLP initialised with `zero_final_layer` set to false has at least one nonzero element in
`2.weight`.

### TheSameSeedGivesTheSameBits

Two MLPs of the same architecture are each created as a state and initialised with the default
policy. Every entry has the same bytes in both states.

### ANormalFillHasTheDeviationItStates

The table of an `Embedding` of 64 rows and width 32 is initialised with the default policy. Over its
2048 values the mean is within 0.0025 of zero and the deviation within 0.002 of
`NN::K_EMBEDDING_DEVIATION`.

### ANormalFillOfNoDeviationIsRefused

An `Embedding` whose weight asks for `ParameterInit::normal(0.0)` is refused by `NN::initialize` as
`InvalidDescription`, with "positive, finite deviation" in the message.

## test_layer_norm.cpp

Suites: `Integration_LayerNorm`.

The file runs LAYER_NORM and the `LayerNorm` layer on an engine with an epsilon of 1e-5 and compares
with a host loop in double: FLOAT32 results within 2e-5 and FLOAT64 results within 1e-12. The
gradient cases normalise a FLOAT64 `[2, 3, 4]` value, reshape it to `[6, 4]`, and compare the
analytic gradient of one leaf with central differences while the other leaves hold seeded noise.

### EachRowIsNormalisedByItsOwnStatistics

A FLOAT32 `[2, 3, 5]` value with a gain and a bias of five elements is normalised and the step is
run twice. The output equals the host's rows.

### TheSavedStatisticsAreTheRowMeanAndInverseDeviation

For the FLOAT64 rows `{1, 2, 3, 4}` and `{-4, -2, 0, 2}` the saved means are 2.5 and -1, and the
saved inverse deviations are `1 / sqrt(1.25 + eps)` and `1 / sqrt(5 + eps)`.

### TheInputGradientMatchesCentralDifferences

The gradient at the `[2, 3, 4]` input.

### TheGainGradientMatchesCentralDifferences

The gradient at the gain of four elements.

### TheBiasGradientMatchesCentralDifferences

The gradient at the bias of four elements.

### ALayerNormalisesWithItsGainAndItsBias

A `LayerNorm` of width 4 over a FLOAT32 `[2, 3, 4]` value keeps the shape, and with its weight and
bias written from noise its output equals the host's rows.

Suites: `Integration_MaxPoolTrailing`.

A `MaxPool2d` with a 2x2 window, a stride of one and one trailing row and column of padding pools a
`[1, 3, 3, 1]` input into nine values. Each is compared with the maximum of its truncated window
within 1e-6.

### TrailingWindowsTakeTheMaximaOfTheirTruncatedTaps

The input `[[5, -1, 2], [-3, 8, 0], [4, -7, 6]]` pools to `[[8, 8, 2], [8, 8, 6], [4, 6, 6]]`.

### AllNegativeTrailingWindowsKeepTheirTrueMaximum

The input runs from -18 to -10. The result is `[[-14, -13, -13], [-11, -10, -10], [-11, -10, -10]]`;
a pad that took part in the maximum as zero would show as 0.

Suites: `Integration_MishValues`.

The `Mish` layer runs over one FLOAT32 row on the engine. Each result is compared with `x *
tanh(log1p(exp(x)))`, computed in double, within 1e-6 of the exact value's magnitude. A zero must be
matched exactly.

### MatchesTheExactFormAtTheAcceptancePoints

The inputs are -20, -1, 0, 1 and 20.

### SweepMatchesTheExactFormAndStatesTheDarknetDifference

The inputs run from -30 to 30 in steps of 0.5, 121 values. The engine's results match the exact
form. The test also evaluates, on the host in double, the form with a softplus cut-off at a
magnitude of 20, records its largest difference from the exact form as the property
`darknet_cut_off_largest_difference`, and expects that difference to be at most 1e-12.

## test_model_state.cpp

Suites: `Integration_ModelState`, `Integration_ModelStateDeathTest`.

`NN::ModelState` lists a model's parameters and buffers in walk order, finds them by name or by
identity, refuses two entries with one name, stops on an entry with no identity, and attaches its
storage to a graph and a pool. The cases pin each of those.

### CreateListsParametersInWalkOrder

A traced `Sequential` of Linear, GELU and Linear lists four parameters in order: `0.weight` [8, 6],
`0.bias` [8], `2.weight` [4, 8] and `2.bias` [4].

### CreateListsBuffersAfterATrainingTrace

A `ModelGraph` holding a batch norm named `bn`, traced in training mode, lists `bn.weight` and
`bn.bias` as parameters. It then lists `bn.running_mean`, with constant initial value 0, and
`bn.running_var`, with constant initial value 1, as buffers.

### AnUntracedModelGivesAnEmptyState

A model that was never traced gives a valid state with no entries.

### APrefixIsJoinedToEveryName

A prefix of "model" passed to `create()` makes the first entry `model.0.weight`.

### TwoEntriesWithOneNameAreRefused

`from_entries` with two entries named "shared" is refused as `InvalidDescription`, and the message
names "shared".

### AnEntryWithNoIdentityIsFatal

`from_entries` with an entry whose identity is 0 stops the process, as the death test expects. The
message contains "'orphan'", the entry's name, and "no identity".

### TensorByNameAndByStateAreTheSameStorage

`tensor()` called with the name and with the identity of one entry returns the same tensor object.

### AnUnknownNameIsRefusedAndNamed

`tensor("missing.weight")` is refused as `InvalidDescription`. The message contains the name and
"4 entries".

### AttachBindsEveryStateLeafToTheStatesStorage

`attach()` records four state leaves on the graph. For each leaf, the pool tensor's data pointer is
the same as the state tensor's.

### AttachRefusesAGraphOfAnotherModel

A graph traced from a second instance of the same architecture is refused by `attach()` as
`Incompatible`.

### TwoPoolsAttachedToOneStateShareStorage

Two pools attached to one state read the same numbers. A value written through a leaf of the first
pool is read back through the second.

## test_add.cpp

That the Add node takes exactly two rank-3 inputs of one shape, emits one ADD reading them in order, returns a failure for values that do not add, and what its factory refuses.

Suites: `Unit_Add`.

Tests `NN::Add` in `modules/layers/nn/include/SushiAI/nn/add.hpp` and its factory
`NN::make_add_node`. Unless a case says otherwise, the node is built over two [8, 6, 3] shapes
and traced over two [2, 8, 6, 3] inputs.

### CreateKeepsTheSharedShapeAsThePort

The node has one port, named "output", of the shape [8, 6, 3].

### CreateRejectsOneInput

`create` over one shape fails with `InvalidDescription`, and the message holds "takes 2 inputs"
and "was given 1".

### CreateRejectsThreeInputs

`create` over three shapes fails with "takes 2 inputs" and "was given 3".

### CreateRejectsNoInput

`create` over an empty list fails with "takes 2 inputs" and "was given 0".

### CreateRejectsUnequalChannels

[8, 6, 3] with [8, 6, 5] fails with `InvalidDescription` and "share one shape".

### CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with "share one shape".

### CreateRejectsUnequalWidth

[8, 6, 3] with [8, 12, 3] fails with "share one shape".

### CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message holding "add input 1" and "rank-3".

### CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [144] fails with a message holding "add input 0" and "rank-3".

### TheInputCountIsTwo

`input_count()` is 2.

### ForwardEmitsOneAddReadingBothInputsInOrder

`forward` over the inputs b and a adds one node. Its operation is `Ops::K_ADD`, its inputs are
b then a, and its one output is the one value `forward` returned.

### ForwardResultIsThePortShapeWithABatchAxis

The traced value has the shape [2, 8, 6, 3], and the port's shape stays [8, 6, 3].

### ForwardRejectsAnInputCountItWasNotBuiltOver

`forward` over one input fails with `InvalidDescription`, "input count is 2" and "given is 1".
It returns no output and adds no node.

### ForwardReturnsTheFailureOfValuesThatDoNotAdd

`forward` over a [2, 8, 6, 3] value and a [2, 8, 6, 5] value returns an `InvalidDescription`
status, no output, and adds no node.

### TheFactoryBuildsANodeFromARowWithNoKey

`make_add_node` over two [8, 6, 3] shapes and an empty row returns an `Add` with two inputs and
one port of the shape [8, 6, 3].

### TheFactoryRefusesAKeyAndShapesThatDoNotMatch

A row with the key "alpha" fails with a message holding "layer 3 'add'", "'alpha'" and
"no key". An empty row over [8, 6, 3] and [8, 6, 5] fails with "layer 3 'add'" and
"share one shape".

### TheStateWalkVisitsNothing

After a trace, `walk_state` calls neither the parameter visitor nor the buffer visitor.

## test_avg_pool2d.cpp

The shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

Suites: `Unit_AvgPool2d`.

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `modules/layers/nn/include/SushiAI/nn/avg_pool2d.hpp` holds a
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

### ForwardTracesThePooledShapeInNHWC

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

### ForwardRejectsAnNCHWInput

A [1, 6, 6, 3] input is converted with `layout_convert` to NCHW, and `forward()` throws on the
converted value.

### ForwardNamesTheLayoutOfAnNCHWInputNarrowerThanTheWindow

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

## test_batch_norm2d.cpp

BatchNorm2d's two trace forms and the running-statistic update.

Suites: `Unit_BatchNorm2d`, `Unit_BatchNorm2dDeathTest`.

Tests BatchNorm2d's two trace forms and the running-statistic update.

This file needs no device. `NN::BatchNorm2d` in `modules/layers/nn/include/SushiAI/nn/batch_norm2d.hpp` is a
modal layer. `trace_training` emits one K_BATCHNORM2D node and requests two buffer updates
through the trace. `trace_evaluation` reads the two running statistics as buffer leaves and
emits a K_BATCHNORM_FOLD and a K_CHANNEL_AFFINE. Both forms record a refusal on the trace and
return their input.

The last two cases read `tests/layers/nn/fixtures/batch_norm2d/batchnorm.sacp` through
`modules/layers/reference_files/include/SushiAI/reference_files/reference_file.hpp` and report themselves skipped when the file is absent.
They compute the updated statistics on the host from the momentum and the source scale the
trace requested. No device runs the update.

Two `static_assert` lines after the helpers hold that `BatchNorm2d` is a `ModalModule` and
neither a `Module` nor a `TraceModule`, and that `Sequential<Conv2d, BatchNorm2d>` is a
`TraceModule` and not a `Module`.

### Constants

`K_REFERENCE_EPSILON` is 1e-3 and `K_REFERENCE_MOMENTUM` is 0.03, the two values the reference
layers ran with. `K_UNIT_CASE_TOLERANCE` is 5.3e-7 and `K_MODEL_CASE_TOLERANCE` is 7.0e-7. Each
is four float32 epsilons of the case's largest running variance, 1.1086 and 1.4578.
`K_EXACT_TOLERANCE` is 1e-15, for a double computed the same way on both sides.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### only_node

`only_node` returns the first node of a graph that carries one operation. Each caller has
asserted a count of one before it.

### count_buffer_leaves

`count_buffer_leaves` counts the values of a graph whose lifetime is `BUFFER`.

### ChannelStatistics

`ChannelStatistics` holds the per-channel mean and biased variance of one NHWC activation.

### batch_statistics

`batch_statistics` computes a `ChannelStatistics` over batch, height and width from a flat
NHWC array, in double.

### updated_vector

`updated_vector` applies `NN::updated_statistic` per channel and rounds each result to
float32.

### view

`view` returns a `SushiRuntime::span<const float>` over a vector, the argument
`Test::near_reference` takes.

### expect_reference_case

`expect_reference_case` reads one case of `batchnorm.sacp`: an input and the running mean and
variance before and after one torch training step. It traces a layer of the input's channel
count with the reference epsilon and momentum, and applies the two requested updates to the
"before" statistics on the host. The results must be within the tolerance of the "after"
statistics. The variance recomputed with a source scale of 1.0 must not be, so the fixture
tells the unbiased update from the biased one.

### refused_training

`refused_training` traces the training form, expects the input id back and returns the trace's
`status()`.

### refused_evaluation

`refused_evaluation` does the same for the evaluation form.

### train_under_an_evaluation_trace

`train_under_an_evaluation_trace` calls `trace_training` with an evaluation trace. It is the
statement of a death test.

### evaluate_over_a_value_the_graph_lacks

`evaluate_over_a_value_the_graph_lacks` calls `trace_evaluation` over value 7 in a graph of
one value. It is the statement of a death test.

### ATrainingTraceEmitsOneBatchNorm

A training trace over [2, 2, 2, 3] emits one K_BATCHNORM2D node and no K_BATCHNORM_FOLD or
K_CHANNEL_AFFINE. The graph gains three nodes: the two parameter leaves and the K_BATCHNORM2D. The
node reads the input, the weight's leaf and the bias's leaf, has four outputs of which the
first is the returned value, and carries the epsilon 1e-3 in attribute 0.

### ATrainingTraceReadsNoBuffer

After a training trace the graph holds no value of lifetime `BUFFER`, and neither running
statistic has a leaf in it.

### ATrainingTraceRequestsTheTwoRunningUpdates

The list holds two updates. The first reads output 1 of the node and targets the running
mean's state, and the second reads output 3 and targets the running variance's state. The two
states are distinct and neither is `K_NO_STATE`. Both momenta are 0.03 and the mean's source
scale is 1.0.

### TheVarianceUpdateCarriesTheUnbiasedFactor

The variance update's source scale is n / (n - 1), where n is the number of values per channel.
For [2, 2, 2, 3] it is 8/7, and for [1, 20, 20, 128] it is 400/399.

### AnEvaluationTraceFoldsTheRunningStatistics

An evaluation trace over [2, 4, 4, 3] emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE and no
K_BATCHNORM2D, and the graph holds two buffer leaves. It gains six nodes: two parameter leaves,
two buffer leaves, the fold and the affine. The fold reads the weight, the bias, the running
mean and the running variance in that order and carries the epsilon 1e-3. The affine reads the
input and the fold's two outputs, and its output is the returned value of shape [2, 4, 4, 3].

### BothFormsNameTheSameState

One layer is traced in training form into one graph and in evaluation form into another. In
the evaluation graph, `state_leaf` of each update's target is the leaf of the matching running
statistic. The weight's and the bias's states are unchanged by the second trace, and each has
a leaf in the evaluation graph.

### TheWalkYieldsTwoParametersThenTwoBuffers

`NN::walk_state(layer, "bn", visitor)` over a traced five-channel layer yields "bn.weight",
"bn.bias", "bn.running_mean" and "bn.running_var" in that order, each of shape [5]. Both
parameters have a constant fill. The starting values are 1.0 for the weight, 0.0 for the bias,
0.0 for the running mean and 1.0 for the running variance.

### TrainingOverOneValuePerChannelFails

A training trace over [1, 1, 1, 3] records `InvalidDescription` with a message containing
"BatchNorm2d(3)", "more than one value per channel" and "got 1". The graph gains no value and
no node, the list holds no update, the weight is not declared, and the graph is marked refused.

### AnIntegerInputFails

Over an INT32 input, both forms record `InvalidDescription` with a message containing
"BatchNorm2d(3)", "half, float32 or float64 input" and "int32". Nothing is added to the graph
or the list, and neither the weight nor the running mean is declared.

### EvaluationOverOneValuePerChannelTraces

An evaluation trace over [1, 1, 1, 3] returns a value of that shape. The limit that
TrainingOverOneValuePerChannelFails holds belongs to the training form.

### AnInputThatDoesNotFitFails

Over a [2, 2, 2, 4] input both forms fail with a message containing "BatchNorm2d(3)" and
"[N, H, W, 3]". Over a [2, 3] input both fail with "rank-4". The graph gains no value and the
list no update.

### AnNCHWInputFails

Over a value converted to NCHW both forms fail with a message containing "BatchNorm2d(3)" and
"NHWC", and the graph gains no value.

### ALayerDescribedWrongFails

A default-constructed layer fails with "positive channel count" and "got 0". A layer with
epsilon 0.0 fails with "positive epsilon". A layer with momentum 1.5 fails with "momentum in
[0, 1]" and "1.5". The graph gains no value.

### ASecondTraceUnderAnotherElementTypeFails

A layer traced over a FLOAT32 input and then, in a second graph, over a FLOAT64 input fails
the second time with a message containing "'weight'" and both type names. The running mean's
type stays FLOAT32.

### TheScaleStartsAtOneAndTheShiftAtZero

After a trace the weight has the role `ParameterRole::Scale` and a constant fill of 1.0, and
the bias has the role `BIAS` and a constant fill of 0.0.

### TheFactoryReadsTheChannelsAndTheTwoConstants

`NN::make_batch_norm2d_node` over the shape [8, 8, 6] with `bn_epsilon` 0.001 and `bn_momentum`
0.03 returns a `ModuleNode<BatchNorm2d>` whose layer has 6 channels and those two options. The
node has one port of the input's shape and an input count of 1.

### TheFactoryRefusesARowThatDoesNotFit

The key `channels` fails with a message containing the context, "'channels'" and "bn_epsilon".
The input shape [48] fails with "[48]" and "[H, W, C]". A `bn_epsilon` of -1.0 fails with
"'bn_epsilon'" and "positive".

### TheTrainingFormUnderAnEvaluationTraceIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_training` given an evaluation
trace stops the process with a report containing "trace_training" and "was given an evaluation
trace".

### ATraceOverAValueTheGraphLacksIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_evaluation` over value 7 stops the
process with a report containing "BatchNorm2d(3)", "value 7" and "holds 1 values".

### TraceLayerPicksTheFormOfTheMode

`NN::trace_layer` under a training trace leaves one K_BATCHNORM2D and no K_BATCHNORM_FOLD in its
graph. Under an evaluation trace it leaves one K_BATCHNORM_FOLD and no K_BATCHNORM2D.

### OutputShapeIsTheInputShape

`output_shape([7, 5, 3])` is [7, 5, 3]. A three-channel layer also returns [7, 5, 4] for
[7, 5, 4]: `output_shape` does not check the channel count.

### UpdatedStatisticBlendsTheScaledBatchValue

`NN::updated_statistic(running, batch, momentum, scale)` is
(1 - momentum) * running + momentum * scale * batch. (1.0, 3.0, 0.25, 2.0) gives 2.25 and
(4.0, 8.0, 0.5, 1.0) gives 6.0. A momentum of 0 returns the running value 4.0 and a momentum
of 1 returns the batch value 8.0. Every operand and result is exact in a double.

### TheUpdatesReproduceTheUnitReferenceCase

`expect_reference_case("unit", K_UNIT_CASE_TOLERANCE)` checks the [2, 2, 2, 3] case. The
test's own comment gives the miss of the unscaled variance on this case as 8.3e-3.

### TheUpdatesReproduceTheModelReferenceCase

`expect_reference_case("model.9.cv1.bn", K_MODEL_CASE_TOLERANCE)` checks the case recorded at
that layer of the reference model. The test's own comment gives the miss of the unscaled
variance on this case as 1.4e-5.

## test_buffer.cpp

Buffer: its description, its one leaf per graph and its identity.

Suites: `Unit_Buffer`, `Unit_BufferDeathTest`.

Tests Buffer: its description, its one leaf per graph and its identity.

This file needs no device. `NN::Buffer` in `modules/layers/nn/include/SushiAI/nn/buffer.hpp` is state a model
owns that takes no gradient. `define()` records its shape, starting value and element type and
returns a `Status`. `read()` declares a leaf of lifetime `Graph::Lifetime::Buffer` in the graph
of the builder it is given.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### read_undefined

`read_undefined` calls `read()` on a buffer under the name "running_mean" with a builder over
a fresh graph. It is the statement of the death test.

### ANewBufferIsUndefined

A default-constructed buffer reports `declared()` false and the state `Graph::K_NO_STATE`.

### DefineRecordsTheDescription

After `define([8], 1.0, FLOAT64)` the buffer is declared and returns that shape, that element
type and the starting value 1.0.

### DefineDefaultsToFloat32

`define([8], 0.0)` with no element type records `DType::Float32`.

### ADefinedBufferAddsNoLeafUntilItIsRead

After `define()` alone a graph holds no value and `id_in(graph)` is `Graph::K_NO_VALUE`.

### DefineDrawsTheIdentity

Two buffers that were only defined have states other than `K_NO_STATE`, and the two states
differ. A second identical `define()` and a `read()` leave the first buffer's state unchanged.

### ReadDeclaresABufferLeaf

`read()` adds one value to the graph. It has the lifetime `BUFFER`, `requires_grad` false and
the shape [8] and type FLOAT64 that were defined. `id_in(graph)` and
`graph.state_leaf(buffer.state())` both return it.

### ASecondReadInOneGraphReturnsTheFirstLeaf

A second `read()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ABufferReadInTwoGraphsHasALeafInEach

The first graph holds a padding input, so the buffer's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, and the buffer's state is
the same after both reads.

### ACopiedBufferHasANewStateID

A copy of a defined and read buffer has the original's shape and starting value and a state
that is neither `K_NO_STATE` nor the original's. `copy.id_in(graph)` is `K_NO_VALUE` until the
copy is read, and its leaf then differs from the original's, which keeps its own.

### AMovedBufferKeepsItsStateID

A buffer move-constructed from a read one has the same state and the same leaf in the graph.
Reading it again returns that leaf and adds no value.

### AMovedFromBufferIsUndefined

After a move construction and after a move assignment the source reports `declared()` false
and the state `K_NO_STATE`. The assigned target is declared and has a state.

### ACopyAssignedBufferHasANewStateID

A defined [2] buffer copy-assigned from a [8] FLOAT64 one takes that shape and type. Its state
differs from the state it had before and from the source's.

### ACopyOfAnUndefinedBufferHasNoIdentity

A copy of a buffer that was never defined is not declared and has the state `K_NO_STATE`.

### DefiningTwiceWithOneDescriptionIsAccepted

A second `define()` with the same shape, value and type returns ok and leaves the description
as it was.

### DefiningWithAnotherShapeFails

`define([4], 0.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as", "[8]" and "[4]". The shape stays [8].

### DefiningWithAnotherElementTypeFails

`define([8], 0.0, FLOAT64)` after a FLOAT32 definition returns `InvalidDescription` with a
message naming both types. The type stays FLOAT32.

### DefiningWithAnotherStartingValueFails

`define([8], 1.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as" and "is now defined as". The starting value stays 0.0.

### ReadingBeforeDefineIsFatal

The case is in the suite `Unit_BufferDeathTest`. `read()` on an undefined buffer stops the
process with a report containing "'running_mean'" and "before define()".

## test_channel_slice.cpp

The ChannelSlice layer's shape inference, the one SLICE_CHANNELS node it emits with its offset, the offsets, counts and ranks it rejects, and its empty state.

Suites: `Unit_ChannelSlice`.

Tests `NN::ChannelSlice` in `modules/layers/nn/include/SushiAI/nn/channel_slice.hpp`, which
keeps `count` channels of an image batch from `offset` on.

Two `static_assert` lines at the top of the file hold that `ChannelSlice` is a `Module` and is
not a `ModuleContainer`.

### TheOffsetAndCountFieldsAreSetInOrder

`ChannelSlice{32, 16}` has `offset` 32 and `count` 16.

### OutputShapeKeepsHeightAndWidthAndTakesTheRange

With offset 32 and count 32, `output_shape([5, 3, 64])` is [5, 3, 32].

### OutputShapeAcceptsTheFirstLastAndWholeRanges

Over [4, 2, 4], the ranges {0, 1} and {3, 1} give [4, 2, 1], {1, 3} gives [4, 2, 3] and {0, 4}
gives [4, 2, 4].

### TheResultIsTaggedNHWC

The traced result over a [2, 5, 3, 64] input has rank 4 and the layout `TensorLayout::NHWC`.

### ForwardEmitsOneSliceNodeCarryingTheOffset

`forward` with offset 32 and count 32 adds one node. Its operation is `Ops::K_SLICE_CHANNELS`,
its one input is the value passed in, its attribute 0 is 32, and the result has the shape
[2, 5, 3, 32].

### TheWholeChannelRangeTracesToTheInputShape

`ChannelSlice{0, 64}` over a [2, 5, 3, 64] input traces to [2, 5, 3, 64].

### OutputShapeAgreesWithTheTracedShape

For {32, 32} over 64 channels and for {1, 2} over 6 channels, `output_shape` equals the traced
shape with its batch axis dropped.

### OutputShapeRejectsANegativeOffset

Offset -1 with count 4 throws an error naming "ChannelSlice", "offset=-1, count=4" and
"64 channels".

### OutputShapeRejectsAZeroCount

Offset 0 with count 0 throws an error naming "ChannelSlice", "offset=0, count=0" and
"64 channels".

### OutputShapeRejectsANegativeCount

Offset 8 with count -2 throws an error naming "ChannelSlice", "offset=8, count=-2" and
"64 channels".

### OutputShapeRejectsARangePastTheChannels

Offset 32 with count 40 over 64 channels throws an error naming "offset=32, count=40" and
"64 channels".

### OutputShapeRejectsAnOffsetAtTheChannelCount

Offset 64 with count 1 over 64 channels throws an error naming "offset=64, count=1".

### OutputShapeRejectsExtremeValuesWithoutOverflow

An offset of the largest `int64_t` with count 1, and offset 1 with a count of the largest
`int64_t`, each throw an error naming "ChannelSlice" and "64 channels".

### OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws an error naming "ChannelSlice", "[H, W, C]" and "[16]".

### OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws `Error`.

### ForwardRejectsARankTwoInput

`forward` over a [2, 12] input throws an error naming "ChannelSlice", "rank-4" and "[2, 12]".

### ForwardRejectsARangePastTheChannels

`forward` with offset 2 and count 2 over a three-channel input throws an error naming
"ChannelSlice", "offset=2, count=2" and "3 channels".

### ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "slice_channels" and "NHWC".

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times and
`NN::parameter_count` is 0.

## test_concat.cpp

The Concat node's shape check, its port and the node it traces.

Suites: `Unit_Concat`.

Tests the Concat node's shape check, its port and the node it traces.

This file needs no device. `NN::Concat` in `modules/layers/nn/include/SushiAI/nn/concat.hpp` is an `NN::Node`
that joins several feature maps along the channel axis. `Concat::create` takes one per-sample
shape per input and returns a `Result`, and `forward` returns a `Status`.

### create_over

`create_over` calls `Concat::create` over a vector of shapes viewed as an `NN::ShapeSpan`.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### CreateSumsTheChannelsOfItsInputs

Over [8, 6, 3], [8, 6, 5] and [8, 6, 1] the node has one port named "output" of shape
[8, 6, 9].

### CreateAcceptsOneInput

Over the one shape [8, 6, 3] the port's shape is [8, 6, 3].

### CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with `InvalidDescription` and a message containing "concat
input 1" and "H and W".

### CreateRejectsUnequalWidth

A third input of [8, 12, 3] fails with a message containing "concat input 2" and "H and W".

### CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message containing "concat input 1" and "rank-3".

### CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [8, 6, 3] fails with a message containing "concat input 0" and "rank-3".

### CreateRejectsNoInput

No shape at all fails with a message containing "at least one input".

### ForwardEmitsOneConcatChannelsReadingEveryInputInOrder

`forward` over the values b and a adds one `Ops::K_CONCAT_CHANNELS` node whose inputs are b then
a. It appends one id to `outputs`, the node's one output.

### ForwardResultIsThePortShapeWithABatchAxis

Over [2, 8, 6, 3] and [2, 8, 6, 5] the traced value has the shape [2, 8, 6, 8], and the port
reports [8, 6, 8].

### ForwardRejectsAnInputCountItWasNotBuiltOver

A node built over two shapes and given one value fails with a message containing "input count
is 2" and "given is 1". `outputs` stays empty and the graph gains no node.

### TheInputCountIsTheNumberOfShapesItWasBuiltOver

`input_count()` is 1 for a node built over one shape and 3 for a node built over three.

### ForwardReturnsTheFailureOfValuesThatDoNotJoin

A node built over two [8, 6, 3] shapes is given a [2, 8, 6, 3] value and a [2, 4, 6, 3] value.
`forward` returns an `InvalidDescription` failure, appends nothing to `outputs` and adds no
node.

### TheFactoryBuildsANodeFromARowWithNoKey

`NN::make_concat_node` over [8, 6, 3] and [8, 6, 5] with the attributes `{}` returns a
`Concat` with an input count of 2 and one port of shape [8, 6, 8].

### TheFactoryRefusesAKeyAndShapesThatDoNotJoin

The attributes `{"axis": 3}` fail with a message containing the context "layer 2 'concat'",
"'axis'" and "no key". The shapes [8, 6, 3] and [4, 6, 5] with no attribute fail with the
context and "H and W".

### TheStateWalkVisitsNothing

After a trace, `walk_state("merge", visitor)` calls neither the parameter callback nor the
buffer callback.

## test_layer_stack.cpp, tests/core/config/test_value.cpp, tests/core/config/test_train_config.cpp, tests/core/config/test_model_config.cpp

How the config layer parses its JSON documents and which ones it refuses.

Suites: `Unit_Config`.

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document in both of its versions, and the layer registry it selects through
- the train and eval documents

### ParsesEveryScalarKind

An object with a whole number, a real, a string, a boolean, a null and an array of two parses
to `Kind::Object`, and each member reads back through `as_int`, `as_real`, `as_string`,
`as_bool`, `is_null` and `as_array`.

### KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

### AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

### AWrongKindNamesBothTheContextAndTheKindFound

`as_int("layer 0 'units'")` on a string raises an error holding "layer 0 'units'" and
"a string".

### AMissingKeyNamesTheKeyAndListsWhatIsPresent

`at("gamma", "model.json")` on an object of `alpha` and `beta` raises an error holding
"model.json", "'gamma'", "'alpha'" and "'beta'".

### ReadingAKeyOffANonObjectSaysWhatItActuallyIs

`at("units", "the model")` on an array raises an error holding "the model" and "an array".

### AnUnknownKeyIsAnErrorThatListsWhatIsAllowed

`reject_unknown_keys` with the allowed keys `type` and `units` over an object holding `unitss`
raises an error holding "layer 0 'linear'", "'unitss'", "'units'" and "'type'".

### AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

### WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

### AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

### MalformedJSONNamesTheOriginAndTheComplaint

`parse_json("{ not json", "train.json")` raises an error holding "train.json" and
"not valid JSON".

### AnObjectBuiltFromMismatchedVectorsIsRefused

`Value::object` of two keys and one value raises an error holding "2 keys" and "1 value".

### ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

### ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

### AConfiguredStackTracesToTheDeclaredOutputWidth

The stack `build_layer_stack` makes from `K_GOOD_MODEL` traces a [4, 6] input to a [4, 4]
value.

### AnUnknownLayerTypeListsEveryTypeThisBuildHas

The model names a layer type "hologram", which no layer row carries. The
message contains "hologram", "layer type", "'linear'" and "'gelu'". The unknown type has to
be a name the registry does not hold, and "conv2d" is a layer row.

### AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

### The version 2 model document

The cases from AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite to
AMalformedOutputIsAnError cover the two versions of the model document that
`Config::parse_model` reads. `Config::K_MODEL_VERSION` is 2. A version 1 document carries the key
`sushiai_model` and an `input` of `features`. A version 2 document carries `format_version`,
an `input` of `shape`, an optional `from` on each row, and the optional top-level keys
`defaults` and `outputs`. Both parse to one `Config::ModelSpec`, in which every row holds its
sources as absolute row indices and the model input is `Config::K_MODEL_INPUT`, which is -1.

### K_FIRST_VERSION_MODEL

`K_FIRST_VERSION_MODEL` is a version 1 document of five rows over 784 features: `unflatten` to
28 by 28 by 1, `conv2d` of 8 filters and kernel 3 with a bias, `relu`, `flatten` and `linear`
of 10 units.

### K_REWRITTEN_MODEL

`K_REWRITTEN_MODEL` is the same model written as version 2, with the input shape [784] and
`"from": -1` on every row.

### scalar_text

`scalar_text` renders one scalar attribute as "int n", "bool true", "bool false" or "string
s", and any other kind as its kind name.

### spec_text

`spec_text` renders a `ModelSpec` as text: one line for the input shape and feature count, one
line per row with its type, its sources as `node:'port'` and its attributes, and one line per
output. Two specs compare equal as text when every one of those fields agrees.

### current_model_from

`current_model_from` wraps a `layers` array, and optionally more top-level keys, in a version
2 document whose input shape is [6].

### ALayerFactoryRefusesANonPositiveWidth

A `linear` row with `units` 0 makes `build_layer_stack` raise an error holding
"layer 0 'linear'" and "'units'".

### UseBiasDefaultsToTrueAndCanBeTurnedOff

After a trace, a `linear` row has two parameters, and one with `use_bias` false has one.

### AMissingModelVersionIsAnErrorThatSaysWhy

A model document without `sushiai_model` raises an error holding "sushiai_model" and
"misread".

### AFutureModelVersionIsAnError

`sushiai_model` 2 raises an error holding "version 2" and "version 1".

### AnUnknownTopLevelModelKeyIsAnError

A version 1 document with the key `optimizer` raises an error holding "'optimizer'" and
"'layers'".

### AModelWithNoLayersIsAnError

An empty `layers` array raises an error holding "no layers".

### ANonPositiveInputWidthIsAnError

`features` 0 raises an error holding "'features'" and "positive".

### ALayerThatIsNotAnObjectIsAnError

A layer given as the string "linear" raises an error holding "layer 0" and "a string".

### AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite

`spec_text` of the two documents is equal. The expected text is asserted as a literal too: the
input is [784] with 784 features, row 0 reads -1, each later row reads the row above it, and
the one output is row 4 with no port and no name.

### AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount

`K_GOOD_MODEL`, a version 1 document of 6 features, parses to `input_shape` [6],
whose `num_elements()` is 6. The case also asserts that `Config::K_MODEL_VERSION` is 2.

### AVersionTwoInputShapeKeepsItsRankAndCountsItsElements

The input shape [28, 28, 1] is kept at rank 3 and its `num_elements()` is 784.

### AnEmptyOrNonPositiveInputShapeIsAnError

An empty `shape` throws with a message containing "'input' 'shape'" and "empty". The shape
[28, 0, 1] throws with "'shape' entry 1" and "positive". A version 2 input that states
`features` throws with "'features'" and "'shape'".

### TheInputElementTypeIsFloat32UnlessTheFileStatesInt32

A version 2 document without `dtype`, and one with `dtype` "float32", both parse to an input
of `Float32`. One with "int32" parses to `Int32`.

### AnInputElementTypeOutsideTheListIsAnError

The `dtype` "int64" raises an error holding "'input' 'dtype'", "'int64'" and
"'float32', 'int32'". The `dtype` 32 raises one holding "'input' 'dtype'".

### AnInputShapeAboveTheRankOrTheElementLimitIsAnError

A shape of `K_MAX_RANK` + 1 extents throws with "'shape'" and "at most". The shape
[4294967296, 4294967296], whose product is 2^64, throws with "'shape'", "entry 1" and "more
elements".

### AModelRootThatIsNotAnObjectIsAnError

The document `[1]` throws with a message containing "model.json", "'format_version'" and "an
array".

### AVersionAboveTheNewestIsRefusedWithBothNumbers

`"format_version": 3` throws with a message containing "model.json", "version 3" and "version
2".

### FormatVersionOneIsRefusedAndNamesTheVersionOneKey

`"format_version": 1` throws with a message containing "'format_version'" and
"'sushiai_model'". Version 1 files carry the second key and never the first.

### AVersionOneDocumentRefusesTheVersionTwoKeys

A version 1 document with a `from` on row 1 throws with "layer 1", "'from'" and "version 1".
One with a top-level `defaults` throws with "'defaults'" and "'layers'".

### EachFromFormResolvesToAnAbsoluteRow

Six rows use every form of `from`. An absent `from` on row 0 and the string "input" resolve to
-1. The number -1 resolves to the row above. An array `[-1, 0]` on row 3 resolves to rows 2
and 0. An object `{"node": 3, "port": "boxes"}` resolves to row 3 with that port. An array may
mix objects, "input" and a relative object. The whole spec is compared as text. Row 0's source
is `K_MODEL_INPUT`, and "from" is not among row 3's attributes.

### ANegativeFromOtherThanMinusOneNamesTheRow

`"from": -2` on row 2 throws with a message containing "layer 2 of 'model.json'", "'from'" and
"row -2". `{"node": -3}` inside an array throws with the layer and "row -3".

### AForwardReferenceNamesTheRow

`"from": 2` on row 1 throws with "layer 1 of 'model.json'", "row 2" and "above row 1".
`"from": [0, 1]` on row 1, which names the row itself, throws with the layer and "row 1".

### AnUnknownKeyOrFormInAFromEntryNamesTheRow

Each of five `from` entries on row 1 throws with "layer 1 of 'model.json'" and one more
needle: the key `prot` with "'prot'" and "'port'"; the string "previous" with "'previous'" and
"'input'"; an empty array with "empty"; the number 0.5 with "it is a number"; and an empty
`port` string with "'port'" and "empty string".

### AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError

A top-level key `output` throws with a message containing "'output'", "'outputs'" and
"'defaults'".

### DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne

The `defaults` block gives `conv` rows `bn_momentum` 3 and `kernel` 5. Row 0 states only
`filters` and ends with the keys `filters`, `bn_momentum`, `kernel` and the values 3 and 5.
Row 1 states `bn_momentum` 7 and keeps it. Row 2 is a `relu` and receives no key.

### AMalformedDefaultsBlockIsAnError

A `defaults` that is an array throws with "'defaults'" and "an array". A type whose default is
the number 3 throws with "'defaults' for 'conv'" and "object". A default that sets `from`
throws with that context and "'from'", and one that sets `type` with "'type'".

### ADefaultsEntryForATypeNoLayerStatesIsAnError

A default for "covn" in a model of `conv` and `relu` rows throws with a message containing
"'defaults'", "'covn'" and "'conv', 'relu'".

### OutputsDefaultToTheLastRowAndKeepWhatTheFileStates

A document with no `outputs` gets one output: the last row, with an empty port and an empty
name. A document that states three outputs keeps each as written: row 0 alone, row 1 with the
port "boxes", and row 1 with the port "boxes" and the name "predictions".

### AMalformedOutputIsAnError

Seven `outputs` values each throw. Row 1 in a one-row model: "'outputs' entry 0", "row 1" and
"0 to 0". The key `nme`: "'nme'" and "'name'". A `name` without a `port`: "'name'" and
"'port'". An empty array: "'outputs'" and "empty". The number 3 as an entry: "a whole number".
An empty `port` string: "'outputs' entry 0 'port'" and "empty string". An empty `name` string:
"'outputs' entry 0 'name'" and "empty string".

### Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

### AModelWithoutATaskHoldsANullTask

A version 2 document without `task`, and `K_FIRST_VERSION_MODEL`, both parse to a null task.

### ATaskIsKeptWholeWithItsKeysInFileOrder

A task stating `classes`, `type` and `heads` in that order parses to an object with those
keys in that order, the type "anchor_detection", 80 classes and one head.

### AMalformedTaskIsAnError

The parse raises an error for a task given as an array ("'model.json's 'task'", "an array"),
as null ("'model.json's 'task'", "null"), without a type ("'model.json's 'task'",
"no 'type'", "'classes'") and with the type 3 ("'task' 'type'", "a whole number").

### AVersionOneDocumentRefusesATask

A version 1 document with a `task` raises an error holding "'task'" and "'layers'".

### ParsesATrainDocumentIntoItsFourTypedBlocks

`K_GOOD_TRAIN` parses to the objective type "cross_entropy", the data type "rings", the
optimizer type "adamw" and 3 epochs. The optimizer attributes hold no `type` and a
`learning_rate` of 0.003, and the data attributes hold a `batch_size` of 8.

### ResolvesRelativePathsAgainstTheConfigsOwnDirectory

`load_train` of a `train.json` written under the system's temporary directory gives a model
path of `model.json` and a checkpoint of `out/model.sushi`, both under that directory.

### LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

### EpochsDefaultsToOneAndMustBePositive

A train document without `epochs` parses to 1 epoch and an empty checkpoint path. `epochs` 0
raises an error holding "'epochs'" and "positive".

### AnUnknownTopLevelTrainKeyIsAnError

The key `epochz` raises an error holding "'epochz'" and "'epochs'".

### AMissingTrainBlockNamesIt

A train document without `data` raises an error holding "'data'".

### ATypedBlockThatIsNotAnObjectIsAnError

An `objective` given as a string raises an error holding "'objective'" and "a string".

### AFutureTrainVersionIsAnError

`sushiai_train` 2 raises an error holding "version 2" and "version 1".

### ParsesAnEvalDocument

`parse_data` of a `sushiai_data` 1 document gives the data type "rings", a `batch_size` of
512 and a `seed` of 3735928575.

### AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

### LoadingAnAbsentFileNamesThePath

`load_train("no/such/train.json")` raises an error holding that path.

## test_depthwise_conv2d.cpp

The DepthwiseConv2d layer's one DEPTHWISE_CONV2D node, its [kernel_h, kernel_w, channels] weight with role and fans, shape inference against the trace, `same_padded`, and the inputs it rejects.

Suites: `Unit_DepthwiseConv2d`.

Tests `NN::DepthwiseConv2d`. Most cases use four channels, a 3x3 window and one pixel of
padding.

Two `static_assert` lines at the top of the file hold that `DepthwiseConv2d` is a `Module` and
is not a `ModuleContainer`.

### ForwardEmitsOneDepthwiseNodeAndNoBiasAdd

After `forward` over a [2, 8, 8, 4] input the graph holds one `Ops::K_DEPTHWISE_CONV2D` node
and no `Ops::K_BIAS_ADD` node.

### TheEmittedNodeReadsTheInputAndTheWeight

The depthwise node has two inputs: the value passed in, and a value of the shape [3, 3, 4]. Its
one output is the value `forward` returned.

### ForwardKeepsTheSpatialExtentsAndTagsNHWC

A padded 3x3 window at stride 1 over [2, 8, 8, 4] yields [2, 8, 8, 4], tagged
`TensorLayout::NHWC`.

### StrideTwoHalvesAnEvenExtent

At stride 2 a [2, 8, 6, 4] input yields [2, 4, 3, 4].

### TheWeightOfAnAnisotropicWindowIsHeightWidthChannels

With a window 2 high and 4 wide over three channels, the weight's shape after `forward` is
[2, 4, 3].

### TheParameterWalkNamesOnlyTheWeight

The parameter walk is empty before `forward`. After it the walk yields "weight:[3, 3, 4]"
alone, and `NN::parameter_elements` is 36.

### TheWeightRecordsTheWeightRole

After `forward`, `weight.role()` is `ParameterRole::Weight`.

### TheWeightRecordsItsWindowAsBothFans

After `forward` with a 3x3 window, `weight.fan_in()` and `weight.fan_out()` are both 9.

### TheWeightAndTheResultTakeTheInputElementType

Over a `Float64` input, the weight's element type and the result's element type are `Float64`.

### OutputShapeAtStrideOneKeepsTheExtents

`output_shape([8, 6, 4])` is [8, 6, 4] for a padded 3x3 window at stride 1.

### OutputShapeAtStrideTwoHalvesTheExtents

`output_shape([8, 6, 4])` is [4, 3, 4] for a padded 3x3 window at stride 2.

### OutputShapeAgreesWithForwardAtStrideOneAndTwo

For the sample [8, 6, 4] in a batch of 3, the traced shape equals `output_shape` with the batch
extent in front, at stride 1 and at stride 2.

### OutputShapeAgreesWithForwardOverAnOddExtent

At stride 2, `output_shape([7, 9, 4])` is [4, 5, 4], and the traced shape of a batch of 2 is
that shape with the batch extent in front.

### SamePaddedBuildsAPaddedLayerOfTheGivenStride

`same_padded(4, 4, 5, 2)` gives 4 channels, a 5x5 window, stride 2 on both axes, a top and left
padding of 2, and dilation 1 on both axes.

### SamePaddedRefusesUnequalChannelCounts

`same_padded(4, 8, 3, 1)` throws an error naming "keeps its channel count".

### ForwardRejectsAnInputThatIsNotRankFour

`forward` over a [2, 16] input throws an error naming "rank-4".

### ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "NHWC".

### ForwardRejectsAChannelCountThatDiffersFromTheLayer

A four-channel layer over a [2, 8, 8, 3] input throws an error naming "[H, W, 4]".

### ForwardRejectsANonPositiveChannelCount

A layer with `channels` 0 throws an error naming "positive channel count".

### ForwardRejectsAWindowThatDoesNotFit

An unpadded 5x5 window over a [2, 4, 4, 4] input throws `Error`.

### OutputShapeRejectsAChannelCountThatDiffersFromTheLayer

`output_shape([8, 8, 5])` on a four-channel layer throws an error naming "[H, W, 4]".

### OutputShapeRejectsASampleOfTheWrongRank

`output_shape([8, 8])` throws an error naming "[H, W, 4]".

## test_dropout.cpp

That Dropout emits one DROPOUT node in training with its rate and site seed, returns its input in evaluation and at rate 0, gives each site its own seed, its rate check and its factory.

Suites: `Unit_Dropout`.

Tests `NN::Dropout` in `modules/layers/nn/include/SushiAI/nn/dropout.hpp` and its factory
`NN::make_dropout_node`. A `static_assert` at the top of the file holds that `Dropout` is a
`ModalModule` and is neither a `Module` nor a `TraceModule`. The training traces state the
seed 5.

### TheTrainingFormEmitsOneDropoutNode

A layer of rate 0.25 over a [2, 3, 4] input adds one DROPOUT node that reads the input. The
node holds 0.25 in `K_DROPOUT_RATE_SLOT`, its seed is `dropout_site_seed(5, 0)`, and the result
has the input's shape. The layer has no parameter and no buffer.

### TheEvaluationFormReturnsItsInputAndEmitsNothing

Under `Trace::evaluation` the layer returns its input, the trace is ok, and the node count and
the value count do not change.

### ARateOfZeroEmitsNothingInTraining

A default `Dropout` under a training trace returns its input and adds no node. The next
`claim_random_site()` of the trace returns 0.

### EachSiteOfATraceGetsItsOwnSeed

Two layers traced one after the other get the seeds `dropout_site_seed(5, 0)` and
`dropout_site_seed(5, 1)`. A second graph traced the same way gets the same two seeds.

### ATraceWithoutAStatedSeedHasSeedZero

`Trace::training` without a seed and `Trace::evaluation` report the seed 0. `Trace::training`
given 5 reports 5.

### ARateOutsideItsIntervalFails

For the rates 1, -0.1 and NaN, `trace_training` and `trace_evaluation` return their input and
both traces fail with `InvalidDescription` and a message containing "rate in [0, 1)". The graph
has a refusal.

### OutputShapeIsTheInputShape

`output_shape([6, 4])` is [6, 4].

### TheFactoryReadsTheRate

`make_dropout_node` over one [6, 4] input and `{"rate": 0.1}` returns a `ModuleNode<Dropout>`
whose layer has the rate 0.1 and whose one port has the shape [6, 4].

### TheFactoryRefusesARowThatDoesNotFit

Four rows fail with `InvalidDescription` and a message naming "layer 3": no rate ("'rate'",
"absent"), a rate of 1 ("'rate'", "[0, 1)"), a `seed` key ("'seed'") and two inputs ("reads
one input", "given 2").

## test_embedding.cpp

The Embedding layer's trace over an int32 index, its table parameter, its shape rule and its factory.

Suites: `Unit_Embedding`, `Unit_EmbeddingDeathTest`.

Tests `NN::Embedding` in `modules/layers/nn/include/SushiAI/nn/embedding.hpp` and its factory
`NN::make_embedding_node`. Every layer has a table of 7 rows of length 3. A `static_assert`
holds that `Embedding` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

### ATraceDeclaresTheTableAndEmitsOneGather

Over a [2, 5] int32 input the trace is ok and adds two nodes, one of them GATHER_ROWS. That node
reads the weight and then the indices, and its result is a float32 value of shape [2, 5, 3].

### TheTableIsAnEmbeddingDrawnFromANormalOfFixedDeviation

After a trace of a float64 layer the weight is declared with the shape [7, 3], the type float64
and the role `ParameterRole::Embedding`. Its initialiser is `Normal` with the deviation
`K_EMBEDDING_DEVIATION`, which is 0.02, and its fan-in and fan-out are both 3. The weight's
value has the lifetime `Parameter` and requires a gradient, and the result is float64.

### TheWalkYieldsTheTable

`NN::walk_state` under the prefix `tokens` visits one name, `tokens.weight`, as a parameter of
shape [7, 3], and visits no buffer.

### TwoGraphsNameTheSameTable

After traces of one layer into two graphs, `weight.state()` is the same `StateID` and is not
`K_NO_STATE`, and `weight.id_in` returns a value in each graph.

### AnInputThatIsNotInt32Fails

A float32 input and an int64 input each fail with `InvalidDescription` and a message containing
"Embedding(7, 3)", "int32 input of indices" and the type given. `forward` returns its input,
the graph gains no value and no node, the weight stays undeclared and the graph has a refusal.

### AnInputOfTheHighestRankFails

A rank-6 int32 input fails with `InvalidDescription`, "rank 1 to 5" and
"[1, 1, 1, 1, 1, 1]", and the weight stays undeclared.

### ALayerDescribedWrongFails

`num_embeddings` 0 and `embedding_dim` -1 each fail with "positive extents" and the field with
its value. An int32 table fails with "half, float32 or float64 table" and "int32". The weights
of the first and the third layer stay undeclared.

### OutputShapeAppendsTheRowLength

`output_shape` maps [5] to [5, 3], [4, 2] to [4, 2, 3] and [1, 1, 1, 1] to [1, 1, 1, 1, 3].

### TheFactoryReadsTheTwoExtents

`make_embedding_node` over one [5] input with `num_embeddings` 11 and `embedding_dim` 4 returns
a `ModuleNode<Embedding>` whose layer holds 11, 4 and float32. The node reads one input and its
one port has the shape [5, 4].

### TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `padding` key, an
absent `embedding_dim`, a `num_embeddings` of 0 ("positive"), two inputs ("reads one input",
"given 2") and a rank-5 input ("[1, 1, 1, 1, 1]", "rank 1 to 4").

### ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"Embedding(7, 3)", "value 7" and "holds 1 values".

### OutputShapeOverARankFiveSampleIsFatal

`output_shape([1, 1, 1, 1, 1])` stops the process with a report containing
"Embedding::output_shape" and "at most 4".

## test_flatten.cpp

The Flatten layer's trace, shape rule, rejections and parameter walk.

Suites: `Unit_Flatten`.

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `modules/layers/nn/include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

### traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

### SatisfiesTheModuleTrait

Two `static_assert` lines require `Flatten` to be a `Module` and not a `ModuleContainer`.

### ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a K_RESHAPE
with the input as its one operand and the result as its one output.

### ARank3InputFoldsItsTwoInnerAxes

A [2, 3, 4] input traces to [2, 12].

### ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

### AnNHWCInputYieldsAnNHWCResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

### ForwardRejectsAnNCHWInput

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

## test_global_avg_pool.cpp

The shapes GlobalAvgPool traces and reports, and the inputs it refuses.

Suites: `Unit_GlobalAvgPool`.

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `modules/layers/nn/include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

### traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

### SatisfiesTheModuleTrait

Three `static_assert` lines require `GlobalAvgPool` to be a `Module`, not to be a
`ModuleContainer`, and to have a `forward()` that is callable on a const object and returns
`Graph::ValueID`.

### ForwardReducesTheSpatialAxes

A [4, 7, 5, 16] input traces to [4, 16].

### ForwardEmitsOneSpatialMeanNode

The trace adds exactly one node, a K_SPATIAL_MEAN with the input as its one operand and the
result as its one output.

### ForwardKeepsTheNHWCLayoutTag

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

### ForwardRejectsAnNCHWInput

A [1, 4, 4, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace, and `parameter_elements` is 0 after it.

## test_host_table.cpp

`Trace::host_table`: a table is a buffer leaf listed with its values, two declarations are two leaves, and a trace without a sink, a wrong type or a wrong count is refused.

Suites: `Unit_HostTable`.

Tests `NN::Trace::host_table` and the `NN::HostTableList` a trace collects into with
`collect_tables`.

### ATableIsALeafListedWithItsValues

A [2, 3] float32 table of six values adds one node. Its leaf has that shape and type, the
lifetime `Buffer` and no gradient, and `graph.state_leaves()` stays empty. The list holds one
table with that leaf and the six values, and the trace is ok.

### EveryDeclarationIsItsOwnLeaf

Two float64 tables declared on a training trace return two different leaves, and the list
holds them in the order they were declared.

### ATableThatNoStepCanFillIsRefused

`host_table` fails with `InvalidDescription` and a message naming `'angles'` three times: on a
trace without `collect_tables` ("collect_tables"), for the type int32 ("float32 or a float64")
and for two values under the shape [2, 3] ("6 elements", "2 values"). The graph gains no node
and the list stays empty.

## test_identity.cpp

That the Identity layer returns its input, emits nothing and holds no state.

Suites: `Unit_Identity`.

Tests that the Identity layer returns its input, emits nothing and holds no state.

This file needs no device. `NN::Identity` is in `modules/layers/nn/include/SushiAI/nn/identity.hpp`.

Three `static_assert` lines at the top of the file hold that `Identity` is a `Module`, is not
a `ModuleContainer`, and that `Identity::fields()` is an empty tuple.

### ForwardReturnsItsInput

`forward` over a [2, 8, 6, 3] input returns the id it was given.

### ForwardEmitsNothing

The graph's node count and value count are the same before and after `forward`.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## test_layer_norm.cpp

The LayerNorm layer's trace, its gain and bias, its refusals and its factory.

Suites: `Unit_LayerNorm`, `Unit_LayerNormDeathTest`.

Tests `NN::LayerNorm` in `modules/layers/nn/include/SushiAI/nn/layer_norm.hpp` and its factory
`NN::make_layer_norm_node`. Every layer normalises over a last extent of 4. A `static_assert`
holds that `LayerNorm` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

### ATraceDeclaresTheGainAndTheBiasAndEmitsOneNode

Over a [2, 3, 4] input a training trace is ok and adds three nodes. The node that writes the
result is LAYER_NORM, reads the input, the weight and the bias in that order and holds
`K_LAYER_NORM_EPSILON` in attribute slot 0. The result is a float32 value of shape [2, 3, 4].

### TheGainIsAScaleAndTheBiasABias

After a trace over a [5, 4] float64 input, the weight and the bias are declared with the shape
[4] and the type float64, with the roles `ParameterRole::Scale` and `ParameterRole::Bias`. A
layer whose `epsilon` is 1e-3 puts 1e-3 in the node's attribute slot 0.

### TheWalkYieldsTheGainThenTheBias

`NN::walk_state` under the prefix `norm` visits `norm.weight` and then `norm.bias`.

### AnInputThatDoesNotFitFails

A [2, 5] input fails with `InvalidDescription`, "LayerNorm(4)", "[..., 4] input" and "[2, 5]".
An int32 input fails with "half, float32 or float64 input" and "int32". `forward` returns its
input, the graph gains no node, the weight stays undeclared and the graph has a refusal.

### ALayerDescribedWrongFails

A default layer, whose `dim` is 0, fails with "positive dim" and "got 0". An `epsilon` of 0
fails with "LayerNorm(4)" and "positive epsilon", and that layer's weight stays undeclared.

### OutputShapeIsTheInputShape

`output_shape([4])` is [4] and `output_shape([6, 4])` is [6, 4].

### TheFactoryReadsTheWidthAndTheEpsilon

`make_layer_norm_node` over one [6, 4] input and `{}` returns a `ModuleNode<LayerNorm>` whose
layer has the `dim` 4 and the epsilon `K_LAYER_NORM_EPSILON`, and whose one port has the shape
[6, 4]. With `{"ln_epsilon": 0.001}` the layer's epsilon is 0.001.

### TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `momentum` key, an
`ln_epsilon` of -1 ("positive"), an `ln_epsilon` of "small" ("a number"), two inputs ("reads
one input", "given 2") and a rank-6 input ("rank 1 to 5").

### ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"LayerNorm(4)", "value 7" and "holds 1 values".

## test_layers.cpp, tests/core/autograd/test_backward.cpp

Suites: `Unit_Backward`.

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`modules/core/graph/include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

The cases from DivReadsItsDivisorAndItsOwnQuotient on each put one operation under an K_NLL_LOSS
root and compare the appended nodes with a literal list of operation names. The first appended
node is always the K_SCALE that K_NLL_LOSS's rule emits, and its result is the gradient the tested
operation receives. The cases then check which operands each appended node reads, since a rule
that emits the right operations over the wrong operands gives a graph of the same shape.

### K_ATTRIBUTE_TOLERANCE

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule writes as an exact
constant.

### K_UNPROVEN_RULES

`K_UNPROVEN_RULES` is 8, the number of rules no finite-difference case names.
EveryRuleInTheTableIsNamedByANumericTest compares the size of its `unproven` list with it, so
adding a rule to that list takes an edit in two places.

### ForwardCase

`ForwardCase` holds a failure label and a function that emits one forward operation over two
parameters.

### Appended

`Appended` holds what one differentiation appended: the index of the first appended node, the
gradient the tested operation receives, and one gradient per requested value.

### differentiate_through

`differentiate_through` declares a `targets` input in the shape of a rank-2 result, puts
`nll_loss(result, targets)` over it and differentiates. It returns an `Appended` whose `seed`
is the first result of the first appended node.

### appended_node

`appended_node` returns the node a given number of places after the first appended one.

### EveryImplementedOperationHasARule

`has_backward_rule` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU.

`has_backward_rule` is false for K_LESS_EQUAL, K_CONSTANT and K_STOP_GRADIENT. Each is a gradient
barrier: the reverse walk steps over it, so no rule is ever asked of it.

`has_backward_rule` is false for K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD. An activation gradient
has no rule, so a second differentiation over one throws.

`has_backward_rule(K_BATCHNORM_UPDATE_STATS)` is false, and that is decision CV-7, not an
omission. A running mean is a buffer the training step writes beside the graph. No gradient
flows through it.

`has_backward_rule(K_BATCHNORM_FOLD)` is false too. The fold is the node an evaluation trace of
`NN::BatchNorm2d` emits, and DifferentiatingThroughBatchNormFoldFails holds what a gradient
request through it does.

`has_backward_rule` is false for K_LEAF and K_ZERO because both produce values from nothing. There
is nothing upstream to hand a gradient to, so there is no rule to write.

`has_backward_rule(K_CAST)` is false because of the order of the pipeline. Casts are inserted
after differentiation (`modules/core/graph/include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

### EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `modules/core/autograd/source/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::K_ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/core/autograd/rules/test_all.cpp` and all run in
FLOAT64, for the reason `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp` derives.

The K_LOG_SOFTMAX entries name the cases of `tests/language/blocks/test_causal_self_attention.cpp`:
the unfused rule against its closed form, a softmax, and a causal attention block.

The `unproven` vector lists the eight rules no finite-difference case names: K_DIV, K_MINIMUM,
K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU. Its size must equal `K_UNPROVEN_RULES`, every
entry must have a rule, and no operation may appear in both the ledger and this list. The loop
over `Ops::K_ALL` accepts a rule that either list names. An entry leaves `unproven` when a
finite-difference case names its rule in the ledger.

### TwoLayerMLPProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, K_ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

### TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one K_MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another K_MATMUL. It is skipped, not
computed and discarded.

### FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered K_ACCUMULATE form.

### ThreeContributorsChainTwoAccumulates

A parameter feeds a ReLU, a tanh and a sigmoid whose sum is the loss operand. After
`differentiate` the graph holds two ACCUMULATE nodes.

### ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no K_ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

### ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

### AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

### ANonScalarRootIsRejected

`differentiate` from a [4, 3] ReLU output throws `Error`.

### LogSoftmaxOutsideTheFusedPairAppendsItsGradientIdentity

A RELU between LOG_SOFTMAX and NLL_LOSS keeps the fusion away. The walk appends SCALE,
RELU_BACKWARD and one LOG_SOFTMAX_BACKWARD over the incoming gradient and the forward result.

### TheFusedPairAppendsNoLogSoftmaxGradient

NLL_LOSS directly over LOG_SOFTMAX appends EXP, SUB and SCALE, and no LOG_SOFTMAX_BACKWARD.

### ALogSoftmaxWhoseOperandNeedsNoGradientAddsNoNode

A log-softmax over an input, multiplied by a parameter, appends no gradient node of its own.

### ASoftmaxDifferentiatesThroughItsExponentialAndItsLogSoftmax

The LOG_SOFTMAX_BACKWARD of a softmax reads the MUL that the rule of EXP emitted and the
forward log-probabilities.

### DivReadsItsDivisorAndItsOwnQuotient

For `c = a / b` with both operands parameters, the appended nodes are K_SCALE, K_DIV, K_MUL, K_SCALE.
The K_DIV reads the seed and `b` and is the gradient of `a`. The K_MUL reads that quotient and the
forward result `c`, and the last K_SCALE negates it with the factor -1 and is the gradient of
`b`. The rule reuses `c` and does not divide by `b` a second time.

### DivEmitsOnlyTheQuotientWhenTheDivisorNeedsNoGradient

With `b` an input, the appended nodes are K_SCALE and K_DIV.

### DivEmitsTheDivisorGradientAloneWhenTheDividendNeedsNone

With `a` an input, the appended nodes are still K_SCALE, K_DIV, K_MUL, K_SCALE, because the divisor's
gradient is computed from the quotient. The one returned gradient is the result of the last
K_SCALE.

### MinimumRoutesByALessEqualBSoATieGoesToTheFirstOperand

For `minimum(a, b)` the appended nodes are K_SCALE, K_LESS_EQUAL, K_ZERO, K_SELECT, K_SELECT. The mask
reads `a` then `b`. The first K_SELECT reads the mask, the seed and the zero and is the gradient
of `a`; the second reads the mask, the zero and the seed and is the gradient of `b`. Where the
operands are equal the mask holds, so the first operand takes the gradient.

### MaximumRoutesByBLessEqualASoATieGoesToTheFirstOperand

For `maximum(a, b)` the node list is the same and the mask reads `b` then `a`. The two K_SELECT
nodes read the same operands as in the K_MINIMUM case, so a tie again goes to the first operand.

### MinimumEmitsOneSelectWhenOnlyTheSecondOperandNeedsAGradient

With `a` an input, one K_SELECT is appended, and it reads the mask, the zero and the seed.

### MaximumEmitsOneSelectWhenOnlyTheFirstOperandNeedsAGradient

With `b` an input, one K_SELECT is appended. The mask reads `b` then `a`, and the K_SELECT reads
the mask, the seed and the zero.

### AbsMultipliesByASignThatIsZeroAtZero

For `abs(x)` the appended nodes are K_SCALE, K_ZERO, K_LESS_EQUAL, K_LESS_EQUAL, K_SUB, K_MUL. The K_ZERO has
the shape of `x`. The first mask reads the zero then `x`, the second reads `x` then the zero,
the K_SUB reads the first mask then the second, and the K_MUL reads the seed and the difference.
The sign is `[0 <= x] - [x <= 0]`, which is 0 at x = 0.

### AtanDividesByOnePlusTheSquaredInput

For `atan(x)` the appended nodes are K_SCALE, K_MUL, K_CONSTANT, K_ADD, K_DIV. The K_MUL reads `x` twice.
The K_CONSTANT has no input, holds 1 and has the shape and element type of `x`. The K_ADD reads the
square and the one, and the K_DIV reads the seed and that sum.

### SoftplusBackwardReadsTheForwardInputNotTheOutput

For `softplus(x)` one K_SOFTPLUS_BACKWARD is appended after the K_SCALE, and it reads the seed and
`x`. The case asserts that `x` and the forward result are different values first, so the
operand check distinguishes the input from the output.

### SiLUBackwardReadsTheForwardInputNotTheOutput

For `silu(x)` one K_SILU_BACKWARD is appended after the K_SCALE, and it reads the seed and `x`.

### SelectRoutesItsGradientByItsOwnCondition

For `select(condition, a, b)` the appended nodes are K_SCALE, K_ZERO, K_SELECT, K_SELECT. The first
K_SELECT reads the condition, the seed and the zero and is the gradient of `a`. The second reads
the condition, the zero and the seed and is the gradient of `b`.

### SelectEmitsOneSelectWhenOnlyTheSecondBranchNeedsAGradient

With `a` an input, the appended nodes are K_SCALE, K_ZERO, K_SELECT, and the K_SELECT reads the
condition, the zero and the seed.

### SelectGivesItsConditionNoGradient

With the condition a parameter and both branches inputs, only the K_SCALE is appended and the
condition's gradient is `K_NO_VALUE`.

### AComparisonUnderASelectLeavesItsOperandsWithoutAGradient

The condition is `less_equal(p, q)` with `p` a parameter. The gradient of `p` is `K_NO_VALUE`,
the gradient of the branch `a` exists, and the appended nodes are K_SCALE, K_ZERO, K_SELECT.

### AComparisonMultipliedIntoAProductEmitsNothingForTheMask

For `mul(less_equal(p, q), a)` differentiation does not throw. The mask does not require a
gradient, the gradient of `p` is `K_NO_VALUE`, and the appended nodes are K_SCALE and one K_MUL that
reads the seed and the mask and is the gradient of `a`. K_MUL's rule emits no node for its mask
operand.

### AConstantAddedToAParameterPathEmitsNothingForTheConstant

For `(x + epsilon) * epsilon`, with `epsilon` a `constant_like` of `x`, differentiation does
not throw. The constant does not require a gradient, and the appended nodes are K_SCALE and one
K_MUL that reads the seed and the constant and is the gradient of `x`.

### MaximumOfXAndAZeroConstantRoutesATieToX

For `maximum(x, floor)` with `floor` a zero K_CONSTANT, the mask reads `floor` then `x`, and the
K_SELECT reads the mask, the seed and the zero. The mask `[floor <= x]` holds at x = 0, so `x`
takes the gradient there.

### MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie

For `maximum(floor, x)` the mask reads `x` then `floor`, and the K_SELECT reads the mask, the
zero and the seed. The mask `[x <= floor]` holds at x = 0, so `x` takes the zero branch there.
The operand order of a clamp therefore decides its gradient at the bound. The case also checks
that no node of the graph, forward or appended, narrows under the mixed-precision policy, so
the clamp stays in the loss's precision.

### NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows

For each of `div`, `minimum`, `maximum`, `abs`, `atan`, `softplus`, `silu` and a `select` on a
`less_equal` mask, the case differentiates the operation over two parameters. The rule appends
a node beyond the K_SCALE, and `Graph::narrows_under_mixed_precision` is false for every appended
node.

### AGradientOnTheBatchVarianceOfBatchNormThrows

`GraphBuilder::batchnorm2d` returns four values, and `batch_variance` is output 3 of the node.
A loss built on a K_SCALE of that value is differentiated against gamma. `differentiate` throws
with a message containing "K_BATCHNORM2D" and "saved statistics received a gradient".

### BatchNormWithFourOutputsStillDifferentiatesThroughItsFirst

A loss built on `sum_channels` of output 0 is differentiated against gamma and beta. Both
gradients are values other than `K_NO_VALUE` and the graph holds one K_BATCHNORM2D_BACKWARD node.
Inputs 3 and 4 of that node are the saved mean and the saved inverse deviation. The batch
variance is not among its operands.

### DifferentiatingThroughBatchNormFoldFails

A loss built on a K_CHANNEL_AFFINE over the two results of `batchnorm_fold` is differentiated
against gamma, a parameter. `differentiate` throws with a message containing "No backward
rule" and "sushiai.batchnorm_fold".

### ABufferInWrtIsRefusedAndAppendsNoNode

A request that lists a weight and a buffer named "running_shift" throws with a message
containing "differentiate()", "running_shift" and "buffer", and the graph's node count is
unchanged. The same graph then differentiates against the weight alone: the one gradient has
the lifetime `GRADIENT`, and the buffer keeps the lifetime `BUFFER`.

## test_layers.cpp, tests/core/autograd/test_backward.cpp

Suites: `Unit_LossRoots`.

Tests which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT
barrier.

This file needs no device. Every case in `Unit_LossRoots` builds a host-side graph, calls
`Autograd::differentiate` and reads the appended nodes or the error. A root produced by
K_NLL_LOSS is seeded by that operation's own rule. Any other one-element root of a real floating
type whose producer has a rule gets a K_CONSTANT of 1 as its gradient. Most refusal cases also
check that the node count is unchanged, so a refused call leaves the graph as it was.

### Constants

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule copies from the
forward node. `K_TERM_WEIGHT` is 2.0, the factor of the K_SCALE term in `WeightedSum`.
`K_ROOT_WEIGHT` is 0.5, the factor of the K_SCALE that ends a loss in the cases that need one.

### op_names

`op_names` returns the operation names of the nodes from a given index to the end. A case
passes the forward node count, so the list covers the appended nodes.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. The seed factor defaults to `K_NO_VALUE` and the provenance to null.

### WeightedSum

`WeightedSum` holds the loss `K_TERM_WEIGHT * a + b * c` over three one-element parameters:
a K_SCALE, a K_MUL and the K_ADD of the two. It keeps the ids of the parameters and of the three
results.

### ASumOfTwoTermsEndingInAddReachesEveryParameter

The root is the K_ADD. Four nodes are appended: K_CONSTANT, K_MUL, K_MUL, K_SCALE. The K_CONSTANT is the
seed. The two K_MUL nodes read the seed with `c` and with `b`, and produce the gradients of `b`
and `c`. The K_SCALE reads the seed with the factor `K_TERM_WEIGHT` and produces the gradient of
`a`. All three gradients have the lifetime GRADIENT.

### ASumOfTwoTermsEndingInScaleReachesEveryParameter

The root is a K_SCALE of the sum by `K_ROOT_WEIGHT`. The appended nodes are K_CONSTANT, K_SCALE, K_MUL,
K_MUL, K_SCALE. The first K_SCALE reads the seed with the factor `K_ROOT_WEIGHT`, and the three nodes
after it read that K_SCALE's result, so both terms under the root receive the scaled seed. No
gradient is `K_NO_VALUE`.

### TheSeedIsAConstantOneShapedAndTypedLikeTheRoot

The root is a FLOAT64 [1, 1, 1, 1] value tagged NCHW. The first appended node is a K_CONSTANT
with no input and the value 1 in attribute slot 0, and its result is [1, 1, 1, 1], FLOAT64 and
NCHW. The graph holds one K_CONSTANT, no K_ZERO and no K_EXP.

### TheSeedFactorMultipliesTheSeedOnce

With a seed factor, the appended nodes are K_CONSTANT, K_DYNAMIC_SCALE, K_MUL, K_MUL, K_SCALE. The
K_DYNAMIC_SCALE reads the K_CONSTANT and the factor, the graph holds one K_DYNAMIC_SCALE, and the
three rule nodes read its result.

### ANonScalarRootThrowsNamingItsShape

A [4, 3] root throws with a message containing "needs a scalar to differentiate" and "[4, 3]".

### ALeafRootThrowsNamingTheLeaf

A parameter passed as the root throws with a message containing "it is a leaf" and
"sushiai.leaf".

### ARootThatNeedsNoGradientThrowsNamingTheCause

The root is an K_ADD of two inputs and the requested parameter is not read by it. The message
contains "requires no gradient" and "sushiai.add".

### ARootWhoseProducerHasNoRuleThrowsNamingTheOperation

The root is a K_DYNAMIC_SCALE of a parameter. The message contains "no backward rule" and
"sushiai.dynamic_scale".

### ARootThatIsNotTheFirstResultThrowsNamingTheProducer

The root is the index result of a K_MAXPOOL2D, its second output. The message contains "not the
first result" and "sushiai.maxpool2d".

### ARootThatIsNotRealFloatingThrowsNamingItsType

The root is a K_SCALE of an INT32 parameter. The message contains "element type is int32" and
"sushiai.scale".

### AFusedCrossEntropyRootEmitsNoSeedNode

A two-layer `NN::Sequential` under `NN::CrossEntropyLoss` appends K_EXP, K_SUB, K_SCALE, K_SUM_ROWS,
K_MATMUL, K_MATMUL, K_RELU_BACKWARD, K_SUM_ROWS, K_MATMUL. The graph holds no K_ZERO and no K_CONSTANT: the
fused rule starts from the root's implicit gradient and needs no seed node.

### AnUnfusedNLLLossRootEmitsNoSeedNode

K_NLL_LOSS over a K_RELU result appends K_SCALE then K_RELU_BACKWARD. The K_SCALE reads the targets
alone, so no seed node precedes it.

### AnNLLLossRootAppliesTheSeedFactorInItsOwnRule

With a seed factor, a K_LOG_SOFTMAX and K_NLL_LOSS pair appends K_EXP, K_SUB, K_SCALE, K_DYNAMIC_SCALE, and
the result of the K_DYNAMIC_SCALE is the gradient of `x`.

### StopGradientEmitsAnIdentityShapedLikeItsOperand

`stop_gradient` of a FLOAT64 NCHW value writes one K_STOP_GRADIENT node that reads the operand.
The result has the operand's shape and is FLOAT64 and NCHW, and `finalise()` accepts the graph.

### StopGradientIsDeclaredAGradientBarrierAndNotDifferentiable

`Ops::K_STOP_GRADIENT` has `blocks_gradient` true and `differentiable` false.
`Ops::is_gradient_barrier` is true for it and false for K_RESHAPE, and its name is
"sushiai.stop_gradient".

### RequiresGradDoesNotCrossStopGradient

After `finalise()`, a parameter requires a gradient and so does a K_SCALE that reads it. The
K_STOP_GRADIENT result and a K_SCALE that reads that result do not.

### StopGradientBlocksAParameterReachableOnlyThroughIt

The loss is `stop_gradient(scale(blocked)) * open`. Differentiation does not throw and appends
K_CONSTANT and K_MUL. The gradient of `blocked` is `K_NO_VALUE`, and the gradient of `open` is the
K_MUL of the seed and the stopped value.

### AValueReadDirectlyAndThroughStopGradientKeepsTheDirectGradient

The loss is `stop_gradient(x) * x`. The appended nodes are K_CONSTANT and K_MUL, the K_MUL reads the
seed and the stopped value, and its result is the gradient of `x`. The graph holds no
K_ACCUMULATE, so the stopped path contributes nothing to the sum.

### ALossReachableOnlyThroughStopGradientThrowsNamingTheBarrier

The root is a K_SCALE of `stop_gradient(x)`. The message contains "requires no gradient" and
"sushiai.scale".

### AStopGradientRootThrowsForRequiringNoGradient

The root is the K_STOP_GRADIENT result. The message contains "requires no gradient" and
"sushiai.stop_gradient".

### AContributionToAStopGradientResultIsDroppedWithoutAskingForARule

The graph is that of StopGradientBlocksAParameterReachableOnlyThroughIt, with `requires_grad`
of the stopped value set by hand after the graph is built. `finalise()` only sets the flag, so
the hand-set value survives it and K_MUL's rule contributes a gradient to the stopped value.
Differentiation does not throw. It appends K_CONSTANT, K_MUL, K_MUL, the gradient of `blocked` is
`K_NO_VALUE`, and no appended node has the K_STOP_GRADIENT node as its origin in the `Provenance`.
The reverse walk steps over the barrier and asks no rule of it.

### TheSeedNodesHaveTheRootProducerAsOrigin

`WeightedSum` is differentiated with a seed factor and a `Provenance`. Five nodes are appended.
The origin of the K_CONSTANT and of the K_DYNAMIC_SCALE is the K_ADD that produces the root. The
origin of the two K_MUL nodes is the forward K_MUL, and the origin of the K_SCALE is the forward
K_SCALE.

### AConstantRootThrowsForRequiringNoGradient

The root is a `constant_like` of a parameter. The message contains "requires no gradient" and
"sushiai.constant".

### AComparisonRootThrowsForRequiringNoGradient

The root is `less_equal(x, bound)` over a parameter. The message contains "requires no
gradient" and "sushiai.less_equal".

### RequiresGradCrossesNoGradientBarrier

After `finalise()`, a K_LESS_EQUAL mask over a parameter, a K_CONSTANT, a K_ZERO and the K_MUL of the
mask with the K_CONSTANT require no gradient. The K_MUL of the mask with the parameter does,
through its second operand.
