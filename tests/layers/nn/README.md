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
the trace throws `Error`. It uses `Test::capture_error` from `tests/common/test_throws.hpp`,
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
`trace_summary` come from `tests/common/test_utils.hpp`.

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
