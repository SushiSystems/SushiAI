# tests/layers/nn/test_node.cpp


Suites: `Small_Node`, `Small_NodeDeathTest`.

Tests ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time.

This file needs no device. `NN::Node` in `modules/layers/nn/include/SushiAI/nn/node.hpp` is the seam a
`ModelGraph` holds its rows through: a node declares its ports, reads a span of values and
appends its results. `NN::ModuleNode<L>` in `modules/layers/nn/include/SushiAI/nn/module_node.hpp` wraps a
layer of any of the three forms, and `NN::LayerNode` in `modules/layers/nn/include/SushiAI/nn/layer_node.hpp`
wraps an `NN::Layer`. Both return a `Status` from `forward`, and a layer's refusal or an `Error`
a trace-form layer throws comes back as a failure.

## BufferedStub

`BufferedStub` lists one `Buffer` as "running" and defines it as [4] zeros in its builder
`forward`.

## BufferedParentStub

`BufferedParentStub` lists a `BufferedStub` as the child "inner".

## ModalStub

`ModalStub` has the modal pair, counts the calls of each form and returns its input.

## TraceStub

`TraceStub` has a trace `forward` that records the mode of each trace it is given.

## ShapelessStub

`ShapelessStub` has `fields()` and a builder `forward` and no `output_shape`.

## RefusingStub

`RefusingStub` records an `InvalidDescription` failure on the trace and returns its input.

## RefusingThenThrowingStub

`RefusingThenThrowingStub` records an `OutOfMemory` failure on the trace and then throws an
`Error` with a different message.

## node_of

`node_of` calls `ModuleNode<L>::create` over a layer and an input shape, expects success and
returns the node.

## span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

## state_names

`state_names` collects the name of every parameter and buffer a node's `walk_state` yields
under a prefix.

## make_test_conv2d

`make_test_conv2d` calls `NN::make_conv2d` with 5 filters and a 3x3 kernel, the layer a
`conv2d` row of a model file builds.

## The two predicates at compile time

`LayerAdapter` asserts that `Detail::HoldsBufferField` is false for its layer. The predicate is
true for `BufferedStub`, for `BufferedParentStub`, for `Sequential<Linear, BufferedStub>` and
for `ModuleList<BufferedStub>`, and false for `Linear` and `Sequential<Linear, ReLU>`.

`ModuleNode` asserts that `Detail::HasOutputShapeOfInput` is true for its layer. The predicate
is true for `Linear` and `TraceStub` and false for `ShapelessStub`.

## ModuleNodeDeclaresOnePortWithTheLayersOutputShape

The node of `Linear{6, 8}` over [6] has one port named "output" of shape [8].

## ALayerNodeOfEitherKindReadsOneValue

`input_count()` is 1 for a `ModuleNode<Linear>` and for a `LayerNode`.

## ModuleNodeCreateReturnsTheFailureOfAShapeTheLayerRefuses

`ModuleNode<Linear>::create(Linear{6, 8}, [5])` returns `InvalidDescription` with a message
containing "Linear(6, 8)", "[5]" and "[6]". `Linear::output_shape` returns that failure, and
`create` passes it on.

## ModuleNodeForwardReturnsTheFailureOfALayerThatRefuses

`forward` over a [2, 5] value returns `InvalidDescription` with "Linear(6, 8)" and "[2, 5]",
and `outputs` stays empty.

## ModuleNodeForwardReturnsTheFailureALayerRecorded

The node of `RefusingStub` returns the failure the stub recorded on the trace, and `outputs`
stays empty although the stub returned an id.

## ModuleNodeForwardReturnsTheRecordedFailureBeforeALaterThrow

The node of `RefusingThenThrowingStub` returns the `OutOfMemory` failure with "the stub's
recorded cause". The `Error` thrown after it is not what the caller sees.

## ModuleNodeForwardAppendsTheLayersResult

`outputs` holds one id before the call and two after it. The first is untouched and the second
is a value of shape [2, 8].

## ModuleNodeTracesAModalLayerInTheModeOfTheTrace

Under a training trace the stub counts one training call and no evaluation call. After an
evaluation trace it counts one of each.

## ModuleNodeHandsTheTraceToATraceModule

A training trace and then an evaluation trace leave `TraceStub::modes` as Training,
Evaluation, and `outputs` as the input id twice.

## ModuleNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("stem", ...)` of the node of `Linear{6, 8}` yields "stem.weight" and
"stem.bias".

## ModuleNodeWalksABufferOfTheLayer

After a trace, `walk_state("bn", ...)` of the node of `BufferedStub` yields "bn.running".

## ModuleNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

## LayerNodeReportsTheShapeOfTheLayerItWraps

A `LayerNode` over the conv2d layer built for [8, 8, 2] has one port named "output" whose
shape is the layer's `output_shape()`, [6, 6, 5].

## LayerNodeForwardAppendsAValueShapedLikeItsPort

`forward` over a [2, 8, 8, 2] value appends one id of shape [2, 6, 6, 5].

## LayerNodeForwardReturnsTheFailureOfALayerThatThrows

A `LayerNode` over `LayerAdapter<Linear>` given a [2, 5] value returns `InvalidDescription`
with "Linear(6, 8)" and "[2, 5]", and `outputs` stays empty.

## LayerNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("0", ...)` yields "0.weight" and "0.bias".

## LayerNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

## LayerNodeRejectsNoInput

`forward` given an empty span returns `InvalidDescription` with "input count is 1" and "given
is 0".

## LayerNodeOverANullLayerIsFatal

The case is in the suite `Small_NodeDeathTest`. Constructing a `LayerNode` from a null pointer
stops the process with a report containing "LayerNode" and "null layer".
