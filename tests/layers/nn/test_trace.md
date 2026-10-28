# tests/layers/nn/test_trace.cpp


Suites: `Small_Trace`, `Small_TraceDeathTest`.

Tests Trace, the buffer-update sinks, the three layer forms and trace_layer.

This file needs no device. `NN::Trace` in `modules/layers/nn/include/SushiAI/nn/trace.hpp` carries a
`Graph::GraphBuilder`, a `TraceMode` and the sink that receives buffer updates. A layer has one
of three forms: `Module`, whose `forward` takes a builder; `TraceModule`, whose `forward` takes
a `Trace`; and `ModalModule`, which states `trace_training` and `trace_evaluation` and no
`forward`. `NN::trace_layer` calls whichever form a layer has.

## K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

## ModalStub

`ModalStub` has the modal pair and an empty `fields()`. `trace_training` emits one ReLU and
`trace_evaluation` returns its input, and each counts its calls.

## CompositeStub

`CompositeStub` lists a `ModalStub` as the child "inner". Its `forward` takes a `Trace` and
passes it to the child through `NN::trace_layer`.

## ByValueTraceStub

`ByValueTraceStub` has a `forward` that takes its `Trace` by value.

## TrainingOnlyStub

`TrainingOnlyStub` states `trace_training` and no `trace_evaluation`.

## ModalWithForwardStub

`ModalWithForwardStub` states the modal pair and a builder `forward` together.

## BothForwardsStub

`BothForwardsStub` has a builder `forward` and a trace `forward` and counts the calls of each.

## FieldlessStub

`FieldlessStub` has the modal pair and no `fields()`.

## RefusingStub

`RefusingStub` has a trace `forward` that records an `InvalidDescription` failure on the trace
through `Trace::fail` and returns its input.

## failure_saying

`failure_saying` returns an `InvalidDescription` failure with a given message.

## defined_buffer

`defined_buffer` returns a `Buffer` defined as [3] with the starting value 0.0 that no graph
has read.

## update_undefined_buffer

`update_undefined_buffer` requests an update of a buffer on which `define()` was never called.
It is the statement of a death test.

## update_from

`update_from` requests an update whose source is a given id, in a graph that holds one value.
It is the statement of two death tests.

## parameter_names

`parameter_names` collects the names `NN::for_each_parameter` yields, in walk order.

## The layer forms at compile time

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

## AFactoryStatesItsModeAndKeepsItsBuilder

`Trace::training(builder, list)` reports `TraceMode::Training` and `Trace::evaluation(builder)`
reports `TraceMode::Evaluation`. `builder()` of each returns the address of the builder it was
made from.

## UpdateUnderAnEvaluationTraceFails

`update()` on an evaluation trace returns `InvalidDescription` with a message containing
"evaluation trace".

## ANewListIsEmpty

A new `BufferUpdateList` has no update and no source.

## EachUpdateAddsOneUpdateAndOneSource

Two `update()` calls on a training trace leave two entries in `updates()` and two in
`sources()`. Entry 0 holds the first buffer's state, the source x, the momentum 0.1 and the
source scale 1.0. Entry 1 holds the second buffer's state, the source y, 0.25 and 1.5.
`sources()` is x then y.

## AnUpdateAddsNothingToTheGraph

An `update()` leaves the graph's value count unchanged and the buffer without a leaf in it.

## ARecordedTargetSurvivesAMoveOfItsBuffer

After the buffer of a recorded update is move-constructed into another, the update's `target`
equals the moved buffer's state and is not `K_NO_STATE`.

## TheDiscardingSinkAcceptsAnUpdateAndIsOneObject

A trace made with `NN::discard_updates()` is in Training mode, its `update()` returns ok and
the graph gains no value. Two calls of `discard_updates()` return the same address.

## AnUpdateOfAnUndefinedBufferIsFatal

The case is in the suite `Small_TraceDeathTest`. `update_undefined_buffer` stops the process
with "define() has not given a shape".

## AnUpdateOfABufferThatWasNeverReadIsRecorded

An update of a defined buffer that no graph has read is recorded with the buffer's state as
its target. The graph gains no value and the buffer has no leaf in it.

## ANewTraceHoldsNoFailure

A new trace's `status()` is ok and the graph has no refusal.

## FailKeepsTheFirstFailureAndMarksTheGraphRefused

After `fail` with "the first reason" and then "the second reason", `status()` carries the
first message and does not contain "second". `graph.has_refusal()` is true.

## ValueOrFailReturnsAHeldValue

`value_or_fail(Result<ValueID>(y), x)` returns y and leaves `status()` ok.

## ValueOrFailRecordsAFailureAndReturnsTheFallback

`value_or_fail` over a failed result returns the fallback x, and `status()` then carries that
failure.

## FailGivenASuccessIsFatal

The case is in the suite `Small_TraceDeathTest`. `trace.fail(Status())` stops the process with
"Trace::fail was given a success".

## SequentialLeavesTheFailureAChildRecordedOnTheTrace

`Sequential{RefusingStub, ModalStub}::forward` over a trace returns `InvalidDescription` nothing and returns its input.
The trace's status is `InvalidDescription` with a message containing "RefusingStub refuses". The
second child's `trace_evaluation` is never called.

## SequentialRecordsTheRefusalOfABuilderFormChild

`Sequential{Linear{4, 8}, ReLU, ModalStub}::forward` over a trace of a [2, 5] input returns `InvalidDescription` nothing.
The trace's status is `InvalidDescription` with a message containing "Linear(4, 8)", the graph is
marked refused, and the `ModalStub` is never called.

## SequentialFoldsAnInputShapeThroughItsChildren

`output_shape([4])` of `Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` is [3]. `output_shape([5])`
returns InvalidDescription with a message containing "Linear(4, 8)" and "[5]".

## AnUpdateWithoutASourceIsFatal

The case is in the suite `Small_TraceDeathTest`. An update whose source is `K_NO_VALUE` stops the
process with "a buffer update reads value".

## AnUpdateFromAValueOutsideTheGraphIsFatal

The case is in the suite `Small_TraceDeathTest`. An update whose source is value 7, in a graph
of one value, stops the process with a report containing "reads value 7" and "holds 1 values".

## TraceLayerCallsTheTrainingFormInTrainingMode

Under a training trace, `trace_layer` calls `ModalStub::trace_training` once and
`trace_evaluation` never. The result differs from the input, because the training form emits a
ReLU.

## TraceLayerCallsTheEvaluationFormInEvaluationMode

Under an evaluation trace, `trace_layer` calls `trace_evaluation` once and `trace_training`
never, and returns its input.

## TraceLayerTracesAModuleThroughTheBuilder

`trace_layer` on a `Linear{4, 3}` and `Linear::forward` on a second one, each in its own graph,
return the same id and leave the same value count. The traced result has the shape [2, 3].

## TraceLayerPassesTheTraceToATraceModule

`CompositeStub` is traced once under an evaluation trace and twice under a training trace. Its
child counts one evaluation call and two training calls.

## TraceLayerPrefersTheTraceForwardOverTheBuilderForward

On `BothForwardsStub`, `trace_layer` calls the trace `forward` once and the builder `forward`
never.

## ANestedSequentialTracesItsModalChildInTheTracesMode

A `ModalStub` inside a `Sequential` inside a `Sequential` counts one evaluation call after an
evaluation trace of the outer container, and one training call after a training trace of it.

## SequentialTracesAModalChildInTheTracesMode

`Sequential{Linear{4, 3}, ModalStub}` traced under each mode calls the stub's matching form
once and yields a [2, 3] value both times. Its parameter walk yields "0.weight" and "0.bias".

## SequentialOfModulesTracesTheSameThroughBothForwards

`Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` traced through the trace `forward` in one graph
and through the builder `forward` in another returns the same id, leaves the same value count
and walks the same parameter names.

## AnEvaluationTraceRefusesAFedLeafAndACarriedValue

An evaluation trace is not carrying. `feed` fails with `InvalidDescription`, "'position'" and
"carrying trace"; `carried_leaf` with "'kept'" and "carrying trace"; `carry` with
"carrying trace". No node is added.

## ATrainingTraceRefusesAFedLeaf

A training trace is not carrying, and `feed` fails with `InvalidDescription` and
"'position'".

## ACarryingTraceIsAnEvaluationTraceThatRefusesABufferUpdate

A carrying trace reports `is_carrying` and the mode `TraceMode::Evaluation`. `update` fails
with `InvalidDescription` and "evaluation trace", and no carried value is recorded.

## AFedLeafIsDeclaredOncePerName

Two `feed` calls for "position" return the same value and add one node between them; `feed`
for "mask" returns another. The value is an `Int32` input leaf. The fed list holds "position"
then "mask", `find("mask")` returns the mask's value and `find("absent")` returns
`K_NO_VALUE`.

## AFedLeafDeclaredAgainAsAnotherLeafIsRefused

After "mask" is fed as [1, 4] `Float32`, feeding it as [1, 5] fails with
`InvalidDescription`, "'mask'", "[1, 4]" and "[1, 5]", and as `Float64` with "'mask'". The
fed list still holds one leaf.

## CarryRecordsTheLeafItsNextValueAndItsInitialValue

`carried_leaf` declares an input leaf that is not in the fed list. After
`carry(leaf, next, 0.5)` the carried list holds one entry with that leaf, that next value and
the initial value 0.5, and `sources` holds the next value.

## CarryRefusesANextValueOfAnotherShape

`carry` of a [4, 3] leaf with a [3, 4] next value fails with `InvalidDescription`, "[4, 3]"
and "[3, 4]", and records nothing.

## CarryingAValueThatIsNoCarriedLeafIsFatal

`carry` of a value that `carried_leaf` did not declare is fatal with "did not declare".
