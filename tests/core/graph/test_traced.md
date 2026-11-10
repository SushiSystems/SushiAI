# tests/core/graph/test_traced.cpp


Suites: `Small_Traced`, `Medium_TracedFatal`.

Tests Traced and the refusal it carries through a chain of emitters.

This file needs no device. `Graph::Traced` in `modules/core/graph/include/SushiAI/graph/traced.hpp` holds a
`ValueID` or a failed `Status`. An emitter that returns one takes its operands as `Traced`
too, so a call can be nested inside another and the first refusal travels to the outermost
result.

Six `static_assert` lines at the top of the file hold that a `Traced` is default-constructible
and constructible from a `ValueID` and from a `Status` without throwing, that both conversions
are implicit, and that its size is two pointers.

The emitter cases run over a real `Graph::GraphBuilder` and read `Graph::value_count()` and
`Graph::has_refusal()`.

## Rig

`Rig` holds an empty `Graph` and the `GraphBuilder` over it.

## emit_below

`emit_below` binds its operand through `SA_TRY_ASSIGN`, refuses through `SA_REFUSE_IF` when
the id is above a limit, and otherwise declares one input.

## emit_over

`emit_over` binds its operand through `SA_TRY_ASSIGN` and declares one input. It has no condition
of its own, so it accepts the absent operand.

## forward_traced

`forward_traced` runs `SA_TRY` on a `Traced` and sets a flag after it.

## declare_below

`declare_below` runs `SA_REFUSE_IF` over a `Graph::GraphBuilder` when an extent is above a
limit, and otherwise declares one input of that extent.

## description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

## HoldsAFailureAndTheAbsentID

A `Traced` built from a failure has no value and reports the failure's code and message.

## SuccessPathCreatesNoFailureBlock

Two chained emitters over an absent operand, a copy and a move yield the id 1, leave the graph
without a refusal and leave `live_failure_count()` where it was.

## AMovedToTracedHoldsTheFailureItWasBuiltWith

A `Traced` move-constructed from a failed one reports that failure.

## MoveAssignmentHandsOverTheId

A `Traced` move-assigned from one holding 4 returns 4.

## AssigningAnIDReplacesAHeldFailure

Assigning `ValueID{3}` to a failed `Traced` leaves a value and the id 3.

## AssigningAFailureReplacesAHeldID

Assigning a failure to a `Traced` holding an id leaves that failure.

## AnAbsentOperandPassesThroughAnEmitter

`emit_over(rig.builder, Traced())` succeeds with the id 0, and the graph holds one value.

## RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing

`emit_below` given the id 9 and the limit 4 fails with "operand 9 is above the limit 4" and
marks the graph. That result is then passed through `emit_below` with the limit 100, nested
inside `emit_over`. The outer result carries the same message. The graph still holds no value.

## NestedCallsEmitWhenNothingIsRefused

`emit_over(builder, emit_below(builder, ValueID{2}, 4))` yields the id 1 with two values in the
graph and no refusal.

## TryForwardsATracedFailureAsAStatus

`forward_traced` over a failed `Traced` returns the failure and leaves its flag false.

## TryPassesASuccessfulTracedValueOn

`forward_traced` over `ValueID{1}` succeeds and sets the flag.

## ValueOnAFailureIsFatalAndPrintsTheFailure

Size (RW-166): the death check runs in a child process and took more than the 0.1 s of a
`small` test on the debug build, so the case is in `Medium_TracedFatal`; a death test is `medium` by its
size rule.

`value()` on a failed `Traced` stops the process with a report containing "Traced::value() was
called on a failure" and the failure's message.

## BuildingFromASuccessStatusIsFatal

Size (RW-166): the death check runs in a child process and took more than the 0.1 s of a
`small` test on the debug build, so the case is in `Medium_TracedFatal`; a death test is `medium` by its
size rule.

`Traced(Status())` stops the process with "a Traced was built from a success Status".

## RefuseMarksTheGraphAndReturnsTheFailure

`GraphBuilder::refuse` returns the failure it was given, with the same message pointer, and
`graph.has_refusal()` goes from false to true.

## RefuseIfOverAGraphBuilderDoesNotMarkTheGraphOnSuccess

`declare_below(builder, 4, 8)` declares value 0 and leaves the graph without a refusal.

## RefuseIfOverAGraphBuilderMarksTheGraphAndEmitsNothingOnRefusal

`declare_below(builder, 9, 8)` fails with "extent 9 is above the limit 8", adds no value and
marks the graph.

## ObserveDoesNotMarkTheGraphForASuccessStatus

`GraphBuilder::observe(Status())` is ok and leaves the graph unmarked.

## ObserveMarksTheGraphForAFailedStatus

`observe` of a failed status returns that failure and marks the graph.

## ObservePassesASuccessfulResultThroughWithoutMarkingTheGraph

`observe(Result<int>(7))` returns a result holding 7 and leaves the graph unmarked.

## ObserveMarksTheGraphForAFailedResult

`observe` of a failed `Result<int>` returns the failure and marks the graph.
