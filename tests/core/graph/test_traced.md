# tests/core/graph/test_traced.cpp


Suites: `Unit_Traced`.

Tests Traced and the refusal it carries through a chain of emitters.

This file needs no device. `Graph::Traced` in `modules/core/graph/include/SushiAI/graph/traced.hpp` holds a
`ValueID` or a failed `Status`. An emitter that returns one takes its operands as `Traced`
too, so a call can be nested inside another and the first refusal travels to the outermost
result.

Six `static_assert` lines at the top of the file hold that a `Traced` is default-constructible
and constructible from a `ValueID` and from a `Status` without throwing, that both conversions
are implicit, and that its size is two pointers.

The cases up to BuildingFromASuccessStatusIsFatal use `StandInBuilder` and involve no graph.
The last four use a `Graph::GraphBuilder` and read `Graph::has_refusal()`.

## TraceLog

`TraceLog` counts the values the stand-in builder emitted and the refusals it recorded.

## StandInBuilder

`StandInBuilder` has two members. `emit()` returns the next id, and `refuse()` counts one
refusal and returns the failure it was given. `SA_REFUSE_IF` is run over it in `emit_below`.

## emit_below

`emit_below` binds its operand through `SA_TRY_ASSIGN`, refuses through `SA_REFUSE_IF` when
the id is above a limit, and otherwise emits one value.

## emit_over

`emit_over` binds its operand through `SA_TRY_ASSIGN` and emits one value. It has no condition
of its own, so it accepts the absent operand.

## forward_traced

`forward_traced` runs `SA_TRY` on a `Traced` and sets a flag after it.

## declare_below

`declare_below` runs `SA_REFUSE_IF` over a `Graph::GraphBuilder` when an extent is above a
limit, and otherwise declares one input of that extent.

## description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

## DefaultHoldsTheAbsentValueAndNoFailure

A default-constructed `Traced` has a value, an ok status and the id `Graph::K_NO_VALUE`.

## HoldsAnIDWithASuccessStatus

`Traced(ValueID{12})` has a value, is ok and returns 12.

## HoldsAFailureAndTheAbsentID

A `Traced` built from a failure has no value and reports the failure's code and message.

## SuccessPathCreatesNoFailureBlock

Two chained emitters over an absent operand, a copy and a move yield the id 1, record no
refusal and leave `live_failure_count()` where it was.

## MovedFromTracedKeepsItsState

A failed `Traced` that was moved from still reports its failure, as does the target. A
`Traced` holding 4 that was move-assigned from still returns 4, as does the target.

## AssignmentReplacesAFailureWithAnID

Assigning `ValueID{3}` to a failed `Traced` leaves a value and the id 3. Assigning a failure to
it afterwards leaves that failure.

## AnAbsentOperandPassesThroughAnEmitter

`emit_over(builder, Traced())` succeeds with the id 0, and the log shows one emitted value.

## RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing

`emit_below` given the id 9 and the limit 4 fails with "operand 9 is above the limit 4",
records one refusal and emits nothing. That result is then passed through `emit_below` with
the limit 100, nested inside `emit_over`. The outer result carries the same message, and its
`message().data()` is the first result's pointer, so the failure was forwarded and not rebuilt.
The log still shows one refusal and no emitted value.

## NestedCallsEmitWhenNothingIsRefused

`emit_over(builder, emit_below(builder, ValueID{2}, 4))` yields the id 1 with two values
emitted and no refusal.

## TryForwardsATracedFailureAsAStatus

`forward_traced` over a failed `Traced` returns the failure and leaves its flag false. Over
`ValueID{1}` it succeeds and sets the flag.

## ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Traced` stops the process with a report containing "Traced::value() was
called on a failure" and the failure's message.

## BuildingFromASuccessStatusIsFatal

`Traced(Status())` stops the process with "a Traced was built from a success Status".

## RefuseMarksTheGraphAndReturnsTheFailure

`GraphBuilder::refuse` returns the failure it was given, with the same message pointer, and
`graph.has_refusal()` goes from false to true.

## RefuseIfOverAGraphBuilderMarksTheGraphOnlyOnARefusal

`declare_below(builder, 4, 8)` declares value 0 and leaves the graph without a refusal.
`declare_below(builder, 9, 8)` fails with "extent 9 is above the limit 8", adds no value and
marks the graph.

## ObserveMarksTheGraphForAFailedStatusOnly

`GraphBuilder::observe(Status())` is ok and leaves the graph unmarked. `observe` of a failed
status returns that failure and marks the graph.

## ObserveMarksTheGraphForAFailedResultOnly

`observe(Result<int>(7))` returns a result holding 7 and leaves the graph unmarked. `observe`
of a failed `Result<int>` returns the failure and marks the graph.
