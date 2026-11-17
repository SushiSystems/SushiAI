# tests/core/eager/test_step.cpp


Suites: `Small_EagerStep`, `Small_EagerStepRefusal`.

Tests what a `Step` accepts while it traces and when it runs.

A step belongs to a `Session`. Every case builds the `SmallLoop` of `eager_run_support.hpp`
over the two-layer model, on the host backend and an executor that runs no region, and opens one
step on its session, so the file is small. `test_session.md` covers training through steps, reading values
and the refusals that involve the session's states.

## AStateIsDeclaredOncePerStep

Asking a step for the same state twice returns one leaf, and the graph binds that state to it.

## AnEmptyInputIsRefused

`input()` with a default tensor returns a failed `Graph::Traced` with `InvalidDescription`,
naming the label "batch" and "no element", and marks the step's graph as refused.

## FeedingTheResultOfAnOperationIsRefused

`feed()` into the result of a K_SCALE fails with `InvalidDescription` and "cannot be fed": only a
leaf takes a feed.

## KeepingAnUnknownValueIsRefused

`keep()` of a value id 100 past the input fails with `InvalidDescription` and "does not exist".

## BackwardIsCalledOnce

A second `backward()` on the same loss fails with `InvalidDescription` and "called once".

## ReadScalarRefusesAValueOfManyElements

`read_scalar()` of a K_RELU over the [8, 6] input runs the step, then fails with
`InvalidDescription` and "holds 48 elements". `has_run()` is true afterwards.

## AnEntryPointReturnsTheRefusalOfAFailedTrace

Rows over `EntryPoint`: `backward()` and `read_scalar()` of a K_MATMUL of the [8, 6] input with
itself, which the builder refused. Each fails with `InvalidDescription`, "MATMUL" and "inner
extents must agree", and the step has not run.

## BackwardRefusesAGraphHoldingARefusal

After a refused K_MATMUL whose result the test drops, `backward()` of the valid loss fails with
`InvalidDescription` and "refused description".

## FinishRunsTheStepOnce

`has_run()` is false before `finish()` and true after it. A second `finish()` succeeds without
running again, and the session counts one step.

## ABlendIntoAnUnboundStateIsRefused

`blend_buffer()` into a state the session never bound fails with `InvalidDescription` and "not
bound".

## ABlendOfASourceOfOtherExtentsIsRefused

The buffer is bound with the extents `[6]`. A source of the [8, 6] input fails with `Incompatible`
and "source".

## ABlendGivesTheStepTrailingWork

One buffer blended from the row sums of the input, then `finish()`. The in-memory executor
journals one step with trailing work, although no update was applied.

## AStepWithNoUpdateAndNoBlendHasNoTrailingWork

A step that only traces the model and keeps its loss journals no trailing work. A step with no
blend and no copy passes the caller's recorder through, which here is none.

## TwoStatesSharingOneGradientValueRunWithoutARefusal

Two trainable states of extents `[8, 3]` meet in one K_ADD, so the K_ADD rule gives both one
gradient value. Backward and `finish()` succeed, and the executor journals trailing work although
no update was applied: the copy of the first state's gradient into the second's is the only work.
