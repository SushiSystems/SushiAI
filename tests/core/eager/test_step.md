# tests/core/eager/test_step.cpp


Suites: `Medium_EagerStep`.

Tests what a `Step` accepts while it traces and when it runs.

A step belongs to a `Session`, which needs an engine on a CPU SYCL device, so the file is
medium. Every case builds the `Loop` of `eager_run_support.hpp` over the two-layer model and
opens one step on its session. `test_session.md` covers training through steps, reading values
and the refusals that involve the session's states.

## AStateIsDeclaredOncePerStep

Asking a step for the same state twice returns one leaf, and the graph binds that state to it.

## AnEmptyInputIsRefused

`input()` with a default tensor fails with `InvalidDescription`, naming the label "batch" and
"no element".

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

## FinishRunsTheStepOnce

`has_run()` is false before `finish()` and true after it. A second `finish()` succeeds without
running again, and the session counts one step.
