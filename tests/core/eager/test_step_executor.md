# tests/core/eager/test_step_executor.cpp


Suites: `Medium_StepExecutor`.

Tests the work a `Session` hands its `StepExecutor` and how it treats a failed run.

`StepExecutor` is an interface, so the cases give a session an `InMemoryStepExecutor`, which runs
nothing and journals each `StepWork` as a `Seen`. A session needs an engine on a CPU SYCL
device, which makes the cases medium. `test_region_executor.md` covers the executor the library
ships.

## Rig

`Rig` owns the in-memory executor, a `ScopePartitioner`, a session over both and a
`Graph::InMemoryDescentRecorder` update, and binds the two-layer model of `eager_run_support.hpp`. Its `step()`
traces the model with one scope per layer, differentiates it, and runs it through `apply()` or
`finish()`.

## EachStepIsOneWorkNumberedByTheStepsBefore

Two finished steps reach the executor as two works with run indices 0 and 1. The first holds
more than one region, and with no update applied it holds no trailing work.

## TheUpdateIsTheTrailingWorkAboveEveryRegion

A step run through `apply()` hands the update over as trailing work, under a key above
the highest region key, so it runs behind every region.

## AFailedRunIsDiscardedAndCountsNoStep

When `execute()` fails, the step returns that failure, the session calls `discard()` once so the
next step builds every region, and the step count stays 0.
