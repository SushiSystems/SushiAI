# tests/core/eager/test_step_executor.cpp


Suites: `Small_StepWork`, `Medium_StepExecutor`.

Tests the work a `Session` hands its `StepExecutor` and how it treats a failed run.

`StepExecutor` is an interface, so the medium cases give a session an executor that runs
nothing and records each `StepWork`. A session needs an engine on a CPU SYCL device, which makes
those cases medium. `test_region_executor.md` covers the executor the library ships.

## SeenWork

`SeenWork` holds what one `execute()` call was given: the region count, the highest region key,
the trailing work, its key and the run index.

## RecordingExecutor

`RecordingExecutor` records a `SeenWork` per call and returns success, or a `DeviceFailed`
failure with "the device was lost" when its `fail` flag is set. `discard()` counts its calls.

## Rig

`Rig` owns the recording executor, a `ScopePartitioner`, a session over both and a
`PlainDescent` update, and binds the two-layer model of `eager_run_support.hpp`. Its `step()`
traces the model with one scope per layer, differentiates it, and runs it through `apply()` or
`finish()`.

## ADefaultWorkIsEmpty

A default `StepWork` holds no region, no dropped key, no storage, a null trailing recorder, the
trailing key `K_NO_REGION` and run index 0.

## EachStepIsOneWorkNumberedByTheStepsBefore

Two finished steps reach the executor as two works with run indices 0 and 1. The first holds
more than one region, and with no update applied its trailing recorder is null.

## TheUpdateIsTheTrailingWorkAboveEveryRegion

A step run through `apply()` hands the update itself as the trailing recorder, under a key above
the highest region key, so it runs behind every region.

## AFailedRunIsDiscardedAndCountsNoStep

When `execute()` fails, the step returns that failure, the session calls `discard()` once so the
next step builds every region, and the step count stays 0.
