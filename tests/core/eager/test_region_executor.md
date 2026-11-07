# tests/core/eager/test_region_executor.cpp


Suites: `Medium_RegionExecutor`.

Tests that `RegionExecutor` builds, keeps, records again and drops regions as a step says.

This file needs an engine. `modules/core/eager/README.md`, under `region_executor.hpp`, gives the
two passes of `execute` and says why a kept region is not touched.

## Harness

`Harness` does a session's planning for a host-traced step: it allocates a filled tensor for
each leaf, plans the step on a `RegionTable`, allocates the created slots and compiles each
built region. `prepare` returns the `StepWork`, and `run` executes it. `expected` computes
`relu(x * weight + bias)` on the host from the leaves; the first leaf of `linear_then_relu` is
`x`, so it sits in `K_LEAF_STORAGE`.

## ABuiltStepWritesItsObservedValue

One region is built, lowered once and composed once, and the observed result matches the host.

## AKeptStepFollowsTheInputItsKernelsRead

The second step traces the same graph. The input tensor was refilled between the steps, and the
result follows it, so the kept kernels read the storage they were recorded with.

## AKeptStepIsNeitherComposedNorRecordedAgain

After the second step the executor has emitted and composed once each.

## AKeptStepAllocatesNoPoolByte

The pool bytes the executor holds are the same after the second step as after the first.

## TwoRegionsRunInKeyOrder

A scope between the linear layer and the activation gives two regions. The result equals the
one-region result, so the region with the lower key ran first.

## ARegionRecordedUnderAnotherKeyKeepsItsProgram

The case hands the executor a `RegionOrder` with the action `Record`, the key raised by 2 and
the old key as `previous_key`, which is what the table emits when a scope appears. The old key
holds no region afterwards and the new one holds a program of the same node count.

## ARegionRecordedUnderAnotherKeyIsLoweredAndComposedAgain

The same order lowers the region twice and composes the graph twice.

## ARegionRecordedUnderAnotherKeyKeepsItsResult

The observed value still matches the host after the region moved to its new key.

## ARegionThatLeftIsDropped

Two regions, then one: the step that joins them drops the activation region by its old key.

## TheTrailingRecorderIsToldAfterEveryRun

An `InMemoryStepRecorder` is handed over as trailing work on two steps under the same key. It is
told twice that a run finished, which shows the executor calls the hook after each run. Which
trailing work is recorded, and when, is `test_trailing_ledger.cpp`.

## TheTrailingWorkRunsBehindEveryRegion

The update subtracts the region's result from a zeroed tensor, so that tensor holds the negated
result only if the update ran after the region. `Graph::InMemoryDescentRecorder` records the axpy that
subtracts, and its tasks are real device tasks, so only they can show the order.

## ALateTrailingTaskReplaysAfterItsRecorderIsGone

`Graph::InMemoryLateIncrementRecorder` records one late-count task, which adds one to as many elements of a tensor as
a host count holds. The executor records it once, through a `TaskRecorder` that lives only inside
`record_trailing`, and runs it on two steps. Both runs add, so the hold the executor owns kept the
task's storages after the recorder was gone.

## DiscardDropsEveryRegion

After `discard()` the executor holds no region.

## AKeptRegionAfterDiscardIsRefused

The harness's table was not told, so it plans a `Keep`, and `execute` returns
`InvalidDescription`. A session calls `RegionTable::forget` with `discard()` for this reason.
