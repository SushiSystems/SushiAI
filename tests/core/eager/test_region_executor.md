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

## AKeptStepRunsAgainWithoutBeingComposedOrRecorded

The second step traces the same graph. The executor composes nothing, lowers nothing and
allocates no pool byte. The input tensor was refilled between the steps, and the result follows
it, so the kept kernels read the storage they were recorded with.

## TwoRegionsRunInKeyOrder

A scope between the linear layer and the activation gives two regions. The result equals the
one-region result, so the region with the lower key ran first.

## ARegionRecordedUnderAnotherKeyKeepsItsProgram

The case hands the executor a `RegionOrder` with the action `Record`, the key raised by 2 and
the old key as `previous_key`, which is what the table emits when a scope appears. The old key
holds no region afterwards, the new one holds a program of the same node count, the region was
lowered twice and the graph composed twice.

## ARegionThatLeftIsDropped

Two regions, then one: the step that joins them drops the activation region by its old key.

## TheTrailingWorkRunsBehindEveryRegionOnEveryRun

The update subtracts the region's result from a zeroed tensor, so that tensor holds the negated
result only if the update ran after the region. The second step hands over the same recorder
under the same key: it is replayed and not recorded again.

## DiscardDropsEveryRegionAndAKeptRegionIsThenRefused

After `discard()` the executor holds no region. The harness's table was not told, so it plans a
`Keep`, and `execute` returns `InvalidDescription`. A session calls `RegionTable::forget` with
`discard()` for this reason.
