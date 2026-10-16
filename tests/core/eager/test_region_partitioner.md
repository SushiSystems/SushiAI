# tests/core/eager/test_region_partitioner.cpp


Suites: `Medium_RegionPartitioner`.

Tests how a `Session` drives the `RegionPartitioner` it is given.

`RegionPartitioner` is an interface, so the cases hand a session a partitioner of their own and
watch the calls. A session needs an engine on a CPU SYCL device, which makes the file medium.
`test_scope_partitioner.md` covers the keys `ScopePartitioner` chooses.

## RecordingPartitioner

`RecordingPartitioner` counts its calls, keeps the segments, the node count and the forward node
count of the last one, and returns what `ScopePartitioner` returns for the same arguments.

## RefusingPartitioner

`RefusingPartitioner` throws `std::invalid_argument` with "this graph has no cut" on every call.

## Rig

`Rig` owns a `RegionExecutor`, a session over the given partitioner and a `PlainDescent` update,
and binds the two-layer model of `eager_run_support.hpp` to the session. Its `step()` traces the
model with one scope per layer, differentiates it at the loss and applies the update.

## TheSessionAsksOncePerStep

Two steps make two calls, and the session counts two steps.

## TheSegmentsCoverTheForwardNodes

The segments of the call start at node 0, each starts where the one before ends, and the last
ends at the forward node count of the provenance. That count is below the graph's node count,
because the backward nodes come after the forward ones and no segment holds them.

## AThrownRefusalFailsTheStep

With `RefusingPartitioner` the step fails with `InvalidDescription` naming "this graph has no
cut", and the session counts no step.
