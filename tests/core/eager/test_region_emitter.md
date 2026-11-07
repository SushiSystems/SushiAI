# tests/core/eager/test_region_emitter.cpp


Suites: `Medium_RegionEmitter`.

Tests that `RegionEmitter` records a region's program on the recorder it is handed.

This file needs an engine: its cases allocate tensors and run a recorded graph. The emitter is
handed the recorder of a static `SushiRuntime::Graph`, which shows that it depends on a
`Recorder` and on nothing of a dynamic graph.

## Recorded

`make_recorded` compiles `relu(x * weight + bias)` as one region, adopts the program's plan in a
`TensorPool` and binds a tensor filled by `fill_wave` to each import and export.
`expected_result` reads the three imports back and computes the result on the host with
`linear_relu`. The imports are in the order the region reads them: `x`, `weight`, `bias`.

## EmitRecordsTheProgramOnTheGraphItIsHanded

After `emit()` the graph runs once and the export holds the host result within `K_TOLERANCE`,
256 unit roundoffs of float, the distance a FLOAT32 sum of six products keeps from the same sum
in double.

## EmitCountsItsCalls

`emit_count()` is 0 before the first call and 2 after two `emit()` calls on two graphs.

## ASecondEmitRecordsTheSameOperations

Two `emit()` calls on two graphs record the same number of operations and the same number of
nodes. A region recorded again under a new key is this second call.

## AProgramWithoutDropoutHasNoPreludes

The program has no dropout, so `emit()` asks for no replay prelude.
