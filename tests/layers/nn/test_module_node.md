# tests/layers/nn/test_module_node.cpp


Suites: `Small_ModuleNode`.

Tests the incremental form of `ModuleNode` and the layer it hands back.

This file needs no device. `create()`, the port, `forward()` in its three layer forms, the state
walk and the refusals of `forward()` are pinned in `test_node.cpp`; the cases here cover
`forward_incremental()` and `layer()`.

## IncrementalProbe

`IncrementalProbe` traces a K_RELU in its ordinary form and a K_GELU in its incremental form, and
keeps the per-sample shape the incremental form was given. Two `static_assert` lines pin that it
has an incremental form and that `Linear` has none.

## node_of

`node_of` builds the `ModuleNode` of a layer over a per-sample shape and expects `create()` to
succeed.

## span_of

`span_of` views a vector of value ids as the inputs of one forward call.

## AnIncrementalTraceGivesTheLayerTheShapeOfCreate

The node is created over [5, 4] and traced incrementally over a [2, 1, 4] slice. The probe sees
[5, 4], the shape `create()` was given and not the slice's, the one result keeps the slice's
shape, and the last node of the graph is a K_GELU, so the incremental form ran.

## AnIncrementalTraceRejectsTwoInputs

Two inputs fail with InvalidDescription naming "input count is 1" and "given is 2", and nothing
is appended.

## ALayerWithoutAnIncrementalFormIsRefused

The node of a `Linear` falls back to the default of `Node` and returns Unsupported naming "no
incremental form", with `outputs` left empty.

## TheNodeHandsBackTheLayerItWraps

`layer()` returns the wrapped `Linear` with its fields as given, and the const and non-const
overloads return the same object.
