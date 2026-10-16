# tests/layers/nn/test_layer_node.cpp


Suites: `Small_LayerNode`.

Tests what `LayerNode` adds to the run-time layer it owns.

This file needs no device. The port shape, the input count, the parameter walk, the refusal of
a wrong input count and the fatal null layer are pinned in `test_node.cpp`; the cases here cover
the rest of the node's surface.

## span_of

`span_of` views a vector of value ids as the inputs of one forward call.

## make_linear

`make_linear` returns a 6-to-8 `Linear` wrapped in a `LayerAdapter`, the run-time layer every
case hands to a node.

## TheNodeHandsBackTheLayerItOwns

The address of the layer before it is moved into the node equals the address `layer()` returns,
through the const and the non-const overload. The one port is named `K_SINGLE_PORT_NAME`.

## ForwardKeepsWhatTheOutputsAlreadyHeld

`outputs` holds `x` before the call. After a forward over `x` it holds two values: `x`, unchanged
in front, then the layer's [2, 8] result. A node that cleared `outputs` would break a model graph
that collects the results of several nodes in one vector.

## TheNodeRefusesIncrementalTracing

A run-time layer has no incremental form, so `forward_incremental` returns Unsupported naming
"no incremental form" and leaves `outputs` empty.
