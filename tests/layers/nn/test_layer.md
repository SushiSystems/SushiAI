# tests/layers/nn/test_layer.cpp


Suites: `Small_Layer`.

Tests the run-time `Layer` interface and the `LayerAdapter` that presents a compile-time module
through it.

This file needs no device; every case traces into a host-side graph. The shapes an adapter is
built from are pinned in `test_layer_stack.cpp`, and the compile-time refusal of a module that
lists a buffer is pinned in `test_node.cpp`, so neither is repeated here.

## FixedShapeLayer

`FixedShapeLayer` implements `Layer` by hand: it returns its input, owns no parameter and reports
the per-sample shape [2, 3, 4]. It reaches `output_features()` without an adapter in between.

## parameter_names

`parameter_names` walks a layer's parameters under a prefix and returns their names in walk
order.

## OutputFeaturesCountsTheShapeOfAnyImplementation

`output_features()` of `FixedShapeLayer`, read through a `Layer` reference, is 24, the element
count of [2, 3, 4]. The count comes from the base class, so a hand-written layer gets it too.

## AnAdapterTracesWhatItsModuleTraces

A `LayerAdapter<Linear>` traced through `Layer::forward` and a bare `Linear` traced on its own,
both 6 to 8 over a [2, 6] input, leave graphs with the same node count and a result of shape
[2, 8]. The adapter adds no node of its own.

## AnAdapterWalksItsModulesParametersUnderThePrefix

After one forward, the adapter's walk under the prefix `block` yields `block.weight` and then
`block.bias`, the parameters `Linear` declared on that trace.

## TheAdapterHandsBackTheModuleItWraps

`layer()` returns the wrapped `Linear` with its fields as given, 6 in and 8 out, and the const
and non-const overloads return the same object.
