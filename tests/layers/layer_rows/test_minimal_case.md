# tests/layers/layer_rows/test_minimal_case.cpp


Suites: `Small_LayerRowsMinimalCase`.

Tests `MinimalCase`, `span_of` and `make_minimal` in
`modules/layers/layer_rows/include/SushiAI/layer_rows/minimal_case.hpp`. The rows come from
`make_layer_registry`. No case needs a device.

## SpanOfViewsEveryShape

`span_of` over two shapes has two elements, starts at the vector's data and holds [2, 3] second.

## AMinimalCaseBuildsItsStatedOutput

A "linear" layer row of 5 units over [6] and an "add" node row over two [6] inputs each build a
node with as many inputs as the case lists and one port of the stated output shape.

## ARefusedRowIsNamedAsLayerZero

A "conv2d" row fed the feature vector [784] fails with `InvalidDescription` naming
"layer 0 'conv2d'" and "[784]".
