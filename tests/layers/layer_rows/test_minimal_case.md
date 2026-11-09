# tests/layers/layer_rows/test_minimal_case.cpp


Suites: `Small_LayerRowsMinimalCase`, `Small_LayerRowsMinimalCaseBuild`.

Tests `MinimalCase` and `make_minimal` in
`modules/layers/layer_rows/include/SushiAI/layer_rows/minimal_case.hpp`. The rows come from
`make_layer_registry`. No case needs a device. `span_of` is a one-line view and has no test of its
own; every case below builds through it.

## BuildRow

Names a minimal case that builds, with the case itself.

## AMinimalCaseBuildsItsStatedOutput

Run once for each row of `Small_LayerRowsMinimalCase`: a "linear" layer row of 5 units over [6] and
an "add" node row over two [2, 2, 3] inputs each build a node with as many inputs as the case lists
and one port of the stated output shape.

## ARefusedRowIsNamedAsLayerZero

A "conv2d" row fed the feature vector [784] fails with `InvalidDescription` naming
"layer 0 'conv2d'" and "[784]".
