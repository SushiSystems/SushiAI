# tests/vision/importers/test_ultralytics_tensors.cpp

Suites: `Small_UltralyticsTensors`.

Each tensor holds 0, 1, 2 and so on in PyTorch's row-major order, so the expected record is
written out from the index arithmetic of the layout.

## ADenseWeightIsLaidOutOutputTapsInput

A [2, 3, 1, 2] weight of one group becomes a [2, 1, 2, 3] record whose values run output, tap,
input: 0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11.

## ADepthwiseWeightIsLaidOutTapsChannel

A [3, 1, 1, 2] weight of three groups becomes a [1, 2, 3] record whose values run tap, channel:
0, 2, 4, 1, 3, 5.

## ABiasKeepsItsLayout

A [3] tensor keeps its name, its shape and its values.

## AGroupedWeightThatIsNotDepthwiseIsRefused

A [4, 2, 1, 1] weight of two groups fails with `Unsupported`, naming the tensor and the groups.
