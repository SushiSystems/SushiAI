# tests/vision/importers/test_darknet_tensors.cpp


Suites: `Small_DarknetTensors`.

Tests `list_darknet_tensors` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_tensors.hpp`. The weights
are a plain convolution of 2 filters over 3 channels with a 2 by 2 kernel, whose weights are
0 to 23, and a normalised convolution of 2 filters over 1 channel with a 1 by 1 kernel. The
targets name `0.weight` and `0.bias` for the first, and `1.weight`, `2.bias`, `2.weight`,
`2.running_mean` and `2.running_var` for the second.

## ListsOneTensorPerNamedTarget

Seven tensors are listed. Each is `Float32` and its payload has the byte count its record
states.

## WeightsMoveToKernelLast

The tensor `0.weight` has the shape [2, 2, 2, 3]. Its 24 values open with 0, 4, 8, 1 and end
with 23.

## EveryVectorLandsUnderItsTarget

`2.weight` has the shape [2] and holds the scales 2, 3. `2.bias` holds 1.5, -1.5,
`2.running_mean` 0.25, 0.75 and `2.running_var` 4, 9. No tensor is named `0.running_mean`.

## AVectorOfTheWrongLengthIsRefused

With a third scale on the second convolution, the listing fails with `Incompatible`,
"layer 1", "'2.weight'" and "3 values".

## AnEntryNamedTwiceIsRefused

With the second target's bias named `0.bias`, the listing fails with `Incompatible`,
"layer 1", "'0.bias'" and "already fills".

## AShapeThatIsNotPositiveIsRefused

With 0 channels in the first shape, the listing fails with `InvalidDescription`, "layer 0"
and "channels 0".

## MismatchedCountsAreRefused

With one target for two convolutions, the listing fails with `Incompatible`,
"2 convolutions" and "the targets 1".
