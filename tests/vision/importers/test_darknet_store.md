# tests/vision/importers/test_darknet_store.cpp


Suites: `Medium_DarknetStore`, in `sushiai_vision_tests`.

The file needs an engine and no reference data. The state is built with `ModelState::from_entries`
for a model of two convolutions: layer 0 with 2 filters, 3 channels and a kernel side of 2, and
layer 1 with 2 filters, 1 channel, a kernel side of 1 and batch norm in its own entries. The weights
are written by hand, and `store_darknet_weights` moves them into the state.

## WeightsMoveToKernelLastAndEveryVectorLands

The weight of layer 0 lands in `0.conv.weight` kernel last: the expected values are built by walking
the destination order and reading the source as `[out, in, kh, kw]`. The biases, the scales, the
rolling mean and the rolling variance land in their entries unchanged.

## AFoldedConvolutionLeavesItsNormalisationEntriesAlone

Layer 1 is folded and its target names only a weight and a bias. Those two entries receive the
folded values, and the four batch-norm entries keep the sentinel -7 written before the store.

## AMissingEntryIsRefusedAndNamesTheLayer

A target that names `1.bn.missing` is refused as `Incompatible`. The message contains "layer 1" and
the entry name.

## AShapeMismatchIsRefusedAndNothingIsWritten

`1.bn.running_var` is given the shape `[3]`. The store is refused as `Incompatible`, the message
names the entry and "shape", and `0.conv.weight` still holds the sentinel in all 24 elements.

## AVectorOfTheWrongLengthIsRefused

Layer 1 is folded but its target still names the batch-norm entries. The store is refused as
`Incompatible`, and the message contains "layer 1", "0 values" and '1.bn.weight'.

## AnEntryNamedTwiceIsRefused

Two targets name `0.conv.bias`. The store is refused as `Incompatible`, and the message contains
"layer 1", the entry name and "already fills".

## MismatchedCountsAreRefused

Two convolutions are stored against a list of one shape. The store is refused as `Incompatible` with
"the weights hold 2 convolutions".
