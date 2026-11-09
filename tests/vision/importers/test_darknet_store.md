# tests/vision/importers/test_darknet_store.cpp


Suites: `Small_DarknetStore`.

The file needs the host engine and no reference data; its fixture holds a `HostExecutionBackend`. The state is built with `ModelState::from_entries`
for a model of two convolutions: layer 0 with 2 filters, 3 channels and a kernel side of 2, and
layer 1 with 2 filters, 1 channel, a kernel side of 1 and batch norm in its own entries. The weights
are written by hand, and `store_darknet_weights` moves them into the state.

## WeightsMoveToKernelLast

The weight of layer 0 lands in `0.conv.weight` kernel last: the source reads as `[out, in, kh, kw]`
with the values 0 to 23, and the 24 destination values are written out in `[out, kh, kw, in]` order.

## EveryOtherVectorLandsUnderItsTarget

The biases of layer 0, the weight of layer 1, and the scales, biases, rolling mean and rolling
variance of layer 1 land in their entries unchanged.

## AFoldedConvolutionLeavesItsNormalisationEntriesAlone

Layer 1 is folded and its target names only a weight and a bias. Those two entries receive the
folded values, and the four batch-norm entries keep the sentinel -7 written before the store.

## AMissingEntryIsRefusedAndNamesTheLayer

A target that names `1.bn.missing` is refused as `Incompatible`. The message contains "layer 1" and
the entry name.

## AMissingEntryLeavesEveryEntryUntouched

All eight entries hold the sentinel -7. A target that names `1.bn.missing` is refused as
`Incompatible`, and every element of every entry still holds the sentinel.

## AShapeMismatchIsRefusedAndNothingIsWritten

`1.bn.running_var` is given the shape `[3]`. The store is refused as `Incompatible`, the message
names the entry and "shape", and `0.conv.weight` still holds the sentinel in all 24 elements.

## AVectorOfTheWrongLengthIsRefused

Layer 1 is folded but its target still names the batch-norm entries. The store is refused as
`Incompatible`, and the message contains "layer 1", "0 values" and '1.bn.weight'.

## AnEntryNamedTwiceIsRefused

Two targets name `0.conv.bias`. The store is refused as `Incompatible`, and the message contains
"layer 1", the entry name and "already fills".

## ARepeatedNameLeavesEveryEntryUntouched

All eight entries hold the sentinel -7. Two targets name `0.conv.bias`, the store is refused as
`Incompatible`, and every element of every entry still holds the sentinel.

## MismatchedCountsAreRefused

Two convolutions are stored against a list of one shape. The store is refused as `Incompatible` with
"the weights hold 2 convolutions".
