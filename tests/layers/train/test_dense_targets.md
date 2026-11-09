# tests/layers/train/test_dense_targets.cpp


Suites: `Small_DenseTargets`, `Small_DenseTargetsRealType`, `Small_DenseTargetsPair`.

Tests `Train::fill_dense_targets` and `Train::fill_binary_targets`. The tensors are allocated on the engine of a `HostExecutionBackend`, so no suite needs a device.
A leaf holds 7 in every element before a fill.

## PairRow

`PairRow` holds the shape and the element type of a source and of a leaf, and the two texts the
refusal of that pairing holds.

## ValuesAreCopiedInOrderIntoEitherRealType

Six float32 values of shape [2, 3] fill a leaf of the same shape, once as float32 and once as
float64, and the leaf reads back the six values in order.

## AColumnLeafTakesOneLabelPerRow

`fill_binary_targets` writes the [4] labels 1, 0, 0, 1 into a [4, 1] leaf.

## AnotherCountOrTypeIsRefusedAndTheLeafKept

`fill_dense_targets` fails with `Incompatible` for six values into a leaf of four ("holds 6
values", "targets hold 4"), for int32 values ("int32", "float32") and for an int32 leaf
("targets are int32"), one row each. The leaf of each row still holds a 7 in every element.

## ALabelThatIsNotZeroOrOneIsRefused

`fill_binary_targets` over the labels 1, 0, 0.5, 1 fails with `Incompatible` and a message
containing "element 2", "0.5" and "0 or 1". The leaf still holds four 7s.
