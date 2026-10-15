# tests/layers/train/test_dense_targets.cpp


Suites: `Unit_DenseTargets`.

Tests `Train::fill_dense_targets` and `Train::fill_binary_targets`. The fixture needs an engine.
A leaf holds 7 in every element before a fill.

## ValuesAreCopiedInOrderIntoEitherRealType

Six float32 values of shape [2, 3] fill a float32 leaf and a float64 leaf of the same shape,
and each leaf reads back the six values in order.

## AColumnLeafTakesOneLabelPerRow

`fill_binary_targets` writes the [4] labels 1, 0, 0, 1 into a [4, 1] leaf.

## AnotherCountOrTypeIsRefusedAndTheLeafKept

`fill_dense_targets` fails with `Incompatible` for six values into a leaf of four ("holds 6
values", "targets hold 4"), for int32 values ("int32", "float32") and for an int32 leaf
("targets are int32"). The float32 leaf still holds four 7s.

## ALabelThatIsNotZeroOrOneIsRefused

`fill_binary_targets` over the labels 1, 0, 0.5, 1 fails with `Incompatible` and a message
containing "element 2", "0.5" and "0 or 1". The leaf still holds four 7s.
