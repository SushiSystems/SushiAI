# tests/layers/train/test_input_match.cpp


Suites: `Small_InputMatch`.

Tests `Train::describe_input_leaf` and `Train::match_input`. The dataset is a local `Published`
class named "published" that publishes the input it is built with, no role and no batch.

## ALeafIsDescribedWithoutItsBatchAxis

A [4, 16] int32 leaf is described as [16] int32, a [2, 8, 8, 3] float32 leaf as [8, 8, 3]
float32 and a [4] float64 leaf as an empty shape and float64.

## ALeafWithNoBatchAxisIsRefused

A leaf of rank 0 fails with `InvalidDescription` and "no batch extent".

## ADatasetThatPublishesTheInputIsAccepted

`match_input` is ok for [16] int32 against a dataset that publishes [16] int32, and for [16]
float32 against one that publishes [16] float32.

## AnotherElementTypeIsRefusedWithBothTypes

A float32 input against the int32 dataset fails with `Incompatible` and a message containing
`'published'`, "as int32 [16] per sample" and "compiled for float32 [16] per sample". An int32
input against the float32 dataset fails with the two types the other way round.

## AnotherShapeIsRefusedWithBothShapes

A [32] int32 input against the [16] int32 dataset fails with `Incompatible`, "as int32 [16] per
sample" and "compiled for int32 [32] per sample".
