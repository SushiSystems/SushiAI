# tests/layers/train/test_input_match.cpp


Suites: `Small_InputMatch`, `Small_InputMatchLeaf`, `Small_InputMatchAccepted`,
`Small_InputMatchRefusal`.

Tests `Train::describe_input_leaf` and `Train::match_input`. The dataset is a
`Data::InMemoryDataset` of no batch and no role that publishes the input it is built with; its
name is "in-memory".

## LeafRow

`LeafRow` holds the shape and the element type of a leaf and the shape of one sample that its
description holds.

## MatchRow

`MatchRow` holds the input a step reads, the input a dataset publishes and two texts of the
refusal.

## ALeafIsDescribedWithoutItsBatchAxis

A leaf is described by its extents past the batch axis and its element type. The rows are a
[4, 16] int32 leaf, described as [16] int32; a [2, 8, 8, 3] float32 leaf, as [8, 8, 3] float32;
and a [4] float64 leaf, as an empty shape and float64.

## ALeafWithNoBatchAxisIsRefused

A leaf of rank 0 fails with `InvalidDescription` and "no batch extent".

## ADatasetThatPublishesTheInputIsAccepted

`match_input` is ok for [16] int32 against a dataset that publishes [16] int32, and for [16]
float32 against one that publishes [16] float32.

## AnotherInputIsRefusedWithBothDescriptions

`match_input` fails with `Incompatible` and a message containing `'in-memory'` and both
descriptions. The rows are a float32 input against an int32 dataset ("as int32 [16] per sample",
"compiled for float32 [16] per sample"), an int32 input against a float32 dataset (the two types
the other way round), and a [32] int32 input against a [16] int32 dataset ("as int32 [16] per
sample", "compiled for int32 [32] per sample").
