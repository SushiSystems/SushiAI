# tests/layers/train/test_feed.cpp


Suites: `Small_Feed`.

Tests `Train::check_feed`, which compares a batch tensor with the leaf a step copies it into.

This file needs no device. `check_feed` reads the rank, the extents and the element type of a
tensor, so every case builds tensor headers that own no storage.

## header_of

`header_of` returns a `SushiBLAS::Tensor` with the given extents and element type and no
storage.

## MatchingTensorsAreAccepted

Two [4, 6] float32 headers pass.

## AnotherShapeIsIncompatibleAndBothShapesAreNamed

A [4, 5] source for a [4, 6] leaf fails with Incompatible naming `inputs` and both shapes. A
[24] source of the same element count fails too, so the check compares shapes and not counts.

## AnotherElementTypeIsIncompatibleAndBothTypesAreNamed

An int32 source for a float32 leaf of the same shape fails with Incompatible naming `targets`,
int32 and float32.

## AnExtentNoShapeHoldsIsAnInvalidDescription

A source with an extent of 0 cannot be held by a `Shape`. The check returns InvalidDescription
naming "positive" instead of letting the error escape.
