# tests/core/data/test_input.cpp


Suites: `Small_DataInput`.

Tests `Data::InputDescription`, the per-sample shape and the element type of a batch's input.

This file needs no device.

## ADefaultDescriptionIsAFloat32Scalar

A default description states float32 and a sample shape of rank 0.

## ADescriptionHoldsWhatItIsGiven

A description built from [7, 2] and int32 holds that shape and that type.
