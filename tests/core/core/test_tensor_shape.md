# tests/core/core/test_tensor_shape.cpp


Suites: `Small_TensorShape`.

Tests `shape_of`, which reads the extents of a `SushiBLAS::Tensor` as a `Shape`. The rules of
`Shape` itself are in `test_shape.cpp` in this folder. Each case fills the `shape` and `rank`
fields of a default tensor, so no storage and no device is involved.

## description

`description` returns a default tensor of rank 3 whose extents are the three it is given.

## ExtentsComeBackOutermostFirstUpToTheRank

A rank-3 tensor of extents 2, 3 and 5, with a 7 written past its rank, reads as `Shape{2, 3, 5}`.
The extent past the rank is not read.

## ARankZeroTensorIsTheScalarShape

A default tensor, of rank 0, reads as the default `Shape`.

## AnExtentThatIsNotPositiveThrows

Extents 2, 0 and 5 throw an `Error` whose message contains "positive" and "0".
