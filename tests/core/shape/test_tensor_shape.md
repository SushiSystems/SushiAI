# tests/core/shape/test_tensor_shape.cpp


Suites: `Small_TensorShape`.

Tests `shape_of` and `checked_shape_of`, which read the extents of a `SushiBLAS::Tensor` as a
`Shape`. The rules of `Shape` itself are in `test_shape.cpp` in this folder. Each case fills the
`shape` and `rank` fields of a default tensor, so no storage and no device is involved.

## description

`description` returns a default tensor of rank 3 whose extents are the three it is given.

## ExtentsComeBackOutermostFirstUpToTheRank

A rank-3 tensor of extents 2, 3 and 5, with a 7 written past its rank, reads as `Shape{2, 3, 5}`.
The extent past the rank is not read.

## ARankZeroTensorIsTheScalarShape

A default tensor, of rank 0, reads as the default `Shape`.

## AnExtentThatIsNotPositiveIsFatal

Extents 2, 0 and 5 stop the process with a report that contains "positive" and "0".

## CheckedShapeOfReadsALegalTensor

A rank-3 tensor of extents 2, 3 and 5 reads as `Shape{2, 3, 5}` through `checked_shape_of`.

## CheckedShapeOfRefusesAnExtentThatIsNotPositive

Extents 2, 0 and 5 return `InvalidDescription` naming "Shape extents must be positive" and "got 0".
