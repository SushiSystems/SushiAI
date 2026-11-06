# tests/core/tensor_values/test_tensor_values.cpp


Suites: `Medium_TensorValues`, `Small_TensorValues`.

Tests `round_to_float32`, `write_tensor`, `read_tensor`, `read_tensor_bytes` and
`first_difference`.

The `Medium_TensorValues` fixture needs an engine to allocate tensors. The writes and reads touch
shared memory from the host, so no case records work on the device.

## AFloat64TensorRoundTripsExactly

Four doubles written to a float64 [4] tensor read back equal, 1e-300 included.

## RoundToFloat32GivesTheNearestFloatOfEachValue

0.1, -2.5 and one third rounded to float32 equal the hexadecimal literals `0x1.99999ap-4`,
`-2.5` and `0x1.555556p-2`. No engine is involved.

## AFloat32TensorHoldsEachValueRoundedToFloat

0.1, -2.5 and one third written to a float32 [3] tensor read back as the same three literals
widened to double, compared exactly.

## TheBytesOfATensorAreItsElementStorage

`read_tensor_bytes` of a float32 [2, 3] tensor written with 1 to 6 equals the 24 bytes of the
floats 1 to 6 laid out in a host array.

## FirstDifferenceFindsTheFirstUnequalByte

`first_difference` returns 2 for buffers that differ at byte 2, the length for two equal
buffers, and the shorter length when one buffer is a prefix of the other.
