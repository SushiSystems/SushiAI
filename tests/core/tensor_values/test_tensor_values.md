# tests/core/tensor_values/test_tensor_values.cpp


Suites: `Medium_TensorValues`, `Small_TensorValues`.

Tests `write_tensor`, `read_tensor`, `read_tensor_bytes` and `first_difference`.

The `Medium_TensorValues` fixture needs an engine to allocate tensors. The writes and reads touch
shared memory from the host, so no case records work on the device.

## AFloat64TensorRoundTripsExactly

Four doubles written to a float64 [4] tensor read back equal, 1e-300 included.

## AFloat32TensorHoldsEachValueRoundedToFloat

0.1, -2.5 and one third written to a float32 [3] tensor read back as each value rounded to
float and widened to double, compared exactly.

## TheBytesOfATensorAreItsElementStorage

`read_tensor_bytes` of a float32 [2, 3] tensor holds 24 bytes, equal to the bytes of its
element storage.

## FirstDifferenceFindsTheFirstUnequalByte

`first_difference` returns 2 for buffers that differ at byte 2, the length for two equal
buffers, and the shorter length when one buffer is a prefix of the other.
