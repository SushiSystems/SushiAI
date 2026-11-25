# tests/core/allocation/test_host_copy.cpp


Suite: `Medium_HostCopy`.

Tests the copies of `host_copy.hpp` between host memory and a tensor allocated with the default
`TensorAccess::Kernels`, which the host cannot address. Every case allocates on the engine
`EngineTest` gives it, on the CPU SYCL device.

## WrittenElementsReadBackUnchanged

The int32 values 3, -1, 7 and 42, written with `write_elements`, read back equal with
`read_elements`.

## ARangeCopyWritesOnlyItsRange

Two floats written from element 1 of a zeroed float32 tensor of four elements read back as
0, 1.5, 2.5, 0.

## RealsRoundTripThroughBothFloatTypes

The doubles 1, -2 and 4, written with `write_reals` into a float32 and into a float64 tensor,
read back equal with `read_reals`.

## AnotherElementWidthIsRefused

Reading a float32 tensor into a span of doubles is refused with `StatusCode::InvalidDescription`.

## ARangePastTheLastElementIsRefused

Writing two floats from element 1 of a tensor of two elements is refused with
`StatusCode::InvalidDescription`.
