# tests/layers/train/test_host_values.cpp


Suites: `Medium_HostValues`.

Tests `Train::require_host_reals` and `Train::host_real` on the two element types they read.

The tensors are allocated on the CPU SYCL device through `EngineTest`, so the file is medium.
The refusal of an int32 tensor and of another element count is pinned in
`test_binary_confusion.cpp`; the cases here pin the reads.

## Medium_HostValues

The fixture derives from `EngineTest`. Its `tensor_of` allocates a tensor of a shape and an
element type and writes the given values into it.

## AFloat32ElementIsReadAsADouble

A float32 tensor of three values is accepted for three elements, and elements 1 and 2 read back
as -1.25 and 3.0 exactly.

## AFloat64ElementIsReadUnrounded

0.1 in a float64 tensor reads back as the double 0.1. The same value in a float32 tensor reads
back as the float nearest 0.1 widened to a double, which differs from 0.1. A read that went
through float for both types would fail the first expectation.
