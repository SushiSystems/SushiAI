# tests/applications/programs/source/test_host_readback.cpp

Suites: `Medium_HostReadback`.

Tests `HostReadback` of `applications/programs/source/host_readback.hpp`, the host copies of the
outputs a report decodes. The tensor is a float32 [4] allocated with `TensorAccess::Kernels`, so
only kernels address its memory.

## ARefreshReturnsTheValuesWrittenToADeviceTensor

The values 1.5, -2, 0 and 42.25 are written into the tensor through `write_elements`. After the
tensor is bound and `refresh()` succeeds, the view that `bind()` returned holds the same four
values.
