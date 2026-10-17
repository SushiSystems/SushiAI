# tensor_values

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The host reads and writes of the elements of a tensor. `write_tensor` and `read_tensor` go
through the shared-USM pointer and not through a queue copy: every tensor a pool hands out is
`Residency::Shared`, and a test waits before it reads. Reading this way also keeps a test from
queueing work into the graph it is inspecting.

`read_tensor_bytes` exists because `read_tensor` widens everything to double, which cannot tell
-0.0 from 0.0 or two NaN payloads apart. A determinism claim is about bits, so the check that
guards it reads bits. `first_difference` gives the first byte where two such reads differ.

## Depends on

`dtype`, for `DType` and `dtype_size`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/tensor_values/tensor_values.hpp` | `write_tensor`, `read_tensor`, `read_tensor_bytes`, `first_difference` |
