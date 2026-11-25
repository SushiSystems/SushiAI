# allocation

## Owns

Tensor and arena allocation on a `SushiBLAS::Engine`, views into an arena, and the host copies of
a tensor's elements. `allocate_tensor` takes a `TensorAccess`: `Kernels`, the default, allocates
device memory; `Host` allocates pinned host memory for a tensor the host writes every step.
`residency_of` in `allocation.cpp` is the only place a residency is chosen.

## Depends on

`dtype`, `failure` and `shape`, public; `seam`, private, to turn an allocation failure into a
`Status`. It links `SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/allocation/allocation.hpp` | `TensorAccess`, `check_tensor_extents`, `allocate_tensor`, `allocate_like`, `allocate_arena`, `view_tensor` |
| `SushiAI/allocation/host_copy.hpp` | `copy_to_host`, `copy_from_host`, `read_elements`, `write_elements`, `read_reals`, `write_reals` |
