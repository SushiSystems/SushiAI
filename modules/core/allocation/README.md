# allocation

## Owns

Tensor and arena allocation on a `SushiBLAS::Engine`, and views into an arena.

## Depends on

`dtype`, `failure` and `shape`, public; `seam`, private, to turn an allocation failure into a
`Status`. It links `SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/allocation/allocation.hpp` | `allocate_tensor`, `allocate_like`, `allocate_arena`, `view_tensor` |
