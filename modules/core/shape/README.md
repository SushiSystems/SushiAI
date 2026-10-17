# shape

## Owns

The `Shape` value type, its hash, and `shape_of`, which reads the shape of a SushiBLAS tensor.

## Depends on

`dtype`, for `K_MAX_RANK`, and `failure`, for the checks of a constructor, both public. It links
`SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/shape/shape.hpp` | `Shape`, `std::hash<SushiAI::Shape>` |
| `SushiAI/shape/tensor_shape.hpp` | `shape_of` |
