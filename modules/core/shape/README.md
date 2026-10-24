# shape

## Owns

The `Shape` value type, its hash (64-bit FNV-1a over the rank and each extent, through the `hash` module), and `shape_of`, which reads the shape of a SushiBLAS tensor.

## Depends on

`dtype`, for `K_MAX_RANK`, `failure`, for the checks of a constructor, and `hash`, for `Shape::hash`, all public. It links
`SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/shape/shape.hpp` | `Shape`, `std::hash<SushiAI::Shape>` |
| `SushiAI/shape/tensor_shape.hpp` | `shape_of` |
