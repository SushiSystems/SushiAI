# shape

## Owns

The `Shape` value type, its hash (64-bit FNV-1a over the rank and each extent, through the `hash` module), `shape_of`, which reads the shape of a SushiBLAS tensor, and `checked_shape_of`, which reads the shape of a tensor a caller supplied.

A constructor of `Shape` stops the process when the rank exceeds `K_MAX_RANK` or an extent is not positive, and so does `shape_of`. `Shape::make` and `checked_shape_of` return `InvalidDescription` for extents that came from a document, a file or a caller.

## Depends on

`dtype`, for `K_MAX_RANK`, `failure`, for the fatal checks of a constructor and the status of `Shape::make`, and `hash`, for `Shape::hash`, all public. It links
`SushiBLAS::sushiblas` and `SushiRuntime::core` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/shape/shape.hpp` | `Shape`, `Shape::make`, `std::hash<SushiAI::Shape>` |
| `SushiAI/shape/tensor_shape.hpp` | `shape_of`, `checked_shape_of` |
