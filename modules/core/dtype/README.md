# dtype

## Owns

The element types of a tensor. `DType` and `Layout` alias the SushiBLAS element types and
layouts, since SushiAI computes on nothing SushiBLAS cannot. `K_MAX_RANK` is the rank SushiBLAS
stores, and `dtype_size`, `dtype_name` and `is_real_float` describe one element type.

## Depends on

The module declares no module dependency. It links `SushiBLAS::sushiblas` as public, through
`EXTERNAL_PUBLIC`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/dtype/dtype.hpp` | `DType`, `Layout`, `K_MAX_RANK`, `dtype_size`, `dtype_name`, `is_real_float` |
