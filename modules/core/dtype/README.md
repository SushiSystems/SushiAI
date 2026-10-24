# dtype

## Owns

The element types of a tensor. `DType` and `Layout` alias the SushiBLAS element types and
layouts, since SushiAI computes on nothing SushiBLAS cannot. `K_MAX_RANK` is the rank SushiBLAS
stores, and `dtype_size`, `dtype_name` and `is_real_float` describe one element type. The eight
types are `Half`, `Float32`, `Float64`, `Complex32`, `Complex64`, `Int32`, `Int64` and `UInt32`;
`UInt32` is a count type that SushiBLAS holds and moves and no operation computes on, so a model
file and a checkpoint refuse it.

## Depends on

The module declares no module dependency. It links `SushiBLAS::sushiblas` as public, through
`EXTERNAL_PUBLIC`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/dtype/dtype.hpp` | `DType`, `Layout`, `K_MAX_RANK`, `dtype_size`, `dtype_name`, `is_real_float` |
