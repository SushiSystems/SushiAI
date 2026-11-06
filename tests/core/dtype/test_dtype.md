# tests/core/dtype/test_dtype.cpp


Suites: `Small_DType`, `Small_DTypeRow`.

Tests the helpers of `modules/core/dtype/include/SushiAI/dtype/dtype.hpp`: `dtype_size`,
`dtype_name`, `is_real_float` and `K_MAX_RANK`.

## HasItsWidthInBytes

For each element type, `UInt32` included, `dtype_size` returns the width in bytes its row gives.

## HasItsDiagnosticName

For each element type, `dtype_name` returns the name its row gives, such as `float32`.

## OnlyTheRealFloatingTypesAreRealFloats

`is_real_float` holds for `Half`, `Float32` and `Float64` and for no other type.

## TheMaximumRankIsTheRankSushiBLASStores

`K_MAX_RANK` equals the tensor rank SushiBLAS stores, `K_MAX_TENSOR_RANK`.
