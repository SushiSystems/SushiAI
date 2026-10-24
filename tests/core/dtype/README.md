# tests/core/dtype

The tests of the module `core/dtype`, one file per header of `modules/core/dtype/include/`. The
suite prefix of a case gives its ctest label.

## test_dtype.cpp

The three `Small_DType` cases check `dtype_size` and `dtype_name` for every element type, `UInt32` included, that only
the three real floating types are real floats, and that `K_MAX_RANK` is the rank SushiBLAS stores.
