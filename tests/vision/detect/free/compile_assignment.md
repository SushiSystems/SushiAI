# tests/vision/detect/free/compile_assignment.cpp


A compile-only unit: it holds no test and the runner never lists it.

Checks at build time that `Assignment` in
`modules/vision/detect/include/SushiAI/vision/detect/free/assignment.hpp` is move constructible.
The header declares a struct and no function. The case of the earlier `test_assignment.cpp` that
stated the default extents and empty arrays is deleted: a default value is not a promise. The
assigner that fills an `Assignment` is tested in `test_assigner.cpp`, and the leaves read from it
in `test_target_leaves.cpp`.

The file needs no device and runs nothing; the program compiles it.
