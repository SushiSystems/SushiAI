# tests/vision/detect/compile_prediction_shape.cpp


A compile-only unit: it holds no test and the runner never lists it.

Tests the shape of `PredictionShape` in
`modules/vision/detect/include/SushiAI/vision/detect/prediction_shape.hpp`. The header declares
one struct of three extents and no function, so no code of the unit runs. The default-value case
and the field-order case of the earlier `test_prediction_shape.cpp` are deleted: they pinned the
defaults of a struct and the order of its fields, which `test_detections_nms.cpp` and
`test_detections_topk.cpp` exercise through positional `PredictionShape` values.

The file asserts at namespace scope that the struct is an aggregate and that its batch, anchor
and class extents are `std::int64_t`. It needs no device and runs nothing; the program compiles
it.
