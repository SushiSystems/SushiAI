# tests/vision/detect/compile_detection_row.cpp


A compile-only unit: it holds no test and the runner never lists it.

Tests the column layout of a detection row in
`modules/vision/detect/include/SushiAI/vision/detect/detection_row.hpp`. The header declares
constants and no function, so no code of the unit runs. The two cases of the earlier
`test_detection_row.cpp` compared constants with literals and are deleted. The column layout is
read by `test_detections_nms.cpp` and `test_detections_topk.cpp`, and the default suppression
values are checked in `test_detections_nms.cpp`.

The file asserts at namespace scope that the class column follows the score column, that the class
column is the last of a row, and that the four corner columns come before the score. It needs no
device and runs nothing; the program compiles it.
