# tests/vision/umbrella/compile_vision.cpp


A compile-only unit: it holds no test and the runner never lists it.

Tests that `modules/vision/umbrella/include/SushiAI/vision.hpp`, included alone, names the anchor
size of detect, the confusion matrix of segment and the image of imaging. The file includes no
other SushiAI header. The header only includes others, so the names are what it pins.

The file asserts at namespace scope that `Detect::Anchor::AnchorSize`, `Segment::ConfusionMatrix`
and `Imaging::Image` are classes. It needs no device and runs nothing; the program compiles it.
The values the earlier test compared moved to the modules that own them: the self-IoU to
`tests/vision/detect/anchor/test_anchor_layout.cpp` and the ignore index to
`tests/vision/segment/test_ignore_index.cpp`; the two constants it also compared are not promises
of the umbrella.
