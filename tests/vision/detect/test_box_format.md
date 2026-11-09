# tests/vision/detect/test_box_format.cpp


Suite: `Small_DetectBoxes`.

Tests `xyxy_to_xywh` and `xywh_to_xyxy` in
`modules/vision/detect/include/SushiAI/vision/detect/box_format.hpp`.

## ConvertsBetweenCornersAndCentreSize

The corners (1, 2, 5, 8) convert to the centre (3, 5) and the size 4 by 6, and the centre and
size convert back to the same corners. The values are exact in float32, so both results are
compared for equality. The case uses no file, device or thread, so its suite is small.
