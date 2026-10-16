# tests/vision/umbrella/test_vision.cpp


Suites: `Small_VisionUmbrella`.

Tests that `modules/vision/umbrella/include/SushiAI/vision.hpp` alone makes the units of several
vision modules reachable.

This file needs no device and includes nothing from the vision domain but the umbrella header.
A header missing from the umbrella, or one that does not compile beside the others, fails the
build of this file.

## K_IOU_TOLERANCE

`K_IOU_TOLERANCE` is 1e-6, the error of an IoU whose terms float32 rounds once each.

## OneIncludeReachesDetectSegmentAndImaging

`shape_iou` of a size with itself is 1, `K_DETECTION_VALUES` is 6, `K_DEFAULT_IGNORE_INDEX` is
255 and `K_MOSAIC_PARTNERS` is 3, each reached through the umbrella.
