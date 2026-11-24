# tests/vision/importers/test_darknet_training

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_recipe.cpp` | Tests the solver and head settings a .cfg yields, the defaults and the refusals. |
| `test_yolov3.cpp` | Tests the head deltas of YOLOv3 and one training step of it under batch statistics against what Darknet dumped of both. |
| `test_yolov3_tiny.cpp` | Tests the head deltas of YOLOv3-tiny and one training step of it under batch statistics against what Darknet dumped of both. |
| `test_yolov4.cpp` | Tests the head deltas of YOLOv4 and one training step of it under batch statistics against what Darknet dumped of both. |
| `test_yolov4_tiny.cpp` | Tests the head deltas of YOLOv4-tiny and one training step of it under batch statistics against what Darknet dumped of both. |
| `test_yolov7.cpp` | Tests the head deltas of YOLOv7 and one training step of it under batch statistics against what Darknet dumped of both. |
| `test_yolov7_tiny.cpp` | Tests the head deltas of YOLOv7-tiny and one training step of it under batch statistics against what Darknet dumped of both. |
