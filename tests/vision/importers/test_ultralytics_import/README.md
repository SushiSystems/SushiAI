# tests/vision/importers/test_ultralytics_import

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_detections.cpp` | Tests the detections of imported Ultralytics checkpoints against Ultralytics' own on coco128 images. |
| `test_import.cpp` | Tests the Ultralytics importer on the YOLO26n checkpoint the reference exporter writes. |
