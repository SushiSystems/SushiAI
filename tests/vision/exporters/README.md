# tests/vision/exporters

The test sources of `modules/vision/exporters` kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_caffe_export.cpp` | Tests the Caffe exporter: the LeNet fixed point, one model per row and the refusals. |
| `test_darknet_export.cpp` | Tests the Darknet exporter: a written .cfg's fixed point, one model per row, the anchors and the refusals. |
| `test_onnx_export.cpp` | Tests the ONNX exporter: the fixed point, one model per row and the refusals, and that the Caffe fixture and its exported copy give the same output. |
