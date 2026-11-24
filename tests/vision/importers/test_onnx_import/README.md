# tests/vision/importers/test_onnx_import

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_import.cpp` | Tests the ONNX importer: its refusals, its rows and weights, and the forward pass of the fixture against the output onnx.reference computed. |
| `test_yolox_detections.cpp` | Tests the detections of an imported YOLOX ONNX file against the reference runtime's own on letterboxed images. |
