# tests/vision/importers/test_darknet_import

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_decode.cpp` | Tests that decode_head reproduces the boxes and scores Darknet dumped before its threshold, for the heads of six models. |
| `test_import.cpp` | Tests that each Darknet parity model imports with the rows, the task and the weights it declares. |
| `test_yolov3_forward.cpp` | Tests the forward pass of the imported YOLOv3 against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
| `test_yolov3_tiny_forward.cpp` | Tests the forward pass of the imported YOLOv3-tiny against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
| `test_yolov3_tiny_layers.cpp` | Tests every layer of an imported Darknet model against the Darknet dump of its output. |
| `test_yolov4_forward.cpp` | Tests the forward pass of the imported YOLOv4 against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
| `test_yolov4_tiny_forward.cpp` | Tests the forward pass of the imported YOLOv4-tiny against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
| `test_yolov7_forward.cpp` | Tests the forward pass of the imported YOLOv7 against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
| `test_yolov7_tiny_forward.cpp` | Tests the forward pass of the imported YOLOv7-tiny against the Darknet reference tensors: the heads, the [yolo] outputs, the detection rows and the Separate form. |
