# tests/vision/importers

The tests of the module `vision/importers`, one file per header of `modules/vision/importers/include/`. The suite prefix of a case gives its ctest label.

## Test files

Each test file has a page beside it that says what its cases pin.

| Test file | Page |
| --- | --- |
| `test_caffe_import.cpp` | [`test_caffe_import.md`](test_caffe_import.md) |
| `test_darknet_config.cpp` | [`test_darknet_config.md`](test_darknet_config.md) |
| `test_darknet_fold.cpp` | [`test_darknet_fold.md`](test_darknet_fold.md) |
| `test_darknet_import.cpp` | [`test_darknet_import.md`](test_darknet_import.md) |
| `test_darknet_model.cpp` | [`test_darknet_model.md`](test_darknet_model.md) |
| `test_darknet_recipe.cpp` | [`test_darknet_recipe.md`](test_darknet_recipe.md) |
| `test_darknet_tensors.cpp` | [`test_darknet_tensors.md`](test_darknet_tensors.md) |
| `test_darknet_training.cpp` | [`test_darknet_training.md`](test_darknet_training.md) |
| `test_darknet_weights.cpp` | [`test_darknet_weights.md`](test_darknet_weights.md) |
| `test_darknet_import/test_decode.cpp` | [`test_darknet_import/test_decode.md`](test_darknet_import/test_decode.md) |
| `test_darknet_import/test_yolov3_tiny_layers.cpp` | [`test_darknet_import/test_yolov3_tiny_layers.md`](test_darknet_import/test_yolov3_tiny_layers.md) |
| `test_darknet_import/test_yolov3_forward.cpp` | [`test_darknet_import/test_yolov3_forward.md`](test_darknet_import/test_yolov3_forward.md) |
| `test_darknet_import/test_yolov3_tiny_forward.cpp` | [`test_darknet_import/test_yolov3_tiny_forward.md`](test_darknet_import/test_yolov3_tiny_forward.md) |
| `test_darknet_import/test_yolov4_forward.cpp` | [`test_darknet_import/test_yolov4_forward.md`](test_darknet_import/test_yolov4_forward.md) |
| `test_darknet_import/test_yolov4_tiny_forward.cpp` | [`test_darknet_import/test_yolov4_tiny_forward.md`](test_darknet_import/test_yolov4_tiny_forward.md) |
| `test_darknet_import/test_yolov7_forward.cpp` | [`test_darknet_import/test_yolov7_forward.md`](test_darknet_import/test_yolov7_forward.md) |
| `test_darknet_import/test_yolov7_tiny_forward.cpp` | [`test_darknet_import/test_yolov7_tiny_forward.md`](test_darknet_import/test_yolov7_tiny_forward.md) |
| `test_darknet_training/test_yolov3.cpp` | [`test_darknet_training/test_yolov3.md`](test_darknet_training/test_yolov3.md) |
| `test_darknet_training/test_yolov3_tiny.cpp` | [`test_darknet_training/test_yolov3_tiny.md`](test_darknet_training/test_yolov3_tiny.md) |
| `test_darknet_training/test_yolov4.cpp` | [`test_darknet_training/test_yolov4.md`](test_darknet_training/test_yolov4.md) |
| `test_darknet_training/test_yolov4_tiny.cpp` | [`test_darknet_training/test_yolov4_tiny.md`](test_darknet_training/test_yolov4_tiny.md) |
| `test_darknet_training/test_yolov7.cpp` | [`test_darknet_training/test_yolov7.md`](test_darknet_training/test_yolov7.md) |
| `test_darknet_training/test_yolov7_tiny.cpp` | [`test_darknet_training/test_yolov7_tiny.md`](test_darknet_training/test_yolov7_tiny.md) |
| `test_darknet_store.cpp` | [`test_darknet_store.md`](test_darknet_store.md) |
| `test_onnx_import/test_import.cpp` | [`test_onnx_import/test_import.md`](test_onnx_import/test_import.md) |
| `test_onnx_import/test_yolox_detections.cpp` | [`test_onnx_import/test_yolox_detections.md`](test_onnx_import/test_yolox_detections.md) |
| `test_ultralytics_checkpoint.cpp` | [`test_ultralytics_checkpoint.md`](test_ultralytics_checkpoint.md) |
| `test_ultralytics_import/test_detections.cpp` | [`test_ultralytics_import/test_detections.md`](test_ultralytics_import/test_detections.md) |
| `test_ultralytics_import/test_import.cpp` | [`test_ultralytics_import/test_import.md`](test_ultralytics_import/test_import.md) |
| `test_ultralytics_model.cpp` | [`test_ultralytics_model.md`](test_ultralytics_model.md) |
| `test_ultralytics_tensors.cpp` | [`test_ultralytics_tensors.md`](test_ultralytics_tensors.md) |

## Fixtures

`fixtures/caffe/` holds the LeNet prototxt and caffemodel of `test_caffe_import.cpp` and the output
OpenCV's `readNetFromCaffe` gives for one input. `tools/reference/make_caffe_lenet_fixture.py`
writes the first two, and `tools/reference/export_caffe_reference.py` writes the third.
