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
| `test_darknet_import/test_import.cpp` | [`test_darknet_import/test_import.md`](test_darknet_import/test_import.md) |
| `test_darknet_import/test_decode.cpp` | [`test_darknet_import/test_decode.md`](test_darknet_import/test_decode.md) |
| `test_darknet_import/test_yolov3_tiny_layers.cpp` | [`test_darknet_import/test_yolov3_tiny_layers.md`](test_darknet_import/test_yolov3_tiny_layers.md) |
| `test_darknet_import/test_yolov3_forward.cpp` | [`test_darknet_import/test_yolov3_forward.md`](test_darknet_import/test_yolov3_forward.md) |
| `test_darknet_import/test_yolov3_tiny_forward.cpp` | [`test_darknet_import/test_yolov3_tiny_forward.md`](test_darknet_import/test_yolov3_tiny_forward.md) |
| `test_darknet_import/test_yolov4_forward.cpp` | [`test_darknet_import/test_yolov4_forward.md`](test_darknet_import/test_yolov4_forward.md) |
| `test_darknet_import/test_yolov4_tiny_forward.cpp` | [`test_darknet_import/test_yolov4_tiny_forward.md`](test_darknet_import/test_yolov4_tiny_forward.md) |
| `test_darknet_import/test_yolov7_forward.cpp` | [`test_darknet_import/test_yolov7_forward.md`](test_darknet_import/test_yolov7_forward.md) |
| `test_darknet_import/test_yolov7_tiny_forward.cpp` | [`test_darknet_import/test_yolov7_tiny_forward.md`](test_darknet_import/test_yolov7_tiny_forward.md) |
| `test_darknet_training/test_recipe.cpp` | [`test_darknet_training/test_recipe.md`](test_darknet_training/test_recipe.md) |
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

## Shared check helpers

The parity suites do not repeat their checks. Three modules hold them, and each test file of
`test_darknet_import/` and `test_darknet_training/` states what its model has and calls them.

- `modules/vision/darknet_import_cases` (`darknet_import.hpp`) imports `<stem>.cfg` and
  `<stem>.weights` in memory and holds the checks of the import: `check_tensors_against_entries`
  and `fill_imported_state` (strict in both ways, naming the tensor or the entry),
  `check_imported_rows`, `check_imported_task`, `check_imported_weights`,
  `check_imported_entries`, and `check_written_files` and `check_written_weights`, which write the
  import into an empty directory and read it back. Its fixture imports the stem a case was
  instantiated with and skips naming the missing file.
- `modules/vision/darknet_forward_cases` (`darknet_forward.hpp`) lists the six parity models,
  finds the files and reference runs of a stem, and runs one forward pass of the imported model
  over a run's `input`. It measures a tensor under `absolute + relative * |reference|` and holds
  four checks: `check_folded_heads`, `check_yolo_outputs`, `check_detections` and
  `check_separate_form`.
- `modules/vision/darknet_training_cases` (`darknet_training.hpp`, `training_units.hpp`) loads what
  a run dumped of the training pass of one head and the `truth` rows, and holds the checks of a
  step: `check_training_settings`, `check_host_deltas`, `check_two_image_deltas`,
  `check_step_heads`, `check_step_deltas`, `check_head_updates`, `check_training_plan`,
  `check_plan_changes_no_value`, `check_first_gradients`, `check_rolling_statistics` and
  `check_one_batch_overfit`. `units` gives a bound in float roundoff units, and
  `K_OVERFIT_COST_SHARE` is the share of its first cost that the cost after the overfit steps must
  fall below.

## Fixtures

`fixtures/caffe/` holds the LeNet prototxt and caffemodel of `test_caffe_import.cpp` and the output
OpenCV's `readNetFromCaffe` gives for one input. `tools/reference/make_caffe_lenet_fixture.py`
writes the first two, and `tools/reference/export_caffe_reference.py` writes the third.
