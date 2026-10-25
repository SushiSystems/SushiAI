# yolov8n: the YOLOv8n detector as a model file

`model.json` describes YOLOv8n in version 2 of the model file: 23 rows, one per row of the
Ultralytics `yolov8.yaml` at scale `n` (depth 0.33, width 0.25, at most 1024 channels), read by
`NN::build_model` (`modules/layers/nn/include/SushiAI/nn/node_registry.hpp`). There is no C++ and no
training configuration in this directory.

No command trains or evaluates this model yet, and no run has been made with it.
`Medium_UltralyticsFiles` in `tests/vision/registration/test_ultralytics_files.cpp` pins the build.

## Rows

| Rows | Type | Notes |
| --- | --- | --- |
| 0, 1, 3, 5, 7 | `conv` | 3x3, stride 2, 16, 32, 64, 128 and 256 filters |
| 2, 4, 6, 8 | `c2f` | 1, 2, 2 and 1 repeats, `shortcut` true |
| 9 | `sppf` | 5x5, three pools |
| 10, 13 | `upsample` | scale 2 |
| 11, 14, 17, 20 | `concat` | rows 6, 4, 12 and 9 |
| 12, 15, 18, 21 | `c2f` | one repeat, `shortcut` false |
| 16, 19 | `conv` | 3x3, stride 2, 64 and 128 filters |
| 22 | `detect` | 80 classes over rows 15, 18 and 21 |

`hidden` of a `c2f` row is half its `filters`, as `C2f` computes it. The `detect` row states
`reg_max` 16, `end2end` false and a `plain` class branch. It publishes six ports at strides 8, 16
and 32.

## Parameters

The built model holds 3,157,184 parameters. The Ultralytics summary line gives 3,157,200; the 16
extra are the fixed ramp of the distribution expectation, which has no state record here.

## Task

The `task` section is `anchor_free_detection` with the default score threshold, 0.25, and
suppression by intersection over union at 0.7. Strides, ports, `reg_max` and the class count come
from the built model.
