# yolo11n: the YOLO11n detector as a model file

`model.json` describes YOLO11n in version 2 of the model file: 24 rows, one per row of the
Ultralytics `yolo11.yaml` at scale `n` (depth 0.50, width 0.25, at most 1024 channels), read by
`NN::build_model` (`modules/layers/nn/include/SushiAI/nn/node_registry.hpp`). There is no C++ and no
training configuration in this directory.

No command trains or evaluates this model yet, and no run has been made with it.
`Medium_UltralyticsFiles` in `tests/vision/registration/test_ultralytics_files.cpp` pins the build.

## Rows

| Rows | Type | Notes |
| --- | --- | --- |
| 0, 1, 3, 5, 7 | `conv` | 3x3, stride 2, 16, 32, 64, 128 and 256 filters |
| 2, 4 | `c3k2` over `bottleneck` | `hidden` is a quarter of `filters`, the `0.25` of the yaml |
| 6, 8 | `c3k2` over `c3k` | `hidden` is half of `filters` |
| 9 | `sppf` | 5x5, three pools, no shortcut |
| 10 | `c2psa` | 256 filters, 2 heads |
| 11, 14 | `upsample` | scale 2 |
| 12, 15, 18, 21 | `concat` | rows 6, 4, 13 and 10 |
| 13, 16, 19 | `c3k2` over `bottleneck` | `c3k` false in the yaml |
| 17, 20 | `conv` | 3x3, stride 2, 64 and 128 filters |
| 22 | `c3k2` over `c3k` | `c3k` true in the yaml |
| 23 | `detect` | 80 classes over rows 16, 19 and 22 |

`shortcut` is true on every `c3k2` row, the default of `C3k2.__init__`, which `yolo11.yaml` does not
state. The `detect` row states `reg_max` 16, `end2end` false and a `depthwise` class branch. It
publishes six ports at strides 8, 16 and 32.

## Parameters

The built model holds 2,624,064 parameters. The Ultralytics summary line gives 2,624,080; the 16
extra are the fixed ramp of the distribution expectation, which has no state record here.

## Task

The `task` section is `anchor_free_detection` with the default score threshold, 0.25, and
suppression by intersection over union at 0.7. Strides, ports, `reg_max` and the class count come
from the built model.
