# yolo11n: the YOLO11n detector as a model file

`model.json` describes YOLO11n in version 2 of the model file: 24 rows, one per row of the
Ultralytics `yolo11.yaml` at scale `n` (depth 0.50, width 0.25, at most 1024 channels), read by
`NN::build_model` (`modules/layers/nn/include/SushiAI/nn/node_registry.hpp`). `train.json` and `eval.json` sit beside it for
`sa train` and `sa inference`. There is no C++ in this directory.

No run has been made with this model.
`Medium_UltralyticsFiles` in `tests/vision/registration/test_registration/test_ultralytics_files.cpp` pins the build.

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

## Training and inference

`train.json` trains from scratch: it names no `weights` file. Its `checkpoint` is
`../../../build/yolo11n/trained.sushi`, and `data.root` is `../../../datasets/coco128`, the
repository's git-ignored `datasets/coco128`; edit it to where coco128 lies on your machine. The path rules are those of `applications/examples/yolov5n/README.md`.

The data block reads coco128 at `[640, 640]` in batches of 16 and states no `fit`, so the
detection default, `letterbox`, applies. The `objective` is `anchor_free` with `box_gain` 7.5,
`class_gain` 0.5, `distribution_gain` 1.5 and `l1_gain` 0. The optimizer is SGD (learning rate 0.01,
momentum 0.937, weight decay 0.0005, Nesterov) for 10 epochs, and the run is deterministic.

```bash
sa train applications/examples/yolo11n/train.json
sa inference applications/examples/yolo11n/model.json build/yolo11n/trained.sushi     applications/examples/yolo11n/eval.json
```

`eval.json` reads the training split, so it measures fit, not generalisation.
