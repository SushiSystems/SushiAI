# yolov5n: the YOLOv5n detector as a model file

`model.json` describes YOLOv5n, release 7.0, in version 2 of the model file: 32 rows read by
`NN::build_model`. It follows `yolov5s.yaml` at depth 0.33 and width 0.25, with 80 classes and a
`[640, 640, 3]` input. `train.json` and `eval.json` sit beside it for `sa train` and `sa inference`. There is no C++ in
this directory, and no run has been made with this model.

## Build

`NN::build_model` builds the file over the registry that `NN::register_layers` and
`Vision::register_nodes` fill.

The file builds into 1,872,157 parameters, the count of a sum over its rows and of the
Ultralytics summary line. `tests/vision/registration/test_registration/test_yolov5n.cpp` pins that count, the three
head shapes and the task section.

## The rows

| Rows | Type | Meaning |
| --- | --- | --- |
| 0 to 2 | `conv2d`, `batchnorm2d`, `silu` | The 6x6 stride-2 stem with padding 2 |
| 3 to 11 | `conv`, `c3`, `sppf` | The backbone; rows 6 and 8 are the P3 and P4 outputs |
| 12 to 25 | `conv`, `upsample`, `concat`, `c3` | The neck, `c3` with `shortcut` false |
| 26 to 28 | `conv2d` | Three 1x1 convolutions of 255 channels with biases |

The stem is three rows because the `conv` row pads by half its kernel, which gives 321 cells
from a 6x6 window at stride 2; the reference pads by 2 and gives 320.

## The heads

Rows 26, 27 and 28 are published as `head_0`, `head_1` and `head_2`, at strides 8, 16 and 32.
A head has 255 channels, three anchors of `x, y, w, h, objectness` and 80 classes, all logits;
the model applies no sigmoid. The task section states `"coding": "logit"`, `"scale_xy": 2.0`
and `"size": "squared_sigmoid"`, which decodes a centre as `2 * sigmoid - 0.5` and a side as
`(2 * sigmoid)^2` times the anchor.

The task section states the nine anchors as fractions of 640 (10x13 is `[0.015625, 0.0203125]`)
and `"assignment": {"rule": "shape_ratio"}`.

## Biases

`Vision::Training::derive_anchor_bias_heads` and `initialize_anchor_biases`
(`modules/vision/training/include/SushiAI/vision/training/anchor_bias_init.hpp`) add
`log(8 / (640 / stride)^2)` to the objectness bias and `log(0.6 / (80 - 0.99))` to the class
biases of every anchor slot of rows 26, 27 and 28.

## Training and inference

`train.json` trains the model from scratch: it names no `weights` file, so every parameter starts
from the initializer and the anchor biases above. Its paths:

| Key | Value | Resolves to |
| --- | --- | --- |
| `model` | `model.json` | Against the directory of `train.json`, this directory |
| `checkpoint` | `../../../build/yolov5n/trained.sushi` | The repository's `build/yolov5n`; the run writes the file there |
| `data.root` | `../../../datasets/coco128` | Against the directory of `train.json`, the repository's `datasets/coco128`, which is git-ignored |

The data block reads coco128 at `[640, 640]` in batches of 16. The model's input block says
`letterbox`, so an image keeps its aspect ratio and is padded to the input. The `objective` is
`anchor` with `box_gain` 0.05, `class_gain` 0.5 and `objectness_gain` 1.0; the optimizer is SGD at
learning rate 0.01, momentum 0.937, weight decay 0.0005 and Nesterov updates, with no schedule.
The file runs 10 epochs and carries no `deterministic` key, so the run is deterministic.

```bash
sa train applications/examples/yolov5n/train.json
sa inference applications/examples/yolov5n/model.json build/yolov5n/trained.sushi     applications/examples/yolov5n/eval.json
```

`eval.json` carries the same data block, so the evaluation split is the training split: it
measures fit, not generalisation. No cost, precision or timing is quoted, because none has been
measured.
