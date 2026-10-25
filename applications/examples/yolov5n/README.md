# yolov5n: the YOLOv5n detector as a model file

`model.json` describes YOLOv5n, release 7.0, in version 2 of the model file: 32 rows read by
`NN::build_model`. It follows `yolov5s.yaml` at depth 0.33 and width 0.25, with 80 classes and a
`[640, 640, 3]` input. There is no C++ and no training configuration in this directory, and no
run has been made with this model.

## Build

`NN::build_model` builds the file over the registry that `NN::register_layers` and
`Vision::register_nodes` fill. No command trains or evaluates this model yet.

The file builds into 1,872,157 parameters, the count of a sum over its rows and of the
Ultralytics summary line. `tests/vision/registration/test_yolov5n.cpp` pins that count, the three
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
