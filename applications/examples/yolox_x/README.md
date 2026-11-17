# yolox_x: YOLOX-X as a model file, for ByteTrack's MOT17 detector

`model.json` describes YOLOX-X (Megvii YOLOX, depth 1.33, width 1.25) with one class, the
detector of ByteTrack's `mot17x` checkpoint, in version 2 of the model file: 25 rows read by
`NN::build_model`. `map.json` names the tensors of that checkpoint. There is no C++ in this
directory, and no run has been made with this model.

## The rows

| Rows | Type | YOLOX part |
| --- | --- | --- |
| 0 | `focus` | `stem`: the four pixel phases stacked into 12 channels, then a 3x3 BaseConv to 80 |
| 1 to 9 | `conv`, `c3`, `sppf` | `dark2` to `dark5`: a 3x3 stride-2 BaseConv and a CSP layer per stage, the spatial pyramid pooling before the last CSP layer; rows 4, 6 and 9 are the outputs at strides 8, 16 and 32 |
| 10 to 23 | `conv`, `upsample`, `concat`, `c3` | The PAFPN: `lateral_conv0`, `C3_p4`, `reduce_conv1`, `C3_p3`, `bu_conv2`, `C3_n3`, `bu_conv1`, `C3_n4`; rows 17, 20 and 23 are its outputs |
| 24 | `decoupled_head` | `YOLOXHead`: a 1x1 stem per level, a class tower and a box tower of two 3x3 BaseConv, and three 1x1 predictors |

The channel widths are 80, 160, 320, 640 and 1280 (64 times 1.25 times 1, 2, 4, 8 and 16). The CSP
layers repeat 4, 12, 12 and 4 times in the backbone (3 times 1.33 rounds to 4) and 4 times each in
the neck.

## The outputs

Row 24 publishes nine values, which the file names `level<l>.classes`, `level<l>.boxes` and
`level<l>.objectness` for levels 0, 1 and 2, the names `yolox_head_contract.hpp` fixes. At the
800 by 1440 input the levels are grids of 100 by 180, 50 by 90 and 25 by 45, at strides 8, 16 and
32. The classes and the objectness are logits, the box is the raw `x`, `y`, `log w`, `log h`; the
model applies no sigmoid and no grid. `decode_yolox` turns the raw values into boxes, and the
`yolox_detection` row of `sushiai inference` runs it (`applications/programs/README.md`, "The run
table").

The `task` section lives in `model.json`, because `Config::load_model` reads the section from there
and from nowhere else. It states a score threshold of 0.1 and an IoU suppression at 0.7, the values
a printout of detections wants.

## The preprocessing

The `input` object states what ByteTrack's `preproc` does to an image: `"fit": "letterbox"` at 800
by 1440, which keeps the image in the top left corner and pads the right and the bottom with 114,
and `"normalise"`, which turns the fitted bytes into the network input as `(byte / 255 - mean) /
std` per channel, with the mean 0.485, 0.456, 0.406 and the standard deviation 0.229, 0.224, 0.225
in red, green, blue order. ByteTrack reads the image as BGR and swaps it to RGB before it divides;
the decoder of this repository delivers RGB, so the file states `"channels": "rgb"` and no swap
happens. The pad value 114 is a byte of the fitted image, so it is normalised with the rest.
`Config::parse_input_normalise` reads the object, and every training, evaluation and inference
read applies it in `write_batch_inputs` (`docs/guides/MODEL_FILE.md`, "The input object").

The values come from ByteTrack's source, not from a run: the `yolox` package was not installed in
the reference environment when this file was written. `tools/reference/export_yolox_reference.py`
accepts `--mean` and `--std`, runs the installed `preproc` and writes the exact input array, so the
first comparison on the real checkpoint also checks these numbers.

## Assumptions

The file is written from the structure of the YOLOX source and from the names of ByteTrack's
`yolox_x` state dictionary. Nothing was read from a checkpoint or run. Each point below is one
the first run on the real file confirms or corrects.

1. Depth 1.33 and width 1.25 give 4, 12, 12 and 4 backbone repeats and a base width of 80, with
   `int` truncation for the width and rounding for the repeats, as `YOLOPAFPN` computes them.
2. Every convolution is a BaseConv: a bias-free convolution, a batch normalisation and a SiLU.
   Epsilon 0.001 and momentum 0.03 are the values YOLOX's model initialiser writes into every
   normalisation, so the file states them in `defaults` for each row type.
3. A `CSPLayer` is a `c3` row: `conv1` and `conv2` narrow to half the output, `conv3` joins the two
   halves, and the bottlenecks are a 1x1 and a 3x3 at the half width. The `shortcut` is true in
   `dark2` to `dark4` and false in `dark5` and in the whole neck.
4. `SPPBottleneck` has no row of its own. Its parallel max pools of 5, 9 and 13 are the `sppf` row
   with kernel 5 and 3 pools, because two chained 5x5 pools make a 9x9 window and three a 13x13
   one. This holds when a pool ignores the cells it pads, as PyTorch's does; the `sppf` rows of
   the YOLOv5 and YOLOv8 examples depend on the same property. `activate_narrowing` is true because
   YOLOX puts a SiLU after the first convolution of the block. The output stacks the input, the
   5, 9 and 13 pools in that order, which is `sppf`'s order.
5. `upsample` is the nearest-neighbour doubling of `nn.Upsample(scale_factor=2)`, and each
   `concat` stacks the upsampled map first and the backbone map second, as `YOLOPAFPN` does. The
   two `concat` rows of the bottom-up path read rows 14 and 10, the outputs of `reduce_conv1` and
   `lateral_conv0`.
6. The head reads the objectness from the box tower, as `YOLOXHead` does, and `hidden` is
   256 times the width, 320, for the stems and both towers.
7. The task thresholds, a score of 0.1 and an IoU of 0.7, are chosen for a printout and were not
   read from ByteTrack's experiment file. A tracking evaluation scores at a lower threshold.

## Import

The checkpoint is a PyTorch file whose state dictionary sits under `model`. `map.json` reads that
key, drops every `num_batches_tracked` and renames each prefix of ByteTrack's names to its row:
`backbone.backbone.` is the CSPDarknet, `backbone.` the PAFPN and `head.` the head. A CSP layer
renames its `conv1`, `conv2`, `conv3` and `m.<k>.conv1`, `m.<k>.conv2` to `cv1`, `cv2`, `cv3` and
`m.<k>.cv1`, `m.<k>.cv2`, and the SPP block renames `conv1` and `conv2` to `cv1` and `cv2`. The map has 132 rules,
generated from the structure above and checked by `Medium_YoloxXMap` in
`tests/vision/registration/test_registration/test_yolox_x.cpp`, which writes the 748 tensors of
the state dictionary by formula and requires them to fill the model's records exactly. A convolution
weight `[out, in, kh, kw]` goes to `[out, kh, kw, in]`.

```bash
sa import weights applications/examples/yolox_x/model.json mot17x.pth.tar \
    --map applications/examples/yolox_x/map.json --output build/yolox_x/mot17x.sushi
```

A tensor the model has no record for, a record the file lacks and a shape that differs stop the
command with one message that names all of them.

## Inference

A MOT17 frame is one image: the third argument of `inference` is the image file, and no other
argument is needed. The class count comes from the `classes` of row 24.

```bash
sa inference applications/examples/yolox_x/model.json build/yolox_x/mot17x.sushi frame.jpg
```

It prints the detections above the score threshold in pixels of the source image, then the line
that says the mean average precision was not computed.

## Reference

`tools/reference/export_yolox_reference.py` runs the checkpoint under PyTorch, outside this
repository, and writes one SACP file with the input and the nine head outputs of one image.
`tools/reference/README.md` describes it. The Large test
`tests/vision/registration/test_registration/test_yolox_x_import.cpp` imports the checkpoint and asserts that nothing is
refused; it reads the folder in `SA_YOLOX_DATA_DIR`.
