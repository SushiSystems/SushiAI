# Darknet-era YOLO: reference run and source findings

**Status:** Forward and `[yolo]` training references exist for all six models; every layer of `yolov3-tiny` is dumped; one training step under batch statistics exists for `yolov3-tiny` and `yolov7-tiny`. Sections 8 to 14 are the second pass.

This report describes what AlexeyAB's Darknet does, in words and formulas, so that the behaviour can be implemented from scratch. It quotes no Darknet code. File, function and line pointers refer to commit `59596d7880f6504768df41d6daa586f5cb2b932f` (2025-12-15) of `https://github.com/AlexeyAB/darknet`, paths relative to the clone's root. The clone, the build and the patch live in `D:\Projects\.sushisystems\models\darknet\_darknet\` and are deleted with it; nothing from them is in this repository.

## 1. Build

- Toolchain: MSVC 14.44.35207 (Visual Studio 2022 Build Tools, `vcvars64`), CMake with the Ninja generator, Release, target `darknet`. CUDA, cuDNN, OpenCV and the vcpkg integration are off. Threads come from the bundled `3rdparty/pthreads`; its DLL directory is put on `PATH` for the run. The first configure and build succeeded.
- OpenMP is on (CMake found it), and Darknet's x86-64 Windows build compiles its AVX paths and selects them at run time. The convolution therefore sums products in the order of Darknet's vectorised matrix product, not in plain index order. Two runs on this machine gave bit-identical head tensors; a different CPU could differ in the last bits.
- `_darknet` is 93 MB: `src` 17 MB, `build` 5.4 MB, `out` 71 MB.
- To delete: `Remove-Item -Recurse -Force D:\Projects\.sushisystems\models\darknet\_darknet`. Move `_darknet\out` away first.

## 2. How the reference was taken

The patch touches one file, `src/detector.c`, and is 177 lines as a unified diff (137 added). It is kept as `_darknet/dump_tensors.patch` and reverses cleanly. It acts inside `detector test` and only when environment variables are set:

| Variable | Effect |
| --- | --- |
| `DARKNET_DUMP_DIR` | Directory that receives the tensors |
| `DARKNET_RAW_INPUT` | A file of `channels * height * width` little-endian `float32` in channel, row, column order, taken in place of a decoded image |
| `DARKNET_TRUTH` | A file of rows of five `float32` (`x_center y_center width height class`); turns on the training-mode pass of section 2.3 |

The input is a raw float file and not a PPM or PNG. An image file would pass through Darknet's decoder, its division by 255 and its resampler, and the reference would then depend on three things SushiAI does not need to reproduce. With an input that already has the network's size, Darknet's resize returns a copy (`src/image.c`, `resize_image`, line 1401), so layer 0 receives the file's floats unchanged. The dumped `input.bin` equals the generated file byte for byte (same SHA-256).

Command, run from the clone's root so that `cfg/coco.data` and `data/labels` resolve:

```
darknet detector test cfg/coco.data <model>.cfg <model>.weights synthetic -dont_show -ext_output [-thresh 0.01]
```

### 2.1 Input and truth recipes

These rebuild the two inputs without Darknet. Every input value is `k / 255` for an integer `k`, computed in integers and divided once in `float32`.

```python
import numpy as np

def synthetic_input(width, height):
    c, y, x = np.meshgrid(np.arange(3), np.arange(height), np.arange(width), indexing="ij")
    value = ((x * x + 3 * y * y + 7 * x * y) // 16 + 50 * c) % 256
    return (value.astype(np.float32) / np.float32(255)).astype("<f4")   # shape [3, height, width]

truth = np.array([(0.50, 0.50, 0.40, 0.60, 0),
                  (0.25, 0.30, 0.10, 0.15, 16),
                  (0.70, 0.75, 0.30, 0.20, 2)], dtype="<f4")            # x y w h class
```

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `input_416x416.bin` | 2 076 672 | `088fdab79ea238c8fd84c8e8e1beb8b33fc782cea96b0b337f39c82cb1b11304` |
| `input_640x640.bin` | 4 915 200 | `2c585b8211e8464048a158f8f6586bbec3d1370e76fd4c9724ac2bd429d0b533` |
| `truth_3_boxes.bin` | 60 | `1afd29ecbfcd6476e8884212b4f0ff369ea52ecb229d824ff0653dda3b59159a` |

The pattern is not a photograph, so no model finds an object at the default threshold of 0.25. Each model was therefore also run at `-thresh 0.01`, which yields rows for the suppression step. The head tensors of the two runs are identical.

### 2.2 Dump layout

Every tensor is two files: `<name>.bin`, raw little-endian `float32` with no header, and `<name>.json`, one JSON object with `name`, `dtype`, `byte_order`, `shape`, `kind` and, for a feature map, `order` (`"CHW"`) and `layer`. Batch is 1 and has no axis.

| Name | Shape | Content |
| --- | --- | --- |
| `input` | `[3, H, W]` | The floats layer 0 received |
| `layer_<k-1>_yolo_input` | `[A * (5 + C), h, w]` | Output of the layer before `[yolo]` layer `k`: the raw head |
| `layer_<k>_yolo` | same | The `[yolo]` layer's own output in inference mode (section 3.6) |
| `detections` | `[rows, 7]` | After threshold and suppression: `x_center y_center width height objectness score class`, boxes as fractions of the network input. One row per detection and class whose score exceeds the threshold, in Darknet's order after its last sort. The JSON holds `thresh`, `hier_thresh`, `nms`, `nms_kind` (0 default, 1 greedy, 2 DIoU), `beta_nms`, `letter_box`, `candidates_before_nms` |
| `truth` | `[rows, 6]` | The truth rows as the `[yolo]` layer read them: the five input columns and the row number |
| `layer_<k>_yolo_delta` | `[A * (5 + C), h, w]` | The layer's delta after its training-mode forward. The JSON holds `cost` and the layer's loss settings |
| `layer_<k>_yolo_assigned_row` | `[A, h, w]` | Truth row number at the anchor slot and cell that the best-anchor rule assigned, -1 elsewhere |
| `layer_<k>_yolo_assigned_class` | `[A, h, w]` | Class id at the same positions, -1 elsewhere |

Channel index within a head: `a * (5 + C) + e`, with `a` the anchor slot within the layer's `mask` and `e` in `x, y, w, h, objectness, class_0 ...`. `training_log.txt` holds the loss line Darknet prints per `[yolo]` layer.

### 2.3 What the training-mode pass is, and what it is not

After the inference forward, each `[yolo]` layer is run once more with training switched on, the same head input and the truth rows. This is the code path a real training step takes for that layer (`src/yolo_layer.c`, `forward_yolo_layer`, line 662; the GPU build pulls the head to the host and calls the same function). The layer's backward step only adds its delta to the delta of the layer before it (`backward_yolo_layer`, line 930), so the dumped delta is what flows back into the body.

The head input comes from the inference forward: batch norm folded into the weights and using the stored rolling statistics. A real training step computes the body under batch statistics. The dumps therefore pin the assignment and the loss exactly, for a known head tensor, and say nothing about batch-statistics forward, weight gradients or the update. Section 5 says where those live.

## 3. Findings on the forward pass

Each item names the SPEC statement it settles.

### 3.1 `.cfg` reading

- Every space, tab, carriage return and line feed is removed from a line, inner ones included, before the line is interpreted (`src/utils.c`, `strip`, line 424; `src/parser.c`, `read_cfg`, line 1800). SPEC 4.1 is right.
- A negative index in `layers` of `[route]` or `from` of `[shortcut]` is relative to the current layer; zero or more is absolute (`parse_route`, line 1085; `parse_shortcut`, line 957).
- Activations in the six cfgs, counted from the files: `yolov3-tiny` 11 leaky, 2 linear; `yolov3` 72 leaky, 26 linear; `yolov4-tiny` 19 leaky, 2 linear; `yolov4` 35 leaky, 26 linear, 72 mish; `yolov7-tiny` 55 leaky, 3 logistic; `yolov7` 89 swish, 3 logistic. **Correction:** `logistic` is used, by the three convolutions before the heads of both v7 models. `swish` occurs in `yolov7` only, not in `yolov7-tiny`.
- Section types in the six cfgs: `convolutional`, `maxpool`, `route`, `shortcut`, `upsample`, `yolo`. No `groups` key on a convolution, no `avgpool`, no `reorg`.

### 3.2 Convolution

- Padding: if `pad` is non-zero the padding is `size / 2` (integer division) and `padding` is ignored; otherwise it is `padding`, default 0 (`parse_convolutional`, lines 189-191). The same count pads all four sides. Output extent `(in + 2 * padding - size) / stride + 1` (`src/convolutional_layer.c`, `convolutional_out_height`, line 78). SPEC 4.1 is right.
- Order of operations: convolution without bias, then batch norm or bias, then the activation (`forward_convolutional_layer`, line 1204; the normalise-or-bias step at 1390-1394, the activation at 1397-1404).
- Weight tensor order: `[out, in / groups, kh, kw]`, row-major. The unfolded input column index is `(c * ksize + kh) * ksize + kw` (`src/im2col.c`, `im2col_cpu`), and filter `f` owns the `f`-th block of `in / groups * size * size` weights (`forward_convolutional_layer`, lines 1222-1235). SPEC 4.2 is right.

### 3.3 Batch normalisation

- Evaluation: `y = scale * (x - mean) / sqrt(variance + 0.00001) + bias`, epsilon inside the root (`src/blas.c`, `normalize_cpu`, line 292; `src/batchnorm_layer.c`, `forward_batchnorm_layer`, lines 205-208). SPEC 4.3 is right for AlexeyAB. pjreddie's form was not read; that repository is not on disk.
- **Addition:** `detector test` folds batch norm into the convolution before the forward pass (`src/detector.c`, `test_detector`, line 1635; `src/network.c`, `fuse_conv_batchnorm`, line 1295). In double precision it sets `bias' = bias - scale * mean / sqrt(variance + 0.00001)` and multiplies every weight of filter `f` by `scale_f / sqrt(variance_f + 0.00001)`, then rounds to `float32`. The reference tensors come from this folded arithmetic. A `BatchNorm2d` evaluated after the convolution is the same function with a different rounding; the parity tolerance has to allow for that or the importer has to fold the same way.
- Training mode keeps rolling statistics as `rolling = 0.9 * rolling + 0.1 * batch` (lines 196-199).

### 3.4 Max pooling

- Defaults: `stride=1`, `size=stride`, `padding=size-1` (`parse_maxpool`, lines 839-843). `padding` is the total over both sides. Output extent `(in + padding - size) / stride + 1` (`src/maxpool_layer.c`, `make_maxpool_layer`, lines 100-101).
- The first window starts at `-(padding / 2)` with C integer division, which truncates toward zero. A tap outside the input never wins. Inference uses a second implementation with the same offset rule (`src/gemm.c`, `forward_maxpool_layer_avx`, both variants, lines 1870 and 2412); the training path is `forward_maxpool_layer`, lines 283-304.
- Size 2, stride 1: padding 1, offset 0, output extent equal to the input. Nothing is padded above or left; one row is padded below and one column on the right. SPEC 2.2 and 4.1 are right.
- Size 2, stride 2: padding 1, offset 0, extent `(in - 1) / 2 + 1`. Sizes 5, 9, 13 at stride 1 (the SPP blocks of v4 and v7): padding `size - 1`, offset `-(size - 1) / 2`, extent unchanged.

### 3.5 Route, shortcut, upsample, activations

- `[route]`: joins the listed layers' outputs along channels, in list order. With `groups = g` and `group_id = i`, each input contributes its channels `[i * C / g, (i + 1) * C / g)`, and the output has the summed channels divided by `g` (`src/route_layer.c`, `forward_route_layer`, lines 84-95; `parse_route`, lines 1091-1110). SPEC 4.1 is right.
- `[shortcut]`: output is the output of the layer before plus the output of layer `from`, element by element, then the activation, default `linear` (`src/shortcut_layer.c`, `forward_shortcut_layer`, lines 156-173; `parse_shortcut`, line 917). When the shapes differ or `weights_type` is set, a general routine adds only the overlapping leading elements and can weight the inputs; whether any of the six cfgs reaches it was not checked.
- `[upsample]`: nearest neighbour, output pixel `(j, i)` reads input pixel `(j / stride, i / stride)` and multiplies by `scale`; `stride` defaults to 2 and `scale` to 1 (`src/blas.c`, `upsample_cpu`, line 483; `parse_upsample`, lines 1063-1065). SPEC 4.1 is right.
- Activations (`src/activations.h`, lines 60-77; `src/activations.c`, lines 138-161):
  - `leaky(x) = x` for `x > 0`, else `0.1 * x`.
  - `logistic(x) = 1 / (1 + exp(-x))`.
  - `swish(x) = x * logistic(x)`.
  - `mish(x) = x * tanh(softplus(x))`, with `softplus(x) = x` for `x > 20`, `exp(x)` for `x < -20`, `log(exp(x) + 1)` otherwise, and `tanh(v)` computed as `2 / (1 + exp(-2 v)) - 1`. PLAN D17's question is settled: the cut-off is 20 on both sides.

### 3.6 `[yolo]` forward

`src/yolo_layer.c`, `forward_yolo_layer`, lines 665-683. The layer copies its input and then, per anchor slot:

- `new_coords = 0`: logistic on `x`, `y`, on objectness and on every class. `w`, `h` stay raw.
- `new_coords = 1`: no logistic at all. The convolution before the layer already has `activation=logistic`, so all `5 + C` entries arrive in (0, 1).
- In both cases `x` and `y` are then mapped to `v * s - (s - 1) / 2` with `s = scale_x_y`. This is stored in the layer's output, so the decode adds no second scaling.

Checked on the dumps: for `yolov3-tiny` the layer output equals the logistic of its input on those entries to 9.4e-8 and equals the input on `w`, `h`; for `yolov7-tiny` it equals `2 * input - 0.5` on `x`, `y` exactly and the input elsewhere.

**Correction:** SPEC 4.1 says `new_coords=1` is refused. The owner's scope now includes v7, so it has to be implemented; the rule above and the decode below are all there is to it in the forward pass.

### 3.7 Box decode and score gates

`get_yolo_box`, lines 131-152, and `get_yolo_detections`, lines 1087-1124. With `X, Y, Wr, Hr` the four box entries of the layer's output (already activated and scaled as in 3.6), cell column `col`, row `row`, grid `w, h`, anchor `(pw, ph) = anchors[mask[a]]` in input pixels:

```
bx = (col + X) / w                       by = (row + Y) / h
new_coords = 0:  bw = exp(Wr) * pw / net_w          bh = exp(Hr) * ph / net_h
new_coords = 1:  bw = Wr^2 * 4 * pw / net_w         bh = Hr^2 * 4 * ph / net_h
score_c = objectness * class_c
```

A cell and anchor is kept when `objectness > thresh` (strict, so a NaN is dropped). Each kept row's `score_c` is set to 0 unless `score_c > thresh`. Both gates of SPEC 4.4 are confirmed. Rows are produced cell by cell in row-major order, and within a cell by anchor slot; layers in file order. The boxes are then mapped back to the image: without letterboxing this is the identity when relative coordinates are asked for, which `detector test` does (`correct_yolo_boxes`, line 939).

### 3.8 Thresholds, suppression, resize

- `detector test` defaults: `thresh = 0.25`, `hier_thresh = 0.5` (`src/detector.c`, `run_detector`, lines 1969 and 1971), NMS threshold 0.45, fixed in the function (`test_detector`, line 1656). pjreddie's default was not read.
- Which suppression runs is decided by `nms_kind` of the last `[yolo]` layer in the file (lines 1677-1685 and 1703-1704), parsed at `src/parser.c` lines 493-499 with `beta_nms` default 0.6.
- All kinds share one frame (`src/box.c`, `do_nms_sort`, line 812, and `diounms_sort`, line 868): rows with zero objectness are moved out; then for each class, sort by that class's score descending, walk the list, skip a row whose score is 0, and zero the score of every later row whose criterion against it exceeds the threshold. No cap on rows, and boxes of different classes never interact.
  - `default` (v3, v3-tiny): criterion IoU.
  - `greedynms` (v4, v4-tiny): criterion `IoU - (d / c)^0.6`, with `d` the squared distance of the centres and `c` the squared diagonal of the smallest enclosing box. **Correction:** the exponent is the constant 0.6 here; `beta_nms` is not read for this kind (`box_diou`, line 195).
  - `diounms` (v7, v7-tiny): criterion `IoU - (d / c)^beta_nms` (`box_diounms`, line 214).
  - When `c` is 0 both fall back to IoU. IoU is intersection over union with no epsilon and is 0 when either is 0 (`box_iou`, line 163).
  - `detector test` honours the cfg's kind: the v7 runs report `nms_kind` 2.
- Resize: `detector test` stretches the image to the network size unless `-letter_box` is given or `[net]` has `letter_box=1` (`test_detector`, lines 1633 and 1674-1675; `src/parser.c`, line 1208). The stretch is bilinear with scale `(in - 1) / (out - 1)` per axis, columns first, then rows (`src/image.c`, `resize_image`, line 1399). The letterbox scales by the smaller ratio with integer extents, fills with 0.5 and embeds at `(net - new) / 2` (`letterbox_image`, line 1006). The decoder gives RGB, channel-first, each byte divided by 255 (`load_image_stb`, line 1521). SPEC 4.6 is right.

### 3.9 `.weights` file

`src/parser.c`, `load_weights_upto`, lines 2254-2274, and `load_convolutional_weights`, lines 2168-2202.

- Header: `major`, `minor`, `revision` as three `int32`; then `seen` as `uint64` when `major * 10 + minor >= 2`, else `uint32`. There is no upper bound on `major` or `minor` in that test. A file with `major > 1000` or `minor > 1000` is read with connected weights transposed.
- Read from the six files: `yolov3-tiny` and `yolov3` are 0.2.0; `yolov4-tiny`, `yolov4`, `yolov7-tiny`, `yolov7` are 0.2.5. All six have a 20-byte header. `seen` is 32 013 312 for the v3 pair, 32 032 000 for `yolov4`, 0 for the other three.
- Per convolution, in layer order: `biases[n]`; with `batch_normalize`: `scales[n]`, `rolling_mean[n]`, `rolling_variance[n]`; then `weights[n * (c / groups) * size * size]`. No other layer type of the six cfgs holds weights.
- Sizes and float counts after the header, measured: `yolov3-tiny` 35 434 956 bytes, 8 858 734; `yolov3` 248 007 048, 62 001 757; `yolov4-tiny` 24 251 276, 6 062 814; `yolov4` 257 717 640, 64 429 405; `yolov7-tiny` 24 967 560, 6 241 885; `yolov7` 147 898 248, 36 974 557. The four of SPEC 5.4 match. Darknet loaded `yolov3-tiny`, `yolov7-tiny` and `yolov7` without complaint; the strict byte count of the other three was not exercised.

## 4. Findings on the `[yolo]` training pass

`src/yolo_layer.c`, `process_batch`, line 388, with `delta_yolo_box` (173), `delta_yolo_class` (308), `compare_yolo_class` (356), `averages_yolo_deltas` (291). The body descends along `-delta`. "Output" below is the layer's inference-mode output of 3.6.

Truth layout: up to `max` rows (default 200) of six floats, `x y w h class track_id`, boxes as fractions of the image. The list ends at the first row whose `x` is 0 (lines 430 and 508), which is why Darknet's loader nudges an `x` of exactly 0 upward (`src/data.c`, `fill_truth_detection`, line 441).

Step 1, every cell and anchor slot (lines 416-503):

- Decode its box. Over the truths find `best_iou`, and `best_match_iou`, the best IoU among truths for which the prediction has some class probability above 0.25. **Correction to SPEC 4.5:** the class test does not look at the truth's class; any class above 0.25 passes.
- Objectness delta `= obj_normalizer * (0 - objectness)`.
- If `best_match_iou > ignore_thresh`: the objectness delta becomes 0, or with `objectness_smooth=1` the larger of itself and `obj_normalizer * (best_match_iou - objectness)`. **Correction:** the ignore rule uses `best_match_iou`, not the plain best IoU, so a prediction with no class above 0.25 is never ignored.
- If `best_iou > truth_thresh` the cell is treated as a positive for that truth. `truth_thresh` is 1 in the cfgs read (`yolov3-tiny`, `yolov4`, `yolov7-tiny`), so this never fires there.

Step 2, every truth (lines 506-636):

- Cell `col = floor(x * w)`, `row = floor(y * h)`.
- Best anchor: among all `num` anchors, the one whose `(pw / net_w, ph / net_h)` has the highest IoU with `(tw, th)`, both centred on the origin; the first wins a tie. If it is in this layer's `mask`, its slot at the cell is assigned.
- With `iou_thresh < 1` (0.213 in v4, 0.2 in v7), every other anchor of this layer's `mask` whose shape IoU with the truth exceeds `iou_thresh` is assigned at the same cell as well. The comparison kind is `iou_thresh_kind`, default IoU.
- For an assigned slot:
  - Box delta, `iou_loss=mse` (v3, v3-tiny), with `k = (2 - tw * th) * iou_normalizer`: `dx = k * (tx - X)`, `dy = k * (ty - Y)`, `dw = k * (tw' - Wr)`, `dh = k * (th' - Hr)`, where `tx = x * w - col`, `ty = y * h - row`, `tw' = log(tw * net_w / pw)`, `th' = log(th * net_h / ph)`. **Correction:** `iou_normalizer` multiplies these terms, and its default is 0.75 (`src/parser.c`, line 469), which the v3-tiny cfg does not override. Checked on the dump: the `x` delta at truth 0's cell equals `1.76 * 0.75 * (0.5 - X)` to the last bit. These deltas add to what the cell already holds.
  - Box delta, other `iou_loss` kinds (`ciou` in v4 and v7): the gradient of the chosen IoU measure with respect to the predicted box, from `src/box.c`, `dx_box_iou` (line 258), times `iou_normalizer`; for `new_coords=0` the `w` and `h` components are also multiplied by `exp(Wr)`, `exp(Hr)`. NaN and infinity become 0, and each component is clipped to `[-max_delta, max_delta]` when `max_delta` is set (2 in v7-tiny). `dx_box_iou` was not read in full; the delta dumps of the v7 models are the check for it. CIoU itself is `IoU - d / c - alpha * v`, `v = (4 / pi^2) * (atan(tw / th) - atan(pw / ph))^2`, `alpha = v / (1 - IoU + v + 0.000001)` (`box_ciou`, line 235).
  - Objectness delta `= obj_normalizer * (1 - objectness)`; with `objectness_smooth=1` it is written only where the delta is still 0.
  - Class deltas: `1 - class_c` for the truth's class and `0 - class_c` for the others, with label smoothing when `label_smooth_eps` is set. If the truth's class delta at that slot is already non-zero (a second truth in the same slot), only that one class is updated.
- Step 3, only with `iou_thresh < 1`: where a slot's objectness delta is non-zero, its four box deltas are divided by the number of classes with a positive delta at that slot (lines 638-653).

What Darknet reports per layer (lines 859-921): `total_loss` is the sum of squared deltas; `class_loss` is `obj_normalizer` times the sum of squared non-box deltas; `iou_loss` is their difference; all three divided by the batch size. The layer's stored cost is the sum of squared deltas for `mse`, and for the IoU kinds `iou_normalizer * mean(1 - IoU) + class_loss`. As SPEC 4.5 says, this cost is a report, not the quantity whose gradient is taken. For `mse`, `-delta` is the gradient of a logit cross-entropy for objectness and classes; for `x`, `y` it is that times `k`; `w`, `h` are half a squared error times `k`.

Not observable from the layer's state: the extra `iou_thresh` assignments and the per-truth IoU. `count` in the log line counts every assignment, so a layer whose `count` exceeds the rows in `assigned_row` had extra ones (`yolov7-tiny` layer 90 has count 1 and no best-anchor row).

## 5. Where the rest of training lives

Not dumped; a second pass while the build exists could add them.

| Quantity | Where |
| --- | --- |
| Truth loading, box clipping, rejection of boxes under one pixel, `track_id` | `src/data.c`, `fill_truth_detection`, line 371; `load_data_detection` for augmentation |
| One training step: forward, backward, cost, update | `src/network.c`, `train_network_datum`, line 361; `forward_network` 269; `backward_network` 338; `get_network_cost` 317; `update_network` 293 |
| Batch-statistics normalisation and rolling update | `src/batchnorm_layer.c`, `forward_batchnorm_layer`, lines 192-208 |
| Convolution update | `src/convolutional_layer.c`, `update_convolutional_layer`, line 1599: the weight update accumulates `-decay * batch * weight`, the weights move by `learning_rate / batch` times the accumulated update, and the update is then scaled by `momentum`; biases and scales the same without decay |
| Memory for training | `src/parser.c`, `parse_network_cfg_custom`, lines 1353-1354: a positive `batch` argument, which `detector test` passes, allocates for inference only. A weight-gradient dump needs the network parsed the way `train_detector` does (`src/detector.c`, line 24) |

## 6. Reference files

Darknet commit `59596d7880f6504768df41d6daa586f5cb2b932f`. All files are under `_darknet/out/<directory>/`; each `.bin` has a `.json` beside it; `dtype` is `float32` everywhere. Statistics were computed with NumPy from the files. `python _darknet/stats.py _darknet/out/<directory>` reprints any table below.

Model files used:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `yolov3-tiny.cfg` | 1 915 | `84eb7a675ef87c906019ff5a6e0effe275d175adb75100dcb47f0727917dc2c7` |
| `yolov3-tiny.weights` | 35 434 956 | `dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c` |
| `yolov7-tiny.cfg` | 7 562 | `9fbd49151721a52d23112717c0d66552363c899befb8920a7c6857f4d9d8ba46` |
| `yolov7-tiny.weights` | 24 967 560 | `06be302b1564f2e2c3e1db1f5ec02477ac05d0ae82405a8c1b2b8e4111101b66` |
| `yolov7.cfg` | 10 766 | `16e66fa7a47b371be3c1f6afb9a8a8bcdad6edf2443b036691868a25f56b0cb2` |
| `yolov7.weights` | 147 898 248 | `4ecf7ca13ec5039ec7b79b0f25b156fda5eaf819d6c2bb6828ba55fe4f928332` |

Five directories: `yolov3_tiny`, `yolov3_tiny_thresh_0p01`, `yolov7_tiny`, `yolov7_tiny_thresh_0p01`, `yolov7_thresh_0p01`. A `_thresh_0p01` directory differs from its sibling only in `detections`; every other file has the same SHA-256. `truth.bin` (72 bytes, `527b9ee0154e5c11760448008bfbddb27b5d5e4b0c8b0152ce5334c062f6f246`) is the same in all five. An empty `detections.bin` has SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

### yolov3_tiny (416 x 416)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `input` | `[3, 416, 416]` | 2076672 | 0 | 1 | 0.499889184 | `088fdab79ea238c8fd84c8e8e1beb8b33fc782cea96b0b337f39c82cb1b11304` |
| `layer_015_yolo_input` | `[255, 13, 13]` | 172380 | -16.698267 | 5.56667948 | -6.97117711 | `b533dbd6e1111561a0d9044a6fb5ba7816f1f06909899b500b5d2550b729e56c` |
| `layer_016_yolo` | `[255, 13, 13]` | 172380 | -1.330724 | 1.14966476 | 0.0218608137 | `86162dbd2136d22764ca038184371beede50ab47d120cdafca66a4de462f0c08` |
| `layer_016_yolo_assigned_class` | `[3, 13, 13]` | 2028 | -1 | 2 | -0.992110454 | `2f06e2c057842022d7ab55cb9136020b5bb061e5ede8b92cb9693af01d9f2cd7` |
| `layer_016_yolo_assigned_row` | `[3, 13, 13]` | 2028 | -1 | 2 | -0.992110454 | `2f06e2c057842022d7ab55cb9136020b5bb061e5ede8b92cb9693af01d9f2cd7` |
| `layer_016_yolo_delta` | `[255, 13, 13]` | 172380 | -0.90544486 | 0.999752641 | -1.32393349e-05 | `7d9099c2fce11d748c07fabad644c023b2af5db58213347afeb1f0bb341a40b3` |
| `layer_022_yolo_input` | `[255, 26, 26]` | 689520 | -16.9854736 | 5.03736782 | -6.62345647 | `8c845b1d4017b435142315f8e5d4c4bae305f38475a449c3ab8e5585f8aecc51` |
| `layer_023_yolo` | `[255, 26, 26]` | 689520 | -1.57172835 | 1.57747066 | 0.0226522634 | `6885f5ae6857824bb6906b1f885c56b4a5e939f8ce4bd24f06c7a4d753a22a80` |
| `layer_023_yolo_assigned_class` | `[3, 26, 26]` | 8112 | -1 | 16 | -0.991617357 | `04485c169df5bcf735660c09edd7be5fa4f84630ae997cb71c7c989da7ca0c3f` |
| `layer_023_yolo_assigned_row` | `[3, 26, 26]` | 8112 | -1 | 1 | -0.999013807 | `1721c1ca9fa8e24ceb4150dfa8fd445b5d376faa0abfc2bac2012ea6085125ae` |
| `layer_023_yolo_delta` | `[255, 26, 26]` | 689520 | -0.23376368 | 0.999951303 | 8.30126834e-06 | `715a5a4cd9316dd7c449eab4565fdb223d2c746c32236cef70910fa81b96186a` |

The two `layer_016` assignment files are identical because truth rows 0 and 2 carry classes 0 and 2.

Largest objectness: 0.127279028 (layer 16), 0.00466298033 (layer 23). At 0.25: 0 detections from 0 candidates, `nms` 0.45, `nms_kind` 0.

Assignment: truth row 2 at layer 16, slot 0, cell row 9, column 9; row 0 at layer 16, slot 1, cell (6, 6); row 1 at layer 23, slot 2, cell row 7, column 6. Stored cost 6.79358196 (675 non-zero deltas) and 2.28762364 (2112).

```
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 16 Avg (IOU: 0.647415), count: 2, class_loss = 5.753386, iou_loss = 1.040196, total_loss = 6.793582
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 23 Avg (IOU: 0.821256), count: 1, class_loss = 2.066177, iou_loss = 0.221447, total_loss = 2.287624
```

`yolov3_tiny_thresh_0p01/detections`: `[16, 7]`, 448 bytes, SHA-256 `768e20cf8ccb2759afc6f0c1ae035e0cc5380b9a0814c87157e5171364bfcb79`, 16 rows from 58 candidates (`x y w h objectness score class`):

```
0.420056134 0.575345993 1.02013958 0.886559129 0.0621844605 0.0161716435 59
0.341266513 0.572691441 1.11309183 0.95305562 0.0599434897 0.0105364611 57
0.342386633 0.342360258 0.988622904 0.8944543 0.0832582861 0.0435806774 28
0.502193511 0.575005352 0.940744877 0.821747005 0.063418664 0.0149255022 28
0.647601902 0.575507402 0.768234432 0.722933948 0.111623786 0.0193525963 27
0.647601902 0.575507402 0.768234432 0.722933948 0.111623786 0.0100756362 56
0.647601902 0.575507402 0.768234432 0.722933948 0.111623786 0.0234565679 57
0.579034507 0.344971269 0.928181887 0.8257128 0.113598026 0.0102171199 59
0.580825746 0.284031481 0.868584335 0.598739982 0.127279028 0.0283768121 15
0.580825746 0.284031481 0.868584335 0.598739982 0.127279028 0.0581324026 25
0.580825746 0.284031481 0.868584335 0.598739982 0.127279028 0.0146738943 27
0.580825746 0.284031481 0.868584335 0.598739982 0.127279028 0.0143221384 57
0.725799799 0.275734574 0.423033834 0.459980875 0.0229510609 0.020980306 27
0.653043389 0.581866443 0.513712049 0.422783643 0.0235172678 0.0211947914 27
0.424147069 0.354057461 0.301349074 0.484948874 0.0734061375 0.0226651467 15
0.424147069 0.354057461 0.301349074 0.484948874 0.0734061375 0.065312162 27
```

### yolov7_tiny (416 x 416, same `input` as above)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_089_yolo_input` | `[255, 52, 52]` | 2758080 | 6.93933515e-08 | 0.948225021 | 0.0364591207 | `4c04055f50409bceb5f64d913261173e4ea394c77eb3d22ae862a6c55bfd411a` |
| `layer_090_yolo` | `[255, 52, 52]` | 2758080 | -0.193316162 | 1.26769745 | 0.0368043951 | `6fc024da8453a9715c3dd1f11b120110f4f921ce372de06e73a74a252474638b` |
| `layer_090_yolo_assigned_class` | `[3, 52, 52]` | 32448 | -1 | -1 | -1 | `db85d9e39fcaa6fee92cbb718c1078ca4dc5614dacbe04966f906e098650c0c8` |
| `layer_090_yolo_assigned_row` | `[3, 52, 52]` | 32448 | -1 | -1 | -1 | `db85d9e39fcaa6fee92cbb718c1078ca4dc5614dacbe04966f906e098650c0c8` |
| `layer_090_yolo_delta` | `[255, 52, 52]` | 2758080 | -0.474152893 | 0.995678842 | -4.35537813e-07 | `70d8c84776d6167e791399a22948116a3495d380cf0333c693a4ca9a67868424` |
| `layer_093_yolo_input` | `[255, 26, 26]` | 689520 | 1.22107195e-07 | 0.930993378 | 0.0366641243 | `fe9dd886b76b7493a7f1fecc61446fabcfcb08e21d50bf454df9eca2b2029bff` |
| `layer_094_yolo` | `[255, 26, 26]` | 689520 | -0.33257103 | 1.31507206 | 0.0370520925 | `2075b49fd365fd0c32ab8cc61584afcea5fb111d8e4943a79ab463d1f902c97a` |
| `layer_094_yolo_assigned_class` | `[3, 26, 26]` | 8112 | -1 | 16 | -0.991617357 | `ffcae4b178553618daa1e154a05821aa59d54b3ce2c958b7a0fab8e6349f4536` |
| `layer_094_yolo_assigned_row` | `[3, 26, 26]` | 8112 | -1 | 1 | -0.999013807 | `f13a744b14eef3b46b263eb4664f9f30da6ac67c19abc6b9b6128bfc557bd848` |
| `layer_094_yolo_delta` | `[255, 26, 26]` | 689520 | -0.474997789 | 0.997657478 | -8.17161997e-06 | `62465483047dfcf4e1e627e9ccf44a2c41c061ddcb8543ddfca2135d60cc072b` |
| `layer_097_yolo_input` | `[255, 13, 13]` | 172380 | 9.64712488e-09 | 0.930199385 | 0.0298295907 | `7fc82c8e12b39703cd6567e4b03bb4c1d9a026e9396accb9aeac9d6e6a517cb4` |
| `layer_098_yolo` | `[255, 13, 13]` | 172380 | -0.301275015 | 1.31922197 | 0.0301769127 | `314e8ad160cab91e82ee634e609021d3d9950da540a2bdebf9a525c000b81128` |
| `layer_098_yolo_assigned_class`, `_assigned_row` | `[3, 13, 13]` | 2028 | -1 | 2 | -0.992110454 | `2f06e2c057842022d7ab55cb9136020b5bb061e5ede8b92cb9693af01d9f2cd7` |
| `layer_098_yolo_delta` | `[255, 13, 13]` | 172380 | -0.657733679 | 0.999972403 | -0.000106230028 | `3fa653340fedafe977315889509447180fcd27f535dad49f5610edabe662e960` |

Largest objectness 0.0158627331, 0.0358091816, 0.248599663 (layers 90, 94, 98). At 0.25: 0 detections, `nms_kind` 2, `beta_nms` 0.6.

Assignment by best anchor: truth row 1 at layer 94, slot 0, cell row 7, column 6; row 2 at layer 98, slot 0, cell (9, 9); row 0 at layer 98, slot 1, cell (6, 6). Stored cost 1.05615366 (8196 non-zero deltas), 5.60516644 (2448), 7.62700224 (1011).

```
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 90 Avg (IOU: 0.471687), count: 1, class_loss = 1.029738, iou_loss = 0.491261, total_loss = 1.520999
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 94 Avg (IOU: 0.264290), count: 5, class_loss = 5.568381, iou_loss = 0.252303, total_loss = 5.820684
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 98 Avg (IOU: 0.264358), count: 6, class_loss = 7.590220, iou_loss = 0.022136, total_loss = 7.612356
```

`yolov7_tiny_thresh_0p01/detections`: `[11, 7]`, 308 bytes, SHA-256 `5720d4d13a541afc73bdd24ff406c4ef4e9a1d62cf3c6e0385790fe13ced039e`, 11 rows from 207 candidates:

```
0.484462351 0.299234062 0.923145652 0.608270526 0.0513224974 0.0191556774 62
0.49363178 0.515957534 0.849302888 0.975347102 0.0307267308 0.0105911288 13
0.78684783 0.692279398 0.427874923 0.50447917 0.0228852537 0.0156961996 15
0.0976252556 0.834269047 0.195340812 0.328508109 0.0281284787 0.0106194802 58
0.0816311687 0.944398284 0.163983271 0.11122293 0.0355440788 0.0223833993 28
0.668080628 0.657954037 0.324686229 0.395360231 0.026394248 0.0230225753 15
0.283455223 0.506075263 0.587219477 0.977767348 0.038107615 0.0237753559 13
0.464676559 0.418760419 0.300290048 0.391841292 0.198583618 0.02720741 28
0.369089127 0.365083277 0.269979537 0.345720798 0.055168014 0.010251685 0
0.283446789 0.718973458 0.57227242 0.556832016 0.0496511571 0.0287045874 13
0.363912374 0.734753251 0.447876632 0.514376879 0.0951366574 0.0102672065 56
```

### yolov7_thresh_0p01 (640 x 640, `-thresh 0.01`)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `detections` | `[7, 7]` | 196 | 0.0106335348 | 79 | 8.3291825 | `06be6657ce727f8c42cffb632261c91cdbf4d4957fa401b90b80b1d8984cd3b7` |
| `input` | `[3, 640, 640]` | 4915200 | 0 | 1 | 0.499765533 | `2c585b8211e8464048a158f8f6586bbec3d1370e76fd4c9724ac2bd429d0b533` |
| `layer_133_yolo_input` | `[255, 80, 80]` | 6528000 | 4.13770929e-09 | 0.975986719 | 0.0352218867 | `67aa9477a4c47da8c77fd4c382365657aa21ed239f63aab4a12ba16b77bdf608` |
| `layer_134_yolo` | `[255, 80, 80]` | 6528000 | -0.281233251 | 1.20360529 | 0.0351824163 | `f3b90cc9cf2b6110d632186987410c489f57e3bd62103b3974cb3c1bbb2c103a` |
| `layer_134_yolo_assigned_class`, `_assigned_row` | `[3, 80, 80]` | 76800 | -1 | -1 | -1 | `ad9a808a8c96d9648d089638cba4972dc5bf0dd77e5cb41fde072a353ebe6b27` |
| `layer_134_yolo_delta` | `[255, 80, 80]` | 6528000 | -0.0207956042 | 0 | -2.41969285e-07 | `b4665734084a3bb5e30f73fb9d26e2d320cfa1f9ad3b51291724b9dab866538e` |
| `layer_137_yolo_input` | `[255, 40, 40]` | 1632000 | 2.35281172e-09 | 0.958701372 | 0.0334636755 | `a2686efe7af6f108307278e195ad651a9a9c696433eefd4a63e96df6530e188a` |
| `layer_138_yolo` | `[255, 40, 40]` | 1632000 | -0.308100522 | 1.39979637 | 0.033228783 | `d48b3825653559aa927b33c461ad9ead3d507a8220fb25ec707bca6c22b959c1` |
| `layer_138_yolo_assigned_class` | `[3, 40, 40]` | 19200 | -1 | 16 | -0.996458333 | `3aff8c223835f11f7a73187094c3dd48a0eac73fe71c43ee41db6a9c7d74355b` |
| `layer_138_yolo_assigned_row` | `[3, 40, 40]` | 19200 | -1 | 1 | -0.999583333 | `966e429efc748143baa2742d4bd0a348b784b4d15c69714debd12089db97759e` |
| `layer_138_yolo_delta` | `[255, 40, 40]` | 1632000 | -0.396278352 | 0.99321717 | -1.94449626e-06 | `ca7617769716987feb2360d1099812a516fe519c88035add7ee62f1f7a7a8300` |
| `layer_141_yolo_input` | `[255, 20, 20]` | 408000 | 2.29633379e-08 | 0.976390004 | 0.0343422141 | `afc57043f67b47bf25500d35a5fe2da2bf2e13c31fe0a7c1a3f4ad8741d334b6` |
| `layer_142_yolo` | `[255, 20, 20]` | 408000 | -0.410442531 | 1.18942904 | 0.0344993762 | `5e2a37705ab1dd47cb31f73f073a7337f416480fee65bb8de88e28ccdc77524e` |
| `layer_142_yolo_assigned_class`, `_assigned_row` | `[3, 20, 20]` | 4800 | -1 | 2 | -0.996666667 | `0bdc2a42e17c757274bef97f1b36cf6605aaa37a3dec466b5e8814c51b6faad2` |
| `layer_142_yolo_delta` | `[255, 20, 20]` | 408000 | -0.858558595 | 0.998154521 | -1.24195384e-05 | `14d9945eb92fe118b8fffe0789f2f3d2ce43ca4e452cae5e54a8da275ef51f30` |

7 rows from 54 candidates, `nms_kind` 2, `beta_nms` 0.6:

```
0.0685434192 0.943559766 0.134694159 0.109933689 0.0328350626 0.0142680733 79
0.0400596224 0.961658001 0.078247413 0.0755259842 0.0245890375 0.011679057 76
0.4979662 0.490348548 0.98757422 0.968777835 0.0531930067 0.0143189467 15
0.0773411766 0.940759003 0.154317006 0.116366945 0.0262436122 0.0130589623 76
0.499173075 0.501578152 0.987037718 0.987164259 0.0514149778 0.0118046524 13
0.025706891 0.972356915 0.0498760566 0.0536170192 0.0235623494 0.0144317579 76
0.0220777523 0.976733029 0.0432271138 0.0452785082 0.0184099488 0.0106335348 61
```

Assignment by best anchor: truth row 1 at layer 138, slot 2, cell row 12, column 10; row 2 at layer 142, slot 0, cell row 15, column 14; row 0 at layer 142, slot 2, cell (10, 10). Stored cost 0.00362131861 (19200 non-zero deltas), 4.08578777 (5136), 6.75602961 (1620).

```
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 134 Avg (IOU: 0.000000), count: 1, class_loss = 0.003621, iou_loss = 0.000000, total_loss = 0.003621
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 138 Avg (IOU: 0.363589), count: 4, class_loss = 4.053967, iou_loss = 0.421772, total_loss = 4.475739
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 142 Avg (IOU: 0.207137), count: 5, class_loss = 6.716386, iou_loss = 0.013420, total_loss = 6.729806
```

## 7. Not done

- Full-body dumps exist for `yolov3-tiny` only. Section 9 gives the size each of the other five would have.
- The dumps are raw files, not SACP, and no manifest in the fixture format was written. `_darknet/out/SHA256SUMS.txt` lists every file's hash.
- The training step of section 13 was run for `yolov3-tiny` and `yolov7-tiny` only, and stops before the weight update. The update rule is described in section 5 from the source and was not exercised.
- `mish` was exercised by the `yolov4` run but not compared against values computed outside Darknet.
- The CIoU delta was reproduced to 1e-5 relative, not to the bit (section 11).
- The GPU path of the convolution backward was not traced; section 13 names the one place where it visibly differs from the CPU path.
- The `(confirm)` marks of SPEC about `region`, `detection`, `reorg`, `local`, `connected`, `avgpool` and pjreddie's defaults were left: v1 and v2 are out of scope and pjreddie's source is not on disk.

## 8. Second pass: what the patch gained

The patch is still one file, `src/detector.c`, now 362 diff lines (319 added). Three additions:

| Switch | Effect |
| --- | --- |
| `DARKNET_DUMP_ALL_LAYERS` set | Writes the output of every layer except the `[yolo]` layers as `layer_<k>_<type>`, `[C, h, w]`, with `type` one of `convolutional`, `maxpool`, `route`, `upsample`, `shortcut`. Darknet's own name table has no entry for the upsample layer, so the patch names it |
| always, with `DARKNET_DUMP_DIR` | Per `[yolo]` layer `k`: `layer_<k>_yolo_boxes` `[h*w*A, 4]`, `layer_<k>_yolo_objectness` `[h*w*A, 1]`, `layer_<k>_yolo_scores` `[h*w*A, C]` |
| `DARKNET_TRAIN_STEP` set | Replaces the inference run by one training-mode forward and backward (section 13) |

With the new switches unset, a run writes the earlier files byte for byte: all 160 `.bin` and `.json` files of the first pass were hashed before the rerun and checked after it, with no difference.

Decoded rows. Row index `(row * w + column) * A + anchor`, the order of SPEC 5.1. Boxes are `x_center y_center width height` as fractions of the network input; scores are `objectness * class probability`, not the class probability alone (that is in `layer_<k>_yolo`). They come from Darknet's own decode, called with a threshold of -1 so that no row and no score is dropped. Checked against section 3.7 from `layer_<k>_yolo`: boxes agree to 6e-8 (`yolov3-tiny` layer 16) and 1.2e-7 (`yolov7-tiny` layer 98), objectness is equal, scores are equal.

A correction to section 4: the `count` in Darknet's loss line is forced to 1 when no truth was assigned, so `count: 1` with `Avg (IOU: 0.000000)` means none (`yolov3` layer 106, `yolov4` layer 139, `yolov7` layer 134).

A correction to section 1: Darknet's CMake adds `/fp:fast` for MSVC (`CMakeLists.txt`, lines 285-286; the GCC and Clang branch uses `-Ofast`, line 299). The compiler may reorder sums and keep wider intermediates. The reference tensors are what this build computed; a from-scratch implementation can match them to a tolerance, not to the bit, wherever a result is a difference of close numbers.

## 9. Weights files and body sizes

Float count the cfg requires (per convolution: `n` biases, `3n` more with batch norm, `n * (c / groups) * size^2` weights), against the file, counted by a walker written for this report (`_darknet/cfg_walk.py`) and not by Darknet:

| Model | Layers | Floats the cfg needs | File | Full-body dump would be |
| --- | --- | --- | --- | --- |
| `yolov3-tiny` | 24 | 8 858 734 | exact | 32 787 352 bytes (written; the files sum to this) |
| `yolov3` | 107 | 62 001 757 | exact | 227 886 360 bytes |
| `yolov4-tiny` | 38 | 6 062 814 | exact | 47 151 000 bytes |
| `yolov4` | 162 | 64 429 405 | exact | 692 320 024 bytes |
| `yolov7-tiny` | 99 | 6 241 885 | exact | 72 222 488 bytes |
| `yolov7` | 143 | 36 974 557 | exact | 602 864 000 bytes |

"Exact" means `20 + 4 * floats` equals the file size. Body sizes are at the cfg's own input size (416, 416, 416, 608, 416, 640).

## 10. `[shortcut]` in the six cfgs

Every `[shortcut]` in `yolov3` (23) and `yolov4` (23) has one `from`, no `weights_type`, and a source whose shape equals the current one. The other four cfgs have none. The general routine of section 3.5 is never reached; the plain element-wise sum is all that is needed.

## 11. The CIoU box gradient

`src/box.c`, `dx_box_iou`, line 258, and its use in `src/yolo_layer.c`, `delta_yolo_box`, lines 191-290. Predicted box `(x, y, w, h)`, truth `(X, Y, W, H)`, both centre and size as fractions of the input. Edges `l = x - w/2`, `r = x + w/2`, `t = y - h/2`, `b = y + h/2`, capitals for the truth.

Overlap part. `A = (b - t)(r - l)`, `A' = (B - T)(R - L)`, `Iw = min(r, R) - max(l, L)`, `Ih = min(b, B) - max(t, T)`, `I = Iw * Ih`, `U = A + A' - I`. For each edge `e`:

```
dA/dt = -(r - l)    dA/db = r - l      dA/dl = -(b - t)    dA/dr = b - t
dI/dt = -Iw if t > T else 0            dI/db = Iw if b < B else 0
dI/dl = -Ih if l > L else 0            dI/dr = Ih if r < R else 0
g_e = (U * dI/de - I * (dA/de - dI/de)) / U^2        (0 when U <= 0)
Gx = g_l + g_r      Gy = g_t + g_b      Gw = g_r - g_l      Gh = g_b - g_t
```

Distance part. `S = (x - X)^2 + (y - Y)^2`, `Cw = max(r, R) - min(l, L)`, `Ch = max(b, B) - min(t, T)`, `C = Cw^2 + Ch^2`. With indicators `Rx = [r > R]`, `Lx = [l < L]`, `By = [b > B]`, `Ty = [t < T]`:

```
Dx = (2 (X - x) C - 2 Cw (Rx - Lx) S) / C^2        Dy = (2 (Y - y) C - 2 Ch (By - Ty) S) / C^2
Dw = Cw (Rx + Lx) S / C^2                          Dh = Ch (By + Ty) S / C^2
```

Aspect part. `theta = atan(W / H) - atan(w / h)`, `v = 4 theta^2 / pi^2`, `alpha = v / (1 - I / U + v + 0.000001)`:

```
Vw = alpha * (8 / pi^2) * theta * h                Vh = -alpha * (8 / pi^2) * theta * w
```

Result. If `C > 0`: `(Gx + Dx, Gy + Dy, Gw + Dw + Vw, Gh + Dh + Vh)`. If the boxes do not overlap (`Iw <= 0` or `Ih <= 0`) the overlap part is dropped and the result is `(Dx, Dy, Dw + Vw, Dh + Vh)`.

This is not the analytic derivative of CIoU, in three places, and an automatic derivative will not reproduce Darknet's deltas:

- `Gw`, `Gh` are twice the derivative with respect to width and height (no factor 1/2).
- In `Dx`, `Dy` the enclosing-box term has the opposite sign of the derivative of `-S / C`.
- `Vw`, `Vh` lack the factor `1 / (w^2 + h^2)`, and `alpha` is treated as a constant.

Chain to the layer's entries (`delta_yolo_box`):

- `x`, `y`: none. The result is used as the delta of the activated entry; neither the grid size nor `scale_x_y` nor a logistic slope is applied in the layer. For `new_coords=1` the logistic slope `out * (1 - out)` is applied afterwards by the backward step of the convolution before the layer, as for every entry.
- `w`, `h`, `new_coords=0`: multiplied by `exp(Wr)`, `exp(Hr)`.
- `w`, `h`, `new_coords=1`: none.
- Then each of the four is multiplied by `iou_normalizer`, a NaN or infinity becomes 0, each is clipped to `[-max_delta, max_delta]` when `max_delta` is set, and it is added to the cell's delta. The factor `2 - W * H` of the `mse` branch is not used here. With `iou_thresh < 1` the four are finally divided by the number of positive class deltas of the slot (section 4, step 3).
- `max_delta` is 2 in both v7 cfgs and unset in the v4 cfgs: `yolov4` layer 150 has a box delta of 1.92664182.

Check on `yolov7-tiny` (`_darknet/check_ciou.py`, `float32` arithmetic in the order written above). Dumped against recomputed:

| Cell | Entry | Dumped | Recomputed | Relative difference |
| --- | --- | --- | --- | --- |
| layer 98, slot 1, cell (6, 6), truth 0, IoU 0.313492 | x | 0.000183408672 | 0.000183410593 | 1.0e-5 |
| | y | -0.00106176059 | -0.00106175675 | 3.6e-6 |
| | w | -0.0385336503 | -0.0385336541 | 9.7e-8 (1 ulp) |
| | h | -0.0332950354 | -0.0332950316 | 1.1e-7 (1 ulp) |
| layer 98, slot 0, cell (9, 9), truth 2, IoU 0.224087 | x | -0.0112515502 | -0.011251552 | 1.7e-7 |
| | y | 0.0061833025 | 0.0061833146 | 2.0e-6 |
| | w | -0.04532497 | -0.0453249812 | 2.5e-7 |
| | h | -0.0397488885 | -0.0397489034 | 3.7e-7 |
| layer 94, slot 0, cell row 7, column 6, truth 1, IoU 0.230543 | x | -0.00749072898 | -0.0074907057 | 3.1e-6 |
| | y | 0.00453934958 | 0.00453935005 | 1.0e-7 |
| | w | -0.097358264 | -0.097358264 | bit-equal |
| | h | -0.0835391209 | -0.0835391209 | bit-equal |

The same computation with double intermediates differs by a similar amount in other entries, so neither precision reproduces the bits. The cause is `/fp:fast` (section 8): `x` and `y` are sums of terms that nearly cancel. The `mse` check of section 4 was bit-equal because that expression has no cancellation. A parity test on CIoU deltas needs a relative tolerance near 1e-4 on the box entries, or Darknet rebuilt with `/fp:precise`, which would change every reference tensor already taken.

## 12. Summary of the three new forward runs

| Model | Largest objectness per head | Rows at 0.25 | Rows at 0.01 | `nms_kind` |
| --- | --- | --- | --- | --- |
| `yolov3` | 0.999769509, 0.00034218264, 0.000142286328 | 2 of 6 candidates | 3 of 9 | 0 |
| `yolov4-tiny` | 0.592214584, 0.00383588765 | 0 of 11 | 16 of 37 | 1 |
| `yolov4` | 0.00665664487, 0.00934946537, 0.943473876 | 1 of 1 | 5 of 5 | 1 |

`yolov3` and `yolov4` do report an object on the synthetic pattern at the default threshold, so their default-threshold directories exercise the whole post-processing.

## 13. One training step under batch statistics

How it was run. The network is parsed the way training parses it (memory for training, batch from the cfg), from a copy of the cfg in which only the first `batch=` line is changed to 2 and `subdivisions` to 1. The weights are loaded, the batch and the truth are fed, and Darknet's forward and backward over the whole network are called once in training mode. The weight update is not called, so every `*_updates` tensor is the accumulated update of this one batch and nothing else.

Batch size. With batch 1 the statistics are still taken over all `h * w` positions of a channel, so batch norm is not degenerate for these networks (the smallest map is 13 by 13). It is degenerate only for a 1 by 1 map, where the variance divides by zero. Batch 2 was used so that the batch axis takes part in the mean, in the `batch * spatial - 1` divisor and in the per-image truth offset.

Recipe (`_darknet/make_train_batch.py`); image 0 is the image of section 2.1:

```python
first  = ((x * x + 3 * y * y + 7 * x * y) // 16 + 50 * c) % 256
second = ((3 * x * x + y * y + 5 * x * y) // 16 + 90 * c) % 256      # same meshgrid as section 2.1
batch = (np.stack([first, second]).astype(np.float32) / np.float32(255)).astype("<f4")   # [2, 3, 416, 416]
truth = np.array([(0, 0.50, 0.50, 0.40, 0.60, 0), (0, 0.25, 0.30, 0.10, 0.15, 16), (0, 0.70, 0.75, 0.30, 0.20, 2),
                  (1, 0.30, 0.60, 0.20, 0.25, 7), (1, 0.65, 0.35, 0.50, 0.45, 41)], dtype="<f4")  # image x y w h class
```

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `train_input_2x416x416.bin` | 4 153 344 | `7b7563a106509747180c421f8fd1be6cec581bcf8aaeed22274976adbb383036` |
| `train_truth_5_boxes.bin` | 120 | `89cec5c9fccba5b3f630907f27ee13e58a6fe59decb892aa2a42d739df21f994` |
| `input_608x608.bin` (section 2.1 recipe) | 4 435 968 | `fd32ed64373eff75ed1709262ee9e00a04008b96b901cd758a5f438e4f6dd3d5` |

Layout. Feature maps are `[N, C, h, w]` with `"order": "NCHW"`; weight updates are `[out, in, kh, kw]` with `"order": "OIHW"`; vectors are `[1, n]`.

| Name | Content |
| --- | --- |
| `input`, `truth` | The batch and the truth rows as read |
| `rolling_mean_before`, `rolling_variance_before`, `..._after` | Rolling statistics of the first batch norm (layer 0) around the step |
| `batch_mean`, `batch_variance` | That layer's statistics of this batch |
| `layer_000_output` | Layer 0 after batch norm and activation |
| `layer_000_weight_updates`, `_scale_updates`, `_bias_updates` | Updates of the first convolution |
| `layer_<k-2>_output` | Input of the last convolution before `[yolo]` layer `k` |
| `layer_<k-1>_yolo_input`, `layer_<k>_yolo`, `layer_<k>_yolo_delta` | Head under batch statistics, the layer's output, its delta |
| `layer_<k-1>_weight_updates`, `_bias_updates` | Updates of the last convolution before the head |
| `layer_<k>_yolo_cost`, `network_cost` | The layer's stored cost; the mean of the layers' costs |

What the dumps confirm, with the measured error (`_darknet/check_train_step.py`):

- Batch statistics (`src/blas.c`, `mean_cpu`, line 253, and `variance_cpu`, line 269): mean over batch and positions; variance divides by `batch * h * w - 1`.
- `rolling_after = 0.9 * rolling_before + 0.1 * batch`: largest deviation 3.6e-8 (mean) and 3.0e-8 (variance) for `yolov3-tiny`, 2.7e-8 and 2.7e-9 for `yolov7-tiny`.
- An update is the descent direction summed over the batch, not divided by it: for the last convolution before a head, `weight_updates[f, c] = sum over images and positions of d[f] * input[c]` and `bias_updates[f] = sum of d[f]`, where `d` is the `[yolo]` delta times the derivative of the convolution's own activation (1 for `linear`, `out * (1 - out)` for `logistic`, `src/activations.h` line 103). Largest deviation: `yolov3-tiny` 7.7e-7 and 2.4e-6 on weights (largest entry 7.08 and 4.44), 1.2e-7 on biases; `yolov7-tiny`, with the logistic factor, 1.1e-7, 2.8e-7, 2.8e-7 on weights.
- The stored cost of an `mse` layer is the sum of squared deltas over the whole batch (12.0947657 against 12.0947686); the printed `total_loss` is that divided by the batch (6.047383). The network cost is the plain mean of the `[yolo]` layers' costs (`src/network.c`, `get_network_cost`, line 317): 7.64737082 and 10.2394142.
- **`layer_000_bias_updates` is exactly zero in both models.** On the CPU path a convolution with batch norm runs the batch-norm backward, which accumulates the scale update and transforms the delta, and never accumulates a bias update (`src/convolutional_layer.c`, `backward_convolutional_layer`, lines 1547-1552; `src/batchnorm_layer.c`, `backward_batchnorm_layer`, line 211). The GPU path has a bias accumulation at `src/convolutional_kernels.cu` line 676 that was not traced. A CPU-built Darknet therefore never trains the shift of a batch-normalised convolution. For SushiAI the mathematical gradient is `sum of delta`; this dump must not be taken as the reference for that one tensor.
- The batch-norm backward (`backward_scale_cpu` line 91, `mean_delta_cpu` 106, `variance_delta_cpu`, `normalize_delta_cpu` 136) was not checked numerically; `layer_000_weight_updates` and `layer_000_scale_updates` are there for that check.

## 14. Reference files of the second pass

Same commit, same conventions as section 6. `_darknet/out/SHA256SUMS.txt` holds the hash of all 728 files. A `_thresh_0p01` directory differs from its sibling only in `detections`.

Model files added:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `yolov3.cfg` | 8 342 | `370c7b08d2350fd105a79af9b106c02217d659cd2a5c1e8319a65fa4ced610fa` |
| `yolov3.weights` | 248 007 048 | `523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297` |
| `yolov4-tiny.cfg` | 3 231 | `f858e3724962eedf3ac44e3b6cb3f0c3d9ed067c306bb831f539c578b924c90e` |
| `yolov4-tiny.weights` | 24 251 276 | `cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87` |
| `yolov4.cfg` | 12 231 | `a6d0f8e5c62cc8378384f75a8159b95fa2964d4162e33351b00ac82e0fc46a34` |
| `yolov4.weights` | 257 717 640 | `e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562` |
| `yolov3-tiny_batch_2.cfg` (derived) | 1 915 | `7021f30bef565b4efd66cabe55d310dae784e68de57ccca405e693862703f80b` |
| `yolov7-tiny_batch_2.cfg` (derived) | 7 561 | `6106ea34a43924c8e9e672399232e4906245d3eb07cf6fac083399d9c66929c8` |

### yolov3_tiny: every layer, and the decoded rows

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_000_convolutional` | `[16, 416, 416]` | 11075584 | -5.12612486 | 49.8875923 | 1.85553978 | `c74a14e82104c742c615a312cb92a89751950c11ab7cb08d9616e6f8e1bd3526` |
| `layer_001_maxpool` | `[16, 208, 208]` | 2768896 | -3.20889711 | 49.8875923 | 4.96929134 | `7e7a3fe1073366bd0a870b9ad62356ca2ba657a3af1c872337a1a2e742b736e3` |
| `layer_002_convolutional` | `[32, 208, 208]` | 5537792 | -7.81382084 | 34.4665642 | -0.10823329 | `daf6c7fa13db692c68296c9fa02240997dae46c3e581668279c8ace981972a0b` |
| `layer_003_maxpool` | `[32, 104, 104]` | 1384448 | -5.80116892 | 34.4665642 | 0.785221536 | `f19d186a00e49625792e5c6001483ba0136b40e502dcade095305f76e7415618` |
| `layer_004_convolutional` | `[64, 104, 104]` | 2768896 | -5.44199038 | 31.6995506 | 0.636704788 | `192bf5233361dda2bae2bf20b3cc898200edf402b68efea19290b3194e66ae10` |
| `layer_005_maxpool` | `[64, 52, 52]` | 692224 | -4.90372467 | 31.6995506 | 1.21208995 | `c28ef119015e4a8250d5d50d16a83976c501272fb45ef7fe4e141fad9554cb67` |
| `layer_006_convolutional` | `[128, 52, 52]` | 1384448 | -4.42487907 | 20.0595398 | 0.145193505 | `a5a9de5b76cce008abb09d53c8a834bb308404e80a43159b38fd3e51fe7c9299` |
| `layer_007_maxpool` | `[128, 26, 26]` | 346112 | -3.69701743 | 20.0595398 | 0.512742079 | `42e8afbd84fc2439d320e22595a1aa520b270a3e15f407e1bc470e407b3f7c1d` |
| `layer_008_convolutional` | `[256, 26, 26]` | 692224 | -2.50564027 | 18.4442711 | -0.0976192959 | `c1322d26652ceb6256e6f73d61db140f7943a0a7c34a10d68286955f77f5fb5f` |
| `layer_009_maxpool` | `[256, 13, 13]` | 173056 | -1.97340417 | 18.4442711 | 0.191322641 | `692ac1a5569a473ad9f5b04a74ea285df38f7cde016a289a6a37b5d8d685aea1` |
| `layer_010_convolutional` | `[512, 13, 13]` | 346112 | -1.91092908 | 6.15734959 | -0.154410784 | `612bb031542fb3b5f6597c7e8c890746056751c7fb929694a45fe47a8ca491ac` |
| `layer_011_maxpool` | `[512, 13, 13]` | 346112 | -1.61518312 | 6.15734959 | -0.0145619142 | `d9a5a94b8105a46ee8be0440e498d34a4a46e8b42e800c62a6157ff20770c287` |
| `layer_012_convolutional` | `[1024, 13, 13]` | 692224 | -2.97461081 | 15.7055035 | 0.0280390475 | `39d4671627ea819f5d9faf4a12fd8c79f8591c3e30def2823223ab5caab5536f` |
| `layer_013_convolutional` | `[256, 13, 13]` | 173056 | -0.526361883 | 3.48680878 | 0.168939153 | `5f1ae03502cc7502fa60837972a6b393fda8bdc5677223e626ac6fb267439d3c` |
| `layer_014_convolutional` | `[512, 13, 13]` | 346112 | -0.793259561 | 7.03454971 | 0.371155426 | `617d1307aad513b4e099df2674edf52b309a26eb596c2d175792db04fc38e005` |
| `layer_015_convolutional` | `[255, 13, 13]` | 172380 | -16.698267 | 5.56667948 | -6.97117711 | `b533dbd6e1111561a0d9044a6fb5ba7816f1f06909899b500b5d2550b729e56c` |
| `layer_016_yolo_boxes` | `[507, 4]` | 8112 | 0.0364500508 | 1.3367275 | 0.475601366 | `f972cd68740ff7796bd8df52d2f2f2d7ba1cbf28d200f878ecbe450032955b96` |
| `layer_016_yolo_objectness` | `[507, 1]` | 2028 | 5.59802338e-08 | 0.127279028 | 0.00590089057 | `e354046af3a93cb0107ae1dc0647715c714631aefe2c1d07b493e9f269d830b0` |
| `layer_016_yolo_scores` | `[507, 80]` | 162240 | 3.09522785e-12 | 0.065312162 | 7.16944489e-05 | `33073de722f7ba74354586a174ed9f7ac9c664fdbeefb19fa59ff15b93147f29` |
| `layer_017_route` | `[256, 13, 13]` | 173056 | -0.526361883 | 3.48680878 | 0.168939153 | `5f1ae03502cc7502fa60837972a6b393fda8bdc5677223e626ac6fb267439d3c` |
| `layer_018_convolutional` | `[128, 13, 13]` | 86528 | -0.424660504 | 5.05288029 | 0.509498737 | `d089fc18ea34e26d81da1a874574973786a234d94e81cf2eebf23ada8ac3a09c` |
| `layer_019_upsample` | `[128, 26, 26]` | 346112 | -0.424660504 | 5.05288029 | 0.509498737 | `cd42e750dbfa2eaeadb5bf0e1bf4d0c2f3be960101d44c78227e05684cd02f26` |
| `layer_020_route` | `[384, 26, 26]` | 1038336 | -2.50564027 | 18.4442711 | 0.104753382 | `c93e35f302f3fe9dfa610bd9f7acb48375db2f0c31a469593dbeecdb6e6e56a3` |
| `layer_021_convolutional` | `[256, 26, 26]` | 692224 | -0.721319199 | 7.86844254 | 0.525727081 | `d72e7bb4cd6fb77545ab0ba2ffd766405d001f3c07350f6e6670c9c9a75a382b` |
| `layer_022_convolutional` | `[255, 26, 26]` | 689520 | -16.9854736 | 5.03736782 | -6.62345647 | `8c845b1d4017b435142315f8e5d4c4bae305f38475a449c3ab8e5585f8aecc51` |
| `layer_023_yolo_boxes` | `[2028, 4]` | 32448 | 0.00520833116 | 0.987848163 | 0.28685676 | `78f1236a97738ad7708ce82179cc841cf55e056e96c3665a10f3b233b76e9bac` |
| `layer_023_yolo_objectness` | `[2028, 1]` | 8112 | 4.20051514e-08 | 0.00466298033 | 0.000100649463 | `9d4a51ad61bd6b90437fe7189df66eb5a3bbdbdd235e44668fdcbcc738ca09d6` |
| `layer_023_yolo_scores` | `[2028, 80]` | 648960 | 5.49305881e-13 | 0.00180915464 | 1.06793148e-06 | `6354cca41510c1921c7d440ff4742d92845458f0929c457203013a54760751f0` |

`layer_015_convolutional` and `layer_022_convolutional` are the same bytes as the earlier `layer_015_yolo_input` and `layer_022_yolo_input`; `layer_017_route` equals `layer_013_convolutional`. The directory now holds 37 470 076 bytes of `.bin`. The six decoded files are repeated in `yolov3_tiny_thresh_0p01`.

### yolov7_tiny: decoded rows (repeated in `yolov7_tiny_thresh_0p01`)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_090_yolo_boxes` | `[8112, 4]` | 129792 | 0.00557872234 | 0.994528413 | 0.285993977 | `69107ef872380aaffb2d4076edcac0d3b7c1a3b42ca343798eccb99310602bf3` |
| `layer_090_yolo_objectness` | `[8112, 1]` | 32448 | 6.93933515e-08 | 0.0158627331 | 9.05447745e-05 | `f9af48cd8eaab9270f86f5b543f3cbfb756b0520e6037ed8c50f1b1e3927f1cf` |
| `layer_090_yolo_scores` | `[8112, 80]` | 2595840 | 9.66764321e-13 | 0.00432451954 | 8.81014348e-07 | `da453a7403432cc5c9c11c98f067951dac8bb99efde2136903be3a6a3b00bec5` |
| `layer_094_yolo_boxes` | `[2028, 4]` | 32448 | 0.0109815923 | 0.989150822 | 0.349459718 | `c0ca51cd4870442d1621ced67ecaa7825232d4d67c8670293b337493aa5f77fe` |
| `layer_094_yolo_objectness` | `[2028, 1]` | 8112 | 1.22107195e-07 | 0.0358091816 | 0.000960633553 | `d5d057b5fe07876a88d8535d5aac9842fe414e794075a4f8d8bd43ee745a17fb` |
| `layer_094_yolo_scores` | `[2028, 80]` | 648960 | 5.84731395e-12 | 0.0223833993 | 1.00608066e-05 | `5d7725758a664f55976d7a566d711711e5fab8372c447598e107900fae581fc7` |
| `layer_098_yolo_boxes` | `[507, 4]` | 8112 | 0.0504878424 | 1.05574667 | 0.461844085 | `3db07f8ed1157f0802041744e595973a204960a1035f5c591939cf0904e93deb` |
| `layer_098_yolo_objectness` | `[507, 1]` | 2028 | 1.34441322e-07 | 0.248599663 | 0.0138530231 | `252efd136993e633fc587153692fde4c5fb4cca8007f5252bab068d755a4a0f2` |
| `layer_098_yolo_scores` | `[507, 80]` | 162240 | 6.28539104e-12 | 0.0287045874 | 5.54910987e-05 | `a3bf67d6e0d92eed9e139febb36687a952799e7f61a740185ed5923bfd854c64` |

### yolov7_thresh_0p01: decoded rows

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_134_yolo_boxes` | `[19200, 4]` | 307200 | 0.00342257391 | 0.996753335 | 0.275022499 | `8b8e7c857fd0c14e80db075728794437769f1012f75419def215cd48489cac98` |
| `layer_134_yolo_objectness` | `[19200, 1]` | 76800 | 4.13770929e-09 | 0.0207956042 | 2.05673892e-05 | `4be66e5bd8d87008b6ab46c9c78720dd307565ecfed6a16063868a2e395c07da` |
| `layer_134_yolo_scores` | `[19200, 80]` | 6144000 | 1.11178705e-13 | 0.0106335348 | 2.25235132e-07 | `db26f895ce183f407aa52f55587cb396ca4d45befa5db7c9b25a6ace2c638570` |
| `layer_138_yolo_boxes` | `[4800, 4]` | 76800 | 0.0111311227 | 0.987112999 | 0.330239643 | `7343c647a1953c1955f9ad04dd9dc2c1466d926ffde4f17840b0a0d7c61b6d2c` |
| `layer_138_yolo_objectness` | `[4800, 1]` | 19200 | 2.35281172e-09 | 0.0328350626 | 0.000153452428 | `c4b4877597596864a66403ff9a935d0c74beb5e3f370853a2637f4ef516baa3c` |
| `layer_138_yolo_scores` | `[4800, 80]` | 1536000 | 9.45494835e-14 | 0.0144317579 | 1.71810651e-06 | `84c31c3ad4773e596a197f39dab7d581afabc56dbefd0a6e3a0ce7229b282147` |
| `layer_142_yolo_boxes` | `[1200, 4]` | 19200 | 0.0281386431 | 1.043576 | 0.466136593 | `e70a94f52905316a56ecdc9a7626d38fa525824c5b33eca22ae0097e48fd9d00` |
| `layer_142_yolo_objectness` | `[1200, 1]` | 4800 | 2.29633379e-08 | 0.0606684014 | 0.000784816625 | `03efad1623580865589b034f1a54ff12cfe702f347d38e8cc616ea4437788c5c` |
| `layer_142_yolo_scores` | `[1200, 80]` | 384000 | 1.1125482e-12 | 0.0143189467 | 1.01616828e-05 | `aa8600b8ff102087380a9e7cdfb4af3692f28d0134cceb1d2722786ba2e68b33` |

### yolov3 (416 x 416; `input` and `truth` are the files of section 6)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `detections` | `[2, 7]` | 56 | 0.294172019 | 72 | 10.6290227 | `e14b46e6a690ee7a16b550664c42e0ad8d74be9563178c2191777f1099d17249` |
| `layer_081_yolo_input` | `[255, 13, 13]` | 172380 | -25.6103439 | 8.37504196 | -6.95477087 | `72eb61fc9b9b308cf6b21dfea2ed487fb1cbd9fff1628e130a6c5cdd9f3f6dba` |
| `layer_082_yolo` | `[255, 13, 13]` | 172380 | -1.20734477 | 1.08569741 | 0.0188064957 | `28fc0d611d4dbe95f3ff7d83c63cb444d1e308fa82cde988cf1a1e6092737240` |
| `layer_082_yolo_assigned_class`, `_assigned_row` | `[3, 13, 13]` | 2028 | -1 | 2 | -0.992110454 | `2f06e2c057842022d7ab55cb9136020b5bb061e5ede8b92cb9693af01d9f2cd7` |
| `layer_082_yolo_boxes` | `[507, 4]` | 8112 | 0.0275863055 | 1.01860547 | 0.487408071 | `3cee7c5091ff60bb9c78023a6b12ea319cc08bcb120cc3f49415400110240307` |
| `layer_082_yolo_delta` | `[255, 13, 13]` | 172380 | -0.999769509 | 0.999998748 | -6.49471772e-05 | `b0ba6274387263b7d56d9088b658f6e178a68d99112d741db4eaa69bfc12f7b8` |
| `layer_082_yolo_objectness` | `[507, 1]` | 2028 | 2.61688143e-10 | 0.999769509 | 0.0109118256 | `656aa783ec4aa19973782011e51a0b2e5a0878bda7728b448de19ace702751b7` |
| `layer_082_yolo_scores` | `[507, 80]` | 162240 | 1.389191e-15 | 0.944021583 | 0.000128147719 | `12d1fca2584a2c8327aaf896fc3d98e07023a426994346ac050c727783c0def3` |
| `layer_093_yolo_input` | `[255, 26, 26]` | 689520 | -23.4514656 | 4.33472776 | -7.39474153 | `93586325100681fab6b2a32af22ee65999df3b8ad14484cfc6f30c7f1026a4ad` |
| `layer_094_yolo` | `[255, 26, 26]` | 689520 | -1.58354616 | 1.64260685 | 0.0188425045 | `f42be4855a7ec70246bbd31105a5e1261557ab619307603b1ac534618f1f8167` |
| `layer_094_yolo_assigned_class` | `[3, 26, 26]` | 8112 | -1 | 16 | -0.991617357 | `ffcae4b178553618daa1e154a05821aa59d54b3ce2c958b7a0fab8e6349f4536` |
| `layer_094_yolo_assigned_row` | `[3, 26, 26]` | 8112 | -1 | 1 | -0.999013807 | `f13a744b14eef3b46b263eb4664f9f30da6ac67c19abc6b9b6128bfc557bd848` |
| `layer_094_yolo_boxes` | `[2028, 4]` | 32448 | 0.00895041879 | 0.989958227 | 0.330960966 | `7536641a43f537a331095ef27fb829ecabb7f03deed80768f823ab62ef5e7b39` |
| `layer_094_yolo_delta` | `[255, 26, 26]` | 689520 | -0.421976805 | 0.999999821 | 1.86104329e-05 | `79da9487065baff70dc321866655ba3a62c1a1ac20f63fefea40cbae1240d62a` |
| `layer_094_yolo_objectness` | `[2028, 1]` | 8112 | 6.53368054e-11 | 0.00034218264 | 1.60941963e-06 | `89bb91480e8cf4e36b95f8a9b17b3f9e3d70eeb173a205b4ab8861f1bb12e9fd` |
| `layer_094_yolo_scores` | `[2028, 80]` | 648960 | 4.93378957e-16 | 5.40923793e-05 | 1.20833799e-08 | `c24f10afbf44c6825440bc9c53ab977b25871effe4f954928c3744cae21cdf48` |
| `layer_105_yolo_input` | `[255, 52, 52]` | 2758080 | -26.2584209 | 5.58423424 | -7.02534606 | `ce1bc11f52402d0affb1179ff6a3a8a7ec9825b490c728e4482faa27fc77cf6e` |
| `layer_106_yolo` | `[255, 52, 52]` | 2758080 | -2.95188141 | 1.76167989 | 0.0195055122 | `1de002cf9855e8a2b4457d31351f017c397f21d04e6b6cad98c280d93369c780` |
| `layer_106_yolo_assigned_class`, `_assigned_row` | `[3, 52, 52]` | 32448 | -1 | -1 | -1 | `db85d9e39fcaa6fee92cbb718c1078ca4dc5614dacbe04966f906e098650c0c8` |
| `layer_106_yolo_boxes` | `[8112, 4]` | 129792 | 0.000937779783 | 0.998794317 | 0.278838109 | `1220d29db1ca845421e50908ef64695117a7ebab738d8dbf41697f1eb47acad5` |
| `layer_106_yolo_delta` | `[255, 52, 52]` | 2758080 | -0.000142286328 | 0 | -1.27770928e-08 | `6d457692f32de76001d5ba3b876088b8547f586008336b25a741e219f3166aaf` |
| `layer_106_yolo_objectness` | `[8112, 1]` | 32448 | 3.94559645e-12 | 0.000142286328 | 1.08605289e-06 | `08003a978630d715dfd55806698465eb02cab37c75ff1cdf4045bf4a3a566a4f` |
| `layer_106_yolo_scores` | `[8112, 80]` | 2595840 | 5.31103752e-21 | 5.71103119e-05 | 1.09228606e-08 | `0c002e24945199ba7607f996fa493e596ca9c16d85467eebf67ae03b4a2a8924` |

At 0.25: 2 rows from 6 candidates, `nms_kind` 0.

```
0.475477934 0.515369534 0.985516846 0.841887534 0.999769509 0.944021583 68
0.456058294 0.518443048 0.94367516 0.848751664 0.983175159 0.294172019 72
```

`yolov3_thresh_0p01/detections`: `[3, 7]`, 84 bytes, `43325da6181ae1fc8e0c9c46e9ac8392ea2e65c509b49647faef56b99efdbac3`, 3 rows from 9 candidates:

```
0.456058294 0.518443048 0.94367516 0.848751664 0.983175159 0.294172019 72
0.475477934 0.515369534 0.985516846 0.841887534 0.999769509 0.944021583 68
0.474889368 0.646638334 0.906094432 0.615450382 0.162092656 0.0103095695 69
```

Assignment: truth row 2 at layer 82, slot 0, cell (9, 9); row 0 at layer 82, slot 1, cell (6, 6); row 1 at layer 94, slot 0, cell row 7, column 6; none at layer 106. Stored cost 10.105978 (675 non-zero deltas), 3.14048457 (2112), 2.64356714e-07 (8112).

```
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 82 Avg (IOU: 0.652228), count: 2, class_loss = 9.343662, iou_loss = 0.762316, total_loss = 10.105978
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 94 Avg (IOU: 0.490963), count: 1, class_loss = 2.178990, iou_loss = 0.961494, total_loss = 3.140485
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 106 Avg (IOU: 0.000000), count: 1, class_loss = 0.000000, iou_loss = 0.000000, total_loss = 0.000000
```

### yolov4_tiny (416 x 416)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `layer_029_yolo_input` | `[255, 13, 13]` | 172380 | -18.0098801 | 2.23715353 | -6.58540421 | `16c64a910a93f2b82f7aa6d565cc590d8fe61a8bfc3703c86469cd08655fe0ec` |
| `layer_030_yolo` | `[255, 13, 13]` | 172380 | -1.02319205 | 1.4202199 | 0.0222652469 | `af993cea3da77c8170ba30eb761a1e32e853ab4fce6d633e7c2a256dd0338069` |
| `layer_030_yolo_assigned_class`, `_assigned_row` | `[3, 13, 13]` | 2028 | -1 | 2 | -0.992110454 | `2f06e2c057842022d7ab55cb9136020b5bb061e5ede8b92cb9693af01d9f2cd7` |
| `layer_030_yolo_boxes` | `[507, 4]` | 8112 | 0.0329437032 | 0.982196033 | 0.468524608 | `de011fdf5bbe15bfbd0768689fc1615aa2df64c7aabf8c57e54f1fb789fbcacf` |
| `layer_030_yolo_delta` | `[255, 13, 13]` | 172380 | -0.687694609 | 0.999980152 | -9.3695811e-05 | `51ded10ecc750fd08020783a9a3f355a774b7ea883a91b38133eadaec6ba8a0b` |
| `layer_030_yolo_objectness` | `[507, 1]` | 2028 | 1.50802464e-08 | 0.592214584 | 0.0110382316 | `46d3514b4aacd7c831439864b13341aa9bc661eee651fe479f8354fee8e76984` |
| `layer_030_yolo_scores` | `[507, 80]` | 162240 | 5.84866574e-13 | 0.186263993 | 0.000108982226 | `d1854b92cb44d270eb0851f2c31d102d8bf91db3ee7263941e096717c8a71b28` |
| `layer_036_yolo_input` | `[255, 26, 26]` | 689520 | -18.7265606 | 4.12847805 | -6.29743766 | `717909554f7a8f0ad83096d7711f623df74d5f4504c8a6a1f4c41d62e7064090` |
| `layer_037_yolo` | `[255, 26, 26]` | 689520 | -1.26012671 | 1.26295245 | 0.0247980797 | `ebf1dfb2dea345c7610e42fb8221f25e85c85d8fb65a7d8473bc5a759e75d3d8` |
| `layer_037_yolo_assigned_class` | `[3, 26, 26]` | 8112 | -1 | 16 | -0.990138067 | `f8991180cdffa3a8fdb9c12e93838c90ded4517cd50b17558ccb851d3ed5215d` |
| `layer_037_yolo_assigned_row` | `[3, 26, 26]` | 8112 | -1 | 2 | -0.997534517 | `5e12b14e7212c9f6b42e83a2810f9270ad6b97305f969bff9cab60a468a783a1` |
| `layer_037_yolo_boxes` | `[2028, 4]` | 32448 | 0.00965559389 | 0.989131391 | 0.323932053 | `40a07675ebe57bcc3c33737d9897879928128c547fd32c8bb6daca924c4bf7a3` |
| `layer_037_yolo_delta` | `[255, 26, 26]` | 689520 | -1.10375893 | 0.999999464 | 1.09690214e-05 | `3ed4c8d638dfdeebb2452d4a55cce4e0e057fc831b6e5b26ad0f87893ae13571` |
| `layer_037_yolo_objectness` | `[2028, 1]` | 8112 | 7.36475148e-09 | 0.00383588765 | 4.8947936e-05 | `c0648c9be1d6e96269dec23932de6d2047eadae11616ed2212e7dbd4a5c9a1f1` |
| `layer_037_yolo_scores` | `[2028, 80]` | 648960 | 9.85626765e-15 | 0.000679959543 | 5.1562293e-07 | `6d757e05049ce5845d22dc9193583f45c45cf98cd4780f8693a74f4234a9b377` |

`input`, `truth` and the empty `detections` are the files of section 6. At 0.25: 0 rows from 11 candidates, `nms_kind` 1. The second head has `mask = 1,2,3`, so anchor 3 belongs to both heads: truth row 2 is assigned at layer 30 (slot 0) and at layer 37 (slot 2).

Assignment: truth row 2 at layer 30, slot 0, cell (9, 9); row 0 at layer 30, slot 1, cell (6, 6); row 1 at layer 37, slot 1, cell row 7, column 6; row 2 at layer 37, slot 2, cell row 19, column 18. Stored cost 6.4185462 (675 non-zero deltas), 4.14668655 (2196).

```
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 30 Avg (IOU: 0.500919), count: 2, class_loss = 6.383611, iou_loss = 0.237106, total_loss = 6.620717
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 37 Avg (IOU: 0.685359), count: 2, class_loss = 4.124661, iou_loss = 2.040528, total_loss = 6.165190
```

`yolov4_tiny_thresh_0p01/detections`: `[16, 7]`, 448 bytes, `fbf45831d2070b62594bceb2a6105211d9056496792cdb0820bd4f0bc32a87f5`, 16 rows from 37 candidates:

```
0.576520085 0.352618992 0.982196033 0.812515855 0.276052862 0.0374743566 68
0.498983681 0.355907559 0.908489347 0.754035592 0.271912277 0.0244752765 28
0.639251709 0.575268269 0.735128641 0.846868634 0.0739169046 0.0114859538 68
0.420948803 0.576464295 0.940784276 0.825694442 0.431879014 0.186263993 28
0.423307508 0.355867118 0.932099581 0.769241512 0.401848614 0.145183966 15
0.421613753 0.496941388 0.938910723 0.836163342 0.582078218 0.0160563812 24
0.421613753 0.496941388 0.938910723 0.836163342 0.582078218 0.0466618873 26
0.575925648 0.579398274 0.944067299 0.851303101 0.386642784 0.146850482 25
0.499933153 0.735780835 0.880882025 0.559920788 0.0362568758 0.0265961476 28
0.574328721 0.507876039 0.894099951 0.856502831 0.301612258 0.0273071323 15
0.422668666 0.420303375 0.959788799 0.817168713 0.592214584 0.0234999042 0
0.422668666 0.420303375 0.959788799 0.817168713 0.592214584 0.0160662793 16
0.422668666 0.420303375 0.959788799 0.817168713 0.592214584 0.0159900431 57
0.422668666 0.420303375 0.959788799 0.817168713 0.592214584 0.05835573 59
0.422668666 0.420303375 0.959788799 0.817168713 0.592214584 0.0119494647 63
0.570651352 0.209588349 0.923376858 0.441316396 0.0237778816 0.0189585108 25
```

### yolov4 (608 x 608)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `detections` | `[1, 7]` | 28 | 0 | 1.06989157 | 0.619299395 | `e5b1de7094b391c10745904c8f70ee9539b60f39917070082ac0cc9417898b9e` |
| `input` | `[3, 608, 608]` | 4435968 | 0 | 1 | 0.499723624 | `fd32ed64373eff75ed1709262ee9e00a04008b96b901cd758a5f438e4f6dd3d5` |
| `layer_138_yolo_input` | `[255, 76, 76]` | 5891520 | -19.9334793 | 3.32445955 | -6.11190742 | `b26dffb414062ac6e15adbf78ef6e1aa55284e3969e0d039da47187300a15eaa` |
| `layer_139_yolo` | `[255, 76, 76]` | 5891520 | -2.20023108 | 1.55204272 | 0.0234692622 | `19e71e5340c7f22cbc6a2899e316efc25bc5d382ef3e2dcd3afd0ae4b334652f` |
| `layer_139_yolo_assigned_class`, `_assigned_row` | `[3, 76, 76]` | 69312 | -1 | -1 | -1 | `d482f6c0cecea58b50289c1992bdb330b54defe0a6dc91e087ddf26106d8e1fc` |
| `layer_139_yolo_boxes` | `[17328, 4]` | 277248 | 0.0012682355 | 0.999221623 | 0.276144012 | `c7a2489e31e468cfaed31b32b48026bad5e4db111cb729bef53ce5ebe4b79ed1` |
| `layer_139_yolo_delta` | `[255, 76, 76]` | 5891520 | -0.00665664487 | 0 | -7.17356942e-08 | `b0e11dc3776494419e9cd270e103a94d7e5181c0878d8c0acbcb4ad2f05aaa96` |
| `layer_139_yolo_objectness` | `[17328, 1]` | 69312 | 2.20292606e-09 | 0.00665664487 | 6.09753401e-06 | `d1065bff87d75f209b5fd90432be877b481a1cded126817ff85377e37d0fd864` |
| `layer_139_yolo_scores` | `[17328, 80]` | 5544960 | 3.51268745e-16 | 0.00257695629 | 8.82088144e-08 | `603149cfa21d7a83f925ff78799c310ea6752edd13b5a836b1b3204f14d235e7` |
| `layer_149_yolo_input` | `[255, 38, 38]` | 1472880 | -18.3986244 | 2.41983223 | -5.55679115 | `0715762a9915159a347e6aaad1e112bca542d6071f5beb841e3628fbba8ae1d8` |
| `layer_150_yolo` | `[255, 38, 38]` | 1472880 | -1.6335634 | 1.43527973 | 0.0303043107 | `f78ad29f8946945f011beb6fc6b5cca3f625dfac2b16325294975362715df6c9` |
| `layer_150_yolo_assigned_class` | `[3, 38, 38]` | 17328 | -1 | 16 | -0.996075716 | `057878694dbe9b9d872fde4a7bf80f0bf356d9ba791446d0a5c6740f11acecc2` |
| `layer_150_yolo_assigned_row` | `[3, 38, 38]` | 17328 | -1 | 1 | -0.999538319 | `49b7a4abecc2bab4e600893403b4d8fd1fea5d38d6f562a14249a566413997ac` |
| `layer_150_yolo_boxes` | `[4332, 4]` | 69312 | 0.0111807613 | 0.991043806 | 0.340124756 | `7d73516c633b70f1d0f03613c7ebe5f07d7105d51a9cca773ff31741374fd1d7` |
| `layer_150_yolo_delta` | `[255, 38, 38]` | 1472880 | -0.795941412 | 1.92664182 | 1.80772841e-05 | `f571681ee6d0ef2adbb53370e690305bd845e5bbd423a58e7e02ae19b7bcdad2` |
| `layer_150_yolo_objectness` | `[4332, 1]` | 17328 | 1.02230127e-08 | 0.00934946537 | 3.08456378e-05 | `0405d8734ec70f566565290e5bd8539fa82588146399ea7fbc0036b83f845596` |
| `layer_150_yolo_scores` | `[4332, 80]` | 1386240 | 8.92931399e-13 | 0.00187492091 | 3.6579586e-07 | `749e920b09055765fd232a698fd6d4e46b327906addc1e57a49996f53f5f2884` |
| `layer_160_yolo_input` | `[255, 19, 19]` | 368220 | -18.3739471 | 3.77599144 | -6.09322476 | `a906d0c8ee2292ac28d161594148850ad3fb703105f8ff92f52bbe8689dbdd9a` |
| `layer_161_yolo` | `[255, 19, 19]` | 368220 | -1.50384712 | 1.45249224 | 0.0238091147 | `6a0ddcdcc32b005183b1fb8d8f70ea799591f16c36b66cb82a111ee4e60f42e9` |
| `layer_161_yolo_assigned_class`, `_assigned_row` | `[3, 19, 19]` | 4332 | -1 | 2 | -0.996306556 | `d8eed43a92e15bef2bd053eab159d5aa8cb847e71858199d551d3d86950077fb` |
| `layer_161_yolo_boxes` | `[1083, 4]` | 17328 | 0.0202482156 | 1.0945884 | 0.463998905 | `4ed69a29a9da3c52381372054264638b2002e950eae0e17f6e6f1e0804062de5` |
| `layer_161_yolo_delta` | `[255, 19, 19]` | 368220 | -0.694345355 | 0.999986231 | 2.60158929e-05 | `0c556ef1b566152f971fc7997083bb4846eef64ce70904eea5430eb245e08585` |
| `layer_161_yolo_objectness` | `[1083, 1]` | 4332 | 1.04784288e-08 | 0.943473876 | 0.0012645214 | `e96e27af5b140ed1c8d3b044ce22c3f5c6298ba432f33e7d0d5a38346d3b9141` |
| `layer_161_yolo_scores` | `[1083, 80]` | 346560 | 2.52532283e-12 | 0.304312766 | 1.22303786e-05 | `7b4a6d5a75aaa885301cb3f018bc9befb95af219309c034b4cfdac93be89f21a` |

At 0.25: 1 row from 1 candidate, `nms_kind` 1.

```
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.304312766 0
```

`yolov4_thresh_0p01/detections`: `[5, 7]`, 140 bytes, `57f651a3f74a6740ef7f655c423b199a22882b181c9713f3e38ea4849d7a7bba`, 5 rows from 5 candidates:

```
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.304312766 0
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.112391718 25
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.126751676 27
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.0691307485 59
0.49611342 0.501353264 1.06989157 1.01995087 0.943473876 0.0124453148 75
```

Assignment by best anchor: none at layer 139; truth row 1 at layer 150, slot 2, cell row 11, column 9; row 2 at layer 161, slot 0, cell row 14, column 13; row 0 at layer 161, slot 1, cell (9, 9). Stored cost 0.000159919146 (17328 non-zero deltas), 9.05432034 (4667), 9.88566303 (1501).

```
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 139 Avg (IOU: 0.000000), count: 1, class_loss = 0.000160, iou_loss = 0.000000, total_loss = 0.000160
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 150 Avg (IOU: 0.585474), count: 4, class_loss = 9.025304, iou_loss = 5.970388, total_loss = 14.995692
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 161 Avg (IOU: 0.362349), count: 5, class_loss = 9.841027, iou_loss = 0.527369, total_loss = 10.368397
```

### yolov3_tiny_train_step (batch 2, 34 340 356 bytes of `.bin`)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 16]` | 64 | -1.11118996 | 0.435080796 | -0.105848577 | `70220ba676fd8177fef817c5ac4e69b87910d6fd49bdd4a87f4f71189188aae3` |
| `batch_variance` | `[1, 16]` | 64 | 0.0560609959 | 2.45413709 | 0.854896924 | `3ffc4169f0f3f14ec367bbbdd09b004735f952c8cc66b0330fda7ef55feb415c` |
| `input` | `[2, 3, 416, 416]` | 4153344 | 0 | 1 | 0.500035784 | `7b7563a106509747180c421f8fd1be6cec581bcf8aaeed22274976adbb383036` |
| `layer_000_bias_updates` | `[1, 16]` | 64 | 0 | 0 | 0 | `f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b` |
| `layer_000_output` | `[2, 16, 416, 416]` | 22151168 | -3.25457239 | 28.9920635 | 0.784267198 | `8dcbcb0ae8b0751f2e417550ab0720a22b4fd067989be1c3f1ffd232c6423915` |
| `layer_000_scale_updates` | `[1, 16]` | 64 | -3.98921537 | 3.45665956 | 0.18084514 | `f1f09fe123fbf702c8be19a6853575e823fb8b46bbb75f62b5481580fdf76da1` |
| `layer_000_weight_updates` | `[16, 3, 3, 3]` | 1728 | -27.9032593 | 19.8602905 | -0.158570385 | `ee9f64385613f3ff46ca029689f4d017e2df79d7df81a0367a0a8656da73526c` |
| `layer_014_output` | `[2, 512, 13, 13]` | 692224 | -0.759213328 | 5.50794983 | 0.445638337 | `ae541ae04642a491479031f24af0f683f0a2a78c57d4fcf83ceef15b03a36aa5` |
| `layer_015_bias_updates` | `[1, 255]` | 1020 | -0.716869473 | 1.38751125 | 0.0293592268 | `fcc0bee3b3e733a79e973423d4701f7047192621fb900fc2ee04e94df4b2d330` |
| `layer_015_weight_updates` | `[255, 512, 1, 1]` | 522240 | -3.21065354 | 7.08322906 | 0.0129693254 | `6eae8b20f562ca02eb24ff0b1cc78cbc33232597092f8b891b0d6136a3b8fa2c` |
| `layer_015_yolo_input` | `[2, 255, 13, 13]` | 344760 | -33.8204575 | 4.24410772 | -6.76585644 | `06fda5143c1c283fd988773398c0690461cb787be4e27e4f9dd4bf0a9c3be16b` |
| `layer_016_yolo` | `[2, 255, 13, 13]` | 344760 | -1.41159189 | 1.36032641 | 0.0166431497 | `62230ae186fa758e9ab28c09491efc67b4e473de30fa697e368cae052a055546` |
| `layer_016_yolo_cost` | `[1, 1]` | 4 | 12.0947657 | | | `d5651bcc6004d0de637be624f9364523c707d20c552117e41bd7b424609cb6d6` |
| `layer_016_yolo_delta` | `[2, 255, 13, 13]` | 344760 | -0.81123364 | 1.0498929 | 8.68616201e-05 | `7da63ddc3f49d3af27f3073d4e348b7d5325b6d885d60d506910d5a92f2b0fb4` |
| `layer_021_output` | `[2, 256, 26, 26]` | 1384448 | -0.86978358 | 7.62519598 | 0.563967734 | `37ac955e4b788d25fa9aa005ed28fcf353f396d2641864a5e007cf5abe835f83` |
| `layer_022_bias_updates` | `[1, 255]` | 1020 | -1.00536764 | 0.994520485 | -0.00896544269 | `d655ef3ba663cadc79b39dc17472011618b5429f2b18a01456275906cb806cfa` |
| `layer_022_weight_updates` | `[255, 256, 1, 1]` | 261120 | -3.72427273 | 4.43896055 | -0.0029364256 | `7eb5a79f85bde74aad302f50c50a4d7827c231dc889064b50a46120119e59210` |
| `layer_022_yolo_input` | `[2, 255, 26, 26]` | 1379040 | -31.3489761 | 4.94152069 | -6.76893745 | `fc1e29d0bc78ca743fda2f7e9a60df38d315c85a6ba98f8ba13b339a10a4600c` |
| `layer_023_yolo` | `[2, 255, 26, 26]` | 1379040 | -2.00156975 | 1.75923431 | 0.0203764467 | `e1eba8da4bb989bc699980f5a7cf0b351eb0980351d2e34ac6a20f5db2bae072` |
| `layer_023_yolo_cost` | `[1, 1]` | 4 | 3.19997597 | | | `6bd4e9e09d334a2849eb02f47ce42c0dbb28021121f9986e66bee7698d60a119` |
| `layer_023_yolo_delta` | `[2, 255, 26, 26]` | 1379040 | -0.834399283 | 0.999999821 | -6.63124504e-06 | `6a334a5f0523a51485e135e63e8d8dcfcdef7a9cae5cf91f03906a2a379bb17a` |
| `network_cost` | `[1, 1]` | 4 | 7.64737082 | | | `c2e10b364513356a433500b53164b304564d9ee736b546b54f61259a036d6b08` |
| `rolling_mean_after` | `[1, 16]` | 64 | -1.0624485 | 0.409146577 | -0.10490644 | `502fa1c4809bfcfdf114af653bdac1bd62282e960d13b0ded2e417139c7cbe35` |
| `rolling_mean_before` | `[1, 16]` | 64 | -1.05703282 | 0.406264991 | -0.10480176 | `87941e7288a5a3a1d4364472e3fd4d0ed2daa1c97604a284c5c2334042a97a81` |
| `rolling_variance_after` | `[1, 16]` | 64 | 0.0579088628 | 0.987287402 | 0.388725711 | `d28904116488e630a55e1a74ebd3f1312a7ea9862faff43fb8ccf7c18e548743` |
| `rolling_variance_before` | `[1, 16]` | 64 | 0.0132524753 | 0.824304104 | 0.336928916 | `ed60f736dc67f07a1da4385863be10277c3a37ef97d7f2c0157cac909d527afe` |
| `truth` | `[5, 6]` | 120 | 0 | 41 | 2.535 | `89cec5c9fccba5b3f630907f27ee13e58a6fe59decb892aa2a42d739df21f994` |

`network_cost.json` also holds `batch` 2, `subdivisions` 1, `learning_rate` 0.001, `momentum` 0.9, `decay` 0.0005.

```
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 16 Avg (IOU: 0.520880), count: 4, class_loss = 4.178824, iou_loss = 1.868559, total_loss = 6.047383
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 23 Avg (IOU: 0.501969), count: 1, class_loss = 1.094116, iou_loss = 0.505872, total_loss = 1.599988
```

### yolov7_tiny_train_step (batch 2, 42 715 972 bytes of `.bin`; `input` and `truth` as above)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 32]` | 128 | -0.30834195 | 0.454256266 | -0.0114759712 | `cf9f068bb79980660247f865c7e35b9f9ed98b4344f78fcdc711e60515df537f` |
| `batch_variance` | `[1, 32]` | 128 | 0.0014501228 | 0.138292879 | 0.0509921597 | `d9bf0eea8faca66b924bc5121474d8c836e8475104ca13fabe6a52396a890aa5` |
| `layer_000_bias_updates` | `[1, 32]` | 128 | 0 | 0 | 0 | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| `layer_000_output` | `[2, 32, 208, 208]` | 11075584 | -1.28664029 | 6.75693989 | 0.520951735 | `822fe23e9df62c6e3c7f361db30aec2de9207151ef911a86ce68d0e36cdaf449` |
| `layer_000_scale_updates` | `[1, 32]` | 128 | -0.00561921066 | 0.00297295977 | 1.04121198e-05 | `1043d4ecef76a1abe501f1c67014c614072c9b44200ccfe23e1e17246239e14f` |
| `layer_000_weight_updates` | `[32, 3, 3, 3]` | 3456 | -0.0361233018 | 0.049819462 | 0.000171468478 | `ea1fcedb9b830f6f908dc980b54b04942482868f300bf58c3ea2f7a39006a37f` |
| `layer_088_output` | `[2, 128, 52, 52]` | 2768896 | -2.0518558 | 19.0545406 | 1.24580735 | `1119f3cfcf87f05e727de8d029f109d1c297822c64f390b8ee337c7890b71ee7` |
| `layer_089_bias_updates` | `[1, 255]` | 1020 | -0.107384957 | 0.110680908 | -0.000409444741 | `4ad448ad0ea46f18a074b89b672acd3215ad11d821ad61d6a2eccc258c58b16c` |
| `layer_089_weight_updates` | `[255, 128, 1, 1]` | 130560 | -0.728258133 | 0.749558151 | -0.000726772044 | `7026c293e7807bfd0ae69487cf9632ebf4e7639ccf50a9bccd520bc142cd7e94` |
| `layer_089_yolo_input` | `[2, 255, 52, 52]` | 5516160 | 4.5183135e-11 | 0.990507603 | 0.0364954595 | `f420772bdee4533f03763e48a8b47339062c8ef6947b08558a07faa9ee772770` |
| `layer_090_yolo` | `[2, 255, 52, 52]` | 5516160 | -0.382298917 | 1.31895316 | 0.0366127831 | `fb1871248a8eae83178ba9c797f46849187233c3ebde34a4e8fb96ebaa53e00a` |
| `layer_090_yolo_cost` | `[1, 1]` | 4 | 1.34784484 | | | `6980af11db5b2b3e8c97ab90232c484a9cbcdc21116c5e9b902d005d6b78d011` |
| `layer_090_yolo_delta` | `[2, 255, 52, 52]` | 5516160 | -0.211934254 | 0.997110248 | -4.66089346e-06 | `3fbc589d9d049f5cbf48421fb2bf2a386a07421868f0d5abf516c13ab8197168` |
| `layer_092_output` | `[2, 256, 26, 26]` | 1384448 | -1.6125176 | 14.6930618 | 0.858670496 | `9eb6fd8c5c00904f127b4476b1087015126d82a8689695bdd6bfaacf293fd358` |
| `layer_093_bias_updates` | `[1, 255]` | 1020 | -0.578628659 | 0.155354008 | -0.00398128228 | `7948874f1e72643fa8be06ed3af86e9e02f398b84a069d150986a798a15515eb` |
| `layer_093_weight_updates` | `[255, 256, 1, 1]` | 261120 | -2.34863067 | 0.628307521 | -0.00262944041 | `a4604db1f798b355c90245f76179730b66f37815c755524e3e2e5a730518c4e7` |
| `layer_093_yolo_input` | `[2, 255, 26, 26]` | 1379040 | 1.29210592e-11 | 0.99763 | 0.0338070292 | `5cb6ea082b2aa8281ae9be7aacbf25a616aca003b5b79d9d8d435b66800931ae` |
| `layer_094_yolo` | `[2, 255, 26, 26]` | 1379040 | -0.389608383 | 1.43459654 | 0.033783843 | `eec82bdbccb6348930a7b2cd5c40f4c241910ac93ffa5d663a82803c6d98454d` |
| `layer_094_yolo_cost` | `[1, 1]` | 4 | 10.1716776 | | | `0afc3ff6a66901ca76941160da433423d7d5e8781303aa647334be486788cc13` |
| `layer_094_yolo_delta` | `[2, 255, 26, 26]` | 1379040 | -0.375335395 | 0.998850048 | -3.30780531e-05 | `ab4c49c2d7aba115c83634b8b996eb40a9a78d1f0ec50e0759177d46639cbfe4` |
| `layer_096_output` | `[2, 512, 13, 13]` | 692224 | -1.22168481 | 7.83986473 | 0.520984576 | `5e6e13ea204b8bab441a9e42eaf709b7a8a161de532566e4829168b29951fe78` |
| `layer_097_bias_updates` | `[1, 255]` | 1020 | -1.2776947 | 0.0539661013 | -0.0137818621 | `ca0cb37b6d84c724d91d4e9e46343dbaa32324ad563a7dca71d54e947c74e4ec` |
| `layer_097_weight_updates` | `[255, 512, 1, 1]` | 522240 | -3.6391449 | 0.512976468 | -0.00659477321 | `1b8b2e0081e75a72d8c65ab244443695a41cd68549479919fb9b0a43dccff84d` |
| `layer_097_yolo_input` | `[2, 255, 13, 13]` | 344760 | 1.28693592e-11 | 0.999780357 | 0.03091215 | `27382abce338befe2c54cb824a8cbe7a0067efae866e0aab7d1f5eade40ee196` |
| `layer_098_yolo` | `[2, 255, 13, 13]` | 344760 | -0.349392116 | 1.40669751 | 0.0309944691 | `4755773068996ecc4f87b13a702b6a0668319d87f0b0b1d98ad8b7ca378262e2` |
| `layer_098_yolo_cost` | `[1, 1]` | 4 | 19.198719 | | | `31225276e567a102ed807e0fed37cb6d77d2961742a85914c436ecd3a85b2641` |
| `layer_098_yolo_delta` | `[2, 255, 13, 13]` | 344760 | -0.910977006 | 0.999859214 | -0.000190074989 | `3481daa6cf5ce6154d6c9f0ebf5e04a0a3249305d98e1dceb343edfc06017656` |
| `network_cost` | `[1, 1]` | 4 | 10.2394142 | | | `a65acc96649d8b6380cb1e1db91b65cea1a64baf5de2f5577f5aa7cc83126294` |
| `rolling_mean_after` | `[1, 32]` | 128 | -0.259400338 | 0.421377748 | -0.0103083738 | `cf214fc9a1ec88ef0a2f54d6c824c8ca144215056d9564a7eeb656c3548cfeba` |
| `rolling_mean_before` | `[1, 32]` | 128 | -0.255859375 | 0.417724609 | -0.0101786405 | `fe12bb8b936a4379c47b78046e4ed516f0e4552620264c437d4b07d04e4d25b4` |
| `rolling_variance_after` | `[1, 32]` | 128 | 0.00100718159 | 0.0580492578 | 0.021931673 | `e736578ebfeeec3d3fe5b0a7de456f232d0df6f2488bf2cc9ffa2f7edeafaf9e` |
| `rolling_variance_before` | `[1, 32]` | 128 | 0.000957965851 | 0.0585021973 | 0.0187027305 | `a6c1b050b08850b68ef49ca0903bf045b56eb5eecb5f8f02a2a25eef70f83947` |

For these CIoU layers the stored cost is `iou_normalizer * mean(1 - IoU) + class_loss`, so it differs from the sum of squared deltas (layer 90: 1.34784484 against 1.62620923).

```
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 90 Avg (IOU: 0.215644), count: 1, class_loss = 0.654314, iou_loss = 0.158790, total_loss = 0.813104
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 94 Avg (IOU: 0.363187), count: 8, class_loss = 5.069919, iou_loss = 0.483121, total_loss = 5.553040
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 98 Avg (IOU: 0.327432), count: 11, class_loss = 9.582545, iou_loss = 0.098301, total_loss = 9.680846
```

### Sizes

`_darknet/out` is 310 MB in 13 directories plus five input files; `_darknet` is 333 MB (`src` 17 MB, `build` 5.5 MB). Budget was 600 MB.
