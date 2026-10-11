# Darknet-era YOLO: reference run and source findings

**Status:** Forward and `[yolo]` training references exist for all six models; every layer of `yolov3-tiny` is dumped; one training step under batch statistics exists for `yolov3-tiny` and `yolov7-tiny`. Sections 8 to 14 are the second pass, section 15 the implementation so far, sections 16 to 24 the third pass, sections 25 to 29 the fourth. Section 25 supersedes every inference-mode dump of `yolov4`, `yolov7` and `yolov7-tiny` taken before it: the `plain_pool` directories are the reference for those three.

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

## 15. Implementation, waves 1 to 4

Written on 2026-10-09 by workflow `wf_5a220fdc-b48` from `SPEC.md`, `PLAN.md` and the sections above; no worker opened the Darknet clone, and each wave's reviewer checked, as a named item, that no code reads as a port and no comment names a foreign function. None was found.

```
dispatch: specification and plan, three revisions — opus low, passed, rung 1
dispatch: Darknet build, dump and source reading, two passes — opus low, passed, rung 1
dispatch: T07 — haiku medium, 10 turns, passed, rung 1
dispatch: F03 — haiku high, 21 turns, passed, rung 1
dispatch: F02 — haiku high, 22 turns, passed, rung 1
dispatch: F08 — haiku high, 35 turns, passed, rung 1
dispatch: F04 — haiku high, 36 turns, passed, rung 1
dispatch: F06 — opus low, 18 turns, passed, rung 1
dispatch: F07 — haiku high, 35 turns, passed, rung 1
dispatch: F09 — haiku high, 43 turns, passed, rung 1
dispatch: F01 — haiku high, 72 turns, passed, rung 1
dispatch: review wave 1 — opus low, 43 turns, passed, rung 1
dispatch: F05 — haiku medium, 25 turns, passed, rung 1
dispatch: A02 — opus low, 9 turns, passed, rung 1
dispatch: T08 — opus low, 17 turns, passed, rung 1
dispatch: A01 — opus low, 15 turns, passed, rung 1
dispatch: T01 — opus low, 20 turns, passed, rung 1
dispatch: F10 — haiku high, 50 turns, passed, rung 1
dispatch: review wave 2 — opus low, 30 turns, passed, rung 1
dispatch: A04 — opus low, 9 turns, passed, rung 1
dispatch: T05 — haiku medium, 29 turns, passed, rung 1
dispatch: A03 — opus low, 16 turns, passed, rung 1
dispatch: T06 — haiku high, 35 turns, passed, rung 1
dispatch: F11 — opus low, 34 turns, passed, rung 1
dispatch: review wave 3 — opus low, 33 turns, passed, rung 1
dispatch: A06 — opus low, 13 turns, passed, rung 1
dispatch: F12 — haiku medium, 33 turns, passed, rung 1
dispatch: F13 — haiku medium, 29 turns, passed, rung 1
dispatch: A05 — opus low, 15 turns, passed, rung 1
dispatch: F14 — opus low, 21 turns, passed, rung 1
dispatch: T09 — opus low, 30 turns, passed, rung 1
dispatch: review wave 4 — opus low, 26 turns, passed, rung 1
dispatch: F20 — haiku high, 28 turns, passed, rung 1
dispatch: F21 — haiku high, 28 turns, passed, rung 1
dispatch: review wave 4b — opus low, 12 turns, passed, rung 1
```

T07, the step schedule, was not written: SPEC 4.8 does not say when a step applies, whether scales compound or how steps meet the warm-up. Other gaps the workers stopped at: a per-layer tolerance, the `[net]` solver defaults, the default of `ignore_thresh`, the head count of the Blob case.

The orchestrator wrote the build files of the two new modules, `vision/importers` and `vision/training`, added their rows to `cmake/SushiAILayers.cmake` and `tools/layering/check_layering.py`, 26 headers to the two umbrellas, 8 root tests to the two test binaries and 7 rows to `site_classes.tsv`.

Build B1, RTX 3080 Ti, SushiBLAS main: the first build had no error.

| Run | Result | Repair |
| --- | --- | --- |
| 1 | `sushiai_tests` 1383 passed, 3 failed; `sushiai_vision_tests` 907 passed, 3 failed, 1 skipped. Every parity case passed | Five expectations: two registry tests that listed 14 layer rows, a diagnostic that reads `-1,-1`, two shortcut texts without a `[yolo]` section. One order of checks: a head given no anchor is refused before the sum is tested |
| 2 | `sushiai_tests` 1386 passed; `sushiai_vision_tests` 910 passed, 1 skipped | |

Without `SA_REFERENCE_DATA_DIR` the parity cases are reported skipped.

Measured against the Darknet dumps, with the bound each test states:

| Comparison | Largest difference | Bound |
| --- | --- | --- |
| `yolov3-tiny`, the two raw heads, on a folded import | 3.8e-5 | 1e-3 + 1e-4 of the reference |
| `yolov3-tiny`, the two `[yolo]` outputs | 6.1e-6 | the same |
| `yolov3-tiny`, 22 body layers, one by one | 5.2e-5, at layer 12 | the same |
| `yolov3-tiny`, 16 detection rows at threshold 0.01 | 1.3e-6 | 1e-4 |
| Separate batch norm against the folded import | 4.4e-5 | 1e-3 + 1e-4 of the reference |
| Decoded boxes of all six models, 14 heads | 1.2e-7 | 1e-6 |
| Scores of all six models | 8.9e-8 | 1e-6 |

The head values reach 17 in magnitude, so 3.8e-5 is about two parts in a million.

## 16. Learning rate and the update step

`src/parser.c`, `parse_net_options`, line 1145; `src/network.c`, `get_current_batch` (65), `get_current_rate` (131), `update_network` (293), `train_network_datum` (361), `train_network_waitkey` (407); `src/convolutional_layer.c`, `update_convolutional_layer` (1599). Read from the source; no run exercised the update.

`[net]` defaults:

| Key | Default | Key | Default |
| --- | --- | --- | --- |
| `batch` | 1 | `learning_rate` | 0.001 |
| `subdivisions` | 1 | `momentum` | 0.9 |
| `max_batches` | 0 | `decay` | 0.0001 |
| `policy` | `constant` | `burn_in` | 0 |
| `power` | 4 | `steps`, `scales` | required when `policy=steps` |

Counters. With `B` the cfg's `batch` and `S` its `subdivisions`, the mini-batch is `m = B / S` (integer division). `seen` counts images and grows by `m` before each forward pass. The iteration counter grows by one after the `S` forward and backward passes of one iteration. Training runs while the iteration counter is below `max_batches`. The schedule reads `n = seen / (m * S)` with integer division, which equals the number of completed iterations including the current one, because `seen` is advanced before the rate is read. The first update therefore sees `n = 1`.

Rate, with `lr` the cfg's `learning_rate`:

```
n < burn_in:              rate = lr * (n / burn_in) ^ power
otherwise, policy steps:  rate = lr * product of scales[i] over every i with steps[i] <= n
otherwise, constant:      rate = lr
```

- A step applies at its listed iteration: the test is `steps[i] <= n`.
- The scales compound, in list order. The walk stops at the first step still ahead, so the list is expected in ascending order.
- The warm-up replaces the policy while `n < burn_in`; the two are not multiplied.
- Other policies exist (`step`, `exp`, `poly`, `sigmoid`, `random`, `sgdr`); none of the six cfgs uses them. `poly` is `lr * (1 - n / max_batches) ^ power`.

The six cfgs:

| Model | `batch` / `subdivisions` | `learning_rate` | `momentum` | `decay` | `burn_in` | `max_batches` | `steps` | `scales` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `yolov3-tiny`, `yolov3` | 1 / 1 as shipped (64 / 2 and 64 / 16 are commented out) | 0.001 | 0.9 | 0.0005 | 1000 | 500200 | 400000, 450000 | 0.1, 0.1 |
| `yolov4-tiny` | 64 / 1 | 0.00261 | 0.9 | 0.0005 | 1000 | 2000200 | 1600000, 1800000 | 0.1, 0.1 |
| `yolov4` | 64 / 8 | 0.0013 | 0.949 | 0.0005 | 1000 | 500500 | 400000, 450000 | 0.1, 0.1 |
| `yolov7-tiny` | 64 / 1 | 0.00261 | 0.9 | 0.0005 | 1000 | 2000200 | 1600000, 1800000 | 0.1, 0.1 |
| `yolov7` | 8 / 1 | 0.00261 | 0.9 | 0.0005 | 1000 | 2000200 | 1600000, 1800000 | 0.1, 0.1 |

One iteration, in order. `U` is a layer's accumulated update for a parameter `W`; it starts at zero and is never cleared, only scaled.

1. For each of the `S` mini-batches: advance `seen` by `m`; forward in training mode; backward, which adds this mini-batch's descent direction to `U` (section 13: summed over images and positions, not divided by anything).
2. Advance the iteration counter. Read `rate` from the schedule.
3. For every convolution, with `B = m * S`:

```
weights:  U <- U - decay * B * W       W <- W + (rate / B) * U       U <- momentum * U
biases:                                b <- b + (rate / B) * Ub      Ub <- momentum * Ub
scales (batch norm):                   g <- g + (rate / B) * Ug      Ug <- momentum * Ug
```

So with `G` the summed descent direction of the iteration, `U_t = momentum * U_(t-1) + G_t - decay * B * W`, and the step is `rate / B` times `U_t`. Decay acts on weights only, and it enters the momentum buffer. A per-layer `learning_rate` key (default 1) multiplies `rate` for that layer; none of the six cfgs sets it. The loss the trainer prints for an iteration is the sum of the per-mini-batch network costs divided by `S * m`.

## 17. `[yolo]` keys: defaults and where each acts

`src/parser.c`, `parse_yolo`, lines 448-534; uses in `src/yolo_layer.c`.

| Key | Default | Where it acts |
| --- | --- | --- |
| `classes` | 20 | Channel count check: the layer before must have `len(mask) * (5 + classes)` channels, or Darknet stops |
| `num` | 1 | Number of anchors in `anchors`; the best-anchor search runs over all of them |
| `mask` | all `num` anchors, in order | Which anchors this layer owns |
| `anchors` | every value 0.5 | Pairs in input pixels; values beyond `2 * num` are ignored |
| `max` | 200 | Truth rows per image the layer reads; also sizes the truth buffer |
| `ignore_thresh` | 0.5 | Training, step 1 of section 4 |
| `truth_thresh` | 1 | Training, step 1; never fires at 1 |
| `iou_thresh` | 1 | Training, step 2: extra anchors, and step 3 runs only below 1 |
| `iou_thresh_kind` | `iou` | Measure used for `iou_thresh`; also `giou`, `diou`, `ciou`; anything else warns and falls back to `iou` |
| `iou_loss` | `mse` | Box delta branch; also `giou`, `diou`, `ciou`; anything else is taken as `iou` |
| `iou_normalizer` | 0.75 | Multiplies the box deltas in every branch, `mse` included |
| `obj_normalizer` | 1 | Multiplies every objectness delta; also scales the reported class loss |
| `cls_normalizer` | 1 | Multiplies the truth class's delta only when `counters_per_class` is given. Without that key it has no effect, so the 0.5 of the v7 cfgs changes nothing in the deltas although the log line prints it |
| `delta_normalizer` | 1 | GPU backward only, where the delta passed down is multiplied by it and by `[net] loss_scale`; the CPU backward adds the delta unscaled |
| `max_delta` | unset (no clipping) | Clips the IoU-branch box deltas; also caps the `counters_per_class` multipliers |
| `label_smooth_eps` | 0 | Class targets become `t * (1 - eps) + 0.5 * eps` |
| `scale_x_y` | 1 | Forward, section 3.6 |
| `new_coords` | 0 | Forward, decode, and the `w`, `h` chain of the deltas |
| `objectness_smooth` | 0 | Training, sections 4 and 11 |
| `focal_loss` | 0 | Replaces the class deltas by a focal form; unused by the six cfgs |
| `counters_per_class` | none | Per-class multiplier `largest count / count`, capped at `max_delta`, on the box and class deltas of a truth of that class |
| `nms_kind` | `default` | Read by `detector test` from the last `[yolo]` layer. Accepted spellings: `default`, `greedynms`, `diounms`. Any other string silently becomes `default` |
| `beta_nms` | 0.6 | Exponent of `diounms` only (section 3.8) |
| `jitter` | 0.2 | Data loader: crop jitter. Not in the layer |
| `random` | 0 | Trainer: multi-scale resizing of the network. Not in the layer |
| `resize` | 1.0 | Data loader. Not in the layer |

The six cfgs set `num` (6 or 9), `jitter` (0.3 or 0.1), `random`, `truth_thresh=1`, and `resize=1.5` in some v4 and v7 heads; none sets `max`, `label_smooth_eps`, `focal_loss` or `counters_per_class`.

## 18. Edge cases

Truths, in the loader (`src/data.c`, `fill_truth_detection`, line 371):

- The image's boxes are shuffled, then cut to `max`. With more truths than `max`, a random subset survives.
- Dropped silently or with a message: class id at or above `classes`; width below `1 / net_w` or height below `1 / net_h`, which covers zero and negative sizes; `x <= 0`, `x > 1`, `y <= 0`, `y > 1`.
- Width or height above 1 is clipped to 1 and kept.
- A negative class id is not tested here.

Truths, in the layer (`src/yolo_layer.c`, `process_batch`):

- The list ends at the first row with `x` equal to 0.
- A row with a class id below 0 or at or above `classes` is skipped.
- A row outside [0, 1] or with negative size prints a warning and is still used.
- The cell is `(int)(x * w)`, `(int)(y * h)`, not clamped. `x = 1` passes the loader and gives column `w`, one past the row: the write lands in column 0 of the next row, or past the buffer in the last row. An implementation should clamp or reject; Darknet does neither.

Grid and stride. The layer stores no stride. It knows its grid from the extents of its input and the network size from `[net]`; the decode divides by the grid, the anchors by the network size. A stride is only ever the quotient of the two.

Suppression exponent. `beta_nms` is any float; nothing validates it. It is used as the exponent of `d / c`, which lies in [0, 1].

Weights header. Nothing validates `major`, `minor` or `revision`. A negative `major` or `minor` makes `major * 10 + minor >= 2` false, so `seen` is read as 32 bits, and the transposed-matrix test (above 1000) is false.

cfg text (`src/option_list.c`, `read_option` 57, `option_find` 96, `option_unused` 84; `src/parser.c`, `read_cfg` 1800, lines 1358, 1546, 1651):

- A line is skipped when, after all white space is removed, it is empty or starts with `#` or `;`.
- There is no comment syntax after a value. The value is everything after the first `=`. A trailing `#text` survives into it: a number still parses, because number parsing stops at the first character that is not part of a number, and the `mask` and `anchors` lists stop counting at `#`; a string value such as an activation name does not match and breaks.
- A line whose only `=` is its last character is reported as unparsable and skipped. A line with no `=` is stored as a key without a value.
- A repeated key in one section: the first occurrence wins; the later one is reported as unused.
- An unknown key: reported on standard error as an unused field after parsing; parsing continues.
- An unknown section: a message is printed and an empty layer takes its index; parsing continues. The section name must match exactly, so trailing text after `]` makes it unknown.
- The first section must be `[net]` or `[network]`, or Darknet stops. A key before any section is not guarded.

## 19. The anchor command

`detector calc_anchors <data> -num_of_clusters K -width W -height H` (`src/detector.c`, `calc_anchors`, line 1440; `src/matrix.c`, `do_kmeans` 301, `dist` 283, `random_centers` 260, `kmeans_maximization` 223). `K` defaults to 5; `W` and `H` are required.

What it does:

- Reads the label file of every image in the `train` list of the data file and collects each box as `(w * W, h * H)`: label sizes are fractions, scaled by the given network size as a plain stretch. Positions, images and classes play no part in the clustering.
- Runs k-means on those pairs. Distance between two sizes is `1 - IoU` of the two boxes sharing a corner. Start: `K` distinct boxes drawn at random from the data, with the random generator seeded from the clock, so two runs differ. The banner says k-means++, but the centres are a plain random draw. Update: each centre becomes the arithmetic mean of its members' widths and heights; a centre that ends up at zero keeps its old value. It stops when no assignment changes, or after 1000 rounds.
- Sorts the centres by area, ascending.
- Prints the mean best IoU of the boxes against the final centres as a percentage, the number of rounds, and the per-class box counts; writes `anchors.txt` (rounded to whole pixels when `W > 32`, four decimals otherwise) and `counters_per_class.txt`.
- A label outside (0, 1] is reported and still used.

What it does not do: it does not write the cfg, choose `num`, split anchors over heads, write `mask`, look at the strides or grids, weigh boxes by anything, deduplicate, repeat the run to keep the best, or evaluate the anchors against the assignment rule of section 4. Ordering by area is the only step toward head assignment; the cfg author does the rest by hand. Anything in SushiAI beyond clustering by `1 - IoU` and sorting by area is its own.

## 20. `mish` against the formula

One `yolov4` layer, the smallest `mish` convolution: layer 87, `[512, 19, 19]`, 739 328 bytes per file. Darknet keeps the pre-activation input of a `mish` or `swish` convolution, so the patch writes both (`DARKNET_DUMP_ACTIVATION_LAYER=87`), into `out/yolov4/`:

| Name | Shape | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `layer_087_activation_input` | `[512, 19, 19]` | 739328 | `68b72b478c6233c5ce4abbc537f55fa1514d790ceaca89c8d00ef2456557e68f` |
| `layer_087_activation_output` | `[512, 19, 19]` | 739328 | `1f045b8e5d84a9550536c3aa6ae362367ae4dc61dd14667a3bc842071ab2a0e1` |

Inputs range from -9.797682 to 5.3294296; outputs from -0.30884358 to 5.3291817, mean -0.153938522. Over the 184 832 values:

- Against the formula of section 3.5 in double precision: largest absolute difference 1.03e-6. Against `x * tanh(log1p(exp(x)))` in double: the same 1.03e-6.
- Against the same formula evaluated in `float32`: 83.7 % of the values are bit-equal, largest difference 1.2e-6.
- The largest relative difference is 5e-4, at strongly negative inputs where the output is near zero. Darknet computes the hyperbolic tangent as `2 / (1 + exp(-2 s)) - 1`, which cancels in `float32` when `s` is small. Compare with an absolute tolerance of a few 1e-6, not a relative one.

The other files of `out/yolov4/` were re-hashed after this run and are unchanged, except `run_log.txt`, which holds a timing.

## 21. Training steps for the other four models

Same procedure, recipe and file set as section 13, batch 2, `subdivisions` 1, no update. All four ran: `yolov3` in 10 s, `yolov4-tiny` in 2 s, `yolov4` in 20 s, `yolov7` in 12 s, on a machine with 15 GB of memory. `layer_000_output` was left out for `yolov4` and `yolov7` (94 MB and 105 MB) to stay inside the size budget; every other file of the set is there. The batch for 608 and 640 comes from the section 13 recipe at that size and is kept as `input` in each directory.

Derived cfgs (first `batch=` line set to 2, `subdivisions` to 1, nothing else):

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `yolov3_batch_2.cfg` | 8 342 | `99b172e546651e2137dc22a80eab89280b4a62e64758f8eddc21bf5f3106a372` |
| `yolov4-tiny_batch_2.cfg` | 3 230 | `3beb41eccf737f3077cdca013939f4a485ab127917b0b5fc46d0e4c282f843ce` |
| `yolov4_batch_2.cfg` | 12 230 | `55e80e4e2a92902cb0cd5adbc1b3916b0a614f1b0761f4acf2c910d608d0d7eb` |
| `yolov7_batch_2.cfg` | 10 766 | `fc53c40b49e462dacf97997452e08fb2f66995126459d42d24b10a8eb0e94682` |

Checks (`check_train_step.py`), largest deviation:

| Model | Rolling mean, variance | Head weight updates vs `sum(d * input)` | Head bias updates | Network cost vs mean of layer costs |
| --- | --- | --- | --- | --- |
| `yolov3` | 3.6e-8, 1.2e-8 | 5.8e-7, 9.1e-7, 1.0e-6 | 1.3e-7, 1.7e-7, 2.9e-7 | 4.55193281 / 4.55193305 |
| `yolov4-tiny` | 4.8e-8, 2.5e-8 | 6.8e-7, 1.3e-6 | 1.7e-7, 1.4e-7 | 7.96330929 / 7.96330905 |
| `yolov4` | 7.2e-8, 2.0e-7 | 3.2e-7, 1.3e-6, 9.0e-7 | 1.6e-7, 3.4e-7, 2.3e-7 | 12.1158934 / 12.1158938 |
| `yolov7` (logistic slope applied) | 1.2e-8, 1.8e-9 | 1.1e-6, 1.0e-6, 2.9e-7 | 3.1e-7, 3.2e-7, 1.1e-7 | 11.8744097 / 11.8744094 |

`layer_000_bias_updates` is exactly zero in all four, as in section 13.

A quirk of this build. `yolov4_train_step/layer_139_yolo` holds 49 NaN among 2 945 760 values. Every one is a class entry whose input is below -88.7 (inputs from -121.09 to -89.21); the most negative input that still gives a number is -88.70. That is where `exp(-x)` overflows `float32`, and under `/fp:fast` the logistic then yields NaN where the mathematical value is 0. Darknet leaves the class delta at 0 for a NaN result, so `layer_139_yolo_delta` has no NaN and equals what a correct logistic would give there. No other dumped file contains a NaN (all were scanned). A parity test on that one file must treat NaN as 0.

### yolov3_train_step (81 702 212 bytes of `.bin`; learning_rate 0.001, momentum 0.9, decay 0.0005)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 32]` | 128 | -0.825920343 | 0.575327396 | -0.094553517 | `a4d65e6c12cb2b08b64365797c1055eeaf8990a5d92a5d53ee257a89da8b5954` |
| `batch_variance` | `[1, 32]` | 128 | 0.0205821786 | 0.882144868 | 0.27155171 | `437c5479f2ab593080f1e00d62fe9731548c0ec1800832310fe1070b5dd5b66c` |
| `input` | `[2, 3, 416, 416]` | 4153344 | 0 | 1 | 0.500035784 | `7b7563a106509747180c421f8fd1be6cec581bcf8aaeed22274976adbb383036` |
| `layer_000_bias_updates` | `[1, 32]` | 128 | 0 | 0 | 0 | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| `layer_000_output` | `[2, 32, 416, 416]` | 44302336 | -1.33403933 | 10.6902962 | 0.759437917 | `712ad3a9749bc2e880147941232d0f7cf7f6a99160f493c892cbdbb200ddf416` |
| `layer_000_scale_updates` | `[1, 32]` | 128 | -4.33340983e-06 | 2.60572597e-06 | -2.76385137e-07 | `fee1ac16e6cae90f93b020bcda6f8f54a945396a1dded7840621558e461a06a2` |
| `layer_000_weight_updates` | `[32, 3, 3, 3]` | 3456 | -6.66875894e-06 | 6.96195548e-06 | -3.10366914e-08 | `f4fe1a7d423340369f0189b3280b4ff4168591b2b64f09ab2040edb023b763c9` |
| `layer_080_output` | `[2, 1024, 13, 13]` | 1384448 | -1.21769607 | 11.5567951 | 0.423962955 | `9a3b0adaf24678007f90c898f2997c653460c10ac977591d0ead0ab666f77b2b` |
| `layer_081_bias_updates` | `[1, 255]` | 1020 | -1.23069179 | 1.79814315 | 0.0089695411 | `74dc01abc6a48336b779b83e49a7895cf3773728fa443b80646d0623542733eb` |
| `layer_081_weight_updates` | `[255, 1024, 1, 1]` | 1044480 | -2.92711735 | 4.41978788 | 0.00263034801 | `eb0d03213d3329696b28e494fd2cf25c0f5453021a84c858dea544400e1a394b` |
| `layer_081_yolo_input` | `[2, 255, 13, 13]` | 344760 | -60.6563034 | 4.06008816 | -7.16453498 | `62576753a959cae110e036b162057b16ff11898022163de046cd5a920b3a9abe` |
| `layer_082_yolo` | `[2, 255, 13, 13]` | 344760 | -1.28249705 | 1.43396962 | 0.0214046511 | `c3eee521ac303d81b3e2b896ad7c60afa345555e3fa31303e1dbf37defa262f9` |
| `layer_082_yolo_cost` | `[1, 1]` | 4 | 10.6514587 | | | `2b01c16546a06d217979c55248d12cadb4570c4d4c69bbb5af7ab1727330cf32` |
| `layer_082_yolo_delta` | `[2, 255, 13, 13]` | 344760 | -0.524310946 | 0.999990642 | 2.65371053e-05 | `6032254c1f546c2489cd01192a8b7090705f9bf58f22e3e4458251b436e0c159` |
| `layer_092_output` | `[2, 512, 26, 26]` | 2768896 | -1.6850332 | 20.6851559 | 0.506221189 | `e47ccfd35640e6eb3cb75563401b5946ffe20f2dad3fc45dc6edf5adce45d199` |
| `layer_093_bias_updates` | `[1, 255]` | 1020 | -0.358004987 | 0.997660339 | 0.00260661082 | `811ceb26d34d425774e4312dc2454bf0baeb032b7ca7e8d2227f89655f458d2f` |
| `layer_093_weight_updates` | `[255, 512, 1, 1]` | 522240 | -0.953438044 | 2.07162499 | 0.00126099336 | `311cd195c184a59ded2be50ec2a4506e134e75fc5b7f4cd303241df8a9e35027` |
| `layer_093_yolo_input` | `[2, 255, 26, 26]` | 1379040 | -68.8421326 | 10.6541281 | -6.8030996 | `c64eb4cb4e3861fb8e6e298117856d9e280897857a036dcb3004f9740eda2fba` |
| `layer_094_yolo` | `[2, 255, 26, 26]` | 1379040 | -1.84795308 | 1.85544276 | 0.0220679497 | `69fa6b9b7f0bfa25d00ba6aaab3796c3aae17d9c710201bc4f8cbc932a4cd4c5` |
| `layer_094_yolo_cost` | `[1, 1]` | 4 | 2.9240768 | | | `575178c5d682fbe207bbf86419920c21d3fc768f3df6e96193cc796d3f31a41b` |
| `layer_094_yolo_delta` | `[2, 255, 26, 26]` | 1379040 | -0.358004987 | 0.999998927 | 1.92796712e-06 | `dcc1e99d0045e85f61e3c987a97bd62d0303ca52fa4d8d310c80305ca6479b04` |
| `layer_104_output` | `[2, 256, 52, 52]` | 5537792 | -2.48770523 | 30.036272 | 0.61528292 | `664b373da6e20f7e995a1706cf7268cbde7f000847b305e9ccac0bede74851e5` |
| `layer_105_bias_updates` | `[1, 255]` | 1020 | -0.90175271 | 0 | -0.0056862408 | `1ad70351cf02881e80e291eb2cdc8db4f50af457405c62309a8bf5d05c8c5504` |
| `layer_105_weight_updates` | `[255, 256, 1, 1]` | 261120 | -3.14908028 | 0.266939789 | -0.00204633404 | `7ef0961d644dc5dfe73678e1afe91b20f98429b72fe7548a2e707ab5b09ce7bb` |
| `layer_105_yolo_input` | `[2, 255, 52, 52]` | 5516160 | -86.8985443 | 29.3514042 | -7.40007407 | `9a76e8e95456d92d904705dc9aaa4a41bb96405f5ebf61ce84b41461f46aa39c` |
| `layer_106_yolo` | `[2, 255, 52, 52]` | 5516160 | -4.65608454 | 4.26966524 | 0.0181380606 | `03881996c7126e77f7c4186fca961a6d01671d22fb4444aeeec7cfe3189fd720` |
| `layer_106_yolo_cost` | `[1, 1]` | 4 | 0.0802636072 | | | `4d874cc66c8c96361821f2317a91633543cd387caf9754aab9cb45e4cda7f32a` |
| `layer_106_yolo_delta` | `[2, 255, 52, 52]` | 5516160 | -0.241720378 | 0 | -1.05145007e-06 | `9b51457d60215935099c247e4b07219262501753835987bba2de4c4b551689cf` |
| `network_cost` | `[1, 1]` | 4 | 4.55193281 | | | `05735461aea2f80076c77ab6d3c0f77075507d6435b66af21cd9c653046b2087` |
| `rolling_mean_after` | `[1, 32]` | 128 | -0.791963875 | 0.552391827 | -0.0881575195 | `837e825f27a3c9653d18cdb5dc2dc2b77df71546230da8c7955ba7e30cc66c41` |
| `rolling_mean_before` | `[1, 32]` | 128 | -0.788190961 | 0.551960528 | -0.0874468533 | `77460ac899c69c76ffaba433c8ac2555290265765576f1d470f9093fc9689d91` |
| `rolling_variance_after` | `[1, 32]` | 128 | 0.0205737483 | 0.292335719 | 0.103262428 | `27e17a334e3bd98833ed282b538d6a7cc4495459f07f2d0770081b7802bddbbf` |
| `rolling_variance_before` | `[1, 32]` | 128 | 0.00214239652 | 0.292869359 | 0.0845636203 | `5d765f03cc6dbbff63ed8b471780fbebda53f952a762945d454c962b809ef2da` |

The first convolution's updates are of order 1e-6 here, against 0.1 to 10 in the other models. I did not look for the reason; the cfg has no `stopbackward`.

```
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 82 Avg (IOU: 0.632826), count: 4, class_loss = 4.285794, iou_loss = 1.039935, total_loss = 5.325729
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 94 Avg (IOU: 0.549921), count: 1, class_loss = 1.123072, iou_loss = 0.338967, total_loss = 1.462038
v3 (mse loss, Normalizer: (iou: 0.75, obj: 1.00, cls: 1.00) Region 106 Avg (IOU: 0.000000), count: 1, class_loss = 0.040132, iou_loss = 0.000000, total_loss = 0.040132
```

### yolov4_tiny_train_step (23 267 012 bytes of `.bin`; learning_rate 0.00261, momentum 0.9, decay 0.0005)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 32]` | 128 | -0.92403686 | 1.12113893 | -0.0895710696 | `e5c78739ad1765399e079d8f7eb1a2ddccef66bd96a38cacab5082f41bf39ca8` |
| `batch_variance` | `[1, 32]` | 128 | 0.00754783768 | 0.970219851 | 0.329844763 | `957d3a5e36eee3308d32787b334bd4f3eeeb732d062519ff425faaae9bb39d70` |
| `input` | `[2, 3, 416, 416]` | 4153344 | 0 | 1 | 0.500035784 | `7b7563a106509747180c421f8fd1be6cec581bcf8aaeed22274976adbb383036` |
| `layer_000_bias_updates` | `[1, 32]` | 128 | 0 | 0 | 0 | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| `layer_000_output` | `[2, 32, 208, 208]` | 11075584 | -2.19677925 | 16.7529621 | 0.672322045 | `7244d5b65d91ab0a5fcc52150e0580142e13fb99ced9603c1572c71973b400ad` |
| `layer_000_scale_updates` | `[1, 32]` | 128 | -3.0261364 | 2.32638693 | 0.0926359822 | `767dcac751396bbda9d44f8342d89ca7d954d36db17bb6a27c9a037f07bd6272` |
| `layer_000_weight_updates` | `[32, 3, 3, 3]` | 3456 | -13.2529964 | 15.9260521 | 0.075996557 | `be5b2fd272ff6eccb58634b2ff37c5afd3b588ad4462e1c90d9b1a279cf4fda7` |
| `layer_028_output` | `[2, 512, 13, 13]` | 692224 | -1.14610803 | 9.66847229 | 0.403434556 | `bd8e0da6e61b9383d28a10d210de3ce5a952ca1fc2459d882fb667b2ac854596` |
| `layer_029_bias_updates` | `[1, 255]` | 1020 | -0.735277057 | 0.9851138 | 0.0104990257 | `cea2668635958a85ef4fa038e5e9524afbd36c1f947ca467d4e1220e48b027a8` |
| `layer_029_weight_updates` | `[255, 512, 1, 1]` | 522240 | -4.73640823 | 5.94756222 | 0.00386814919 | `edb5c2d9b8f546b34985fd019fdbbd983332094f87513649c858605553b2a45c` |
| `layer_029_yolo_input` | `[2, 255, 13, 13]` | 344760 | -38.4170837 | 4.11102676 | -6.48877487 | `e56533eeedcf5df1b02b754d4d73fdad6529130c3f08662df9678a56393a90a3` |
| `layer_030_yolo` | `[2, 255, 13, 13]` | 344760 | -1.56632376 | 1.20744073 | 0.0206213466 | `113fefc0e0e01bd7d5e618f334635dd85be739d8cff047da2fd54b62b8fa2afd` |
| `layer_030_yolo_cost` | `[1, 1]` | 4 | 9.44664097 | | | `cabab3f345b99b53f2977573cf2f1d077c59c65fc0248153240ec7044463b9ab` |
| `layer_030_yolo_delta` | `[2, 255, 13, 13]` | 344760 | -0.824786961 | 0.999999583 | 3.10622026e-05 | `825fb2900ba688d97baee66c3b90e8c99fb3bbe88170e52dc57c888f24f3601a` |
| `layer_035_output` | `[2, 256, 26, 26]` | 1384448 | -0.869753301 | 10.4237146 | 0.488171287 | `2fc0bc2a40c6d6310e17e6d50de956cb8c576c00401c6c683cf69653869c7304` |
| `layer_036_bias_updates` | `[1, 255]` | 1020 | -1.22999346 | 1.16564274 | 0.00597841695 | `e84d3e947ccf7198042e337fb77d0b54bac51b758f49eecd0c30f736d62ccf43` |
| `layer_036_weight_updates` | `[255, 256, 1, 1]` | 261120 | -5.89411306 | 5.00641727 | 0.00422349784 | `3e2c0127aa22b4269081469377d3420a47f8849640ea17f58749b89ac5a81d74` |
| `layer_036_yolo_input` | `[2, 255, 26, 26]` | 1379040 | -38.2666321 | 5.63766003 | -6.70029549 | `428818124ce7281fe7a494885473840a30782abc5ca041d5b3d32ecbd493e5e5` |
| `layer_037_yolo` | `[2, 255, 26, 26]` | 1379040 | -1.47480309 | 1.49849796 | 0.0223370562 | `50c61d0dc8a1e04c9a708b3245a76912a3bc21f8037bb1892aebff9749d9875e` |
| `layer_037_yolo_cost` | `[1, 1]` | 4 | 6.47997713 | | | `764549a1814bed224d8add48ab3f38f0e161b250cf8f8a8e22cf665032f093cc` |
| `layer_037_yolo_delta` | `[2, 255, 26, 26]` | 1379040 | -1.22999346 | 1 | 4.42190583e-06 | `ae6f1b0a5a3bba25b53c91bd80d59871929fc717b485691dad8692d755424770` |
| `network_cost` | `[1, 1]` | 4 | 7.96330929 | | | `4b80051352f4b21bf38e34c103510e6656e68fffeb9d1f036613e4d0b61a9bd9` |
| `rolling_mean_after` | `[1, 32]` | 128 | -0.780066788 | 1.06339359 | -0.0981364004 | `7aee95368276df6da09ae4f1be20357aec5b5936b7a30a4568964d6403fe4cf4` |
| `rolling_mean_before` | `[1, 32]` | 128 | -0.764070094 | 1.05697739 | -0.0990881119 | `308ea913bd4902b9d77ab0c9a01082f4a10f4fa1a176b81d1c9f0c0245bc93e5` |
| `rolling_variance_after` | `[1, 32]` | 128 | 0.0430846848 | 0.431661993 | 0.16171436 | `734ba2d679b57b265bf77ab7612666c68fc2bc081a7c71d07afaeaf156a4980e` |
| `rolling_variance_before` | `[1, 32]` | 128 | 0.00913077779 | 0.395603091 | 0.143033209 | `5d7e529adf9d12f73c1021412fd603285b0d6b63ef5c389236b5aec1baa2d662` |

```
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 30 Avg (IOU: 0.514073), count: 4, class_loss = 4.706313, iou_loss = 0.327233, total_loss = 5.033546
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 37 Avg (IOU: 0.674733), count: 3, class_loss = 3.228604, iou_loss = 1.346554, total_loss = 4.575159
```

### yolov4_train_step (77 804 356 bytes of `.bin`; learning_rate 0.0013, momentum 0.949, decay 0.0005; no `layer_000_output`)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 32]` | 128 | -1.808599 | 1.92694926 | -0.000444416197 | `4e6ffe46f7fcd5c5cd94065968bbd0c2c9d09754deaaffd0b510eef7532d50a5` |
| `batch_variance` | `[1, 32]` | 128 | 0.00145650236 | 6.3291769 | 1.67370414 | `d1766ab881a4e37832a3c30b4a04df66cc772cb6f0533ea7f14a8782921605ac` |
| `input` | `[2, 3, 608, 608]` | 8871936 | 0 | 1 | 0.499941621 | `f83b41004e60d3d1f160b979d121cddb58350237b71de46b4a5559ce2244464b` |
| `layer_000_bias_updates` | `[1, 32]` | 128 | 0 | 0 | 0 | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| `layer_000_scale_updates` | `[1, 32]` | 128 | -0.0865207985 | 0.0495658368 | 2.03963809e-05 | `47f065a06f8cdc8281b119d359dded9e19e8e0b66c7dc132f3d49cead23a9520` |
| `layer_000_weight_updates` | `[32, 3, 3, 3]` | 3456 | -0.179587483 | 0.0914738029 | -0.000155693846 | `6a29b2d2f61f5e297ecab91a1c8a8cef12baadfde7130472f690a561be6309f5` |
| `layer_137_output` | `[2, 256, 76, 76]` | 11829248 | -12.6184406 | 44.5065689 | 0.26587585 | `7ed022dc8301f9c3cfdbfff7c330a53f6f14a24fad7ed5c03f64dd8e24a1ff08` |
| `layer_138_bias_updates` | `[1, 255]` | 1020 | -4.01882076 | 0 | -0.0316438422 | `f383f4b6f02f618d36bd2595a740923115b231f161be0d048f780826d1dca7e1` |
| `layer_138_weight_updates` | `[255, 256, 1, 1]` | 261120 | -3.7060585 | 0.666231275 | -0.00311850076 | `fa4a95214fe46c7c8f14bdd4058dbb2b64bd66ef146191b11bb9989e5b50f7bd` |
| `layer_138_yolo_input` | `[2, 255, 76, 76]` | 11783040 | -121.091614 | 10.796751 | -7.09564231 | `a5c167d37e8cc224eb3f987ef4a8c211798d2d0bdd31800043707f2e64b2a002` |
| `layer_139_yolo` (49 NaN) | `[2, 255, 76, 76]` | 11783040 | | | | `d214db0f17eccca881d4c9446eb3fdbdb5910d004746f504e6aa4f9f07516772` |
| `layer_139_yolo_cost` | `[1, 1]` | 4 | 0.395545483 | | | `7eac0de8cfb6e0a97efa9fa3ec264b5197dce820de8b6577f67dd6080eccb50c` |
| `layer_139_yolo_delta` | `[2, 255, 76, 76]` | 11783040 | -0.326673448 | 0 | -2.73925235e-06 | `7e2b9bdffc91b437614c1fe1491bcff596b7ad3bf5139a5e7bd762cd212f0173` |
| `layer_148_output` | `[2, 512, 38, 38]` | 5914624 | -5.40936852 | 33.3811646 | 0.197715624 | `e827a77ce5e76338889efa13c17bd8cce8c2117b5bf8d766c2eefbb00786668a` |
| `layer_149_bias_updates` | `[1, 255]` | 1020 | -2.68337297 | 1.3329972 | -4.09236988e-05 | `2a28f1e7b4b28428b4691e4d06b7554e6d76789edad71317045753c58b025a64` |
| `layer_149_weight_updates` | `[255, 512, 1, 1]` | 522240 | -2.13048792 | 2.98437357 | 0.00440575801 | `a9b4c0e6fced933c35ea228d44e6591503d7a32539ab8041bfd3a3288228cdae` |
| `layer_149_yolo_input` | `[2, 255, 38, 38]` | 2945760 | -74.2264786 | 10.438118 | -7.63408398 | `f2247ca136aeaed4319cc6aa8f1905984de0d6b1d3aa42d7e22b0a6bab5529b4` |
| `layer_150_yolo` | `[2, 255, 38, 38]` | 2945760 | -4.7988677 | 2.41801715 | 0.0254339359 | `b17b81cc4be73f5663de9fedbf3f56401c7930517f9697f50b7ea37c83b733cd` |
| `layer_150_yolo_cost` | `[1, 1]` | 4 | 15.0076895 | | | `6ec71451ab4bdff204333e1a2abb738b3bee254d33fdb24bd0e7909cdabcd349` |
| `layer_150_yolo_delta` | `[2, 255, 38, 38]` | 2945760 | -0.774721026 | 0.999998093 | -1.41707085e-08 | `0997ac4f77d368e242de5db5633fed82940c6b79076ce5be4d7fc2ba9688322b` |
| `layer_159_output` | `[2, 1024, 19, 19]` | 2957312 | -3.40499473 | 28.9078064 | 0.183057072 | `a5a3b4e2eb04eb4930a93d8ba126b7597d3f6c572cb63539b7c19e65171334e6` |
| `layer_160_bias_updates` | `[1, 255]` | 1020 | -0.975110292 | 1.8579402 | 0.0126094966 | `09bd3de18d990f4fb617e336b3ed0b7bb00199a2fc397382578c3ac97767c22c` |
| `layer_160_weight_updates` | `[255, 1024, 1, 1]` | 1044480 | -2.81816149 | 4.37316942 | 0.00384405929 | `4c6dcc012d5c1cf4376d12196ae025b7f9109feae00a66bc9abc0566f2bcf0a0` |
| `layer_160_yolo_input` | `[2, 255, 19, 19]` | 736440 | -48.4606743 | 14.6299171 | -7.45708193 | `1c5d49fa66e002fbad237c4996be1f9071bf457b0178a01a32708adbf05eb60f` |
| `layer_161_yolo` | `[2, 255, 19, 19]` | 736440 | -3.02021289 | 2.1755774 | 0.0152260745 | `8f86322453d323fecf21f8a009f870f8562c14da965829ff6a989855bf2414b0` |
| `layer_161_yolo_cost` | `[1, 1]` | 4 | 20.9444466 | | | `48c087dc17645da84878491bd16834b727522e05b3267b6d5e2b3aee1912ccf6` |
| `layer_161_yolo_delta` | `[2, 255, 19, 19]` | 736440 | -0.953395545 | 0.999998391 | 1.74646746e-05 | `035e04e710f1ffb36ad2e512456fd9b629ac5b4ccad5934faad6bbc2bd03704d` |
| `network_cost` | `[1, 1]` | 4 | 12.1158934 | | | `e6dfa773c11480ce7563875f8565d7596e5bd40e17fee241ed329fc65ada9e23` |
| `rolling_mean_after` | `[1, 32]` | 128 | -1.47389162 | 1.82498217 | 0.00482733879 | `020d657e5444dcf790c058ba6756d5e727821a31725443238e880187cd7198fb` |
| `rolling_mean_before` | `[1, 32]` | 128 | -1.43670189 | 1.81365252 | 0.00541309621 | `f3000c373df69cf00e2306e54490d808b8ce2bcc8e996547a5fd1790a8627602` |
| `rolling_variance_after` | `[1, 32]` | 128 | 0.00300991512 | 2.34199667 | 0.749540856 | `ad8406790398fb59cfca7d2eef1820783cf72dbd5528e4d35f21c67197bd74c2` |
| `rolling_variance_before` | `[1, 32]` | 128 | 0.00318251667 | 2.09600306 | 0.646856052 | `095e765b38c7bfad6e09075b8eb8aa9baacac853aa382913db8401b18890441e` |

```
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 139 Avg (IOU: 0.000000), count: 1, class_loss = 0.197773, iou_loss = 0.000000, total_loss = 0.197773
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 150 Avg (IOU: 0.453356), count: 6, class_loss = 7.484712, iou_loss = 1.896934, total_loss = 9.381646
v3 (iou loss, Normalizer: (iou: 0.07, obj: 1.00, cls: 1.00) Region 161 Avg (IOU: 0.395026), count: 9, class_loss = 10.451049, iou_loss = 0.389297, total_loss = 10.840345
```

### yolov7_train_step (86 011 516 bytes of `.bin`; learning_rate 0.00261, momentum 0.9, decay 0.0005; no `layer_000_output`)

| Name | Shape | Bytes | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| `batch_mean` | `[1, 32]` | 128 | -0.219165415 | 0.270234704 | -0.0133869264 | `79746e4e6e6a8f4bc6166b2e6c7726a10557bd31f14f299b4e900b37dd9b89fc` |
| `batch_variance` | `[1, 32]` | 128 | 0.00175370963 | 0.126476899 | 0.0432153958 | `45038f07005c5d6a88ce11c575e74aa28cd193167b626b083f2cb2c609e8b6b1` |
| `input` | `[2, 3, 640, 640]` | 9830400 | 0 | 1 | 0.499923213 | `462eb4f97a655fd0c5c7d888eab55dbed0d45bf208ff969984fdedc0cf74a14a` |
| `layer_000_bias_updates` | `[1, 32]` | 128 | 0 | 0 | 0 | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| `layer_000_scale_updates` | `[1, 32]` | 128 | -0.00830242131 | 0.0085215196 | 0.000503225431 | `bde2cb76cca8191cbeddc20ce9d13cbcc4fda6ebed8ef6908a1fe56efd321a7f` |
| `layer_000_weight_updates` | `[32, 3, 3, 3]` | 3456 | -0.120069996 | 0.100294963 | 0.00117627134 | `bb59c2accf96f965c6d67b25b3ff2c2a0b8ef485320d1ba061e6a73e899740d9` |
| `layer_132_output` | `[2, 256, 80, 80]` | 13107200 | -0.278464556 | 21.2574291 | 0.656168833 | `67ed11c48c133fe89974e6a3591b22943568ca425c25428a8657595ea407a185` |
| `layer_133_bias_updates` | `[1, 255]` | 1020 | -0.793023884 | 0 | -0.00706977634 | `8da5139834a24df616c7ad02f6dc4c8cf079b03a9b397255a417d0c2888bfba2` |
| `layer_133_weight_updates` | `[255, 256, 1, 1]` | 261120 | -4.16600466 | 0.154136285 | -0.00616115814 | `68d4c24b7918da3b6e4f08274dfb44ddcbd680442d75e61f156a4ae2f138aeaa` |
| `layer_133_yolo_input` | `[2, 255, 80, 80]` | 13056000 | 9.67126544e-19 | 0.998885095 | 0.0319109824 | `319fe4ea88c48dd2813f94671b6bb213ff926b63663005c9685868eac98ef4b5` |
| `layer_134_yolo` | `[2, 255, 80, 80]` | 13056000 | -0.387484789 | 1.40034163 | 0.0319472654 | `527962db514242df8926c6db5d1d3149fdd7b2afd57db2e9fed87fbfd6969510` |
| `layer_134_yolo_cost` | `[1, 1]` | 4 | 2.03216767 | | | `100ce9c93b49ef35e8ca70c49fb630ec4f5524f085d20c22cee234db107f6f7b` |
| `layer_134_yolo_delta` | `[2, 255, 80, 80]` | 13056000 | -0.252521008 | 0 | -1.07865802e-05 | `0399448980c8fafa31b604b3a6fb77d0ba437c89a0f70bfc94b3a79b6f2a4c1d` |
| `layer_136_output` | `[2, 512, 40, 40]` | 6553600 | -0.278464556 | 16.2083778 | 0.437475369 | `23e1b8e5c94df8f033e46870ee31721a9ca9601050c3f91b5daa04cc2dc26390` |
| `layer_137_bias_updates` | `[1, 255]` | 1020 | -2.67797613 | 0.119615234 | -0.0273514561 | `dfe5b19ffcf7bc10618fb6c6068c1409705fd07d07006a70ddc83d93e852374a` |
| `layer_137_weight_updates` | `[255, 512, 1, 1]` | 522240 | -7.27674866 | 0.735461652 | -0.0137457679 | `c52878363daee91e596efcea1e815c3bcb71dbc8f76c7d84bc2747cf5d13ce65` |
| `layer_137_yolo_input` | `[2, 255, 40, 40]` | 3264000 | 5.22590807e-24 | 0.999866486 | 0.0314928842 | `93c2555c4e18f05020e8949c667db9f16842e173d86380e88bc224c29456d95f` |
| `layer_138_yolo` | `[2, 255, 40, 40]` | 3264000 | -0.422270596 | 1.40814781 | 0.03143261 | `936e7148ac26b0064145b64642edcb77b8b742b6bf0d573735625deb54a489f5` |
| `layer_138_yolo_cost` | `[1, 1]` | 4 | 17.5290718 | | | `7d9d9ca6cee1469e611ab4d4e9222aa111a33221dfc57ac6f77e785de8075ad5` |
| `layer_138_yolo_delta` | `[2, 255, 40, 40]` | 3264000 | -0.802548051 | 0.999624848 | -5.79927902e-05 | `49bf725af07a7fb4f9b43a2eff5a80e19f332cc7a3157719f72b890bf82c7047` |
| `layer_140_output` | `[2, 1024, 20, 20]` | 3276800 | -0.278464556 | 16.7893257 | 0.269453315 | `d4ccfed4279b34b27403408cb7cf7133a343c59bfd5bd683e723997cdb6a23e9` |
| `layer_141_bias_updates` | `[1, 255]` | 1020 | -0.956705928 | 0.0626785532 | -0.0101967976 | `6675d46c9a36889574b8aca3bb7b5c374117ab30f41cd22d6c69379a1288f70a` |
| `layer_141_weight_updates` | `[255, 1024, 1, 1]` | 1044480 | -3.25752163 | 0.622430086 | -0.00213415826 | `fa01ae3eff70c362c67e484074e01262f230f40f6160cb8509f41c0c3172bc51` |
| `layer_141_yolo_input` | `[2, 255, 20, 20]` | 816000 | 6.82800077e-15 | 0.999947906 | 0.0296005533 | `f01dc6e465500f50a797721c5a49b51205c32f46799db725f19ebca253502933` |
| `layer_142_yolo` | `[2, 255, 20, 20]` | 816000 | -0.432404965 | 1.43087089 | 0.0294549541 | `06c8b386292d366c35b64a49f978dc52f68e067ee9d9e3dec922d1dcbec84796` |
| `layer_142_yolo_cost` | `[1, 1]` | 4 | 16.0619888 | | | `b137ed965c2ed6e72fe9e5bcb978f59ed92508507cb39b8df4f4864fa573683f` |
| `layer_142_yolo_delta` | `[2, 255, 20, 20]` | 816000 | -0.867874563 | 0.99972862 | -5.36265388e-05 | `6a2db8d229064382ff4848f3a4e902b6192b474ad86112c028831052ad929186` |
| `network_cost` | `[1, 1]` | 4 | 11.8744097 | | | `66aa041febc8f64f84b62df1da1f942a475ab34e3f0bef9f69cdf3e829d08f95` |
| `rolling_mean_after` | `[1, 32]` | 128 | -0.20297122 | 0.249826193 | -0.0134676555 | `91d283ffc2427d03da97e60a81c142fa35ebceea9bb7553f6ac057b74d77ee54` |
| `rolling_mean_before` | `[1, 32]` | 128 | -0.201171875 | 0.247558594 | -0.0134766251 | `23999a9800916726bf2b825d32db546fd98b054148e12a1f0c61ea37ab69ddd2` |
| `rolling_variance_after` | `[1, 32]` | 128 | 0.0029469938 | 0.0351911858 | 0.0114601919 | `f5203a1c612063214f4f3bb8b2c8ab1cd76b692e945845dd4e4d2af2a0a35c71` |
| `rolling_variance_before` | `[1, 32]` | 128 | 0.000277042389 | 0.0290985107 | 0.00793183595 | `cffdcb342e9fa5e428758d31dd613a27b41f62f994923e7b5db0ec2bd6c211f2` |

```
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 134 Avg (IOU: 0.000000), count: 1, class_loss = 1.016084, iou_loss = 0.000000, total_loss = 1.016084
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 138 Avg (IOU: 0.350895), count: 6, class_loss = 8.748308, iou_loss = 0.456499, total_loss = 9.204807
v3 (iou loss, Normalizer: (iou: 0.05, obj: 1.00, cls: 0.50) Region 142 Avg (IOU: 0.297029), count: 9, class_loss = 8.013420, iou_loss = 0.120522, total_loss = 8.133943
```

`truth` in all four directories is the 120-byte file of section 13 (`89cec5c9fccba5b3f630907f27ee13e58a6fe59decb892aa2a42d739df21f994`).

## 22. Every head

Built from the cfgs with the defaults of section 17 (`_darknet/heads_table.py`). The loss settings of every row were compared with the values Darknet parsed, which the dumps hold in `layer_<k>_yolo_delta.json`; all agree. Stride is network width over grid width. `truth_thresh` is 1 and `iou_thresh_kind` is `iou` everywhere.

| Model | Layer | Grid h, w | Stride | mask | Anchors (px) | new_coords | scale_x_y | iou_loss | iou_thresh | ignore_thresh | nms_kind | beta_nms | max_delta | objectness_smooth | iou / obj / cls normalizer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `yolov3-tiny` | 16 | 13, 13 | 32 | 3,4,5 | 81x82, 135x169, 344x319 | 0 | 1 | mse | 1 | 0.7 | default | 0.6 | unset | 0 | 0.75 / 1 / 1 |
| `yolov3-tiny` | 23 | 26, 26 | 16 | 0,1,2 | 10x14, 23x27, 37x58 | 0 | 1 | mse | 1 | 0.7 | default | 0.6 | unset | 0 | 0.75 / 1 / 1 |
| `yolov3` | 82 | 13, 13 | 32 | 6,7,8 | 116x90, 156x198, 373x326 | 0 | 1 | mse | 1 | 0.7 | default | 0.6 | unset | 0 | 0.75 / 1 / 1 |
| `yolov3` | 94 | 26, 26 | 16 | 3,4,5 | 30x61, 62x45, 59x119 | 0 | 1 | mse | 1 | 0.7 | default | 0.6 | unset | 0 | 0.75 / 1 / 1 |
| `yolov3` | 106 | 52, 52 | 8 | 0,1,2 | 10x13, 16x30, 33x23 | 0 | 1 | mse | 1 | 0.7 | default | 0.6 | unset | 0 | 0.75 / 1 / 1 |
| `yolov4-tiny` | 30 | 13, 13 | 32 | 3,4,5 | 81x82, 135x169, 344x319 | 0 | 1.05 | ciou | 1 | 0.7 | greedynms | 0.6 | unset | 0 | 0.07 / 1 / 1 |
| `yolov4-tiny` | 37 | 26, 26 | 16 | 1,2,3 | 23x27, 37x58, 81x82 | 0 | 1.05 | ciou | 1 | 0.7 | greedynms | 0.6 | unset | 0 | 0.07 / 1 / 1 |
| `yolov4` | 139 | 76, 76 | 8 | 0,1,2 | 12x16, 19x36, 40x28 | 0 | 1.2 | ciou | 0.213 | 0.7 | greedynms | 0.6 | 5 | 0 | 0.07 / 1 / 1 |
| `yolov4` | 150 | 38, 38 | 16 | 3,4,5 | 36x75, 76x55, 72x146 | 0 | 1.1 | ciou | 0.213 | 0.7 | greedynms | 0.6 | 5 | 0 | 0.07 / 1 / 1 |
| `yolov4` | 161 | 19, 19 | 32 | 6,7,8 | 142x110, 192x243, 459x401 | 0 | 1.05 | ciou | 0.213 | 0.7 | greedynms | 0.6 | 5 | 0 | 0.07 / 1 / 1 |
| `yolov7-tiny` | 90 | 52, 52 | 8 | 0,1,2 | 10x13, 16x30, 33x23 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |
| `yolov7-tiny` | 94 | 26, 26 | 16 | 3,4,5 | 30x61, 62x45, 59x119 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |
| `yolov7-tiny` | 98 | 13, 13 | 32 | 6,7,8 | 116x90, 156x198, 373x326 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |
| `yolov7` | 134 | 80, 80 | 8 | 0,1,2 | 12x16, 19x36, 40x28 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |
| `yolov7` | 138 | 40, 40 | 16 | 3,4,5 | 36x75, 76x55, 72x146 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |
| `yolov7` | 142 | 20, 20 | 32 | 6,7,8 | 142x110, 192x243, 459x401 | 1 | 2 | ciou | 0.2 | 0.7 | diounms | 0.6 | 2 | 1 | 0.05 / 1 / 0.5 |

Three things the table shows that are easy to miss: the head order is coarse-to-fine in the v3 and v4-tiny cfgs and fine-to-coarse in `yolov4` and both v7; `yolov4-tiny` layer 37 has `mask = 1,2,3`, so anchor 3 belongs to both heads and anchor 0 to none; `yolov4` uses a different `scale_x_y` per head.

## 23. Corrections to sections 4 and 11

- Section 11 said `max_delta` is unset in the v4 cfgs. It is unset in `yolov4-tiny` and 5 in `yolov4`. The delta of 1.92664182 at `yolov4` layer 150 is inside that bound.
- Sections 4 and 11 named 0.213 as the `iou_thresh` "in v4". That holds for `yolov4` only; `yolov4-tiny` has the default 1, so it assigns no extra anchors and skips step 3. Its second head still shows two assignments in section 14 because of the shared anchor above.
- The `truth_thresh` statement of section 4 now covers all six cfgs: it is 1 in every head.

## 24. Files and sizes after the third pass

- New directories: `yolov3_train_step`, `yolov4_tiny_train_step`, `yolov4_train_step`, `yolov7_train_step`. New files in `yolov4`: the two of section 20.
- `out/SHA256SUMS.txt` now lists 994 files.
- `_darknet/out` is 569 MB; `_darknet` is 591 MB (`src` 17 MB, `build` 5.5 MB), inside the 600 MB budget with 9 MB to spare. Another pass needs `out` moved first.
- The patch is 377 diff lines (334 added), SHA-256 `5e721e000eb370bf146840b39671ee19e964744169753545009fda44a2cdeba8`. New switches: `DARKNET_DUMP_ACTIVATION_LAYER=<k>` and `DARKNET_TRAIN_STEP_NO_FIRST_OUTPUT`. With them unset the earlier files are written unchanged: the 728 files of the second pass were re-checked after a rerun of `yolov4`, and only its `run_log.txt` (a timing) differed.

## 25. A defect in Darknet's CPU max pooling, and what it does to the references

Section 3.4 said inference uses a second pooling implementation "with the same offset rule". The offset rule is the same; the border handling is not. This section corrects that.

Where. `src/maxpool_layer.c`, `forward_maxpool_layer`, line 276: in inference, with equal strides, the layer calls the routine in `src/gemm.c`, `forward_maxpool_layer_avx`, line 1870. For stride 1 on a CPU with AVX that routine handles eight output columns at a time (lines 1886-1904), then finishes the row one column at a time.

What goes wrong. A block of eight starts at output column `j0 = 0, 8, 16, ...` and runs while `j0 < out_w - 8 - (size - 1)`. Inside a block, each tap of the window is tested for being inside the input once, for the block's first column, and a tap that fails is skipped for all eight columns. With `p = padding / 2` columns of padding on the left:

- In the block at `j0 = 0`, taps `0 .. p - 1` are outside the input for column 0, so they are skipped for columns 1 to 7 as well, where some of them are inside.
- Output column `j`, for `1 <= j <= 7`, therefore reads input columns `j .. j + size - 1 - p` instead of `j - p .. j + size - 1 - p`. Rows are handled correctly.
- Column 0 and every column from 8 on are correct. Later blocks start inside the input, and the right border is handled by the one-column loop.

When it triggers: stride 1, inference, AVX available, and `out_w > size + 7`, so that at least one block runs.

| Model | SPP map | size 5 | size 9 | size 13 |
| --- | --- | --- | --- | --- |
| `yolov7-tiny` (416) | 13 x 13 | wrong in columns 1-7 | correct | correct |
| `yolov4` (608) | 19 x 19 | wrong in columns 1-7 | wrong in columns 1-7 | correct |
| `yolov7` (640) | 20 x 20 | wrong in columns 1-7 | wrong in columns 1-7 | correct |
| `yolov3-tiny` (416) | 13 x 13, size 2 | correct: its left padding is 0, so no tap is ever outside on the left | | |

`yolov3` and `yolov4-tiny` have no stride-1 pool. The size-2, stride-2 pools take another branch and recompute exactly.

Measured on the dump of `yolov7-tiny` (`_darknet/check_layers.py`), each pool recomputed from its dumped input with the rule of section 3.4:

| Layer | Largest difference, default path | Largest difference, plain path |
| --- | --- | --- |
| 36, size 5 | 3.697319746 (9110 of 43 264 cells differ, all in columns 1 to 7; each of those columns equals the maximum over input columns `j .. j + 2` and rows `i - 2 .. i + 2`) | 0 |
| 38, size 9 | 0 | 0 |
| 40, size 13 | 0 | 0 |
| 9, size 2, stride 2 | 0 | 0 |

What is and is not affected.

- Affected: every inference-mode dump of `yolov4`, `yolov7` and `yolov7-tiny` from the first three passes: heads, `[yolo]` outputs, decoded rows, detections, and the training-mode `[yolo]` deltas of section 2.3, which were computed on those heads.
- Not affected: everything of `yolov3-tiny`, `yolov3`, `yolov4-tiny`; all six batch-statistics training steps (sections 13 and 21), because training mode takes the layer's own loop; `layer_087` of `yolov4` (section 20), which lies before the SPP block.
- Size of the effect on the raw heads, default path against plain path: `yolov7-tiny` 0.143, 0.249, 0.220; `yolov7` 0.183, 0.292, 0.448; `yolov4` 1.565, 1.760, 3.859. These are the magnitudes SushiAI was off by.
- The GPU pooling kernel was not read. The layer's own loop, which training uses, follows section 3.4.

The plain path. The patch gained one switch in a second file (`src/maxpool_layer.c`): with `DARKNET_PLAIN_MAXPOOL` set, inference uses the layer's own loop. With it unset the binary reproduces `yolov7_tiny_layers` byte for byte (246 files compared after the rebuild). The patch is now 398 diff lines over two files, SHA-256 `f0cf21bb3f28f56c56c0e5d2292940078df9b217c1f7a4e82520da9119c54dad`.

Which reference to use is the owner's decision. The `*_plain_pool*` directories below are what section 3.4 describes and what SushiAI should equal if it implements that rule. The earlier directories are what a CPU-built Darknet prints on an AVX machine. Reproducing the defect in SushiAI would mean copying a bug.

## 26. Border rules, stated exactly and checked

Max pooling, stride 1, size `s` (5, 9, 13): padding is `s - 1` in total, `(s - 1) / 2` before and the same after, since `s` is odd. Output cell `(i, j)` is the maximum over input rows `i - (s - 1) / 2 .. i + (s - 1) / 2` and columns `j - (s - 1) / 2 .. j + (s - 1) / 2`, intersected with the input; a position outside never wins. The output has the input's extent. Checked: table above, plain path, difference 0 for all three.

Route with several inputs: the output's channels are the inputs' channels laid end to end, in the order the cfg lists them, not in layer order. Layer 41 lists `-1,-3,-5,-6`, which is layers 40, 38, 36, 35: channels 0-255 are the size-13 pool, 256-511 the size-9 pool, 512-767 the size-5 pool, 768-1023 the unpooled map. Checked by concatenating the dumped inputs: difference 0 for layer 41, for layer 7 (inputs 2, 4, 5, 6) and for layer 70 (inputs 69, 56, a relative and an absolute index mixed).

Upsample: layer 46 equals layer 45 with every value repeated twice along rows and columns; difference 0.

Convolution with `size=3`, `stride=2`, `pad=1`: padding is 1 on every side. Output cell `(i, j)` reads input rows `2i - 1 .. 2i + 1` and columns `2j - 1 .. 2j + 1`; the first tap of the first cell is at index -1, in the padding. With an even input extent the last cell reads rows `H - 3 .. H - 1`, so the padding after the last row and column is never read. Checked on layer 0 by reading its weights and statistics from the `.weights` file and computing the layer in double precision (convolution, batch norm with epsilon 1e-5 inside the root, leaky): largest difference 8.1e-6 against the dump, and `output[0, 0, 0]` 0.91194731 against 0.91194725. With the padding placed after only (first tap at index 0) the difference is 24 and the first value 0.0167. A convolution with `size=1` and `pad=1` has padding 0, because padding is `size / 2`.

## 27. `yolov7-tiny`: every layer

`out/yolov7_tiny_layers/`, 416 x 416, the section 2.1 input, inference, batch norm folded, default pooling path. 99 layers; every shape below was compared with the dump's JSON. File `layer_<k>_<type>.bin` holds `4 * C * h * w` bytes; the directory holds 85 244 348 bytes of `.bin`, of which the 99 layer outputs are 72 222 488, the figure section 9 predicted. Every route has `groups` 1 and `group_id` 0. No `padding` key is written on any max pool. `bn` is `batch_normalize`.

The 27 files this directory shares with the earlier `yolov7_tiny` (input, the three head inputs and outputs, deltas, assignments, decoded rows, detections, truth) have the same SHA-256 as in the 994-line list: 54 `.bin` and `.json` files compared, all equal.

| Layer | Type | Keys | Output `[C, h, w]` | min | max | mean | SHA-256 of `layer_<k>_<type>.bin` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | convolutional | filters 32, size 3, stride 2, pad 1 (1 per side), leaky, bn 1 | `[32, 208, 208]` | -3.25297332 | 26.6152515 | 0.914168501 | `6f2658103110c109c682788192dfcaaeeacf8e8037b8465db3ddbdb421a6f3ba` |
| 1 | convolutional | filters 64, size 3, stride 2, pad 1 (1 per side), leaky, bn 1 | `[64, 104, 104]` | -3.48546457 | 38.3997002 | 1.3533651 | `d875ceed3bd78a2aee69a499f3c3364c124c56ab676f1e3db0cc1c964890dea1` |
| 2 | convolutional | filters 32, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[32, 104, 104]` | -2.54590392 | 26.0234032 | 0.529891711 | `7c0e378d9b25ee20c987b0857c8bb6b62af7071880e2e89b6f18de38a919809e` |
| 3 | route | layers -2 -> 1 | `[64, 104, 104]` | -3.48546457 | 38.3997002 | 1.3533651 | `d875ceed3bd78a2aee69a499f3c3364c124c56ab676f1e3db0cc1c964890dea1` |
| 4 | convolutional | filters 32, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[32, 104, 104]` | -2.73239183 | 15.6770372 | 0.52010961 | `ba232b1ae95e0d64a67766fbb57204d13266a977c8f7b8e2f3adf9d58ec1ff23` |
| 5 | convolutional | filters 32, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[32, 104, 104]` | -2.6756475 | 16.2227688 | 0.570644605 | `3b41d58f025303f4675d9759ae44abac0e741167044603f5bd636470f0b3e5a8` |
| 6 | convolutional | filters 32, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[32, 104, 104]` | -2.08770418 | 11.4448986 | 1.00395808 | `08005bbeb6318a308fae0a391058453d115d5bc30791575d81f7e59df673b224` |
| 7 | route | layers -5,-3,-2,-1 -> 2, 4, 5, 6 | `[128, 104, 104]` | -2.73239183 | 26.0234032 | 0.656151002 | `da2ad90cc3edaa156efa71fd6f6508dba33a43d4020cc4476d7116354a5b5408` |
| 8 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 104, 104]` | -1.85142982 | 17.6683102 | 0.742621737 | `c616e55407c2b1cfcc7c95a78760ac320f5af9efa6826510369c1e5f15945ebe` |
| 9 | maxpool | size 2, stride 2, padding 1 total (0 before, 1 after), first window at 0 | `[64, 52, 52]` | -1.17073321 | 17.6683102 | 1.26177337 | `add4756c775542637f8f76981b56b96bb514bf668b2a41100f01aa0aa80fc385` |
| 10 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 52, 52]` | -1.19710529 | 14.7395821 | 0.493997301 | `ac621c3483db0219d656b130f363ade9b8013036e665e98462697d1bf24dec30` |
| 11 | route | layers -2 -> 9 | `[64, 52, 52]` | -1.17073321 | 17.6683102 | 1.26177337 | `add4756c775542637f8f76981b56b96bb514bf668b2a41100f01aa0aa80fc385` |
| 12 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 52, 52]` | -1.13198698 | 11.0223866 | 0.661258157 | `4d48c7fe1a197dbd4232e29cbe44723ea42666381db6a91b50b00052bba0a6dd` |
| 13 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 52, 52]` | -1.31379998 | 13.6470833 | 0.174889297 | `bc3a8e13d369d83e66fa1df669fc16e8df2137fee8a033a9d88260d2a13a989f` |
| 14 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 52, 52]` | -1.06378973 | 13.2747498 | 0.460082042 | `8d18bdbb70122096071d817090e0998706ba0bf050092731fb4a192aaf2afd68` |
| 15 | route | layers -5,-3,-2,-1 -> 10, 12, 13, 14 | `[256, 52, 52]` | -1.31379998 | 14.7395821 | 0.447556699 | `5b2a983bc3f7b5922b1217bc0d067e9db4bbe841d8d3752a9d2f147efe3bb5fa` |
| 16 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 52, 52]` | -1.67432153 | 13.8384848 | 0.170441149 | `fe4e009747f90ef66f2493ec443cca3969f81dee6c8de57cf20f353b8f86d472` |
| 17 | maxpool | size 2, stride 2, padding 1 total (0 before, 1 after), first window at 0 | `[128, 26, 26]` | -1.08588886 | 13.8384848 | 0.392495838 | `e3a994b24e97f5acbf814bf716c9d702695b5ea10be34b63ebbbb4a265a91485` |
| 18 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 26, 26]` | -0.651999891 | 7.62829685 | 0.291544986 | `e1d102cbb7b004cec0a4eb8d96f350351489872e623223b3d9648bdd2ada6820` |
| 19 | route | layers -2 -> 17 | `[128, 26, 26]` | -1.08588886 | 13.8384848 | 0.392495838 | `e3a994b24e97f5acbf814bf716c9d702695b5ea10be34b63ebbbb4a265a91485` |
| 20 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 26, 26]` | -1.10680473 | 6.1606555 | 0.0793852068 | `7ae203828604f434496d0ed848618a91fa37e60451fc8986529787e86ce324e3` |
| 21 | convolutional | filters 128, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[128, 26, 26]` | -0.789677322 | 5.20680904 | 0.0642220043 | `bdda26abf43825c7566198122183771d3f8c405a3c327ad502c5490642c827fc` |
| 22 | convolutional | filters 128, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[128, 26, 26]` | -0.747401714 | 8.35359955 | 0.205167344 | `792fc3fcc339cf6b78764893bc4cff76fa050a3571176148e28324ddc76f62a2` |
| 23 | route | layers -5,-3,-2,-1 -> 18, 20, 21, 22 | `[512, 26, 26]` | -1.10680473 | 8.35359955 | 0.160079885 | `ca984ffd2300f3a73cf4b1193b86a4e9a311244072836db6c65736e8550458ff` |
| 24 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 26, 26]` | -0.75498718 | 7.57341623 | 0.016887986 | `b4e440e2dc8f10f45aca61a96d769434223d1b384e4e7e207895f41c82853817` |
| 25 | maxpool | size 2, stride 2, padding 1 total (0 before, 1 after), first window at 0 | `[256, 13, 13]` | -0.496919394 | 7.57341623 | 0.12870278 | `2c0a5dc543f4c85c743a1975e9dfdb36e081a5fd91a8f584c55874ccaec80a8b` |
| 26 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.606224716 | 6.27018833 | 0.196835915 | `3fd87ab9cfa614fa37358c9c5248b31f8631e95922fcf21f091d9043969c6451` |
| 27 | route | layers -2 -> 25 | `[256, 13, 13]` | -0.496919394 | 7.57341623 | 0.12870278 | `2c0a5dc543f4c85c743a1975e9dfdb36e081a5fd91a8f584c55874ccaec80a8b` |
| 28 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.42688784 | 3.65867949 | 0.00706621002 | `cad6842609767196119a8793d67ef8b22a856fc1bae79b2afc609fdb53f77760` |
| 29 | convolutional | filters 256, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[256, 13, 13]` | -0.499327719 | 3.56019568 | 0.0457020174 | `dee5d0eff313ad523e7b964a506ff0abd9ac1d6923d9923afc5661527492944a` |
| 30 | convolutional | filters 256, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[256, 13, 13]` | -0.578418672 | 4.47313309 | 0.137137956 | `0cd5d578c28ecfec5848e6db9134f184ab846a64056d6332dac0fc93d196fd5a` |
| 31 | route | layers -5,-3,-2,-1 -> 26, 28, 29, 30 | `[1024, 13, 13]` | -0.606224716 | 6.27018833 | 0.0966855247 | `68a35c5feade5dddccddcab3c9fd35937e84e5bd47440656be7a9731c3efd4c5` |
| 32 | convolutional | filters 512, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[512, 13, 13]` | -0.558000743 | 3.93006849 | 0.0834438995 | `fb7b407b335b602e9572c28cef6f11826f15100bf3352a059223389a21180270` |
| 33 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.407811731 | 4.09909344 | 0.0513682383 | `1c28501fa58eb941838b7fdc1fe84692e8cb9ef06ad1290b7702783f8fe337de` |
| 34 | route | layers -2 -> 32 | `[512, 13, 13]` | -0.558000743 | 3.93006849 | 0.0834438995 | `fb7b407b335b602e9572c28cef6f11826f15100bf3352a059223389a21180270` |
| 35 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.615638435 | 5.87882328 | 0.209671163 | `b93dcffebf83b62865a2eda1a5b257a134cae025f8db7f2b1ef283e0563ae555` |
| 36 | maxpool | size 5, stride 1, padding 4 total (2 before, 2 after), first window at -2 | `[256, 13, 13]` | -0.294125766 | 5.87882328 | 0.63806357 | `a4ccc8a42f8d21023c259b65fbf7131184c3bdda028979e025cc4edf0b6d4296` (defective path) |
| 37 | route | layers -2 -> 35 | `[256, 13, 13]` | -0.615638435 | 5.87882328 | 0.209671163 | `b93dcffebf83b62865a2eda1a5b257a134cae025f8db7f2b1ef283e0563ae555` |
| 38 | maxpool | size 9, stride 1, padding 8 total (4 before, 4 after), first window at -4 | `[256, 13, 13]` | -0.21193257 | 5.87882328 | 0.956591842 | `38d4957d1b332e56d2bd6ccfaf2503778f0b4c2003aff0cede427e1c147b26e3` |
| 39 | route | layers -4 -> 35 | `[256, 13, 13]` | -0.615638435 | 5.87882328 | 0.209671163 | `b93dcffebf83b62865a2eda1a5b257a134cae025f8db7f2b1ef283e0563ae555` |
| 40 | maxpool | size 13, stride 1, padding 12 total (6 before, 6 after), first window at -6 | `[256, 13, 13]` | -0.187990367 | 5.87882328 | 1.12313272 | `33e72a723e54b419f2266191e6374e09ad5d8614e99484b22ea961b17fb49f0a` |
| 41 | route | layers -1,-3,-5,-6 -> 40, 38, 36, 35 | `[1024, 13, 13]` | -0.615638435 | 5.87882328 | 0.731864825 | `4e10042ab176916a555e96291fe403bf6b23338d30c6b2f6aca04f3ecda63b14` |
| 42 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.522141635 | 2.83669281 | -0.00480134933 | `5e6c746ba8a2662bcddd71cca2fd3548e12494d3e5cc16dd08e824a75865021a` |
| 43 | route | layers -10,-1 -> 33, 42 | `[512, 13, 13]` | -0.522141635 | 4.09909344 | 0.0232834445 | `f378be0c396b2850e2ffadfcb314ae0bd78c95dfa1f6fb748ab4fc39816364e2` |
| 44 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.539699078 | 3.53327084 | 0.0866131305 | `363d0cb076a98921a80c7454371f775adc77c8d7c06d3049866b3f006b863337` |
| 45 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 13, 13]` | -0.427424908 | 3.98749924 | 0.278791225 | `d183e82e7c5e8d05edbdd59f79a2368e7c0d0598b586ed4d075cfabc70c60818` |
| 46 | upsample | stride 2 | `[128, 26, 26]` | -0.427424908 | 3.98749924 | 0.278791225 | `55c6107cdd6b0aa567ddfc5eb00526512a54cf74f32748165a0e733412c049df` |
| 47 | route | layers 24 -> 24 | `[256, 26, 26]` | -0.75498718 | 7.57341623 | 0.016887986 | `b4e440e2dc8f10f45aca61a96d769434223d1b384e4e7e207895f41c82853817` |
| 48 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 26, 26]` | -0.76914072 | 4.57938766 | 0.214671863 | `19a52a9f14ed23f7a72b30655a769c1fdc107650361a6216fbe967141789c4a6` |
| 49 | route | layers -1,-3 -> 48, 46 | `[256, 26, 26]` | -0.76914072 | 4.57938766 | 0.246731544 | `54828cd8f34cd792e7e8c16cffc9015923efca8cb4a4ee7d77c4e351d3f6c5bc` |
| 50 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 26, 26]` | -0.451583833 | 4.50717211 | 0.150817767 | `2ce32911b72c4391f9084d21a834ce6e7caa0f69cb29f0cd3b50ef2e98b85e44` |
| 51 | route | layers -2 -> 49 | `[256, 26, 26]` | -0.76914072 | 4.57938766 | 0.246731544 | `54828cd8f34cd792e7e8c16cffc9015923efca8cb4a4ee7d77c4e351d3f6c5bc` |
| 52 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 26, 26]` | -0.50212425 | 2.97415352 | 0.262340772 | `edc85e6fae000578e50656e9d4d78cc0a5eedb478deffb7347cfa600277a63c6` |
| 53 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 26, 26]` | -0.490791231 | 2.09351683 | 0.0442920174 | `866d862d9c703be6f6bfd2db4b3f7f1dab6475a1b51a3d04a51f61699bfefc9d` |
| 54 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 26, 26]` | -0.347845703 | 3.17029476 | 0.0940979396 | `dc8abbd9a2aa3008bcd3c87d5466326018476301a30c93a492fba445a794875a` |
| 55 | route | layers -5,-3,-2,-1 -> 50, 52, 53, 54 | `[256, 26, 26]` | -0.50212425 | 4.50717211 | 0.137887124 | `f82fae3d8c34c01fbb6354c6055af94c291142752b7519bedfc9213a6a4a33b9` |
| 56 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 26, 26]` | -0.306569755 | 2.81682754 | 0.258521416 | `73c9737b472986071a00782f16d6eb81f2dc3c6fea43aae082311d417c22cba3` |
| 57 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 26, 26]` | -0.339276075 | 3.06734657 | 0.462728737 | `7bd88724077dd03512f2ce1dae62a688b3ddd7783ef8f3e9cf57158a71a0e73e` |
| 58 | upsample | stride 2 | `[64, 52, 52]` | -0.339276075 | 3.06734657 | 0.462728737 | `a0b9bfc3d4b8244bd219919b5b7ba7125c4ebffcf768916a8460168d47cef1dd` |
| 59 | route | layers 16 -> 16 | `[128, 52, 52]` | -1.67432153 | 13.8384848 | 0.170441149 | `fe4e009747f90ef66f2493ec443cca3969f81dee6c8de57cf20f353b8f86d472` |
| 60 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 52, 52]` | -0.970102727 | 9.06834602 | 0.501249825 | `a950ff61a1de195493ee8a7e547d8bc11b9c7c544bb362abeed2119c124ebbc4` |
| 61 | route | layers -1,-3 -> 60, 58 | `[128, 52, 52]` | -0.970102727 | 9.06834602 | 0.481989281 | `d934e5215322471a21cda320e5bf29ed3551c6262ca5e3a458f5f25472ce4140` |
| 62 | convolutional | filters 32, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[32, 52, 52]` | -0.649720669 | 4.02220392 | 0.617484744 | `216ccf589b97a297960b31766e187c91565a14858f40b7aff4f02e6e6f51e012` |
| 63 | route | layers -2 -> 61 | `[128, 52, 52]` | -0.970102727 | 9.06834602 | 0.481989281 | `d934e5215322471a21cda320e5bf29ed3551c6262ca5e3a458f5f25472ce4140` |
| 64 | convolutional | filters 32, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[32, 52, 52]` | -0.646864951 | 4.92564774 | 0.538270521 | `53ff248cca6f03df7c9c0f639de0fd2dd6927505e1edd7964d8647cec82ecc7c` |
| 65 | convolutional | filters 32, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[32, 52, 52]` | -0.516792417 | 4.28896904 | 0.118686452 | `80300b5901434b39624507ed8b62450c0bbbb07b70ae9d506e801251bc065fd3` |
| 66 | convolutional | filters 32, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[32, 52, 52]` | -0.659052074 | 6.78535175 | 0.33765615 | `47f2a6b3d6846188aa56362ce32312c592928a9ea741dd95cc98974b57af2578` |
| 67 | route | layers -5,-3,-2,-1 -> 62, 64, 65, 66 | `[128, 52, 52]` | -0.659052074 | 6.78535175 | 0.403024467 | `eba7fdab53507b5662cdca0dc3e73726b1bc6c8535c3434e2bcdaf032952999b` |
| 68 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 52, 52]` | -0.674794614 | 5.13327074 | 0.55726828 | `f6f44d0f5369afd86d87db05d0305c6100367bb4c61460668f4084a63bb03862` |
| 69 | convolutional | filters 128, size 3, stride 2, pad 1 (1 per side), leaky, bn 1 | `[128, 26, 26]` | -0.582573354 | 3.80255222 | 0.0407975785 | `743b099aa72a13b29c2dcc61259df9d66ceb1a822202383aee375dfc0753f925` |
| 70 | route | layers -1,56 -> 69, 56 | `[256, 26, 26]` | -0.582573354 | 3.80255222 | 0.149659497 | `3ac7399e9de02c6147e534eff772a9c56c27576c919b02358c010ff029c7390a` |
| 71 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 26, 26]` | -0.327524036 | 3.25308371 | 0.240402502 | `ca6ef8726f303776eeb68301d0c86618633716575303820b6917af3b346ff722` |
| 72 | route | layers -2 -> 70 | `[256, 26, 26]` | -0.582573354 | 3.80255222 | 0.149659497 | `3ac7399e9de02c6147e534eff772a9c56c27576c919b02358c010ff029c7390a` |
| 73 | convolutional | filters 64, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[64, 26, 26]` | -0.352882475 | 2.80834866 | 0.192239494 | `3bc45e50c5fc335a5dd0555fbd6c3cb803945e5e4bc5dad9deb7c8637f860074` |
| 74 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 26, 26]` | -0.525291622 | 3.11334276 | 0.0440795814 | `db2576c5fe3fbdb6b3b8d692e4f67d8c87c47d188af60222a00dd7dcfdfc2f79` |
| 75 | convolutional | filters 64, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[64, 26, 26]` | -0.539809883 | 3.90393949 | 0.250275943 | `e241f8ad8fe2accec4116e81bbc8a526d5aac077d2fec4512900366c9c172353` |
| 76 | route | layers -5,-3,-2,-1 -> 71, 73, 74, 75 | `[256, 26, 26]` | -0.539809883 | 3.90393949 | 0.18174938 | `192fbbf3505bf916d126773c442359caa0701a3937676b42a2e030e3a85eac2b` |
| 77 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 26, 26]` | -0.579899907 | 3.44631052 | 0.263736558 | `51c2505ef5d0a13475493092ad4e4567d0b3cded55b4a022b201b6037d80982b` |
| 78 | convolutional | filters 256, size 3, stride 2, pad 1 (1 per side), leaky, bn 1 | `[256, 13, 13]` | -0.384415746 | 3.19467831 | 0.0496673954 | `5f3e19cbc618cc08f36f73aa9d0183bed4a2182e7d6c4a185b4976768ba6c390` |
| 79 | route | layers -1,44 -> 78, 44 | `[512, 13, 13]` | -0.539699078 | 3.53327084 | 0.068140263 | `65da6390990ce0c571ee9bac7354c0215b4c179696d8379c9d3ed01a0e06f85d` |
| 80 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 13, 13]` | -0.301520228 | 3.46449637 | 0.289977411 | `b94289623560ea385b525c897fe4035ac1e7f6ac6f2efbf99747b70ac34bc787` |
| 81 | route | layers -2 -> 79 | `[512, 13, 13]` | -0.539699078 | 3.53327084 | 0.068140263 | `65da6390990ce0c571ee9bac7354c0215b4c179696d8379c9d3ed01a0e06f85d` |
| 82 | convolutional | filters 128, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[128, 13, 13]` | -0.301734298 | 2.44972587 | 0.0827794281 | `aa619645e21ee875b2dbebbe2ec6ec4a09614b924dabc9b48e3eaee0199b9877` |
| 83 | convolutional | filters 128, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[128, 13, 13]` | -0.325627536 | 2.19875431 | 0.0335912937 | `4938dad19a028edaf3c9b0eef58f2b676ad6688e7c4c53b9a3dac7fe292e2e33` |
| 84 | convolutional | filters 128, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[128, 13, 13]` | -0.50227809 | 3.46492171 | 0.141005863 | `d01e32d8b590d2197071738d08908357c6d6d56c0670761e1567b6991d92bb4a` |
| 85 | route | layers -5,-3,-2,-1 -> 80, 82, 83, 84 | `[512, 13, 13]` | -0.50227809 | 3.46492171 | 0.136838499 | `c30fc9477c59c058afb93c07e17a2c6d60c812475278020120be785d1f522397` |
| 86 | convolutional | filters 256, size 1, stride 1, pad 1 (0 per side), leaky, bn 1 | `[256, 13, 13]` | -0.52466923 | 4.66847229 | 0.172332178 | `fcab1f83dde55dbc1e4aa9d2ac1fb568acb33807383a230ca9b2436d673cb797` |
| 87 | route | layers 68 -> 68 | `[64, 52, 52]` | -0.674794614 | 5.13327074 | 0.55726828 | `f6f44d0f5369afd86d87db05d0305c6100367bb4c61460668f4084a63bb03862` |
| 88 | convolutional | filters 128, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[128, 52, 52]` | -1.64464784 | 17.4323769 | 1.26924496 | `51efae9c04063c842286e99a7fb4404146f694242353cb8afb8c56c360d1a1c7` |
| 89 | convolutional | filters 255, size 1, stride 1, pad 1 (0 per side), logistic, bn 0 | `[255, 52, 52]` | 6.93933515e-08 | 0.948225021 | 0.0364591207 | `4c04055f50409bceb5f64d913261173e4ea394c77eb3d22ae862a6c55bfd411a` |
| 90 | yolo | mask 0,1,2 | `[255, 52, 52]` | -0.193316162 | 1.26769745 | 0.0368043951 | `6fc024da8453a9715c3dd1f11b120110f4f921ce372de06e73a74a252474638b` |
| 91 | route | layers 77 -> 77 | `[128, 26, 26]` | -0.579899907 | 3.44631052 | 0.263736558 | `51c2505ef5d0a13475493092ad4e4567d0b3cded55b4a022b201b6037d80982b` |
| 92 | convolutional | filters 256, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[256, 26, 26]` | -1.20480406 | 9.68115807 | 0.697500661 | `e01ece1b7a0ace38716916a82681b0823b6263df365f703f84ef1fde55e52bcc` |
| 93 | convolutional | filters 255, size 1, stride 1, pad 1 (0 per side), logistic, bn 0 | `[255, 26, 26]` | 1.22107195e-07 | 0.930993378 | 0.0366641243 | `fe9dd886b76b7493a7f1fecc61446fabcfcb08e21d50bf454df9eca2b2029bff` |
| 94 | yolo | mask 3,4,5 | `[255, 26, 26]` | -0.33257103 | 1.31507206 | 0.0370520925 | `2075b49fd365fd0c32ab8cc61584afcea5fb111d8e4943a79ab463d1f902c97a` |
| 95 | route | layers 86 -> 86 | `[256, 13, 13]` | -0.52466923 | 4.66847229 | 0.172332178 | `fcab1f83dde55dbc1e4aa9d2ac1fb568acb33807383a230ca9b2436d673cb797` |
| 96 | convolutional | filters 512, size 3, stride 1, pad 1 (1 per side), leaky, bn 1 | `[512, 13, 13]` | -0.903474987 | 13.3566236 | 0.547234298 | `788b67bdceea36f1305fef3760bd3ebcf4f5f5a440de9cc47916934167be9412` |
| 97 | convolutional | filters 255, size 1, stride 1, pad 1 (0 per side), logistic, bn 0 | `[255, 13, 13]` | 9.64712488e-09 | 0.930199385 | 0.0298295907 | `7fc82c8e12b39703cd6567e4b03bb4c1d9a026e9396accb9aeac9d6e6a517cb4` |
| 98 | yolo | mask 6,7,8 | `[255, 13, 13]` | -0.301275015 | 1.31922197 | 0.0301769127 | `314e8ad160cab91e82ee634e609021d3d9950da540a2bdebf9a525c000b81128` |

In this directory layers 0 to 35, 37 to 40 follow section 3.4; layer 36 carries the defect, and every layer from 41 on that depends on it differs from the plain path. Layers 47, 59, 87 and the layers that read only from before layer 36 are unaffected.

## 28. Plain-path references for the three affected models

Written with `DARKNET_PLAIN_MAXPOOL=1`, otherwise as before. Per-file shapes, bytes, statistics and SHA-256 of all six new directories are in `out/TABLES_PASS4.md` (45 KB) and every hash is in `out/SHA256SUMS.txt`; the heads are repeated here.

| Directory | Content | Bytes of `.bin` |
| --- | --- | --- |
| `yolov7_tiny_layers` | every layer, default path (section 27) | 85 244 348 |
| `yolov7_tiny_layers_plain_pool` | every layer, plain path, threshold 0.25 | 85 244 348 |
| `yolov7_tiny_plain_pool_thresh_0p01` | first-pass file set, plain path | 16 642 176 |
| `yolov4_plain_pool`, `yolov4_plain_pool_thresh_0p01` | first-pass file set, plain path | 35 548 492, 35 548 632 |
| `yolov7_plain_pool_thresh_0p01` | first-pass file set, plain path | 39 389 124 |

In `yolov7_tiny_layers_plain_pool`, layers 0 to 35 and 37 to 40 are byte-identical to `yolov7_tiny_layers`; the first file that differs is `layer_036_maxpool.bin`.

| File | Shape | min | max | mean | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `yolov7_tiny_layers_plain_pool/layer_036_maxpool` | `[256, 13, 13]` | | | | `f701328a93a76c19571ff88223771120c5cbc19a7c46fc48f92891653913c2e0` |
| `.../layer_041_route` | `[1024, 13, 13]` | | | | `ec2fac582f75a3b0d80cae8ecfc5c63260ea7d27407a447b169c61b286247633` |
| `.../layer_042_convolutional` | `[256, 13, 13]` | | | | `df55d895080d7efbc5ef7d926f2b5d6caf275726df33d6adf0a0bfd67fc5cfbe` |
| `.../layer_089_yolo_input` | `[255, 52, 52]` | 6.28939532e-08 | 0.947838306 | 0.036531848 | `778dad4d007fe50cb66ada2e9d982e00afdeaf59876cb771cd87ff56d34d4826` |
| `.../layer_090_yolo` | `[255, 52, 52]` | -0.1882433 | 1.26681018 | 0.0368733522 | `2234765c66464c9ceb2ff9755f29ea1c8ec3254c6f0c88e9ecf62c329093e9ba` |
| `.../layer_093_yolo_input` | `[255, 26, 26]` | 7.76270923e-08 | 0.934977233 | 0.0366608781 | `fb1413dedebcb49912a842e69befe82f8bbebaedbda4e4cecf5acee26060ca8d` |
| `.../layer_094_yolo` | `[255, 26, 26]` | -0.333949983 | 1.33521724 | 0.0370387837 | `14677c188725e271c50004231cf49e13fe27c98008fc3f5760d95b8fdf1c2463` |
| `.../layer_097_yolo_input` | `[255, 13, 13]` | 3.25077432e-09 | 0.9207955 | 0.0296797664 | `80b88c4dccd7cee88b94ba0f6db539781a9f2393a6321c65f91e579d5cf00b09` |
| `.../layer_098_yolo` | `[255, 13, 13]` | -0.298743814 | 1.31911814 | 0.029974146 | `49d17e8c0f4229bbf4f954cae3aa330790f09fb68d931338f3989e586a179870` |
| `yolov4_plain_pool/layer_138_yolo_input` | `[255, 76, 76]` | -19.9334793 | 3.32445955 | -6.0723723 | `20f737e921873cd64ce4904758d8441f0cf6ffa21219456b1831a6611ed44a8e` |
| `.../layer_139_yolo` | `[255, 76, 76]` | -2.20020413 | 1.55204272 | 0.0238557158 | `2af95c66e90122a52f8b900a52fd866f4b4acd201c1f6380ac027c8dd9f9dcbd` |
| `.../layer_149_yolo_input` | `[255, 38, 38]` | -18.4016609 | 2.44846129 | -5.53230514 | `620c033b33e9a03f942add7634a338a2b4f1e52e2e19bb009994f1c84857756b` |
| `.../layer_150_yolo` | `[255, 38, 38]` | -1.63356149 | 1.44611228 | 0.0305583276 | `51e249339c6c2bb09f294dea5eafdcc347358429b5da97cbc9353ad794643553` |
| `.../layer_160_yolo_input` | `[255, 19, 19]` | -18.3739471 | 3.78838444 | -6.00890135 | `61744328f8f1b7c7c03beb22f153741698ca0034b92915cec02164ef1429601e` |
| `.../layer_161_yolo` | `[255, 19, 19]` | -1.50353241 | 1.46000004 | 0.0227780386 | `345c0f7104e6861dbb4f592a577837302bd3f4f2e98bd7043366b57f26add9c9` |
| `yolov7_plain_pool_thresh_0p01/layer_133_yolo_input` | `[255, 80, 80]` | 3.78334875e-09 | 0.976695359 | 0.0351272274 | `c80fd0ea07149926d6b2026921222e0313fa05415f5f42dd2b4723a89c3fb9b0` |
| `.../layer_134_yolo` | `[255, 80, 80]` | -0.293618619 | 1.20404863 | 0.0350866648 | `fcb9e62b3601983c35821e3b5ef49fdaa1e5801a3179b378ca6989351efd751b` |
| `.../layer_137_yolo_input` | `[255, 40, 40]` | 2.35281172e-09 | 0.958701372 | 0.0334698779 | `893faf29e5293e8c36fc7b9bef897c6945c319a038e1830a2d4553fc4de583c8` |
| `.../layer_138_yolo` | `[255, 40, 40]` | -0.318634033 | 1.40052402 | 0.0332446538 | `52c3b89582d78881dd355d9f7fa72683d1335b9eb9d0f07b2cebe4647e4c463e` |
| `.../layer_141_yolo_input` | `[255, 20, 20]` | 1.95132071e-08 | 0.973696887 | 0.0345568625 | `e1d6630b9025dfdc09789a9f80c16e1725c7bfa9ee03eef919c42b7a722d997d` |
| `.../layer_142_yolo` | `[255, 20, 20]` | -0.40904358 | 1.25043416 | 0.0346289467 | `845326a5ba1772922bb7a07f51ae145834a212d2f2b4c5657af0d79c8b941397` |

Detections on the plain path (`x y w h objectness score class`):

- `yolov7-tiny`, 0.25: 0 rows from 2 candidates. At 0.01: 12 rows from 203 candidates, file SHA-256 `516e2804425eed221c9064c8466d4dbb1b56d5df7505156a00a3b3c5f1cc7d69`.
- `yolov4`, 0.25: 1 row from 1 candidate, `0.496728897 0.501437247 1.07234502 1.01939344 0.946602404 0.315725982 0`, SHA-256 `8c49891a6a996e7bce62655c10308df708fcb512bee7f36db34f8c364123f732`. At 0.01: 6 rows from 11 candidates, SHA-256 `e2d0ead5561ac44bd2bd2cd63c26a143a9701d59f2ed9f57594fecd0dcc99219`.
- `yolov7`, 0.01: 9 rows from 53 candidates, SHA-256 `65347687afcbab913265f785155251c1fbcad3ed8878b5ca55bec596ea8b9c6d`.

The rows are in `out/TABLES_PASS4.md`.

Stored cost of the training-mode `[yolo]` pass on the plain-path heads: `yolov7-tiny` 1.05197299, 5.69186687, 7.90927315; `yolov4` 0.000435579073, 8.904109, 9.70098877; `yolov7` 0.00564183062, 4.12445354, 6.84989786.

## 29. Housekeeping of the fourth pass

- Copy check: `sha256sum -c` of the 994-line list inside `D:\Projects\.sushisystems\models\darknet\references\` reported 994 files `OK` and none failing. Only then were the 17 model directories under `_darknet/out` deleted. The top-level input and truth files were kept.
- `out/SHA256SUMS.txt` was regenerated and now has 726 lines, covering what `out` holds now. The 994-line list of the third pass is at `_darknet/SHA256SUMS_pass3.txt` and in `references`. Copy the new list under another name.
- `_darknet/out` is 300 MB; `_darknet` is 323 MB.
- A full-body dump of `yolov4` would be 692 320 024 bytes (section 9). It was not written and does not fit beside the current content.
- New helpers in `_darknet`: `layer_table.py`, `check_layers.py`. Scratchpad: nothing of mine.

## 30. Implementation: the six forward passes

Written on 2026-10-09 by workflow `wf_4451d03d-0c6` and two single dispatches, under the same rule on sources as section 15.

```
dispatch: shared forward runner — opus low, passed, rung 1
dispatch: T02 the IoU gradient in two forms — opus low, passed, rung 1
dispatch: A07 plan_anchors — opus low, passed, rung 1
dispatch: F15 to F19, forward parity of five models — haiku high, passed, rung 1
dispatch: review — opus low, passed, rung 1
dispatch: layer-by-layer parity over a table of models — opus low, passed, rung 1
dispatch: Darknet passes three and four — opus low, passed, rung 1
```

The first run against the references of the first three passes: `yolov3` and `yolov4-tiny` matched; `yolov4`, `yolov7` and `yolov7-tiny` were off by 0.1 to 4 at their heads. Section 25 is the cause: those references carried Darknet's pooling defect. Against the plain-path references of section 28 the same SushiAI code matches, with no change to the library. Three row counts of the detection cases, written from the defective dumps, were corrected to those of section 28 (6, 9 and 12 rows at threshold 0.01).

Last run, RTX 3080 Ti, SushiBLAS `983d430`, `SA_REFERENCE_DATA_DIR` set: `sushiai_tests` 1386 passed; `sushiai_vision_tests` 950 passed, 1 skipped.

| Model | Raw heads, largest difference | Detection rows at 0.25 and 0.01 | Largest row difference |
| --- | --- | --- | --- |
| `yolov3-tiny` | 3.8e-5 | 0, 16 | 1.3e-6 |
| `yolov3` | 5.4e-5 | 2, 3 | 2.0e-6 |
| `yolov4-tiny` | 5.1e-5 | 0, 16 | 1.2e-6 |
| `yolov4` | 7.8e-5 | 1, 6 | 1.7e-6 |
| `yolov7-tiny` | 5.6e-6 | 0, 12 | 2.0e-6 |
| `yolov7` | 5.8e-6 | none taken, 9 | 3.6e-7 |

The bound on a head is 1e-3 plus 1e-4 of the reference, and on a row 1e-4. Every body layer of `yolov3-tiny` (22) and of `yolov7-tiny` (its 99-layer plain-path dump) is compared one by one; the largest difference over all of them is 5.8e-5. Each model's Separate form, with batch norm kept as layers, computes the heads of its Folded form within the same bound.

The test support that only vision tests use sits in `tests/vision_support/`, so `tests/common/` still names no domain and the layering check stays clean without an exemption.

Open:

- The decode parity of `test_darknet_decode.cpp` still reads the first-pass directories of the three affected models. It compares SushiAI's decode with Darknet's decode of the same head tensor, so the defect upstream of the head does not affect what it pins.
- T07, the step schedule, can now be written from section 16.
- The training units (T03, T04, T10 to T16) are not written. The training-mode `[yolo]` deltas of `yolov4`, `yolov7` and `yolov7-tiny` from before the fourth pass were computed on defective heads; the plain-path directories hold their replacements.
- REPORT section 11 does not say what the gradient returns when the enclosing box has no diagonal.

## 31. Implementation: training, written and not yet built

Written on 2026-10-09 and 2026-10-10 by workflow `wf_424939ca-ab4` and three single dispatches; the workflow's units ran on the models PLAN section 4 names for them. Nothing in this section has been compiled or run: each file passed `clang++ -fsyntax-only` alone, and the layering, contract, documentation and comment checkers are clean. The build stopped twice outside SushiAI, first in SushiRuntime's working tree while it was being moved to modules, then in SushiBLAS, which still calls the members SushiRuntime marked deprecated.

```
dispatch: workflow wf_424939ca-ab4, units T03, T04, T07, T10 to T15 and their review — passed, rung 1
dispatch: the box gradient option and its tests — opus low, passed, rung 1
dispatch: compute_head_delta returns a status — opus low, passed, rung 1
dispatch: the one-batch overfit check — opus low, 22 turns, passed, rung 1
```

What was written:

- `Optim::StepSchedule` and `validate_step_schedule`: the rate of section 16 at an iteration, with the warm-up power and the step scales.
- The delta of a head whose box loss is an IoU form, and of a head in the activated coding. `HeadTrainingOptions::box_gradient` chooses between `BoxGradient::Analytic`, the derivative of the stated loss and the default of the library, and `BoxGradient::DarknetCompatible`, the three departures of section 11, which `read_darknet_training` selects so that an imported model trains as its file was trained.
- `compute_head_delta` returns a `Status`. A truth box whose centre lies outside the input, or whose width or height is not positive, is `InvalidDescription`; Darknet writes out of bounds on the first and the library does not reproduce that.
- `HeadTrainingOptions::class_weight` is removed: the key it carried has no effect in Darknet (section 17), and an option that does nothing is not offered.
- Six suites, `tests/integration/test_darknet_<model>_training.cpp`, over `tests/vision_support/darknet_training.hpp`. Each compares the settings read from the cfg, the delta of a dumped head, the delta of a two-image head, the heads under batch statistics, the deltas of the step, the updates of the head convolutions and of the first convolution, and the rolling statistics after one step. The last three are the comparison T16 asked for, on all six models.
- The one-batch overfit check of PLAN 4.7, one test per suite: 200 steps on the batch of the step run, at the fixed rate of that run's dump with no momentum and no decay, and the last summed cost under a tenth of the first.

Open:

- Whether 200 plain steps at the dumped rate bring the cost under a tenth is unmeasured, and so is the time 200 two-pass steps of the three full-size models take.
- The first build decides every bound of the six suites. A bound that fails is reported with the measured difference, not widened.
- `Optim::SGD` decays every slot, and Darknet decays the weights only. The update comparison is expected to show it on the biases and the batch norm scales; the fix is parameter groups in `core/optim`, a public interface and so the owner's decision.
- SushiAI's own calls into SushiRuntime were moved to the v0.4.0 spellings in fifteen files. Each passes `-fsyntax-only` against SushiRuntime `281c2fa` with the deprecation errors of SushiBLAS's headers demoted to warnings; none has been built.

