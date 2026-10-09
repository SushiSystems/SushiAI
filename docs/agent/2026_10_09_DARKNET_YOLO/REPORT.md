# Darknet-era YOLO: reference run and source findings

**Status:** Forward and `[yolo]` training references exist for `yolov3-tiny`, `yolov7-tiny` and `yolov7`. `yolov3`, `yolov4` and `yolov4-tiny` were not run. Batch-statistics forward and weight gradients were not dumped.

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

- `yolov3`, `yolov4`, `yolov4-tiny` were not run. The patch and `run_dump.sh` take them unchanged; `yolov4` needs a 608 by 608 input from the recipe.
- No body layer was dumped (SPEC 5.1 asks for layers 11 and 12 of `yolov3-tiny`), and no decoded `boxes` and `scores` before the threshold.
- No batch-statistics forward, weight gradient or update step (SPEC 5.3).
- `mish` was read but not exercised: only `yolov4` uses it.
- `dx_box_iou` was not read in full.
- The `(confirm)` marks of SPEC about `region`, `detection`, `reorg`, `local`, `connected`, `avgpool` and pjreddie's defaults were left: v1 and v2 are out of scope and pjreddie's source is not on disk.
- The dumps are raw files, not SACP, and no manifest in the fixture format was written.
