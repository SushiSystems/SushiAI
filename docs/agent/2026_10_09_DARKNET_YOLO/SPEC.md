# Darknet-era YOLO: specification

**Status:** Open — stage 6 of `docs/design/PROGRAMME/README.md`; RW-45 to RW-50.

## 1. Requirements

Ruled by the owner on 2026-10-09:

| # | Requirement |
| --- | --- |
| R1 | `yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7` run on the library from Darknet's own `.cfg` and `.weights` files. YOLOv3-tiny's forward parity is the first milestone |
| R2 | No excerpt from AlexeyAB's Darknet enters the repository: no patch, code, comment or line-by-line port. The implementation is written from the behaviour `REPORT.md` states in formulas |
| R3 | A reference is produced by running Darknet and is never typed. The reference tensors, the weights and the cfg files stay outside the repository (section 6). A test that needs one reads `SA_REFERENCE_DATA_DIR` and is reported skipped without it, never passed |
| R4 | The anchor code lives inside `modules/vision/detect`, one module and one target in three parts: the root `Vision::Detect` (what both families use), `free/` `Vision::Detect::Free` (the anchor-free path of YOLO26) and `anchor/` `Vision::Detect::Anchor`. There is no module `vision/detect_anchor`. `vision/detect` depends on `core` only |
| R5 | One anchor unit with options structures serves the six models; no class per version |
| R6 | The detection objective, the host stage and the training step live in a new module `vision/training`, which depends on `train` and `vision/detect`. YOLO26's objective lands beside the anchor one |
| R7 | The importer is the `vision/importers` module `docs/design/DOMAIN_MODULES/README.md` foresees; it emits a `Config::ModelSpec` |
| R8 | Anchoring is as automatic as the data allows (section 5): derived anchors, a chosen count, a computed head assignment, a fit report with a proposal, fractional storage, derived thresholds. An imported model keeps the anchors of its cfg, since parity is measured with them |
| R9 | Every model trains. Training is checked by assignment parity, cost parity, gradient parity at the heads and overfitting one batch. No full COCO run |
| R10 | A domain adds layers and algorithms; an operation, a gradient rule or a lowering is written in the core tier (`DOMAIN_MODULES.md`, "Open") |

Out of scope: YOLOv1 (`local`, `connected`, `detection`, `dropout`), YOLOv2 (`reorg`, `region`), Darknet's data loader and augmentation, a COCO training run.

**Marks.** A statement that cites "REPORT n" was read from Darknet's source or measured on its run by the agent that wrote `REPORT.md`. **(second pass)** marks what the REPORT did not read or run; a second Darknet pass is running and section 6.2 lists what it adds. Statements about this repository were read from the cited file on 2026-10-09.

## 2. What exists

Paths are relative to `modules/` unless they start with another root.

### 2.1 Layers (`layers/nn`)

| Need | What the repository has | Verdict |
| --- | --- | --- |
| Convolution | `NN::Conv2d`, weight `[out, kh, kw, in]`, optional bias; one pad per axis applied to both sides. Row `conv2d` with `filters`, `kernel`, `stride`, `padding`, `dilation`, `use_bias` | Covers every convolution of the six cfgs: `pad=1`, size 1 or 3, stride 1 or 2, no `groups` key |
| Batch normalisation | `NN::BatchNorm2d`: `scale = gamma / sqrt(variance + epsilon)`, `shift = beta - mean * scale`, epsilon `1e-5`, row key `bn_epsilon`. Training through `K_BATCHNORM2D` and the running-statistics update | The same function as Darknet's (REPORT 3.3), with another rounding than its folded form (4.3) |
| `leaky` | `NN::LeakyReLU`, row `leaky_relu` with `slope` | `slope` 0.1 |
| `swish` | `NN::SiLU`, row `silu` | Covered: `x * logistic(x)` with no cut-off (REPORT 3.5) |
| `logistic` | Row `sigmoid`, `K_SIGMOID` with a backward rule | Covered |
| `mish` | `softplus`, `tanh`, `mul` in the builder; no layer, no row | Missing: a `Mish` layer and its row |
| Max pool | `NN::MaxPool2d` holds a symmetric `ConvGeometry`. Row keys `kernel`, `stride`, `padding` | Missing: trailing padding (2.2) |
| Upsample | `NN::Upsample2d`, nearest, row `upsample` | Covered: every `[upsample]` is `stride=2`, and `scale` is 1 |
| `route` | `NN::Concat`, row `concat`; a row's `from` takes absolute and relative indices | Covered up to the six inputs of `yolov7.cfg`. A one-layer `route` is a `from` entry, not a row |
| Grouped `route` | `GraphBuilder::slice_channels`; no layer, no row | Missing: a `ChannelSlice` layer and its row |
| `shortcut` | `GraphBuilder::add`; no node wraps it | Missing: a two-input `Add` node and its row |

A Darknet `[convolutional]` becomes two or three `nn` rows, so `vision/blocks` is not a dependency.

### 2.2 Pools with trailing padding

REPORT 3.4 gives one rule for every `[maxpool]`: `padding` defaults to `size - 1` and is the total over both sides; the output extent is `(in + padding - size) / stride + 1`; the first window starts at `-(padding / 2)`, integer division truncating toward zero; a tap outside the input never wins. The leading pad is therefore `padding / 2` and the trailing pad is what the extent implies:

| Pool | Leading | Trailing | Extent |
| --- | --- | --- | --- |
| 2/1 (`yolov3-tiny` layer 11) | 0 | 1 | `in` |
| 2/2 on an even extent | 0 | 0 | `in / 2` |
| 2/2 on an odd extent | 0 | 1 | `(in - 1) / 2 + 1` |
| 5/1, 9/1, 13/1 | `(size - 1) / 2` | `(size - 1) / 2` | `in` |

`GraphBuilder::maxpool2d_with_extents` takes the output extents from the caller, and SushiBLAS's reference pool skips a tap outside the input (`D:\Projects\sushiblas\modules\kernels\spatial_kernels\include\SushiBLAS\spatial_kernels\reference\pool_max.hpp`). `core/autograd/tests/test_spatial_shapes.cpp` pins the 2/1 case. No operation and no kernel is missing; `MaxPool2d` lacks the field and the model file lacks the key. The translation computes the trailing pad from the rule, so an odd extent is served and not refused.

### 2.3 Model graph, state, evaluation

`NN::ModelGraph::add(name, node, from)` reads any earlier port and `output(name, from)` publishes a head. `NN::build_model(registry, Config::ModelSpec)` accepts a spec built by hand. `ModelState::tensor(name)` returns the tensor `IO::import_weights` writes through `data()`. `Train::EvaluationStep` runs a forward-only pass. Whether `data()` is host-addressable on every device was not checked beyond that one call site.

### 2.4 `vision/detect` after the split of 2026-10-09

| Part | Namespace | Headers |
| --- | --- | --- |
| Root, `include/SushiAI/vision/detect/` | `Vision::Detect` | `box_format`, `iou`, `detections_nms`, `detections_topk`, `top_k`, `detection_row`, `ground_truth`, `average_precision` |
| `free/` | `Vision::Detect::Free` | `anchors`, `anchor_leaves`, `box_coding`, `assigner`, `assigner_config`, `assignment`, `target_leaves`, `level_rows`, `prediction_shape` |
| `anchor/` | `Vision::Detect::Anchor` | None yet; everything of sections 4.4 to 5 |

The module's `CMakeLists.txt` has `PRIVATE_DEPENDS core` and `EXTERNAL_PUBLIC SushiRuntime::SushiRuntime`. Root bricks used here: `intersection_over_union`, `complete_iou`, `K_IOU_EPS = 1e-7` (`iou.hpp`); `suppress_overlaps` and `select_nms_detections(boxes [B, A, 4] centre and size, scores [B, A, C], shape, NmsConfig, detections)` (`detections_nms.hpp`); `GroundTruthBatch` with boxes as pixel corners (`ground_truth.hpp`). `NmsConfig` has no criterion field. Class awareness shifts each box by `class * class_offset` (7680), so boxes reach it in pixels.

Read on 2026-10-09: `detections_nms.hpp:37` and `detections_topk.hpp:37` still include `<SushiAI/vision/detect/prediction_shape.hpp>`, which the move put under `free/`. The root then reads a header of `free/`; where `PredictionShape` belongs is the orchestrator's to settle with the move, before wave 1.

### 2.5 What training finds in the tree

| Need | In the tree |
| --- | --- |
| Constants, barriers | `constant`, `constant_like`, `stop_gradient`, `select` in `GraphBuilder` |
| A scalar from a tensor | `mul`, `reshape`, `sum_rows`. Whether the training graph takes a one-element value that is not `NLL_LOSS` as its root was not checked |
| An objective with several outputs and declared leaves | `Train::Objective`: `declare`, `loss(builder, graph, outputs, leaves)`, `score`, `name` (`layers/train/include/SushiAI/train/objective.hpp:118-154`) |
| A host stage between a forward-only run and the training replay | None. `vision/training` composes `Train::EvaluationStep` and `Train::TrainingStep` |
| A learning-rate schedule | None; `Optim::SGD` exposes its rate |
| A status and an expected type | `StatusCode` (`InvalidDescription`, `FileNotFound`, `InvalidFile`, `Incompatible`, `Unsupported`, ...) and `Result<T>` in `core/core` |

## 3. What the six cfgs contain

Measured from the files in `D:\Projects\.sushisystems\models\darknet\` on 2026-10-09. Layer indices are Darknet's.

| | v3-tiny | v3 | v4-tiny | v4 | v7-tiny | v7 |
| --- | --- | --- | --- | --- | --- | --- |
| Layers | 24 | 107 | 38 | 162 | 99 | 143 |
| Input | 416 | 416 | 416 | 608 | 416 | 640 |
| `convolutional` | 13 | 75 | 21 | 110 | 58 | 92 |
| `maxpool` | 6 | 0 | 3 | 3 | 6 | 8 |
| `route` | 2 | 4 | 11 | 21 | 30 | 38 |
| `shortcut` | 0 | 23 | 0 | 23 | 0 | 0 |
| `upsample`, `yolo` | 1, 2 | 2, 3 | 1, 2 | 2, 3 | 2, 3 | 2, 3 |
| Activations of `convolutional` | leaky 11, linear 2 | leaky 72, linear 3 | leaky 19, linear 2 | mish 72, leaky 35, linear 3 | leaky 55, logistic 3 | swish 89, logistic 3 |
| Pools, size/stride | 2/2 x5, 2/1 x1 | none | 2/2 x3 | 5/1, 9/1, 13/1 | 2/2 x3; 5/1, 9/1, 13/1 | 2/2 x5; 5/1, 9/1, 13/1 |
| Inputs of a `route` | 1, 2 | 1, 2 | 1, 2 | 1, 2, 4 | 1, 2, 4 | 1, 2, 3, 4, 6 |
| Grouped `route` | no | no | layers 3, 11, 19: `groups=2`, `group_id=1` | no | no | no |

REPORT 3.1 counts 26 `linear` in v3 and v4: the 23 `[shortcut]` sections carry `activation=linear` too. `swish` occurs in `yolov7` only. The convolutions without batch normalisation are the ones before each `[yolo]`: 255 filters, `linear` in v3 and v4, `logistic` in both v7 models. No other section type and no `groups` key on a convolution occurs.

`[yolo]` keys:

| Key | v3-tiny, v3 | v4-tiny | v4 | v7-tiny, v7 |
| --- | --- | --- | --- | --- |
| `num`, masks | 6; 9 | 6; masks `3,4,5` and `1,2,3` | 9 | 9 |
| `scale_x_y` | absent | 1.05 | 1.2, 1.1, 1.05 | 2.0 |
| `new_coords` | absent | absent | absent | 1 |
| `nms_kind`, `beta_nms` | absent | `greedynms`, 0.6 | `greedynms`, 0.6 | `diounms`, 0.6 |
| `iou_loss` | absent (`mse`) | `ciou` | `ciou` | `ciou` |
| `iou_normalizer`, `cls_normalizer`, `obj_normalizer` | absent (0.75, 1, 1) | 0.07, 1.0, absent | 0.07, 1.0, absent | 0.05, 0.5, 1.0 |
| `iou_thresh` | absent (1) | absent | 0.213 | 0.2 |
| `objectness_smooth`, `max_delta` | absent | absent | absent, 5 | 1, 2 |
| `ignore_thresh`, `truth_thresh` | 0.7, 1 | 0.7, 1 | 0.7, 1 | 0.7, 1 |

Anchors: v3-tiny and v4-tiny share six, v3 and v7-tiny share nine, v4 and v7 share nine. The first mask of `yolov4-tiny` leaves anchor 0 to no head and gives anchor 3 to both; 5.3 measures what that costs.

YOLOv7 uses no `shortcut`, no `mish` and no grouped `route`, so nothing under `layers/nn` is written for it. Over v4 it adds the `logistic` head with the `new_coords` coding, `swish`, the `diounms` kind and, in training, `objectness_smooth` with deltas taken at an activated value.

## 4. Formats and behaviour

### 4.1 `.cfg` grammar

A text file read line by line. Every space, tab, carriage return and line feed is removed from a line, inner ones included, before it is read (REPORT 3.1); `yolov4.cfg` writes `12, 16, 19, 36`. An empty line and a line starting with `#` or `;` are skipped. A line starting with `[` opens a section. Any other line is `key=value`, split at the first `=`. A list value is comma-separated. The first section is `[net]` or `[network]`; every later one is a layer, numbered from 0. In a `route` or `shortcut` a negative value `v` in layer `i` names layer `i + v`; zero or more is absolute.

`[net]` keys read: `width`, `height`, `channels`; for training `batch`, `subdivisions`, `momentum`, `decay`, `learning_rate`, `burn_in`, `max_batches`, `policy`, `steps`, `scales`. Accepted and unused: `angle`, `saturation`, `exposure`, `hue`, `mosaic`. `letter_box=1` is refused: parity is taken on a tensor of the network's size.

| Section | Keys and Darknet defaults | Rule |
| --- | --- | --- |
| `convolutional` | `filters=1`, `size=1`, `stride=1`, `pad=0`, `padding=0`, `batch_normalize=0`, `activation=logistic` | With `pad` non-zero the padding is `size / 2` and `padding` is ignored. Extent `(in + 2 * padding - size) / stride + 1`. Order: convolution without bias; batch norm or bias; activation (REPORT 3.2). `groups` is refused |
| `maxpool` | `stride=1`, `size=stride`, `padding=size-1` | 2.2 |
| `upsample` | `stride=2`, `scale=1` | Output `(j, i)` reads input `(j / stride, i / stride)`. A `scale` other than 1 is refused |
| `route` | `layers`, `groups=1`, `group_id=0` | Joins the listed layers along channels in list order; with `groups = g`, `group_id = i` each input contributes channels `[i * C / g, (i + 1) * C / g)` |
| `shortcut` | `from`, `activation=linear` | Adds layer `from` to the layer before. Different shapes, `weights_type` and a non-linear activation are refused; whether a cfg reaches Darknet's general routine was not checked (REPORT 3.5), and a refusal would show it |
| `yolo` | The keys of section 3 | Reads the layer before, which has `len(mask) * (5 + classes)` channels. `label_smooth_eps`, `iou_thresh_kind` and any key outside section 3 are refused |

Activations (REPORT 3.5): `leaky(x) = x` for `x > 0`, else `0.1 * x`; `logistic(x) = 1 / (1 + exp(-x))`; `swish(x) = x * logistic(x)`; `mish(x) = x * tanh(softplus(x))`, where Darknet's `softplus` is `x` above 20 and `exp(x)` below -20. The library's `softplus` has no cut-off; the two differ by less than `exp(-20)`, about 2.1e-9, and the `Mish` test states the measured difference. `mish` was read and not run **(second pass, with `yolov4`)**.

A section type or a key outside this table is refused with a status naming it and its line. Darknet only warns; refusing is this importer's choice, so that a cfg it does not fully understand never runs.

### 4.2 `.weights` binary

Little-endian, no padding (REPORT 3.9). Header: `major`, `minor`, `revision` as three `int32`; then `seen`, a `uint64` when `major * 10 + minor >= 2`, else a `uint32`. The two v3 files are 0.2.0 and the other four 0.2.5, so all six have a 20-byte header. A file with `major` or `minor` above 1000 is refused as `Unsupported`: Darknet reads it with another layout.

Per `convolutional` in layer order, `float32`: `biases[n]`; with `batch_normalize`, `scales[n]`, `rolling_mean[n]`, `rolling_variance[n]`; then `weights[n * c * size * size]` as `[out, in, kh, kw]`. No other section holds weights. The reader fails with `InvalidFile` on a file that ends early or has bytes left over, naming the layer and both byte counts. Darknet loaded `yolov3-tiny`, `yolov7-tiny` and `yolov7`; the byte count of the other three is exercised by the reader's own strict check **(second pass for Darknet's side)**.

| File | Bytes | Floats after the header | SHA-256 |
| --- | --- | --- | --- |
| `yolov3-tiny.weights` | 35 434 956 | 8 858 734 | `dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c` |
| `yolov3.weights` | 248 007 048 | 62 001 757 | `523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297` |
| `yolov4-tiny.weights` | 24 251 276 | 6 062 814 | `cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87` |
| `yolov4.weights` | 257 717 640 | 64 429 405 | `e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562` |
| `yolov7-tiny.weights` | 24 967 560 | 6 241 885 | `06be302b1564f2e2c3e1db1f5ec02477ac05d0ae82405a8c1b2b8e4111101b66` |
| `yolov7.weights` | 147 898 248 | 36 974 557 | `4ecf7ca13ec5039ec7b79b0f25b156fda5eaf819d6c2bb6828ba55fe4f928332` |

`weights` `[out, in, kh, kw]` becomes `Conv2d::weight` `[out, kh, kw, in]` by the permutation `(0, 2, 3, 1)`. `biases` goes to `BatchNorm2d::bias` when normalised and to `Conv2d::bias` otherwise; `scales`, `rolling_mean`, `rolling_variance` go to `BatchNorm2d::weight`, `running_mean`, `running_var`. The state names `build_model` gives were not read from the code; the translation's test pins them against `ModelState::entries()`.

### 4.3 Batch normalisation and how parity treats it

Darknet evaluates `y = scale * (x - mean) / sqrt(variance + 0.00001) + bias`, epsilon inside the root. Before an inference forward it folds this into the convolution, in double: `bias' = bias - scale * mean / sqrt(variance + 0.00001)`, every weight of filter `f` times `scale_f / sqrt(variance_f + 0.00001)`, then a rounding to `float32` (REPORT 3.3). Every reference tensor comes from the folded arithmetic.

**Decision: forward parity is measured on a folded model.** The translation takes a choice of two forms. `Separate` emits `conv2d` then `batchnorm2d` with `bn_epsilon` `1e-5`; it is the default and the form training uses, since batch statistics need the layer. `Folded` emits `conv2d` with `use_bias` and no `batchnorm2d` row, and the importer folds each convolution by the formula above, in double, before it stores the weights. The parity tests use `Folded`. Reasons:

- The references were made with this arithmetic. With the same fold, what separates the two programs is the order in which a convolution sums its products (REPORT 1: Darknet's AVX order), so the tolerance can stay tight enough to catch a wrong pad or a wrong permutation in a late layer.
- An unfolded `BatchNorm2d` adds a rounding at each normalised convolution: 11 in `yolov3-tiny`, 72 in `yolov3`, 107 in `yolov4`. The tolerance would have to grow with depth and be chosen per model, and a tolerance chosen after seeing the difference proves little.
- The fold is a pure function over five vectors with a unit test of its own.

The first parity test also runs `yolov3-tiny` in the `Separate` form on the same input and states the largest difference between the two forms at each head. That pins the two as one function and measures what the library's own `BatchNorm2d` rounding costs. In training mode Darknet keeps `rolling = 0.9 * rolling + 0.1 * batch`; the batch-statistics forward was not dumped **(second pass, perhaps)**.

### 4.4 `[yolo]` forward and decode

Head input, NHWC: `t[n, row, col, a * (5 + C) + e]`, with `a` the anchor slot within the head's mask and `e` in `x, y, w, h, objectness, class_0 ...`. Darknet stores it NCHW with that channel index. `(pw, ph)` is the slot's anchor as a fraction of the network input (5.5), `w, h` the head's grid, `s = scale_x_y`.

The layer's output (REPORT 3.6), which the `layer_<k>_yolo` dump holds:

```
new_coords = 0:  X = logistic(tx) * s - (s - 1) / 2    Y likewise    O = logistic(to)    P_c = logistic(tc)    Wr = tw    Hr = th
new_coords = 1:  X = tx * s - (s - 1) / 2              Y likewise    O = to              P_c = tc              Wr = tw    Hr = th
```

With `new_coords = 1` the layer applies no logistic: the convolution before it has `activation=logistic`, so all `5 + C` entries arrive in (0, 1). The decode (REPORT 3.7):

```
bx = (col + X) / w                 by = (row + Y) / h
new_coords = 0:  bw = exp(Wr) * pw          bh = exp(Hr) * ph
new_coords = 1:  bw = Wr^2 * 4 * pw         bh = Hr^2 * 4 * ph
score_c = O * P_c
```

Boxes are centre and size as fractions of the network input. Rows come cell by cell in row-major order and, within a cell, by anchor slot; heads in file order. Darknet keeps a cell and anchor when `O > thresh` and then zeroes each `score_c` that does not exceed `thresh`. Since `P_c <= 1`, `score_c > thresh` implies `O > thresh`, so the one score gate of `select_nms_detections` gives the same rows and the decode takes no threshold. The caller multiplies the boxes by the network size before suppression, because of the class offset of 2.4.

Checked by the REPORT on the dumps: for `yolov3-tiny` the layer output equals the logistic of its input to 9.4e-8; for `yolov7-tiny` it equals `2 * input - 0.5` on `x`, `y` exactly. The decoded boxes and scores before the threshold were not dumped **(second pass)**.

### 4.5 Thresholds and suppression

`detector test` uses `thresh = 0.25` and a suppression threshold of 0.45 (REPORT 3.8). The kind is the `nms_kind` of the last `[yolo]` in the file. All kinds share one frame: per class, sort by that class's score descending, walk the list, skip a row whose score is 0, zero the score of every later row whose criterion against it exceeds the threshold. There is no cap on rows and boxes of different classes never interact.

| `nms_kind` | Models | Criterion |
| --- | --- | --- |
| absent | v3, v3-tiny | `IoU` |
| `greedynms` | v4, v4-tiny | `IoU - (d / c)^0.6`. The exponent is the constant 0.6; `beta_nms` is not read for this kind |
| `diounms` | v7, v7-tiny | `IoU - (d / c)^beta_nms` |

`d` is the squared distance of the centres and `c` the squared diagonal of the smallest enclosing box; with `c = 0` the criterion is `IoU`. Darknet's `IoU` has no epsilon and is 0 when the intersection or the union is 0; `K_IOU_EPS` here is `1e-7`, so a pair exactly on the threshold could be decided differently. The parity test reports such a pair instead of loosening a tolerance. The library needs one criterion with an exponent field: the importer sets 0.6 for `greedynms` whatever `beta_nms` says and `beta_nms` for `diounms`, and lifts `max_detections` and `max_candidates` above the row count.

`detector test` stretches an image to the network size unless `-letter_box` is given. The references start from a tensor of the network's size, so no resize is part of parity.

### 4.6 Training: the delta of a `[yolo]` layer

Darknet writes a delta per head entry and the body descends along `-delta` (REPORT 4). The delta is at the layer's input: for `new_coords = 0` that is the raw head, for `new_coords = 1` the output of the `logistic` convolution, through whose activation the gradient then passes. `X, Y, Wr, Hr, O, P_c` are the layer's output of 4.4. Truths are `x y w h class` as fractions of the image.

Step 1, every cell and anchor slot:

- Decode its box. `best_iou` is its largest IoU with any truth. `best_match_iou` is the largest IoU among the truths, counted only when some class probability of the prediction exceeds 0.25; the class test does not look at the truth's class.
- `dO = obj_normalizer * (0 - O)`.
- If `best_match_iou > ignore_thresh`: `dO = 0`, or with `objectness_smooth` the larger of `dO` and `obj_normalizer * (best_match_iou - O)`. A prediction with no class above 0.25 is never ignored.
- If `best_iou > truth_thresh` the cell is a positive for that truth. `truth_thresh` is 1 in the six cfgs, so this never fires and the library refuses another value.

Step 2, every truth:

- Cell `col = floor(x * w)`, `row = floor(y * h)`.
- Best anchor: among all `num` anchors, the one whose shape has the highest IoU with `(tw, th)`, both centred on the origin; the first wins a tie. If this head's mask holds it, its slot at the cell is assigned.
- With `iou_thresh < 1`, every other anchor of this head's mask whose shape IoU with the truth exceeds `iou_thresh` is assigned at the same cell too.
- Box delta, `iou_loss=mse`, with `k = (2 - tw * th) * iou_normalizer`: `dx = k * (tx - X)`, `dy = k * (ty - Y)`, `dw = k * (log(tw / pw) - Wr)`, `dh = k * (log(th / ph) - Hr)`, where `tx = x * w - col`, `ty = y * h - row`. They add to what the entry holds. `iou_normalizer` defaults to 0.75, which the v3 cfgs do not override; on the dump the `x` delta at truth 0's cell equals `1.76 * 0.75 * (0.5 - X)` to the last bit.
- Box delta, `iou_loss=ciou`: the gradient of CIoU with respect to the predicted box, times `iou_normalizer`; for `new_coords = 0` the `w`, `h` components are also multiplied by `exp(Wr)`, `exp(Hr)`. NaN and infinity become 0, and each component is clipped to `[-max_delta, max_delta]` when `max_delta` is set. `CIoU = IoU - d / c - alpha * v`, `v = (4 / pi^2) * (atan(tw / th) - atan(pw / ph))^2`, `alpha = v / (1 - IoU + v + 0.000001)`. The gradient routine was not read in full **(second pass)**; the v7 delta dumps check it.
- `dO = obj_normalizer * (1 - O)`; with `objectness_smooth` it is written only where the delta is still 0.
- Class deltas: `1 - P_c` for the truth's class, `0 - P_c` for the others. If the truth's class delta at the slot is already non-zero, only that one class is updated. Where `cls_normalizer` multiplies is not stated by the REPORT **(second pass; the v7 dumps, with `cls_normalizer` 0.5, check it)**.

Step 3, only with `iou_thresh < 1`: where a slot's objectness delta is non-zero, its four box deltas are divided by the number of classes with a positive delta at that slot.

Darknet's truth list ends at the first row whose `x` is 0; the library reads `GroundTruthBatch::valid` and treats no coordinate specially.

**Decision: the library computes this delta on the host and trains on it.** The host stage of 4.8 computes the delta of every head, and the training graph's root is `-(1 / batch) * sum(delta * head)` with `delta` a constant leaf, so the gradient at each head's published output is `-delta / batch` by construction. A closed-form loss exists for `mse` (a cross-entropy on logits for objectness and classes, that times `k` for `x`, `y`, half a squared error times `k` for `w`, `h`), but the clip, the division of step 3, the "only where still 0" rule and the larger-of rule of `objectness_smooth` are not the gradient of any one expression, and the six models would need three emitters where this needs one. What the root's value is has no meaning; the reported number is the cost of 4.7. The IR CIoU emitter YOLO26 will need is not written by this work; the host CIoU value and gradient in the root of `vision/detect` are the oracle it can be tested against.

### 4.7 What Darknet reports

Per layer, each divided by the batch size (REPORT 4): `total_loss` is the sum of squared deltas, `class_loss` is `obj_normalizer` times the sum of squared non-box deltas, `iou_loss` their difference. The layer's stored `cost`, in the dump's JSON, is the sum of squared deltas for `mse`, and for the IoU kinds `iou_normalizer * mean(1 - IoU) + class_loss`. Which IoU measure that mean uses was not stated **(second pass; the stored costs of the v7 dumps check it)**. The cost is a report: no gradient is taken of it. `count` in the log line counts every assignment, the extra `iou_thresh` ones included, which the `assigned_row` dump does not show.

### 4.8 The training step

A forward-only pass under batch statistics gives each head; the host stage computes the deltas; the replay descends. Darknet's update (REPORT 5): the weight update accumulates `-decay * batch * weight`, the weights move by `learning_rate / batch` times the accumulated update, the update is then scaled by `momentum`; biases and scales the same without decay. Whether `Optim::SGD` is that rule was not compared; it matters only to the update parity of the second pass. The schedule is `policy=steps` with `steps` and `scales`, and a warm-up of `learning_rate * (iteration / burn_in) ^ power` until `burn_in`; the power is recalled as 4 and is not in the REPORT **(second pass)**, so it is a field.

## 5. Automatic anchoring

The owner's emphasis (R8). Every function here is a free function on value types in `Vision::Detect::Anchor`, holds no global state and returns a status where an input can be wrong. A label box is a width and a height as fractions of the network input, after whatever resize the caller's pipeline applies. `shape_iou(a, b)` is the IoU of two such sizes centred on the origin.

### 5.1 Requirements

| # | Requirement | Test pins |
| --- | --- | --- |
| AR1 | `cluster_anchors` derives `count` anchors from label boxes: k-means++ seeding under the distance `1 - shape_iou`, Lloyd iterations, then the refinement of 5.2. Same boxes in the same order, same options and same seed give the same anchors | Recovery of known centres; the fit never falls in the refinement; two runs are bit-identical on one build |
| AR2 | `choose_anchor_count` picks the count per head when the caller gives none: from `min_per_head` (1) it grows while the gain in mean best IoU from one more anchor per head is at least `min_gain` (0.02), up to `max_per_head` (5) | Six separated clusters and two heads give 3; one blob gives 1; the trials are returned |
| AR3 | `assign_anchors_to_heads` computes each head's anchor indices from the strides and the anchors' areas. Nothing types a mask | Reproduces the masks of five cfgs; differs from `yolov4-tiny`'s and says by how much |
| AR4 | `measure_anchor_fit` reports, for a dataset under an anchor layout: mean best IoU, best possible recall, the share of boxes whose best anchor no head owns, and each anchor's share. `review_anchors` compares a layout with a proposal for the same counts and returns both with a verdict; it changes nothing | A fitted set is kept; a dataset whose boxes fall between the anchors gets `Propose` with both fits; the input layout is unchanged |
| AR5 | An anchor is stored as a fraction of the network input. Pixels appear in `vision/importers` alone, where a cfg's anchors are divided by `width` and `height` | One layout decodes the same fractional box on a 13 and a 19 grid |
| AR6 | `derive_assignment_thresholds` gives defaults for the match and ignore thresholds from the fit; a caller overrides either | The stated formula on a known fit; the clamps |

An imported Darknet model keeps the anchors, masks and thresholds of its cfg. `review_anchors` can still be run on it against a new dataset, and it never replaces anything.

### 5.2 Clustering (AR1)

1. Boxes with a non-finite side or a side at or below `min_side` are left out and counted. Fewer distinct boxes than `count` is `Incompatible`.
2. Seeding: the first centre is the box at index `next() % n`; each later centre is drawn with probability proportional to the squared distance to its nearest centre, by a cumulative sum in index order.
3. Lloyd: each box goes to the centre of smallest distance, the lower index on a tie. A centre becomes the component-wise median of its boxes' widths and heights; the median resists the few very large boxes a mean follows. An empty cluster takes the box farthest from its centre. Stop when no box changes cluster or at `max_iterations` (300).
4. Refinement, a pattern search on the fit measure `F`, the mean over boxes of the best `shape_iou` with any anchor: for each anchor in index order and for width then height, try the factors `1 + step` and `1 - step` and keep a change that raises `F`. `step` starts at 0.1 and halves after a sweep without a gain; stop below 0.001 or at `max_refinement_sweeps` (64). `F` cannot fall.
5. The anchors are returned in ascending area.

Determinism rule: the random source is a SplitMix64 the unit defines itself, seeded from the options (default `0x5355534849`), because the standard distributions differ between libraries. All sums run in index order on one thread, in double. Bit identity is promised for one build; between compilers the tests compare to 1e-6.

### 5.3 Count, heads, fit, proposal (AR2 to AR4)

Count: for `p` from `min_per_head`, cluster `p * heads` anchors; stop at the first `p` whose successor gains less than `min_gain`, or at `max_per_head`. The count fixes the head's channel count, so it is chosen before a model is built and is never changed for an existing one.

Heads: anchors in ascending area and heads in ascending stride; with `K = q * heads + r`, the `r` heads of smallest stride take `q + 1` consecutive anchors and the others `q`. This gives the masks of `yolov3-tiny`, `yolov3`, `yolov4`, `yolov7-tiny` and `yolov7`. For `yolov4-tiny` it gives `0,1,2` and `3,4,5` where the cfg has `1,2,3` and `3,4,5`.

Fit: a box is recallable when its best anchor is owned by a head and their `shape_iou` is at least `recall_iou` (0.25: a box whose sides are both within a factor of two of an anchor's reaches it). Best possible recall is the share of recallable boxes.

Proposal: `review_anchors` clusters the dataset to the layout's own count, assigns heads to the layout's own per-head counts, and returns `Propose` when the current recall is below `min_recall` (0.9) or the proposal's mean best IoU exceeds the current one by more than `min_gain` (0.05); otherwise `Keep`. Because the counts are kept, accepting changes no tensor shape. To accept, the caller assigns `review.proposed` to the layout it passes to the objective and re-initialises the convolution before each head: the box entries of those weights were trained against the old anchors. Nothing does either step for the caller.

### 5.4 Thresholds (AR6)

`match_threshold` (Darknet's `iou_thresh`) is the tenth percentile of the boxes' best `shape_iou`, clamped to `[0.2, 1]`: where the anchors fit loosely more of a head's anchors share a truth, and where they fit tightly the best one is enough. `ignore_threshold` is the mean best IoU clamped to `[0.5, 0.7]`: a prediction that overlaps a truth more than a bare anchor does on average is not pushed toward zero objectness. Both are this work's rules with no Darknet counterpart, and whether they train better than 0.2 and 0.7 is not measured here. Without a dataset the defaults are 1 and 0.7.

### 5.5 Public surface

```cpp
struct AnchorSize { float width = 0.0f; float height = 0.0f; };   // fractions of the network input
[[nodiscard]] float shape_iou(AnchorSize a, AnchorSize b) noexcept;
// anchors: every anchor of the model. head_anchors: per head, indices into anchors (Darknet's mask).
struct AnchorLayout { std::vector<AnchorSize> anchors; std::vector<std::vector<std::int64_t>> head_anchors; };
[[nodiscard]] Status validate_anchor_layout(const AnchorLayout& layout);
struct ClusteringOptions { std::int64_t count = 0; std::uint64_t seed = 0x5355534849ull; std::int64_t max_iterations = 300;
                           std::int64_t max_refinement_sweeps = 64; float min_side = 0.0f; };
struct AnchorClustering { std::vector<AnchorSize> anchors; double mean_best_iou = 0.0; double mean_best_iou_before_refinement = 0.0;
                          std::int64_t iterations = 0; std::int64_t refinement_sweeps = 0;
                          std::int64_t boxes_used = 0; std::int64_t boxes_rejected = 0; };
[[nodiscard]] Result<AnchorClustering> cluster_anchors(SushiRuntime::span<const AnchorSize> boxes, const ClusteringOptions& options);
struct CountOptions { std::int64_t min_per_head = 1; std::int64_t max_per_head = 5; double min_gain = 0.02; ClusteringOptions clustering; };
struct CountTrial { std::int64_t per_head = 0; double mean_best_iou = 0.0; };
struct AnchorCountChoice { std::int64_t per_head = 0; std::vector<CountTrial> trials; AnchorClustering clustering; };
[[nodiscard]] Result<AnchorCountChoice> choose_anchor_count(SushiRuntime::span<const AnchorSize> boxes, std::int64_t heads, const CountOptions& options);
[[nodiscard]] Result<AnchorLayout> assign_anchors_to_heads(SushiRuntime::span<const AnchorSize> anchors,
                                                           SushiRuntime::span<const std::int64_t> strides, SushiRuntime::span<const std::int64_t> per_head);  // empty: the even split
// unowned_share: boxes whose best anchor no head owns. anchor_share: per anchor, the share of boxes it fits best.
struct FitOptions { float recall_iou = 0.25f; };
struct AnchorFit { double mean_best_iou = 0.0; double best_possible_recall = 0.0; double unowned_share = 0.0;
                   double best_iou_tenth_percentile = 0.0; std::vector<double> anchor_share; std::int64_t boxes = 0; };
[[nodiscard]] Result<AnchorFit> measure_anchor_fit(SushiRuntime::span<const AnchorSize> boxes, const AnchorLayout& layout, const FitOptions& options);
enum class AnchorVerdict : std::uint8_t { Keep, Propose };
struct ReviewOptions { double min_gain = 0.05; double min_recall = 0.9; FitOptions fit; ClusteringOptions clustering; };
struct AnchorReview { AnchorFit current; AnchorLayout proposed; AnchorFit proposed_fit; AnchorVerdict verdict = AnchorVerdict::Keep; };
[[nodiscard]] Result<AnchorReview> review_anchors(SushiRuntime::span<const AnchorSize> boxes, const AnchorLayout& current,
                                                  SushiRuntime::span<const std::int64_t> strides, const ReviewOptions& options);
struct AssignmentThresholds { float match_threshold = 1.0f; float ignore_threshold = 0.7f; };
[[nodiscard]] AssignmentThresholds derive_assignment_thresholds(const AnchorFit& fit) noexcept;
struct AnchorPlanOptions { std::int64_t per_head = 0; CountOptions count; FitOptions fit; };   // per_head 0: chosen
struct AnchorPlan { AnchorLayout layout; AnchorFit fit; AssignmentThresholds thresholds; std::vector<CountTrial> trials; };
[[nodiscard]] Result<AnchorPlan> plan_anchors(SushiRuntime::span<const AnchorSize> boxes, SushiRuntime::span<const std::int64_t> strides, const AnchorPlanOptions& options);
```

The structures are written on one line here for space; in source each field has its own line and comment. `plan_anchors` is the one call a new model makes: count, clustering, heads, fit and thresholds, each from the function above. A bad option is `InvalidDescription`; boxes and counts that disagree are `Incompatible`. No status code is added.

### 5.6 Synthetic distributions the tests use

Every distribution is generated in the test from a seed written there.

| Name | Boxes | Used for |
| --- | --- | --- |
| Ladder | `K` centres on a geometric ladder from `0.03 x 0.05`, ratio 1.8, 400 boxes each, each side jittered by up to 2 % | AR1: every centre recovered within 5 %, `F > 0.95`; AR2 with `K = 6`; AR3 |
| Blob | 2 000 boxes, each side log-uniform within 10 % of `0.2 x 0.3` | AR2: one anchor per head |
| Outliers | Ladder with 1 % of the boxes at `0.9 x 0.9` | AR1: the median centre stays within 5 % |
| Shifted | Ladder with every side times 0.75, halfway between two rungs, measured against a layout fitted to Ladder | AR4: `Propose`, the current recall and both fits within 1e-6 of the values the test computes directly |
| Degenerate | No box; `count - 1` distinct boxes; a zero and a NaN side | The statuses and `boxes_rejected` |

The six anchors and two masks of `yolov4-tiny`, written in the test as input data, measured on 100 boxes at each of the six anchor sizes: `unowned_share` is 1/6, the boxes of anchor 0, and the computed masks bring it to 0.

### 5.7 Against Darknet's own tools

| Behaviour | Darknet | Here |
| --- | --- | --- |
| Anchors from label boxes | `detector calc_anchors` clusters widths and heights for a count the user gives and prints an average IoU. This is recalled, not read: neither Darknet pass covers that tool | AR1, with the refinement and the stated determinism |
| Count per head | Typed by the user | AR2 |
| Masks | Typed in the cfg | AR3 |
| Fit before training, recall, a proposal | None beyond that printed average | AR4 |
| Anchor units | Pixels of the network input, so a change of `width` asks for new numbers | AR5 |
| `iou_thresh`, `ignore_thresh` | Typed in the cfg | AR6 |

Proposals of this revision, beyond the six requirements:

- The clustering leaves out degenerate boxes and returns their count, so a broken label file shows in a number (5.2 step 1). Darknet's loader rejects boxes under one pixel (REPORT 5).
- `AnchorFit::anchor_share` shows an anchor no box prefers, which the v4-tiny mask case needs.
- `review_anchors` keeps the counts, so a proposal can be accepted without rebuilding the model.

## 6. References

### 6.1 Where they are and what they hold

Outside the repository, under `D:\Projects\.sushisystems\models\darknet\`, which `SA_REFERENCE_DATA_DIR` names: the six `.cfg` and `.weights` files and `coco.names` directly in it, and `references/<model>[_thresh_0p01]/` with one raw little-endian `float32` `<name>.bin` and one `<name>.json` per tensor (REPORT 2.2). `SA_REFERENCE_DATA_DIR` is an environment variable read at run time, so no path is compiled in. Darknet commit `59596d7880f6504768df41d6daa586f5cb2b932f`.

| Name | Shape | Content |
| --- | --- | --- |
| `input` | `[3, H, W]` | The floats layer 0 received |
| `layer_<k-1>_yolo_input` | `[A * (5 + C), h, w]` | The raw head, the published output of the model |
| `layer_<k>_yolo` | same | The `[yolo]` layer's output of 4.4 |
| `detections` | `[rows, 7]` | `x y w h objectness score class` after threshold and suppression; the JSON holds `thresh`, `nms`, `nms_kind`, `beta_nms`, `candidates_before_nms` |
| `truth` | `[rows, 6]` | The truth rows and the row number |
| `layer_<k>_yolo_delta` | `[A * (5 + C), h, w]` | The delta of 4.6; the JSON holds `cost` |
| `layer_<k>_yolo_assigned_row`, `_assigned_class` | `[A, h, w]` | The best-anchor assignment, -1 elsewhere |

`<k>` is written with three digits. Feature maps are `CHW` and are permuted to `HWC` by the test's loader. The input and the truth are rebuilt without Darknet by the recipes of REPORT 2.1, whose SHA-256 values the loader does not recompute.

The head input of the training-mode dumps comes from the folded inference forward, so they pin the assignment, the delta and the cost for a known head tensor and say nothing about batch statistics, weight gradients or the update (REPORT 2.3).

### 6.2 What exists and what the second pass adds

On disk on 2026-10-09: `yolov3_tiny`, `yolov3_tiny_thresh_0p01`, `yolov7_tiny`, `yolov7_tiny_thresh_0p01`, `yolov7_thresh_0p01`. The synthetic input finds nothing at 0.25, so suppression parity uses the `_thresh_0p01` directories: 16 rows from 58 candidates for `yolov3-tiny`, 11 from 207 for `yolov7-tiny`, 7 from 54 for `yolov7`.

The second pass, running now, is to add: the directories of `yolov3`, `yolov4-tiny` and `yolov4`; every layer of `yolov3-tiny`; decoded boxes and scores before the threshold; the CIoU gradient in full; perhaps one batch-statistics training step. Its tensor names beyond the table above are taken from the revised REPORT, not guessed here.

### 6.3 Tolerances

A tolerance is written in the test before the first build, absolute and relative, and the build reports the measured maximum beside it. A failure is reported; no tolerance is widened to pass. Starting values: a head tensor of a folded model, `1e-3` absolute plus `1e-4` relative; a delta computed from the dumped head, `1e-6` absolute; a detection row, `1e-4` on each box entry and score with equal class.

## 7. Constraints on the work

- The working tree has uncommitted edits under `modules/vision/blocks/source/` and the move of the nine `free/` headers staged. No unit touches `vision/blocks`.
- The orchestrator edits every build file and `modules/vision/umbrella/include/SushiAI/vision/vision.h`; `sushiai_check_umbrella` stops the configure for a header the umbrella leaves out.
- The tier table exists twice, in `cmake/SushiAILayers.cmake` and `tools/layering/check_layering.py`. Both gain `vision/importers` and `vision/training` and no row for the anchor code.
- Builds and test runs are the owner's. A worker checks single files with `clang -fsyntax-only`.
- The files under section 6.1 have no `format_version`: they are a foreign dump read by test code only, and no module of the library owns or reads them.
