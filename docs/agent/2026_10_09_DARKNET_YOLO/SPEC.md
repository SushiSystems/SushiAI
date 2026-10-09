# Darknet-era YOLO: specification

**Status:** Draft, revised 2026-10-09 to the owner's rulings of that day. Nothing here is built or run. Stage 6 of `docs/design/PROGRAMME.md`; backlog section "Darknet-era YOLO" of `docs/design/REMAINING_WORK.md`. The units and waves are in `PLAN.md` beside this file.

## 1. Requirements

Ruled by the owner on 2026-10-09 (`docs/design/PROGRAMME.md:38`, `:52`):

| # | Requirement |
| --- | --- |
| R1 | `yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7` run on the library from Darknet's own `.cfg` and `.weights` files, in that order |
| R2 | One anchor-based detection unit with an options structure serves them; no class per version. It is its own module, `vision/detect_anchor` |
| R3 | YOLOv3-tiny is first: its forward pass on imported weights equals a reference |
| R4 | A reference is produced by running Darknet and is never typed |
| R5 | A domain adds layers and algorithms; an operation, a gradient rule or a lowering is written in the core tier (`docs/design/DOMAIN_MODULES.md:126-128`) |
| R6 | The importer is the `importers` module the domain shape foresees (`DOMAIN_MODULES.md:75`) and emits a `Config::ModelSpec` |
| R7 | Every model trains. The anchor unit holds the host assigner and the loss, as IR built from existing operations. Each model has a forward-parity part first and a training part after it |
| R8 | The weights and cfg files stay outside the repository. A test that needs one reads `SA_REFERENCE_DATA_DIR` and is reported skipped without it, never passed |
| R9 | The suppression criterion is chosen in the configuration. The CIoU emitter and the home of the objective are shared with YOLO26 |

Out of scope:

- YOLOv1 (`local`, `connected`, `detection`, `dropout`).
- YOLOv2 (`reorg`, `region`).
- A full COCO training run and Darknet's augmentation, subject to question Q3 of the plan.

**The reference.** AlexeyAB's Darknet is being built outside the repository, CPU only, with a patch that dumps the network input, each `[yolo]` layer's raw output and the final detections. The build is deleted when every reference exists. The agent doing that writes `REPORT.md` beside this file, with the dump layout and the answer to every statement marked **(REPORT)** below. Such a statement is written from memory of Darknet; the REPORT settles it, and no unit that depends on one is dispatched before the REPORT exists. Every statement about this repository or about a `.cfg` file was read from the cited file on 2026-10-09.

## 2. What exists

Paths are relative to `modules/` unless they start with another root.

### 2.1 Layers (`layers/nn`)

| Need | What the repository has | Verdict |
| --- | --- | --- |
| Convolution | `NN::Conv2d`, `layers/nn/include/SushiAI/nn/conv2d.hpp:42`. Weight `[out, kh, kw, in]`, optional bias `[out]`. Geometry is `Graph::ConvGeometry` with one pad per axis applied to both sides (`core/graph/include/SushiAI/graph/builder.hpp:50-60`). Row `conv2d` with `filters`, `kernel`, `stride`, `padding`, `dilation`, `use_bias` (`layers/nn/source/layer_registry.cpp:69-70`) | Covers every convolution of the six cfgs: each has `pad=1`, size 1 or 3, stride 1 or 2, and none has a `groups` key (3.1). `NN::DepthwiseConv2d` exists and is not needed |
| Batch normalisation at inference | `NN::BatchNorm2d::trace_evaluation` emits `BATCHNORM_FOLD` then `CHANNEL_AFFINE`: `scale = gamma / sqrt(variance + epsilon)`, `shift = beta - mean * scale`. Epsilon defaults to `1e-5`, row key `bn_epsilon` | Covers AlexeyAB's inference form if its epsilon sits inside the root at `1e-5` **(REPORT)** |
| `leaky` | `NN::LeakyReLU`, slope default `0.01`, row `leaky_relu` with key `slope` (`layer_registry.cpp:455`) | `{"type": "leaky_relu", "slope": 0.1}` |
| `swish` (v7) | `NN::SiLU`, `x * sigmoid(x)`, `layers/nn/include/SushiAI/nn/silu.hpp:41`; operation `K_SILU` (`core/core/include/SushiAI/core/ops.hpp:311`); row `silu` (`layer_registry.cpp:463`) | Covered, if Darknet's `swish` is `x * logistic(x)` with no cut-off **(REPORT)** |
| `logistic` (v7 heads) | `GraphBuilder::sigmoid` (`builder.hpp:350`), `K_SIGMOID` with a backward rule (`ops.hpp:176`, `:427`); row `sigmoid` (`layer_registry.cpp:454`) | Covered |
| `mish` (v4) | `softplus`, `tanh`, `mul` (`builder.hpp:306`, `:353`, `:234`); no layer and no row | Missing: a `Mish` layer composed of the three, and its row |
| Max pool | `NN::MaxPool2d` holds only a `ConvGeometry` (`max_pool2d.hpp:41-44`) and calls the symmetric emitter. Row keys `kernel`, `stride`, `padding` (`layer_registry.cpp:73`) | Covers size 2 stride 2 on an even extent and the pools 5, 9, 13 of stride 1 with `padding = size / 2`. Does not cover size 2 stride 1 (2.2) |
| Upsample | `NN::Upsample2d`, nearest, `scale` default 2, row `upsample` (`layer_registry.cpp:462`) | Covered: every `[upsample]` of the six cfgs is `stride=2` |
| `route` over several layers | `NN::Concat` node, any number of `[H, W, C_i]` inputs joined in order, row `concat` (`layer_registry.cpp:465`). A row's `from` takes absolute indices, negative relative ones and `"input"` | Covered, up to the six inputs `yolov7.cfg` joins. A `route` with one layer is not a row: the rows that read it name that layer in `from` |
| Grouped `route` (v4-tiny) | `GraphBuilder::slice_channels` (`builder.hpp:652`), `K_SLICE_CHANNELS` (`ops.hpp:241`); no layer and no row | Missing: a `ChannelSlice` layer and its row |
| `shortcut` (v3, v4) | `GraphBuilder::add` (`builder.hpp:228`), `K_ADD`; no node wraps it. The rows `register_layers` adds are `linear`, `gelu`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `unflatten`, `conv2d`, `maxpool2d`, `avgpool2d`, `global_avgpool`, `flatten`, `upsample`, `silu`, `batchnorm2d`, `concat` (`layer_registry.cpp:450-465`) | Missing: a two-input `Add` node and its row |
| Conv + BN + activation as one block | `Vision::Blocks::ConvNorm`; `vision/blocks` also holds `Bottleneck`, `C3k`, `C3k2`, `SPPF`, `Attention`, `PSABlock`, `C2PSA`, `DetectHead` | Not used. A Darknet `[convolutional]` becomes two or three rows (`conv2d`, `batchnorm2d`, activation), so the importer names `nn` rows only and `vision/blocks` is not a dependency |

### 2.2 The size 2, stride 1 pool

Only `yolov3-tiny.cfg` has it, at layer 11. Darknet keeps the extent by padding one row below and one column right **(REPORT: the offset rule)**.

- `GraphBuilder::maxpool2d_with_extents(x, g, out_h, out_w)` (`builder.hpp:562`) takes the output extents from the caller and checks only that they are positive. `ConvGeometry` has no bottom or right pad, so the trailing padding is whatever the stated extents imply.
- `core/autograd/tests/test_spatial_shapes.cpp:208-224` pins this case: a 2x2 window of stride 1 over `[1, 13, 13, 256]` with extents 13x13.
- SushiBLAS skips a tap outside the input and starts the running best at the lowest value of the type (`D:\Projects\sushiblas\modules\kernels\spatial_kernels\include\SushiBLAS\spatial_kernels\reference\pool_max.hpp:59-70`), so a padded tap never wins.
- What is missing is above the builder: `MaxPool2d` has no field for trailing padding and the model file has no key for it.

No IR operation and no kernel is missing for this pool.

### 2.3 Model graph, state, evaluation

| Need | What the repository has |
| --- | --- |
| A graph with branches | `NN::ModelGraph::add(name, node, from)` reads any earlier ports (`model_graph.hpp:157`), so a `route` to an earlier layer is a `from` entry |
| Two or three heads | `ModelGraph::output(name, from)` (`:180`); `trace` returns `ModelOutputs` (`:196`). In a model file, `outputs` rows carry `node`, `port`, `name` |
| A graph from a description | `NN::build_model(registry, Config::ModelSpec)` (`node_registry.hpp:114`) accepts a spec built by hand and names each row by its index |
| State filled by a foreign reader | `ModelState::create(engine, model)` (`model_state.hpp:118`); `ModelState::tensor(name)` returns the `SushiBLAS::Tensor*` (`:163`), and `IO::import_weights` writes through `tensor->data()` (`layers/io/source/import_weights.cpp:104`, `:130`) |
| Forward-only run | `Train::EvaluationStep::make(engine, traced, state, pool)`, `run(inputs)`, `output(name)` (`layers/train/include/SushiAI/train/evaluation_step.hpp:69-103`) |

Whether `data()` is host-addressable on every device the suite runs on was not checked beyond that one call site.

### 2.4 Detection bricks (`vision/detect`), all on host spans of `float`

| Brick | Signature site | Input |
| --- | --- | --- |
| IoU | `intersection_over_union` (`iou.hpp:45`), `complete_iou` (`:53`), `elementwise_complete_iou` (`:68`), `pairwise_iou` (`:77`); `K_IOU_EPS = 1e-7` | Boxes as x1 y1 x2 y2 |
| Greedy suppression | `suppress_overlaps(boxes, scores, iou_threshold, kept)`, `detections_nms.hpp:84` | `[N, 4]`, `[N]` |
| Class-aware NMS | `select_nms_detections(boxes, scores, shape, config, detections)`, `:96`; `NmsConfig` (`:67`) holds confidence, IoU threshold, label and class modes, `max_detections`, `max_candidates`, `class_offset` 7680 | `[B, A, 4]` centre and size, `[B, A, C]` probabilities; writes `[n, 6]` rows x1 y1 x2 y2 score class |
| Ground truth | `GroundTruthBatch`, `GroundTruthRows`, `ground_truth.hpp` | |

`anchors.hpp`, `anchor_leaves.hpp`, `assigner.hpp` and `target_leaves.hpp` belong to the anchor-free head: anchor points, the task-aligned assigner and its leaves. `vision/detect_anchor` keeps anchor boxes apart from them. Class awareness shifts each box by `class * class_offset`, so boxes must be in pixels below 7680. `NmsConfig` has no criterion field and there is no distance-IoU suppression.

### 2.5 Letterbox (`vision/imaging`)

`plan_letterbox` and `apply_letterbox` (`vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp:94-113`) work on 8-bit images and move labels forward into the letterboxed image. There is no inverse for a box. The resize kernel is not Darknet's, so a parity fixture starts from the tensor Darknet fed its first layer (5.1).

### 2.6 What training finds in the tree

YOLO26's forward pass is committed (`1dbc84b`). Searched on 2026-10-09 for what its training half was to bring:

| Need of the anchor loss | In the tree | Consequence |
| --- | --- | --- |
| Gradient barriers and constants | `K_CONSTANT`, `K_STOP_GRADIENT` (`ops.hpp:153`, `:203`); `constant`, `constant_like`, `stop_gradient`, `select` (`builder.hpp:333`, `:341`, `:719`, `:275`) | Present |
| Binary cross-entropy on logits | `softplus`, `mul`, `sub` (`builder.hpp:306`, `:234`, `:231`): `softplus(x) - x * t` | Composable |
| CIoU as IR | `DIV`, `MINIMUM`, `MAXIMUM`, `ATAN`, `EXP` exist (`ops.hpp:308-309`). CIoU exists on the host only (`iou.hpp:53`); a search of `modules/` for an emitter found none | The emitter is written once here, in `vision/detect`, for YOLO26 to call too (R9) |
| A scalar loss | `nll_loss` is the only emitter documented as a loss (`builder.hpp:404`). `sum_rows` reduces `[N, C]` to `[C]` (`:225`), so `reshape` to `[M, 1]` then `sum_rows` gives one element | Composable on paper. Whether the training graph takes a one-element value that is not `NLL_LOSS` as its root was not checked |
| An objective with several outputs and declared leaves | `Train::Objective` has `declare`, `loss(builder, graph, outputs, leaves)`, `score`, `name` (`layers/train/include/SushiAI/train/objective.hpp:121-154`) | Present |
| A host stage between a forward-only run and the training replay | No hook in `Objective`; `Trainer::step` is "feed, replay, read metrics, update" (`trainer.hpp:180`). No detection objective exists in `modules/` | Designed here; question Q4 of the plan asks for its home |
| A schedule with `burn_in` and `steps` | `Optim::SGD` exposes its rate "for a learning-rate schedule" (`core/optim/include/SushiAI/optim/sgd.hpp:96`); no schedule type exists | Written here; question Q5 |
| Batch normalisation in training | `K_BATCHNORM2D` and the running-statistics update (`PROGRAMME.md:40`) | Library |

## 3. What the six cfgs contain

Measured from the files in `D:\Projects\.sushisystems\models\darknet\` on 2026-10-09. Layer indices are Darknet's.

### 3.1 Sections, activations, pools

| | v3-tiny | v3 | v4-tiny | v4 | v7-tiny | v7 |
| --- | --- | --- | --- | --- | --- | --- |
| Layers | 24 | 107 | 38 | 162 | 99 | 143 |
| Input | 416 | 416 | 416 | 608 | 416 | 640 |
| `convolutional` | 13 | 75 | 21 | 110 | 58 | 92 |
| `maxpool` | 6 | 0 | 3 | 3 | 6 | 8 |
| `route` | 2 | 4 | 11 | 21 | 30 | 38 |
| `shortcut` | 0 | 23 | 0 | 23 | 0 | 0 |
| `upsample` | 1 | 2 | 1 | 2 | 2 | 2 |
| `yolo` | 2 | 3 | 2 | 3 | 3 | 3 |
| Activations | leaky 11, linear 2 | leaky 72, linear 3 | leaky 19, linear 2 | mish 72, leaky 35, linear 3 | leaky 55, logistic 3 | swish 89, logistic 3 |
| Pools, size/stride | 2/2 x5, 2/1 x1 | none | 2/2 x3 | 5/1, 9/1, 13/1 | 2/2 x3; 5/1, 9/1, 13/1 | 2/2 x5; 5/1, 9/1, 13/1 |
| Inputs of a `route` | 1, 2 | 1, 2 | 1, 2 | 1, 2, 4 | 1, 2, 4 | 1, 2, 3, 4, 6 |
| Grouped `route` | no | no | layers 3, 11, 19: `groups=2`, `group_id=1` | no | no | no |

No other section type occurs. A `[convolutional]` carries only `batch_normalize`, `filters`, `size`, `stride`, `pad`, `activation`; `pad` is 1 everywhere and no `groups` key occurs. A `[maxpool]` carries only `size` and `stride`, a `[shortcut]` only `from` and `activation=linear`, an `[upsample]` only `stride=2`. The only convolutions without batch normalisation are the ones before each `[yolo]`: 255 filters, `linear` in v3 and v4, `logistic` in v7 and v7-tiny.

Whether every 2/2 pool meets an even extent was not traced layer by layer; the translation refuses an odd one.

### 3.2 `[yolo]` keys

| Key | v3-tiny, v3 | v4-tiny | v4 | v7-tiny, v7 |
| --- | --- | --- | --- | --- |
| `num`, masks | 6; 9 | 6; masks `3,4,5` and `1,2,3` | 9 | 9 |
| `scale_x_y` | absent | 1.05 | 1.2, 1.1, 1.05 | 2.0 |
| `new_coords` | absent | absent | absent | 1 |
| `nms_kind`, `beta_nms` | absent | `greedynms`, 0.6 | `greedynms`, 0.6 | `diounms`, 0.6 |
| `iou_loss` | absent | `ciou` | `ciou` | `ciou` |
| `iou_normalizer`, `cls_normalizer`, `obj_normalizer` | absent | 0.07, 1.0, absent | 0.07, 1.0, absent | 0.05, 0.5, 1.0 |
| `iou_thresh` | absent | absent | 0.213 | 0.2 |
| `objectness_smooth` | absent | absent | absent | 1 |
| `max_delta` | absent | absent | 5 | 2 |
| `ignore_thresh`, `truth_thresh` | 0.7, 1 | 0.7, 1 | 0.7, 1 | 0.7, 1 |
| `jitter`, `random`, `resize` | 0.3, 1, absent | 0.3, 0, 1.5 | 0.3, 1 on the last head, absent | 0.1, absent, 1.5 |

Anchors: v3-tiny and v4-tiny share six, v3 and v7-tiny share nine, v4 and v7 share nine.

### 3.3 What YOLOv7 needs

| What the cfg has | Forward pass | Training | Served by |
| --- | --- | --- | --- |
| `activation=swish` | `x * sigmoid(x)` | The rule of `K_SILU` | `NN::SiLU`, row `silu` (2.1) |
| `activation=logistic` on the head convolution | The head's values reach `[yolo]` already inside (0, 1) | The gradient passes through `K_SIGMOID` | Row `sigmoid` |
| `new_coords=1` | The `[yolo]` layer applies no logistic of its own, and the box size is not `exp(tw)` but a function of the activated value. The exact centre and size formulae are those of Darknet's `get_yolo_box`, which the REPORT gives **(REPORT)** | The box, objectness and class terms are taken on activated values, so they are not the cross-entropy on logits of v3 (4.5) | A second coding in `vision/detect_anchor` |
| `scale_x_y=2.0` | The centre offset spans two cells, half a cell beyond each edge | Enters the decoded box the IoU reads | The `scale_xy` option v4 already needs |
| `nms_kind=diounms`, `beta_nms=0.6` | Suppression by a distance-IoU criterion, if `detector test` honours the key **(REPORT, with what `greedynms` and `beta_nms` select in v4)** | None | A criterion field in `NmsConfig` and one new function in `vision/detect` |
| `iou_loss=ciou`, `iou_normalizer` | None | The box term is `1 - CIoU`, weighted | The shared CIoU emitter (2.6) |
| `objectness_smooth=1` | None | The objectness target of an assigned prediction is taken from its IoU with the truth and not fixed at 1; the exact expression is **(REPORT)** | An objectness-target option of the assigner |
| `iou_thresh=0.2` | None | Every anchor of a head whose shape IoU with a truth exceeds it is assigned, beside the best one | The multi-anchor match v4 already needs |
| `obj_normalizer`, `cls_normalizer`, `max_delta` | None | Term weights, and a clip on the box delta that gradient parity has to reproduce **(REPORT)** | Fields of the loss options |
| Pools 2/2 and 5, 9, 13 of stride 1 | Symmetric padding only; no stride-1 pool of size 2 | Library | `MaxPool2d` as it is |
| Routes of up to six inputs, none grouped | Concatenation | Library | `Concat` |

YOLOv7 uses no `shortcut`, no `mish` and no grouped `route`: nothing under `layers/nn` is written for it. What it adds over v4 is the `logistic` head with the `new_coords` coding, `swish`, the `diounms` kind, and in training the IoU-derived objectness target with terms on activated values.

## 4. Formats

### 4.1 `.cfg` grammar

A text file read line by line. Darknet strips spaces, tabs and line endings from a line before reading it **(REPORT: inner spaces too)**; `yolov4.cfg` writes its anchors as `12, 16, 19, 36`. An empty line and a line starting with `#` or `;` are skipped. A line starting with `[` opens a section. Any other line is `key=value`, split at the first `=`. A list value is comma-separated. The first section is `[net]` or `[network]`; every later one is a layer, numbered from 0 in file order.

`[net]` keys the forward pass reads: `width`, `height`, `channels`. Training settings: `batch`, `subdivisions`, `momentum`, `decay`, `learning_rate`, `burn_in`, `max_batches`, `policy`, `steps`, `scales`. Augmentation keys, accepted and unused: `angle`, `saturation`, `exposure`, `hue`, `mosaic`. These are all the `[net]` keys of the six files.

A negative value `v` in layer `i` of a `route` or `shortcut` names layer `i + v`; a value of zero or more is absolute.

| Section | Keys and Darknet defaults | Forward rule |
| --- | --- | --- |
| `convolutional` | `filters=1`, `size=1`, `stride=1`, `pad=0`, `padding=0`, `batch_normalize=0`, `activation=logistic` | With `pad` non-zero the padding is `size / 2`. Output extent `(in + 2 * padding - size) / stride + 1`. Order: convolution without bias; with `batch_normalize`, normalise, multiply by `scales`, add `biases`; without it, add `biases`; then the activation, one of `leaky` (slope 0.1), `linear`, `mish`, `swish`, `logistic`. A `groups` key is refused |
| `maxpool` | `stride=1`, `size=stride`, `padding=size-1` | `padding` is the total over both sides. Output extent `(in + padding - size) / stride + 1`; the first window starts at `-(padding / 2)` **(REPORT)**. A padded tap never wins |
| `upsample` | `stride=2` | Nearest neighbour by `stride`. A `scale` key is refused |
| `route` | `layers` (list), `groups=1`, `group_id=0` | Joins the listed layers along channels, in list order. With `groups > 1` each input contributes channels `[group_id * C / groups, (group_id + 1) * C / groups)` |
| `shortcut` | `from`, `activation=linear` | Adds layer `from` to the layer before. Inputs of different shapes and a non-linear activation are refused |
| `yolo` | Forward: `mask`, `anchors`, `num`, `classes`, `scale_x_y=1`, `new_coords=0`, `nms_kind`, `beta_nms`. Training: the other keys of 3.2 | Reads the layer before, which has `len(mask) * (5 + classes)` channels. Anchors are in pixels of the network input. Decode in 4.4 |

A section type or a key outside this table and 3.2 is refused with a status naming it and its line. Darknet only warns about an unused key; refusing is this importer's choice, so that a cfg it does not fully understand never runs.

### 4.2 `.weights` binary

Little-endian, no padding.

| Field | Type | Note |
| --- | --- | --- |
| `major`, `minor`, `revision` | 3 x `int32` | |
| `seen` | `uint64` when `major * 10 + minor >= 2`, else `int32` | The six files are expected to carry the 20-byte header **(REPORT, per file)** |
| Per `convolutional`, in layer order | `float32` | No other section of the six cfgs holds weights |

Per `convolutional` with `n` filters, `c` input channels and window `size`: `biases[n]`; with `batch_normalize`, `scales[n]`, `rolling_mean[n]`, `rolling_variance[n]`; then `weights[n * c * size * size]` laid out `[out, in, kh, kw]`. The reader fails with `Incompatible` on a file that ends early and on one with bytes left over, naming the layer and the byte counts.

| Darknet tensor | Library tensor | Rule |
| --- | --- | --- |
| `weights` `[out, in, kh, kw]` | `Conv2d::weight` `[out, kh, kw, in]` | Permute `(0, 2, 3, 1)`, the rule of the YOLO26 exporter (`tools/reference/README.md`, "Layouts") |
| `biases` with `batch_normalize` | `BatchNorm2d::bias` | Copy |
| `scales`, `rolling_mean`, `rolling_variance` | `BatchNorm2d::weight`, `running_mean`, `running_var` | Copy |
| `biases` without `batch_normalize` | `Conv2d::bias`, `use_bias = true` | Copy |

State names follow `build_model`'s row index. The exact strings were not read from the code; the translation unit pins them in a test against `ModelState::entries()`.

### 4.3 Batch normalisation epsilon

`BatchNorm2d` computes `(x - mean) / sqrt(variance + epsilon)`. The importer sets `bn_epsilon` to the value AlexeyAB uses, expected `1e-5` inside the root **(REPORT)**. A form with epsilon outside the root is not expressible by a change of epsilon and would need a second fold in the core tier.

### 4.4 Decodes

Head input, NHWC: `t[n, h, w, a * (5 + C) + e]`, with `a` the anchor within the head and `e` in `x, y, w, h, objectness, class_0 ...`. Darknet stores the same tensor as NCHW with that channel index. `H`, `W` are the head's grid, `net_w`, `net_h` the network input, `sigma` the logistic function, `s = scale_x_y`, `(pw, ph) = anchors[mask[a]]`.

Logit coding (`new_coords=0`: v3, v3-tiny, v4, v4-tiny):

```
bx = (w + sigma(tx) * s - (s - 1) / 2) / W        by = (h + sigma(ty) * s - (s - 1) / 2) / H
bw = exp(tw) * pw / net_w                         bh = exp(th) * ph / net_h
objectness = sigma(to)                            score_c = objectness * sigma(tc)
```

Activated coding (`new_coords=1`: v7, v7-tiny): `t` already holds logistic values; the centre uses them where the logit coding uses `sigma(t)`, and the size formula is the one in `get_yolo_box` **(REPORT)**. It is not written here.

A row is dropped when `objectness <= thresh`; a kept row's `score_c` is set to 0 unless it exceeds `thresh` **(REPORT: both gates)**. Boxes are centre x, centre y, width, height as fractions of the network input. The library multiplies by `net_w`, `net_h` before suppression, because `select_nms_detections` separates classes by a pixel offset (2.4).

Whether a dumped "raw output" of a `[yolo]` layer is its input or its own output, which in the logit coding has the logistic applied to `x`, `y`, objectness and classes, is stated by the REPORT.

### 4.5 Training: assignment and loss

All of this subsection is **(REPORT)** and is the part recalled least exactly.

Per head and image, in the logit coding with squared-error boxes (v3, v3-tiny):

- For every prediction, the best IoU of its decoded box with any truth. Where it exceeds `ignore_thresh` the prediction has no objectness term; elsewhere its objectness target is 0.
- For every truth, the anchor among all `num` anchors whose width and height overlap the truth's best, both centred on the origin. If that anchor is in this head's `mask`, the cell `(floor(x * W), floor(y * H))` is assigned: `tx = x * W - col`, `ty = y * H - row`, `tw = log(w * net_w / pw)`, `th = log(h * net_h / ph)`, box weight `2 - w * h`, objectness target 1, a one-hot class target.
- Darknet writes deltas, not a loss. The loss whose gradient at the head's input equals minus those deltas is binary cross-entropy on logits for x, y, objectness and classes and half a squared error for w and h, each times its mask and weight. The update divides by the batch size.

v4 and v4-tiny replace the four box terms with `iou_normalizer * (1 - CIoU)` on the decoded box, weight the class term by `cls_normalizer`, and (v4) assign every anchor of the head above `iou_thresh`. `max_delta` clips the box delta.

v7 and v7-tiny keep those and add `objectness_smooth` and `obj_normalizer`. Their head input is a logistic output, so Darknet's delta there is the gradient with respect to an activated value. The loss that reproduces it is fixed from the REPORT, not here.

Optimiser and schedule from `[net]`: `learning_rate`, `momentum`, `decay`, `burn_in` (the rate times `(batch / burn_in) ^ 4` until `burn_in`, exponent **(REPORT)**), `policy=steps` with `steps` and `scales`. Training parity is taken on a fixed batch without augmentation.

### 4.6 Thresholds and suppression

| Setting | Darknet |
| --- | --- |
| Confidence `thresh` | 0.25 in `detector test` **(REPORT)** |
| NMS IoU | 0.45 **(REPORT)** |
| NMS form | Per class: sort by that class's score, zero the score of a box whose criterion against a kept box exceeds the threshold. No cap on the number of rows **(REPORT)** |
| Criterion | IoU, or a distance-IoU form selected by `nms_kind` with `beta_nms` **(REPORT)** |
| Letterbox | `detector test` stretches the image unless `-letter_box` is given **(REPORT)** |

Darknet's `box_iou` has no epsilon term; `K_IOU_EPS` here is `1e-7`. A pair exactly on the threshold could be decided differently. The parity test reports such a pair instead of loosening a tolerance.

## 5. References

### 5.1 What a Darknet run dumps

The patch dumps three things, in the layout the REPORT states:

| Dump | Used for |
| --- | --- |
| The network input, the floats the first layer received | Removes image decoding and resizing from the comparison |
| Each `[yolo]` layer's raw output | Parity of the whole body, of the imported weights and, decoded, of the anchor unit |
| The final detections | Parity of the thresholds and the suppression |

No body layer is dumped. A mismatch at a head is located by importing a truncated cfg, not by a reference of an inner layer; for v3-tiny that leaves the stride-1 pool pinned by its own unit test and by the first head.

### 5.2 Fixtures

`tests/fixtures/yolo26/` holds `MANIFEST.json` and SACP files; git tracks the manifest alone (`.gitignore:44`). The Darknet fixtures copy that shape:

```
tests/fixtures/darknet/<model>/MANIFEST.json        tracked
tests/fixtures/darknet/<model>/input.sacp           ignored
tests/fixtures/darknet/<model>/forward.sacp         ignored
tests/fixtures/darknet/<model>/training.sacp        ignored (5.3)
tools/reference/export_darknet_reference.py         turns a dump directory into the three files
```

`<model>` is `yolov3_tiny`, `yolov3`, `yolov4_tiny`, `yolov4`, `yolov7_tiny`, `yolov7`. The manifest has `format_version` 1 and the keys of the YOLO26 one (`tools/reference/README.md`, "Manifest format"), with `darknet_repository` and `darknet_commit` in place of the Ultralytics pin, `cfg` and `weights` entries carrying the `sha256` and `size_bytes` of 5.4 and `"location": "external"`, the image's `sha256`, the thresholds and the suppression kind. The largest fixture is `yolov7` at 640: an input of 4.9 MB and three heads of 8.6 MB together, under the 64 MiB ceiling the YOLO26 exporter applies to a fixture directory.

Reading a fixture that git does not track, and weights through `SA_REFERENCE_DATA_DIR`, is an exception to rule 4 of the `testing` skill ("no test depends on ... a file outside the repository"); question Q1 of the plan.

### 5.3 Training references

The patch as described dumps nothing in training mode. Training parity against Darknet needs, per model, one more dump taken before the build is deleted: for one fixed batch (the fixture input, its ground truth, no augmentation, the network in training mode) each head's input under batch statistics and each `[yolo]` layer's `delta` after the forward. The library is compared at three depths: the assigner's targets and masks against the pattern and values of `delta`; the loss summed over the heads against Darknet's printed terms where the REPORT shows they are comparable; the gradient at each head's input against `-delta / batch`. Overfitting the one batch is the last check and has no Darknet counterpart. Question Q2 of the plan asks for this dump.

### 5.4 Published files

On disk in `D:\Projects\.sushisystems\models\darknet\`, with `coco.names`. None enters the repository.

| File | Bytes | Floats after a 20-byte header | SHA-256 |
| --- | --- | --- | --- |
| `yolov3-tiny.weights` | 35 434 956 | 8 858 734 | `dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c` |
| `yolov3.weights` | 248 007 048 | 62 001 757 | `523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297` |
| `yolov4-tiny.weights` | 24 251 276 | 6 062 814 | `cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87` |
| `yolov4.weights` | 257 717 640 | 64 429 405 | `e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562` |
| `yolov7-tiny.weights` | 24 967 560 | 6 241 885 | `06be302b1564f2e2c3e1db1f5ec02477ac05d0ae82405a8c1b2b8e4111101b66` |
| `yolov7.weights` | 147 898 248 | 36 974 557 | `4ecf7ca13ec5039ec7b79b0f25b156fda5eaf819d6c2bb6828ba55fe4f928332` |

The byte counts match the directory listing of 2026-10-09. The hashes are the ones the owner's ruling states; they were not recomputed here. The float count is what the strict reader checks against the cfg's shapes.

## 6. Constraints on the work

- The working tree has uncommitted edits under `modules/vision/blocks/source/`. No unit of the plan touches `vision/blocks`.
- The orchestrator's skeleton step edits `modules/vision/umbrella/include/SushiAI/vision/vision.h`; `sushiai_check_umbrella` stops the configure for a header the umbrella leaves out (`cmake/SushiAIModule.cmake`).
- The tier table exists twice and both copies need the two new rows: `cmake/SushiAILayers.cmake:41-46` and `tools/layering/check_layering.py:41-46`.
- Builds and test runs are the owner's (`PROGRAMME.md:30`). A worker checks single files with `clang -fsyntax-only`.
- Darknet's code is third-party. The patch lives outside the repository and is deleted with the build; if its text is kept (question Q2), it is listed in `NOTICE.md` with the licence read from the pinned commit.
