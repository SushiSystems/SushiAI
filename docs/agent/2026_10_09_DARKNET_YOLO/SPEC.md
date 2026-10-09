# Darknet-era YOLO: specification

**Status:** Draft, 2026-10-09. Nothing here is built or run. Stage 6 of `docs/design/PROGRAMME.md`; backlog section "Darknet-era YOLO" of `docs/design/REMAINING_WORK.md`. The units and waves are in `PLAN.md` beside this file.

## 1. Requirements

The owner ruled on 2026-10-09 (`docs/design/PROGRAMME.md:38`, `:52`, `:59-65`):

| # | Requirement |
| --- | --- |
| R1 | YOLOv1, v2, v3, v3-tiny, v4 and v4-tiny run on the library from Darknet's own `.cfg` and `.weights` files |
| R2 | One anchor-based detection unit with an options structure serves them; no class per version |
| R3 | YOLOv3-tiny is first: its forward pass on imported weights equals a reference |
| R4 | A reference is produced by running Darknet and is never typed |
| R5 | A domain adds layers and algorithms; an operation, a gradient rule or a lowering is written in the core tier (`docs/design/DOMAIN_MODULES.md:126-128`) |
| R6 | The importer is the `importers` module the domain shape foresees (`DOMAIN_MODULES.md:75`) |
| R7 | The anchor unit holds the host assigner (best-anchor IoU, ignore threshold) and the loss, as IR built from existing operations where possible. Each model has a forward-parity part first and a training part after it |
| R8 | The published Darknet weights are downloaded and stay outside the repository |

Training is in scope by the owner's ruling on Q2 of the plan, 2026-10-09 (R7). The published weights may be downloaded by the same ruling (R8).

A statement marked **(confirm)** is written from memory of Darknet and has to be checked against Darknet's source before the unit that depends on it is accepted. Every other statement about this repository was read from the cited file on 2026-10-09.

## 2. What exists

Paths are relative to `modules/` unless they start with another root.

### 2.1 Layers (`layers/nn`)

| Need | What the repository has | Verdict |
| --- | --- | --- |
| Convolution | `NN::Conv2d`, `layers/nn/include/SushiAI/nn/conv2d.hpp:42`. Weight `[out, kh, kw, in]` (`:62`), optional bias `[out]` (`:54`, `:66`). Geometry is `Graph::ConvGeometry` with one pad per axis applied to both sides (`:50`; `core/graph/include/SushiAI/graph/builder.hpp:50-60`). Model-file row `conv2d` with `filters`, `kernel`, `stride`, `padding`, `dilation`, `use_bias` (`layers/nn/source/layer_registry.cpp:69-70`, `:372-399`) | Covers every Darknet convolution of the six models: Darknet pads both sides by the same count. No `groups` key; none of the six cfgs needs one **(confirm)** |
| Batch normalisation at inference | `NN::BatchNorm2d::trace_evaluation` emits `BATCHNORM_FOLD` then `CHANNEL_AFFINE` (`batch_norm2d.hpp:103-109`): `scale = gamma / sqrt(variance + epsilon)`, `shift = beta - mean * scale` (`builder.hpp:695-699`). Epsilon defaults to `1e-5` (`batch_norm2d.hpp:50`) and is the row key `bn_epsilon` (`:122`). State: `weight`, `bias`, `running_mean`, `running_var` (`:84-92`) | Covers Darknet's inference form when epsilon sits inside the root. See 4.3 for the Darknet variant that adds it outside |
| Leaky ReLU 0.1 | `NN::LeakyReLU`, `leaky_relu.hpp:41`, slope field default `0.01` (`:44`), row key `slope` (`layer_registry.cpp:63`, `:346-356`), legal in (0, 1) (`builder.hpp:107-111`) | Covered by `{"type": "leaky_relu", "slope": 0.1}` |
| Conv + BN + leaky as one block | `Vision::Blocks::ConvNorm<Convolution, Activation>`, `vision/blocks/include/SushiAI/vision/blocks/conv_norm.hpp:55`. The activation is default-constructed (`:65`), so `ConvNorm<Conv2d, LeakyReLU>` would run slope 0.01. The file is modified in the working tree by the YOLO26 workflow | Not used. A Darknet `[convolutional]` becomes two or three rows: `conv2d`, `batchnorm2d`, activation. That needs no file YOLO26 edits |
| Max pool | `NN::MaxPool2d`, `max_pool2d.hpp:41`, holds only a `ConvGeometry` (`:44`). `forward` calls `builder.maxpool2d` (`layers/nn/source/max_pool2d.cpp:80`); `output_shape` and `forward` both use `window_extents` (`:55`, `:85`), which pads both sides equally (`core/graph/source/builder.cpp:80-82`). Row keys `kernel`, `stride`, `padding`, shared with `avgpool2d` (`layer_registry.cpp:73`, `:241-255`) | Covers size 2 stride 2 on an even extent and the SPP pools 5, 9, 13 of stride 1 with `padding = size / 2`. Does not cover size 2 stride 1; see 2.2 |
| Upsample | `NN::Upsample2d`, nearest, `scale` default 2 (`upsample2d.hpp:42-48`), row `upsample` (`layer_registry.cpp:76`, `:429-439`) | Covered |
| `route` over several layers | `NN::Concat` node, any number of `[H, W, C_i]` inputs, joined in order (`concat.hpp:46-55`), row `concat` (`layer_registry.cpp:465`). A row's `from` takes absolute indices, negative relative ones and `"input"` (`layers/nn/tests/test_node_registry.cpp:119`, `:137`); a parsed `SourceSpec` holds the absolute index (`core/config/include/SushiAI/config/model_config.hpp:50-57`) | Covered. A `route` with one layer is not a row: the rows that read it name that layer in `from` |
| `shortcut` | `GraphBuilder::add` exists (`builder.hpp:228`). No node or layer wraps it: the rows `register_layers` adds are `linear`, `gelu`, `relu`, `tanh`, `sigmoid`, `leaky_relu`, `unflatten`, `conv2d`, `maxpool2d`, `avgpool2d`, `global_avgpool`, `flatten`, `upsample`, `silu`, `batchnorm2d`, `concat` (`layer_registry.cpp:450-465`) | Missing: a two-input `Add` node and its row |
| Identity | `NN::Identity`, emits no node (`identity.hpp:41`). No row | Not needed: the importer maps Darknet layer indices to rows, so a skipped layer (`dropout`, a one-layer `route`) leaves no row |

### 2.2 The size 2, stride 1 pool

Darknet's `[maxpool] size=2 stride=1` keeps the extent: it pads one row below and one column right, none above or left **(confirm the offset rule, 4.2)**.

- `GraphBuilder::maxpool2d_with_extents(x, g, out_h, out_w)` (`builder.hpp:562`) takes the output extents from the caller and checks only that they are positive (`core/graph/source/builder.cpp:851-870`). `ConvGeometry` has no bottom or right pad, so the trailing padding is whatever the stated extents imply. `im2col_with_extents` and `col2im_with_extents` (`builder.hpp:477`, `:498`) follow the same rule.
- `core/autograd/tests/test_spatial_shapes.cpp:208-224` pins exactly this case: a 2x2 window of stride 1 over `[1, 13, 13, 256]` with extents 13x13 gives `[1, 13, 13, 256]`.
- The lowering hands the output tensor to SushiBLAS (`core/graph/source/lowering/spatial.cpp:131`), which reads the extents from it. SushiBLAS refuses a window that starts past the last row or column and a window that holds only padding (`D:\Projects\sushiblas\modules\operations\spatial\include\SushiBLAS\spatial\window.hpp:137-146`, `:182-190`); a tap outside the input is skipped and the running best starts at the lowest value of the type (`modules/kernels/spatial_kernels/include/SushiBLAS/spatial_kernels/reference/pool_max.hpp:59`, `:65`, `:70`; SYCL kernel `sycl/pool_max.hpp:74`). A padded tap therefore never wins, which is Darknet's `-FLT_MAX` behaviour.
- What is missing is above the builder: `MaxPool2d` has no field for trailing padding, calls only the symmetric emitter, and the model file has no key for it.

No IR operation and no kernel is missing for this pool.

### 2.3 Model graph, state, evaluation

| Need | What the repository has |
| --- | --- |
| A graph with branches | `NN::ModelGraph` (`model_graph.hpp:122`): `add(name, node, from)` reads any earlier ports (`:157`), so a `route` to an earlier layer by index is a `from` entry |
| Two or three heads | `ModelGraph::output(name, from)` publishes any port under a name (`:180`); `trace` returns them as `ModelOutputs` (`:196`). In a model file, `outputs` rows carry `node`, `port`, `name` (`model_config.hpp:60-70`) |
| A graph from a description | `NN::build_model(registry, Config::ModelSpec)` (`node_registry.hpp:114`) accepts "a spec built by hand, which is checked the same way" (`:109-110`), and names each row by its index (`:107`) |
| State filled by a foreign reader | `ModelState::create(engine, model)` walks a traced model (`model_state.hpp:118`); `from_entries` allocates from a list (`:129`). `ModelState::tensor(name)` returns the `SushiBLAS::Tensor*` (`:163`). `IO::import_weights` reads SACP only (`layers/io/include/SushiAI/io/import_weights.hpp:68`) and writes through that same pointer: `state.tensor(target)`, then `reader.read_payload(record, tensor->data())` (`layers/io/source/import_weights.cpp:104`, `:130`) |
| Forward-only run | `Train::EvaluationStep::make(engine, traced, state, pool)` (`layers/train/include/SushiAI/train/evaluation_step.hpp:69`), `run(inputs)` (`:96`), `output(name)` returns the tensor of a published output (`:103`). `tests/integration/test_evaluation_step.cpp:145-152` shows the sequence |

A Darknet reader can fill a state without SACP: it looks each tensor up by name and writes host floats into `tensor->data()`, as `import_weights.cpp` does. Whether `data()` is host-addressable on every device the suite runs on was not checked beyond that one call site.

### 2.4 Detection bricks (`vision/detect`), all on host spans of `float`

| Brick | Signature site | Input |
| --- | --- | --- |
| Box formats | `xyxy_to_xywh`, `xywh_to_xyxy`, `box_format.hpp:55` | `[N, 4]` |
| IoU | `intersection_over_union` (`iou.hpp:45`), `complete_iou` (`:53`), `pairwise_iou` (`:77`); `K_IOU_EPS = 1e-7` | Boxes as x1 y1 x2 y2 |
| Greedy suppression | `suppress_overlaps(boxes, scores, iou_threshold, kept)`, `detections_nms.hpp:84` | `[N, 4]` x1 y1 x2 y2, `[N]` scores |
| Class-aware NMS | `select_nms_detections(boxes, scores, shape, config, detections)`, `:96`; `NmsConfig` (`:67-76`) holds confidence, IoU threshold (default 0.7), `LabelMode`, `ClassMode`, `max_detections` 300, `max_candidates` 30000, `class_offset` 7680 | `[B, A, 4]` centre x, centre y, width, height and `[B, A, C]` probabilities; writes `[n, 6]` rows x1 y1 x2 y2 score class per image |
| Top-k | `select_top_k` (`top_k.hpp:47`), `select_top_detections` (`detections_topk.hpp:58`) | Values, or `[B, A, 4]` x1 y1 x2 y2 with `[B, A, C]` |

`vision/detect` already has `anchors.hpp` and `anchor_leaves.hpp`. They hold the anchor points of the anchor-free head, not anchor boxes. The name of the new module has to stay clear of them.

Class awareness is done by shifting each box by `class * class_offset`, so boxes must be in pixels below 7680. There is no distance-IoU suppression, which YOLOv4's cfg asks for (4.6).

### 2.5 Letterbox (`vision/imaging`)

`plan_letterbox`, `apply_letterbox(image)` and `apply_letterbox(labels, plan)` (`vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp:94`, `:105`, `:113`) work on 8-bit images, pad with a settable byte that defaults to 114 (`:45`), and move labels forward into the letterboxed image. There is no inverse that moves a box from the network frame back to the source image; a search for one under `modules/vision` found nothing. The resize kernel is the reference bilinear of the YOLO26 pipeline, not Darknet's `resize_image`, so a parity fixture starts from the tensor Darknet fed its first layer, not from an image file (5.1).

### 2.6 What training can reuse

Read from `docs/agent/2026_10_07_YOLO26/SPEC.md` section 4 and its `PLAN.md`. The YOLO26 units named there are not all written, so each row is shared when it lands.

| Need of the anchor loss | What YOLO26 brings or the library has | Shared? |
| --- | --- | --- |
| A loss that is a fixed graph over full-shape leaves, with the host writing targets and masks | YOLO26 SPEC "The loss as a fixed graph" (`:213-225`); the general loss root and `CONSTANT` (YOLO26 PLAN, "Loss roots and stop-gradient", "Constants, gradient barriers and the loss graph") | Yes: the same root, and the same masking by a leaf that is zero off the assigned rows |
| Binary cross-entropy on logits | `SUB(SOFTPLUS(x), MUL(x, t))`, YOLO26's class term (its PLAN `:450-451`); emitters `softplus`, `mul`, `sub` (`builder.hpp:306`, `:234`, `:231`) | Yes. Darknet's `yolo` deltas for x, y, objectness and classes are `t - sigma(x)`, the gradient of this term **(confirm)** |
| Squared error on raw `tw`, `th` | `sub`, `mul`, `scale` (`builder.hpp:231`, `:234`, `:283`) | Library; no new operation |
| CIoU box term (v4, v4-tiny: `iou_loss=ciou`) | YOLO26's CIoU graph over `DIV`, `MINIMUM`, `MAXIMUM`, `ATAN` (`builder.hpp:242-303`) and its guard that keeps unassigned rows finite (YOLO26 SPEC `:225`) | Yes, if YOLO26 exposes that emitter as a function of `vision/detect`; question Q12 of the plan |
| Ground truth on the host | `Detect::GroundTruthBatch`, `GroundTruthRows` (`vision/detect/include/SushiAI/vision/detect/ground_truth.hpp:44`, `:54`); the IoU bricks of 2.4 | Yes |
| Assignment that reads predictions | The ignore mask needs each predicted box's best IoU with the truths, so it needs the forward-only step and the host stage between two replays of YOLO26 SPEC `:227-237` | Yes. The best-anchor match itself reads no prediction |
| An objective with several outputs, declared leaves fed every step, a detection batch, a rank-4 input | `Train::Objective::loss` takes `ModelOutputs` and `ObjectiveLeaves` (`layers/train/include/SushiAI/train/objective.hpp`); the rest is YOLO26 SPEC "What each type must carry" (`:266-278`) | Yes; none of it is planned twice |
| SGD with momentum and weight decay; a schedule with `burn_in` and `steps` | SGD exists; the scheduler object is missing for YOLO26 too (its SPEC `:205-206`) | The object is shared; Darknet's policy is one more schedule |
| Batch normalisation in training | `K_BATCHNORM2D` and the running-statistics update (`PROGRAMME.md:40`) | Library |
| Not shared | `TaskAlignedAssigner`, the ltrb distance terms, MuSGD, EMA, and YOLO26's augmentation as a parity target | |

## 3. What is missing, per model

"Composable" means the builder already has every emitter the piece needs, so the piece is a layer in `nn` and no file under `modules/core` changes. SushiBLAS was searched under `D:\Projects\sushiblas\modules\*\*\include`.

| Model | Missing piece | Home | IR operation? | SushiBLAS kernel |
| --- | --- | --- | --- | --- |
| v3-tiny | 2x2 stride-1 pool: trailing padding on `MaxPool2d` and a row key | `layers/nn` | No; `maxpool2d_with_extents` | `pool_max` takes it (2.2) |
| v3-tiny | Decode of a `yolo` head with anchors and a mask; two heads | new `vision/detect_anchor` | No; host code | None needed |
| v3-tiny | `.cfg` reader, `.weights` reader, translation to a `ModelSpec` | new `vision/importers` | No | None needed |
| v3 | `shortcut`: an `Add` node and its row | `layers/nn` | No; `GraphBuilder::add` (`builder.hpp:228`) | `K_ADD` is an existing operation (`core/core/include/SushiAI/core/ops.hpp:308`); its kernel was not looked up |
| v3 | Three heads | none | No | Covered by the v3-tiny pieces |
| v4-tiny | `route` with `groups=2, group_id=1`: a channel slice layer and its row | `layers/nn` | No; `slice_channels` (`builder.hpp:652`) | `slice_channels` (`spatial/spatial.hpp:187`) |
| v4-tiny, v4 | `scale_x_y` in the decode | `vision/detect_anchor` | No | None needed |
| v4 | `Mish`: `x * tanh(softplus(x))` | `layers/nn` | No; `softplus`, `tanh`, `mul` (`builder.hpp:306`, `:353`, `:234`). Three kernels per activation where one fused kernel would do; a `K_MISH` is a later optimisation, not a requirement | `softplus` (`nonlinear/nonlinear.hpp:185`); no `mish` found |
| v4 | SPP: pools of 5, 9, 13 at stride 1 | none | No | Covered by `MaxPool2d` with `padding = size / 2` |
| v4 | Distance-IoU suppression (`nms_kind=greedynms`, `beta_nms`) **(confirm)** | `vision/detect` | No; host code | None needed |
| v2 | `reorg` stride 2 | `layers/nn` | No, if Darknet's channel order is block-wise: `im2col_with_extents` then `reshape` gives `[N, H/2, W/2, 4C]` in (row, column, channel) order (`builder.hpp:477`, `:712`); a different order adds `slice_channels` and `concat_channels`. The index rule is **(confirm)** (4.2). Whether a reshape of an IM2COL result keeps a usable NHWC layout tag was not checked | `im2col` (`spatial/spatial.hpp:65`) |
| v2 | `region` decode: softmax over classes, anchors in grid cells, no objectness gate | `vision/detect_anchor`, by options | No; host code | None needed |
| v1 | `connected` | `layers/nn` | Exists: `NN::Linear`, weight `[out, in]` (`layers.hpp:50-62`), `flatten` row | Exists |
| v1 | `local`: a convolution whose weights differ at every output position | `core` and `layers/nn` | Yes, or a general axis permutation: the IR has `LAYOUT_CONVERT` for rank 4 only (`builder.hpp:422`) and no transpose of a rank-3 value, so IM2COL rows cannot be regrouped by position for the rank-3 `matmul` (`:207`) | None found |
| v1 | `detection` decode | host code beside the anchor unit | No | None needed |
| v1 | `dropout` | none | No; it is the identity at inference and leaves no row | None needed |
| all, training | Host assigner: best-anchor match, ignore mask, target and mask leaves | `vision/detect_anchor` | No; host code | None needed |
| all, training | Loss as IR: cross-entropy on logits, squared error, masked sums | `vision/detect_anchor` | No, given the loss root and `CONSTANT` YOLO26 is landing (2.6) | `softplus` exists; the rest are elementwise operations already lowered for YOLO26, not looked up here |
| v4, v4-tiny, training | CIoU term; `iou_thresh` assignment | `vision/detect_anchor`, on YOLO26's CIoU emitter | No, same condition | As YOLO26 |
| v2, training | `region` terms: softmax cross-entropy as a term of a sum, rescored objectness, the early anchor pull | `vision/detect_anchor` | Open: `NLL_LOSS` cannot be a term of a sum (YOLO26 PLAN `:440-448`), and `GraphBuilder::softmax` refuses to be differentiated unless `NLL_LOSS` follows it (`builder.hpp:392`). Either that rule is lifted in the core tier or the class term needs its own formulation | Not looked up |
| v1, training | `detection` loss; dropout in training mode | `vision/detect_anchor`; `layers/nn` | Dropout in training needs a random mask leaf or an operation; not measured | Not looked up |
| all, training | Objective, detection batch, two-pass host stage, scheduler | Where YOLO26's detection objective lands (`DOMAIN_MODULES.md:123-125`) | No | None needed |

YOLOv1 is a different family. It has no anchors, no convolutional head and no `route`: a fully connected layer writes `S * S * (C + B * 5)` numbers that the `detection` layer reads as class probabilities per cell, then confidences, then boxes whose size may be stored as a square root. The options structure of R2 does not describe it; its decoder is a second function. Its `local` layer is the only piece of the six models that needs work in the core tier, and `yolov1.weights` is the largest file of the set. Question Q6 of the plan asks how much of it belongs to 0.1.0.

## 4. Formats

### 4.1 `.cfg` grammar

A text file read line by line. Darknet strips every space, tab and line ending from a line before reading it **(confirm: inner spaces too)**. An empty line and a line starting with `#` or `;` are skipped. A line starting with `[` opens a section named by the text between the brackets. Any other line is `key=value`, split at the first `=`, and belongs to the open section. A list value is comma-separated. The first section is `[net]` or `[network]`. Every later section is a layer; layers are numbered from 0 in file order.

`[net]` keys the forward pass reads: `width`, `height`, `channels` (default 3). The rest (`batch`, `subdivisions`, `momentum`, `decay`, `learning_rate`, `burn_in`, `max_batches`, `policy`, `steps`, `scales`, `angle`, `saturation`, `exposure`, `hue`, `mosaic`) are training settings and are accepted and ignored.

Layer index rule, for `route` and `shortcut`: a negative value `v` in layer `i` names layer `i + v`; a value of zero or more is absolute.

| Section | Keys and defaults | Forward rule |
| --- | --- | --- |
| `convolutional` | `filters=1`, `size=1`, `stride=1`, `pad=0`, `padding=0`, `batch_normalize=0`, `activation=logistic`, `groups=1` | If `pad` is non-zero, padding is `size / 2` (integer division) and `padding` is ignored; otherwise padding is `padding`. Output extent `(in + 2 * padding - size) / stride + 1`. Order: convolution without bias; with `batch_normalize`, normalise, multiply by `scales`, add `biases`; without it, add `biases`; then the activation. `leaky` has slope 0.1. Activations in the six cfgs: `leaky`, `linear`, `mish` **(confirm none uses `logistic` or `relu`)** |
| `maxpool` | `stride=1`, `size=stride`, `padding=size-1` | `padding` is the total over both sides. Output extent `(in + padding - size) / stride + 1`. The first window starts at `-(padding / 2)`, so `padding / 2` rows are above and the rest below **(confirm; an older pjreddie revision computed the extent differently)**. A padded tap never wins |
| `upsample` | `stride=2`, `scale=1` | Nearest neighbour by `stride`; `scale` multiplies the values and is 1 in the six cfgs **(confirm)** |
| `route` | `layers` (list), `groups=1`, `group_id=0` | Joins the outputs of the listed layers along channels, in list order. With `groups > 1` each input contributes channels `[group_id * C / groups, (group_id + 1) * C / groups)` |
| `shortcut` | `from`, `activation=linear` | Adds the output of layer `from` to the output of the layer before. The importer refuses inputs whose shapes differ; Darknet resamples them |
| `yolo` | `mask` (list; default all), `anchors` (pairs), `num=1`, `classes=20`, `scale_x_y=1`, `nms_kind`, `beta_nms`, `new_coords=0`; training keys `ignore_thresh`, `truth_thresh`, `jitter`, `random`, `iou_loss`, `iou_normalizer`, `cls_normalizer`, `iou_thresh`, `resize`, `max` | Reads the layer before, which has `len(mask) * (5 + classes)` channels. Anchors are in pixels of the network input. Decode in 4.4. `new_coords=1` is refused |
| `region` | `anchors` (pairs), `num=1`, `classes=20`, `coords=4`, `softmax=0`; training keys `bias_match`, `jitter`, `rescore`, `object_scale`, `noobject_scale`, `class_scale`, `coord_scale`, `absolute`, `thresh`, `random` | Reads the layer before, `num * (coords + 1 + classes)` channels. Anchors are in cells of this layer's grid. Decode in 4.4. `softmax_tree`, `log`, `sqrt`, `background` are refused when set |
| `reorg` | `stride=1`, `reverse=0`, `flatten=0`, `extra=0` | Moves each `stride x stride` block of pixels into channels: `[H, W, C]` to `[H / stride, W / stride, C * stride * stride]`. The exact element order is **(confirm)**: pjreddie's `reorg_cpu` computes its indices from the input extents in a way that is not a plain space-to-depth, and AlexeyAB keeps that form as `reorg_old` |
| `connected` | `output=1`, `activation=logistic`, `batch_normalize=0` | `y = W x + b`, `W` as `[output, inputs]`; the input is the layer before flattened in channel, row, column order |
| `local` | `filters=1`, `size=1`, `stride=1`, `pad=0`, `activation=logistic` | Unshared convolution. `pad` is the padding itself, not `size / 2`. Output extent `(in - (pad ? 1 : size)) / stride + 1` **(confirm)** |
| `detection` | `classes=1`, `coords=1`, `num=1`, `side=7`, `sqrt=0`, `softmax=0`; training keys `rescore`, `jitter`, `object_scale`, `noobject_scale`, `class_scale`, `coord_scale`, `forced`, `random` | Decode in 4.4 |
| `dropout` | `probability=0.5` | Identity at inference |
| `avgpool` | none | Mean over height and width, `[H, W, C]` to `[C]`. Used by classification cfgs, by none of the six **(confirm)** |

A section type outside this table is refused with a status that names it and its line. So is a key that is neither a forward key nor in the list of training keys. Darknet itself only warns about an unused key; refusing is this importer's choice, so that a cfg the importer does not fully understand never runs. `[crop]` appears in early YOLOv1 cfgs and rescales pixel values **(confirm whether `yolov1.cfg` has it)**.

### 4.2 `.weights` binary

All values little-endian, no padding.

| Field | Type | Note |
| --- | --- | --- |
| `major`, `minor`, `revision` | 3 x `int32` | |
| `seen` | `uint64` when `major * 10 + minor >= 2`, else `int32` | pjreddie also requires `major < 1000 && minor < 1000` for the 64-bit form **(confirm)**. YOLOv3 and v4 files are 0.2.0 (20-byte header); YOLOv2 and v1 files are older **(confirm per file)** |
| Per layer, in layer order | `float32` | Only `convolutional`, `connected`, `local` (and a standalone `batchnorm`, in none of the six) hold weights |

Per `convolutional` with `n` filters, `c` input channels and window `size`:

1. `biases[n]`
2. with `batch_normalize`: `scales[n]`, `rolling_mean[n]`, `rolling_variance[n]`
3. `weights[n * (c / groups) * size * size]`, laid out `[out, in, kh, kw]`

Per `connected`: `biases[outputs]`, `weights[outputs * inputs]` as `[out, in]`, then with `batch_normalize` `scales`, `rolling_mean`, `rolling_variance` each `[outputs]`: the weights come before the statistics here, unlike a convolution **(confirm)**. A file whose `major` or `minor` exceeds 1000 stores this matrix transposed **(confirm)**; the reader refuses such a file.

Per `local`: `biases[out_h * out_w * n]`, then `weights[out_h * out_w * n * c * size * size]` **(confirm order and layout)**.

The reader is strict in both directions: a file that ends early and a file with bytes left over both fail with `Incompatible`, naming the layer and the byte counts.

Conversion to this library:

| Darknet tensor | Library tensor | Rule |
| --- | --- | --- |
| Convolution `weights` `[out, in, kh, kw]` | `Conv2d::weight` `[out, kh, kw, in]` (`conv2d.hpp:62`) | Permute `(0, 2, 3, 1)`, the rule of the YOLO26 exporter (`tools/reference/README.md`, "Layouts") |
| `biases` with `batch_normalize` | `BatchNorm2d::bias` | Copy |
| `scales` | `BatchNorm2d::weight` | Copy |
| `rolling_mean`, `rolling_variance` | `running_mean`, `running_var` | Copy |
| `biases` without `batch_normalize` | `Conv2d::bias`, `use_bias = true` | Copy |
| Connected `weights` `[out, in]` after a feature map `[C, H, W]` | `Linear::weight` `[out, H * W * C]` | Column `c * H * W + h * W + w` moves to `(h * W + w) * C + c`, because `Flatten` folds NHWC without moving data (`flatten.hpp:40`) |

State names follow `build_model`'s row index, `"<row>.weight"` and so on. The exact strings were not read from the code; the translation unit pins them in a test against `ModelState::entries()`.

### 4.3 Batch normalisation epsilon

AlexeyAB computes `(x - mean) / sqrt(variance + 0.00001)`; pjreddie computes `(x - mean) / (sqrt(variance) + 0.000001)` **(confirm both)**. `BatchNorm2d` has the first form only. With the reference taken from AlexeyAB (question Q4) the importer sets `bn_epsilon` to `1e-5` and no new code is needed. The pjreddie form is not expressible by a change of epsilon.

### 4.4 Decodes

Raw head output, NHWC: `t[n, h, w, a * (5 + C) + e]`, with `a` the anchor within the head and `e` in `x, y, w, h, objectness, class_0 ...`. Darknet stores the same tensor as NCHW with that channel index, so the dump permutes `(0, 2, 3, 1)` and nothing else. `H`, `W` are the head's grid, `net_w`, `net_h` the network input, `sigma` the logistic function.

`yolo` (v3, v3-tiny, v4, v4-tiny), with `s = scale_x_y` and anchor `(pw, ph) = anchors[mask[a]]`:

```
bx = (w + sigma(tx) * s - (s - 1) / 2) / W        by = (h + sigma(ty) * s - (s - 1) / 2) / H
bw = exp(tw) * pw / net_w                         bh = exp(th) * ph / net_h
objectness = sigma(to)                            score_c = objectness * sigma(tc)
```

A row is dropped when `objectness <= thresh`; a kept row's `score_c` is set to 0 unless it exceeds `thresh` **(confirm both gates)**. Darknet's `yolo` layer applies the logistic to `x`, `y`, objectness and classes in place and leaves `w`, `h` raw, so its own output is half decoded; the reference records the convolution output before it (5.1).

`region` (v2), anchors in cells:

```
bx = (w + sigma(tx)) / W                          by = (h + sigma(ty)) / H
bw = exp(tw) * pw / W                             bh = exp(th) * ph / H
objectness = sigma(to)                            score_c = objectness * softmax(tc)_c
```

`score_c` is set to 0 unless it exceeds `thresh`; there is no separate objectness gate **(confirm)**.

`detection` (v1), over the `S * S * (C + B * 5)` outputs of the last `connected`, laid out as `S * S * C` class probabilities, then `S * S * B` confidences, then `S * S * B * 4` boxes **(confirm the layout)**: `bx = (col + x) / S`, `by = (row + y) / S`, `bw = w`, `bh = h`, each squared when `sqrt=1`; `score_c = confidence * class_c`.

All boxes are centre x, centre y, width, height as fractions of the network input. The library multiplies by `net_w`, `net_h` before suppression, because `select_nms_detections` separates classes by a pixel offset (2.4).

### 4.5 Training: assignment and loss

All of this subsection is **(confirm)**. It is the part of Darknet recalled least exactly, and the reference of 5.3 settles it.

`yolo` head, per image:

- For every prediction, the best IoU of its decoded box with any truth. Its objectness target is 0, and it has no objectness loss where that IoU exceeds `ignore_thresh` (0.7 in the v3 cfgs).
- For every truth, the anchor among all `num` anchors whose width and height overlap the truth's best, both centred on the origin. If that anchor is in this head's `mask`, the cell `(floor(x * W), floor(y * H))` is assigned: `tx = x * W - col`, `ty = y * H - row`, `tw = log(w * net_w / pw)`, `th = log(h * net_h / ph)`, box weight `2 - w * h`, objectness target 1, a one-hot class target. A truth whose best anchor belongs to another head is not assigned here.
- Darknet writes deltas, not a loss: `weight * (t - sigma(tx))` for x and y, `weight * (t - tw)` for w and h, `t - sigma(.)` for objectness and classes. The cost it prints is the sum of squared deltas and is not what is minimised. The loss whose gradient at the raw head equals minus those deltas is binary cross-entropy on logits for x, y, objectness and classes and half a squared error for w and h, each times its mask and weight. The update divides by the batch size.

v4 and v4-tiny add `iou_loss=ciou` with `iou_normalizer`, `cls_normalizer` and `obj_normalizer` as term weights, `iou_thresh`, which also assigns every anchor of the head whose shape IoU with the truth exceeds it, and `scale_x_y` in the decoded box the IoU reads.

`region` (v2): the same best-anchor match over one head; `object_scale`, `noobject_scale`, `coord_scale`, `class_scale` as term weights; with `rescore=1` the objectness target is the IoU of the assigned prediction; classes through a softmax; while `seen` is below 12 800 images every box is also pulled toward its anchor with a small weight. `detection` (v1): per cell the box with the best IoU is responsible; `coord_scale` 5, `noobject_scale` 0.5, sizes compared as square roots.

Optimiser and schedule from `[net]`: `learning_rate`, `momentum`, `decay`, `burn_in` (the rate times `(batch / burn_in) ^ 4` until `burn_in`), `policy=steps` with `steps` and `scales`. The augmentation keys (`jitter`, `random`, `hue`, `saturation`, `exposure`, `mosaic`) describe Darknet's own loader, which is not reproduced; training parity is taken on a fixed batch.

### 4.6 Thresholds and suppression

| Setting | Darknet default |
| --- | --- |
| Confidence `thresh` | 0.25 in AlexeyAB's `detector test`, 0.5 in pjreddie's **(confirm)** |
| `hier_thresh` | 0.5; unused without a class tree |
| NMS IoU | 0.45 |
| NMS form | Per class: sort by that class's score, zero the score of a box whose IoU with a kept box exceeds the threshold (`do_nms_sort`). No cap on the number of rows **(confirm)** |
| YOLOv4 and v4-tiny | The cfg sets `nms_kind=greedynms` and `beta_nms=0.6`, which selects a distance-IoU criterion (`diounms_sort`) **(confirm, and confirm that `detector test` honours it)** |
| Letterbox | pjreddie letterboxes every image; AlexeyAB's `detector test` stretches the image unless `-letter_box` is given **(confirm)**. Darknet pads with 0.5 in float RGB scaled to [0, 1] and embeds at `(net - new) / 2` |

Darknet's `box_iou` has no epsilon term; `K_IOU_EPS` here is `1e-7`. A pair that sits exactly on the threshold could be decided differently. The parity test reports such a pair instead of loosening a tolerance.

## 5. References

### 5.1 What a Darknet run dumps

| Tensor | Shape, as written | Why |
| --- | --- | --- |
| `input.image` | `[1, net_h, net_w, 3]` NHWC, the floats Darknet's first layer received | Removes image decoding and resizing from the comparison |
| `head.<k>.raw` | `[1, H, W, A * (5 + C)]`, the output of the layer before each `yolo` or `region` | Parity of the whole body and of the imported weights |
| `head.<k>.activated` | Same shape, the `yolo` or `region` layer's own output | Shows where a decode difference starts |
| `head.<k>.boxes`, `head.<k>.scores` | `[1, H * W * A, 4]`, `[1, H * W * A, C]`, row `(h * W + w) * A + a` | Parity of the anchor unit, before any threshold |
| `detections` | `[n, 6]` x1 y1 x2 y2 score class in network pixels, after `thresh` and NMS | Parity of the post-processing |
| A few body layers (`layer.<i>.output`) | `[1, H, W, C]` | Locates a mismatch. For v3-tiny: layers 11 and 12, either side of the stride-1 pool |

The manifest records the thresholds, the NMS form, whether the image was letterboxed and the number of rows before and after suppression. SACP cannot hold an empty tensor (`tools/reference/README.md`, "Validation row counts"), so an image is chosen that yields detections: `data/dog.jpg` of the Darknet tree, which stays outside this repository.

### 5.2 The YOLO26 fixture, and the Darknet one as its sibling

`tests/fixtures/yolo26/` holds `MANIFEST.json` and five SACP files; git tracks the manifest alone (`.gitignore`: `tests/fixtures/yolo26/*.sacp`). `tools/reference/export_yolo26_reference.py` writes them, refuses more than 64 MiB in the output directory (`:40`), writes `weights.sacp` elsewhere under `--external-weights DIR` and marks it `"location": "external"` (`:267-277`). The manifest has `format_version` 1, `snake_case` keys, the generator's path, the pinned commit and versions, the seed, and per file `name`, `location`, `sha256`, `size_bytes`, `context` and `tensors[]` with `name`, `dtype`, `shape`, `original_shape`, `layout`, `source`, `min`, `max`, `mean` (`tools/reference/README.md`, "Manifest format"). `tools/tests/test_export_yolo26_reference.py` tests the torch-free parts.

The Darknet fixture copies that shape:

```
tests/fixtures/darknet/<model>/MANIFEST.json        tracked
tests/fixtures/darknet/<model>/input.sacp           ignored, regenerated by the tool
tests/fixtures/darknet/<model>/forward.sacp         ignored, regenerated by the tool
tests/fixtures/darknet/<model>/training.sacp        ignored, regenerated by the tool (5.3)
tools/reference/export_darknet_reference.py
tools/tests/test_export_darknet_reference.py
```

Differences from YOLO26, each recorded in the manifest: `darknet_commit` and `darknet_repository` in place of the Ultralytics pin; `cfg` and `weights` entries with `sha256`, `size_bytes` and `"location": "external"`; no seed, since the weights are the published ones; the image's `sha256`. `<model>` is `yolov3_tiny`, `yolov3`, `yolov4_tiny`, `yolov4`, `yolov2`, `yolov1`.

### 5.3 Training references

One fixed batch per model: the fixture image as Darknet fed it, its ground truth, no augmentation, the network in training mode. Darknet dumps, per head, the raw output under batch statistics and the `delta` tensor after the forward (`[1, H, W, A * (5 + C)]`, permuted as the raw output is); the weight gradient of the first convolution and of the last convolution before each head; one batch-norm layer's rolling statistics after the step. The library is compared at three depths: the assigner's targets and masks against the pattern and the values of `delta`; the gradient of the loss at each raw head against `-delta / batch`; the two weight gradients. Overfitting the one batch is the last check and has no Darknet counterpart. `training.sacp` joins `input.sacp` and `forward.sacp` in the fixture, with `learning_rate`, `momentum`, `decay`, `seen` and the batch size in its `context`.

### 5.4 Published files

Sizes are recalled; each is **(confirm at download)**, where the tool writes the size and the SHA-256 into the manifest. The parameter count is `(size - header) / 4` and is what the strict reader checks.

| File | Bytes | Floats after the header |
| --- | --- | --- |
| `yolov3-tiny.weights` | 35 434 956 | 8 858 734 |
| `yolov3.weights` | 248 007 048 | 62 001 757 |
| `yolov4-tiny.weights` | 24 251 276 | 6 062 814 |
| `yolov4.weights` | 257 717 640 | 64 429 405 |
| `yolov2.weights` | 203 934 260 | about 50 983 560; the header length is not settled |
| `yolov1.weights` | about 750 MB | not recalled |

None of these enters the repository. `yolov3-tiny.weights` with its dumps is under the 64 MiB ceiling the YOLO26 exporter applies to a fixture directory; the others are not. Question Q8 of the plan asks where the downloads live and how a test that needs them is reported when they are absent.

## 6. Constraints on the work

- The YOLO26 workflow is editing `modules/vision/blocks`, `modules/vision/registration`, `modules/core/core/include/SushiAI/core/ops.hpp` and `modules/core/graph/{include,source}/.../builder.*`. No forward unit of the plan edits those except the `local` layer. The training units build on operations, the loss root and the trainer changes YOLO26 is still landing (2.6), so every one that emits IR or binds the trainer is "after YOLO26 waves".
- The orchestrator's skeleton step edits `modules/vision/umbrella/include/SushiAI/vision/vision.h`, which the YOLO26 workflow must also edit: its four new block headers are not in the umbrella yet (`vision.h:70-74`), and `sushiai_check_umbrella` stops the configure for a header the umbrella leaves out (`cmake/SushiAIModule.cmake:193-217`).
- The tier table exists twice and both copies need the new rows: `cmake/SushiAILayers.cmake:28-47` and `tools/layering/check_layering.py:41-46`.
- Builds and test runs are the owner's (`PROGRAMME.md:30`). A worker checks single files with `clang -fsyntax-only`.
- Darknet's code is third-party. A patch against it, and any file ported from it, is listed in `NOTICE.md` with the licence read from the pinned commit; the licence was not read for this document.
