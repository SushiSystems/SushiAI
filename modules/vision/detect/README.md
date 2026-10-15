# detect

The module has three parts: the root, which holds what both detection families share, in namespace
`SushiAI::Vision::Detect`; `free/`, which holds the anchor-free path of YOLO26, in
`SushiAI::Vision::Detect::Free`; and `anchor/`, which holds anchor-box detection, in
`SushiAI::Vision::Detect::Anchor`. Headings below name each header with its folder, so `free/` and
`anchor/` headers carry the prefix and root headers keep their bare names.

The parts of a detector's training and inference that run on the host over plain arrays: anchor
grids, box coding, IoU, label assignment, the leaves a masked loss reads, both postprocess paths
and the validation metric. Every unit is a function or a small class over `SushiRuntime::span`,
with the shape written in its comment. Nothing here reaches an engine, SYCL or `graph/lowering`,
so every unit is tested without a device.

The module depends on `core` (`SA_THROW_IF`, `Error`, `Status`, `SA_FATAL_IF`) and on `config`
(`Config::Value`, which `anchor/anchor_task.hpp` reads and writes). Nothing depends on
it yet. It adds no IR operation, so nothing in it waits for a lowering. The requirements are sections 2, 4 and 7 of
`docs/agent/2026_10_07_YOLO26/SPEC.md`; the rulings that put label assignment on the host and ask
for both validation heads are in `docs/design/PROGRAMME.md`.

Each unit is a port of Ultralytics at commit `ce1399439b93c490b648b55dedd0a6334f0199da`. `U/` below
is `ultralytics/` at that commit, installed read-only in the conda environment
`sushiai-yolo26-ref`. Quoted lines are the ones ported. Arithmetic is float32 in the reference's
operation order; the build compiles with `-ffp-contract=off`, so no multiply and add is fused.

| Header | Holds |
| --- | --- |
| `free/anchors.hpp` | `FeatureLevel`, `K_POINT_VALUES`, `count_anchors`, `make_anchors`, `scale_points` |
| `free/level_rows.hpp` | `LevelPlacement`, `join_level`, `split_level` |
| `free/anchor_leaves.hpp` | `AnchorLeaves`, `build_anchor_leaves` |
| `box_format.hpp` | `xyxy_to_xywh`, `xywh_to_xyxy` |
| `free/box_coding.hpp` | `decode_xyxy`, `decode_xywh`, `encode_distances`, `scale_boxes` |
| `iou.hpp` | `intersection_over_union`, `complete_iou`, the elementwise forms, `pairwise_iou` |
| `iou_gradient.hpp` | `K_ASPECT_TRADE_OFF_EPS`, `BoxGradient`, `complete_iou_gradient` |
| `top_k.hpp` | `select_top_k` |
| `ground_truth.hpp` | `GroundTruthRows`, `GroundTruthBatch`, `pad_ground_truth` |
| `free/assigner_config.hpp` | `AssignerConfig` and the two branch presets |
| `free/assignment.hpp` | `Assignment` |
| `free/assigner.hpp` | `AssignerInputs`, `TaskAlignedAssigner` |
| `free/target_leaves.hpp` | `TargetLeafInputs`, `TargetLeaves`, `build_target_leaves`, `K_SAFE_BOX` |
| `detection_row.hpp` | The six columns of a detection, `K_MAX_DETECTIONS`, `K_VALIDATION_CONFIDENCE` |
| `prediction_shape.hpp` | `PredictionShape` |
| `detections_topk.hpp` | `count_top_detections`, `select_top_detections`, `keep_confident_detections` |
| `suppression_criterion.hpp` | `K_DEFAULT_DISTANCE_EXPONENT`, `SuppressionCriterion`, `measure_suppression` |
| `detections_nms.hpp` | `LabelMode`, `ClassMode`, `NMSConfig`, `suppress_overlaps`, `select_nms_detections` |
| `average_precision.hpp` | `K_IOU_THRESHOLDS`, `ClassMetrics`, `MetricsSummary`, `match_detections`, `integrate_precision`, `AveragePrecision` |
| `anchor/anchor_size.hpp` | `AnchorSize`, `shape_iou` |
| `anchor/anchor_layout.hpp` | `AnchorLayout`, `validate_anchor_layout` |
| `anchor/head_options.hpp` | `HeadCoding`, `HeadOptions`, `HeadShape`, `NetworkSize`, the entry constants, `validate_head`, `count_head_channels`, `count_head_rows` |
| `anchor/head_decode.hpp` | `decode_head` |
| `anchor/head_training_options.hpp` | `BoxLoss`, `ObjectnessTarget`, `HeadTrainingOptions`, `validate_head_training` |
| `anchor/head_truth.hpp` | `validate_head_truth` |
| `anchor/head_delta.hpp` | `HeadDelta`, `compute_head_delta` |
| `anchor/head_cost.hpp` | `HeadCost`, `measure_head_cost` |
| `anchor/anchor_clustering.hpp` | `ClusteringOptions`, `AnchorClustering`, `cluster_anchors` |
| `anchor/anchor_count.hpp` | `CountOptions`, `CountTrial`, `AnchorCountChoice`, `choose_anchor_count` |
| `anchor/head_assignment.hpp` | `assign_anchors_to_heads` |
| `anchor/anchor_fit.hpp` | `FitOptions`, `AnchorFit`, `measure_anchor_fit` |
| `anchor/assignment_thresholds.hpp` | `AssignmentThresholds`, `derive_assignment_thresholds` |
| `anchor/anchor_review.hpp` | `AnchorVerdict`, `ReviewOptions`, `AnchorReview`, `review_anchors` |
| `anchor/anchor_plan.hpp` | `AnchorPlanOptions`, `AnchorPlan`, `plan_anchors` |
| `anchor/anchor_task.hpp` | `K_ANCHOR_TASK_TYPE`, `AnchorTaskHead`, `AnchorTask`, `read_anchor_task`, `write_anchor_task` |
| `anchor/head_training_section.hpp` | `K_ANCHOR_OBJECTIVE_TYPE`, `read_head_training_section`, `write_head_training_section` |
| `anchor/seeded_sequence.hpp` | `Detail::SeededSequence` |

## free/anchors.hpp

### count_anchors

The anchor count A of a list of levels is the sum of `height * width`: one anchor per cell
(`U/utils/tal.py:453`, `h * w` rows per level). It throws on a level without cells or with a
stride that is not positive. Every other unit sizes its `[A, ...]` arrays by this count.

`K_POINT_VALUES` is 2, the width of an anchor point, x then y.

### make_anchors

One anchor per cell, at the cell centre in grid units, levels in order and row-major inside a
level. Anchor `h * W + w` of a level is element `[b, h, w, :]` of that level's NHWC head output.

```
U/utils/tal.py:449  sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset  # shift x
U/utils/tal.py:450  sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset  # shift y
U/utils/tal.py:451  sy, sx = torch.meshgrid(sy, sx, indexing="ij") ...
U/utils/tal.py:452  anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
U/utils/tal.py:453  stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
```

The offset is 0.5 at both call sites (`U/utils/loss.py:424`, `U/nn/modules/head.py:223`), so it is
the constant `K_GRID_CELL_OFFSET` and not a parameter. Strides are written as `[A]`, not `[A, 1]`.

### scale_points

The assigner takes anchor points in pixels: `anchor_points * stride_tensor`
(`U/utils/loss.py:442`).

## free/level_rows.hpp

### join_level, split_level

The reference concatenates the levels on the anchor axis (`U/nn/modules/head.py:176`). SushiAI's
head outputs and loss leaves are per level, `[B, H, W, C]`. `join_level` copies one level into its
place in `[B, A, C]` for the assigner and the postprocess; `split_level` copies a slice of an
assigner output back out as one level's leaf. Both are plain copies and do not reorder inside a
level.

## free/anchor_leaves.hpp

### build_anchor_leaves

The loss graph decodes a level's `[B, H, W, 4]` distances into corners with `K_SUB` and `K_ADD`, and
both take operands of one shape. The anchor points therefore reach the graph with a batch axis.
`build_anchor_leaves` writes them once per image:

| Leaf | Joined | Per level | Value |
| --- | --- | --- | --- |
| `points` | `[B, A, 2]` | `[B, H, W, 2]` | cell centre, x then y, grid units, as `make_anchors` writes it |
| `strides` | `[B, A, 4]` | `[B, H, W, 4]` | the anchor's stride in each of the four values |

`split_level` with 2 and 4 channels cuts them into a level's leaves, the same way it cuts the
target leaves. `points` is what the two halves of the distances are subtracted from and added to
(`U/utils/tal.py:470-471`). `strides` has the shape of the decoded box it multiplies
(`U/nn/modules/head.py:226`); inside one level it is a single value, so a per-level graph can use
`K_SCALE` by the level's stride and leave this leaf unbound.

Both arrays depend on the input size and the batch size alone. A trainer builds them once and
rebuilds them when either changes. The call throws on a batch below 1 and on an empty level list.

## box_format.hpp

### xyxy_to_xywh, xywh_to_xyxy

```
U/utils/ops.py:257  y[..., 0] = (x1 + x2) / 2  # x center
U/utils/ops.py:259  y[..., 2] = x2 - x1  # width
U/utils/ops.py:281  wh = x[..., 2:] / 2  # half width-height
U/utils/ops.py:282  y[..., :2] = xy - wh  # top left xy
U/utils/ops.py:283  y[..., 2:] = xy + wh  # bottom right xy
```

## free/box_coding.hpp

### decode_xyxy, decode_xywh

```
U/utils/tal.py:469  lt, rb = distance.chunk(2, dim)
U/utils/tal.py:470  x1y1 = anchor_points - lt
U/utils/tal.py:471  x2y2 = anchor_points + rb
U/utils/tal.py:473  c_xy = (x1y1 + x2y2) / 2
U/utils/tal.py:474  wh = x2y2 - x1y1
```

The one-to-many head decodes to xywh and the one-to-one head to xyxy
(`U/nn/modules/head.py:247`, `xywh=not self.end2end and not self.xyxy`). The loss decodes to xyxy
for both (`U/utils/loss.py:410`). With `reg_max = 1` the DFL layer is the identity, so the
distances are the raw box outputs.

The `[A, 2]` anchor points repeat for every image: row `r` of `[B, A, 4]` uses point `r % A`.
`decode_xywh` is `decode_xyxy` followed by `xyxy_to_xywh` in place, the two steps of
`U/utils/tal.py:470-474`.

### scale_boxes

Decoded boxes are in grid units until they are multiplied by the stride:
`self.decode_bboxes(...) * self.strides` (`U/nn/modules/head.py:226`) and
`pred_bboxes.detach() * stride_tensor` (`U/utils/loss.py:441`).

### encode_distances

```
U/utils/tal.py:490  x1y1, x2y2 = bbox.chunk(2, -1)
U/utils/tal.py:491  dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
```

The clamp at `U/utils/tal.py:493` is not ported: the L1 branch calls `bbox2dist` without `reg_max`
(`U/utils/loss.py:143`).

## iou.hpp

### intersection_over_union, complete_iou

`bbox_iou` with `xywh=False`. `K_IOU_EPS` is its default `eps = 1e-7` (`U/utils/metrics.py:115`).
The eps enters each height, the union, the squared hull diagonal and the trade-off divisor, in
that order of operations:

```
U/utils/metrics.py:145  w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
U/utils/metrics.py:149  inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
U/utils/metrics.py:150      b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
U/utils/metrics.py:154  union = w1 * h1 + w2 * h2 - inter + eps
U/utils/metrics.py:157  iou = inter / union
U/utils/metrics.py:159  cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
U/utils/metrics.py:162  c2 = cw.pow(2) + ch.pow(2) + eps
U/utils/metrics.py:164  (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
U/utils/metrics.py:165  ) / 4
U/utils/metrics.py:167  v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
U/utils/metrics.py:169  alpha = v / (1 - iou + v + eps)
U/utils/metrics.py:170  return iou - (rho2 / c2 + v * alpha)
```

`4 / math.pi**2` is a Python double that torch narrows to float32 before the multiply, so
`K_ASPECT_FACTOR` is that double cast to `float`. `std::atan` on `float` and torch's `atan` may
round a last bit apart; the tests state a tolerance for that.

The scalar forms check no size. A caller passes spans of four values.

### pairwise_iou

`box_iou`, which differs from `bbox_iou`: the heights carry no eps and only the divisor does.

```
U/utils/metrics.py:101  wh, wh1, wh2 = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0), a2 - a1, b2 - b1
U/utils/metrics.py:105  return inter / (wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter + eps)
```

No unit of this module calls it yet. The suppression uses its own formula (see
`detections_nms.hpp`).

## iou_gradient.hpp

### BoxGradient, complete_iou_gradient

`complete_iou_gradient(box, target, form, gradient)` writes the gradient of the complete IoU of
two `x_centre y_centre width height` boxes at `box`. `gradient` receives four components in the
order of `box`, and the measure rises along them. All four are 0 when both boxes are one point.
Each span must hold four values.

`BoxGradient` names the form. `Analytic` is the derivative of the measure. `DarknetCompatible`
is the gradient section 11 of `docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md` states.
`K_ASPECT_TRADE_OFF_EPS` is 1e-6, the term that keeps the trade-off of the aspect penalty
finite.

`test_detect_iou_gradient.cpp` compares the analytic form with a central difference of step
1e-6, within 1e-6 relative plus 1e-7 absolute, on fixed pairs and on 100 seeded pairs.

## top_k.hpp

### select_top_k

Returns the indices of the k largest values, largest first, and orders equal values by ascending
index. `torch.topk` and `argsort` leave the order of equal values unspecified, so this rule is
SushiAI's own, and it is where a result can differ from the reference. Four call sites inherit it:

| Call site | Reference line | What a tie decides |
| --- | --- | --- |
| first top-k of the assigner (`assigner.cpp`, `select_candidates`) | `U/utils/tal.py:263` | which candidates of equal metric a box keeps |
| second top-k of the assigner (`assigner.cpp`, `keep_second_top_k`) | `U/utils/tal.py:376` | which held anchors of equal metric survive `topk2` |
| candidate limit of the suppression (`detections_nms.cpp`, `select_image_detections`) | `U/utils/nms.py:141` | which boxes of equal score pass `max_candidates` |
| visiting order of the suppression (`detections_nms.cpp`, `suppress_overlaps`) | `U/utils/nms.py:269` | which of two overlapping boxes of equal score is kept |

The two selections of `select_top_detections` (`U/nn/modules/head.py:285-287`) use it as well.

It matters in the fixture. The reference model has random weights and nearly constant scores: 255
of the 300 rows of `one2one.detections` share their score with another row, and 11 anchor and
class pairs tie for the last 5 places. `test_detect_postprocess.cpp` therefore compares the score
and class columns row by row, compares the rows above the last score as a set, and checks that
every row is some anchor's box with that anchor's score.

## ground_truth.hpp

### GroundTruthRows

The label rows of a batch as a data loader holds them: the image index, the class and the box of
each row, the box as centre x, centre y, width, height, each a fraction of the image side. These
are `batch_idx`, `cls` and `bboxes` of the reference batch (`U/utils/loss.py:431`).

### GroundTruthBatch

The same rows as `[B, M]` arrays, M being the largest box count of an image: labels, pixel
corner boxes and the 0/1 flag the reference calls `mask_gt`. The struct holds what the assigner
reads and nothing else. `pad_ground_truth` counts the rows of each image in a local `[B]` array,
one allocation of B integers per call.

### pad_ground_truth

`v8DetectionLoss.preprocess` and the two lines after its call:

```
U/utils/loss.py:394  out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
U/utils/loss.py:399  out[batch_idx, within_idx] = targets[:, 1:]
U/utils/loss.py:400  out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
U/utils/loss.py:432  ... scale_tensor=imgsz[[1, 0, 1, 0]])
U/utils/loss.py:434  mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
```

The four values are scaled by the image width and height and converted with `xywh_to_xyxy`.
A row is valid when the sum of its four pixel corners is above zero, as the reference derives it.
Rows of one image keep their order. The reference computes a row's slot as its index minus the
image's first index, which assumes rows grouped by image; this port counts per image, which gives
the same slots for grouped rows and stays defined for ungrouped ones. Labels are `int64` here and
float in the reference.

## free/assigner_config.hpp

### AssignerConfig

The member defaults are the constructor defaults of `TaskAlignedAssigner`
(`U/utils/tal.py:33-39`): `topk = 13`, `num_classes = 80`, `alpha = 1.0`, `beta = 6.0`,
`eps = 1e-9`, `topk2 = topk`. `stride_val` is the second stride, or the only one
(`U/utils/tal.py:58-59`), which is 16 for the default strides 8, 16 and 32.

### one_to_many_assigner_config, one_to_one_assigner_config

The detection loss builds both assigners with `alpha=0.5, beta=6.0` and the model's strides
(`U/utils/loss.py:374-381`), and `E2ELoss` chooses the candidate counts:

```
U/utils/loss.py:1327  self.one2many = loss_fn(model, tal_topk=10)
U/utils/loss.py:1328  self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
```

`MANIFEST.json` records the same values as read back from the reference's assigner objects.

## free/assignment.hpp

### Assignment

The five outputs of `TaskAlignedAssigner.forward` at full shape over all anchors, so a fixed-shape
loss can feed them as leaves. `target_boxes` are in pixels. A row off the foreground holds the box
and the label of ground-truth row 0 of its image and index 0, as in the reference
(`U/utils/tal.py:297-300`, `381`).

## free/assigner.hpp

### AssignerInputs

The arguments of one `TaskAlignedAssigner.forward` call (`U/utils/tal.py:64`) with their
extents written out: `pd_scores` as `[B, A, C]`, `pd_bboxes` as `[B, A, 4]`, `anc_points` as
`[A, 2]`, `gt_labels` as `[B, M]`, `gt_bboxes` as `[B, M, 4]` and `mask_gt` as `[B, M]`. The
spans are views; the caller keeps the arrays alive for the call.

### TaskAlignedAssigner

`assign` runs the reference's `_forward` image by image. The inputs are what the loss passes
(`U/utils/loss.py:439-446`): class probabilities after sigmoid, predicted boxes in pixel xyxy,
anchor points in pixels, and the padded ground truth.

**No ground truth.** With `max_boxes == 0` every label is `num_classes` and everything else is
zero (`U/utils/tal.py:90-97`).

**Candidates.** An anchor is a candidate of a box when its centre lies inside the box by more
than `1e-9`. A valid box with a side below `stride_val` is first widened to `stride_val` about
its centre. Every box goes through the xywh round trip, enlarged or not, and the port keeps that
arithmetic by calling `xyxy_to_xywh` and `xywh_to_xyxy` of `box_format.hpp`.

```
U/utils/tal.py:335  wh_mask = gt_bboxes_xywh[..., 2:] < self.stride_val
U/utils/tal.py:336  gt_bboxes_xywh[..., 2:] = torch.where((wh_mask * mask_gt).bool(), stride_val, ...)
U/utils/tal.py:344  mask = xy_centers[:, 0] - lt[..., 0] > eps
U/utils/tal.py:347  mask &= rb[..., 1] - xy_centers[:, 1] > eps
```

The `1e-9` here is the default argument of `select_candidates_in_gts` (`U/utils/tal.py:317`), not
`AssignerConfig::eps`; it is `K_INSIDE_MARGIN` in `assigner.cpp`.

**Metric.** For a candidate, the overlap is the complete IoU of the original box with the
prediction, clamped at zero, and the metric is `score^alpha * overlap^beta`. Both are zero off
the candidates.

```
U/utils/tal.py:226  bbox_scores = pd_scores[indices[0], indices[2], gt_labels[...].long()]
U/utils/tal.py:228  align_values = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)
U/utils/tal.py:246  return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
```

The power is a square root for the exponent 0.5, the value for 1, a product for 2 and `std::pow`
for anything else. The two exponents the detection loss uses, 0.5 and 6.0, are the ones the tests
pin, and the two reference tests compare their result with torch's on the fixture within `1e-5`.
An exponent outside that list is not claimed to round as torch rounds it.

**First top-k.** Each valid box keeps the `topk` anchors of largest metric that are also
candidates (`U/utils/tal.py:263-275`, `202`). When a box has fewer than `topk` positive metrics
the reference fills the rest with zero-metric anchors in an unspecified order; here they are the
lowest indices, and they survive only if they are candidates.

**Tie break.** An anchor kept by more than one box goes to the box of largest overlap, the first
one on equal overlaps. The maximum is taken over every box, including one that did not keep the
anchor.

```
U/utils/tal.py:367  mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
U/utils/tal.py:368  max_overlaps_idx = overlaps.max(1).indices
U/utils/tal.py:371  mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
```

**Second top-k.** When `topk2 != topk`, each box then keeps the `topk2` anchors of largest metric
among those it still holds (`U/utils/tal.py:373-379`).

**Targets and normalisation.** The foreground flag and the box index are the column maximum of
the final mask (`U/utils/tal.py:381`). The target score of a foreground anchor, in the class of
its box, is its metric times the box's largest overlap over the box's largest metric plus `eps`:

```
U/utils/tal.py:170  align_metric *= mask_pos
U/utils/tal.py:171  pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
U/utils/tal.py:173  pos_overlaps = overlaps.amax(dim=-1, keepdim=True)
U/utils/tal.py:174  align_metric.mul_(pos_overlaps).div_(pos_align_metrics + self.eps)
U/utils/tal.py:176  target_scores = target_scores * norm_align_metric
```

**Not ported.** The out-of-memory retry (`U/utils/tal.py:99-138`) and
`RotatedTaskAlignedAssigner`.

**Refusals the reference does not make by name.** `assign` throws when the anchor count is below
`topk` or `topk2` (torch's `topk` would fail there) and when a valid box has a class outside
`[0, num_classes)`. Every refusal happens before `assignment` is written.

**Cost.** One image allocates nothing after the first call: the `[M, A]` metric, overlap and mask
arrays are members and are reused. The complete IoU runs only on candidates. Each valid box costs
one `O(A log topk)` selection.

## free/target_leaves.hpp

### build_target_leaves

The arrays the masked loss of SPEC section 4, "The loss as a fixed graph", reads for one branch,
over `[B, A]`. The class leaf is `Assignment::target_scores` itself and is not copied.

| Leaf | Value | From |
| --- | --- | --- |
| `weight` | sum of a row's target scores | `U/utils/loss.py:133` |
| `target_scores_sum` | `[1]`: `max(sum of all target scores, 1)` | `U/utils/loss.py:448` |
| `normalized_weight` | `weight / target_scores_sum` | `U/utils/loss.py:135` |
| `box_mask` | 1 in the four values of a foreground row, else 0 | `U/utils/loss.py:132` |
| `target_boxes` | `target_boxes / stride` on the foreground, `K_SAFE_BOX` off it | `U/utils/loss.py:461` |
| `target_distances` | `encode_distances` of that box on the foreground, 0 off it | `U/utils/loss.py:143` |
| `l1_scale` | `stride / side * 0.25 * normalized_weight`, 0 off the foreground | `U/utils/loss.py:145-154` |
| `safe_boxes` | `K_SAFE_BOX` in every row | SPEC section 8, D3 |

`side` is the image width for the left and right distances and the height for the top and bottom
(`U/utils/loss.py:146-147`). The 0.25 is the mean over four sides (`U/utils/loss.py:152`).

`K_SAFE_BOX` is `(0, 0, 1, 1)` in grid units. Ruling D3 replaces the operands of an unassigned row
with a fixed valid box before the IoU arithmetic; `safe_boxes` is that box as a full-shape leaf
for the prediction side, and `target_boxes` already carries it on the target side. Its complete
IoU with itself is finite: the union is 1 + 3e-7 and the aspect term is 0.

`target_scores_sum` is accumulated in double and narrowed once. The reference sums in float32 in
an order torch does not specify, so the two may differ in the last bits.

`l1_scale` is the SPEC's own derivation of the L1 term, not a port.

### Leaf rank and element type

The arrays are stored over the joined anchor axis, because the assigner works on it. The graph
multiplies per level, and its binary operations take operands of one shape, so each array reaches
the graph through `split_level`:

| Stored | `LevelPlacement::channels` | Leaf of one level | Arrays |
| --- | --- | --- | --- |
| `[B, A, 4]` | 4 | `[B, H, W, 4]` | `box_mask`, `target_boxes`, `target_distances`, `l1_scale`, `safe_boxes`, `AnchorLeaves::strides` |
| `[B, A, 2]` | 2 | `[B, H, W, 2]` | `AnchorLeaves::points` |
| `[B, A]` | 1 | `[B, H, W, 1]` | `weight`, `normalized_weight` |
| `[B, A, C]` | C | `[B, H, W, C]` | `Assignment::target_scores` |
| `[1]` | not cut | `[1]` | `target_scores_sum` |

`target_scores_sum` is a `std::array<float, 1>`, so an objective binds its storage as a one-element
leaf and divides the class sum by it.

Every leaf is FLOAT32. `GraphBuilder::select` takes its condition in the element type of its
branches, so `box_mask` fits a FLOAT32 loss as it is. A loss built in another element type casts
each leaf to that type with `K_CAST` before its first use; the module writes no second copy.

## detection_row.hpp

A detection is six floats: x1, y1, x2, y2, score, class index (`U/nn/modules/head.py:259`).
`K_MAX_DETECTIONS` is `Detect.max_det = 300` (`U/nn/modules/head.py:83`).
`K_VALIDATION_CONFIDENCE` is 0.001, the confidence the validator sets when none is given
(`U/engine/validator.py:144-145`); both validation paths use it.

## prediction_shape.hpp

### PredictionShape

The batch, the anchor count and the class count of `[B, A, 4]` boxes and `[B, A, C]` scores. Both
postprocess paths take one, and each throws unless its two arrays hold exactly that many values,
so a score array of another class count is refused.

## detections_topk.hpp

Both postprocess paths take a whole batch: the boxes, the scores, their `PredictionShape`, then
the settings of the path. `select_top_detections` writes a fixed `[B, k, 6]` array;
`keep_confident_detections` and `select_nms_detections` write one row vector per image, because
their row counts differ between images.

### count_top_detections

Returns the rows per image, `k = min(max_det, anchors)` (`U/nn/modules/head.py:279`), and throws
on an extent or a limit that is not positive. The limit is an argument of the two top-k functions;
the reference's value is `K_MAX_DETECTIONS`.

### select_top_detections

`Detect.postprocess` on the one-to-one head, the path that is not class-agnostic and not grouped:

```
U/nn/modules/head.py:279  k = min(max_det, anchors)
U/nn/modules/head.py:285  ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1]
U/nn/modules/head.py:286  scores = self._gather(scores, ori_index)
U/nn/modules/head.py:287  scores, index = self._grouped_topk(scores.flatten(1), k, groups)
U/nn/modules/head.py:288  return scores[..., None], (index % nc)[..., None].float(), self._gather(ori_index, index // nc)
```

The k anchors of largest best-class score are kept; their k by C scores are flattened; the k
largest of those give the class as `index % C` and the anchor as entry `index / C` of the first
selection. One anchor can therefore yield several detections. The inputs are decoded pixel xyxy
boxes and probabilities, which is what `_inference` hands over (`U/nn/modules/head.py:216-217`).

### keep_confident_detections

Validation of the one-to-one head is `select_top_detections` followed by this filter at
`K_VALIDATION_CONFIDENCE`. The reference validator passes the head's rows through
`non_max_suppression` with `end2end=True` (`U/models/yolo/detect/val.py:168-176`), whose
end-to-end branch suppresses nothing and keeps the rows above the confidence:

```
U/utils/nms.py:74  mask = pred[:, 4] > conf_thres
U/utils/nms.py:77  idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
U/utils/nms.py:78  output.append(pred[idx].float())
```

A score equal to the threshold is dropped and the rows keep their order. The function takes the
rows and the batch size and nothing else: the rows per image are the row count over the batch,
and it throws when the batch does not divide them.

The `[:max_det]` cut is not ported. The caller keeps `k`, the limit it gave
`select_top_detections`, at or below the validator's row limit. Both are 300 in the reference
(`U/nn/modules/head.py:83`, `U/cfg/default.yaml:57`), so the cut removes nothing there. With a
larger top-k limit this function keeps rows the validator drops. The `classes` filter of
`U/utils/nms.py:75-76` is not ported either.

A validation consumer of the one-to-one head has to call this filter. Without it the metric is
computed over rows the reference drops: in the fixture no score reaches 0.001, so the reference
validator keeps none of the 300 rows (`validation.row_count` in `MANIFEST.json`).

## suppression_criterion.hpp

### SuppressionCriterion, measure_suppression

`measure_suppression(box1, box2, criterion, distance_exponent)` is the number a suppression
compares with its threshold, for two x1 y1 x2 y2 boxes:

```
IoU         = intersection / union, and 0 when the intersection or the union is 0
Iou:          IoU
DistanceIoU:  IoU - (d / c)^distance_exponent, and IoU when c is 0
```

`d` is the squared distance of the two centres and `c` the squared diagonal of the smallest box
that encloses both. The IoU carries no eps, unlike `intersection_over_union` of `iou.hpp`, so
this file computes its own. `DistanceIoU` of two boxes that do not overlap is negative, so it
never exceeds a threshold of 0 or more. `K_DEFAULT_DISTANCE_EXPONENT` is 0.6.
`distance_exponent` is not checked; with an exponent of 0 the term is 1 for every pair. The
behaviour is section 4.5 of `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`.

## detections_nms.hpp

### LabelMode, ClassMode

`LabelMode` is the reference's `multi_label` flag: `MultiLabel` lets an anchor contribute every
class above the threshold, `BestClass` only its highest (`U/utils/nms.py:117-123`). `ClassMode`
is its `agnostic` flag: `ClassAware` suppresses within a class, `ClassAgnostic` across classes
(`U/utils/nms.py:146`). Each replaces a boolean parameter.

### NMSConfig

The defaults are what the validator passes, not the defaults of `non_max_suppression`:

| Field | Value | From |
| --- | --- | --- |
| `confidence_threshold` | 0.001 | `U/engine/validator.py:145` |
| `iou_threshold` | 0.7 | `U/cfg/default.yaml:56` |
| `label_mode` | `MultiLabel` | `U/models/yolo/detect/val.py:173` |
| `class_mode` | `ClassAware` | `U/models/yolo/detect/val.py:174`, `U/cfg/default.yaml:69` |
| `max_detections` | 300 | `U/cfg/default.yaml:57` |
| `max_candidates` | 30000 | `U/utils/nms.py:23` |
| `class_offset` | 7680 | `U/utils/nms.py:24` |
| `criterion` | `IoU` | the reference has one criterion |
| `distance_exponent` | 0.6 | read by `DistanceIoU` only |

The function's own defaults are 0.25, 0.45 and best class only (`U/utils/nms.py:15-19`).

### select_nms_detections

`non_max_suppression` for a detect head without extra channels. It takes `[B, A, 4]` boxes,
`[B, A, C]` scores and their `PredictionShape`, as `select_top_detections` does, and writes one
`[n, 6]` vector per image. Each image goes through:

```
U/utils/nms.py:86   xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
U/utils/nms.py:106  x[:, :4] = xywh2xyxy(x[:, :4])
U/utils/nms.py:118  i, j = torch.where(cls > conf_thres)
U/utils/nms.py:123  conf, j = cls.max(1, keepdim=True)
U/utils/nms.py:141  filt = x[:, 4].argsort(descending=True)[:max_nms]
U/utils/nms.py:146  c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
U/utils/nms.py:152  boxes = x[:, :4] + c  # boxes (offset by class)
U/utils/nms.py:160  i = i[:max_det]  # limit detections
```

A score must exceed the confidence threshold; an equal score is dropped. Multi-label mode applies
only with more than one class (`U/utils/nms.py:92`). Class-aware suppression shifts each box by
its class index times `class_offset`, so boxes of different classes never overlap; the output
rows carry the unshifted boxes.

Not ported: the `classes` filter, the time limit (`U/utils/nms.py:165-167`), rotated boxes and
extra channels. The end-to-end branch (`U/utils/nms.py:71-80`) is `keep_confident_detections` in
`detections_topk.hpp`.

### suppress_overlaps

`TorchNMS.nms`, which the reference describes as matching torchvision's:

```
U/utils/nms.py:266  areas = (x2 - x1) * (y2 - y1)
U/utils/nms.py:269  order = scores.argsort(0, descending=True)
U/utils/nms.py:293  if inter.sum() == 0:
U/utils/nms.py:297  iou = inter / (areas[i] + areas[rest] - inter)
U/utils/nms.py:299  order = rest[iou <= iou_threshold]
```

The IoU has no eps. A box survives when its IoU with the kept box is at most the threshold, so an
IoU equal to the threshold survives. A pair whose IoU is not a number is dropped, as in the
reference. Equal scores are visited by ascending index; the reference's `argsort` leaves that
order unspecified.

The four-argument form is the six-argument form under `SuppressionCriterion::IoU`. The
six-argument form takes the criterion and its exponent and walks the same order: a kept box
removes every later box whose `measure_suppression` against it exceeds the threshold. A row whose
score is 0 is never kept and removes nothing; the reference keeps such a row.
`select_nms_detections` passes only scores above a threshold of at least 0, so no zero score
reaches the suppression from it. The four-argument form throws on boxes that are not `[N, 4]`
for `[N]` scores; the six-argument form stops the process on those and on a criterion outside
the enumeration.

`non_max_suppression` calls `torchvision.ops.nms` in place of `TorchNMS.nms` when torchvision is
already imported (`U/utils/nms.py:154-159`). The exporter's run reached `TorchNMS.nms`
(`validation.suppression` in `MANIFEST.json`); a full validator run imports torchvision through
its data pipeline. This port follows `TorchNMS.nms`. torchvision's kernel is compiled and was not
read, so whether the two agree on an IoU that is not a number is not established here.

## average_precision.hpp

The validation metric: detections are matched to ground truth at ten IoU thresholds, the matches
of every image are accumulated, and one table comes out with precision, recall, AP50 and AP50-95
per class and their means. The reference computes this part in numpy float64, so the unit works
in `double`; only the IoU and the thresholds are float32. `Unit_AveragePrecision` in
`tests/vision/detect/test_average_precision.cpp` holds its tests. No fixture carries a
reference metric table, so those tests compare with values derived by hand.

### K_IOU_THRESHOLDS

The ten thresholds are the float32 values of `torch.linspace(0.5, 0.95, 10)`:

```
U/models/yolo/detect/val.py:59  self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
```

The header writes them as the decimal literals 0.5f to 0.95f, which round to the same float32
bits. `K_IOU_THRESHOLD_COUNT` is 10. `K_RECALL_POINTS` is 101 and `K_CONFIDENCE_POINTS` is 1000,
the lengths of the two `np.linspace` grids below. `K_METRIC_EPS` is `eps = 1e-16`, the default
argument of `ap_per_class`.

### match_detections

Takes one image: `[N, 6]` detection rows, `[M, 4]` ground-truth boxes in the same frame and `[M]`
classes. It writes `[N, K_IOU_THRESHOLD_COUNT]` flags, 1 where the detection is a true positive
at that threshold. The IoU is `pairwise_iou` of `iou.hpp`, as in the reference:

```
U/models/yolo/detect/val.py:393  iou = box_iou(batch["bboxes"], preds["bboxes"])
U/engine/validator.py:335        iou = iou * correct_class  # zero out the wrong classes
U/engine/validator.py:346        matches = np.nonzero(iou >= threshold)  # IoU >= threshold and classes match
U/engine/validator.py:350        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
U/engine/validator.py:351        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
U/engine/validator.py:352        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
U/engine/validator.py:353        correct[matches[:, 1].astype(int), i] = True
```

Line 351 keeps one pair per detection, the one of largest IoU. Line 352 then keeps one pair per
box, and because the pairs are by then in detection order, the box goes to its first detection.
The port does the same in two steps. Each detection first takes the box of its class with the
largest IoU that is at least 0.5; of two boxes with equal IoU it takes the later one. Then, at
each threshold, the detections are visited in row order and a detection whose IoU reaches the
threshold takes its box unless an earlier row holds it. Three consequences follow, and the tests
pin each:

- A detection whose best box is taken stays unmatched. It is not offered its second-best box.
- A box goes to the first detection that reaches the threshold, not to the one of largest IoU.
- A detection of another class never matches, whatever its overlap.

A detection whose IoU is not a number matches at no threshold. `iou >= threshold` is false for
NaN in the reference, and both comparisons of the port are written in that direction.

The function throws when the detection array is not a whole number of rows, when the box count
and the class count disagree, when a ground-truth class is negative, when a detection's score is
not finite, when a detection's class is not a whole number in `[0, 2^24)`, and when `correct`
does not hold ten values per detection. A box coordinate that is NaN or infinite is not refused:
such a row is matched to nothing and counts as a false positive, as in the reference.

Not ported: the `use_scipy` branch (`U/engine/validator.py:338-344`).

### integrate_precision

`compute_ap` with `method = "interp"`:

```
U/utils/metrics.py:782  mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
U/utils/metrics.py:783  mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
U/utils/metrics.py:786  mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
U/utils/metrics.py:791  x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
U/utils/metrics.py:793  ap = func(np.interp(x, mrec, mpre), x)  # integrate
```

The curve gets a point at recall 0 with precision 1 and two closing points with precision 0.
The envelope makes precision non-increasing from the right, it is sampled at 101 recall points,
and the 100 trapezoids are summed. The last sample is at recall 1, where the envelope is 0, so a
detector that finds every box scores 0.99 + 0.01 / 2 = 0.995 and not 1. An empty curve gives 0.5,
the area under the line from 1 to 0.

The function takes recall and precision after each detection in descending score and throws
when the two differ in length. It does not check that recall never decreases.

### AveragePrecision

`add_image` checks one image as `match_detections` does, matches it and stores the scores, the
classes and the flags of its detections, the classes of its boxes and the distinct classes the
image holds. A refused image leaves the accumulator as it was. The caller passes detections that
already went through the validation confidence and the per-image limit (`K_VALIDATION_CONFIDENCE`
and `K_MAX_DETECTIONS` of `detection_row.hpp`), in the frame of the boxes; the class applies
neither. `clear` drops every image and `image_count` returns how many were added.

`summarize` is `ap_per_class` over everything accumulated:

```
U/utils/metrics.py:843  i = np.argsort(-conf)
U/utils/metrics.py:847  unique_classes, nt = np.unique(target_cls, return_counts=True)
U/utils/metrics.py:868  recall = tpc / (n_l + eps)  # recall curve
U/utils/metrics.py:869  r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
U/utils/metrics.py:872  precision = tpc / (tpc + fpc)  # precision curve
U/utils/metrics.py:873  p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score
U/utils/metrics.py:884  f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
U/utils/metrics.py:892  i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
```

The table has one row per class that has ground truth, in ascending class index. A class with
detections and no ground truth gets no row and no share of a mean. A class with ground truth and
no detections gets a row of zeros, which enters every mean. With no ground truth at all the
table has no rows and the four means are 0.

For a class with both, the detections are ranked by descending score, recall and precision are
accumulated down the ranking at each threshold, and `integrate_precision` gives the AP of each.
`ap50` is the AP at IoU 0.5 and `ap50_95` is the mean of the ten. The recall and precision at
IoU 0.5 are also sampled at 1000 confidence values from 0 to 1.

Precision and recall are not read at a fixed confidence. The F1 curve of each class is averaged
over the rows, smoothed, and the confidence point where the result is largest is the one every
row is read at:

```
U/utils/metrics.py:670  nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
U/utils/metrics.py:672  yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
U/utils/metrics.py:673  return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
```

With 1000 points and `f = 0.1` the box filter is 101 points wide, and the curve is padded with
50 copies of its first and of its last value. Of equal maxima the first is taken, as `argmax`
does.

The means are those of the reference's `Metric`: `precision` and `recall` are the means of the
rows (`U/utils/metrics.py:963`, `972`), `map50` is the mean of the rows' AP at
IoU 0.5 (`U/utils/metrics.py:981`) and `map50_95` is the mean over rows and thresholds
(`U/utils/metrics.py:999`). `ClassMetrics::images` counts the images that hold a box of the
class, from `"target_img": np.unique(cls)` (`U/models/yolo/detect/val.py:258`), and `instances`
counts its boxes.

**Equal scores.** `np.argsort(-conf)` does not say in which order equal scores come out. The
port ranks detections of equal class and score in the order they were added, so its result for
such an input is defined and the reference's is not.

**Summation order.** The 100 trapezoids, the mean of a row's ten APs and the four means of the
table are summed in the order numpy's pairwise summation adds them: left to right below 8
values, eight lanes from 8 to 128 values, and a split in two above that. The F1 mean over the
rows and the taps of the box filter are summed left to right.

**Not ported.** The plots, and the true and false positive counts `ap_per_class` also returns.

**Cost.** The accumulator holds one float, one `int64` and ten bytes per detection, one `int64`
per box and one `int64` per distinct class of each image. `summarize` sorts the detections once,
and for each class with detections it runs ten passes over that class's detections and fills two
curves of 1000 points.

## The anchor part

`anchor/` holds detection with anchor boxes, the family of YOLOv3, v4 and v7. It is written from
the behaviour `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md` states in formulas (sections 4.4 and
5.5) and ports no code. A content failure comes back as a `Status` of `core`; a call that breaks
a stated precondition stops the process through `SA_FATAL_IF`. Nothing in this part throws.

An anchor is stored as a fraction of the network input, so one layout serves every input size
and every grid. Pixels do not appear in this part.

## anchor/anchor_size.hpp

### AnchorSize, shape_iou

`AnchorSize` is a width and a height as fractions of the network input. A label box takes the
same form. `shape_iou(a, b)` is the IoU of two sizes laid on one centre:

```
intersection = min(a.width, b.width) * min(a.height, b.height)
union        = a.width * a.height + b.width * b.height - intersection
shape_iou    = intersection / union, and 0 when union is not above 0
```

It is float32 and carries no eps. Two sizes without area give 0, where the quotient would be 0/0.

## anchor/anchor_layout.hpp

### AnchorLayout

`anchors` lists every anchor of the model. `head_anchors[k]` lists the anchors head `k` uses, as
indices into `anchors`; entry `a` of that list is the head's anchor slot `a`, which fixes the
order of its channels and of its decoded rows.

### validate_anchor_layout

Returns `InvalidDescription` for a layout without anchors or without heads, an anchor whose side
is not finite and above zero, a head without anchors, and an index below 0 or past the anchors.
The message names the anchor, head and slot. Two heads may share an anchor, and an anchor may
belong to no head: a layout read from a file can have both, and `validate_anchor_layout` accepts
it. A head that lists one anchor twice is not refused.

## anchor/head_options.hpp

### HeadCoding, HeadOptions

`HeadCoding` names how the published values of a head encode a box. Under `Logit` the offsets,
the objectness and the classes are logits and the sizes are exponents. Under `Activated` the
layer before the head already applied a logistic, every entry arrives in (0, 1), and the decode
applies none. `HeadOptions` adds the class count and `scale_xy`, the stretch of an offset about
the cell centre.

### HeadShape, NetworkSize

`HeadShape` is the batch, the grid rows and the grid columns of a head's
`[B, H, W, A * (5 + C)]` output. `NetworkSize` is the input in pixels; the decode does not read
it, and a caller that needs pixel boxes multiplies by it.

### The entries of a slot

A head has `A` anchor slots per cell and each slot has `5 + C` entries. `K_X_ENTRY`,
`K_Y_ENTRY`, `K_WIDTH_ENTRY`, `K_HEIGHT_ENTRY` and `K_OBJECTNESS_ENTRY` are 0 to 4;
`K_FIRST_CLASS_ENTRY` is 5, and class `c` is entry `5 + c`. Entry `e` of slot `a` is channel
`a * (5 + C) + e`.

### validate_head, count_head_channels, count_head_rows

`validate_head` returns the failure of `validate_anchor_layout`, or `InvalidDescription` for a
head index outside the layout, a class count below 1, or a `scale_xy` that is not finite and
above zero. `count_head_channels` is `A * (5 + C)`, the channel count the convolution before the
head must produce. `count_head_rows` is `H * W * A`, the rows one image decodes to; it does not
multiply by the batch. Both counts are `noexcept` and stop the process on a head index outside
the layout.

## anchor/head_decode.hpp

### decode_head

Reads a head's `[B, H, W, A * (5 + C)]` output and writes `[B, H * W * A, 4]` boxes and
`[B, H * W * A, C]` scores. With `t` the entries of one slot in cell `(row, col)`, `(pw, ph)` the
slot's anchor, `s = scale_xy` and `act` the logistic under `Logit` and the identity under
`Activated`:

```
X  = act(t[x]) * s - (s - 1) / 2        Y  = act(t[y]) * s - (s - 1) / 2
bx = (col + X) / W                      by = (row + Y) / H
Logit:      bw = exp(t[w]) * pw         bh = exp(t[h]) * ph
Activated:  bw = t[w]^2 * 4 * pw        bh = t[h]^2 * 4 * ph
score_c = act(t[objectness]) * act(t[class_c])
```

A box is centre x, centre y, width and height as fractions of the network input. Row
`(row * W + col) * A + a` of an image holds slot `a` of cell `(row, col)`. The decode applies no
threshold and clamps nothing: with `scale_xy` above 1 a centre can leave its cell by
`(s - 1) / 2` of a cell, and a row whose objectness is near 0 is still written.

The arithmetic is float32 in the order written above. The two codings are two rule types with
the same pair of static functions, and the cell loop is a template over the rule, so the coding
is chosen once per call and not per entry. The call allocates nothing. The rules and the decode
of one slot's box are defined once, in the private `source/anchor/slot_decode.hpp`, and the
training delta reads a slot through the same definitions.

`decode_head` stops the process when `validate_head` fails on its arguments, when an extent of
the shape is below 1, and when a span does not hold exactly the values its shape gives.

`test_detect_anchor_decode.cpp` pins each formula with inputs at zero or at saturation, to 1e-6
absolute. A logit of 100 is the saturated input: its float32 logistic is exactly 1, and that of
-100 exactly 0. No test compares with a reference tensor.

## Training an anchor head

### HeadTrainingOptions, compute_head_delta

`anchor/head_training_options.hpp` holds the thresholds, the box loss, the objectness target and
the weights. `compute_head_delta` in `anchor/head_delta.hpp` reads a head's
`[B, H, W, A * (5 + C)]` output and the truths and writes a delta of the same shape; training
descends along `-delta`. The steps are those of SPEC 4.6. `X, Y, O, P_c` are a slot's offsets,
objectness and class probabilities as `decode_head` reads them, so under `Activated` they are
the published entries themselves.

Step 1, every slot. `best` is the largest IoU of the slot's decoded box with a truth of the
image. The slot is ignored when `best > ignore_threshold` and some `P_c > class_gate`.

```
dO = object_weight * (0 - O)
ignored, One:      dO = 0
ignored, BestIoU:  dO = max(dO, object_weight * (best - O))
```

Step 2, every truth. The slot of the truth's best anchor is assigned at the truth's cell when
the head owns that anchor, and below a `match_threshold` of 1 so is every other slot of the head
whose anchor's `shape_iou` with the truth exceeds the threshold. For an assigned slot:

```
One:      dO = object_weight * (1 - O)
BestIoU:  dO = object_weight * (1 - O) only where dO is still 0
dP_c = (c == label ? 1 : 0) - P_c
```

When the delta of the truth's class is already non-zero, that class alone is rewritten. Step 3
runs below a `match_threshold` of 1: a slot with a non-zero `dO` divides its four box deltas by
its count of positive class deltas.

The box delta of an assigned slot adds to what the entries hold. One row per box loss and coding
supplies it:

| Box loss | Coding | Box delta |
| --- | --- | --- |
| `SquaredError` | `Logit` | `k * (target - entry)` per entry, `k = (2 - tw * th) * box_weight` |
| `CompleteIoU` | `Logit` | the gradient `g` of complete IoU at the decoded box: `g_x`, `g_y`, `g_w * exp(t[w])`, `g_h * exp(t[h])`, each times `box_weight` |
| `CompleteIoU` | `Activated` | `g_x`, `g_y`, `g_w`, `g_h`, each times `box_weight` |

`g` is `complete_iou_gradient` in the form `HeadTrainingOptions::box_gradient` names. The default
is `BoxGradient::Analytic`, the derivative of the measure. `BoxGradient::DarknetCompatible` is the
gradient of REPORT section 11, which the cfg importer sets on every head it reads. The field has
no effect under `SquaredError`.

A `CompleteIoU` component that is not finite adds 0, and with `max_delta` above 0 each is
clipped to `[-max_delta, max_delta]`. No row exists for `SquaredError` under `Activated`,
because the SPEC states no target for it, and `compute_head_delta` stops the process on that
pair. Under `Activated` the delta is at the output of the logistic layer before the head; the
slope of that logistic belongs to that layer's backward pass.

The options hold no class weight. REPORT section 17 has Darknet's `cls_normalizer` act only
together with per-class counters, which this library does not have.

`assigned_row` and `assigned_class` record the truth of a slot's best anchor alone;
`assignments` and `iou_sum` count every assigned slot.

`validate_head_truth` in `anchor/head_truth.hpp` checks the truth a head is trained against. A
row counts when its `valid` flag is set; the list of an image ends at `max_boxes`, and no value
of a row ends it. It returns `InvalidDescription` naming the image, the row and the value for a
class outside the head's classes, a centre outside [0, 1) on either axis, and a width or height
that is not positive.

`compute_head_delta` returns a `Status`. A truth comes from a dataset, so the delta returns what
`validate_head_truth` says about it and leaves `result` untouched. The caller's own arguments
are preconditions and stop the process: options that fail `validate_head` or
`validate_head_training`, an extent or a network side below 1, a coding, objectness target or
box loss without a row, and a head span whose length is not the shape's. Those are checked
before the truth, so a refused truth never hides one of them.

`test_detect_anchor_delta.cpp`, `test_detect_anchor_delta_iou.cpp` and
`test_detect_anchor_delta_activated.cpp` compute each expected value in double in the test, the
box gradient by a central difference that `tests/vision/detect/anchor/anchor_delta_support.hpp` defines once. An
objectness or class delta is held to 1e-6 absolute, a box delta to 1e-5 absolute plus 1e-5
relative. `test_detect_anchor_delta_gradient.cpp` runs both gradient forms through the delta of
one slot and holds the compatible form to the twelve dumped deltas of REPORT section 11 at 1e-4
relative. `test_detect_anchor_head_truth.cpp` covers each refusal of a truth row.

### HeadCost, measure_head_cost

`measure_head_cost` in `anchor/head_cost.hpp` reads a `HeadDelta` with the head's shape, options
and training options and returns the cost the head reports for one batch. No gradient is taken
of it. `B` is the batch size:

```
total   = sum of every squared delta entry / B
non_box = object_weight * sum of the squared non-box entries / B
box     = total - non_box
stored, SquaredError:  sum of every squared delta entry
stored, CompleteIoU:   box_weight * mean of (1 - IoU) over the assignments
                       + object_weight * sum of the squared non-box entries
```

The IoU term of `stored` is 0 when the head has no assignment, and no batch size divides
`stored`. The delta must hold `batch * height * width * slots * (5 + C)` values for some slot
count, and the batch, the grid and the class count must be positive.

`test_detect_anchor_cost.cpp` holds the four numbers to 1e-12 for a delta built by hand.

## Automatic anchoring

A new model gets its anchors from the label boxes of its dataset. Six functions each do one step,
and `plan_anchors` calls them in order. The rules are in SPEC sections 5.2 to 5.4. A box is an
`AnchorSize`: the width and height of a label as fractions of the network input.

| Step | Function | Gives |
| --- | --- | --- |
| Count | `choose_anchor_count` | Anchors per head: it grows from `min_per_head` while one more per head raises the mean best IoU by at least `min_gain` |
| Anchors | `cluster_anchors` | `count` anchors in ascending area, from a seeded clustering under `1 - shape_iou` and a refinement that cannot lower the fit |
| Heads | `assign_anchors_to_heads` | An `AnchorLayout`: runs of anchors in ascending area, the smallest to the smallest stride |
| Fit | `measure_anchor_fit` | Mean best IoU, best possible recall, the share of boxes whose best anchor no head owns, each anchor's share |
| Thresholds | `derive_assignment_thresholds` | `match_threshold` and `ignore_threshold` from the fit |
| Review | `review_anchors` | The fit of an existing layout, a proposal of the same counts, and `Keep` or `Propose`; it changes nothing |

An imported model keeps the anchors, heads and thresholds of its file. `review_anchors` is the
call for such a model, and `plan_anchors` is the call for a model that has no anchors yet.

## anchor/anchor_plan.hpp

### plan_anchors

Takes the boxes, one stride per head and `AnchorPlanOptions`, and returns an `AnchorPlan`:

```
heads      = strides.size()
anchors    = per_head == 0 ? choose_anchor_count(boxes, heads, count).clustering.anchors
                           : cluster_anchors(boxes, count.clustering with count = per_head * heads)
layout     = assign_anchors_to_heads(anchors, strides, even split)
fit        = measure_anchor_fit(boxes, layout, fit options)
thresholds = derive_assignment_thresholds(fit)
```

`plan_anchors` holds none of the rules itself. Head `k` of the layout belongs to `strides[k]`, so
the strides may come in any order: under strides 32 and 16 head 0 takes the three largest of six
anchors. `trials` holds the counts `choose_anchor_count` tried, the refused successor last. With
`per_head` above 0 one clustering runs, `trials` is empty, and of `count` only `count.clustering`
is read. The count fixes the channel count of every head, so the plan is made before the model is
built.

The fit is measured on every box passed in. `cluster_anchors` leaves out a box with a side that
is not finite or not above `min_side`, and the plan does not report how many it left out; a
caller that needs the number calls `cluster_anchors` itself.

`plan_anchors` returns `InvalidDescription` for no stride, a negative `per_head`, a `per_head`
whose product with the head count does not fit in 64 bits, and whatever a step refuses as a bad
option or stride. It returns `Incompatible` for no box and for fewer distinct usable boxes than a
count that was tried. The steps run in the order above and the first failure is returned, so
boxes too few for a count are reported before a stride of zero.

The same boxes in the same order with the same strides and options give the same plan, bit for
bit on one build: the only random source is the seeded sequence of `cluster_anchors`.

`test_detect_anchor_plan.cpp` generates the Ladder and the Blob of SPEC 5.6 from seeds written in
the file. Six rungs under strides 16 and 32 give two heads of three anchors, a recall of 1 and
four trials. The test compares a plan with the four steps called one by one to 1e-9, and two
plans of the same arguments with a tolerance of 0.

## anchor/anchor_task.hpp

### AnchorTask, read_anchor_task, write_anchor_task

`AnchorTask` holds what decoding the heads of an anchor-box model needs: the class count, the
`AnchorLayout`, per head of the layout the name of the model output it reads and its
`HeadOptions`, and the suppression criterion with its exponent. It holds nothing about
training. The two functions convert it to and from the `task` object of a model file
(`docs/guides/MODEL_FILE.md`), whose `type` is `K_ANCHOR_TASK_TYPE`, `anchor_detection`:

| Key | Holds | Field |
| --- | --- | --- |
| `type` | `"anchor_detection"` | none |
| `classes` | A whole number of at least 1 | `num_classes`, and `options.num_classes` of every head |
| `anchors` | A list of `[width, height]`, fractions of the input, finite and above zero | `layout.anchors` |
| `heads` | A list of at least one head object, in the layout's head order | `heads`, `layout.head_anchors` |
| `heads[k].output` | The published name of the output head `k` reads | `heads[k].output` |
| `heads[k].anchors` | Indices into `anchors`, at least one; entry `a` is slot `a` | `layout.head_anchors[k]` |
| `heads[k].coding` | `"logit"` or `"activated"` | `heads[k].options.coding` |
| `heads[k].scale_xy` | A number, finite and above zero | `heads[k].options.scale_xy` |
| `suppression.criterion` | `"iou"` or `"distance_iou"` | `criterion` |
| `suppression.distance_exponent` | A finite number, under `distance_iou` only | `distance_exponent` |

Every key is required, except that `distance_exponent` is required under `distance_iou` and
refused under `iou`, which does not read it. A key outside the table is refused. Two heads may
share an anchor and an anchor may belong to no head, as `validate_anchor_layout` allows. Each
enumeration has one table of enumerator and string in `anchor_task.cpp`, which both functions
look up.

Both functions run the same check of the task's values and then `validate_head` on every head.
Each refusal is `InvalidDescription` and names the key and the entry, as in
`task 'heads' entry 1 'anchors' entry 2 names anchor 9; the task has 6 anchors`. The writer
also refuses a head whose class count is not the task's, a head count that is not the layout's,
and an enumerator outside its table.

A float is written as the double of equal value, and the reader narrows a number to float32,
so `read_anchor_task(write_anchor_task(task))` returns every float bit for bit. One field is
outside that: under `IoU` the exponent is not written, and it reads back as
`K_DEFAULT_DISTANCE_EXPONENT`. `write_anchor_task(read_anchor_task(section))` returns the
section it read when the section's numbers are reals that float32 holds exactly and its keys
are in the order of the table; a whole number where the table says number is read and written
back as a real. A number float32 cannot hold as a finite value is refused, and one that narrows
to zero fails the check for a value above zero.

`test_detect_anchor_task.cpp` round-trips three heads over nine anchors and two heads over six
in both directions with a tolerance of zero, reads and writes each string of both
enumerations, and has a case for each refusal.

## anchor/head_training_section.hpp

### read_head_training_section, write_head_training_section

The two functions convert a `HeadTrainingOptions` to and from the `objective` object of a
training file, whose `type` is `K_ANCHOR_OBJECTIVE_TYPE`, `anchor`
(`docs/design/MODEL_IMPORT.md`). The section holds the training-only values of an anchor model;
what decoding needs is in the task section above. Each key is the name of the field it states,
and the two thresholds are stated beside the others, not under a `thresholds` object:

| Key | Holds | Field |
| --- | --- | --- |
| `type` | `"anchor"` | none |
| `match_threshold` | A number in (0, 1] | `thresholds.match_threshold` |
| `ignore_threshold` | A number in [0, 1] | `thresholds.ignore_threshold` |
| `box_loss` | `"squared_error"` or `"complete_iou"` | `box_loss` |
| `box_gradient` | `"analytic"` or `"darknet_compatible"` | `box_gradient` |
| `objectness_target` | `"one"` or `"best_iou"` | `objectness_target` |
| `box_weight` | A finite number, 0 or more | `box_weight` |
| `object_weight` | A finite number, 0 or more | `object_weight` |
| `max_delta` | A finite number, 0 or more; 0 sets no bound | `max_delta` |
| `class_gate` | A number in [0, 1] | `class_gate` |

```json
"objective": {
    "type": "anchor",
    "match_threshold": 0.213,
    "ignore_threshold": 0.7,
    "box_loss": "complete_iou",
    "box_gradient": "darknet_compatible",
    "objectness_target": "best_iou",
    "box_weight": 0.07,
    "object_weight": 1.0,
    "max_delta": 5.0,
    "class_gate": 0.25
}
```

The reader gives a key left out the default of its field, so `{ "type": "anchor" }` reads as
`HeadTrainingOptions{}`. It accepts the object with `type` or with `type` taken out, as
`Config::TrainSpec` hands a typed block to its factory; a `type` that is stated must be
`"anchor"`. A key outside the table is refused with the keys the section accepts. The writer
states `type` and every key, in the order of the table, `box_gradient` included under a box
loss that does not read it.

One section holds one `HeadTrainingOptions`, which every head of the model then trains under.
Options that differ from head to head have no form in the section.

Both functions run `validate_head_training`. Every refusal is `InvalidDescription` and starts
with `objective`; a wrong kind, a string no enumerator has and a number float32 cannot hold as
a finite value name the key, as in
`objective 'box_loss' is 'ciou'; it is one of 'squared_error', 'complete_iou'`. A refusal of
`validate_head_training` keeps that function's message behind the prefix `objective: `, which
names the field in words (`the match threshold`), not the key.

A float is written as the double of equal value and the reader narrows a number to float32, so
`read(write(options))` returns every float bit for bit. `write(read(section))` returns the
section it read when the section states every key in the order of the table and its numbers
are reals float32 holds exactly. One table in `head_training_section.cpp` holds each key with
its reader and its writer, and each enumeration has one table of enumerator and string.

`test_detect_anchor_training_section.cpp` round-trips the defaults and a set of options with no
default left, with a tolerance of zero, reads and writes each string of the three
enumerations, and has a case for each refusal.

## Reference tensors the assigner tests read

`tools/reference/export_yolo26_reference.py` hooks both assigners and writes the first three
arguments of each call into `loss.sacp`, beside the five outputs:

| Tensor | Shape | Holds |
| --- | --- | --- |
| `one2many.assignment.input.scores` | `[1, 8400, 80]` | first argument: `pred_scores.detach().sigmoid()` |
| `one2many.assignment.input.boxes` | `[1, 8400, 4]` | second argument: `pred_bboxes.detach() * stride_tensor` |
| `one2one.assignment.input.scores` | `[1, 8400, 80]` | the same, one-to-one branch |
| `one2one.assignment.input.boxes` | `[1, 8400, 4]` | the same, one-to-one branch |
| `anchors.points` | `[8400, 2]` | third argument, the same for both branches |

`Unit_DetectAssigner.ReproducesTheReferenceOneToManyAssignment` and
`ReproducesTheReferenceOneToOneAssignment` run the assigner on them and compare all five outputs.
The padded ground truth is rebuilt from `batch.batch_idx`, `batch.cls` and `batch.bboxes` through
`pad_ground_truth`, and `Unit_DetectGroundTruth.PadsTheReferenceLabelsToItsTargetBoxes` checks
that rebuild against the target boxes and labels the reference wrote.

The fixture's fourth ground-truth box is 9.6 by 8 px, below `stride_val`. Enlarged, it contains
four anchors, all with a positive metric; unenlarged it contains none. The one-to-many
branch assigns it 4 anchors and the one-to-one branch 1, and both tests pin those counts. No two
positive metrics of the fixture are equal, and no box depends on a zero-metric candidate for its
`topk`, so the tie rule of `select_top_k` does not decide any row of this comparison.

## Reference rows the suppression test reads

The largest class score of the fixture is `1.9e-4`, below the validation confidence, so the
reference validator keeps no row of either head. Two tests pin that count, 0, one per path.

For a comparison with rows in it the exporter runs `non_max_suppression` on the one-to-many decode
a second time, with the validator's arguments and the confidence lowered to `1.8e-4`, and writes
the result as `one2many.suppressed.detections`, `[300, 6]`, in `forward.sacp`. `suppression` in
`MANIFEST.json` records the arguments and the counts: 1542 anchor and class pairs pass, and 300
rows are kept. `Unit_DetectPostprocess.SuppressionReproducesTheReferenceRows` runs
`select_nms_detections` at that confidence and compares.

Twelve kept rows share the last score with nine candidates the limit cut, and the reference's
`argsort` does not say which of the 21 come first. The test compares the score column in order and
the 288 rows above the last score as a set.

The comparison covers the candidate rows, the box conversion, the order and the 300-row limit. It
does not cover a removal or the class shift. No two candidates overlap above 0.7 (the largest IoU
is 0.59), so the reference removes nothing, and the 300 kept rows are all of class 56. The hand
tests cover both. Lowering the IoU threshold would make the reference remove rows, but the 1542
scores hold 282 distinct values and boxes of equal score overlap, so the surviving rows would
depend on the order `argsort` gives equal scores.

## Fixture location

`SushiAI/reference_files/reference_file.hpp` names a file by its path below
`SA_TEST_FIXTURE_DIR`, the `fixtures/` folder of the test program, here
`tests/vision/detect/fixtures/`. Without the file, a fixture test is reported as skipped with the
missing name. With the file present, every
tensor is read with `SA_READ_REFERENCE_OR_FAIL`, so a missing or renamed tensor fails the test.
