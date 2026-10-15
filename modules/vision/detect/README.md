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

[`include/SushiAI/vision/detect/free/anchors.md`](include/SushiAI/vision/detect/free/anchors.md) describes it.

## free/level_rows.hpp

[`include/SushiAI/vision/detect/free/level_rows.md`](include/SushiAI/vision/detect/free/level_rows.md) describes it.

## free/anchor_leaves.hpp

[`include/SushiAI/vision/detect/free/anchor_leaves.md`](include/SushiAI/vision/detect/free/anchor_leaves.md) describes it.

## box_format.hpp

[`include/SushiAI/vision/detect/box_format.md`](include/SushiAI/vision/detect/box_format.md) describes it.

## free/box_coding.hpp

[`include/SushiAI/vision/detect/free/box_coding.md`](include/SushiAI/vision/detect/free/box_coding.md) describes it.

## iou.hpp

[`include/SushiAI/vision/detect/iou.md`](include/SushiAI/vision/detect/iou.md) describes it.

## iou_gradient.hpp

[`include/SushiAI/vision/detect/iou_gradient.md`](include/SushiAI/vision/detect/iou_gradient.md) describes it.

## top_k.hpp

[`include/SushiAI/vision/detect/top_k.md`](include/SushiAI/vision/detect/top_k.md) describes it.

## ground_truth.hpp

[`include/SushiAI/vision/detect/ground_truth.md`](include/SushiAI/vision/detect/ground_truth.md) describes it.

## free/assigner_config.hpp

[`include/SushiAI/vision/detect/free/assigner_config.md`](include/SushiAI/vision/detect/free/assigner_config.md) describes it.

## free/assignment.hpp

[`include/SushiAI/vision/detect/free/assignment.md`](include/SushiAI/vision/detect/free/assignment.md) describes it.

## free/assigner.hpp

[`include/SushiAI/vision/detect/free/assigner.md`](include/SushiAI/vision/detect/free/assigner.md) describes it.

## free/target_leaves.hpp

[`include/SushiAI/vision/detect/free/target_leaves.md`](include/SushiAI/vision/detect/free/target_leaves.md) describes it.

## detection_row.hpp

[`include/SushiAI/vision/detect/detection_row.md`](include/SushiAI/vision/detect/detection_row.md) describes it.

## prediction_shape.hpp

[`include/SushiAI/vision/detect/prediction_shape.md`](include/SushiAI/vision/detect/prediction_shape.md) describes it.

## detections_topk.hpp

[`include/SushiAI/vision/detect/detections_topk.md`](include/SushiAI/vision/detect/detections_topk.md) describes it.

## suppression_criterion.hpp

[`include/SushiAI/vision/detect/suppression_criterion.md`](include/SushiAI/vision/detect/suppression_criterion.md) describes it.

## detections_nms.hpp

[`include/SushiAI/vision/detect/detections_nms.md`](include/SushiAI/vision/detect/detections_nms.md) describes it.

## average_precision.hpp

[`include/SushiAI/vision/detect/average_precision.md`](include/SushiAI/vision/detect/average_precision.md) describes it.

## The anchor part

`anchor/` holds detection with anchor boxes, the family of YOLOv3, v4 and v7. It is written from
the behaviour `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md` states in formulas (sections 4.4 and
5.5) and ports no code. A content failure comes back as a `Status` of `core`; a call that breaks
a stated precondition stops the process through `SA_FATAL_IF`. Nothing in this part throws.

An anchor is stored as a fraction of the network input, so one layout serves every input size
and every grid. Pixels do not appear in this part.

## anchor/anchor_size.hpp

[`include/SushiAI/vision/detect/anchor/anchor_size.md`](include/SushiAI/vision/detect/anchor/anchor_size.md) describes it.

## anchor/anchor_layout.hpp

[`include/SushiAI/vision/detect/anchor/anchor_layout.md`](include/SushiAI/vision/detect/anchor/anchor_layout.md) describes it.

## anchor/head_options.hpp

[`include/SushiAI/vision/detect/anchor/head_options.md`](include/SushiAI/vision/detect/anchor/head_options.md) describes it.

## anchor/head_decode.hpp

[`include/SushiAI/vision/detect/anchor/head_decode.md`](include/SushiAI/vision/detect/anchor/head_decode.md) describes it.

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

[`include/SushiAI/vision/detect/anchor/anchor_plan.md`](include/SushiAI/vision/detect/anchor/anchor_plan.md) describes it.

## anchor/anchor_task.hpp

[`include/SushiAI/vision/detect/anchor/anchor_task.md`](include/SushiAI/vision/detect/anchor/anchor_task.md) describes it.

## anchor/head_training_section.hpp

[`include/SushiAI/vision/detect/anchor/head_training_section.md`](include/SushiAI/vision/detect/anchor/head_training_section.md) describes it.

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
