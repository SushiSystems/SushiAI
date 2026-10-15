# modules/vision/detect/include/SushiAI/vision/detect/free/assigner.hpp


## AssignerInputs

The arguments of one `TaskAlignedAssigner.forward` call (`U/utils/tal.py:64`) with their
extents written out: `pd_scores` as `[B, A, C]`, `pd_bboxes` as `[B, A, 4]`, `anc_points` as
`[A, 2]`, `gt_labels` as `[B, M]`, `gt_bboxes` as `[B, M, 4]` and `mask_gt` as `[B, M]`. The
spans are views; the caller keeps the arrays alive for the call.

## TaskAlignedAssigner

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
