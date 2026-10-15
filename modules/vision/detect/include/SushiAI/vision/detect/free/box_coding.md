# modules/vision/detect/include/SushiAI/vision/detect/free/box_coding.hpp


## decode_xyxy, decode_xywh

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

## scale_boxes

Decoded boxes are in grid units until they are multiplied by the stride:
`self.decode_bboxes(...) * self.strides` (`U/nn/modules/head.py:226`) and
`pred_bboxes.detach() * stride_tensor` (`U/utils/loss.py:441`).

## encode_distances

```
U/utils/tal.py:490  x1y1, x2y2 = bbox.chunk(2, -1)
U/utils/tal.py:491  dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
```

The clamp at `U/utils/tal.py:493` is not ported: the L1 branch calls `bbox2dist` without `reg_max`
(`U/utils/loss.py:143`).
