# modules/vision/detect/include/SushiAI/vision/detect/iou.hpp


## intersection_over_union, complete_iou

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

## pairwise_iou

`box_iou`, which differs from `bbox_iou`: the heights carry no eps and only the divisor does.

```
U/utils/metrics.py:101  wh, wh1, wh2 = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0), a2 - a1, b2 - b1
U/utils/metrics.py:105  return inter / (wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter + eps)
```

No unit of this module calls it yet. The suppression uses its own formula (see
`detections_nms.hpp`).
