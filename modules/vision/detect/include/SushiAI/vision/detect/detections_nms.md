# modules/vision/detect/include/SushiAI/vision/detect/detections_nms.hpp


## LabelMode, ClassMode

`LabelMode` is the reference's `multi_label` flag: `MultiLabel` lets an anchor contribute every
class above the threshold, `BestClass` only its highest (`U/utils/nms.py:117-123`). `ClassMode`
is its `agnostic` flag: `ClassAware` suppresses within a class, `ClassAgnostic` across classes
(`U/utils/nms.py:146`). Each replaces a boolean parameter.

## NMSConfig

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

## select_nms_detections

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

## suppress_overlaps

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
