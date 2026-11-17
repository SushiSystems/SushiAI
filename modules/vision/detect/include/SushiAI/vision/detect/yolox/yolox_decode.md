# yolox_decode.hpp

`decode_yolox` turns the three outputs of each head level into the arrays `select_nms_detections`
reads. A `YoloxLevel` views one level's `[N, H, W, C]` class logits, `[N, H, W, 4]` box values and
`[N, H, W, 1]` objectness logits, with the grid and the stride of the level.

For the cell at column `gx` and row `gy` of a level of stride `s`, with the raw box `(px, py,
pw, ph)`:

| Value | Formula |
| --- | --- |
| Centre x | `(gx + px) * s` |
| Centre y | `(gy + py) * s` |
| Width | `exp(pw) * s` |
| Height | `exp(ph) * s` |
| Score of class `c` | `sigmoid(objectness) * sigmoid(class logit c)` |

Anchors run level by level, each level row by row, so `[B, A, 4]` boxes and `[B, A, C]` scores
come out in the layout `select_nms_detections` takes, with `A` the cell count over all levels.

`make_yolox_nms_config` maps a `YoloxTask` to an `NMSConfig`: the best class of each anchor, kept
class by class, with the task's score threshold, suppression threshold, criterion and exponent.
`select_yolox_detections` decodes and then calls `select_nms_detections`, so the suppression is
the one every detection path of the module shares. A score must exceed the threshold; the
reference keeps a score equal to it.

`decode_yolox` refuses no level, a batch or class count below one, a grid or stride that is not
positive, and a span whose size does not fit its level. Each refusal names the level.
