# yolox_task.hpp

`YoloxTask` holds what a YOLOX model's graph cannot state. `read_yolox_task` and
`write_yolox_task` convert it to and from the `task` section of a model file whose `type` is
`"yolox_detection"` (`K_YOLOX_TASK_TYPE`). The strides, the class count and the ports come from
the built model and its input size, never from here.

| Key | Value | Field |
| --- | --- | --- |
| `score_threshold` | A number from 0 to 1 | `score_threshold` |
| `suppression.criterion` | `"iou"` or `"distance_iou"` | `suppression.criterion` |
| `suppression.threshold` | A number from 0 to 1 | `suppression.threshold` |
| `suppression.distance_exponent` | A finite number not below zero, under `distance_iou` only | `suppression.distance_exponent` |

All keys but `suppression.distance_exponent` are required. A key outside the table is refused.
Every refusal is `InvalidDescription` and names the key. The `suppression` object is read by
[`suppression_section.hpp`](suppression_section.md).

The writer writes the exponent only under `distance_iou`. Floats are written as the double of
equal value, so `read_yolox_task(write_yolox_task(task))` returns every field bit for bit.
