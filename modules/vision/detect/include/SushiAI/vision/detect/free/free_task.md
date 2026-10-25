# free_task.hpp

`FreeTask` holds what an anchor-free model's graph cannot state. `read_free_task` and
`write_free_task` convert it to and from the `task` section of a model file whose `type` is
`"anchor_free_detection"` (`K_FREE_TASK_TYPE`). Strides, ports, `reg_max` and the class count
are derived from the traced model by `Vision::Training::derive_free_model_facts`, never stated
here.

| Key | Value | Field |
| --- | --- | --- |
| `score_threshold` | A number from 0 to 1 | `score_threshold` |
| `suppression.criterion` | `"iou"` or `"distance_iou"` | `criterion` |
| `suppression.threshold` | A number from 0 to 1 | `suppression_threshold` |
| `suppression.distance_exponent` | A finite number, under `distance_iou` only | `distance_exponent` |
| `assignment.one_to_many` | `{top_k, top_k2, alpha, beta}` | `one_to_many` |
| `assignment.one_to_one` | `{top_k, top_k2, alpha, beta}` | `one_to_one` |

`type`, `score_threshold`, `suppression.criterion` and `suppression.threshold` are required.
`assignment` and each of its branches are optional; an absent branch keeps the preset of
`assigner_config.hpp`: top 10 of 10 for one-to-many, top 7 then 1 for one-to-one, alpha 0.5,
beta 6. A stated branch carries all four keys. `top_k` is at least 1, `top_k2` is from 1 to
`top_k`, `alpha` and `beta` are finite and not below zero. A key outside the table is refused.
Every refusal is `InvalidDescription` and names the key.

The writer writes both branches, and the exponent only under `distance_iou`. Floats are
written as the double of equal value, so `read_free_task(write_free_task(task))` returns every
field bit for bit.
