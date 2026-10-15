# modules/vision/detect/include/SushiAI/vision/detect/anchor/anchor_task.hpp


## AnchorTask, read_anchor_task, write_anchor_task

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
