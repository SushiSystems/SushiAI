# modules/vision/detect/include/SushiAI/vision/detect/anchor/head_training_section.hpp


## read_head_training_section, write_head_training_section

The two functions convert a `HeadTrainingOptions` to and from the `objective` object of a
training file, whose `type` is `K_ANCHOR_OBJECTIVE_TYPE`, `anchor`
(`docs/design/MODEL_IMPORT/README.md`). The section holds the training-only values of an anchor model;
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
