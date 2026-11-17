# decision_model: an encoder that answers typed questions, from JSON

Two JSON files describe the decision model of `docs/design/LANGUAGE/README.md`: one transformer
encoder over the text of a question, and three heads over its state. The choice head scores each
option of a list against the question, the yes-or-no head gives one logit, and the number head gives
one value. The model answers in one forward pass and has no generation loop.

The files hold the small extents the tests use. No run has been made with them, and the values are
untuned first guesses.

## The files

| File | Describes |
| --- | --- |
| `model.json` | Four extra inputs, one `decision_heads` row at the test extents that reads them, and the task `decision` |
| `train.json` | The sum of the three objectives, the `decisions` data block, the optimizer and the epochs |

`model.json` reads a question of 8 int32 ids as its input. The `inputs` object names the four other
values the row reads, each fed from the batch role of the same name: `lengths`, the ids of each
question, `options`, the ids of up to four options, `option_lengths` and `option_valid`, the ids and the
presence of each option. The row has a vocabulary of 32 ids, a width of 16, two heads, one block, a
feed-forward width of 32 and `options` of 4: a choice holds up to four options. The row is one node of
five inputs, which its `from` lists in order, and three ports, `choice` of `[4]`, `yes_no` of `[1]` and
`number` of `[1]`, which `modules/language/blocks/README.md` describes.

`train.json` trains the three ports at once. Each term reads one output by its `output` key, reads its
targets from the batch role its `target` key names, and keeps the rows of one column of the role
`type_mask` by its `type_mask` key: column 0 for a choice, 1 for a yes-or-no question and 2 for a number
question. A row of another type does not reach the loss of a term. The weights are 1.

## The full extents

The ruling LANGUAGE 6 of `docs/agent/2026_11_15_DESIGN_COMPLETION/PLAN.md` sets the model at

```json
{ "type": "decision_heads", "vocabulary": 4096, "dim": 256, "heads": 4, "blocks": 4,
  "hidden": 1024, "options": 8 }
```

with an input of 128 ids, the extra inputs `options` of `[8, 128]` and `option_lengths` and `option_valid` of `[8]`, and a `window` of 128 and `options` of 8 in the data block. The `vocabulary` is the size of the
vocabulary file the questions are encoded with, and `options` is the most options a question has.
The number of parameters at these extents is not counted.

## Calibration

After training, `calibrate_decision_model` fits one temperature for each classifying head on a split
the model was not trained on and writes both into the task of the model file:

```json
"task": { "type": "decision", "calibration": { "choice": 1.31, "yes_no": 0.87 } }
```

The numbers above only show the shape of the keys. `modules/language/decision_models/README.md`,
section "Calibration", states them.

## What runs these files

`sushiai train` trains the model from these two files once the rows `decision_heads` and `decisions`
are in `language/registration`: the model file's `inputs` are declared and fed by the trainer, and the
objective terms name their roles. The test `tests/language/decision_models/test_decision_model.cpp`
trains the model of `model.json` under the objective block of `train.json` through `DecisionSession`,
over decision files that `write_decision_set` draws, and calibrates it; `DecisionSession` makes the
calls `sushiai train` makes and reads no other description of the model.

No command writes a decision file. The generator of `modules/language/decision_models` is for tests.
