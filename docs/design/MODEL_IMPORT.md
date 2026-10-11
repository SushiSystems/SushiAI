# Model import

**Status:** Open — phase 1, all four phases written, not yet built

How a model of another framework (Darknet today; ONNX, Caffe and Ultralytics later) enters
SushiAI once, through one door, and is trained and run afterwards from SushiAI's own files with
no file of the other framework in sight.

## Rulings

Made by the owner on 2026-10-11.

| Subject | Ruling |
| --- | --- |
| The door | Every foreign format enters through the same import step. `train` and `inference` read SushiAI's files only |
| The files | A model is a model file in JSON and a weight file, as the hand-written models are. No single-file container |
| The task section | Inside the model file |
| The extensions | The model file is `.json`. The weight file is `.sushi`, the official SushiAI weight file; it replaces `.ckpt` and `weights.sacp`. The bytes are the checkpoint format unchanged |
| Darknet's `.cfg` | An input of the import step. Training a Darknet-era model does not need it afterwards |

## The files a model is made of

| File | Holds | Written by |
| --- | --- | --- |
| Model file (`model.json` today) | The input shape, the layer rows, the published outputs, and the task section below | A person, or the import step |
| `<name>.sushi` | Parameters and buffers by name | The import step, `train` |
| Training file (`train.json` today) | Objective, data, optimizer, schedule, epochs | A person; the import step writes a starting one |

The three are separate because they change at different times: the architecture when the model
is designed, the weights at every checkpoint, the training recipe at every run.

## The import step

```
sushiai import <format> <files...> --output <directory>
```

It writes `model.json`, `weights.sushi` and `train.json` into the directory. `<format>` names an
importer; `darknet` takes a `.cfg` and optionally a `.weights`.

An importer is one unit per format in `modules/<domain>/importers/`, and each has the same
shape: it reads the foreign files and returns an `ImportedModel`.

| Part of `ImportedModel` | Type | Exists today |
| --- | --- | --- |
| The rows and the outputs | `Config::ModelSpec` | Yes; `translate_darknet_model` fills it |
| The task section | A `Config::Value` object the domain wrote | Yes; `ModelSpec::task`, `write_anchor_task` |
| The weights, by parameter name | What `store_darknet_weights` writes into a `ModelState` | Yes |
| The training recipe | A `Config::Value` in the shape of the training file | No; `read_darknet_training` returns structs |

What is written once and shared by every importer: the writer of a `ModelSpec` to JSON, the
writer of the task section, the command. A new format adds an importer and one row of the
importer table; it edits no other importer and no writer.

## The task section

A detection model is not usable from its rows alone. Decoding a head needs the anchors, the
coding and the offset stretch; training it needs the thresholds and the weights of the delta.
Today those live only in `DarknetModel` and `DarknetTraining`, which is why the `.cfg` is still
needed.

The model file gains one optional section, `task`, which the layers tier carries as an opaque
value and a domain reads:

```json
"task": {
    "type": "anchor_detection",
    "classes": 80,
    "anchors": [[0.024, 0.031], [0.055, 0.072], "..."],
    "heads": [
        { "output": "head_0", "anchors": [3, 4, 5], "coding": "logit", "scale_xy": 1.05 }
    ],
    "suppression": { "criterion": "iou" }
}
```

- Anchors are fractions of the input, as `AnchorLayout` holds them, so a model file does not
  change when the input size does.
- The training-only values (`box_loss`, `box_gradient`, thresholds, weights) go to the
  objective of the training file, `{ "type": "anchor", ... }`, beside the optimizer they are
  tuned with. A value left out takes the default of `HeadTrainingOptions`.
- `task.type` is looked up in a registry the domain fills, as node and dataset types are, so
  `config` and `nn` name no detection type. YOLO26's anchor-free head is a second row,
  `anchor_free_detection`.

## Phases

| Phase | Work | Needs a build |
| --- | --- | --- |
| 1 | `ModelSpec` to JSON, and the round trip `parse(write(spec)) == spec` | To run its tests |
| 2 | The task section: its reader and writer in `vision/detect`'s terms, the registry row, and `inference` decoding a model from it | To run its tests |
| 3 | The `anchor` objective and the step schedule in the training file; `train` building `AnchorTrainingStep` from the files | To run its tests |
| 4 | `sushiai import darknet`, and the six parity suites reading the imported files instead of the `.cfg` | Yes |

The code of every phase can be written before SushiBLAS follows SushiRuntime v0.4.0; none can be
verified before it.

## What phase 3 decided and still lacks

Decided, written and not yet built:

- **A file with a `schedule` states its rate there only.** `optimizer.learning_rate` beside a
  schedule is refused, even when the two agree, and so is a schedule without a `base_rate`
  above zero. No file trains at a rate it does not state.
- **The rate is set before each update** through `Optimizer::set_learning_rate`; the iteration
  is the optimizer's own step count.
- **Head shapes come from the probe trace**, which needs no engine, so an objective is still
  made without one.
- **Every objective factory takes an `ObjectiveContext`**: the batch, the class count, the
  input shape, the model's task section and the published output shapes. The objective table
  is a registry a domain adds rows to; `vision/registration` adds `anchor`.
- **Detection has its own loop**, `run_anchor_training`. `Trainer`'s step is classification
  down to its feed, and an anchor update runs two passes.

Still lacking:

- **The run has no test.** `sushiai train` chooses its run from a table by objective type
  (`cross_entropy`, `anchor`), and the programs have no test target of their own.
- **A training file for a detector cannot run yet**: its model file and weights come from
  phase 4.
- **Subdivisions and `max_batches`** have no key; accumulation needs a replay without the
  update and an accumulate kernel.
- **One `HeadTrainingOptions` for every head**; a `heads` array in the section would lift it.
- **A resumed run does not replay its batch order.** Version 2 of the weight file carries a
  run-state section, the optimizer's step count and its own tensors, and `resume` restores
  it; the epoch index is not in it, so a shuffling dataset starts its order over.

## What phase 4 decided

- **`ImportedModel` holds weights as records and bytes**, the pair the checkpoint writer takes,
  so an importer and the import program need no engine.
- **`train.json` is written with `"data": null`.** `sushiai train` refuses it by name until a
  person fills the block; no dataset is invented.
- **A value with no key is a note the command prints**, with its line: `batch`,
  `subdivisions`, `max_batches`, `jitter`, `random` and the augmentation keys.
- **Heads whose training options differ are refused**, naming the two heads and the key.
- **`sushiai inference` reports by the model's task type**; a model without a task takes the
  class report. Its score and suppression thresholds are constants of the program, since
  `eval.json` has no key for them, and boxes are printed in pixels of the network input.
- **One image file cannot be given to `inference`**; it reads the split of `eval.json`.

## Open

Each is the owner's decision.

- **Where the importer interface lives.** `ImportedModel` and the importer table name no
  domain, so they fit the layers tier (`io`); each importer stays in its domain.
- **Formats after Darknet, and their order.** ONNX needs a protobuf reader, which is an
  external dependency.
