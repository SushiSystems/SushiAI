# Model import

**Status:** Open — phase 4 of 9

## Problem

How a model of another framework (Darknet today; PyTorch weight files, ONNX and Caffe by the
format engine below) enters SushiAI once, through one door, and is trained and run afterwards
from SushiAI's own files with no file of the other framework in sight.

## Decision

Made by the owner on 2026-10-11.

| Subject | Ruling |
| --- | --- |
| The door | Every foreign format enters through the same import step. `train` and `inference` read SushiAI's files only |
| The files | A model is a model file in JSON and a weight file, as the hand-written models are. No single-file container |
| The task section | Inside the model file |
| The extensions | The model file is `.json`. The weight file is `.sushi`, the official SushiAI weight file; it replaces `.ckpt` and `weights.sacp`. The bytes are the checkpoint format unchanged |
| Darknet's `.cfg` | An input of the import step. Training a Darknet-era model does not need it afterwards |

## Design

### The files a model is made of

| File | Holds | Written by |
| --- | --- | --- |
| Model file (`model.json` today) | The input shape, the layer rows, the published outputs, and the task section below | A person, or the import step |
| `<name>.sushi` | Parameters and buffers by name | The import step, `train` |
| Training file (`train.json` today) | Objective, data, optimizer, schedule, epochs | A person; the import step writes a starting one |

The three are separate because they change at different times: the architecture when the model
is designed, the weights at every checkpoint, the training recipe at every run.

### The import step

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

### The task section

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

### What phase 3 decided and still lacks

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

### What phase 4 decided

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

### The format engine

Ruled by the owner on 2026-10-12: SushiAI reads every foreign format with readers of its own
and takes no external dependency for it, Caffe is among the formats, and the design is written
before any of it is built. Nothing in this section exists yet.

#### Two kinds of format

| Kind | Formats | The file holds | The import needs |
| --- | --- | --- | --- |
| Weight file | `.pth`, `.pth.tar`, `.safetensors`, `.npz` | Tensors by name | A model file a person wrote, and a name map |
| Graph file | ONNX, Caffe, Darknet | The layers and the weights | A translation of each foreign operator into rows |

A weight file cannot produce a model file, so the two kinds enter by two commands:

```
sushiai import <format> <files...> --output <directory>
sushiai import weights <file> --model <model.json> --map <map.json> --output <file.sushi>
```

The first is the command above. The second detects the container from the file's first
bytes, reads the tensors, renames them by the map and writes the `.sushi` file. It builds the
model file's state records first and refuses the import when a record has no tensor, a tensor
has no record, or a shape differs, naming every such entry in one message.

#### The readers

Each reader turns bytes into a structure and knows no model. They live in `io`, one unit each.

| Reader | Reads | Used by |
| --- | --- | --- |
| `ZipArchive` | The entries of a zip file, stored and deflated | `.pth`, `.npz` |
| `PickleReader` | A pickle stream of protocol 2 to 5, into a tree of values | `.pth` |
| `NpyReader` | One `.npy` array: its header and its raw data | `.npz` |
| `SafetensorsReader` | The length, the JSON header and the raw data | `.safetensors` |
| `ProtobufWire` | Varints, fixed fields and length-delimited fields, one field at a time | ONNX, Caffe weights |
| `ProtobufText` | The protobuf text format, into a tree of fields | Caffe `.prototxt` |

- **`PickleReader` runs no code.** A pickle names callables; the reader holds a table of the
  ones it accepts (`collections.OrderedDict`, PyTorch's tensor, storage and parameter
  rebuilders) and builds a value for each itself. Any other name is refused with that name.
- **`ProtobufWire` has no schema compiler.** ONNX and Caffe each state the field numbers of the
  messages they read in a table of their own: `ModelProto`, `GraphProto`, `NodeProto`,
  `TensorProto`, `AttributeProto` and `ValueInfoProto` for ONNX; `NetParameter`,
  `LayerParameter`, `V1LayerParameter` and `BlobProto` for Caffe. An unknown field is skipped
  by its wire type.
- **A tensor is read where it lies.** The readers return an offset and a length into the file
  for each tensor, and the bytes are copied once, into the `.sushi` writer.

#### The name map

A JSON file beside the model file. It holds prefix rules in the shape of `RenameRule`, a list
of names to drop (`num_batches_tracked` of every PyTorch normalisation), and the key of the
state dictionary inside the file (`model` for the ByteTrack checkpoints). The layout of a
weight follows from the record it lands in: a convolution weight stored `OIHW` is permuted to
the layout the record states, and the map says nothing about it.

#### The graph importers

ONNX and Caffe each get an importer in the shape of the Darknet one, and each holds an
operator table: one row per foreign operator, naming the function that writes its rows of the
model file. An operator without a row is refused; the message lists every such operator of
the file with the nodes that use it, so one run shows the whole gap.

Shared by the two, written once:

- **Axis translation.** Both formats are `NCHW` and SushiAI's activations are `NHWC`. One unit
  rewrites an axis attribute (`Concat`, `Slice`, `Softmax`, `Transpose`, `Reshape`) and
  permutes a weight.
- **Constant folding of shape arithmetic.** An ONNX export computes shapes with `Shape`,
  `Gather`, `Unsqueeze` and `Concat` nodes that feed a `Reshape` or a `Resize`. One unit
  evaluates the nodes whose inputs are all constants and removes them before translation.
- **The task section.** Neither format states one. `--task <task.json>` gives it; without it
  the model has no task and `inference` gives the class report.

Caffe reads the layers from the `.prototxt` and the weights from the `.caffemodel`, joined by
layer name. Its `BatchNorm` and `Scale` pair becomes one normalisation row.

#### The first model: YOLOX-X

The first use of the weight command is YOLOX-X with a ByteTrack checkpoint of SushiTrack
(`mot17x.pth.tar`, a zip container of 793 MB).

| Part | State |
| --- | --- |
| Convolution with normalisation and SiLU, bottleneck, upsample, concat | `conv`, `bottleneck`, `upsample`, `concat` |
| CSP layer | `c3k` has the same parts; whether its options reach the 1x1 and 3x3 bottleneck is not checked |
| SPP with 5, 9 and 13 pools | `sppf` with three 5x5 pools gives the same values; not checked against a run |
| Focus stem | Missing: a row that stacks the four pixel phases into channels |
| Decoupled head | Missing: a stem and separate class, box and objectness branches per level |
| Box decoding | Missing: a third task type; the offset is added to the cell and the size is an exponential, both times the stride |

The model file is written by hand from the paper and from the shapes of the checkpoint's
tensors. A PyTorch run outside the repository dumps the three head outputs for one input, and
a parity suite compares them, as the Darknet suites do. No YOLOX source enters the repository.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| 1 | `ModelSpec` to JSON, and the round trip `parse(write(spec)) == spec`. Needs a build: To run its tests | built |
| 2 | The task section: its reader and writer in `vision/detect`'s terms, the registry row, and `inference` decoding a model from it. Needs a build: To run its tests | built |
| 3 | The `anchor` objective and the step schedule in the training file; `train` building `AnchorTrainingStep` from the files. Needs a build: To run its tests | built |
| 4 | `sushiai import darknet`, and the six parity suites reading the imported files instead of the `.cfg`. Needs a build: Yes | building |
| 5 | `ZipArchive`, `PickleReader`, `SafetensorsReader`, the name map, `sushiai import weights`. Proven by: A `.pth` and a `.safetensors` fixture of a small model give the `.sushi` file `train` wrote for it | planned |
| 6 | YOLOX-X: the focus row, the decoupled head, the task type, the model file and its map. Proven by: The three head outputs against the dumped ones; detections on a MOT17 frame | planned |
| 7 | `ProtobufWire`, axis translation, constant folding, the ONNX importer. Proven by: The ONNX export of YOLOX-X imports to the outputs of phase 6 | planned |
| 8 | `ProtobufText`, the Caffe importer. Proven by: A Caffe model's outputs against a dump of a Caffe run | planned |
| 9 | `NpyReader` and `.npz`; Ultralytics `.pt`, which pickles module objects and widens the accepted names. Proven by: The YOLO26 fixture read from its `.pt` | planned |

The code of every phase can be written before SushiBLAS follows SushiRuntime v0.4.0; none can be
verified before it.

## Open questions

Each is the owner's decision.

- **When phases 5 to 9 run** against tag 0.1.0. Moving SushiTrack's inference from PyTorch to
  SushiAI is after 0.1.0; phases 5 and 6 are what the YOLOX-X run needs.
- **The first Caffe model** of phase 8.
- **One image as the input of `inference`.** It reads the split of `eval.json` today; the
  YOLOX-X run on a MOT17 frame needs either a split of one image or a new argument.
- **TensorFlow Lite**, which needs a FlatBuffers reader, and the order of formats after
  phase 9.
