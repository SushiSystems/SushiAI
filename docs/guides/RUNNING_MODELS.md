# Running models

The `sushiai` commands that train, run, import and tokenize from files: `train`, `inference`, `import` and `tokenize`. The build and test commands are in the [CLI guide](CLI_GUIDE.md).

## `sushiai train` and `sushiai inference`: config-driven runs

```bash
sushiai train      examples/heavy_mlp/train.json
sushiai inference  examples/heavy_mlp/model.json \
                   applications/examples/heavy_mlp/heavy_mlp.sushi \
                   examples/heavy_mlp/eval.json
```

`sa` and `sushiai` are the same program (`cli/pyproject.toml` binds both), so
these work under either name; they are spelled `sushiai` here because that is
how the design document names them.

[`examples/mnist_cnn/`](../../applications/examples/mnist_cnn/README.md) holds a second set of the three
files, for a convolutional network on MNIST. Its README says where the IDX files go.

### What the two commands are for

A model's architecture lives in a JSON file rather than in the C++ type
system, and these two drivers consume it. The point is not convenience: it is
that `applications/programs/source/train.cpp` contains **no layer, loss, dataset or optimizer type
name** and therefore never has to be edited as the library grows. Every one of
those arrives through a constant registry table keyed by a string out of the
configuration, so adding a layer type or a detection loss is a table row and
a factory function in the library.

### The three documents

| File | Describes | Named by |
|---|---|---|
| `model.json` | the architecture alone | `train.json`, and `inference`'s first argument |
| `train.json` | objective, data, optimizer, epochs, checkpoint | `sushiai train` |
| `eval.json` | the data to evaluate on | `inference`'s third argument |

```json
{
  "sushiai_model": 1,
  "input": { "features": 6 },
  "layers": [
    { "type": "linear", "units": 256 },
    { "type": "gelu" },
    { "type": "linear", "units": 4 }
  ]
}
```

```json
{
  "sushiai_train": 3,
  "model": "model.json",
  "objective": { "type": "cross_entropy" },
  "data":      { "type": "rings", "batch_size": 64, "batches": 64,
                 "classes": 4, "seed": 3735928559 },
  "optimizer": { "type": "adamw", "learning_rate": 3e-3, "weight_decay": 0.01 },
  "epochs": 50,
  "checkpoint": "heavy_mlp.sushi"
}
```

Four things about the schema are worth knowing before you write one, and each
closes a way of being silently wrong:

- **Each document leads with a version key.** A format change is then an error
  rather than a misread. The training file states version 4, which may carry a
  `cluster` section, the `deterministic` key and the `checkpoint_every` key; a
  file that states version 3, 2 or 1 still reads, a `checkpoint_every` key in
  version 3 or older is an error, a `deterministic` key in version 2 or older is
  an error, and a `cluster` section in a version 1 file is an error.
- **`units` is an output width only.** The input width is chained from
  `input.features`, so a dimension mismatch between adjacent layers is not
  expressible.
- **An unknown key is an error, not an ignored one.** A typed `"unitss": 256`
  must not train silently against a default, so the message lists what the
  layer *does* accept. An unknown `type` likewise lists every type the build
  has.
- **Relative paths resolve against the configuration file's own directory**,
  not your working directory, so `sushiai train examples/heavy_mlp/train.json`
  works from the repository root and a configuration directory can be moved
  without editing what is inside it. This holds for `model`, `weights` and
  `checkpoint`.
  The `data` block is passed to its factory as written, so the `directory` of
  an `mnist` dataset resolves against the working directory of the process.

### Starting weights, a schedule and data from files

Seven optional parts of `train.json`:

| Key | Holds |
| --- | --- |
| `weights` | A `.sushi` file loaded by name before training, in place of the initialiser. A name, shape or element type that differs is an error |
| `resume` | A `.sushi` checkpoint a run of the same model, optimizer and `seed` wrote. It loads the model state as `weights` does and the run state with it: the optimizer's step count and its momentum buffers or moments, and where the run stood. A file states `weights` or `resume`; both is an error |
| `seed` | An integer of 0 or above, 0 when absent. It seeds the random sites of the training graph, which today are the dropout layers. The `seed` of a `data` block is a different key: it seeds the data |
| `schedule` | `{ "type": "step", "base_rate": ..., "warmup_iterations": ..., "warmup_power": ..., "steps": [...], "scales": [...] }`. A file with a schedule states its rate there: `learning_rate` in `optimizer` is then an error |
| `data` with `reader` | A dataset read from files, in place of `type` |
| `deterministic` | Version 3 only. A boolean, `true` when absent. While it is `true`, the same files, seed and device give the same weights after every step, bit for bit, and a key that asks for an unordered path is refused when the file is read, naming the key. `false` opens such paths. One key asks for one: a `cluster` replica with `"if_absent": "local"` runs a lost worker's replica on the master's device, so a file that states it also states `"deterministic": false`. The data loader hands batches over in index order whatever `workers` and `prefetch` say |
| `cluster` | Version 2 and later. The replicas that train the model on several machines. Its fields are in `modules/core/distribution/README.md` |
| `checkpoint_every` | Version 4 only, and only with a `checkpoint` path. `{ "batches": N }` or `{ "epochs": N }`, a positive integer: the run writes `checkpoint` after every N-th update, or after the last batch of every N-th epoch, and not after the last update, where it writes the final checkpoint anyway. Each write replaces the file whole, and a run that stops can go on from it through `resume`. A write stalls the run for the time of the write, 70 MB for YOLOv3-tiny. A file with a `cluster` section refuses the key |

A `data` block with `reader`:

| Key | Default | Meaning |
| --- | --- | --- |
| `reader` | required | `yolo_txt`, `image_folder`, `idx` or `cifar` |
| `task` | required | `detection` or `classification` |
| `root` | required | The dataset's directory, resolved against the directory of the file that states it |
| `split` | `"train"` | As the reader takes it |
| `batch_size` | 16 | |
| `input` | required | `[height, width]` of the network input |
| `channels` | 3 | |
| `fit` | The model file's, else `letterbox` for `detection` and `segmentation` and `resize` for the other tasks | `resize` scales each axis on its own; `letterbox` scales both by one ratio and pads; `darknet_letterbox` is Darknet's `letterbox_image`: the scaled sides and the offsets truncate, the padding is 128 and the image is scaled up as well as down. A `fit` other than the model file's is an error |
| `letterbox_padding`, `letterbox_scale_up` | 114, false | With the `letterbox` fit, the padding byte in [0, 255] and whether an image smaller than `input` is scaled up; an error beside another fit |
| `classes` | The count the samples name | `yolo_txt` requires it |
| `max_objects` | required for `detection` | The object slots of one image; an image with more is an error |
| `cache`, `cache_side` | none, 0 | A directory of raw images written on the first pass and read afterwards; `cache_side` bounds the longest side |
| `shuffle`, `seed` | false, 0 | A sample order drawn anew at each epoch from the seed |
| `pipeline` | `"none"` | `"default"` runs mosaic, the affine warp, flips and colour on each sample. `"darknet"` crops each sample by Darknet's jitter, resizes the crop to `input` and distorts its colour; it reads `jitter`, `hue`, `saturation` and `exposure` |
| `jitter`, `hue`, `saturation`, `exposure` | 0, 0, 1, 1 | Keys of the `"darknet"` pipeline, refused beside any other. `jitter` in [0, 0.5) moves each crop edge by up to that fraction of the side, past the image where the edge pixel repeats; `hue` in [0, 1] shifts the hue by up to that fraction of the circle either way; `saturation` and `exposure`, each at least 1, scale by a factor drawn in [1/s, s] as Darknet's `rand_scale` does. A sample with a mask is an error. `sushiai import darknet` writes the four from the `.cfg`, with Darknet's defaults 0.2, 0, 1 and 1 for what it omits |
| `workers`, `prefetch` | 0, the worker count | Threads that read batches ahead, and how many batches they hold |

`applications/examples/yolov7_tiny/` holds a training file that uses all three.

### One objective over several outputs

A model that publishes several outputs trains through objectives that each name the one they read. The
four dense objectives, `cross_entropy`, `binary_cross_entropy`, `multi_label` and `mean_squared_error`,
share three keys, and the `sum` objective adds the losses of its `terms`:

| Key | Holds |
| --- | --- |
| `output` | The name of the model output the term reads. Without it the term reads the model's only output |
| `type_mask` | `true`, or `{ "columns": k, "column": c }`: the loss keeps the rows the batch role `type_mask` marks with 1, read as `[N]` or as column `c` of `[N, k]` |
| `target` | The name of the batch role the term reads its targets from, in place of `labels` (`values` for `mean_squared_error`) |

```json
"objective": {
  "type": "sum",
  "terms": [
    { "type": "cross_entropy", "output": "choice", "target": "choice",
      "type_mask": { "columns": 3, "column": 0 } },
    { "type": "mean_squared_error", "output": "number", "target": "number",
      "type_mask": { "columns": 3, "column": 2 } }
  ]
}
```

A dataset that mixes several kinds of question publishes one role of targets for each kind, and each
term keeps the rows of its own kind with `type_mask`. The roles a model reads beside its input are the
`inputs` of its model file (`docs/guides/MODEL_FILE.md`). `applications/examples/decision_model/`
holds both files for a model that answers a choice, a yes or no and a number.

### Training a detector

A detector's `objective` names its engine. `{ "type": "anchor" }` trains a model whose `task` is
`anchor_detection`, with the deltas of each head computed on the host; `{ "type": "anchor_free" }`
trains one whose `task` is `anchor_free_detection`, with the loss evaluated as a graph on the device
and the label assignment on the host. Both start the biases of a fresh run from the strides the
trace gives. The loss gains are keys of the objective; the detect and training READMEs list them.
`sushiai inference` reports the detections of every image and the mAP at IoU 0.5. Give it one image
file (`.jpg`, `.jpeg` or `.png`) in place of the evaluation file and it lists that image's
detections in the pixels of the file and scores nothing.

The example folders `applications/examples/yolov3_tiny`, `yolov4_tiny`, `yolov5n`, `yolov8n`,
`yolo11n` and `yolo26` hold a model, a training file and an evaluation file over coco128, and each
README names the commands. None of them is a preset: each model is the JSON its rows compose.

### Training on a cluster

A `cluster` section trains one model over several machines. Only a distributed build
reads it; any other build refuses the file and names the flag that builds it.

```bash
sa build --distributed
```

On each worker machine, start `sushiai_worker` with the master's training file. `--master` and
`--name` are required, and the other flags are SushiRuntime's worker flags, which
SushiRuntime's distributed guide (`DISTRIBUTED_GUIDE.md`) describes:

```bash
sushiai_worker train.json --master <host>:<port> --name <machine>
```

`--name` is the `worker` of one replica in the section. On the master, start
`sushiai_train train.json`. Replica 0 stays on the master. A restored `resume` checkpoint
also restores the replay counter of the cluster run.

The master waits until every worker the section names has attached, or until the section's
`attach_timeout_ms` passes (60000 by default, raised to the largest `wait_ms` of its `wait`
rules). A worker still missing then ends the run with `DeviceFailed` naming it when a replica on
it is `fail`; a `local` or `wait` replica goes as its rule says. A `local` replica runs on the master's
hardware, so the file is refused unless it states version 3 and `"deterministic": false`, naming
both keys.

### Resuming a run

`sushiai train` writes run state into every checkpoint, so the file `checkpoint` names is
also the file a later run can state as `resume`:

```json
"resume": "heavy_mlp.sushi",
"checkpoint": "heavy_mlp.sushi"
```

The resumed run goes on at the iteration the saved one stopped at. A `schedule` reads that
count, so warm-up is not repeated and a step that was passed stays passed; AdamW keeps its
moments and SGD its momentum.

It also goes on at the epoch index the saved one stopped at. A dataset with `shuffle` draws
the order of an epoch from its own seed and the epoch index, so the resumed run reads the
orders the saved run would have read next, and a dropout layer draws the masks it would have
drawn next. A run of 20 epochs, saved and resumed for 30 more, trains on the same batches
under the same masks as one run of 50. `epochs` counts the epochs of this run, not of both.

The `seed` has to be the one the saved run stated. A checkpoint written under `"seed": 7`
and resumed under `"seed": 8`, or under no `seed`, is refused with both numbers.

The optimizer has to be the one that wrote the file. A checkpoint of `sgd` resumed under
`adamw` is refused with both names, and so is SGD with momentum resumed from a run without
it, with the name of the record that is missing. A weight file with no run state, such as
the `weights.sushi` of `sushiai import` or a checkpoint written before version 2 of the
format, is refused as `resume` and says to state it as `weights`. So is a checkpoint whose
run state holds no `run.seed` record, which a build from before the seed existed wrote.

`sushiai inference` and `weights` read the same file and leave the run state unread.

### Layer types

`NN::register_layers` in `modules/layers/nn/include/SushiAI/nn/layer_registry.hpp` adds
sixteen layer rows; the table below lists the ones a dense or convolutional chain uses. The shapes below are per sample and leave out the batch axis. The chain
starts at `[input.features]`, and each layer receives the shape the layer
before it reports.

| `type` | Keys | Input | Output |
|---|---|---|---|
| `linear` | `units` (required, positive), `use_bias` (default `true`) | `[features]` | `[units]` |
| `gelu`, `relu`, `tanh`, `sigmoid` | none | any shape | the same shape |
| `leaky_relu` | `slope` (default `0.01`, inside the open range 0 to 1) | any shape | the same shape |
| `unflatten` | `height`, `width`, `channels` (all required) | `[height * width * channels]` | `[height, width, channels]` |
| `conv2d` | `filters`, `kernel` (required); `stride` (default `1`), `padding` (default `0`), `dilation` (default `1`), `trailing_padding` (default `0`), `use_bias` (default `true`) | `[H, W, C]` | `[out_h, out_w, filters]` |
| `maxpool2d`, `avgpool2d` | `kernel` (required); `stride` (default: the kernel), `padding` (default `0`), `trailing_padding` (default `0`) | `[H, W, C]` | `[out_h, out_w, C]` |
| `global_avgpool` | none | `[H, W, C]` | `[C]` |
| `flatten` | none | any shape | `[element count]` |
| `upsample` | `scale` (default `2`, at least `1`) | `[H, W, C]` | `[H * scale, W * scale, C]` |

`kernel`, `stride`, `padding` and `dilation` each take an integer, which sets
both axes, or a two-entry `[h, w]` array. A padding applies to both sides of
its axis. A convolution reads its input channel count from the shape before
it, so there is no key for it.

Images enter a model flat. A model that convolves starts with `unflatten`,
and a `linear` layer after an image-shaped layer needs a `flatten` before it.
The factories refuse both mistakes and name the missing layer.

### Dataset types

`Data::register_datasets` adds `rings`, and the vision registration adds `mnist`.

| `type` | Keys |
|---|---|
| `rings` | `batch_size` (default `64`), `batches` (`64`), `classes` (`4`), `distractors` (`4`), `ring_width` (`1.0`), `position_noise` (`0.12`), `seed`; all optional |
| `mnist` | `directory` (required); `split` (`"train"` or `"test"`, default `"train"`), `batch_size` (default `64`), `max_samples` (default `4096`; `0` loads every sample) |

`mnist` reads the four uncompressed IDX files under their original names and
stops with an error that names them when the split's two files are not in
`directory`.

### Why inference takes its data as an argument

The architecture file carries no data block, because it describes a model. So
`inference` needs to be told what to run on, and an optional flag that always
errored when omitted would be a lie about optionality. It traces the forward
pass only: no loss, no backward pass, no optimizer.

It reports the predicted-class distribution **and** top-1 agreement with the
labels the dataset supplies. The distribution alone would be useless: on
class-balanced data a perfect model and a random one both give a uniform
spread. Reading the labels names no loss and no objective: labels are part of
the `Data::Dataset` seam, and top-1 agreement is a property of the data and
the output rather than of what the model was trained to minimise. It is
reported only when the model's output width matches the dataset's class count,
because otherwise the two are not comparable and a number would be fiction.

### Profiling a run

Both commands take `--profile <profile.json>`, which records the run with SushiProfiler. The flag measures the run and changes nothing in it. [`PROFILING.md`](PROFILING.md) has the configuration and how to read the record.

### Exit codes and output

Both exit `0` on success and non-zero on any configuration or runtime error,
printing the library's own diagnostic, which names the file, the key and what
was found. Both print a `compiles: 1` line; two CTest cases pin it, the same
way `sa demo mlp`'s is pinned.

## `sushiai import`: a model of another framework

```bash
sushiai import darknet yolov7-tiny.cfg yolov7-tiny.weights --output models/yolov7_tiny
```

| Argument | Meaning |
| --- | --- |
| `<format>` | The importer to read with. `darknet` is the one a build with the vision domain has; a format with no importer is refused with the formats that have one |
| `<files...>` | The files of the other framework, in the order the format states them. `darknet` takes a `.cfg` and optionally a `.weights`; `caffe` takes a `.prototxt` and optionally a `.caffemodel`; `onnx` takes one `.onnx` |
| `--output`, `-o` | The directory that receives the files; it is created when absent |
| `--task` | A JSON file holding an object with a string `type`, which becomes the `task` section of `model.json`. An importer that writes a task section itself refuses the flag |
| `--force` | Replace a file that is already there. Without it an existing file stops the import before anything is written |

It writes three files into the directory:

| File | Holds |
| --- | --- |
| `model.json` | The architecture, with its task section |
| `weights.sushi` | The weights, when the foreign files held any |
| `train.json` | `model`, `weights`, `"data": null`, `"checkpoint": "trained.sushi"` and the training recipe the foreign files state |

The run prints the format, each input file, the size of the model and the tensor count,
one `wrote <file>` line per file, and then a list of notes. A note names each value of the
foreign files that none of the three files holds; for Darknet these are `batch`,
`subdivisions`, `max_batches` and the augmentation keys.

`train.json` is not ready to run. No foreign file names a dataset, so the import writes
`"data": null`, and `sushiai train` refuses the file until you fill the block. Without a
`.weights` file there is no `weights` key and a run initialises the model.

The command creates no engine and needs no device.

### `sushiai import caffe`: a Caffe net

```bash
sushiai import caffe lenet.prototxt lenet.caffemodel --output models/lenet
```

The command reads a deploy prototxt and, when given, the caffemodel with its weights, and writes
the three files above. The layers it translates are `Input`, `Convolution`, `Pooling`, `ReLU`,
`InnerProduct`, `Concat`, `Eltwise`, `Softmax`, `Dropout`, and `BatchNorm` with the
`Scale` after it, which become one normalisation row. A net with a layer outside that list is
refused with one message that names every such layer, so one run shows the whole gap.

Three differences from Caffe show in the written files. The model reads `[H, W, C]` images, so
convolution weights are stored `[out, kh, kw, in]` and the inputs of an inner product follow the
order of a flattened `[H, W, C]` image. A `Softmax` becomes a `softmax` row, and a net that ends in
it publishes the probabilities. `Eltwise` with `SUM`, `PROD` or `MAX` becomes `add`, `mul` or
`maximum` rows, and a grouped `Convolution` states `groups`. The import writes no task section; a
Caffe net states none.

### `sushiai import onnx`: an ONNX graph

```bash
sushiai import onnx detector.onnx --output models/detector --task detector_task.json
```

The command reads an ONNX file of opset 13 or newer with one float32 image input `[N, C, H, W]`
whose C, H and W are stated, and writes the three files above. The operators it translates are
`Conv`, `BatchNormalization`, `Relu`, `Sigmoid`, `Mul` (only as the SiLU of a `Sigmoid` and its own
input), `Add`, `Sub`, `Div`, `Softmax`, `Transpose`, `Concat`, `MaxPool`, `Resize`, `Reshape`,
`Slice`, `Split`, `Flatten`, `Gemm`, `MatMul`, `Identity` and `Constant`, each in the forms
[`modules/vision/importers/README.md`](../../modules/vision/importers/README.md) lists. A graph with
a node outside those forms is refused with one message that names every such node, so one run shows
the whole gap.

The model reads `[H, W, C]` images, so convolution weights are stored `[out, kh, kw, in]`, and an
output flattened from an image lists its features as `(H, W, C)` where ONNX lists `(C, H, W)`. An
ONNX file states no task and no training recipe, so `model.json` has no `task` section until you
pass `--task`, and `train.json` states no recipe.

### `sushiai import weights`: a weight file for a model you wrote

A weight file holds tensors and no architecture, so it cannot produce a model file. When you
have written the model file yourself, `import weights` fills its state records from the weight
file and writes the `.sushi` file.

```bash
sushiai import weights model.json yolox_x.pth.tar --map yolox_map.json --output yolox_x.sushi
```

| Argument | Meaning |
| --- | --- |
| `<model.json>` | The model file whose parameters and buffers the weights fill |
| `<weights-file>` | A PyTorch file (`.pth`, `.pt`, `.pth.tar`) or a `.safetensors` file. The command reads the container from the first bytes of the file, not from its extension |
| `--map` | The name map, below |
| `--output`, `-o` | The `.sushi` file to write; its folder is created when absent |
| `--force` | Replace the output when it exists |

The name map is a JSON object with three optional keys:

```json
{
  "state_dict": "model",
  "rename": [{"from": "backbone.", "to": "body."}],
  "drop": ["num_batches_tracked"]
}
```

| Key | Meaning |
| --- | --- |
| `state_dict` | The key of the state dictionary inside a PyTorch file, such as `model` in a ByteTrack checkpoint. Leave it out when the file is the dictionary. A safetensors file has no nesting and ignores it |
| `rename` | Prefix rules applied to each tensor name, in order; the first rule that matches wins. An empty `from` matches every name |
| `drop` | Names the model has no record for. An entry matches a tensor whose name is the entry, or ends in a dot and the entry, so `num_batches_tracked` drops that buffer of every normalisation. Drops are read against the name as the file states it, before any rename |

The import is strict both ways. It fails, and writes nothing, when a record of the model has
no tensor, a tensor has no record, two tensors rename to one record, or a shape differs. One
message names every such entry, grouped by kind, so one run shows the whole gap. A tensor whose
shape is its record's is copied as it is. A convolution weight stored `[out, in, kh, kw]`, the
PyTorch layout, is permuted to the record's `[out, kh, kw, in]`, and a depthwise one stored
`[channels, 1, kh, kw]` to `[kh, kw, channels]`; any other difference of shape is refused as
misshapen. Values are read as float32 and stored in the record's element type, float32 or
float64.

## `sushiai tokenize`: a vocabulary and a token file

```bash
sushiai tokenize train  corpus.txt --vocabulary-size 4096 --output vocab.savb
sushiai tokenize encode corpus.txt --vocabulary vocab.savb --output corpus.satk
```

The command exists in a build with the language domain.

| Argument | Verb | Meaning |
| --- | --- | --- |
| `<corpus>` | both | A text file; its bytes are read as they are. Exactly one |
| `--vocabulary-size <N>` | `train` | The tokens of the learned vocabulary, bytes and end-of-text included |
| `--vocabulary <file>` | `encode` | The vocabulary file `tokenize train` wrote |
| `--output <file>` | both | The file written |
| `--force` | both | Replace an output file that is already there. Without it an existing file stops the run before anything is read |

`train` learns a byte-pair vocabulary and prints the corpus with its size in bytes, the
token count with how many of the tokens are merges, and `wrote <file>`. `encode` prints the
corpus, the vocabulary with its token count, the number of tokens written, and
`wrote <file>`. The token file is what a `tokens` data block of a training file names.

`train` holds the whole corpus in memory. `encode` reads it a mebibyte at a time and holds
the ids, four bytes a token, until it writes them.

A verb other than `train` and `encode`, a missing corpus or output, a second corpus and a
flag with no value are each refused with the usage line.
