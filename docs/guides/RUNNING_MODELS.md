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
  "sushiai_train": 2,
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
  rather than a misread. The training file states version 2, which may carry a
  `cluster` section; a file that states version 1 still reads, and a `cluster`
  section in it is an error.
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

Six optional parts of `train.json`:

| Key | Holds |
| --- | --- |
| `weights` | A `.sushi` file loaded by name before training, in place of the initialiser. A name, shape or element type that differs is an error |
| `resume` | A `.sushi` checkpoint a run of the same model, optimizer and `seed` wrote. It loads the model state as `weights` does and the run state with it: the optimizer's step count and its momentum buffers or moments, and where the run stood. A file states `weights` or `resume`; both is an error |
| `seed` | An integer of 0 or above, 0 when absent. It seeds the random sites of the training graph, which today are the dropout layers. The `seed` of a `data` block is a different key: it seeds the data |
| `schedule` | `{ "type": "step", "base_rate": ..., "warmup_iterations": ..., "warmup_power": ..., "steps": [...], "scales": [...] }`. A file with a schedule states its rate there: `learning_rate` in `optimizer` is then an error |
| `data` with `reader` | A dataset read from files, in place of `type` |
| `cluster` | Version 2 only. The replicas that train the model on several machines. Its fields are in `modules/core/distribution/README.md` |

A `data` block with `reader`:

| Key | Default | Meaning |
| --- | --- | --- |
| `reader` | required | `yolo_txt`, `image_folder`, `idx` or `cifar` |
| `task` | required | `detection` or `classification` |
| `root` | required | The dataset's directory, resolved against the working directory |
| `split` | `"train"` | As the reader takes it |
| `batch_size` | 16 | |
| `input` | required | `[height, width]` of the network input |
| `channels` | 3 | |
| `fit` | The model file's, else `letterbox` for `detection` and `segmentation` and `resize` for the other tasks | `resize` scales each axis on its own; `letterbox` scales both by one ratio and pads. A `fit` other than the model file's is an error |
| `classes` | The count the samples name | `yolo_txt` requires it |
| `max_objects` | required for `detection` | The object slots of one image; an image with more is an error |
| `cache`, `cache_side` | none, 0 | A directory of raw images written on the first pass and read afterwards; `cache_side` bounds the longest side |
| `shuffle`, `seed` | false, 0 | A sample order drawn anew at each epoch from the seed |
| `pipeline` | `"none"` | `"default"` runs mosaic, the affine warp, flips and colour on each sample |
| `workers`, `prefetch` | 0, the worker count | Threads that read batches ahead, and how many batches they hold |

`applications/examples/yolov7_tiny/` holds a training file that uses all three.

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
it is `fail`; a `local` or `wait` replica goes as its rule says.

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
| `conv2d` | `filters`, `kernel` (required); `stride` (default `1`), `padding` (default `0`), `dilation` (default `1`), `use_bias` (default `true`) | `[H, W, C]` | `[out_h, out_w, filters]` |
| `maxpool2d`, `avgpool2d` | `kernel` (required); `stride` (default: the kernel), `padding` (default `0`) | `[H, W, C]` | `[out_h, out_w, C]` |
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
| `<files...>` | The files of the other framework, in the order the format states them. `darknet` takes a `.cfg` and optionally a `.weights` |
| `--output`, `-o` | The directory that receives the files; it is created when absent |
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
