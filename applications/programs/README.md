# programs

The four drivers behind `sushiai train`, `sushiai inference`, `sushiai import` and
`sushiai tokenize`. The first two read JSON configuration; the third writes it; the fourth
prepares the data of a language model. None of the first three names a layer type, a dataset
type or a foreign format.
[`applications/examples/tiny_decoder/`](../examples/tiny_decoder/README.md) holds the files
of a decoder.
[`applications/examples/yolov7_tiny/`](../examples/yolov7_tiny/README.md) holds a training file
for a detector.
[`applications/examples/heavy_mlp/`](../examples/heavy_mlp/README.md)
and [`applications/examples/mnist_cnn/`](../examples/mnist_cnn/README.md) each hold a
configuration to run
them with, and [`docs/guides/CLI_GUIDE.md`](../../docs/guides/CLI_GUIDE.md) documents the schema.

| File | Command |
| --- | --- |
| `train.cpp` | `sushiai train <train.json>` |
| `inference.cpp` | `sushiai inference <model.json> <model.sushi> <eval.json>` |
| `import.cpp` | `sushiai import <format> <files...> --output <directory> [--force]` |
| `sushiai_worker.cpp` | `sushiai_worker <train.json> --master host:port --name <worker> [flags]`, built only when `SA_ENABLE_DISTRIBUTED` is on |
| `tokenize.cpp` | `sushiai tokenize train <corpus> --vocabulary-size <N> --output <vocab.savb>` and `sushiai tokenize encode <corpus> --vocabulary <vocab.savb> --output <corpus.satk>` |

| File | Holds |
| --- | --- |
| `registries.hpp`, `registries.cpp` | `Registries`: the node, dataset, objective, run, report and importer rows a program looks up, and `compose_registries`, which fills them |
| `inference_run.hpp`, `inference_run.cpp` | `PreparedInference`, `InferenceRegistry`, `report_inference` and `print_inference_header`: the report table |
| `training_run.hpp`, `training_run.cpp` | `PreparedRun`, `RunSummary`, `RunRegistry` and `run_training`: the run table; `finished_progress` and `check_start_batch`, which both runs call |
| `epoch_loop.hpp`, `epoch_loop.cpp` | `run_epochs`: the epoch, schedule and report loop the trainer run and the cluster run share, and `is_reported_epoch`, the cadence the vision run prints at |
| `trainer_run.hpp`, `trainer_run.cpp` | `register_runs`, which sets the fallback run: `Train::Trainer` |
| `run_start.hpp`, `run_start.cpp` | `start_state` and `describe_start`: a run initialised, started from `weights` or resumed, and the `IO::RunProgress` it starts at |
| `domains.hpp` | `register_vision_domain` and `register_language_domain`, one per domain a build may link |
| `domains_vision.cpp` | The vision rows, the `anchor` run and the `anchor_detection` and `anchor_free_detection` reports; compiled when `SA_BUILD_VISION` is on |
| `domains_vision_none.cpp` | No row; compiled when it is off |
| `domains_language.cpp` | The language rows and the `next_token_generation` report; compiled when `SA_BUILD_LANGUAGE` is on |
| `domains_language_none.cpp` | No row; compiled when it is off |
| `cluster_training.hpp`, `cluster_training.cpp`, `cluster_run.cpp`, `master_cluster.cpp`, `served_kernel.cpp`, `training_replica_model.cpp`, `replica_tensors.cpp`, `worker_arguments.cpp` | Training a cluster section over its replicas and serving a replica; compiled when `SA_ENABLE_DISTRIBUTED` is on |
| `training_cluster_section.hpp`, `training_cluster_section.cpp` | `read_training_cluster`: reads the cluster section of a training file and refuses a replica with `if_absent: "local"` while `deterministic` is true; compiled when `SA_ENABLE_DISTRIBUTED` is on |
| `cluster_training_none.cpp` | Refuses a cluster section and names `sa build --distributed`; compiled when it is off |

A distributed build links the first set into every program. `sushiai_train` then trains a
`cluster` section on the master, and `sushiai_worker` serves the replica named by its `--name`
on each worker machine; the other flags go to SushiRuntime's worker. The training file comes
first: `-h`, `--help`, `-V` or `--version` there prints and exits 0, and any other flag there is
refused with the usage line.

Before the first step the master waits until every worker the section names has attached, or
until the attach bound passes, whichever comes first. The bound is the section's
`attach_timeout_ms`, 60000 when absent, raised to the largest `wait_ms` of the section's `wait`
replicas. The master tells the workers apart by the name each sends when it attaches. When the
bound passes with a worker missing whose `if_absent` is `fail`, the run ends with `DeviceFailed`
naming every missing worker; a missing worker under `local` or `wait` lets the run go on, and
its replica goes as its rule says.

`ReplicaSource::fingerprint`, which the master and every worker must agree on, hashes what a
replica's result depends on: the architecture file's bytes, the objective's `type` and its keys,
the data block's `type` and its keys, the run's `seed`, the run index (`K_RUN_INDEX`, 0 for
every run), and the `rows` and `micro_batch` of every replica. It leaves out the optimizer, the
schedule, `epochs`, `weights`, `resume`, `checkpoint`, and the section's `listen`,
`secret_file`, `attach_timeout_ms`, `worker`, `priority`, `if_absent` and `wait_ms`, which only
the master reads or which name where output goes.

`train.cpp` builds the `TrainingSetup` of a run with `SetupScope::Step` and the master of a
cluster run with `SetupScope::Results`. A `Results` setup binds the state, a gradient per
parameter, the loss and the sources of the buffer updates through `Train::bind_step_results`,
plans no memory and holds no optimizer; the replicas run the step, and the master only reads
what comes back. The run is described in
[`docs/guides/RUNNING_MODELS.md`](../../docs/guides/RUNNING_MODELS.md).

The sections below hold the reasoning that the source files cite.

## train.cpp

### The property the file exists to have

The driver is written once and does not change as the library grows. It names no layer type,
no loss, no dataset and no optimizer. Each of those arrives through a registry lookup keyed
by a string out of a JSON file, so adding a layer, a detection loss or a COCO reader is a
table row and a factory function in the library, and `train.cpp` stays as it is. The
`conv2d` layer row of the `NN::NodeRegistry` and the `mnist` row of the `Data::DatasetRegistry`
are rows of that kind.

The property can be checked: search the file for `Linear`, `GELU`, `Conv2d`, `CrossEntropy`,
`AdamW`, `SGD`, `Rings`, `MNIST`, `Sequential`, `Anchor` or `Vision` and there is nothing to
find. If an edit adds one, the driver has stopped being what the config-driven design set out
to build.

### The steps

1. Load the training file and the model file it names.
2. Fill the registries and build the dataset of the `data` block.
3. Build the model and trace it once without a loss (`build_probe_graph`), which gives the
   shape of every published output and needs no engine.
4. Fill a `Train::ObjectiveContext`: the batch, the class count of the dataset's `labels` or
   `classes` role (0 when it has neither), the sample shape, the `task` section of the model
   and the output shapes. Build the objective from it through the objective registry.
5. Read the schedule with `Optim::read_rate_schedule`, which refuses a rate stated twice and returns a step or a cosine schedule; both loops take either.
6. Build the training graph, match the roles the objective reads against the dataset, create
   the state and bind the parameters. The probe graph
   and the training graph take the element type of their input leaf from `input.dtype` of the
   model file, so a model whose first row is an `embedding` states `"dtype": "int32"`. Both
   take `GraphOptions::seed` from the `seed` of the training file through
   `Config::read_run_seed`.
7. Build the optimizer, then give the state its starting values with `start_state`:
   initialised, loaded from `weights`, or loaded with the run state from `resume`. It fills
   the `IO::RunProgress` the run starts at. Bind the buffer updates.
8. Hand all of it to the run the objective type names, then write the checkpoint with
   `IO::save_run_checkpoint` and the progress the run returned.

### The run table

An objective that fills its own leaves from a batch trains through `Train::Trainer`, which
replays one compiled step. An anchor update does not fit that loop: it runs a forward-only
pass, computes a delta per head on the host and then replays the training step. The driver
therefore holds no loop. It fills a `PreparedRun` and calls `run_training`, which looks the
objective type up in a `RunRegistry`. A type with a row has a step of its own and the row's
function runs it. Every other type goes to the fallback run, so the table names no objective
the trainer can feed, and a domain adds one by registering the objective alone.

| Objective type | Run | Added by |
| --- | --- | --- |
| `anchor` | `Vision::Training::AnchorTrainingStep` under `run_anchor_training` | `register_vision_domain`, `domains_vision.cpp` |
| Any other: `cross_entropy`, `binary_cross_entropy`, `multi_label`, `mean_squared_error`, `next_token` | `Train::Trainer::run_step` per batch | `register_runs`, `trainer_run.cpp` |

An objective that keeps the default `Train::Objective::feed` and has no row fails its first
step with `Unsupported`, naming the objective. A registry with no fallback refuses a type
with no row and lists the types that have one.

A run prints its own progress lines and returns a `RunSummary`: the compile count of its
training step and the `IO::RunProgress` it stands at.

### One seed, three graphs

The driver builds every graph a run trains through, so the seed is set in one place. The probe
graph and the training graph are traced in `train.cpp` with the same `GraphOptions`. The
classifier run and the language run (`next_token` trains through the fallback) replay the
training graph. The `anchor` run replays both: its forward-only pass is the probe graph, which
holds the same dropout sites under the same seed. No run traces a training graph of its own.

The objective registry returns a `Train::Objective`, and `AnchorTrainingStep::make` takes a
`Vision::Training::AnchorObjective`. The anchor run calls `build_anchor_objective` with the
same context and section the registry row was given, so it holds the typed objective without a
cast. The two are built from the same values and declare the same leaves.

### The rate before each update

When the training file has a `schedule`, both runs set the optimizer's rate before every
update with `Optim::apply_step_schedule`, whose iteration is the optimizer's own step count.
The trainer run calls it between the read of a batch and `Trainer::run_step`;
`Train::Trainer` needed no hook, because the loop that calls `run_step` belongs to the run.
`run_anchor_training` does the same inside the vision library.

### Starting weights

`"weights"` in the training file names a `.sushi` file, resolved like `model`. The driver loads
it with `IO::load_checkpoint` in place of `NN::initialize`. The load is strict both ways: a
record the model lacks, an entry the file lacks, and a shape or element type that differs each
stop the run with a status that names the entry. A file without the key initialises as before.

### Resuming a run

`"resume"` names a checkpoint a run of the same model and optimizer wrote. `start_state` loads
it with `IO::load_run_checkpoint`: the model state as `weights` would, the run state of the
optimizer, which is its step count and its momentum buffers or moments, and the
`IO::RunProgress` of the saved run. The step schedule
reads the step count, so the resumed run goes on at the iteration and the rate the saved run
would have had next. A file states `weights` or `resume`; `Config::check_start` refuses both.

`start_state` then calls `IO::check_run_seed`: a checkpoint written under another `seed` than
the training file states is refused with the file and both seeds. The progress it hands back
goes into `PreparedRun::start`, and each run does three things with it before its first
update. `check_start_batch` refuses a batch the dataset lacks. The run restores the run count
of its compiled step, `Trainer::restore_run_count` in the trainer run and
`AnchorTrainingStep::restore_run_count` for both passes of the `anchor` run, so a dropout site
draws the mask of the run index the saved run would have reached. Its first pass calls
`Data::begin_epoch` with the saved epoch index and reads from the saved batch on.

The driver owns the save and the restore because it owns the checkpoint call: it creates the
state and the optimizer, hands both to whichever run the table picks, and writes the file when
the run returns. A run of a domain trains through the same optimizer, so its state is saved
without the run knowing. Every checkpoint the driver writes carries run state; `sushiai
inference` and `"weights"` read the model state of the same file and leave the rest unread.

`epochs` counts the epochs of this run. A run that starts at epoch index `e` reads the epochs
`e` to `e + epochs - 1`, and `finished_progress` records `e + epochs` and batch 0 as where it
stands. The driver writes its checkpoint when the run returns, so the batch it writes is
always 0; the format and both loops carry a batch above 0 for a caller that saves inside an
epoch, and the first pass of such a run is the rest of that epoch. The progress lines number
the epochs of this run from 1.

A checkpoint whose run state has no progress records is refused as `resume` and loads as
`weights`; `modules/layers/io/README.md` has the message.

### Why it takes no flags

`applications/examples/source/demo_mlp.cpp` is the flag-driven program, and its flags are a contract the test
suite pins. In `train.cpp` every knob lives in the configuration file, because a knob that
exists in two places drifts.

### The metric belongs to the objective

The trainer run hands `Train::Trainer::make` the objective the training graph was built with.
`run_step` returns the loss and the `Train::Metrics` the objective measured, and the run
prints every one of them after the loss: `accuracy` for `cross_entropy`, `perplexity` for
`next_token`, nothing for an objective that measures nothing. The run computes no number
itself.

The epoch loop of that run calls `run_step` itself and does not call `fit()`, because it sets
the scheduled rate before each step. The `anchor` run prints the cost `run_anchor_training`
reports for each update.

### The compile-count line

The driver prints `compiles: N` in the shape `sa demo mlp` prints it, because the CTest case
`Medium_App.TrainRunsAndReportsOneCompile` in `tests/CMakeLists.txt` pins the line. `N` is
the compile count of the training step the run returns. The `anchor` run also holds a
forward-only step and fails unless that one was compiled exactly once.

## inference.cpp

### Three positional arguments

The data file is one of the three positional arguments. The architecture file describes a
model and carries no data block, so without the data file there would be nothing to run on.
An optional flag that always errors when omitted would misstate what is optional.

### Forward pass only

The driver traces the forward pass and nothing else: no loss, no backward pass, no
optimizer. Inference minimises nothing, so there is no objective, and asking a configuration
for a loss the driver will not use would be asking for a value with no meaning.

Like `train.cpp`, it names no layer type and no dataset type.

### The report table

What the outputs of a model mean depends on its task, so the driver holds one report and looks
the others up. It builds the model, loads the weights, compiles the forward pass and fills a
`PreparedInference`. A model file without a `task` section is reported as a classifier, by
`report_classes` in `inference.cpp`; that path prints what it printed before the table
existed. A model file with a `task` section is reported by the row of `task.type` in an
`InferenceRegistry`. A type with no row is refused with the types that have one.

| `task.type` | Report | Added by |
| --- | --- | --- |
| none | Predicted-class counts and top-1 agreement | `inference.cpp` |
| `anchor_detection` | The detections of every image | `register_vision_domain`, `domains_vision.cpp` |
| `anchor_free_detection` | The detections of every image | `register_vision_domain`, `domains_vision.cpp` |
| `next_token_generation` | A prompt and its continuation | `register_language_domain`, `domains_language.cpp` |

The `anchor_detection` row reads the section with `read_anchor_task`, binds each head to the
output it names, decodes every head of a batch with `decode_head` and passes the rows to
`select_nms_detections` under the criterion of the section. It prints one line per detection:
the class, the score and the box as `x1 y1 x2 y2` in pixels of the network input, which is
the image fitted as the model's `fit` says and not the file on disk. A detection is printed when its score is
above 0.25, and a box is suppressed above a measure of 0.45. Both numbers are constants of
`domains_vision.cpp`; no file states them.

When the data carries a `boxes` role, the row then prints the mean average precision at IoU 0.5
over the split: `truth_from_roles` reads the labels of each batch, `add_batch` feeds them with the
detections to an `AveragePrecision`, and the last line gives `map50`, the box count and the image
count. The metric reads the detections of the validation confidence, 0.001, and not only those
above 0.25: the suppression runs once at 0.001 and the printout keeps the rows above 0.25, which
is the same set because a box is suppressed only by a box of higher score. Truth and detections
are both in pixels of the network input, so the number does not depend on the size of the file on
disk. Data without a `boxes` role prints a line saying the mean was not computed.

The `anchor_free_detection` row reads the section with `read_free_task` and takes the levels,
strides, reg_max, class count and one-to-many port names from `derive_free_model_facts` over
the shapes the step publishes; the section states none of them. `FreeBatchDecoder` turns the
ports of a batch into centre-size boxes in input pixels and the sigmoid of each class logit.
`select_nms_detections` keeps the best class of each anchor, class-aware, at the section's
`suppression_threshold`, and the printout keeps the rows above its `score_threshold`. The
printing and the mean average precision are those of the `anchor_detection` row. The one-to-one
ports are not read.

The command still takes a data file, so it reports a dataset split. It does not take one
image file.

The `next_token_generation` row reads the section with `read_generation_task`. The prompt
reaches it as the data: `eval.json` holds a `prompt` data block, whose `text` the dataset
encodes with the vocabulary file it names, and the row reads batch 0 and the `prompt_length`
role of it. The command takes a data file for every model, and a prompt is the data of a
generation, so no key was added to the parser of `modules/core/config`.

```json
{ "sushiai_data": 1,
  "data": { "type": "prompt", "text": "Once upon a time", "vocabulary": "vocab.savb",
            "window": 256 } }
```

The row refuses a `window` that is not the model's input shape, a model that publishes more
than one output and a vocabulary file that holds more tokens than the model scores. It
resolves the `vocabulary` of the task section against the directory of the model file; the
`vocabulary` of the data block is relative to the working directory, as every path of a data
block is. The two keys name the same file: the data block encodes with it and the report
decodes with it. The row prints the prompt followed by the continuation.

The row picks its generator from a table of two rows by the `cache` of the task section,
which is true when the section leaves it out:

| `cache` | Generator | Runs |
| --- | --- | --- |
| `true` | `Language::Generation::CachedGenerator` | One position per token over a step the row traces with `Train::build_incremental_graph` and compiles into its own pool |
| `false` | `Language::Generation::WindowGenerator` | One forward pass of the window per token over the step of `PreparedInference` |

The report names the generator on its `Generator` line.

`PreparedInference` carries the published outputs and the engine beside what it held, since
this row names its one output and allocates what it feeds. It also has two pointers that are
null unless the driver sets them: `built_model`, the `NN::ModelGraph`, and `state`, the loaded
`NN::ModelState`. The cached row traces a second form of the model from the first and binds
it to the second, and returns `Unsupported` when either is null. `inference.cpp` does not set
them yet, so a model whose section does not state `"cache": false` is refused until it does;
the window step it compiles is then unused by the cached row.

The check that a model publishes exactly one output now applies to a model without a task
only. A detector publishes one output per head.

### Two summaries

The driver prints the predicted-class distribution and top-1 agreement with the labels. The
distribution alone cannot tell a perfect model from a random one on class-balanced data,
because both give a uniform spread, so it is printed beside the agreement and not in its
place.

Reading `Batch::labels` names no loss and no objective. Labels are part of the `Dataset`
seam, and top-1 agreement is a property of the data and the output, not of what the model
was trained to minimise. The driver reports it only when the model's output width equals the
dataset's class count. When they differ the two are not comparable, and the driver says so
and prints no number.

## tokenize.cpp

### Two verbs

`train` reads a corpus and writes the vocabulary `Language::Text::train_bpe` learns from it.
`encode` reads a corpus and a vocabulary file and writes a token file. A table of the two
verbs picks the function, and a verb outside it is refused with both names. The driver links
`sushiai_language` and is built only when `SA_BUILD_LANGUAGE` is on. It composes no registry
and creates no engine.

A file that exists at `--output` stops the run before anything is read; `--force` replaces it.

### Memory

`train` holds the whole corpus, since the trainer reads one text: 100 MB for a corpus of
100 MB, and beside it what the trainer keeps, four bytes per byte of a distinct chunk and its
pair table, which `modules/language/text/README.md` states and nobody has measured on such a
corpus.

`encode` reads the corpus through `Language::Text::encode_stream`, one mebibyte at a time, and
holds the ids until it writes them, because `write_token_file` takes all of them at once: the
header of a token file states the count before the ids. That is four bytes per token, and
the vector holds its old and its new storage at once each time it grows. A
corpus of 100 MB is at most 100 million tokens and 400 MB when no merge applies, and about
100 MB at four bytes of text per token. A writer that appends would bound it at the piece; the
token file module has none.

## import.cpp

### What it writes

`sushiai import <format> <files...> --output <directory>` writes `model.json`, `weights.sushi`
and `train.json` into the directory, as
[`docs/design/MODEL_IMPORT/README.md`](../../docs/design/MODEL_IMPORT/README.md) describes. The driver looks
`<format>` up in the `IO::ImporterRegistry` of `Registries`, hands the files to the row, and
passes the `IO::ImportedModel` it gets back to `IO::write_imported_model`. It names no format:
`darknet` is a row `register_vision_domain` adds in `domains_vision.cpp`, and a format with no row is
refused with the formats that have one.

The driver creates no engine. An imported model holds its weights as bytes on the host, and
the weight file is written from those.

### The arguments

Everything before `--output` (or `-o`) that is not a flag is a foreign file, in the order the
format states them. `darknet` takes a `.cfg` and optionally a `.weights`. Without a `.weights`
no weight file is written and `train.json` has no `weights` key, so a run initialises the
model.

A file that already exists in the directory stops the import before anything is written.
`--force` replaces it.

### The notes

After the files, the driver prints a list of notes. An importer adds one for each value of the
foreign files that none of the three files holds; the Darknet importer does so for `batch`,
`subdivisions`, `max_batches` and the augmentation keys. The driver adds one of its own: the
written `train.json` states `"data": null`.

### The data block is null

No foreign file names a dataset, and the import invents none. `train.json` carries the `data`
key with the value `null`, so the file shows the key a person has to fill.
`sushiai train` refuses the file as it is: `Config::parse_train` reports that `data` must be
an object and is null, and names the file.

## Designs

<!-- designs:begin -->
- [LAYERED_EXECUTION](../../docs/design/LAYERED_EXECUTION/README.md): SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and it trains one model over the owner's five machines through SushiRuntime's distributed...
<!-- designs:end -->
