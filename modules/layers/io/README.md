# io

## Owns

The io module reads and writes model state on disk. `save_checkpoint` writes every entry of a
`NN::ModelState`, parameters and buffers alike, in the SACP binary format. `load_checkpoint`
reads such a file back and requires that every entry is supplied exactly once, strictly in both
directions. `checkpoint_format.hpp` defines the format, with the magic word "SACP", format
version 2, at most 4096 model-state records and 16384 run-state records per file, and names of
at most 256 bytes; a reader reads versions 1 and 2 and refuses any other with the versions it
reads. `import_weights` loads a file into a state after it rewrites each record name
through the caller's `RenameRule`s, and it fails on a record that matches no entry, repeats, or
differs in dtype or shape, and on an entry that no record supplies.

`save_run_checkpoint` writes the model state and, in the run-state section of the file, what an
`Optim::Optimizer` keeps across updates and a `RunProgress`: the seed of the run, the run count
of its training step, and the epoch index and the batch its next update reads.
`load_run_checkpoint` reads the model state and the optimizer's records back into a model
state and an optimizer of the same type, strictly both ways, and returns the `RunProgress`
through a pointer. `check_run_seed` refuses a progress whose seed is not the seed of the run
that resumes. The private header
`source/model_records.hpp` holds the writer and the strict reader of the model-state records,
which the two writers and the three loaders share.

The module also owns the door a model of another framework enters through,
`docs/design/MODEL_IMPORT.md`. `ImportedModel` is what an importer returns: a
`Config::ModelSpec` with its task section, the weights, the training recipe and a list of
notes. `ImporterRegistry` holds one row per format name, and a domain adds its rows to it;
this module adds none. `write_imported_model` writes `model.json`, `weights.sushi` and
`train.json` into a directory.

An imported model holds each weight as an `ImportedTensor`: a `TensorRecord` and its bytes on
the host. A `NN::ModelState` cannot hold them, because a state is created on an engine from a
built model, and an importer has neither. `CheckpointWriter::write` takes a record and a
pointer to its bytes, so the writer needs no engine either.

`compose_training_file` returns the training file as a value: `sushiai_train`, `model`,
`weights` when the model has tensors, `data`, `checkpoint`, then the keys of the recipe in
their order. `data` is null, because no foreign file names a dataset; `Config::parse_train`
refuses a null `data`, so a person fills the block before a run. A recipe that states one of
the writer's five keys is refused.

`write_imported_model` checks everything before it writes anything: it composes both texts,
checks every tensor (a name, at most 256 bytes, stated once, and as many bytes as its shape
needs) and, under `ExistingFiles::Refuse`, stops at the first of its files that exists. It
then creates the directory and writes the model file, the weight file when there are tensors,
and the training file.

## The file, byte by byte

Every integer is little-endian and nothing is padded or aligned. A payload is the elements of
the tensor in row-major order, each little-endian. The writer and the reader copy tensor memory
as it is, so both return `Unsupported` on a host that is not little-endian.

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | The magic: `53 41 43 50`, ASCII "SACP" |
| 4 | 4 | `format_version`, uint32: 2 |
| 8 | 4 | M, the count of model-state records, uint32, at most 4096 |
| 12 | 4 | R, the count of run-state records, uint32, at most 16384; 0 for a file with no run state |
| 16 | | M records, in the order `NN::ModelState::entries` lists them |
| After them | 4 | Only when R is above 0: L, the length of the owner name, uint32, 1 to 256 |
| | L | The owner name, with no terminator: the optimizer type, `sgd` or `adamw` |
| | | R records: the optimizer's, then the four of the run's progress |

A record has one layout in both sections and in both versions:

| Bytes | Field |
| --- | --- |
| 4 | N, the length of the name, uint32, at most 256 |
| N | The name, with no terminator |
| 4 | The element type, uint32: the value of its `DType` enumerator |
| 4 | K, the rank, uint32, at most `K_MAX_RANK` |
| 8 x K | The extents, int64 each, every one positive |
| Elements x element size | The payload |

| Version | Shipped in | Layout |
| --- | --- | --- |
| 1 | Unreleased | A header of 12 bytes, magic, version and M, then M records |
| 2 | Unreleased | The table above: R joins the header, and the run-state section follows the model records. The four progress records joined the section before version 2 shipped, so the number did not change |

`CheckpointReader::open` reads either header and upgrades a version 1 header in memory to one
with R of 0, so a version 1 file reads as model state with no run state. `CheckpointWriter`
writes version 2 only.

## The run-state section

The owner name is `Optimizer::type_name()`. The records are the scalars and then the tensors
of `Optimizer::run_state()`, and then the four fields of a `RunProgress`, which close the file:

| Record | Element type and shape | Holds |
| --- | --- | --- |
| `step_count` | Int64, `[1]` | The updates applied; the schedule and Adam's bias correction read it |
| `loss_scale` | Float64, `[1]` | The scale of the attached loss scaler; absent without one |
| `loss_scale.good_steps` | Int64, `[1]` | The clean steps since the scale last moved; absent without one |
| `<parameter>.momentum` | As the parameter | SGD's momentum buffer; absent when momentum is 0 |
| `<parameter>.first_moment`, `<parameter>.second_moment` | As the parameter | AdamW's moments |
| `run.seed` | Int64, `[1]` | The seed the training graph of the run was traced with |
| `run.step_runs` | Int64, `[1]` | The runs the compiled training step has finished; a dropout mask is drawn from it |
| `run.epoch` | Int64, `[1]` | The index of the epoch the next batch is read in, from 0; at most 2^63 - 1 |
| `run.batch` | Int64, `[1]` | The index of the next batch of that epoch, from 0 |

A progress record is 28 bytes and its name: 36 for `run.seed`, 41 for `run.step_runs`, 37 for
`run.epoch` and 37 for `run.batch`, 151 bytes in all. Its payload is the count as an unsigned
64-bit integer, low byte first, as `step_count` is written.

| Reader | Version 1 file | Version 2, R of 0 | Version 2 with run state | Version 3 or above |
| --- | --- | --- | --- | --- |
| `load_checkpoint`, `import_weights` | Model state | Model state | Model state; the section is not read | `InvalidFile`, "reads versions 1 to 2" |
| `load_run_checkpoint` | `Incompatible`: no run state | `Incompatible`: no run state | Model state and run state | `InvalidFile`, "reads versions 1 to 2" |

`load_run_checkpoint` refuses, with `Incompatible` and the name at fault, a section another
optimizer type wrote, a record neither the optimizer nor the progress keeps, a record stated
twice, a record of another element type or shape, and a tensor, a scalar or a progress record
that the section does not supply. A `run.epoch` above 2^63 - 1 is `InvalidFile`. A failed load
leaves the state and the optimizer partly loaded and the caller's `RunProgress` as it was. The
counters a loss scaler keeps for its report, skipped steps, backoffs and growths, are not
written.

A version 2 file whose section holds the optimizer's records and no progress records cannot be
resumed: the load returns `Incompatible`, names `run.seed` as the record the file does not
supply, and says the file can be the `weights` a run starts from. Only a working tree between
the two changes wrote such a file. `load_checkpoint` and `import_weights` read it as model
state.

`save_run_checkpoint` takes the progress as its last argument, and a caller that states none
writes a run at its start under the seed 0. The loader requires the four records whether or not
the caller passes a pointer for them. The module compares no seed during a load:
`check_run_seed(saved, seed, file)` returns `Incompatible` with the file and both seeds, and the
program that resumes calls it.

## Depends on

`CMakeLists.txt` lists `PUBLIC_DEPENDS core config nn`; `optim` has to join it for
`run_checkpoint.hpp`. The module uses four modules:

- `core`: `Status`, `Result`, `Shape`, `DType` and `shape_of`, through `core/status.hpp`,
  `core/result.hpp`, `core/shape.hpp`, `core/common.hpp` and `core/tensor_shape.hpp`.
- `nn`: `NN::ModelState`, the state the checkpoint functions write and fill, through
  `nn/model_state.hpp`. `checkpoint.hpp` and `import_weights.hpp` include it;
  `checkpoint_format.hpp` includes only `core` headers.
- `config`: `Config::ModelSpec` and `Config::Value`, which `imported_model.hpp` holds, and
  `write_model`, `write_json` and `K_TRAIN_VERSION`, which `source/imported_model_writer.cpp`
  calls.
- `optim`: `Optim::Optimizer` and `Optim::RunState`, which `run_checkpoint.hpp` and
  `source/run_checkpoint.cpp` read and restore. No other header of the module includes it.
  `run_progress.hpp` includes `core/status.hpp` alone.

Two third-party libraries appear in the module without being named in the list. `checkpoint.hpp`
includes `SushiBLAS/engine.hpp`, and `checkpoint.hpp` and `import_weights.hpp` include
`SushiRuntime/core/span.hpp`. `source/import_weights.cpp` includes `SushiBLAS/tensor.hpp`. `core`
links SushiBLAS and SushiRuntime as external public dependencies, so both reach the module
through `core`.

## Public entry points

The headers sit in `modules/layers/io/include/SushiAI/io/`.

| Header | Declares |
| --- | --- |
| `checkpoint.hpp` | `save_checkpoint`, `load_checkpoint` |
| `checkpoint_format.hpp` | `K_CHECKPOINT_MAGIC`, `K_CHECKPOINT_FIRST_VERSION`, `K_CHECKPOINT_VERSION`, `K_MAX_TENSOR_COUNT`, `K_MAX_RUN_STATE_COUNT`, `K_MAX_NAME_LENGTH`, `TensorRecord`, `CheckpointWriter`, `CheckpointReader` |
| `run_checkpoint.hpp` | `save_run_checkpoint`, `load_run_checkpoint` |
| `run_progress.hpp` | `K_RUN_SEED_STATE`, `K_RUN_STEP_RUNS_STATE`, `K_RUN_EPOCH_STATE`, `K_RUN_BATCH_STATE`, `RunProgress`, `check_run_seed` |
| `import_weights.hpp` | `RenameRule`, `renamed`, `import_weights` |
| `imported_model.hpp` | `ImportedTensor`, `ImportedModel` |
| `importer_registry.hpp` | `ImporterFunction`, `ImporterEntry`, `ImporterRegistry`, `import_model` |
| `imported_model_writer.hpp` | `K_IMPORTED_MODEL_FILE`, `K_IMPORTED_WEIGHTS_FILE`, `K_IMPORTED_TRAINING_FILE`, `K_IMPORTED_CHECKPOINT_FILE`, `ExistingFiles`, `WrittenImport`, `compose_training_file`, `write_imported_model` |
