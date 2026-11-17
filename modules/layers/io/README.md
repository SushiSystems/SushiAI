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
which the two writers and the three loaders share. The private header `source/committed_file.hpp`
holds the write every file of the module goes through: the bytes go to `<name>.partial` and the
file takes its name by a rename only when the whole write succeeded, so a save that fails or is
refused leaves a file already of that name as it was and no partial file behind.

The module also owns the door a model of another framework enters through,
`docs/design/MODEL_IMPORT/README.md`. `ImportedModel` is what an importer returns: a
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
their order. `data` is `ImportedModel::data`, which holds the pipeline an importer states and is null
otherwise, because no foreign file names a dataset; `Config::parse_train` refuses a `data` block
without a reader, so a person fills the block before a run. A recipe that states one of
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
| 4 | The element type, uint32: the value of its `DType` enumerator, 0 to 6; the count type `UInt32` (7) is not carried, the writer refuses it and the reader refuses it as "does not carry" |
| 4 | K, the rank, uint32, at most `K_MAX_RANK` |
| 8 x K | The extents, int64 each, every one positive |
| Elements x element size | The payload |

| Version | Shipped in | Layout |
| --- | --- | --- |
| 1 | Before 0.1.0, untagged | A header of 12 bytes, magic, version and M, then M records |
| 2 | v0.1.0 | The table above: R joins the header, and the run-state section follows the model records. The four progress records joined the section before version 2 shipped, so the number did not change |

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

`CMakeLists.txt` lists `PUBLIC_DEPENDS config core nn optim`. The module uses these four modules:

- `failure`, `shape` and `dtype`: `Status`, `Result`, `Shape`, `DType` and `shape_of`, through
  `failure/status.hpp`, `failure/result.hpp`, `shape/shape.hpp`, `dtype/dtype.hpp` and
  `shape/tensor_shape.hpp`.
- `nn`: `NN::ModelState`, the state the checkpoint functions write and fill, through
  `nn/model_state.hpp`. `checkpoint.hpp` and `import_weights.hpp` include it;
  `checkpoint_format.hpp` includes only `failure`, `dtype`, `shape` and `seam` headers. Of
  the core modules, `seam` is private (`PRIVATE_DEPENDS`).
- `config`: `Config::ModelSpec` and `Config::Value`, which `imported_model.hpp` holds, and
  `write_model`, `write_json` and `K_TRAIN_VERSION`, which `source/imported_model_writer.cpp`
  calls.
- `optim`: `Optim::Optimizer` and `Optim::RunState`, which `run_checkpoint.hpp` and
  `source/run_checkpoint.cpp` read and restore. No other header of the module includes it.
  `run_progress.hpp` includes `failure/status.hpp` alone.

Two third-party libraries appear in the module without being named in the list. `checkpoint.hpp`
includes `SushiBLAS/tensor/tensor.hpp`, and `checkpoint.hpp` and `import_weights.hpp` include
`SushiRuntime/core/span.hpp`. `source/import_weights.cpp` includes `SushiBLAS/tensor/tensor.hpp`.
The module's dependencies link SushiBLAS and SushiRuntime as external public dependencies, so
both reach the module through them.

`import_weights` and `load_checkpoint` also take the bytes of a checkpoint and a name for them,
`origin`, which a failure message uses where the path form uses the path. `MemoryInput` is the
forward-only stream they read through; it copies nothing.

The module also reads PyTorch files, with no dependency outside the repository. `ZipArchive` lists
the entries of a zip archive held in memory, plain or zip64, and hands out their bytes. A stored
entry (method 0) is a view of the archive's bytes. A deflated entry (method 8) is decoded by
`inflate` when `read` runs, and `contents` returns the archive's own copy of the decoded bytes; its
size is checked against the entry's stated size and its CRC-32 against the archive's, and
`ZipEntry::deflated` marks it. `read` refuses with `InvalidFile` a deflated entry whose stream is
corrupt, whose decoded size or CRC-32 differs from the archive's, or whose stated size is more than
1032 times its compressed length, each naming the entry; an entry of any other method is refused
with `Unsupported`, naming the entry and the method. The CRC-32 of a stored entry is not checked.
`read_pickle` turns a pickle stream
of protocol 2 to 5 into a tree of `PickleValue` nodes and calls nothing the stream names: a global
is admitted only when the caller's list names it, whole or by a module prefix ending in `.`, and a
call becomes a `Call` node that holds its arguments and what the stream builds onto it.
`pickle_to_value` converts the plain data of such a tree into a `Config::Value`. `TorchArchive`
joins the two for a file `torch.save` wrote: it finds the `data.pkl` entry of its folder, refuses a byte
order other than little-endian, admits PyTorch's own rebuilders and storages beside the caller's
globals, and reads a tensor of half, float32 or float64 elements, at any stride, as row-major
float32. `copy_kernel_last` and `copy_depthwise_kernel_last` move a convolution weight from
`[out, in, kh, kw]` to `[out, kh, kw, in]` and a depthwise one from `[C, 1, kh, kw]` to
`[kh, kw, C]`, for every importer. The design is `docs/design/MODEL_IMPORT/ULTRALYTICS.md`.

`SafetensorsReader` reads a safetensors file held in memory: the 8-byte little-endian length of
the header, the JSON header, then the data. It lists each tensor, in the order of its bytes, as a
`SafetensorsTensor`: name, `SafetensorsType` (F16, BF16, F32 or F64; bfloat16 is no `DType`, so the
type is its own enumeration), shape, the offset of its first byte from the start of the file and its
length in bytes. It copies no data; `contents` returns a view of the caller's bytes, and
`read_float32` widens a tensor to float32 the way `TorchArchive::read_float32` does. The header's
`__metadata__` entry is skipped. `read` refuses with `InvalidFile` a file with no length field, a
header longer than the file or no JSON object, a tensor that lacks `dtype`, `shape` or
`data_offsets`, an extent that is not positive, a range that runs backwards, ends outside the data
or does not hold the bytes its shape needs, and two ranges that overlap; it refuses with
`Unsupported` a dtype other than the four above. Each message names the tensor, or the header. A
gap between two ranges is accepted. The private header `source/float_widening.hpp` holds the exact
widening of binary16 and bfloat16 that both readers share.

`inflate` decodes a raw DEFLATE stream, RFC 1951, into an output the caller sized: stored, fixed
Huffman and dynamic Huffman blocks, up to the block that states it is the last, and it returns the
bytes written. It decodes bit by bit with canonical Huffman tables built per block and allocates
nothing. It refuses with `InvalidFile`, naming the stream and the bit offset where the fault is
found: a block of the reserved type 3, a stored block whose length and check disagree, a code that
has more codes than its lengths allow or leaves bit patterns unused, a repeat that has no previous
length or runs past the lengths, a dynamic block with no end-of-block code, a symbol or distance
the tables do not hold, a match that reaches back before the start of the output, input that ends
inside a block, and a block that would write past the end of the output. The private header
`source/crc32.hpp` holds the CRC-32 that `ZipArchive` checks.

`NpyReader` reads one NumPy `.npy` file held in memory, format version 1.0, 2.0 or 3.0. It parses
the header, a Python dictionary literal of `descr`, `fortran_order` and `shape`, and returns an
`NpyArray` through `array`: the `NpyType`, the `Shape`, `fortran_order`, and the offset and length
of the data in the caller's bytes, copying none of them; `contents` returns the data. The types are
the little-endian `<f2`, `<f4` and `<f8` and the integers `|i1`, `|u1`, `<i2`, `<u2`, `<i4`, `<u4`,
`<i8` and `<u8`. `read_float32` returns the elements as row-major float32, widening half exactly,
rounding float64 and the integers to the nearest float32 and transposing a column-major array.
`read` refuses with `Unsupported`, naming the descriptor, an object array (`|O`), a big-endian
array, a structured type and any other type; with `Unsupported` an extent of 0, an extent or element
count above 2^40, a rank above `K_MAX_RANK` and a format version other than 1.0, 2.0 and 3.0; and
with `InvalidFile` a file without the magic, a header that is cut short or is not a dictionary of
exactly those three keys, each once, and data shorter than the shape needs. Bytes after the data
are allowed. Like the other readers, it copies elements with `memcpy` and so assumes a
little-endian host. The private header `source/float_widening.hpp` supplies the half widening.

Two readers know no message schema, ONNX's or Caffe's: the importers that need them bring their
own tables. `ProtobufWire` reads the binary wire format forward over a byte view and copies
nothing. `next_field` returns a field number and a `WireType`; `read_varint`, `read_zigzag`,
`read_fixed32`, `read_fixed64` and `read_length_delimited` read the value, and `float_from_bits`
and `double_from_bits` give a fixed value its float. `message` returns a reader over a
length-delimited value, and a packed repeated field is read through it until `at_end`. `skip`
passes a field of any wire type, a group through its end tag by counting depth. Every refusal is
`InvalidFile` and names the byte offset from the start of the outermost bytes, also inside a
sub-message: a truncated varint, a varint above 64 bits, a fixed value or a length that runs past
the end, a field number of 0 or above 2^29 - 1, a wire type of 6 or 7, a group with no end tag and
an end tag with no group. `parse_protobuf_text` reads the text format of a Caffe prototxt into a
tree of `TextField`: `name: value` and `name { ... }`, the colon before a brace optional, a value
a `Number` (as written, with the type left to the schema), a `String` with its C, octal, hex and
`\u` escapes decoded and adjacent strings joined, or an `Identifier` such as an enum value; `#`
comments; `;` or `,` after a field; a repeated field as repeated entries in document order, found
with `find_field` and `find_fields`. Every field records the line and column of its name. A list in
brackets, an extension name, the angle-bracket form of a message and nesting past 100 braces are
refused, as are a stray or unclosed brace, a bad token, a malformed number or string and a
document that ends after a field name, each as `InvalidFile` naming the line and column. The design
is `docs/design/MODEL_IMPORT/README.md`.

A weight file of another framework enters through `WeightSource`, one interface and one
implementation per container. A source lists its tensors as `SourceTensor` (name, shape, the stored
type as the file spells it) and reads any of them as float32 by index; it owns the bytes it was
built from. `TorchWeightSource` reads the state dictionary of a PyTorch file through `TorchArchive`:
the `state_key` option names the dictionary inside the file (`model` for a ByteTrack checkpoint),
empty takes the root, and an entry that is no tensor is refused with a message that names the key
to state. `SafetensorsWeightSource` reads a safetensors file through `SafetensorsReader` and ignores
`state_key`. `NpzWeightSource` reads a `.npz` file, stored or deflated, through `ZipArchive` and
`NpyReader`: one tensor per `<name>.npy` entry, named without the extension, and `state_key` is
ignored. Its matcher accepts a zip only when it holds entries and every one ends in `.npy`, so a
PyTorch zip (it holds `data.pkl`) is not taken; both are zips, so a caller adds the npz row before
the torch row. Each container ships a matcher over the file's first bytes (a zip magic; an 8-byte
length that fits the file followed by `{`), an opener and a `WeightSourceEntry` constant
(`K_TORCH_WEIGHT_SOURCE_ROW`, `K_SAFETENSORS_WEIGHT_SOURCE_ROW`, `K_NPZ_WEIGHT_SOURCE_ROW`). `WeightSourceRegistry` holds the
rows in the order the caller adds them; `open_weight_source` opens the bytes through the first row
whose matcher accepts them and, when none does, names the origin and every format the registry
holds; `open_weight_file` reads a path first. The module adds no row to a registry: the caller
does, so a new container adds its own files and one `add` and edits no source of the module.

`NameMap` reads the JSON map beside a model file: `state_dict`, the key of the state dictionary;
`rename`, prefix rules in the shape of `RenameRule`; and `drop`, names the model has no record for.
A drop entry matches a tensor whose name is the entry or ends in a dot and the entry, read against
the name as the file states it. `import_state_weights` matches a source against the state records
of a model (`NN::StateEntry`, which `ModelState` lists, so no engine is needed): drops, then renames,
then looks each tensor up. It returns one `ImportedTensor` per record in the model's order, in
the record's element type (float32 or float64), and it fails with `Incompatible` in one message
that groups every record the file does not supply, every tensor no record takes (and every one of
two tensors that rename to one record) and every tensor whose shape differs from its record's.
A rank-4 tensor [out, in, kh, kw] whose permutation [out, kh, kw, in] is the record's shape is a
PyTorch convolution weight and is moved to the record's kernel-last layout with `copy_kernel_last`
(the permutation wins when both layouts fit). A rank-4 tensor [C, 1, kh, kw] whose record is the
rank-3 [kh, kw, C] is a depthwise weight and goes through `copy_depthwise_kernel_last`; shapes are
compared after it. Every other tensor is
copied in the layout the file stores.
`write_imported_weights` writes such tensors as one `.sushi` file, through the checks and the
committed write the three-file writer uses.

## Public entry points

The headers sit in `modules/layers/io/include/SushiAI/io/`.

| Header | Declares |
| --- | --- |
| `checkpoint.hpp` | `save_checkpoint`, `load_checkpoint` (a path, or bytes in memory) |
| `checkpoint_format.hpp` | `K_CHECKPOINT_MAGIC`, `K_CHECKPOINT_FIRST_VERSION`, `K_CHECKPOINT_VERSION`, `K_MAX_TENSOR_COUNT`, `K_MAX_RUN_STATE_COUNT`, `K_MAX_NAME_LENGTH`, `TensorRecord`, `CheckpointWriter`, `CheckpointReader` |
| `run_checkpoint.hpp` | `save_run_checkpoint`, `load_run_checkpoint` |
| `run_progress.hpp` | `K_RUN_SEED_STATE`, `K_RUN_STEP_RUNS_STATE`, `K_RUN_EPOCH_STATE`, `K_RUN_BATCH_STATE`, `RunProgress`, `check_run_seed` |
| `import_weights.hpp` | `RenameRule`, `renamed`, `import_weights` (a path, or bytes in memory) |
| `memory_input.hpp` | `MemoryInput` |
| `zip_archive.hpp` | `ZipEntry`, `ZipArchive` |
| `inflate.hpp` | `inflate` |
| `npy_reader.hpp` | `NpyType`, `npy_type_size`, `npy_type_name`, `NpyArray`, `NpyReader` |
| `pickle_reader.hpp` | `PickleKind`, `PickleValue`, `PickleRef`, `is_call_of`, `find_entry`, `read_pickle` |
| `pickle_data.hpp` | `pickle_to_value` |
| `torch_archive.hpp` | `TorchTensor`, `TorchArchive` |
| `safetensors_reader.hpp` | `SafetensorsType`, `safetensors_type_size`, `safetensors_type_name`, `SafetensorsTensor`, `SafetensorsReader` |
| `protobuf_wire.hpp` | `WireType`, `WireField`, `ProtobufWire`, `float_from_bits`, `double_from_bits` |
| `protobuf_text.hpp` | `TextKind`, `TextField`, `find_field`, `find_fields`, `parse_protobuf_text` |
| `weight_source.hpp` | `SourceTensor`, `WeightSource`, `WeightSourceOptions`, `WeightSourceMatcher`, `WeightSourceOpener`, `WeightSourceEntry`, `WeightSourceRegistry`, `open_weight_source`, `open_weight_file` |
| `torch_weight_source.hpp` | `is_torch_file`, `open_torch_source`, `K_TORCH_WEIGHT_SOURCE_ROW` |
| `npz_weight_source.hpp` | `is_npz_file`, `open_npz_source`, `K_NPZ_WEIGHT_SOURCE_ROW` |
| `safetensors_weight_source.hpp` | `is_safetensors_file`, `open_safetensors_source`, `K_SAFETENSORS_WEIGHT_SOURCE_ROW` |
| `name_map.hpp` | `NameMap` |
| `weight_import.hpp` | `import_state_weights` |
| `filter_layout.hpp` | `FilterExtents`, `copy_kernel_last`, `copy_depthwise_kernel_last` |
| `imported_model.hpp` | `ImportedTensor`, `ImportedModel` |
| `importer_registry.hpp` | `ImporterFunction`, `ImporterEntry`, `ImporterRegistry`, `import_model` |
| `in_memory_importer.hpp` | `count_files_importer`, `refuse_no_file_importer`, `K_COUNTING_IMPORTER_ROW`, `K_REFUSING_IMPORTER_ROW` |
| `imported_model_writer.hpp` | `K_IMPORTED_MODEL_FILE`, `K_IMPORTED_WEIGHTS_FILE`, `K_IMPORTED_TRAINING_FILE`, `K_IMPORTED_CHECKPOINT_FILE`, `ExistingFiles`, `WrittenImport`, `compose_training_file`, `check_imported_tensors`, `write_imported_weights`, `write_imported_model` |
