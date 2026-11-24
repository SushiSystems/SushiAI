# Model export: plan

**Status:** Open — written 2026-11-24; waits on the owner's rulings in SPEC section 9, and no task has started
**Designs:** MODEL_IMPORT

Spec: `SPEC.md` in this folder. The plan assumes the recommended answer to every question of
SPEC section 9; a different ruling changes the rows it names, listed under "Rulings that change
the plan". The owner picks inline or parallel.

No task builds or runs a test. Each task writes its code, its tests and its module README rows,
and lists the build lines it needs; the orchestrator holds every `CMakeLists.txt` and
`CHANGELOG.md`. One `sa build` and one `sa test` follow task 22, then
`python -m sushicore.tools check .`; an acceptance that names a test is met when that run
passes it.

## Routing

The owner asked for `sonnet` medium wherever the matrix of `model-routing` allows it. The matrix
gives `sonnet` medium to "one module, a local choice open, several files", and each task below
that fits the row takes it. A task that is spec-complete in one or two files fits the `haiku`
row; it is marked so, and moving it to `sonnet` medium is the owner's call, since it lies above
its row. Cross-module work takes `opus` low. No task takes `opus` high.

## Tasks

Hierarchy order: the build lines, then the shared interfaces in `io`, `nn` and `interchange`,
then the writers, then the exporters, then registration, the program and the command line.

| Id | Task | Module | Depends on | Acceptance | Model, effort |
| --- | --- | --- | --- | --- | --- |
| 1 | Build lines: tier row `vision/exporters vision` in `cmake/SushiAILayers.cmake`; `modules/vision/exporters/CMakeLists.txt` with SPEC section 6's dependencies and the sources of tasks 15 to 17; the new `io`, `interchange` and `nn` sources in their `CMakeLists.txt`; `tests/vision/exporters/CMakeLists.txt`; the `sushiai_export` target in `applications/programs/CMakeLists.txt` | orchestrator | Rulings 2 to 4 | `git diff --stat` lists exactly those files, and every source named in them is one a task below creates | orchestrator, `opus` medium |
| 2 | `ExportedFile`, `ExportedModel`, `ExporterFunction`, `ExporterEntry`, `ExporterRegistry`, `export_model` (`exported_model.hpp`, `exporter_registry.hpp`, sources, `test_exporter_registry.cpp`) | `io` | 1 | `test_exporter_registry.cpp` holds a case for an added row found by name, a second row of one format refused, and an unknown format refused with every format named | `haiku` medium (spec-complete, two units) |
| 3 | `write_exported_model` (`exported_model_writer.hpp`, source, `test_exported_model_writer.cpp`) through `source/committed_file.hpp` | `io` | 2 | The test holds cases for an existing file refused before any is written, `--force` replacing it, and two files of one name refused | `haiku` medium (spec-complete) |
| 4 | `read_model_files` (`model_files.hpp`, source, test) | `io` | 2 | The test reads a model file, a `.sushi` file and a training file the test writes into an `ImportedModel` whose `write_model` text and tensors equal the ones written | `sonnet` medium |
| 5 | `copy_kernel_first` and `copy_depthwise_kernel_first` in `filter_layout.hpp` and their tests | `io` | 1 | A test moves seeded kernels both ways and gets every element back for `[2, 3, 3, 4]` and `[3, 3, 5]` | `haiku` medium (spec-complete) |
| 6 | `StateEntry::layout` and its enumeration; `Conv2d` and `DepthwiseConv2d` set it; `nn` README rows | `nn` | Ruling 10 | A test lists the entries of a model of a `conv2d`, a depthwise `conv2d` and a `linear` and reads `KernelLast`, `DepthwiseKernelLast` and `Plain` | `sonnet` medium |
| 7 | `channels_first_axis`, `channels_first_permutation`, `filter_to_source_layout`, `flatten_weights_to_channels_first` in `axis_translation.hpp`; tests | `interchange` | 5 | Each function composed with its channel-last counterpart returns its input, tested over every axis of rank 4 and seeded weights | `sonnet` medium |
| 8 | `ProtobufWireWriter` (`protobuf_wire_writer.hpp`, source, test) | `io` | 1 | Every field kind the writer writes is read back by `ProtobufWire` with the same field number, wire type and value, including a nested message and a packed float field | `haiku` medium (spec-complete) |
| 9 | `write_protobuf_text` (`protobuf_text_writer.hpp`, source, test) | `io` | 1 | `parse_protobuf_text(write_protobuf_text(tree))` equals the tree for every `TextKind`, an escaped string and a nested message | `haiku` medium (spec-complete) |
| 10 | `ZipWriter` (`zip_writer.hpp`, source, test): stored entries, zip64, CRC-32, alignment | `io` | 1 | `ZipArchive` reads back every entry the writer wrote, its bytes and CRC-32 equal, and each aligned entry's data starts at a multiple of 64 | `sonnet` medium |
| 11 | `write_pickle` (`pickle_writer.hpp`, source, test) | `io` | 1 | `read_pickle(write_pickle(tree))` gives the same tree for each `PickleKind` the writer takes, and the writer refuses the rest | `sonnet` medium |
| 12 | `write_safetensors` (`safetensors_writer.hpp`, source, test) | `io` | 1 | `SafetensorsReader` reads back the name, type, shape and bytes of an `F32`, an `F64` and an `F16` tensor, and the data starts at a multiple of 8 | `haiku` medium (spec-complete) |
| 13 | `write_torch_state_dict` (`torch_writer.hpp`, source, test) | `io` | 10, 11 | `TorchWeightSource` reads back every tensor of a three-tensor state dictionary, at the root and under a `state_key` | `sonnet` medium |
| 14 | `export_state_weights` (`weight_export.hpp`, source, test): kernel layout by entry, reversed renames, drop notes | `io` | 5, 6, 12, 13 | SPEC section 7, item 4, passes for both formats, including a kernel whose two layouts have the same shape | `sonnet` medium |
| 15 | `write_onnx_graph` (`onnx_writer.hpp`, source, test) | `interchange` | 8 | `read_onnx_graph(write_onnx_graph(graph))` equals the graph for nodes of every attribute kind, initializers and a symbolic batch extent | `sonnet` medium |
| 16 | `write_caffe_prototxt` and `write_caffe_model` (`caffe_writer.hpp`, source, test) | `interchange` | 8, 9 | `read_caffe_prototxt` and `read_caffe_model` read back the net of `lenet.prototxt` and `lenet.caffemodel`, layers, parameters and blobs equal | `sonnet` medium |
| 17 | ONNX exporter: `onnx_export.hpp`, the row table and one file per row of SPEC section 3.1, `export_onnx_files`; `vision/exporters` README | `vision/exporters` | 2, 7, 15 | SPEC section 7, items 1 to 3, pass for ONNX | `sonnet` medium |
| 18 | Caffe exporter: `caffe_export.hpp`, the row table of SPEC section 3.2, `export_caffe_files`; README rows | `vision/exporters` | 2, 7, 16 | SPEC section 7, items 1 to 3, pass for Caffe | `sonnet` medium |
| 19 | Darknet exporter: `darknet_export.hpp`, the section writers of SPEC section 3.3, the `.cfg` and `.weights` writers, the `[yolo]` section from the task and the training file; README rows | `vision/exporters` | 2 | SPEC section 7, items 1 to 3, pass for Darknet, and the anchors of a written `.cfg` import to the same float32 fractions | `sonnet` medium |
| 20 | `register_exporters` in `vision/registration` and its test | `vision/registration` | 17, 18, 19 | The test finds the rows `onnx`, `caffe` and `darknet` in a registry the function filled | `haiku` medium (spec-complete) |
| 21 | `export.cpp`, `Registries::exporters`, the line in `domains_vision.cpp`, `compose_weight_writers`; `sa export` in `cli/sushiai/cli.py`, its row in `cli/tests/test_help_screen.py` and `cli/tests/test_export.py`; a program test of SPEC section 7, item 1, through the registry; the round-trip medium test of item 5 | `applications/programs`, `cli` | 3, 4, 14, 20 | The program test exports and re-imports one fixture per format through the registries, and `test_export.py` holds a case per usage line | `opus` low (crosses the program, the registries and the Python command) |
| 22 | `tools/reference/check_exported_files.py` and its row in `tools/reference/README.md` | `tools` | Ruling 11 | The script's `--help` names the three readers it needs, and it exits 1 naming any file a reader refuses | `haiku` medium (spec-complete) |
| 23 | One `sa build`, one `sa test`, `python -m sushicore.tools check .` | orchestrator | 1 to 22 | All three pass | orchestrator |
| 24 | Review: spec compliance and checker output | — | 23 | Every acceptance above is met, with the test names and output pasted | `haiku` high |
| 25 | Review: SOLID shape, public surface and humanizer register, as named items | — | 23 | Each public header of SPEC section 6 has one reason to change, each row file one row, and each README and comment reads in the repository's register | `opus` low |
| 26 | The owner's run of the script of task 22 on an exported YOLOv3-tiny (Darknet), LeNet (Caffe) and `silu_resize` (ONNX) | owner | 22, 23 | The script reports every file read and the ONNX output equal to the one `sushiai inference` gives | owner |

Tasks 2 to 16 touch different files and can run in parallel within each depth; tasks 17 to 19
can run in parallel once 15 and 16 are in. No file belongs to two tasks: the `io` README is
written by task 2 and each later `io` task hands its rows back in its report, which the
orchestrator merges; the same holds for `interchange` (task 7 writes, 15 and 16 hand back) and
`vision/exporters` (task 17 writes, 18 and 19 hand back).

## Rulings that change the plan

| Ruling | Change |
| --- | --- |
| 1, a new topic | Task 1 runs `sa design new MODEL_EXPORT` in place of a page and phases |
| 2, no new module | Tasks 17 to 19 move into `vision/importers`, and task 1 edits that module's build file |
| 4, a separate module | Tasks 2 to 5 and 8 to 14 move to that module, and task 1 adds it |
| 5, a new type | Task 2 adds `IO::ModelFiles`, and tasks 4, 14 and 17 to 19 take it |
| 7, a wider export | Tasks 17 to 19 gain rows with no round trip, and SPEC section 7 needs another proof for them |
| 10, no layout field | Task 6 leaves the plan; task 14 writes SushiAI's layout and section 7, item 4, changes |
| 11, no script | Tasks 22 and 26 leave the plan |

## What the orchestrator writes

### Before task 1

```
sa design page MODEL_IMPORT EXPORT
sa design phase MODEL_IMPORT E1 "The export door and the weight writers: sushiai export weights to safetensors and PyTorch"
sa design phase MODEL_IMPORT E2 "The ONNX exporter"
sa design phase MODEL_IMPORT E3 "The Caffe exporter"
sa design phase MODEL_IMPORT E4 "The Darknet exporter"
```

The arguments follow `sa design page --help` (topic, page) and `sa design phase --help` (topic,
id, text). The topic numbers its phases 1 to 9; `E1` to `E4` keep the export phases apart, and
`10` to `13` serve as well if the owner prefers numbers. The work folder's `**Designs:**` lines then become
`**Implements:** MODEL_IMPORT#E1, MODEL_IMPORT#E2, MODEL_IMPORT#E3, MODEL_IMPORT#E4`.

### Backlog entries

```
sa backlog add work --area MODEL_IMPORT --priority P3 --title "Export a SushiAI model to ONNX, Caffe, Darknet, safetensors and PyTorch files" --done-when "Each format round-trips its fixture through sushiai export and sushiai import to the same model text and tensors" --source docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md
sa backlog add work --area MODEL_IMPORT --priority P2 --title "List the modules of the MODEL_IMPORT topic" --done-when "sa design owner returns MODEL_IMPORT for io, interchange, vision/importers and vision/exporters" --source docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md
sa backlog add work --area MODEL_IMPORT --priority P3 --title "Read and write leaky_relu, mish, avgpool2d, global_avgpool, maximum, dropout, tanh and gelu in ONNX" --done-when "Each of the eight rows round-trips through the ONNX exporter and importer" --source docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md
sa backlog add work --decision --area MODEL_IMPORT --priority P3 --title "Rule on lowering composite vision rows for ONNX and Caffe export" --done-when "The owner rules whether vision/blocks gains a lowering per block, and in which phase" --source docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md
```

The first is the entry the folder's commits name. The owner sets its priority; P3 is the
improvement band of the `backlog` skill. The second answers the `none` that `sa design owner`
returned for the import modules.

### Changelog lines, written at close

```
- <date> — model_import: Added `sa export onnx`, `sa export caffe` and `sa export darknet`, which write a model file and its weights as another framework's files and refuse each row that format's importer cannot read back (<ID>).
- <date> — model_import: Added `sa export weights safetensors` and `sa export weights torch`, which write the records of a `.sushi` file with convolution kernels in PyTorch's layout (<ID>).
- <date> — io: Added the writers of protobuf wire, protobuf text, zip, pickle, safetensors and PyTorch state dictionaries, and `ExporterRegistry`, `read_model_files` and `write_exported_model` (<ID>).
- <date> — interchange: Added `write_onnx_graph`, `write_caffe_prototxt`, `write_caffe_model` and the channel-first counterparts of the axis rewrites (<ID>).
- <date> — nn: Added `StateEntry::layout`, which `Conv2d` and `DepthwiseConv2d` set to their kernel layout (<ID>).
```

### Pages at close

SPEC section 8 lists them. The topic's Decision section takes the rulings of SPEC section 9 in
the same commit.
