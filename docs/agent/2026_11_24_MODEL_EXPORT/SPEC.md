# Model export: specification

**Status:** Open — written 2026-11-24; section 9 ruled 2026-11-24; plan tasks 2 to 16 implemented, tasks 17 to 22 open
**Designs:** MODEL_IMPORT

`sushiai import` turns another framework's files into `model.json`, `weights.sushi` and
`train.json`. This page specifies the other direction: `sushiai export` writes a SushiAI model
as the files of ONNX, Caffe, Darknet, safetensors and PyTorch. Every boundary, public interface
and dependency named here is a proposal until the owner rules on section 9; the plan is
`PLAN.md` in this folder.

## 1. What this rests on

| Fact | Where |
| --- | --- |
| An importer is a function from paths to an `IO::ImportedModel`: a `Config::ModelSpec`, the weights as `ImportedTensor` (a `TensorRecord` and its host bytes), a training recipe, a `data` object and notes | `modules/layers/io/include/SushiAI/io/imported_model.hpp`, `importer_registry.hpp` |
| `ImporterRegistry` holds one row per format; `Vision::register_importers` adds `darknet`, `ultralytics`, `onnx` and `caffe` | `modules/vision/registration/source/registration.cpp:106-116` |
| `import.cpp` reads the command line, looks the format up and hands the result to `IO::write_imported_model`; it names no format | `applications/programs/source/import.cpp:319-344` |
| `sa import` forwards its arguments to `sushiai_import` | `cli/sushiai/cli.py:350-368` |
| `write_model` writes a `ModelSpec` back as model-file text, and `parse(write(spec)) == spec` is phase 1 of MODEL_IMPORT | `docs/guides/MODEL_FILE.md:7-8`; `sa design show MODEL_IMPORT` |
| `interchange` holds `OnnxGraph`, `CaffeNet`, `TensorBytes` and the NCHW-to-NHWC rewrites; each unit returns values and none writes a file | `modules/layers/interchange/README.md`, "Owns" |
| `io` holds the readers `ProtobufWire`, `parse_protobuf_text`, `ZipArchive`, `read_pickle`, `TorchArchive`, `SafetensorsReader`, the kernel moves `copy_kernel_last` and `copy_depthwise_kernel_last`, and `import_state_weights` | `modules/layers/io/README.md` |
| No writer exists for protobuf wire, protobuf text, zip, pickle, safetensors, a Darknet `.cfg` or `.weights` | Every public header listed in the io, interchange and importers READMEs |
| Each importer refuses an operator it has no row for and names every such node in one message | `modules/vision/importers/README.md`, "Refusals", "The Caffe importer", "The ONNX importer" |
| The ONNX importer needs opset 13 or newer and one float32 input `[N, C, H, W]` with C, H and W stated | `modules/vision/importers/README.md:243-244` |
| `NN::StateEntry` holds name, kind, state id, shape, dtype, init and fans; it carries no weight layout | `modules/layers/nn/include/SushiAI/nn/model_state.hpp:64-89` |
| `import_state_weights` moves a rank-4 `[out, in, kh, kw]` tensor to the record's `[out, kh, kw, in]` when the permutation fits, the permutation winning when both fit, and `[C, 1, kh, kw]` to `[kh, kw, C]` | `modules/layers/io/README.md:274-279` |
| `sa design owner` returns `none` for `modules/layers/io`, `modules/layers/interchange`, `modules/vision/importers`, `modules/vision/registration` and `modules/core/config`, and `LAYERED_EXECUTION` for `applications/programs`; MODEL_IMPORT lists `**Modules:** none` | `sa design owner <path>` and `sa design show MODEL_IMPORT`, run 2026-11-24 |
| Fixtures exist for ONNX (`silu_resize.onnx`) and Caffe (`lenet.prototxt`, `lenet.caffemodel`), with the Python scripts that wrote them | `tests/vision/importers/fixtures/`, `tools/reference/make_onnx_fixture.py`, `make_caffe_lenet_fixture.py` |

## 2. Formats

| Target | Feasible | Writes | Needs |
| --- | --- | --- | --- |
| ONNX | Yes | `<stem>.onnx`, opset 13 | A protobuf wire writer and a serialiser of `OnnxGraph` |
| Caffe | Yes | `<stem>.prototxt`, and `<stem>.caffemodel` with weights | A protobuf text writer, the wire writer and a serialiser of `CaffeNet` |
| Darknet | Yes, for the rows of section 3.3 | `<stem>.cfg`, and `<stem>.weights` with weights | A `.cfg` text writer and a `.weights` writer |
| safetensors | Yes, weights only | One `.safetensors` file | A safetensors writer and the layout of each record (section 4) |
| PyTorch state dict | Yes, weights only | One `.pth` file that `torch.load(weights_only=True)` reads | A zip writer (stored entries), a pickle writer and the layout of each record |
| Ultralytics `.pt` | No; question 9 | — | A pickle of Ultralytics' own module objects, tied to the class layout of each Ultralytics version |

A graph format exports the rows; a weight format exports the records. The two kinds enter
by two commands, as they do for import.

The rule of section 3 is that an exporter writes only what the importer of the same format
reads back. Every exported file then round-trips through SushiAI's own importer, and the round
trip is the test (section 7). This rule is question 7.

## 3. What each graph format expresses

Every exporter walks the rows in order, looks each `type` up in its own row table and refuses
the model with `Unsupported` when a row has no entry or states a key its entry cannot write. As
the importers do, it does not stop at the first: it collects every refused row and names each
as `row <i> '<type>': <reason>` in one message. A model with an `inputs` object, an `int32`
input, or an input whose rank the format cannot state is refused before any row.

A row table has one file per row type, `source/<format>_row_<type>.cpp`, and one row in
`source/<format>_row_table.cpp`. A new row type is a new file and a new row.

The values a format cannot carry and that do not change the computation (the `task` section
for ONNX and Caffe, `input.fit` other than Darknet's, `input.normalise`, `train.json`) are left
out with one note each, as an import notes what it drops. Whether they are notes or refusals is
question 14.

### 3.1 ONNX

The graph input is `input`, float32 `[batch, C, H, W]` with `batch` symbolic, from
`input.shape` `[H, W, C]`. An input of rank other than 3 is refused. Weights are initializers
named `<row>.<field>`, the names the importer gives back. Graph outputs take the names the
model publishes.

| Row | ONNX | Refused when |
| --- | --- | --- |
| `conv2d` | `Conv`: weight `[out, kh, kw, in / g]` back to `[out, in / g, kh, kw]`, depthwise `[kh, kw, C]` to `[C, 1, kh, kw]`; `pads` `[top, left, top + trailing_h, left + trailing_w]`; `group`; bias initializer when the row has one | A dilation other than 1 |
| `batchnorm2d` | `BatchNormalization`, `epsilon` from `bn_epsilon`, the four vectors | — |
| `relu`, `sigmoid` | `Relu`, `Sigmoid` | — |
| `silu` | `Sigmoid` and `Mul` of it with its input, the pair the importer fuses back into one `silu` row | — |
| `add`, `sub`, `mul`, `div` | `Add`, `Sub`, `Mul`, `Div` in operand order | The second operand is a broadcast prefix; the importer reads equal extents only |
| `softmax` | `Softmax` with `axis` moved to NCHW | — |
| `transpose` | `Transpose` with `perm` moved to NCHW, batch axis first | — |
| `concat` | `Concat`, `axis` 1 | — |
| `maxpool2d` | `MaxPool`, `ceil_mode` 0; `padding` and `trailing_padding`, or `leading_padding` and `end_padding`, become `pads` | — |
| `upsample` | `Resize`, mode `nearest`, `scales` `[1, 1, s, s]` | `mode` `bilinear` |
| `flatten` | `Flatten`, `axis` 1, and every `linear` that reads it gets its weight columns moved from `(H, W, C)` to `(C, H, W)` | The flatten is published or read by a row other than `linear`: ONNX's element order differs from SushiAI's |
| `slice_channels` | `Slice` on axis 1 | — |
| `linear` | `Gemm`, `transB` 1, constant weight and bias | The input is not a vector |

Refused, because the ONNX importer has no row for them: `leaky_relu`, `mish`, `gelu`, `tanh`,
`avgpool2d`, `global_avgpool`, `maximum`, `dropout`, `unflatten`, `embedding`, `layernorm`,
the vision rows `conv`, `bottleneck`, `c3`, `c3k`, `c3k2`, `c2f`, `sppf`, `c2psa`, `focus`,
`detect`, `decoupled_head`, `segment_head`, and every language row. A model imported from
Ultralytics is made of vision rows, so it is refused whole until question 8 is ruled.

ONNX needs its weights in the file: an export without `weights.sushi` is refused.

### 3.2 Caffe

The input is an `Input` layer of shape `[1, C, H, W]` from `[H, W, C]`, or `[1, K]` from
`[K]`. Layers are named `row_<i>` and each writes the blob of its name; the published outputs
must be exactly the blobs no later layer reads, since that is what the importer publishes, and
a model that publishes another set is refused.

| Row | Caffe | Refused when |
| --- | --- | --- |
| `conv2d` | `Convolution`: `num_output`, `kernel_h`, `kernel_w`, `stride`, `pad`, `group`, `bias_term`; weight back to `[out, in / g, kh, kw]` | `trailing_padding` above 0; Caffe pads both sides alike |
| `maxpool2d` | `Pooling` `MAX` | The row's end padding differs from what Caffe's rounding up adds |
| `avgpool2d` | `Pooling` `AVE` | The window count needs rounding up; Caffe divides the last window by a smaller area |
| `global_avgpool` | `Pooling` `AVE`, `global_pooling` true | — |
| `relu`, `leaky_relu` | `ReLU`, with `negative_slope` for `leaky_relu` | — |
| `flatten` then `linear` | One `InnerProduct`; the flatten writes no layer, and the weight columns move from `(H, W, C)` to `(C, H, W)` | The flatten is published or read by a row other than `linear` |
| `linear` over a vector | `InnerProduct` | — |
| `concat` | `Concat`, `axis` 1 | — |
| `add`, `mul`, `maximum` | `Eltwise` `SUM`, `PROD`, `MAX` over two bottoms | — |
| `softmax` | `Softmax`, `axis` moved to NCHW | The batch axis |
| `dropout` | `Dropout`, `dropout_ratio` from `rate` | — |
| `batchnorm2d` | `BatchNorm` (`eps`, blobs mean, variance and a factor of 1) and a `Scale` with `bias_term` (blobs weight and bias) on its top | — |

Refused, because the Caffe importer has no row for them: `sigmoid`, `silu`, `mish`, `gelu`,
`tanh`, `upsample`, `slice_channels`, `sub`, `div`, `transpose`, `unflatten`, `embedding`,
`layernorm`, every vision row and every language row.

### 3.3 Darknet

`[net]` takes `width`, `height` and `channels` from `[H, W, C]`, and `letter_box` 1 for the fit
`darknet_letterbox` and 0 for `resize`; the fit `letterbox` is refused. With `--training`
(section 5), the solver keys of `[net]` and the training keys of each `[yolo]` come from
`train.json` through `read_head_training_section` and the step schedule; without it the `.cfg`
holds the architecture alone and the export notes it.

| Rows | Section | Refused when |
| --- | --- | --- |
| `conv2d`, then an optional `batchnorm2d` read only by it, then an optional activation row read only by those | `[convolutional]`: `filters`, `size`, `stride`, `pad` 1 when the padding is `size / 2` and `padding` otherwise, `groups`, `batch_normalize`, `activation` | `trailing_padding` above 0; `bn_epsilon` other than 1e-5; a `batchnorm2d` on a convolution with a bias |
| Activation after a convolution | `leaky` (slope 0.1), `mish`, `swish` for `silu`, `logistic` for `sigmoid`, `linear` for none | `leaky_relu` of another slope, `relu`, or an activation row after anything but a convolution |
| `maxpool2d` | `[maxpool]` `size`, `stride` and the `padding` total the importer halves | End padding that the importer's rule does not give back |
| `upsample` | `[upsample]` `stride` | `mode` `bilinear` |
| `concat` | `[route]` `layers` | — |
| `slice_channels` | `[route]` with `groups` and `group_id` | A slice that is not group `group_id` of `groups` equal parts |
| `add` | `[shortcut]` `from`, `activation` linear | — |
| The rows an `anchor_detection` task names as heads | `[yolo]` after the head's section: `mask`, `anchors` (each fraction times `width` or `height`), `classes`, `num`, `scale_x_y`, `nms_kind` and `beta_nms` from the suppression | A task of another type, or a head that `read_anchor_task` reads but `[yolo]` cannot state |

Every other row is refused. A model with no task is refused before any row, as `Unsupported`
with "no task".
Ruled: a model with no task is refused, because the Darknet importer refuses a `.cfg` without
a `[yolo]` section, so no such file comes back through a round trip.

The written anchors must give back the same float32 fractions: each is the shortest decimal
that the importer's division turns into the stored fraction, bit for bit.

`.weights` holds the header major 0, minor 2, revision 0 and a 64-bit `seen` of 0, then for
each `[convolutional]` in file order: the biases (the batch-norm shift when normalised), then
for a normalised one the scales, the running means and the running variances, then the weights
in `[out, in / g, kh, kw]`. That is the order `read_darknet_weights` reads.

## 4. Weight formats

`sushiai export weights` writes the state records of `weights.sushi`, named by their record
names, in the element type of the record: `F32` for float32, `F64` for float64, `F16` for half.
A record of another type is refused.

A convolution kernel goes back to PyTorch's layout: `[out, kh, kw, in]` to `[out, in, kh, kw]`
and a depthwise `[kh, kw, C]` to `[C, 1, kh, kw]`. Every other record keeps its layout.
`import_state_weights` then moves each kernel back, since the permutation wins when both
layouts fit, so the round trip holds for every shape.

Which records are kernels is not in the records: `NN::StateEntry` has no layout and a rank-3
record need not be a depthwise kernel. Question 10 proposes a `layout` field that `Conv2d` and
`DepthwiseConv2d` set when they declare the weight.

`--map <map.json>` reads the same `NameMap` that `import weights` takes and applies its `rename`
rules from the new prefix to the old one. A `drop` entry names tensors the file had and SushiAI
has not, so it cannot be written back; each one is a note. `state_dict` names the key the
dictionary is written under, and the root when empty.

- **safetensors**: the 8-byte little-endian header length, the JSON header with `dtype`,
  `shape` and `data_offsets` per tensor in record order and `__metadata__` `{"format": "pt"}`,
  padded with spaces to a multiple of 8, then the data with no gap.
- **PyTorch**: a zip of stored entries under the folder `<stem>/`: `data.pkl`, a protocol 2
  pickle of an `OrderedDict` of `torch._utils._rebuild_tensor_v2` calls over persistent ids
  `('storage', torch.FloatStorage, '<k>', 'cpu', <elements>)`; `data/<k>`, the bytes of each
  tensor, aligned to 64 bytes as `torch.save` aligns them; `version` `3`; and `byteorder`
  `little`. Every name the pickle states is one `TorchArchive` admits, so the import reads it.

## 5. The commands

```
sushiai export <format> <model.json> [<weights.sushi>] --output <directory>
               [--training <train.json>] [--name <stem>] [--force]
sushiai export weights <safetensors|torch> <model.json> <weights.sushi> --output <file>
               [--map <map.json>] [--force]
```

`sa export` forwards to `sushiai_export`, as `sa import` forwards to `sushiai_import`. The first
form looks `<format>` up in an `IO::ExporterRegistry`; `onnx`, `caffe` and `darknet` are rows
`Vision::register_exporters` adds, and a format with no row is refused with the formats that
have one. `--name` sets the stem of the written files, `model` when absent. The second form is
chosen by `weights` as the first argument, as `import weights` is; no exporter row may take
that name. It needs the model file to find the kernel records (section 4).

The command checks every file it would write before it writes one, refuses a file that exists
unless `--force` is given, and writes each through `<name>.partial` and a rename, as
`write_imported_model` does. It prints the format, the inputs, the files written and every note,
and creates no engine.

## 6. Where the code lives (proposed)

| Unit | Module | Public header |
| --- | --- | --- |
| `ProtobufWireWriter`: varint, zigzag, fixed32, fixed64, length-delimited and packed fields, nested messages | `io` | `protobuf_wire_writer.hpp` |
| `write_protobuf_text`, the inverse of `parse_protobuf_text` over `TextField` | `io` | `protobuf_text_writer.hpp` |
| `ZipWriter`: stored entries, zip64 past 4 GiB, CRC-32 through `source/crc32.hpp`, an alignment per entry | `io` | `zip_writer.hpp` |
| `write_pickle`: protocol 2 from a `PickleValue` tree of the kinds `read_pickle` returns | `io` | `pickle_writer.hpp` |
| `write_safetensors`, `write_torch_state_dict` | `io` | `safetensors_writer.hpp`, `torch_writer.hpp` |
| `copy_kernel_first`, `copy_depthwise_kernel_first`, the inverses of the two kernel moves | `io` | `filter_layout.hpp` |
| `ExportedFile` (a name and bytes), `ExportedModel` (the files and the notes) | `io` | `exported_model.hpp` |
| `ExporterFunction`, `ExporterEntry`, `ExporterRegistry`, `export_model` | `io` | `exporter_registry.hpp` |
| `read_model_files`: model file, optional `.sushi`, optional training file into an `ImportedModel` | `io` | `model_files.hpp` |
| `write_exported_model`: the checks and the committed writes of section 5 | `io` | `exported_model_writer.hpp` |
| `export_state_weights`: records, entries and a map into the tensors of section 4 | `io` | `weight_export.hpp` |
| `write_onnx_graph`: an `OnnxGraph` to the bytes of a `ModelProto` | `interchange` | `onnx_writer.hpp` |
| `write_caffe_prototxt`, `write_caffe_model` | `interchange` | `caffe_writer.hpp` |
| `channels_first_axis`, `channels_first_permutation`, `filter_to_source_layout`, `flatten_weights_to_channels_first` | `interchange` | `axis_translation.hpp` |
| `StateEntry::layout` and its enumeration | `nn` | `model_state.hpp` |
| ONNX, Caffe and Darknet exporters with their row tables, `K_*_EXPORT_FORMAT` | `vision/exporters` (new) | `onnx_export.hpp`, `caffe_export.hpp`, `darknet_export.hpp` |
| `register_exporters` | `vision/registration` | `registration.hpp` |
| `export.cpp`, `Registries::exporters`, `compose_weight_writers` | `applications/programs` | — |
| `sa export` | `cli/sushiai/cli.py` | — |

`ExporterFunction` is `Result<ExportedModel> (*)(const ImportedModel& model, std::string_view
stem)`. An exporter returns bytes and writes no file, so every test runs in memory.

`vision/exporters` declares `PUBLIC_DEPENDS failure io interchange`,
`PRIVATE_DEPENDS config dtype optim vision/detect` and `EXTERNAL_PUBLIC SushiRuntime::core`
(`modules/vision/exporters/CMakeLists.txt`); it does not depend on `vision/importers`. Its
Darknet writer states the `.cfg` keys itself, and the format names are its own constants with
the same text as the importers' (`"onnx"`, `"caffe"`, `"darknet"`).

## 7. The round trip

Each exporter task carries its own tests, in `tests/vision/exporters/` or the module's test
folder, and every one runs in memory.

1. **Fixed point.** For each format: import a fixture (`silu_resize.onnx`; `lenet.prototxt`
   with `lenet.caffemodel`; a `.cfg` and a `.weights` the test writes, as
   `test_darknet_import` does), export it, import the bytes back. The two models give the same
   `write_model` text and every tensor is equal bit for bit. A second export gives the same
   bytes as the first.
2. **One model per row.** For each row of a table in section 3, a `ModelSpec` built in code
   with seeded weights goes through export and import, and the text and the tensors come back
   equal.
3. **Refusals.** A model with two refused rows fails with `Unsupported`, and the message names
   both rows by index and type.
4. **Weights.** A model of a `conv2d`, a depthwise `conv2d`, a `batchnorm2d` and a `linear`, with
   seeded weights, exported to safetensors and to PyTorch and read back by
   `import_state_weights`, gives every record bit for bit, including a kernel whose two layouts
   have the same shape.
5. **Same output.** A medium test builds the fixture model and its round-tripped copy on the
   test engine and compares their outputs for one seeded input bit for bit; the rows are the
   same, so the kernels are.
6. **Outside readers.** A script under `tools/reference/` loads the exported `.onnx` with
   `onnx.checker` and `onnx.reference.ReferenceEvaluator`, the `.pth` with
   `torch.load(weights_only=True)` and the `.safetensors` with `safetensors`, and compares the
   ONNX output with the one `sushiai inference` gives. It is run by hand and not in CI
   (question 11).

## 8. Pages this work changes

`applications/programs/README.md` (the file and command tables, lines 16-35, and a new
`export.cpp` section before its "Designs" heading), `docs/guides/CLI_GUIDE.md` line 71 and section
"`sushiai train`, `inference`, `import` and `tokenize`", `docs/guides/RUNNING_MODELS.md` (its
import section), the READMEs of `io`, `interchange`, `nn`, `vision/registration` and the new
`vision/exporters`, the MODEL_IMPORT topic, and `docs/reference/CHANGELOG.md`.
`docs/guides/MODEL_FILE.md` stays: the model file does not change.

## 9. Questions for the owner

Each is the owner's decision. The plan cannot start before 1 to 8, 10 and 12 are ruled. The
owner handed the rulings to the implementing agent on 2026-11-24; each ruling follows its
question.

1. **Design home.** Recommended: a page `EXPORT.md` and phases E1 to E4 in MODEL_IMPORT
   (`sa design page`, `sa design phase`), since export is the other half of the same door.
   The topic should also list its modules: today `sa design owner` returns `none` for `io`,
   `interchange` and `vision/importers`. The alternative is a new topic, MODEL_EXPORT.
   Ruled: a page `EXPORT.md` and phases E1 to E4 in MODEL_IMPORT, because the importer and the
   exporter of a format share the row tables' subject and the round-trip test.
2. **A new module `modules/vision/exporters`**, tier `vision`, with the dependencies of
   section 6. Recommended. It mirrors `vision/importers`, which keeps the operator tables in
   the domain that owns the rows, and keeps the importers module at one reason to change.
   Ruled: yes, `modules/vision/exporters` with the dependencies of section 6, because
   `vision/importers` changes when a foreign file is read differently and the exporters change
   when one is written differently.
3. **`interchange` serialises.** Recommended: it gains `write_onnx_graph` and the Caffe writers,
   which return bytes or text and still write no file, so its rule holds; its README's "Owns"
   says "read" today.
   Ruled: yes, because the writers return bytes and text and the module's rule, no file
   written, still holds.
4. **The io writers, door types and loader** of section 6, each beside its reader.
   Recommended. A separate `layers/export` module is the alternative; it would split each
   container's reader from its writer.
   Ruled: in `io`, each writer beside its reader, because a change to a container format then
   touches one module.
5. **The exporter's input type.** Recommended: `IO::ImportedModel`, which already holds a model
   in memory, with `read_model_files` filling it. The alternative is a new `IO::ModelFiles` of
   the same fields, which reads better and duplicates the type.
   Ruled: `IO::ImportedModel`, because its fields are exactly the three files and a second type
   of the same fields would have two reasons to drift.
6. **No external dependency.** Recommended: the ruling of 2026-10-12 (own readers, no library)
   extends to the writers, so no protobuf, ONNX or zip library is linked.
   Ruled: no external dependency, because the readers are already the repository's own and each
   writer is tested against its reader.
7. **Export only what the importer reads back.** Recommended, because every exported file is
   then tested by a round trip. The cost: ONNX refuses `leaky_relu`, `mish`, `avgpool2d`,
   `global_avgpool`, `maximum`, `dropout`, `tanh` and `gelu` until each gains an importer row
   and an exporter row together.
   Ruled: export only what the importer reads back, because a row with no round trip has no
   test in this repository.
8. **Composite vision rows** in ONNX and Caffe. Recommended: refused in E1 to E4. A later
   phase gives each block a public function that lowers it to primitive rows, owned by
   `vision/blocks`, and the exporters call it. Until then no Ultralytics-imported model exports
   to ONNX.
   Ruled: refused in E1 to E4; the lowering is a later decision entry, because it adds a public
   function to every block.
9. **Ultralytics `.pt`.** Recommended: not written; the PyTorch state dict is offered instead.
   Ruled: not written, because the file pickles Ultralytics' module classes, which change with
   each Ultralytics version.
10. **The layout of a record.** Recommended: `NN::StateEntry` gains `layout` (`Plain`,
    `KernelLast`, `DepthwiseKernelLast`), set by `Conv2d` and `DepthwiseConv2d`, a public change
    in `nn`. The alternatives: write SushiAI's layout as it is, which PyTorch cannot load into
    a convolution, or guess from the rank.
    Ruled: `NN::StateLayout` with `Plain`, `KernelLast` and `DepthwiseKernelLast`, and
    `StateEntry::layout`. `describe_parameter` builds an entry from a `Parameter` alone
    (`modules/layers/nn/source/model_state.cpp:50-62`), so `Parameter` carries the layout too,
    through `set_layout`, which `Conv2d` and `DepthwiseConv2d` call after they declare the
    weight. A rank guess fails on a `[kh, kw, C]` record of another meaning.
11. **The outside-reader script** of section 7, item 6, run by hand with `onnx`, `torch` and
    `safetensors` installed outside the repository. Recommended.
    Ruled: yes, by hand and not in CI, because CI installs none of the three packages.
12. **The command line** of section 5, including `export weights` and `--name`. Recommended.
    Ruled: the command line of section 5, because it mirrors `import` and `import weights`.
13. **`num_batches_tracked`.** PyTorch's batch normalisation has it and SushiAI has no record
    for it. Recommended: not written, with a note that `load_state_dict(strict=False)` loads the
    file; writing zeros would state a count no run made.
    Ruled: not written, with the note, because a zero count is a value no run produced.
14. **Lost values.** Recommended: the task section, a fit or normalisation the format cannot
    state, and the training file become notes, and the export goes on. Refusing them would make
    ONNX and Caffe refuse every detector.
    Ruled: notes, and the export goes on, because none of these values changes what the
    exported graph computes.
