# Export

**Status:** Phase E1 building, E2 to E4 planned; the rulings of 2026-11-24 are recorded in
section 9 of `docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md`

`sushiai import` turns another framework's files into `model.json`, `weights.sushi` and
`train.json`. Export is the other direction through the same door: `sushiai export` writes a
SushiAI model as the files of ONNX, Caffe, Darknet, safetensors and a PyTorch state dictionary.
The full specification, with the row tables of each format, is
`docs/agent/2026_11_24_MODEL_EXPORT/SPEC.md`; its plan is `PLAN.md` in the same folder.

## The rule

An exporter writes only what the importer of the same format reads back, so every exported file
is tested by importing it again. A row the format's importer has no entry for is refused, and the
refusal names every such row of the model in one message, as the importers name every operator
they lack. Values a format cannot carry and that do not change the computation (the task section
in ONNX and Caffe, an input fit or normalisation the format cannot state, the training file) are
left out with one note each.

## The two commands

```
sushiai export <format> <model.json> [<weights.sushi>] --output <directory>
               [--training <train.json>] [--name <stem>] [--force]
sushiai export weights <safetensors|torch> <model.json> <weights.sushi> --output <file>
               [--map <map.json>] [--force]
```

A graph format exports the rows; a weight format exports the records. `<format>` is looked up in
an `IO::ExporterRegistry` whose rows a domain adds, as `ImporterRegistry` holds the importers. An
exporter returns bytes and writes no file; the command checks every file before writing one and
writes each through `<name>.partial` and a rename, as `write_imported_model` does.

## Rulings

Ruled on 2026-11-24, by the owner through the implementing agent; the reasons are in SPEC
section 9.

| Subject | Ruling |
| --- | --- |
| Design home | This page and phases E1 to E4 of MODEL_IMPORT |
| Dependencies | None outside the repository; each writer is tested against its own reader |
| What is exported | Only what the importer of the same format reads back |
| Composite vision rows | Refused in ONNX and Caffe in E1 to E4; a lowering per block is a later decision |
| Ultralytics `.pt` | Not written; the PyTorch state dictionary is offered instead |
| `num_batches_tracked` | Not written, with a note |
| Values a format cannot state | A note each, and the export goes on |
| The outside-reader script | Run by hand, not in CI |

## Where the code goes

| Part | Module |
| --- | --- |
| Protobuf wire and text writers, zip and pickle writers, safetensors and PyTorch writers, the inverse kernel moves, the exporter door and the model-file loader | `modules/layers/io`, each writer beside its reader |
| `write_onnx_graph`, the Caffe writers, the channel-first axis rewrites | `modules/layers/interchange` |
| The ONNX, Caffe and Darknet exporters and their row tables | `modules/vision/exporters`, a new module that mirrors `vision/importers` |
| `export.cpp` and `sa export` | `applications/programs`, `cli/sushiai` |

A record carries no layout today, so a weight export cannot tell a kernel from another rank-4
record. The ruling adds `NN::StateLayout` (`Plain`, `KernelLast`, `DepthwiseKernelLast`) to
`StateEntry` and to `Parameter`, set by `Conv2d` and `DepthwiseConv2d` when they declare the
weight.

## Phases

| Id | Delivers | Proven by |
| --- | --- | --- |
| E1 | The export door and the weight writers: `sushiai export weights` to safetensors and PyTorch | A model of a convolution, a depthwise convolution, a normalisation and a linear row read back by `import_state_weights` bit for bit |
| E2 | The ONNX exporter | `silu_resize.onnx` imported, exported and imported again to the same model text and tensors |
| E3 | The Caffe exporter | The LeNet fixture through the same round trip |
| E4 | The Darknet exporter | A `.cfg` and `.weights` written by the test through the same round trip |

When `modules/vision/exporters` exists, it joins the topic's `**Modules:**` line.
