# yolo26: the YOLO26n detector as a model file

`model.json` describes YOLO26n in version 2 of the model file: 24 rows, one per row of the
Ultralytics `yolo26.yaml` at scale `n`, read by `NN::build_model`
(`include/SushiAI/nn/node_registry.hpp`). There is no C++ and no training configuration in
this directory.

The file does not build yet. Rows 10, 22 and 23 name blocks this build does not have, and
`build_model` stops at the first of them with a status that names the row. No command trains
or evaluates this model today, and no run has been made with it.

## What builds

| Rows | Type | State |
| --- | --- | --- |
| 0 to 9 | `conv`, `c3k2`, `sppf` | Build |
| 10 | `c2psa` | Fails: `layer 10 'c2psa'` has a type this build does not have |
| 11 to 21 | `upsample`, `concat`, `c3k2`, `conv` | Build once row 10 exists |
| 22 | `c3k2` with `"inner": "bottleneck_psa"` | Fails: `'bottleneck_psa'` names no inner block |
| 23 | `detect` | Fails: `layer 23 'detect'` has a type this build does not have |

The three rows wait for three emitters the graph builder lacks: a depthwise convolution, a
rank-3 matrix product and a differentiable softmax.
`tests/functional/unit/test_node_registry.cpp` pins the three messages. It also pins the
shapes below by building the file with row 10 replaced by a `silu` row, which keeps the shape
`c2psa` keeps.

## The rows

Activations are NHWC. The shapes leave out the batch axis and follow from a `[640, 640, 3]`
input.

| Row | Type | Reads | Output shape |
| --- | --- | --- | --- |
| 0 | `conv`, 3x3, stride 2 | input | `[320, 320, 16]` |
| 1 | `conv`, 3x3, stride 2 | 0 | `[160, 160, 32]` |
| 2 | `c3k2` over `bottleneck` | 1 | `[160, 160, 64]` |
| 3 | `conv`, 3x3, stride 2 | 2 | `[80, 80, 64]` |
| 4 | `c3k2` over `bottleneck` | 3 | `[80, 80, 128]` |
| 5 | `conv`, 3x3, stride 2 | 4 | `[40, 40, 128]` |
| 6 | `c3k2` over `c3k` | 5 | `[40, 40, 128]` |
| 7 | `conv`, 3x3, stride 2 | 6 | `[20, 20, 256]` |
| 8 | `c3k2` over `c3k` | 7 | `[20, 20, 256]` |
| 9 | `sppf`, three 5x5 pools | 8 | `[20, 20, 256]` |
| 10 | `c2psa` | 9 | `[20, 20, 256]` |
| 11 | `upsample`, scale 2 | 10 | `[40, 40, 256]` |
| 12 | `concat` | 11, 6 | `[40, 40, 384]` |
| 13 | `c3k2` over `c3k` | 12 | `[40, 40, 128]` |
| 14 | `upsample`, scale 2 | 13 | `[80, 80, 128]` |
| 15 | `concat` | 14, 4 | `[80, 80, 256]` |
| 16 | `c3k2` over `c3k` | 15 | `[80, 80, 64]` |
| 17 | `conv`, 3x3, stride 2 | 16 | `[40, 40, 64]` |
| 18 | `concat` | 17, 13 | `[40, 40, 192]` |
| 19 | `c3k2` over `c3k` | 18 | `[40, 40, 128]` |
| 20 | `conv`, 3x3, stride 2 | 19 | `[20, 20, 128]` |
| 21 | `concat` | 20, 10 | `[20, 20, 384]` |
| 22 | `c3k2` over `bottleneck_psa` | 21 | `[20, 20, 256]` |
| 23 | `detect`, 80 classes | 16, 19, 22 | twelve ports |

Rows 16, 19 and 22 are the three feature levels the head reads, at strides 8, 16 and 32.

## Where the numbers come from

Every `filters` and `hidden` value, and every `from` index, follows from the convolution
weight shapes in `tests/fixtures/yolo26/MANIFEST.json`, which a run of the pinned Ultralytics
commit produced. No weight shape carries `stride`, `shortcut`, `pools` or `heads`; those four
are read from the Ultralytics sources at commit `ce13994`, and no reference run has confirmed
them here.

`shortcut` is `true` on every `c3k2` row. `yolo26.yaml` writes rows 2 and 4 as
`C3k2, [256, False, 0.25]` and `C3k2, [512, False, 0.25]`: the second argument is `c3k` and the
third is the expansion `e`. `shortcut` is the last parameter of `C3k2.__init__` and defaults to
`True`, and no row of `yolo26.yaml` states it, so the one `Bottleneck(16, 16)` of row 2 and the one
`Bottleneck(32, 32)` of row 4 add their input.
`Unit_NodeRegistry.Yolo26RowsTwoAndFourAddTheirBottlenecksInput` pins the two additions.

`defaults` gives every block the batch-normalisation constants Ultralytics sets, epsilon 0.001
and momentum 0.03. A row that states either key keeps its own value.

A graph built from this file names its state by row index, so a checkpoint record is
`0.conv.weight` where the Ultralytics state dictionary has `model.0.conv.weight`.

The design is `docs/agent/2026_10_07_MODEL_GRAPH/SPEC.md`, sections 2.6 and 2.10.
