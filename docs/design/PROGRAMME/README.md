# SushiAI programme

**Status:** Open — stage 1 of 7 built and passing; the naming half of stage 2 is done
**About:** What SushiAI is being built towards, in what order, and the owner's rulings that fix the shape of the work.
**Modules:** none

## Problem

What SushiAI is being built towards, in what order, and the owner's rulings that fix the shape of
the work. Open items are in [`REMAINING_WORK.md`](../../backlog/README.md); each stage's design is in the
work folder named beside it. This document replaces `ROADMAP.md`, which planned the
construction of SushiBLAS and SushiAI up to the MLP showcase and is kept as the record of that
period.

SushiAI trains Ultralytics YOLO26 from scratch and runs its forward pass, on the stack's own
kernels, and runs the Darknet-era YOLO models. The first tag, 0.1.0, is cut when that works. Along the way the library becomes both
compiled and eager: a model is traced once and replayed, or traced on every step with plain C++
control flow, over one IR, one rule table and one lowering.

## Decision

Each was made by the owner on 2026-10-07.

| Subject | Ruling |
| --- | --- |
| Front ends | Compiled and eager. The eager front end runs on SushiRuntime's `DynamicGraph`, cut into regions where the caller opens a scope |
| Eager surface | It must not break after it ships. What a caller writes (`Session`, `Step`, `Scope`) is the public surface; the units behind it are not promised |
| Python front end | Not now |
| First tag | 0.1.0, when the YOLO26 programme and the Darknet-era models are finished |
| Error contract | Status values everywhere: every operation that can fail returns a status, and no exception crosses the public surface. A wrong model is a status whether it comes from JSON or from C++; a programmer's misuse is fatal. Gradient rules stay `void` and `differentiate()` returns the failure |
| Naming | The `cpp-code-style` rule for the whole library, applied in one wave: `K_` constants, PascalCase enumerators, no `I` prefix; SushiBLAS v0.1.0 is asked to rename its `DataType`, `Layout` and `ExecutionMode` enumerators the same way |
| Builds | The owner builds and tests, once the programme's code is written. Until then work is checked by reading and by single-file syntax checks |
| Sibling repositories | SushiRuntime and SushiBLAS become public; CI access follows from that and is not this programme's work |
| YOLO26 scope | Training from scratch and the forward pass. Depthwise convolution in SushiBLAS. Mixed precision in the first run, parity checks in fp32 |
| YOLO26 loss | Fixed-shape and masked, with a selection guarding unassigned rows |
| YOLO26 label assignment | On the host, in two passes |
| YOLO26 validation | Both heads: one-to-one with top-k, and one-to-many with NMS |
| YOLO26 data | The default augmentation pipeline in full; images decoded once into a raw cache by a separate tool with libjpeg-turbo |
| YOLO26 references | Produced by running the pinned Ultralytics commit, never stated |
| Darknet-era YOLO | Ruled on 2026-10-09: YOLOv3, YOLOv4, YOLOv7 and their tiny variants run and train on the library, YOLOv3-tiny first, through one anchor-based detection unit; they are inside the feature freeze of 0.1.0. YOLOv1 and YOLOv2 are outside it. References come from AlexeyAB's Darknet, built outside the repository for that purpose and deleted afterwards |
| Domain modules | Ruled on 2026-10-09: a field of use is a top folder and its own library target; vision is the first; layers stay in `nn`. The design is [`DOMAIN_MODULES.md`](../DOMAIN_MODULES/README.md) |
| Model import | Ruled on 2026-10-11: every foreign format (Darknet, later ONNX, Caffe, Ultralytics) enters through one import step that writes a model file in JSON and a `.sushi` weight file; training and inference read those alone. `.sushi` is the official weight extension. The design is [`MODEL_IMPORT.md`](../MODEL_IMPORT/README.md) |
| Datasets | Ruled on 2026-10-11: a dataset is a task and a format reader; `Data::Batch` breaks to carry roles; classification, binary, multi-label, detection, semantic segmentation and regression are inside the feature freeze of 0.1.0; libjpeg-turbo and libpng are dependencies. The design is [`DATASETS.md`](../DATASETS/README.md) |
| Language | Ruled on 2026-10-11: a language domain is inside the feature freeze of 0.1.0. A decoder-only transformer trains on the library; the reference for parity is a nanoGPT-scale PyTorch model run outside the repository. Training across several machines through SushiRuntime's distributed layer is wanted; ruled on 2026-10-19 to go through SushiBLAS, designed in [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION/README.md) |
| Batch normalisation | The batch variance is an output of `K_BATCHNORM2D`; the running statistics are updated beside the graph |

## Design

### Order of the stages

Stages 1 and 2 touch the same files and are planned as one pass over the library. Stages 3 to 5
each have units that call no engine and are written ahead of stage 1; their lowerings are written
once, after it.

### Feature freeze of 0.1.0

The first tag holds what stages 1 to 9 name and nothing else: the compiled and the eager front
end, the status contract, convolutional networks from a model file, YOLO26 training and forward
pass, the Darknet-era YOLO models through the anchor-based detection unit, the import step
of stage 7, the datasets and tasks of stage 8 and the language domain of stage 9. A feature outside
this list waits for a later version; the owner adds to the list, as he did on 2026-10-09 for
stage 6 and on 2026-10-11 for stages 7 to 9.

## Phases

| Id | Phase | State |
| --- | --- | --- |
| 0 | The owner's build. Ends when: The tree builds, the suites pass on a named device, and the "No kernel named" failure in the test logs is explained. Design: `REMAINING_WORK.md`, first section | planned |
| 1 | SushiBLAS v0.1.0 migration. Ends when: Every call site uses the recorder, the lowering is per-domain tables, `TensorPool` has its own header. Design: sushiblas: `docs/agent/2026_10_07_V010/` | built |
| 2 | Error contract and naming. Ends when: No exception crosses the public surface; one naming form. Design: `docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md` | building |
| 3 | Convolutional networks. Ends when: A CNN trains on MNIST from a JSON file; batch normalisation, multi-input layers and a rank-4 input exist. Design: `docs/agent/2026_10_07_CV_CLOSE/SPEC.md`, the model graph spec in the version history | building |
| 4 | Eager front end. Ends when: A model traced under scopes trains to the same loss as the compiled path, and a changed scope rebuilds only its own regions. Design: `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` | building |
| 5 | YOLO26. Ends when: Stages a to e of its map: body forward, imported-weight parity, loss parity, overfit, full run. Design: `docs/agent/2026_10_07_YOLO26/SPEC.md` | building |
| 6 | Darknet-era YOLO. Ends when: YOLOv3-tiny runs its forward pass on imported Darknet weights and equals a reference, then trains; v3, v4-tiny, v4, v7-tiny and v7 follow through the same anchor unit. Design: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md` and `PLAN.md` | building |
| 7 | Model import. Ends when: A Darknet model is imported once and its six parity suites pass from the imported files. Design: [`MODEL_IMPORT.md`](../MODEL_IMPORT/README.md) | building |
| 8 | Datasets and tasks. Ends when: A YOLOv7-tiny trains from a model file, a `.sushi` file and a folder of images; a binary and a segmentation model each overfit one batch. Design: [`DATASETS.md`](../DATASETS/README.md) | planned |
| 9 | Language. Ends when: A small decoder-only transformer equals its reference on one forward pass and one step, then trains on one machine; the multi-machine run follows [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION/README.md). Design: [`LANGUAGE.md`](../LANGUAGE/README.md) | planned |
| 10 | Release 0.1.0. Ends when: The readiness list is closed. Design: the release readiness report in the version history | planned |

### Work

<!-- work:begin -->
No open work.
<!-- work:end -->

## Open questions

None. The open items are in [`REMAINING_WORK.md`](../../backlog/README.md).
