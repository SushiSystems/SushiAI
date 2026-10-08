# SushiAI programme

**Status:** Open — stage 1 of 7 built and passing; the naming half of stage 2 is done

What SushiAI is being built towards, in what order, and the owner's rulings that fix the shape of
the work. Open items are in [`REMAINING_WORK.md`](REMAINING_WORK.md); each stage's design is in the
work folder named beside it. This document replaces [`ROADMAP.md`](ROADMAP.md), which planned the
construction of SushiBLAS and SushiAI up to the MLP showcase and is kept as the record of that
period.

## Target

SushiAI trains Ultralytics YOLO26 from scratch and runs its forward pass, on the stack's own
kernels, and runs the Darknet-era YOLO models. The first tag, 0.1.0, is cut when that works. Along the way the library becomes both
compiled and eager: a model is traced once and replayed, or traced on every step with plain C++
control flow, over one IR, one rule table and one lowering.

## Rulings

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
| Darknet-era YOLO | Ruled on 2026-10-09: the Darknet-era models run on the library, YOLOv3-tiny first, through one anchor-based detection unit; they are inside the feature freeze of 0.1.0 |
| Batch normalisation | The batch variance is an output of `K_BATCHNORM2D`; the running statistics are updated beside the graph |

## Stages

| # | Stage | Ends when | Design |
| --- | --- | --- | --- |
| 0 | The owner's build | The tree builds, the suites pass on a named device, and the "No kernel named" failure in the test logs is explained | `REMAINING_WORK.md`, first section |
| 1 | SushiBLAS v0.1.0 migration | Every call site uses the recorder, the lowering is per-domain tables, `TensorPool` has its own header | sushiblas: `docs/agent/2026_10_07_V010/` |
| 2 | Error contract and naming | No exception crosses the public surface; one naming form | `docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md` |
| 3 | Convolutional networks | A CNN trains on MNIST from a JSON file; batch normalisation, multi-input layers and a rank-4 input exist | `docs/agent/2026_10_07_CV_CLOSE/SPEC.md`, `docs/agent/2026_10_07_MODEL_GRAPH/SPEC.md` |
| 4 | Eager front end | A model traced under scopes trains to the same loss as the compiled path, and a changed scope rebuilds only its own regions | `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md` |
| 5 | YOLO26 | Stages a to e of its map: body forward, imported-weight parity, loss parity, overfit, full run | `docs/agent/2026_10_07_YOLO26/SPEC.md` |
| 6 | Darknet-era YOLO | YOLOv3-tiny runs its forward pass on imported Darknet weights and equals a reference; the other Darknet-era models follow through the same anchor unit | Not written; `REMAINING_WORK.md`, section "Darknet-era YOLO" |
| 7 | Release 0.1.0 | The readiness list is closed | `docs/agent/2026_10_07_RELEASE_READINESS/REPORT.md` |

Stages 1 and 2 touch the same files and are planned as one pass over the library. Stages 3 to 5
each have units that call no engine and are written ahead of stage 1; their lowerings are written
once, after it.

## Feature freeze of 0.1.0

The first tag holds what stages 1 to 6 name and nothing else: the compiled and the eager front
end, the status contract, convolutional networks from a model file, YOLO26 training and forward
pass, and the Darknet-era YOLO models through the anchor-based detection unit. A feature outside
this list waits for a later version; the owner adds to the list, as he did on 2026-10-09 for
stage 6.
