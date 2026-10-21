# Graph capabilities

**Status:** Open — phase 0 of 4; plan written and its decisions ruled, nothing implemented

## Problem

The changes below SushiAI's layers tier that lift the limits found while the language, dataset,
model-import and Darknet work was written in October 2026. The session that owns SushiRuntime
and SushiBLAS carries out the three pages for them; the SushiAI session carries out the graph page.

How to read a path: one that starts with `sushiblas/` or `sushiruntime/` is in that sibling
checkout, and every other path is in this repository. A path under a heading "Adds" does not
exist yet. Line numbers are those of the checkouts when the plan was written; their commits are in
`docs/agent/2026_10_21_GRAPH_CAPABILITIES_SOURCES/REPORT.md`.

Nothing in this plan was built or measured: builds are closed until SushiBLAS follows
SushiRuntime v0.4.0. Every cost stated here is read from the code or is an estimate, and each
estimate names the benchmark that would replace it.

## Decision

No item changes or removes an existing public declaration of SushiBLAS or SushiRuntime, so no
consumer, SushiEngine included, has to change to keep compiling. The author of this plan did
not read SushiEngine; that statement rests on the plan adding declarations only.

Ruled on 2026-10-11. The owner approved the new operation modules and delegated the rest to
the engineer of the SushiAI session, who took the recommendation in each case:

| # | Subject | Ruling | By |
| --- | --- | --- | --- |
| 1 | Two operation modules in SushiBLAS, `indexing` and `broadcast`, and one kernel module, `index_kernels` | As written. Not members on `SparseOps` and `ElementwiseOps`, whose contracts say the opposite of what the new operations do | Owner |
| 2 | An index out of range | Defined in every build type: zero on a gather, skipped on a transpose. SushiAI adds its own named error above it (A5) | Engineer |
| 3 | `BroadcastOps` and `ColumnMajor` | Refused in the first release. If SushiEngine needs it, the geometry rule gets a second arm | Engineer |
| 4 | Bilinear sampling | Half-pixel centres, integer scales. An `align_corners` form would be a further pair of operations, added when a model needs it | Engineer |
| 5 | The positioned random fill | `PositionedRandomOps` as its own view with one fill (Bernoulli); more distributions on evidence | Engineer |
| 6 | R1 and layer 2 of `GRAPH_REPLAY.md` | Open: the session that owns that design confirms the sentence or rewords it | That session |
| 7 | B6, the error-function GELU | Not now. It waits for a model trained with that form to be imported | Engineer |
| 8 | Releases | SushiBLAS tags a MINOR release at the end of phase 2 and of phase 3; SushiAI moves its pin in one `build(deps)` commit each time | Engineer |
| 9 | B7 and B8 | Inside the last build of the v0.4.0 migration: two test files and two paragraphs | Engineer |

## Design

### The items

Owners: **R** is SushiRuntime, **B** is SushiBLAS, **G** is SushiAI's graph
(the `core` tier: `modules/core/graph`, `modules/core/autograd` and the modules beneath them), **A** is SushiAI above
the graph. Size counts units, a unit being a file added or an existing file edited, tests
included: small is 1 to 3, medium 4 to 8, large more than 8.

| Id | Limit | Owner | Unblocks in SushiAI | Size | Depends on |
| --- | --- | --- | --- | --- | --- |
| R1 | Host-written memory read by a replayed plan is relied on and stated nowhere | R | G1, and what the gather offsets, the mask vectors and `StepScalars` already do | small | none |
| R2 | A run over several machines | R | Moved to [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION.md) | not sized | that design's phase 0 |
| B1 | A random fill fixes its seed and offset when recorded | B | Dropout (G1) | medium | R1 |
| B2 | Row gather goes through a sparse product: int32 `r * V + label`, no bounds rule, host-filled offsets | B | Losses over integer labels of any size (G5), the embedding | large | none |
| B3 | No broadcasting: operands of one elementwise call share one shape | B | Broadcast operations in the IR (G3), shorter special forms (G4) | large | none |
| B4 | Upsampling is nearest only | B | A bilinear segmentation head (G7) | medium | none |
| B5 | No index of the maximum along an axis | B | Segmentation metrics without `R * C` host reads (G10) | small | none |
| B6 | GELU is the tanh approximation only | B | A model trained with the error-function form | small | none |
| B7 | Tensor creation under concurrent calls is not stated or tested | B | The loader on threads (A8) | small | none |
| B8 | Host access to a Shared tensor between executions is not stated or tested | B | The same as R1, at the level SushiAI calls | small | R1 |
| B9 | No mask of the k largest of each row | B | The gate of a mixture in one node (G11) | small | none |
| B10 | No index of the rows a mask selects | B | An expert run on its own rows (G12) | medium | B2 |
| G1 | A lowering row cannot run host work before each replay; the IR has no random operation. Written with a host-drawn mask; the device fill waits on B1 | G | `NN::Dropout` (A11) | large | B1 |
| G2 | No square root, logarithm or power in the IR | G | Compositions that need them; a reciprocal root is a power | medium | none; now |
| G3 | `add`, `sub`, `mul`, `div` take equal shapes | G | The position table, a fed mask, the key and value cache's scores | medium | B3 |
| G4 | Special forms spell a broadcast with ones vectors, rank-4 views and `[N, C]` scratch | G | Fewer tasks in layer normalisation, log-softmax and its gradient | medium | B3, a measurement |
| G5 | A loss over labels builds `r * V + label`; `N * T * V` and `N * H * W * C` must fit in int32 | G | Next-token and per-pixel cross entropy at any batch size | large | B2 |
| G6 | `K_SCATTER_ADD_ROWS` has no backward rule | G | Dice without the `R * C` one-hot leaf | medium | none; now |
| G7 | `K_UPSAMPLE2D` is nearest | G | `SegmentHead` with class boundaries finer than its scale | medium | B4 |
| G8 | The masked score becomes minus infinity in half precision | G | Attention under `MixedPrecisionPolicy` | small | none; now |
| G9 | Whether a step made of recorders alone compiles is not pinned | G | The update step of gradient accumulation (A2) | small | none; now |
| G10 | No argmax in the IR | G | `measure_segmentation` reading `R` indices | medium | B5 |
| G11 | No top-k or row maximum in the IR; spelled as a pooling window, an upsample, a comparison and a select, 4 nodes a level | G | `MoEFeedForward`'s gate in one node | medium | B9 |
| G12 | The rows of an expert and their count come from the data, and a graph is traced for fixed shapes | G | An expert that runs on a bounded count of rows | large | B10, B2 |
| A1 | The key and value cache: one fed leaf, buffer updates in training only, one trace form | A (`train`, `nn`, `language/blocks`) | Written since, on a fed `[1, T]` mask leaf and without G3; a slide still rebuilds it | done | none |
| A2 | Every replay rewrites the gradients and the update sits behind backward | A (`optim`, `train`, `config`) | `subdivisions`, `max_batches` | medium | G9 |
| A3 | The optimizer's step count was not restorable | A (`optim`, `io`) | Written since, as the run-state section of version 2 of the weight file | done | none |
| A4 | The whole-document JSON parser needed about 8 GB for a 450 MB COCO file, by estimate | A (`config`, `vision/datasets`) | Written since, as `Config::read_json_events` | done | none |
| A5 | No check of a token id against the table before a step | A (`language/training`) | A named error where B2 gives a defined zero | small | none |
| A6 | `Dataset::read` has no epoch argument | A (`data`) | Window starts redrawn at each epoch | medium | none |
| A7 | A training file named one objective | A (`train`) | Written since, as the `sum` objective | done | none |
| A8 | The loader on threads | A (`data`) | Written since, as `Data::ThreadedLoader`; B7's test still confirms what it relies on | done | B7 |
| A9 | `ParameterInit` had no normal initialiser of fixed deviation | A (`nn`) | Written since, as `ParameterInit::normal`; `Embedding` starts at 0.02 | done | none |
| A10 | The clip squares each norm, so a norm above the square root of the largest float gives a factor of 0 | A (`optim`) | Clipping of an exploded gradient | small | none |
| A11 | `Dropout` passes its input through. Written | A (`nn`) | A model that states a rate | small | G1 |
| A12 | Objectives, heads and metrics still use the forms G5, G6, G7 and G10 replace | A (`language/training`, `vision/training`, `vision/blocks`) | The limits above, in the files a user trains from | medium | G5, G6, G7, G10 |

How the reported limits map to items: dropout is B1, G1 and A11. Broadcasting and the missing
math are B3, G2, G3 and G4. Bilinear upsampling is B4 and G7. The row gather's index width,
its bounds and its host-filled offsets are B2 and G5; no device iota is asked for, because the
offsets exist only to build the sparse matrix B2 replaces. The scatter-add rule is G6. The
`[N, C]` scratch of the log-softmax gradient goes with G4. The key and value cache is A1, with
G3 for its mask. Gradient accumulation is G9 and A2 and needs no kernel: `Level1::axpy` is the
accumulate kernel (`sushiblas/modules/operations/blas/include/SushiBLAS/blas/level1.hpp:47`).
The masked score is G8. Concurrent allocation is B7, and the code already answers it (section
4, B7). Host-filled scratch is R1 and B8. The step count is A3, the JSON parser A4, several
machines R2.

Found in the source documents beyond that list: B5 with G10
(`modules/vision/training/README.md:190-195`), B6 (`docs/design/LANGUAGE.md:79`), A6
(`docs/design/LANGUAGE.md:90-91`), A7 (`docs/design/DATASETS.md:159-161`), A9
(`docs/design/REMAINING_WORK.md:255-256`) and A10 (`modules/core/optim/README.md:185-186`).

Added on 2026-10-11 by the mixture of experts, after this plan was handed over, and not yet
sectioned below: B9 as `TaskHandle ReductionOps::top_k_mask_along(const Tensor& t, int64_t k,
Tensor& mask)` over a rank-2 `t`, the mask of `t`'s type holding 1 at the k largest of each
row, the lowest index winning a tie; B10 as `TaskHandle IndexOps::select_rows(const Tensor&
mask, Tensor& index, Tensor& count)`, `index` an INT32 `[C]` of the first C selected row
numbers ascending with the row count in unused slots. `ReductionOps::max_along` exists
(`sushiblas/modules/operations/reductions/include/SushiBLAS/reductions/reductions.hpp:158`);
a top-k, a sort and a compaction do not.

### Where the items are written

| Page | Holds |
| --- | --- |
| [RUNTIME.md](RUNTIME.md) | R1 and R2, SushiRuntime |
| [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md) | The rules common to every B item, B1 and B2 |
| [BLAS_BROADCAST_AND_OPERATIONS.md](BLAS_BROADCAST_AND_OPERATIONS.md) | B3 to B8 |
| [GRAPH.md](GRAPH.md) | G1 to G10, SushiAI's graph |

### What the plan leaves out

- **A design for several machines.** [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION.md) holds it.
- **Removing a special form.** `K_LAYER_NORM`, `K_CAUSAL_MASK`, `K_BIAS_ADD` and
  `K_LOG_SOFTMAX_BACKWARD` stay as nodes. G4 changes how they lower, row by row, and only on a
  measured gain. Whether a composition from G2 and G3 could replace a node is not claimed: it
  would add IR values the memory plan has to place, and nothing here measures that.
- **A fused row-wise log-softmax, a fused `K_NLL_LOSS`, bfloat16, skipping a whole step on
  overflow, feeding a batch without two copies, a contiguous `K_RESHAPE` as a view.** They are
  older entries of `docs/design/REMAINING_WORK.md` under "Memory planning", "Fusion and the
  IR" and "Convolutional networks", outside the sections this plan was asked to read. The
  batch copy grows with A1, which feeds three leaves.
- **A row write for the key and value cache.** B2 in [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md): the copy it would save is 98 304
  elements per buffer per token.
- **A parallel `add_rows`.** B2 in [BLAS_RANDOM_AND_INDEX.md](BLAS_RANDOM_AND_INDEX.md): gated on a profile.
- **Sine and cosine in the IR.** They serve rotary positions, which the owner has not chosen
  over the learned table (`docs/design/LANGUAGE.md:161-162`). SushiBLAS has both kernels; the
  work would be the shape of G2.
- **Restoring the random stream on resume.** After G1 a resumed run draws its masks from
  offset 0, because `RNGStream::reseed` rewinds the offset and nothing sets it
  (`sushiblas/modules/engine/engine/include/SushiBLAS/engine/rng_stream.hpp:57-62`).
  `sushiblas/modules/README.md`, under `RNGStream::reseed`, already names the shape such a
  method would take. It belongs with A3 and is raised when A3 is designed.
- **Top-k for the mixture of experts.** Phase 7 of `docs/design/LANGUAGE.md` is not started
  and no document states what it lacks.
- **Everything in the A rows** beyond the items table above. They are SushiAI's own and are
  planned in `docs/design/LANGUAGE.md`, `docs/design/DATASETS.md` and
  `docs/design/MODEL_IMPORT.md`.

## Phases

No item changes an existing signature of SushiBLAS or SushiRuntime, so every item of R and B
can be written beside the v0.4.0 migration of SushiBLAS and none edits a file the migration
moves, apart from the one-line registrations each section names. An item marked **free** can be
scheduled at any time around the migration; one marked **after** needs the migrated tree to
build before its tests can run.

| Id | Phase | State |
| --- | --- | --- |
| 1 | Items: R1 (free); B7, B8 (after); G2, G6, G8, G9 (written now). Closes with: `sr build` and `sr test` with the R1 test; `sb build` and `sb test --suite unit` with the B7 and B8 tests. SushiAI then checks: `sa build`, `sa test --suite unit`: the gradient checks of G2 and G6, the half-precision mask test of G8, the recorder-only step of G9 | planned |
| 2 | Items: B1, B4, B5, B6 (after); then G1, G7, G10, A11. Closes with: `sb build`, `sb test`, and a tagged SushiBLAS release. SushiAI then checks: The pin in `cli/sushistack.deps.toml` moves to that tag; a dropout step draws two different masks from one compile; a bilinear head passes the one-batch overfit test of segmentation; `measure_segmentation` equals its host form | planned |
| 3 | Items: B2, B3 (after); then G5, G3, G4, A12. Closes with: `sb build`, `sb test`, the two benchmarks of B2 and B3 run on the RTX 3080 Ti, and a tagged release. SushiAI then checks: The next-token and per-pixel losses and gradients equal their current values within the tolerances of their tests, with the int32 refusals removed; each re-lowering of G4 lands only with its before and after timing | planned |
| 4 | Items: R2. Closes with: Phases of [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION.md). SushiAI then checks: That design's Phases table | planned |

Phases 2 and 3 do not depend on each other and may swap. Inside a phase the B items are
independent of one another. Rule 4 of the `api-stability` skill holds: SushiAI consumes a phase
only from a SushiBLAS tag, never from a branch.

## Open questions

- Decision 6 above: the session that owns `GRAPH_REPLAY.md` confirms the sentence on R1 and layer 2, or rewords it.
