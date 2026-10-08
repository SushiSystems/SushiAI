# Remaining work

The single backlog of this repository. Work and decisions that are still open are listed here;
a defect that exists today is in [`KNOWN_ISSUES.md`](../reference/KNOWN_ISSUES.md). The design
behind the library items is in [`ROADMAP.md`](ROADMAP.md) and
[`ARCHITECTURE.md`](../architecture/ARCHITECTURE.md).

## What the owner's build must confirm

Builds and test runs have been off since 2026-10-07, so everything written since then was checked
by reading and by single-file syntax checks only. The first build has to settle:

- **The suites pass.** Done on 2026-10-09: the tree built with no error and all 1803 tests
  passed on an RTX 3080 Ti; `docs/agent/2026_10_08_SUSHIBLAS_MIGRATION/REPORT.md` has the runs.
- **The `No kernel named` failure.** The test logs of 2026-09-16 and later show nearly every test
  failing with a SYCL "No kernel named" error; no document names the cause (stale binary, device
  selection or kernel bundle).
- **The new unit suites**: the eager units, provenance, the eight layers, the layer registry and
  the changed layer stack, module, config and spatial shape cases.
- **The pinned references.** SushiBLAS reseeds by rewinding its counter and draws four values per
  Philox block, so `test_learning_curve.cpp`, `test_determinism.cpp` and the demo transcript may
  have moved.
- **A convolution step on a device**, and the first `sa train examples/mnist_cnn/train.json`. The
  list is in `docs/agent/2026_10_07_CV_CLOSE/SPEC.md` section 4.

## SushiBLAS v0.1.0 migration

SushiBLAS removes the engine facades, makes every operation take a `TaskRecorder` and removes
`Engine::get_graph()`. Its design is in the sushiblas repository under
`docs/agent/2026_10_07_V010/`.

The library, the applications, the examples and the tests were moved on 2026-10-08, unbuilt. The
design is `docs/agent/2026_10_08_SUSHIBLAS_MIGRATION/SPEC.md`; section 7 of it lists the sixteen
things the first build has to confirm, and `REPORT.md` beside it holds each wave's review.

- **Run what the suite does not reach.** On 2026-10-09 all 1803 tests passed against SushiBLAS
  `7f04834`. The demo transcript, `sa train` on MNIST and a HALF comparison under
  `ForceReference` have not run.
- **The model graph's engine units**: `ModelState`, `initialize` over it, buffers in the
  checkpoint, the step builders over `ModelOutputs`, and both applications on `build_model`.
  The plan, fifteen tasks in seven waves with fourteen open questions, is
  `docs/agent/2026_10_07_MODEL_GRAPH/PLAN.md`.
- **Finish the status contract.** 398 `SA_THROW_IF` sites in 66 files still throw
  `SushiAI::Error` (the builder, the differentiator, `Config`, `build_layer_stack`,
  `train/objective.cpp`), `nn/throwing_bridge.hpp` bridges eleven of them, and `Scope`'s
  destructor can throw through `ScopeTrace::close()`. `docs/reference/KNOWN_ISSUES.md` lists them.
- **Run on the device what was written on 2026-10-09 and not built**: the fourteen cases of
  `test_lowered_operations.cpp`, the tail tolerance of the learning curve, the status-returning
  `TensorPool::alias` and `request_constant`, the memory plan without
  `values_needing_own_allocation` (its byte counts move), and the engine discard in the seam.
- **Observe that a discarded recording releases storage and scratch.** SushiBLAS `494ff8e` adds
  `Engine::discard_recorded()`; no test on either side watches the release.

- **Decide whether `DeviceFusionSelector` still declines GEMM epilogues under `ForceReference`**
  once the build shows the fused and plain forms agree there.
- **State what the gradient check's reference mode proves.** The host reference path of v0.1.0
  shares the element functor with the SYCL path, so it separates launch and indexing errors and
  not an error in the functor's arithmetic.

## Eager front end

The design is `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`. The host-lane units exist; nothing
executes a partition.

- **`Session`, `Step`, `StepExecutor`, `RegionExecutor`, `RegionEmitter`**, after the migration.
- **Match a boundary value with its storage of the previous step**, so a region survives the
  rebuild of the region that feeds it.
- **Measure the host cost of a step whose structure did not change**, and of renumbering.
- **`Scope`'s destructor can throw** through `ScopeTrace::close()`.

## Convolutional networks

- **Batch normalisation**: a non-trainable `Buffer`, training and evaluation modes, a ones
  initialiser and the running-statistics update (`K_BATCHNORM_UPDATE_STATS` has an identity and no
  emitter or lowering).
- **Layers with more than one input**: `Concat`, a residual block, named children.
- **A rank-4 input leaf**, so images need no `Unflatten` layer.
- **Asymmetric padding in the layers and the model file.** The builder's `_with_extents` emitters
  express it; `Conv2d`, the pools and the `padding` key do not.
- **Make a contiguous `K_RESHAPE` a view in the lowering.** It copies today, so each step copies
  every convolution kernel and its gradient.
- **One home for the layer-name prefix on a refused window**: `Conv2d`, both pools and
  `Upsample2d` each catch and re-throw the builder's message, and the two pools hold the same
  `fit_window`.
- **One order for setting and rank checks across the layers**; it needs the geometry check of
  `window_extents` as its own function.
- **The leaky slope check compares in double**, so a slope that narrows to 0 or 1 in float32
  passes.
- **Call `rematerialise_im2col` from `build_graph`.**
- **The ResNet-18 on CIFAR-10 gate** of the August design, with its loader and augmentation.

## YOLO26

The requirements map is `docs/agent/2026_10_07_YOLO26/SPEC.md`, with the owner's rulings in its
section 8 and the staged plan in section 7; what has been written is in `PLAN.md` beside it. The
reference exporter is `tools/reference/export_yolo26_reference.py`.

Written and never run: the loss operations and their rules, any scalar loss root, stop-gradient,
the `detect` module and the `imaging` module.

- **The detection objective.** No code builds the loss graph of SPEC section 4 from the
  operations; `PLAN.md` walks each term on paper.
- **Two-pass label assignment in the training loop**: a forward-only replay, the host assigner,
  then the full step, and what `CompiledStep` and `Trainer` need for it.
- **The rest of the model graph.** Units U1 to U12 of `docs/agent/2026_10_07_MODEL_GRAPH/SPEC.md`
  are written and never built; `PLAN.md` there lists what waits: the engine-side units U13 to U17
  (the lowering of the fourth `K_BATCHNORM2D` output and of `K_BATCHNORM_FOLD`, an `initialize` that
  fills a scale with one and fills buffers, checkpoints with buffers, the weight importer), the
  attention and head blocks behind rows 10, 22 and 23 of `examples/yolo26/model.json`, and how
  `BatchNorm2d` and a nested `ModelGraph` refuse a misfit when added (owner's ruling).
- **A detection dataset and a threaded loader** over the imaging pipelines: a directory-backed
  sample source, the build pass that parses and rejects labels once, one pipeline per thread.
- **The image decode tool** that writes the raw cache, with libjpeg-turbo through vcpkg.
- **MuSGD, parameter groups, the learning-rate schedule with warmup, and the weight EMA.**
- **The tolerance of a HALF comparison between the reference path and the device.** The
  toolchain's host float-to-half conversion truncates below 2^-13, so a HALF result of smaller
  magnitude computed under `ForceReference` sits one HALF unit in the last place under the
  device's value. A mixed-precision check that compares the two paths allows one HALF ulp below
  2^-13 and bit equality above it (sushiblas: `docs/reference/KNOWN_ISSUES.md`).
- **Reference coverage of NMS.** The fixture's detections are one class with no pair above the
  IoU threshold, so a removal by suppression and the class shift are covered by hand tests only.
- **The detect module's open interface points** are listed in `PLAN.md` under "Detection host
  lane".

## Memory planning

- **Pack the lowering's scratch.** `MemoryPlan::build` packs graph IR values into arenas.
  `TensorPool::scratch()` and `TensorPool::ones()` are not IR values, so each still gets one
  `Storage`; on the demo that is about 2 KiB of 130 KiB. Packing them needs a host-lane
  description of each operation's scratch, shared by the plan and `Lowering`. Today those shapes
  are written inline in `record_log_softmax()` and `record_nll_loss()`
  (`src/graph/memory_plan.cpp`, `src/graph/lowering.cpp`).
- **Feed a batch without two copies.** A step issues two `blas().copy()` calls to move the batch
  into the plan's input buffers, because a recorded SushiBLAS call captures its operand addresses
  by value and every batch is a different source tensor. It needs either an operand-address late
  binding that device tasks can use, or a `Dataset` that writes each batch into the two buffers
  the plan owns. SushiRuntime's `LateBindings` re-points a node's iteration count, base index
  and enabled flag, not the addresses its kernel captured, and
  `TaskGraph::add_offload_dynamic()` carries re-pointable lists only for offloadable tasks with
  a host fallback (`src/train/trainer.cpp`).

## Fusion and the IR

- **Fuse row-wise log-softmax.** `K_LOG_SOFTMAX` lowers to nine SushiBLAS calls, the largest term
  in a step. `max_along` and `sum_along` exist in SushiBLAS, so what is missing is a fusion
  catalog entry (`include/SushiAI/graph/fusion.hpp`, `src/graph/lowering.cpp`).
- **Fuse `K_NLL_LOSS`.** It lowers to three calls because the SushiBLAS whole-tensor reduction
  takes no transform functor, so the elementwise multiply cannot ride into the sum
  (`src/graph/lowering.cpp`).
- **Differentiate standalone `K_LOG_SOFTMAX`.** The rule is `dx = dy - exp(L) * rowsum(dy)` and a
  backward rule writes IR, so it needs an IR node for a row-wise sum: an op identity, a builder
  entry, a shape rule, a lowering and the rule (`include/SushiAI/core/ops.hpp`,
  `src/autograd/rules/loss.cpp`).

## Training and precision

- **Skip the whole step on overflow.** `LossScaler` screens each parameter's gradient and skips
  that parameter's update. Freezing every parameter when any gradient overflows needs a
  dependency edge `execute_zip_inplace` cannot declare: one hand-written task that reads the
  count buffer and writes a gradient (`include/SushiAI/optim/loss_scaler.hpp`).
- **bfloat16.** `MixedPrecisionPolicy` takes its narrow type as a constructor argument, and
  `SushiBLAS::Core::DataType` has no bfloat16 entry (`include/SushiAI/graph/precision.hpp`).

## Data and objectives

- **Widen `Dataset` beyond classification.** `Data::Dataset::num_classes()` is pure virtual and
  `Data::Batch` carries classification targets; a regression task or a detector has no answer
  for either (`include/SushiAI/data/dataset.hpp`).
- **Feed arbitrary objective leaves.** `Train::Trainer` writes `StepSpec::targets` each step. An
  objective that declares another leaf in `ObjectiveLeaves` has nothing writing it
  (`include/SushiAI/train/trainer.hpp`, `include/SushiAI/train/objective.hpp`).

## Diagnostics

- **Pin a reference loss curve.** `sa demo mlp` reports the compile count and the steady-state
  allocations; no test compares a run's loss with a recorded one (`examples/demo_mlp.cpp`).
- **Print per-node timings in the demo.** The table needs a `RuntimeSettings::profiling`
  context, which the demo does not build (`examples/demo_mlp.cpp`, `src/train/trainer.cpp`).

## Open questions from the roadmap

[`ROADMAP.md`](ROADMAP.md) section 24 holds the wording. FFT backend for SushiBLAS: own SYCL
FFT, optional oneMKL DFT, or none. Value-dependent control flow in autograd: out of scope for
v1, to revisit for PINN and GNN work.

## Documentation

- **Make the root README a front door.** It is 584 lines and holds the demo transcript, the
  fusion and mixed-precision measurements and the directory tree. The measurements belong in
  the manual; the owner decides what stays (`README.md`).
- **Module READMEs.** No folder beside the code has one except `tools/` and
  `examples/heavy_mlp/`. The owner decides which folders are modules: the nine areas under
  `include/SushiAI/`, `cli/`, `tests/`, `apps/`.
- **Decide who owns the roadmap.** `ROADMAP.md` plans SushiBLAS and SushiAI together; sections 5
  to 15 are about other repositories.
- **Re-home the archived August plans.** `docs/archive/superpowers/` holds five files named in
  lower-kebab; the archive shape for agent work is `docs/archive/agent/`.
- **Humanizer pass** over `README.md`, `docs/guides/CLI_GUIDE.md`,
  `docs/architecture/ARCHITECTURE.md` and `docs/design/ROADMAP.md`, which run 11 to 16 em dashes
  per thousand words.
- **Compare the manual with the code.** The audit did not check the numbers in `TUTORIAL.md`,
  the prose of `ARCHITECTURE.md` sections 1 to 6, or the flag tables of `CLI_GUIDE.md`.

## Layout, tools and release

- **Let CI reach the siblings.** SushiRuntime and SushiBLAS are private and the workflow has no
  token, so the functional job has never compiled a SushiAI source
  (`.github/workflows/ci.yml`).
- **`Shape::hash` starts from 1469598103934665603**; the 64-bit FNV-1a offset basis is
  14695981039346656037, one digit longer (`include/SushiAI/core/shape.hpp`).
- **Release readiness** is assessed in `docs/agent/2026_10_07_RELEASE_READINESS/REPORT.md`.

- **Declare the tiers.** `K_TIER_ORDER` in `tools/layering/check_layering.py` is empty, and the
  checker looks for `modules/<tier>/` and `source/`, which this tree does not have.
- **Folder names.** `src/`, `apps/`, `examples/`, `tests/functional/` and the root `Doxyfile`
  depart from the `repository-layout` skill, as they do in SushiRuntime and SushiBLAS. One
  decision covers the three repositories.
- **Run the checkers in CI**, add a Windows lane and a release workflow
  (`.github/workflows/ci.yml`).
- **Tag a first release.** Every changelog entry sits under `## Unreleased` and the repository
  has no tag; the declared version is 0.1.0.
- **Tie the CLI version to the library version**, or state that they move apart
  (`CMakeLists.txt`, `cli/pyproject.toml`).
- **Add `.gitattributes`** to pin line endings to the `lf` that `.editorconfig` asks for.

## Estate programmes

The estate audit of 2026-10-05 is in
`docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`, which is kept out of git.
Its "Code shape" section, 22 findings from the lowering's dispatch chain to the error model, is
the input of the code programme and is not restated here.
