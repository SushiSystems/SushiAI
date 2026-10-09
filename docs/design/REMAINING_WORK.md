# Remaining work

The single backlog of this repository. Work and decisions that are still open are listed here;
a defect that exists today is in [`KNOWN_ISSUES.md`](../reference/KNOWN_ISSUES.md). The design
behind the library items is in [`ROADMAP.md`](ROADMAP.md) and
[`ARCHITECTURE.md`](../architecture/ARCHITECTURE.md).

## What a build has confirmed

Builds and test runs happen when the owner allows one. The last, on 2026-10-09 against SushiBLAS
`494ff8e` on an RTX 3080 Ti: the tree built with no error, `sushiai_tests` ran 1875 tests (1874
passed, 1 skipped for the absent MNIST files), and `sushiai_train`, `sushiai_inference` and
`sushiai_demo_mlp` each reported one compile on the test fixtures. The runs are in
`docs/agent/2026_10_07_MODEL_GRAPH/REPORT.md`.

Not yet run:

- **`sa train examples/mnist_cnn/train.json`**, the version 2 model file and the `image` layout
  read from the IDX files. The list is in `docs/agent/2026_10_07_CV_CLOSE/SPEC.md` section 4.
- **The demo transcript** against its recorded text, and a HALF comparison under
  `ForceReference`.
- **`sa test`**, which runs the CTest cases around the three programs; the programs were run by
  hand instead.

## SushiBLAS v0.1.0 migration

SushiBLAS removes the engine facades, makes every operation take a `TaskRecorder` and removes
`Engine::get_graph()`. Its design is in the sushiblas repository under
`docs/agent/2026_10_07_V010/`.

The library, the applications, the examples and the tests were moved on 2026-10-08 and built on
2026-10-09. The design is `docs/agent/2026_10_08_SUSHIBLAS_MIGRATION/SPEC.md`, and `REPORT.md`
beside it holds each wave's review and the runs.

- **Finish the status contract.** 398 `SA_THROW_IF` sites in 66 files still throw
  `SushiAI::Error` (the builder, the differentiator, `Config`, `build_layer_stack`,
  `train/objective.cpp`), `nn/throwing_bridge.hpp` bridges eleven of them, and `Scope`'s
  destructor can throw through `ScopeTrace::close()`. `docs/reference/KNOWN_ISSUES.md` lists them.
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

Built and run: the blocks `conv`, `c3k2` and `sppf`, the model graph and `build_model` over
`applications/examples/yolo26/model.json`, the model state, the evaluation step, the weight
import, the `detect` and `imaging` modules. Twenty-two of the model file's 24 rows build.

- **The `c2psa` block** (row 10) and **the `detect` head** (row 23), the two row types no
  registered module has.
- **Forward parity**: load `tests/fixtures/yolo26/weights.sacp` through `import_weights` with
  the rename rules of the Ultralytics names, feed `input.sacp`, compare with `forward.sacp`.
- **The detection objective.** No code builds the loss graph of SPEC section 4 from the
  operations; `PLAN.md` walks each term on paper.
- **Two-pass label assignment in the training loop**: a forward-only replay, the host assigner,
  then the full step, and what `CompiledStep` and `Trainer` need for it.
- **How `BatchNorm2d` and a nested `ModelGraph` refuse a misfit when added** (owner's ruling).
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

## Domain modules

The design and the owner's rulings are in [`DOMAIN_MODULES.md`](DOMAIN_MODULES.md); the tree is
`modules/<tier>/<module>/` and was built on 2026-10-09. New vision code is written into
`modules/vision/`.

- **Carry the domain-tier key of the layering checker to SushiSkills.** The shared copy reads
  `<SushiAI/vision/detect/x.hpp>` as a module named `vision`.
- **Split `sushiai_add_module`** into one function per concern (key, target, dependencies,
  tests, install).
- **One home for the join that lists type names**, in `node_registry.cpp`,
  `layer_registry.cpp` and `dataset_registry.cpp`.
- **Give each module's tests their own README**, and take their sections out of
  `tests/unit/README.md`.

## Darknet-era YOLO

Inside the feature freeze of 0.1.0 by the owner's ruling of 2026-10-09; stage 6 of
[`PROGRAMME.md`](PROGRAMME.md). No design document exists yet.

- **Decide where the anchor unit lives.** The owner's decision, to be brought with the design:
  inside `detect`, or in `detect/anchor/` beside the anchor-free head of YOLO26, sharing the box,
  IoU and NMS bricks.
- **Write the anchor-based detection unit**: a decoder (cell offset through a sigmoid, size
  through an exponential times the anchor), a host assigner by best-anchor IoU with an ignore
  threshold, and a loss. One unit with an options structure for the `region` form of v2 and the
  `yolo` form of v3 and v4, not a class per version.
- **Write the Darknet importer**: the `.cfg` text to a model graph and the `.weights` binary to
  parameters and buffers, with batch normalisation stored as bias, scale, mean, variance and
  then the weights.
- **Add the layers the models need**: `reorg` for v2, `Mish` and a grouped `route` for v4, the
  fully connected detection head of v1, and letterbox preprocessing with its inverse for boxes.
- **Check `MaxPool2d` against YOLOv3-tiny's 2x2 pool of stride 1**, which Darknet pads on one
  side only.
- **Produce the references by running Darknet**, as the YOLO26 references are produced, never
  stated. YOLOv3-tiny on its published weights is the first.

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
