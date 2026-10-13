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

The design is `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`, and
`modules/core/eager/README.md` describes the units. `Session`, `Step`, `StepExecutor`,
`RegionExecutor` and `RegionEmitter` are written, with the boundary matching in `SlotLedger` and
`RegionTable`. They have passed a syntax check and have not been built or run;
`docs/agent/2026_10_07_EAGER_FRONT_END/REPORT.md` lists what the first build has to confirm.

- **Build the new units and run their tests**: nine test files under
  `modules/core/eager/tests/`, five of which need an engine.
- **Read the host cost off a run.** `test_step_cost.cpp` prints the cost of an unchanged step for
  two and for fifty layers, and of a scope that appears and disappears. No number exists yet.
- **Phases.** A step runs once; tracing on after a read returns `Unsupported`. The design keeps
  one dynamic graph per stretch between two reads. Two things are undecided: which values of a
  closed phase stay readable for a later one, and how `Autograd::differentiate` treats the
  values of an earlier phase as constants, since `Graph::finalise` propagates `requires_grad`
  through every node.
- **The debug option that flushes after every node** rests on the phases.
- **Compare an eager loop with `Train::Trainer`.** `test_session.cpp` compares it with
  `Graph::CompiledStep`; a test that names `Trainer` crosses modules and belongs under `tests/`.
- **`signature_of` scans every node of the graph for each region**, so a step costs
  nodes times regions on the host. One pass that signs every region would make it linear.
- **Buffer updates on the eager path.** A running statistic is blended by a recorder that holds
  the tensor of a value of one compiled graph; the eager path has no such tensor across steps.
- **Two states whose gradient is one graph value** make `Session::run` return `Unsupported`.
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

The forward pass is built and equals the reference. On 2026-10-09 the 24 rows of
`applications/examples/yolo26/model.json` built, the weights of `tests/fixtures/yolo26/` loaded
through `import_weights`, and `Integration_Yolo26Forward` matched the Ultralytics reference: the
feature maps of rows 16, 19 and 22 and the twelve raw head outputs within 1e-4, the decoded
boxes within 1.3e-4, the scores within 1e-6.

- **A detection report in `sushiai inference`**, which refuses a model with several outputs.
- **The detection objective.** No code builds the loss graph of SPEC section 4 from the
  operations; `PLAN.md` walks each term on paper.
- **Two-pass label assignment in the training loop**: a forward-only replay, the host assigner,
  then the full step, and what `CompiledStep` and `Trainer` need for it.
- **How `BatchNorm2d` and a nested `ModelGraph` refuse a misfit when added** (owner's ruling).
- **A detection dataset and a threaded loader** over the imaging pipelines: a directory-backed
  sample source, the build pass that parses and rejects labels once, one pipeline per thread.
- **The image decode tool** that writes the raw cache, with libjpeg-turbo through vcpkg.
- **MuSGD, parameter groups, the cosine learning-rate schedule with warmup, and the weight
  EMA.**
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
[`PROGRAMME.md`](PROGRAMME.md). The scope is YOLOv3, v4, v7 and their tiny forms, with training.
The specification, the plan and the report are in `docs/agent/2026_10_09_DARKNET_YOLO/`; the
module rulings are in [`DOMAIN_MODULES.md`](DOMAIN_MODULES.md).

Built and passing on 2026-10-09: the cfg and weights importer, the anchor-box path of
`vision/detect`, the automatic anchoring, and the forward pass of the six models against
Darknet's tensors.

- **Build and run the training parity tests.** The step schedule, the delta over IoU boxes and
  over the activated coding, the box gradient option, the six
  `test_darknet_<model>_training.cpp` suites and the one-batch overfit check are written and
  syntax-checked, and have never been compiled. The build waits on SushiBLAS following
  SushiRuntime v0.4.0.
- **Follow SushiRuntime v0.4.0.** The spellings SushiAI uses are already the new ones
  (`devices()`, `memory()`, `TaskGraph()`, `Observability::`, `Memory::`); they are unverified
  until that build.
- **Confirm weights-only decay against Darknet** once a parity suite states a decay; the
  rule is written. Was: **Decay the weights only.** `Optim::SGD` decays every slot it updates; the Darknet solver
  decays the convolution weights and neither the biases nor the batch norm scales. It needs
  parameter groups, which the YOLO26 line below also asks for.
- **Drive `StepSchedule` from a training loop.** The schedule returns a rate per iteration; no
  loop reads it yet.
- **Train a model from a cfg in `sushiai train`**, and report its detections in
  `sushiai inference`.
- **Letterbox preprocessing with its inverse for boxes.**
- **Delete the temporary Darknet checkout** outside the repository once the training tests
  pass; the references and their SHA-256 lists are already copied.
- **An exception to rule 4 of the `testing` skill** for the suites that read
  `SA_REFERENCE_DATA_DIR`, or the references inside the repository: the owner's decision. The
  suites skip when the directory is absent.

## Model import

Ruled on 2026-10-11; the design and its four phases are in
[`MODEL_IMPORT.md`](MODEL_IMPORT.md).

- **Build and run the writers and the task section.** `write_json`, `write_model`, `AnchorTask`
  and their tests are written and have not been built.
- **The registry row of the `task` section**, and `inference` decoding a model from it.
- **The `anchor` objective and the step schedule in the training file.**
- **The `idx` reader row cannot name its two files**, and no row sets the letterbox.
- **What phase 3 still lacks**, listed in `MODEL_IMPORT.md`: subdivisions, options per head,
  a restorable step count.
- **Build and run the import**: the exact text of the three written files, and whether
  `load_checkpoint` accepts the written weights of all six models.
- **Thresholds of the detection report in `eval.json`**, and inference on one image file.
- **`sushiai import darknet`**, and the parity suites reading the imported files.

## Datasets and tasks

Ruled on 2026-10-11; the design and its seven phases are in [`DATASETS.md`](DATASETS.md).

- **Build and run `Batch` with roles and `vision/codecs`.** Both are written and have not been
  built; `test_learning_curve`, `test_determinism` and `test_compile_once` cover the synthetic
  data that changed its construction.
- **Check that the engine allocates under concurrent calls**, which the first read of a
  `ReaderDataset` into a new `Batch` does; the loader on threads depends on it.
- **A death test inside an engine fixture**: `ABatchOutsideTheEpochStopsTheProcess` is the
  first, and whether its child process gets through engine start-up is not known.
- **Turn the detection roles into a `GroundTruthBatch`**, in `vision/training`: int32 classes to
  int64, the float `valid` to bytes.
- **Give the imaging functions a status.** They throw, and `fit_sample` reports any of their
  errors as a file fault.
- **A resize that averages on reduction.** `resize_bilinear` aliases a large image fitted to a
  small input.
- **Write the raw cache from a folder of images**, the rest of phase 2.
- **One unit for the exception boundary of the decoders**, which four functions of
  `vision/codecs` repeat, and a `raw_decoder` unit out of `decode_image.cpp`.
- **A 16-bit PNG without a `gAMA` chunk** has no test.
- **A way for a collator to say it reads no mask**; one is decoded whenever a sample has it.
- **Build and run the loader**: no test allocates from two threads, and the claim that it is
  safe is read from SushiRuntime's allocator, not run.
- **The detection collator and `train` over `AnchorTrainingStep`.**
- **A checkpoint inside an epoch.** The run state carries a batch index, but `sushiai train`
  saves only when the run returns, so the batch it writes is 0.
- **A test of the program loops from a start other than zero**, in `test_program_runs.cpp`;
  the bit-for-bit resume test drives its own loop over `Trainer::run_step`.
- **`Trainer::fit` begins at epoch 0 on every call**; the CLI does not use it.
- **`tools/reference/export_yolo26_reference.py` still writes version 1** of the weight file;
  its files read.
- **A refused save truncates the old file**: the target is opened before the writer can refuse.
- **A way for an objective to name its metrics before it measures**, so a sum of two terms
  does not compute mean IoU twice.
- **Build and run segmentation**: the objectives, the head and the overfit test are written.
- **A bilinear upsample in the graph**, for a head whose boundaries are finer than its scale.
- **Several polygons per object in a label**, for outlines of more than one part.
- **Keys for the probabilities and gains of the `default` pipeline**, and for its mosaic epochs.

## Language

Ruled into 0.1.0 on 2026-10-11; stage 9 of [`PROGRAMME.md`](PROGRAMME.md). The design and its
seven phases are in [`LANGUAGE.md`](LANGUAGE.md). Phases 1 to 5 and 7 are written and not yet built.

- **Phase 6 of the design**, the decision model: the first work after the build passes, by
  the owner's ruling of 2026-10-11.
- **Per-term losses in a step's view**, so perplexity inside a sum is that of the fit alone.
- **A port that is one value per step**, not per sample; a balance stands behind an axis of 1.
- **Usage metrics without a balance term**: a sum refuses a weight of 0.
- **Rotary leftovers**: a position past 2^28 is not refused; rotary under mixed precision is
  not exercised (`fill_host_tables` refuses a half leaf); `tiny_decoder` stays on the
  learned table; the window form declares its tables per block.
- **An in-place row write in the graph**, in place of the scatter-add into zeros.
- **A token-file writer that appends**, so `sushiai tokenize encode` holds one piece and not
  every id of the corpus.
- **One factory signature for every node**, once vision's rows can change; linked rows then go.
- **Call `match_input` from the anchor loop**, which matches roles only; the step holds its
  input leaf privately.
- **The epoch in the dataset seam**, without which window starts cannot be redrawn: one pure
  virtual, 8 overrides and 68 call sites.
- **One home for `at_boundary` and the little-endian readers**, which `vision/datasets`,
  `language/text` and `language/datasets` each hold privately.
- **Measure `train_bpe` on a corpus of tens of megabytes.**
- **The points phase 2 left open**, listed in `LANGUAGE.md`.
- **Dropout: the device fill.** The mask is drawn on the host until SushiBLAS lands B1. No
  test covers dropout in HALF or FLOAT64 on an engine.
- **Tests whose body and name disagree**: `DropoutRefusesOperandsThatDoNotFit` asserts no
  empty input; the second half of `AMergeOfALaterTokenIsRefused` breaks another byte;
  `AMissingFileFails` of the tokens section checks no code or message;
  `ARefusedDiceFillChangesNothing` reads back four of the five leaves.
- **`TheYOLO26FileNamesEachMissingBlock` has no successor** since the YOLO26 cases moved to
  `vision/registration`; decide whether a model file with a missing block still needs a case.
- **The older sections of `tests/unit/README.md`**: about 190 were checked for headings and
  suites only, not sentence by sentence against their sources; 106 lines pass 100 columns.
- **The Darknet sections of `tests/integration/README.md`** were written from the case briefs
  and model descriptions, not from the bodies of the shared check helpers.
- **Check a token id against the table's rows before a step.** An index outside the table reads
  or writes out of bounds in the sparse product; nothing validates it yet.
- **An int32 input in the model file.** `Config::ModelSpec` states no input type, so an
  `embedding` row works from C++ only.
- **Deferred by the owner: how a run spans machines** on SushiRuntime's distributed layer: what is shipped per
  step, and what the network between the owner's five machines can carry. Mixture of experts is
  asked about; it needs a top-k router and a balance term.

## Memory planning

- **Write the scratch rules of the rows that have none.** A lowering row states the buffers it
  writes on every run in a scratch rule, and the plan packs them (`lowering/rule.hpp`,
  `ScratchRule`). Batch norm and its gradient have a rule. The other rows that call
  `LoweringContext::scratch()`, in `lowering/norm.cpp`, `loss.cpp`, `linalg.cpp` and
  `random.cpp`, have none, so each of their buffers still gets one `Storage`, as does every
  shared `ones()` tensor. A rule must leave out a buffer the host fills once or a replay
  prelude writes.
- **Find why a full-size training step leaves Darknet.** The step bounds of YOLOv4 and YOLOv7
  are 1.5 times what the run of 2026-10-13 measured, by the owner's ruling of that day: a head
  of YOLOv4 is up to 0.205 from the dump on values near 40, a head of YOLOv7 up to 0.0052.
  YOLOv3 below differs by as much as YOLOv4, and the three tiny models hold 1e-3. The memory
  plan is ruled out for the tiny models, where a planned step equals an unplanned one bit for
  bit; no full-size model fits unplanned. The owner asked for the cause to be found. It needs a
  Darknet dump of each layer's output under batch statistics and a case that walks them, the
  same work as the YOLOv3 item below.
- **Close the rest of the step time against Darknet.** On the RTX 3080 Ti with a batch of 2,
  measured 2026-10-13 over 200 steps: YOLOv4 at 608 by 608 takes 0.84 s a step against 0.21 s
  for an iteration of a CUDA build of Darknet without cuDNN, YOLOv7 at 640 by 640 0.74 s
  against 0.18 s, YOLOv4-tiny 0.08 s against 0.02 to 0.04 s. The steps took 3.2 s, 3.0 s and
  0.25 s before SushiBLAS gained a `gemm` that divides the depth of a product with a small
  result, which the weight gradient of a convolution is. `RuntimeConfig::profiling` gives the
  time of each node of a step and was how that product was found; a step is device time, with
  no host share. Arenas of `Residency::Device` change nothing. Left: profile a step again and
  take the next largest share. A step runs the forward pass twice and recomputes each column
  matrix (`GraphOptions::rematerialise`).
  `docs/agent/2026_10_13_PLAN_PACKED_SCRATCH/REPORT.md` has how both sides were timed.
- **Make the one-batch overfit of YOLOv7 hold.**
  `Integration_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` fails since
  2026-10-13: the stored cost is 11.1 at step 199 against a bound of 3.56. It reached 1.27
  with the `gemm` SushiBLAS had before. The two runs agree to 1e-7 of the cost at step 0 and
  3e-8 at step 1, then part: 6e-5 at step 2, 1e-3 at step 3, 2 % at step 4. Both costs jump
  between 14 and 60 over the first 25 steps, so the run is unstable on this batch and the last
  bits of a sum decide where it stands at step 199. The one-step cases of the suite pass. To
  find: what makes the run unstable, the learning rate and the warm-up first. The bound waits
  on the owner.
- **Measure where YOLOv3's training step leaves Darknet, layer by layer.** The step bounds of
  `test_darknet_yolov3_training.cpp` are 1.5 times what the run of 2026-10-12 measured: a head
  is up to 0.243 from the dump on values near 49. No layer before the heads is compared, so
  whether the difference grows evenly over the 75 convolutions or enters at one layer is not
  known. It needs a Darknet dump of each layer's output under batch statistics and a case that
  walks them.
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
