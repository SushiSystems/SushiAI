# Remaining work

Next id: RW-130

## Decisions waiting for the owner

### RW-1 Tag a first release
- State: decision
- Done when: the owner has approved 0.1.0 and `## Unreleased` becomes its section

### RW-2 Rule how `BatchNorm2d` and a nested `ModelGraph` refuse a misfit when added
- State: decision
- Done when: the owner has ruled and both refuse that way
- Source: `docs/agent/2026_10_07_YOLO26/SPEC.md`

### RW-3 Rule where the Darknet reference data lives
- State: decision
- Done when: the suites that read `SA_REFERENCE_DATA_DIR` have an exception to rule 4 of the
  `testing` skill, or their references are inside the repository

The suites skip when the directory is absent.

### RW-4 Rule how a run spans machines
- State: decision
- Done when: the owner has ruled what a step ships on SushiRuntime's distributed layer and what
  the network between the five machines carries

Deferred by the owner. Mixture of experts is asked about; it needs a top-k router and a balance
term.

### RW-5 Rule on value-dependent control flow in autograd
- State: decision
- Done when: the owner has ruled it in or out of scope; it was out of scope for v1, to revisit
  for PINN and GNN work

### RW-6 Decide whether a model file with a missing block needs a case
- State: decision
- Done when: `TheYOLO26FileNamesEachMissingBlock` has a successor under `vision/registration`,
  or the owner has ruled it unneeded

### RW-7 Decide whether the archived August plans are renamed
- State: decision
- Done when: the five lower-kebab files under `docs/archive/superpowers/` move to the
  `docs/archive/agent/` shape, or the owner leaves the frozen archive as it is

## Runs not yet made

### RW-8 Train the MNIST CNN from its model file
- State: unmeasured
- Done when: `sa train examples/mnist_cnn/train.json` has run on the version 2 model file and the
  `image` layout read from the IDX files
- Source: `docs/agent/2026_10_07_CV_CLOSE/SPEC.md`

### RW-9 Check the demo transcript against a run
- State: unmeasured
- Done when: the transcript of `docs/guides/DEMO.md` matches a run, its node counts among them,
  and a HALF comparison under `ForceReference` has run

### RW-10 Run `sa test` around the three programs
- State: unmeasured
- Done when: the CTest cases around `sushiai_train`, `sushiai_inference` and the demo pass

## Failure contract

### RW-11 Finish the status contract
- State: open
- Done when: no `SA_THROW_IF` site throws `SushiAI::Error` where the contract asks for a `Status`
  or a `Result`, and no `try` or `catch` sits outside `seam/seam.hpp` but the ones the contract
  allows
- Source: `docs/design/PROGRAMME.md`

Stage 2 of the programme: 398 sites in 66 files, `nn/throwing_bridge.hpp` bridging eleven, and
`Scope`'s destructor throwing through `ScopeTrace::close()`.

### RW-12 Observe that a discarded recording releases storage and scratch
- State: unmeasured
- Done when: a test watches `Engine::discard_recorded()` release both

### RW-13 Decide whether fusion declines GEMM epilogues under `ForceReference`
- State: unmeasured
- Done when: a build shows whether the fused and plain forms agree under `ForceReference`, and
  `DeviceFusionSelector` follows

### RW-14 State what the gradient check's reference mode proves
- State: open
- Done when: the gradient check's documentation says the reference path shares the element
  functor with the SYCL path, so it separates launch and index errors, not formula errors

## Eager front end

### RW-15 Build the eager units and run their tests
- State: unmeasured
- Done when: the nine test files of the eager module pass, five of them on an engine
- Source: `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`

### RW-16 Read the host cost of an eager step off a run
- State: unmeasured
- Done when: `test_step_cost.cpp` has printed the cost of an unchanged step for two and fifty
  layers, and of a scope that appears and disappears

### RW-17 Design the phases of an eager session
- State: open
- Done when: the design says which values of a closed phase stay readable, and how
  `Autograd::differentiate` treats an earlier phase's values as constants while
  `Graph::finalise` propagates `requires_grad`
- Source: `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`

### RW-18 Add the debug option that flushes after every node
- State: blocked
- Blocked by: RW-17
- Done when: an eager session can flush after each node

### RW-19 Compare an eager loop with `Train::Trainer`
- State: open
- Done when: a test under `tests/` compares the two

### RW-20 Sign every region in one pass
- State: open
- Done when: `signature_of` costs one walk of the graph per step, not one per region

### RW-21 Update buffers on the eager path
- State: open
- Done when: a running statistic is blended across eager steps

### RW-22 Run a session whose two states share one gradient value
- State: open
- Done when: `Session::run` handles it instead of returning `Unsupported`

## Convolutional networks

### RW-24 Train batch normalisation
- State: open
- Done when: a non-trainable `Buffer`, training and evaluation modes, a ones initialiser and the
  running-statistics update exist; `K_BATCHNORM_UPDATE_STATS` has an emitter and a lowering

### RW-25 Take layers with more than one input
- State: open
- Done when: `Concat`, a residual block and named children exist

### RW-26 Give images a rank-4 input leaf
- State: open
- Done when: an image model needs no `Unflatten` layer

### RW-27 Pad asymmetrically in the layers and the model file
- State: open
- Done when: `Conv2d`, the pools and the `padding` key take the extents the builder's
  `_with_extents` emitters express

### RW-29 Give the layer-name prefix of a refused window one home
- State: open
- Done when: `Conv2d`, both pools and `Upsample2d` stop catching and re-throwing the builder's
  message, and the pools share one `fit_window`

### RW-30 Check settings and ranks in one order across the layers
- State: open
- Done when: the geometry check of `window_extents` is its own function and every layer checks
  in one order

### RW-31 Check the leaky slope in the element type
- State: open
- Done when: a slope that narrows to 0 or 1 in float32 is refused

### RW-32 Call `rematerialise_im2col` from `build_graph`
- State: open
- Done when: `build_graph` runs the pass, as the eager region program does

### RW-33 Pass the ResNet-18 on CIFAR-10 gate
- State: open
- Done when: the gate of the August design passes, with its loader and augmentation

## YOLO26

### RW-34 Report detections in `sushiai inference`
- State: open
- Done when: inference accepts a model with several outputs and reports its detections
- Source: `docs/agent/2026_10_07_YOLO26/SPEC.md`

### RW-35 Build the detection objective
- State: open
- Done when: code builds the loss graph of section 4 of the YOLO26 specification
- Source: `docs/agent/2026_10_07_YOLO26/PLAN.md`

### RW-36 Assign labels in two passes in the training loop
- State: open
- Done when: a forward-only replay, the host assigner and the full step run in the loop, with
  what `CompiledStep` and `Trainer` need

### RW-37 Load a detection dataset on threads
- State: open
- Done when: a directory-backed sample source, a build pass that parses and rejects labels once
  and one pipeline per thread exist

### RW-38 Write the image decode tool
- State: open
- Done when: a tool writes the raw cache through libjpeg-turbo from vcpkg

### RW-39 Add MuSGD, parameter groups, a cosine schedule with warm-up and the weight EMA
- State: open
- Done when: each exists with tests; parameter groups also give the Darknet weight-only decay

### RW-40 Allow one HALF unit below 2^-13 between the reference path and the device
- State: open
- Done when: a mixed-precision check that compares the two paths allows one HALF ulp below
  2^-13 and bit equality above it (`sushiblas KI-3`)

### RW-41 Cover suppression and the class shift of NMS against the reference
- State: open
- Done when: a fixture has a pair above the IoU threshold and more than one class

### RW-42 Close the detect module's open interface points
- State: open
- Done when: the points under "Detection host lane" are closed
- Source: `docs/agent/2026_10_07_YOLO26/PLAN.md`

## Domain modules

### RW-43 Split `sushiai_add_module` by concern
- State: open
- Done when: key, target, dependencies, tests and install are one function each
- Source: `docs/design/DOMAIN_MODULES.md`

### RW-44 Give the join that lists type names one home
- State: open
- Done when: `node_registry.cpp`, `layer_registry.cpp` and `dataset_registry.cpp` share it

## Darknet-era YOLO

### RW-45 Follow SushiRuntime v0.4.0 and the first SushiBLAS tag
- State: blocked
- Blocked by: sushiruntime RW-3, sushiblas RW-8
- Done when: the spellings this tree uses build against SushiRuntime v0.4.0, and CI pins
  `SUSHIRUNTIME_REF` and `SUSHIBLAS_REF` to tags
- Source: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`

### RW-46 Confirm weight-only decay against Darknet
- State: unmeasured
- Done when: a parity suite states a decay and matches Darknet, which decays convolution weights
  and neither biases nor batch norm scales

### RW-47 Drive `StepSchedule` from a training loop
- State: open
- Done when: a loop reads the rate the schedule returns per iteration

### RW-48 Train a model from a cfg in `sushiai train`
- State: open
- Done when: `sushiai train` takes a cfg and `sushiai inference` reports its detections

### RW-49 Letterbox the input, with the inverse for boxes
- State: open
- Done when: preprocessing letterboxes and boxes map back

### RW-50 Delete the temporary Darknet checkout
- State: open
- Done when: the checkout outside the repository is gone, once the training tests pass

## Model import

### RW-51 Build and run the writers and the task section
- State: unmeasured
- Done when: `write_json`, `write_model`, `AnchorTask` and their tests pass
- Source: `docs/design/MODEL_IMPORT.md`

### RW-52 Give the `task` section a registry row
- State: open
- Done when: the row exists and inference decodes a model from it

### RW-53 Put the `anchor` objective and the step schedule in the training file
- State: open
- Done when: both are keys of the training file

### RW-54 Let the `idx` reader row name its two files, and a row set the letterbox
- State: open
- Done when: both rows exist

### RW-55 Close what phase 3 of the import lacks
- State: open
- Done when: subdivisions, options per head and a restorable step count exist
- Source: `docs/design/MODEL_IMPORT.md`

### RW-56 Build and run the import
- State: unmeasured
- Done when: the three written files match their expected text and `load_checkpoint` accepts the
  written weights of all six models

### RW-57 Set detection thresholds in `eval.json` and infer on one image file
- State: open
- Done when: both work

### RW-58 Add `sushiai import darknet`
- State: open
- Done when: the command exists and the parity suites read the imported files

## Datasets and tasks

### RW-59 Build and run `Batch` with roles and `vision/codecs`
- State: unmeasured
- Done when: both pass, with `test_learning_curve`, `test_determinism` and `test_compile_once`
- Source: `docs/design/DATASETS.md`

### RW-60 See the engine allocate under concurrent calls
- State: unmeasured
- Done when: a test allocates from two threads, as the first read of a `ReaderDataset` into a
  new `Batch` does

### RW-61 Run a death test inside an engine fixture
- State: unmeasured
- Done when: `ABatchOutsideTheEpochStopsTheProcess` is seen to pass engine start-up in its child

### RW-62 Turn the detection roles into a `GroundTruthBatch`
- State: open
- Done when: `vision/training` converts int32 classes to int64 and the float `valid` to bytes

### RW-63 Give the imaging functions a status
- State: open
- Done when: they return a status, and `fit_sample` reports each error as what it is

### RW-64 Average on reduction when resizing
- State: open
- Done when: `resize_bilinear` does not alias a large image fitted to a small input

### RW-65 Write the raw cache from a folder of images
- State: open
- Done when: the rest of phase 2 of the datasets design exists
- Source: `docs/design/DATASETS.md`

### RW-66 Give the decoders one exception boundary
- State: open
- Done when: the four functions of `vision/codecs` share one unit, and `raw_decoder` leaves
  `decode_image.cpp`

### RW-67 Test a 16-bit PNG without a `gAMA` chunk
- State: open
- Done when: the case exists

### RW-68 Let a collator say it reads no mask
- State: open
- Done when: a collator that reads no mask stops a mask being decoded

### RW-69 Build and run the loader on two threads
- State: unmeasured
- Done when: a test allocates from two threads through the loader

### RW-70 Collate detections and train over `AnchorTrainingStep`
- State: open
- Done when: the detection collator exists and `train` runs over the step

### RW-71 Checkpoint inside an epoch
- State: open
- Done when: `sushiai train` saves at a batch and the written batch index is that batch

### RW-72 Test the program loops from a start other than zero
- State: open
- Done when: `test_program_runs.cpp` starts a loop past zero

### RW-73 Let `Trainer::fit` resume past epoch 0
- State: open
- Done when: `fit` begins at the epoch it is given

### RW-74 Write version 2 of the weight file from the YOLO26 exporter
- State: open
- Done when: `tools/reference/export_yolo26_reference.py` writes version 2

### RW-75 Keep the old file when a save is refused
- State: open
- Done when: a refused save leaves the target as it was

### RW-76 Let an objective name its metrics before it measures
- State: open
- Done when: a sum of two terms computes mean IoU once

### RW-77 Build and run segmentation
- State: unmeasured
- Done when: the objectives, the head and the overfit test pass

### RW-78 Upsample bilinearly in the graph
- State: open
- Done when: a head can have boundaries finer than its scale (G7 of the graph capabilities plan)

### RW-79 Take several polygons per object in a label
- State: open
- Done when: an outline of more than one part loads

### RW-80 Give the `default` pipeline keys for its probabilities, gains and mosaic epochs
- State: open
- Done when: each is a key

## Language

### RW-81 Design the decision model, phase 6
- State: open
- Done when: phase 6 is designed, the first work after the build passes by the owner's ruling
- Source: `docs/design/LANGUAGE.md`

### RW-82 Show per-term losses in a step's view
- State: open
- Done when: perplexity inside a sum is that of the fit alone

### RW-83 Add a port that is one value per step
- State: open
- Done when: a balance needs no axis of 1

### RW-84 Allow usage metrics without a balance term
- State: open
- Done when: a sum accepts a weight of 0

### RW-85 Finish rotary positions
- State: open
- Done when: a position past 2^28 is refused, rotary runs under mixed precision, `tiny_decoder`
  uses rotary and the window form shares its tables

### RW-86 Write a row in place in the graph
- State: open
- Done when: an in-place row write replaces the scatter-add into zeros

### RW-87 Append to a token file
- State: open
- Done when: `sushiai tokenize encode` holds one piece at a time

### RW-88 Give every node one factory signature
- State: open
- Done when: vision's rows take it and the linked rows go

### RW-89 Call `match_input` from the anchor loop
- State: open
- Done when: the loop matches inputs as well as roles

### RW-90 Pass the epoch through the dataset seam
- State: open
- Done when: window starts can be redrawn per epoch: one pure virtual, 8 overrides, 68 call sites

### RW-91 Give `at_boundary` and the little-endian readers one home
- State: open
- Done when: `vision/datasets`, `language/text` and `language/datasets` share them

### RW-92 Measure `train_bpe` on a corpus of tens of megabytes
- State: unmeasured
- Done when: the time and memory are recorded

### RW-93 Close the points phase 2 left open
- State: open
- Done when: the points the language design lists are closed
- Source: `docs/design/LANGUAGE.md`

### RW-94 Fill the dropout mask on the device
- State: open
- Done when: the mask is drawn on the device (B1 and G1 of the graph capabilities plan) and a test
  covers dropout in HALF and FLOAT64 on an engine

B1 belongs to SushiBLAS; its entry there waits until that repository's current work is done.

### RW-95 Make the tests' bodies match their names
- State: open
- Done when: `DropoutRefusesOperandsThatDoNotFit`, `AMergeOfALaterTokenIsRefused`,
  `AMissingFileFails` and `ARefusedDiceFillChangesNothing` check what they name

### RW-96 Check the older test pages sentence by sentence
- State: open
- Done when: the about 190 older sections of the test READMEs match their sources and none of
  their lines passes 100 columns (106 do)

### RW-97 Write the Darknet sections of the importer test page from the helpers
- State: open
- Done when: `tests/vision/importers/README.md` describes the shared check helpers from their
  bodies

### RW-98 Check a token id against the table before a step
- State: open
- Done when: an id outside the table is refused before the sparse product reads it (A5)

### RW-99 Take an int32 input in the model file
- State: open
- Done when: `Config::ModelSpec` states an input type, so an `embedding` row works from a file

## Memory planning and speed

### RW-100 Write the scratch rules of the rows that have none
- State: open
- Done when: the rows of `lowering/norm.cpp`, `loss.cpp`, `linalg.cpp` and `random.cpp` that call
  `LoweringContext::scratch()`, and the shared `ones()` tensors, have scratch rules

A rule leaves out a buffer the host fills once or a replay prelude writes.

### RW-101 Find why a full-size training step leaves Darknet
- State: open
- Done when: a Darknet dump of each layer's output under batch statistics and a case that walks
  them show where YOLOv4 and YOLOv7 leave it

A head of YOLOv4 is up to 0.205 from the dump on values near 40, one of YOLOv7 up to 0.0052; the
tiny models hold 1e-3, and the memory plan is ruled out for them.

### RW-102 Close the step time against Darknet
- State: open
- Done when: a full-size step takes the time of a Darknet iteration on the same card
- Source: `docs/agent/2026_10_13_PLAN_PACKED_SCRATCH/REPORT.md`

RTX 3080 Ti, batch 2, 200 steps: YOLOv4 0.76 s a step against 0.21 s, YOLOv7 0.70 s against
0.18 s. The host stage costs YOLOv4 115 ms to read the heads and 120 ms to write the deltas, all
on `Residency::Shared` pages; batch normalisation and its gradient, lowered to primitives, are
most of the 4 980 small nodes. Left, by return: copy heads and deltas through
`Residency::Host` tensors (an allocation of that residency in `allocation.hpp`), fuse batch
normalisation and its gradient, compute the deltas on the device.

### RW-103 Make the one-batch overfit of YOLOv7 hold
- State: open
- Done when: `Large_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` passes, the cause of
  the instability found and the bound ruled by the owner

### RW-104 Measure where YOLOv3's training step leaves Darknet, layer by layer
- State: unmeasured
- Done when: a per-layer dump under batch statistics shows whether the difference grows over the
  75 convolutions or enters at one

### RW-105 Feed a batch without two copies
- State: blocked
- Blocked by: sushiruntime RW-59
- Done when: a step writes the batch into the plan's input buffers without two `blas().copy()`
  calls

A `Dataset` that writes each batch into the two buffers the plan owns is the other way.

## Fusion and the IR

### RW-106 Fuse row-wise log-softmax
- State: open
- Done when: a fusion catalog entry replaces the nine calls `K_LOG_SOFTMAX` lowers to
  (`modules/core/graph/include/SushiAI/graph/fusion.hpp`)

### RW-107 Fuse `K_NLL_LOSS`
- State: open
- Done when: it lowers to fewer than three calls

The SushiBLAS whole-tensor reduction takes no transform functor.

### RW-108 Skip the whole step on overflow
- State: open
- Done when: one overflowed gradient freezes every parameter for that step, through a task that
  reads the count buffer and writes a gradient

### RW-109 Add bfloat16
- State: open
- Done when: `MixedPrecisionPolicy` takes bfloat16

`SushiBLAS::Core::DataType` has no bfloat16 entry; its entry there waits until that
repository's current work is done.

### RW-123 Differentiate `K_LOG_SOFTMAX` on its own
- State: open
- Done when: it has a backward rule outside the fused pair with `K_NLL_LOSS`

## Data and objectives

### RW-110 Widen `Dataset` beyond classification
- State: open
- Done when: a regression task or a detector can implement `Dataset` without `num_classes()` and
  classification targets (`modules/core/data/include/SushiAI/data/dataset.hpp`)

### RW-111 Feed arbitrary objective leaves
- State: open
- Done when: `Trainer` writes every leaf an objective declares in `ObjectiveLeaves`

## Diagnostics

### RW-112 Pin a reference loss curve
- State: open
- Done when: a test compares a run's loss with a recorded curve

### RW-113 Print per-node timings in the demo
- State: open
- Done when: `applications/examples/source/demo_mlp.cpp` builds a profiling context and prints
  the table

## Graph capabilities

### RW-120 Carry out the SushiAI items of the graph capabilities plan
- State: open
- Done when: the G and A items of section 5 are closed
- Source: `docs/design/GRAPH_CAPABILITIES.md`

R1 is `sushiruntime RW-62`; R2 is the owner's deferred question, `RW-4`. The B items belong to
SushiBLAS, and their entry there waits until that repository's current work is done.

### RW-121 Move onto SushiBLAS's views, enums and `BLASScalar`
- State: open
- Done when: this tree builds on that interface

### RW-122 Give `Status` a code for a runtime busy with another run
- State: open
- Done when: a call that meets a busy runtime returns that code

## Documentation

### RW-114 Compare the manual with the code
- State: unmeasured
- Done when: the numbers of `TUTORIAL.md`, sections 1 to 6 of `ARCHITECTURE.md` and the flag
  tables of `CLI_GUIDE.md` are checked against the code

### RW-124 Correct the agent files and a test comment
- State: open
- Done when: `CLAUDE.md` and `AGENTS.md` name `build/docs/api-site` and `sa docs`, and
  `tests/CMakeLists.txt` cites `docs/guides/DEMO.md` for the demo's output

### RW-125 Point the security page at the shared policies
- State: open
- Done when: `docs/reference/SECURITY.md` links each sibling's policy where it is

### RW-126 Record the `sushicore` floor raise in the changelog
- State: open
- Done when: `docs/reference/CHANGELOG.md` records the move from 0.4.0 to 0.7.0

## Layout, tools and release

### RW-115 Fix the offset basis of `Shape::hash`
- State: open
- Done when: it starts from 14695981039346656037, the 64-bit FNV-1a basis

### RW-127 Set the documentation's version
- State: open
- Done when: `PROJECT_NUMBER` in `.config/doxygen/Doxyfile` is the library's version

### RW-128 Drop the ignore rules of `docs/api-site/`
- State: open
- Done when: `.gitignore` names no folder nothing writes

### RW-129 Spell the repository URL in `sushi-module.toml` as GitHub does
- State: open
- Done when: the manifest names `SushiSystems/SushiAI`

## Test tree

### RW-116 Run the Darknet parity scenarios of YOLOv3, YOLOv4 and YOLOv7 after the move
- State: unmeasured
- Done when: the scenarios in `tests/vision/importers/test_darknet_training/` and
  `test_darknet_import/` have run on the moved tree

### RW-117 Name every suite after its unit
- State: open
- Done when: no suite such as `Small_Diagnostics` or `Unit_<Module><Header>` starts with
  another name

### RW-118 List the new test pages and give the new test folders a README
- State: open
- Done when: each folder README that keeps a test-file table lists its new pages, and the 18 new
  folders have a README

### RW-119 Move three tests into their mirror files
- State: open
- Done when: the cases of `check.hpp`, `load_sample_mask.hpp` and `perplexity.hpp` leave
  `test_status.cpp`, `test_fit_mask.cpp` and `test_next_token_targets.cpp`
