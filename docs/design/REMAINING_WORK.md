# Remaining work

Next id: RW-137

## Next

- RW-116 The Darknet parity scenarios are the YOLOv3-tiny gate, and the last run skipped them
- RW-56 The import's written files are what a user trains from, and no test loads them
- RW-132 Neither end of the anchor path has a program-level test
- RW-133 YOLOv3-tiny has no training file a run can start from
- RW-48 The milestone: YOLOv3-tiny imported, trained and run from the command line

## Decisions waiting for the owner

### RW-1 Tag a first release
- State: decision
- Done when: the owner has approved 0.1.0 and `## Unreleased` becomes its section

### RW-38 Choose whether the raw cache needs a tool of its own
- State: decision
- Done when: the owner has ruled that the dataset build's `write_sample_cache` is the tool, or
  named a command that writes the cache without starting a run

The cache is written through libjpeg-turbo when a dataset with a `cache` key is built
(`modules/vision/datasets/source/task_dataset.cpp`). The YOLO26 ruling named a separate tool.

## Rulings to carry out

### RW-130 Carry out layered execution across the stack
- State: open
- Done when: tasks A0 to A7 of `docs/design/LAYERED_EXECUTION/README.md` are closed, and with them
  `sushiruntime RW-65`, SushiBLAS's B1 to B13 and SushiEngine's E1 to E17
- Source: `docs/design/LAYERED_EXECUTION/README.md`

The four repositories each hold a `docs/design/LAYERED_EXECUTION/README.md` with their share. The owner
ruled on 2026-10-19 how a run spans machines: through SushiRuntime's distributed layer, carried by
SushiBLAS, YOLOv3-tiny first. The runtime's measurements (its R13) come after the placement
labels, as ruled on 2026-10-20. A0 to A4 are built; A5 to A7 need two and five machines.

### RW-2 Make `BatchNorm2d` and a nested `ModelGraph` refuse a misfit with a `Status`
- State: open
- Done when: `NN::ModuleNode::create` returns `InvalidDescription` for a `BatchNorm2d` whose
  input is not `[H, W, channels]` and for a nested `ModelGraph` whose input does not fit or that
  has no single output
- Where: `modules/layers/nn/source/batch_norm2d.cpp`, `modules/layers/nn/source/model_graph.cpp`

Today both `output_shape` functions return a shape unchecked, and the misfit is refused only when
the model is traced.

### RW-5 Design value-dependent control flow in autograd
- State: open
- Done when: a design says how `Autograd::differentiate` treats a branch or loop that depends on
  a value; the owner took it into scope on 2026-10-19

### RW-6 Test a model file with a missing block
- State: open
- Done when: a test under `tests/vision/registration/` loads a model file with a block missing
  and checks that the `Status` names the block

## Runs not yet made

### RW-8 Train the MNIST CNN from its model file
- State: unmeasured
- Done when: `sa train examples/mnist_cnn/train.json` has run on the version 2 model file and the
  `image` layout read from the IDX files
- Source: `docs/agent/2026_10_07_CV_CLOSE/SPEC.md`

`Medium_VisionMnist.AnImageLayoutYieldsRankFourBatches` skips without the IDX files.

### RW-9 Check the demo transcript against a run
- State: unmeasured
- Done when: the transcript of `docs/guides/DEMO.md` matches a run, its node counts among them,
  and a HALF comparison under `ForceReference` has run

`Medium_Demo.MLPRunsAndReportsOneCompile` pins the compile count and not the node counts.

### RW-10 Run `sa test` around the three programs
- State: unmeasured
- Done when: `Medium_App.TrainRunsAndReportsOneCompile`,
  `Medium_App.InferenceLoadsACheckpointAndReportsOneCompile` and
  `Medium_Demo.MLPRunsAndReportsOneCompile` pass in a recorded run

## Failure contract

### RW-11 Finish the status contract
- State: open
- Done when: no `SA_THROW_IF` site throws `SushiAI::Error` where the contract asks for a `Status`
  or a `Result`, and no `try` or `catch` sits outside `seam/seam.hpp` but the ones the contract
  allows
- Source: `docs/design/PROGRAMME/README.md`

Stage 2 of the programme: 425 sites in 69 files under `modules/`, `NN::Detail::bridge_throwing`
called from 61 sites, `try` or `catch` in 30 files outside the seam, and `Scope`'s destructor
throwing through `ScopeTrace::close()`.

### RW-12 Observe that a dropped recording releases storage and scratch
- State: open
- Done when: a test counts the engine's storage and scratch before and after a failed
  `record_and_run()` and sees both back where they started

`tests/core/execution/test_execution.cpp` proves that a failed recording's tasks do not run; it
does not watch the memory.

### RW-13 Show that the fused and plain GEMM epilogues agree under `ForceReference`
- State: unmeasured
- Done when: a test runs a fused and a plain GEMM epilogue under `ForceReference`, and the
  comment of `DeviceFusionSelector` cites its result

The selector already declines every GEMM epilogue under `ForceReference`
(`modules/core/graph/source/fusion_selector.cpp`).

### RW-14 State what the gradient check's reference mode proves
- State: open
- Done when: the gradient check's documentation says the reference path shares the element
  functor with the SYCL path, so it separates launch and index errors, not formula errors

`gradcheck.hpp` names no reference mode; the entry means the engine's `ForceReference` under which
the checked graph runs.

## Eager front end

### RW-15 Build the eager units and run their tests
- State: unmeasured
- Done when: the 19 test files of the eager module pass, seven of them on an engine
- Source: `docs/agent/2026_10_07_EAGER_FRONT_END/SPEC.md`

### RW-16 Read the host cost of an eager step off a run
- State: unmeasured
- Done when: the `Medium_EagerStepCost` cases of `tests/core/eager/test_session.cpp` have printed
  the cost of an unchanged step for two and fifty layers, and of a scope that appears and
  disappears

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

### RW-27 Pad convolution and average pooling asymmetrically
- State: open
- Done when: `Conv2d` and `AvgPool2d` take trailing pads through the builder's `_with_extents`
  emitters, and the model file accepts them as `trailing_padding`, as `maxpool2d` does

### RW-29 Give the layer-name prefix of a refused window one home
- State: open
- Done when: `Conv2d`, both pools and `Upsample2d` stop catching and re-throwing the builder's
  message, and the four copies of `fit_window` become one

### RW-30 Check settings and ranks in one order across the layers
- State: open
- Done when: the geometry check of `window_extents` is its own function and every layer checks
  in one order

### RW-31 Check the leaky slope in the element type
- State: open
- Done when: a slope that narrows to 0 or 1 in float32 is refused

### RW-33 Pass the ResNet-18 on CIFAR-10 gate
- State: open
- Done when: a ResNet-18 model file and a CIFAR-10 training file exist, and the gate of the
  August design passes on them

The CIFAR reader and the augmentation pipeline exist; no basic block or projection shortcut does.

## YOLO26

### RW-35 Build the detection objective
- State: open
- Done when: code builds the loss graph of section 4 of the YOLO26 specification
- Source: `docs/agent/2026_10_07_YOLO26/PLAN.md`

### RW-36 Assign labels in two passes in the training loop
- State: open
- Done when: an anchor-free training step replays forward only, assigns with `detect/free`, and
  runs the full step in the loop

The Darknet loop splits one replay at the heads (`AnchorTrainingStep`); the YOLO26 loop is not
written.

### RW-39 Add MuSGD, learning rates per parameter group and the weight EMA
- State: open
- Done when: MuSGD and the weight EMA exist with tests, and parameter groups carry their own
  learning rates

The cosine schedule with warm-up and the weight-only decay exist.

### RW-40 Allow one HALF unit below 2^-13 between the reference path and the device
- State: open
- Done when: a mixed-precision check that compares the two paths allows one HALF ulp below
  2^-13 and bit equality above it (`sushiblas KI-3`)

### RW-41 Cover suppression and the class shift of NMS against the reference
- State: open
- Done when: a fixture has a pair above the IoU threshold and more than one class

### RW-42 Close the detect module's open interface points
- State: open
- Done when: the owner has ruled on the limits header, and `suppress_overlaps` and the
  torchvision `nms` are compared with a reference
- Source: `docs/agent/2026_10_07_YOLO26/PLAN.md`

## Domain modules

### RW-43 Split `sushiai_add_module` by concern
- State: open
- Done when: key, target, dependencies, tests and install are one function each
- Source: `docs/design/DOMAIN_MODULES/README.md`

### RW-44 Give the join that lists names one home
- State: open
- Done when: every registry and reader table builds its list of names through one function

At least eight copies: the node, layer, dataset, importer and objective registries, the reader
section, `csv.cpp` and the sampler section.

## Darknet-era YOLO

### RW-45 Follow the first SushiBLAS tag
- State: blocked
- Blocked by: sushiblas RW-8
- Done when: CI pins `SUSHIRUNTIME_REF` and `SUSHIBLAS_REF` to tags
- Source: `docs/agent/2026_10_09_DARKNET_YOLO/SPEC.md`
- Where: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`

Both refs are `main`. SushiRuntime has tag `v0.4.0`; SushiBLAS has no release tag.

### RW-46 Confirm weight-only decay against Darknet
- State: unmeasured
- Done when: a parity suite states a decay and matches Darknet, which decays convolution weights
  and neither biases nor batch norm scales

The training parity cases set momentum and decay to 0
(`modules/vision/darknet_training_cases/source/darknet_training.cpp`).

### RW-48 Train YOLOv3-tiny from the command line
- State: open
- Done when: `sushiai import darknet` of `yolov3-tiny.cfg`, `sushiai train` and
  `sushiai inference` have run end to end on a small YOLO-layout dataset and printed detections

Every link exists; the chain has never run.

### RW-49 Map boxes back through the letterbox
- State: open
- Done when: a function maps boxes from the letterboxed input to the source image, and
  `sushiai inference` prints them in source pixels

### RW-50 Delete the temporary Darknet checkout
- State: open
- Done when: the checkout outside the repository is gone, once the training tests pass

## Model import

### RW-51 Run the writers and the task section
- State: unmeasured
- Done when: `write_json`, `write_model`, `AnchorTask` and their tests pass in a recorded run
- Source: `docs/design/MODEL_IMPORT/README.md`

### RW-54 Let the `idx` reader row name its two files, and a row set the letterbox
- State: open
- Done when: both rows exist

### RW-55 Add subdivisions and options per head
- State: open
- Done when: the training file takes `subdivisions`, and each head of the anchor objective takes
  its own options
- Source: `docs/design/MODEL_IMPORT/README.md`

The restorable step count exists. A stock `yolov3-tiny.cfg` has two identical heads, and a
smaller `batch_size` stands in for subdivisions.

### RW-56 Load the files the import writes
- State: unmeasured
- Done when: the three files the import writes for each of the six models equal their expected
  text, and `load_checkpoint` accepts each written `weights.sushi`

`ParityModels/Medium_DarknetImportEndToEnd` compares the translation in memory, not the files.

### RW-57 Set detection thresholds in `eval.json` and infer on one image file
- State: open
- Done when: both work

The score threshold 0.25 and the suppression threshold 0.45 are constants in
`applications/programs/source/domains_vision.cpp`.

### RW-58 Read the imported files in the parity suites
- State: open
- Done when: the forward and training parity suites load the `model.json` and `weights.sushi`
  the import wrote, not the `.cfg`

### RW-132 Run both ends of the anchor path in a program test
- State: open
- Done when: a test under `tests/applications/programs/` trains an anchor model over a small
  YOLO-layout fixture through the train program's anchor row, and runs
  `report_anchor_detections` on the checkpoint it wrote

### RW-133 Give YOLOv3-tiny an example directory
- State: open
- Done when: `applications/examples/yolov3_tiny/` holds a `train.json` with a filled `yolo_txt`
  data block and a README a run can follow

The import writes `"data": null`, since no `.cfg` names a dataset.

### RW-134 Fit images as the imported `.cfg` says
- State: open
- Done when: the import writes the `.cfg`'s `letter_box` into the task section, and the
  detection task stretches an image under 0 and letterboxes it under 1

The owner ruled on 2026-10-25 that the model follows its `.cfg`. Today the detection task always
letterboxes (`modules/vision/datasets/source/task_table.cpp`) and the import refuses
`letter_box=1`, while Darknet stretches under the stock `letter_box=0`.

### RW-135 Report mean average precision in `sushiai inference`
- State: open
- Done when: inference prints mAP at an IoU of 0.5 against the labels of the evaluated split

### RW-136 Carry the `.cfg`'s augmentation into the training file
- State: open
- Done when: `jitter`, `hue`, `saturation` and `exposure` are training-file keys the import
  writes

The import prints a note for each and drops it.

## Datasets and tasks

### RW-59 Run `Batch` with roles and `vision/codecs`
- State: unmeasured
- Done when: both pass, with `test_learning_curve`, `test_determinism` and `test_compile_once`
- Source: `docs/design/DATASETS/README.md`

### RW-60 See the loader allocate from two threads
- State: unmeasured
- Done when: `Medium_DatasetsReaderSection.WorkersReadTheSameBatchesAhead` passes in a recorded
  run

### RW-61 Run a death test inside an engine fixture
- State: unmeasured
- Done when: `ABatchOutsideTheEpochStopsTheProcess` is seen to pass engine start-up in its child

### RW-63 Give the imaging functions a status
- State: open
- Done when: they return a status, and `fit_sample` reports each error as what it is

### RW-64 Average on reduction when resizing
- State: open
- Done when: `resize_bilinear` does not alias a large image fitted to a small input

Darknet's resize is a two-tap bilinear too, so averaging moves away from the Darknet reference.

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

### RW-71 Checkpoint inside an epoch
- State: open
- Done when: `sushiai train` saves at a batch and the written batch index is that batch

### RW-72 Test the cluster loop and the trainer run from a start other than zero
- State: open
- Done when: a test starts the cluster loop and the `Trainer` fallback run past epoch 0 and past
  batch 0, as `test_epoch_loop.cpp` does

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

### RW-77 Run segmentation
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

### RW-81 State the frame budget and training data of phase 6
- State: open
- Done when: the language design states the frame budget and the training data of phase 6, and
  names the first task after the build passes
- Source: `docs/design/LANGUAGE/README.md`

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
- Done when: `language_head`, the one linked row, takes the signature of the others and
  `LinkedNodeFactory` goes

### RW-89 Call `match_input` from the anchor loop
- State: open
- Done when: the loop matches inputs as well as roles

### RW-90 Redraw a token dataset's windows per epoch
- State: open
- Done when: `TokenDataset` implements `Data::EpochAware` and redraws its window starts at each
  epoch

### RW-91 Give `at_boundary` and the little-endian readers one home
- State: open
- Done when: the three `at_boundary` copies (`vision/datasets`, `language/text`,
  `language/datasets`) and the two `little_endian.hpp` copies are one each

### RW-92 Measure `train_bpe` on a corpus of tens of megabytes
- State: unmeasured
- Done when: `sushiai tokenize train` over a corpus of at least 20 MB is timed, and its time and
  peak memory are in `modules/language/text/README.md`

### RW-93 Settle the masked attention score
- State: open
- Done when: the masked score is a value that is minus infinity in HALF, or the language design
  records -1e9 as a decision
- Source: `docs/design/LANGUAGE/README.md`

### RW-94 Fill the dropout mask on the device
- State: open
- Done when: the mask is drawn on the device (B1 and G1 of the graph capabilities plan) and a test
  covers dropout in HALF and FLOAT64 on an engine

B1 belongs to SushiBLAS; its entry there waits until that repository's current work is done.

### RW-95 Make the tests' bodies match their names
- State: open
- Done when: `AMissingFileFails` checks a `FileNotFound` code and the path,
  `ARefusedDiceFillChangesNothing` checks all five leaves, and `DropoutRefusesOperandsThatDoNotFit`
  has the empty-input case its comment promises

### RW-96 Check the older test pages sentence by sentence
- State: open
- Done when: the test pages under `tests/` match their sources and none of their lines passes 100
  columns (227 do)

### RW-97 Describe the shared check helpers on the importer test page
- State: open
- Done when: `tests/vision/importers/README.md` links the test files that exist and describes
  the helpers of the three `darknet_*_cases` modules from their bodies

### RW-98 Check a token id against the embedding before a step
- State: open
- Done when: an input id outside `Embedding::num_embeddings` is refused with a named error before
  the step

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
- Where: `tests/vision/importers/test_darknet_training/test_yolov4.cpp`,
  `tests/vision/importers/test_darknet_training/test_yolov7.cpp`

A head is held to 0.31 of the dump in YOLOv4 and 0.0079 in YOLOv7. The differences behind those
bounds (0.205 and 0.0052) were measured on 2026-10-13, before SushiBLAS's later work, and are to
be measured again. The tiny models' differences are explained by Darknet's variance divisor and
its backward clearing a route's delta (`test_yolov3_tiny.md`).

### RW-102 Close the step time against Darknet
- State: open
- Done when: a full-size step takes the time of a Darknet iteration on the same card

Measured on 2026-10-13 and to be measured again: RTX 3080 Ti, batch 2, 200 steps, YOLOv4 0.76 s a
step against 0.21 s. Left: `Residency::Host` tensors for the heads and deltas (SushiRuntime has
the residency, `allocation.cpp` does not offer it), batch normalisation and its gradient fused
(SushiBLAS has no kernel), the deltas on the device.

### RW-103 Make the one-batch overfit of YOLOv7 hold
- State: open
- Done when: `Large_DarknetYolov7Training.TheCostOfOneBatchFallsUnderATenth` passes, the cause of
  the instability found and the bound ruled by the owner
- Where: `tests/vision/importers/test_darknet_training/test_yolov7.cpp`

It read 11.1 at step 199 against 3.56 on 2026-10-13; the run is to be made again.

### RW-104 Measure where YOLOv3's training step leaves Darknet, layer by layer
- State: unmeasured
- Done when: a per-layer dump under batch statistics shows whether the difference grows over the
  75 convolutions or enters at one
- Where: `tests/vision/importers/test_darknet_training/test_yolov3.cpp`

A head is held to 0.36 of the dump, 1.5 times the one measured difference.

### RW-105 Feed a batch without two copies
- State: blocked
- Blocked by: sushiruntime RW-59
- Done when: a step writes the batch into the plan's input buffers without two `blas().copy()`
  calls

## Fusion and the IR

### RW-106 Fuse row-wise log-softmax
- State: open
- Done when: a fusion catalog entry replaces the nine calls `K_LOG_SOFTMAX` lowers to
  (`modules/core/graph/include/SushiAI/graph/fusion.hpp`)

### RW-107 Fuse `K_NLL_LOSS`
- State: open
- Done when: it lowers to fewer than three calls

SushiBLAS's folding reduction takes a transform and a finalize functor in its dispatch, and its
public `ReductionOps` does not.

### RW-108 Skip the whole step on overflow
- State: open
- Done when: one overflowed gradient freezes every parameter for that step, through a task that
  reads the count buffer and writes a gradient
- Where: `modules/core/optim/include/SushiAI/optim/loss_scaler.hpp`

An overflowed gradient in mixed precision skips one parameter's update only, so one step updates
some layers and not others.

### RW-109 Add bfloat16
- State: open
- Done when: `MixedPrecisionPolicy` takes bfloat16

`SushiBLAS::Core::DataType` has no bfloat16 entry.

## Data and objectives

### RW-111 Check that an objective writes every leaf it declares
- State: open
- Done when: `Trainer` returns a `Status` when `feed` leaves a leaf of `ObjectiveLeaves::all()`
  unwritten, with a test whose objective skips one

## Diagnostics

### RW-113 Print per-node timings in the demo
- State: open
- Done when: `applications/examples/source/demo_mlp.cpp` turns on `RuntimeConfig::profiling` and
  prints the per-node table of `CompiledStep::last_report()`

## Graph capabilities

### RW-120 Carry out the SushiAI items of the graph capabilities plan
- State: open
- Done when: G2 to G5, G7 to G12, A2, A5, A6 and A12 of section 5 are closed
- Source: `docs/design/GRAPH_CAPABILITIES/README.md`

R1 is `sushiruntime RW-62`; R2 is `docs/design/LAYERED_EXECUTION/README.md`, `RW-130`. The B items belong to
SushiBLAS.

### RW-121 Build on SushiBLAS's views, enums and `BLASScalar`
- State: unmeasured
- Done when: `sa build` of this tree is clean against SushiBLAS `main` at 802e59d or later

No old spelling is left in the source.

### RW-122 Give `Status` a code for a runtime busy with another run
- State: open
- Done when: a call that meets a busy runtime returns that code

## Documentation

### RW-114 Compare the manual with the code
- State: unmeasured
- Done when: the numbers of `TUTORIAL.md`, sections 1 to 6 of `ARCHITECTURE.md` and the flag
  tables of `CLI_GUIDE.md` are checked against the code

## Test tree

### RW-116 Run the Darknet parity scenarios
- State: unmeasured
- Done when: the scenarios in `tests/vision/importers/test_darknet_training/` and
  `test_darknet_import/` have run with `SA_REFERENCE_DATA_DIR` set, YOLOv3-tiny's first

Without the variable they skip; the last run skipped all of them.

### RW-117 Name every suite after its unit
- State: open
- Done when: `Small_Diagnostics` in `test_backward.cpp` and `test_builder.cpp` is named after the
  unit each file tests

### RW-118 Give every test folder a README
- State: open
- Done when: the 28 test folders that hold sources and no README have one, and each README with a
  test-file table lists its pages

### RW-119 Move three tests into their mirror files
- State: open
- Done when: the cases of `check.hpp`, `load_sample_mask.hpp` and `perplexity.hpp` leave
  `test_status.cpp`, `test_fit_mask.cpp` and `test_next_token_targets.cpp`

## Documentation architecture

### RW-131 Write the About line of each design topic by hand
- State: open
- Done when: each topic has an `**About:**` line of one sentence written for a cold reader

Four of the seven end in the generator's cut.
