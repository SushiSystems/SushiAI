# tests/applications/programs

The tests of `applications/programs`. The suite prefix of a case gives its ctest label.

## source/test_training_run.cpp, source/test_inference_run.cpp, source/test_run_start.cpp

Suites: `Small_TrainingRun`, `Medium_TrainingRun`, `Small_InferenceRun`, `Medium_InferenceRun`,
`Small_RunStart` and `Medium_RunStart`, in `sushiai_programs_tests`.

The cases exercise the tables of `applications/programs`: the run table and its fallback, the report
table, and the start of a run. The pieces of a run are an MLP of 6 features, 12 hidden units and 4
classes, its classifier graph for a batch of 8, SGD at a rate of 0.1 and a synthetic dataset of 8
batches. A classification run takes 2 epochs.

### ASecondRowForOneTypeIsRefused

In `Small_TrainingRun`, a second row for 'anchor' is refused as `InvalidDescription` with
"already registered". `Small_InferenceRun` holds the same case for 'classification' in the report
table.

### ARunRunsTheRowOfItsType, ARunOfAnotherTypeRunsTheFallback

With both a row for 'anchor' and a fallback registered, 'anchor' runs the row's stub, which reports
a compile count of 7, and 'cross_entropy' runs the fallback's, which reports 9.

### AnUnknownTypeWithoutAFallbackIsRefusedNamingTheKnownOne

Without a fallback, the type 'cross_entropy' is refused with a message that names it and 'anchor'.

### AClassificationRunCompilesOnceAndTakesEveryStep, AClassificationRunPrintsAFallingLoss

`register_runs` sets a fallback. A 'cross_entropy' run through it reports one compile, and the
optimizer has taken 16 steps. The run prints two losses, and the last is below the first.

### TheReportRunsTheRowOfItsType, AnUnknownTaskTypeIsRefusedNamingTheKnownOne

A report for 'classification' runs its row. A report for 'pose' is refused as `InvalidDescription`,
and the message names "model.json", 'pose' and 'classification'. A second row for 'classification'
is refused with "already registered".

### Small_RunStart and Medium_RunStart

`describe_start` returns "initialised" for a spec without files, the path for a spec with `weights`
and "resumed from " and the path for a spec with `resume`. A run is trained and saved with its run
state to a file in the temporary directory. `start_state` leaves the optimizer at step 0 for a spec
without files and for `weights`, and for `resume` it restores the trained run's 16 steps.

## source/test_training_setup.cpp

Suites: `Medium_TrainingSetup`.

### TracesTheDataBatchWhenNoRowsAreGiven, BindsTheParametersAndTheStateOfTheModel

A setup without rows traces the batch of the data, 4. It binds parameters and holds a state with
entries.

### HoldsOnlyTheResultsOfTheStepInTheResultsScope

A setup of `SetupScope::Results` over the cluster test files has no plan in its pool, binds the
loss and not the inputs, and holds a gradient with data for every parameter slot, as many slots as
a setup of `SetupScope::Step`. The pool of the `Results` setup allocates fewer bytes than the pool
of the `Step` setup, which has a plan.

## source/test_training_cluster_section.cpp

Suites: `Small_TrainingClusterSection` and `Small_TrainingClusterSectionLocalFallback`.

### IsRefusedNamingBothKeys

A version 3 file that does not state `deterministic`, one that states `"deterministic": true` and a
version 2 file, which cannot state the key, each have a second replica stating
`if_absent: "local"`. Each is refused with `InvalidDescription`; the message names replica 1, the
key `'if_absent'` and the key `'deterministic'`.

### ALocalFallbackIsReadWhenTheRunIsNotDeterministic

The same section in a version 3 file with `"deterministic": false` is read.

### WaitAndFailAreReadUnderADeterministicRun

A section whose second replica states `fail`, or `wait` with `wait_ms` 250, is read under the
default.

## source/test_training_replica_model.cpp

Suites: `Medium_TrainingReplicaModel`, `Medium_ReplicaFingerprintEntry`,
`Medium_ReplicaFingerprintChanged` and `Medium_ReplicaFingerprintUnchanged`.

### AModelReportsTheFingerprintOfItsSource

The model a source builds reports the fingerprint of that source.

### TwoReadsOfTheSameFilesGiveOneFingerprint

Two sources made from the same files, through `load` and through `make`, carry one fingerprint.

### AChangedPartChangesTheFingerprint, AnUnrelatedPartLeavesTheFingerprintAlone

One row per part of the training file or of the architecture file that moves the fingerprint, and
one per part that leaves it alone.

## source/test_training_program.cpp, source/test_inference_program.cpp

Train the `tiny_yolo` fixture, two convolutions and one anchor head over four 16 by 16 images,
for two steps through the `anchor` run row (`run_training_file`), then report its checkpoint
through the `anchor_detection` inference row (`run_inference_file`).
`source/tiny_yolo_files.hpp` holds the training and evaluation files both share.
[`source/test_training_program.md`](source/test_training_program.md) and
[`source/test_inference_program.md`](source/test_inference_program.md) describe each case.

## source/test_detection_report.cpp, source/test_detection_reports.cpp, source/test_vision_training_runs.cpp

Test the headers `detection_report.hpp`, `detection_reports.hpp` and `vision_training_runs.hpp`
directly, without the program assembly. `source/prepared_inference_harness.hpp` calls
`prepare_inference` over the `tiny_yolo` fixture with initialised weights, and
`source/prepared_run_harness.hpp` calls `prepare_run` over its training file; the programs call the
same two functions, so no build sequence exists twice. A test changes the
member of the harness a promise reads: the task section, the input shape, the thresholds, the start.
`source/role_hidden_dataset.hpp` serves the fixture's batches without one role, for the cases that
need data with no `fit` or no `boxes`. The cases are described in
[`source/test_detection_report.md`](source/test_detection_report.md),
[`source/test_detection_reports.md`](source/test_detection_reports.md) and
[`source/test_vision_training_runs.md`](source/test_vision_training_runs.md).

## source/test_inference_preparation.cpp, source/test_run_preparation.cpp

Suites: `Medium_InferencePreparation` and `Medium_RunPreparation`, in `sushiai_programs_tests`.
They test `prepare_inference` and `prepare_run` over the `tiny_yolo` fixture. The cases are described in
[`source/test_inference_preparation.md`](source/test_inference_preparation.md) and
[`source/test_run_preparation.md`](source/test_run_preparation.md).

## source/test_image_input.cpp

Builds the data block of one image file from a model of an 8 by 6 input and a task of 3 classes, and
reads the classes from the task or, when the task has none, from the last layer that has them, and refuses a model with neither. `is_image_path` is decided by the extension alone.

## source/test_profile_session.cpp

Suites: `Medium_ProfileSession`, in `sushiai_programs_tests`.

Each case opens `ProfileSession` on the shared runtime with a configuration written to a scratch
folder. The good configuration follows `train.step` for two periods and writes `run.sprec.json`
there.

### AMissingConfigNamesTheFile

`open` on a path that does not exist fails with `FileNotFound`, and the message names the file.

### ABadJsonNamesTheFile

A file whose JSON ends early fails with `InvalidDescription` naming the file.

### AnUnknownKeyNamesTheFile

A configuration with the key `bogus_key` fails with `InvalidDescription`; the message names the file
and the key.

### EndingAFailedRunReturnsItsOwnFailure

Three `train.step` marks run under the session, and `end_profiled_run` is given a `DeviceFailed`
status. It returns that status, not the result of the close.

### ARunThatFailsStillWritesItsRecord

After the same `end_profiled_run` call, the record file holds two periods.

### ADroppedSessionStillWritesItsRecord

A session that goes out of scope without `close()`, as it does when an exception leaves the run,
writes a record of two periods.

### ASecondCloseSucceeds

Two `close()` calls both succeed, and the record is the one the first wrote.

## source/test_profile_arguments.cpp

Suites: `Small_ProfileArguments`, `Small_ProfileArgumentsPosition` and
`Small_ProfileArgumentsRefusal`, in `sushiai_programs_tests`.

### ALineWithoutTheFlagIsKept

`train.json` alone comes back as the one positional argument, with no configuration.

### ThePairIsTakenOutWhereverItStands

`--profile p.json` before, between and after three positional arguments, one row each, leaves the
three in order and reads `p.json`.

### ALineThatBreaksTheFlagIsRefused

A trailing `--profile`, a `--profile` followed by another flag and two `--profile` pairs, one row
each, fail with `InvalidDescription`, the first two with "needs the path" and the last with "given
twice".

## source/test_host_readback.cpp

Suite: `Medium_HostReadback`. Its cases are described in
[source/test_host_readback.md](source/test_host_readback.md). They read a device-resident tensor
back through `HostReadback`.

## source/test_model_state_entries.cpp

Suite: `Small_ModelStateEntries`. Its cases are described in
[source/test_model_state_entries.md](source/test_model_state_entries.md). They list the state records
of the tiny run's model file without an engine.

## source/test_weights_import.cpp

Suite: `Small_WeightsImport`. Its cases are described in
[source/test_weights_import.md](source/test_weights_import.md). They check the composed weight
sources and that a failed import writes no file.

## source/test_weights_export.cpp

Suite: `Small_WeightsExport`. Its cases are described in
[source/test_weights_export.md](source/test_weights_export.md). They check the composed weight
writers and that a refused export writes no file.

## source/test_model_export.cpp

Suite: `Small_ModelExport`, in a build with vision. Its cases are described in
[source/test_model_export.md](source/test_model_export.md). They export one fixture per format
through the composed registries and import the files back.

## source/test_share_runtime.cpp

Suite: `Small_ShareRuntime`. `narrow_to_split` over a configuration of 3 workers and no
accelerator: a file without `device_split` leaves the filter and the worker count as they were;
a split whose first share is `gpu` admits GPUs alone, and one whose first share is `cpu` admits
CPUs alone.

## source/test_device_split_run.cpp

Suite: `Medium_DeviceSplitRun`. The suite opens a runtime narrowed to GPUs once and checks that a
runtime narrowed to CPUs finds a CPU device; on a machine without both, every case is skipped.
Each case trains the tiny run's model file over `rings` (batches of 8, two batches, two epochs,
`adamw`) through `run_training_file` on the GPU backend.

- `TheSplitStepRunsOnTheGpuAndOnTheCpu`: a split of 6 GPU rows and 2 CPU rows reports two shares,
  GPU then CPU, each with its rows and 4 replays, and the optimizer takes 4 steps.
- `TheSplitStepReachesTheGpuOnlyLoss`: the last printed loss of that split is within 1e-3 of the
  last printed loss of the same file without a split. The CPU and the GPU sum in other orders, so
  the two differ by rounding only.
- `WithoutASplitTheRunReportsNoShare`: a file without `device_split` reports no share and prints
  neither the `Devices` line nor a `share` line.
- `SharesThatMissTheBatchSizeAreRefused`: 6 and 1 rows return `InvalidDescription` naming 7 rows
  and batches of 8.

## source/test_row_feed.cpp

Suite: `Medium_RowFeed`, on the shared test engine. Each case builds the tiny run's setup traced
over 2 rows and enters epoch 0 of its `rings` data.

- `AFeedCopiesItsRowsIntoTheInputs`: a feed of rows 6 and 7 of batch 0 leaves the input leaf equal
  to what the dataset reads for those rows.
- `AFeedPastTheBatchIsRefused`: a feed of rows 7 to 9 of a batch of 8 returns
  `InvalidDescription`.

## source/test_share_step.cpp

Suite: `Medium_ShareStep`. Each case builds a CPU share of the tiny run that takes the last 2 rows
of each batch, on a runtime of its own; on a machine without a CPU device the case is skipped.

- `AShareReplaysItsStepOncePerBatch`: after batches 0 and 1 the share reports the CPU class, 2
  rows, 2 replays and a device name.
- `AShareHoldsEveryParameterOfTheModel`: the share holds 4 parameters, a weight and a bias for each
  linear layer.
- `AShareRefusesABatchPastTheEpoch`: batch 2 of a file with two batches returns
  `InvalidDescription`.

The tiny run's training file the three split-step suites build is in `source/tiny_run_files.hpp`.
