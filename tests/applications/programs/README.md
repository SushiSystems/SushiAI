# tests/applications/programs

The tests of `applications/programs`. The suite prefix of a case gives its ctest label.

## source/test_training_run.cpp, source/test_inference_run.cpp, source/test_run_start.cpp

Suites: `Integration_ProgramRuns`, in `sushiai_programs_tests`.

The cases exercise the tables of `applications/programs`: the run table and its fallback, the report
table, and the start of a run. The pieces of a run are an MLP of 6 features, 12 hidden units and 4
classes, its classifier graph for a batch of 8, SGD at a rate of 0.1 and a synthetic dataset of 8
batches. A classification run takes 2 epochs.

### TheRunTablePicksARowOrTheFallback

A second row for 'anchor' and a second fallback are each refused as `InvalidDescription` with
"already registered". Without a fallback, the type 'cross_entropy' is refused with a message that
names it and 'anchor'. With both registered, 'anchor' runs the row's stub, which reports a compile
count of 7, and 'cross_entropy' runs the fallback's, which reports 9.

### AClassificationRunTrainsThroughTheFallback

`register_runs` sets a fallback. A 'cross_entropy' run through it reports one compile, and the
optimizer has taken 16 steps. The run prints two losses, and the last is below the first.

### TheReportTableRefusesAnUnknownTask

A second row for 'classification' is refused with "already registered". A report for
'classification' runs. A report for 'pose' is refused as `InvalidDescription`, and the message names
"model.json", 'pose' and 'classification'.

### ARunStartsFromWhatItsFileStates

A run is trained and saved with its run state to a file in the temporary directory. For a spec
without files `describe_start` returns "initialised" and the optimizer stays at step 0. With
`weights` set it returns the path, and the step count stays 0. With `resume` set it returns "resumed
from " and the path, and the step count equals the trained run's 16.


## source/test_training_setup.cpp

Suites: `Medium_TrainingSetup`.

### HoldsOnlyTheResultsOfTheStepInTheResultsScope

A setup of `SetupScope::Results` over the cluster test files has no plan in its pool, binds the
loss and not the inputs, and holds a gradient with data for every parameter slot, as many slots as
a setup of `SetupScope::Step`. The pool of the `Results` setup allocates fewer bytes than the pool
of the `Step` setup, which has a plan.

## source/test_training_cluster_section.cpp

Suites: `Small_TrainingClusterSection`.

### ALocalFallbackUnderTheDefaultIsRefusedNamingBothKeys

A version 3 file that does not state `deterministic` and whose second replica states
`if_absent: "local"` is refused with `InvalidDescription`; the message names replica 1,
`'if_absent': 'local'` and `'deterministic' is true`. A file that states `"deterministic": true`
and a version 2 file, which cannot state the key, are refused the same way.

### ALocalFallbackIsReadWhenTheRunIsNotDeterministic

The same section in a version 3 file with `"deterministic": false` is read. A section whose
second replica states `fail`, or `wait` with `wait_ms` 250, is read under the default.

## source/test_training_replica_model.cpp

Suites: `Medium_TrainingReplicaModel`.

### TheFingerprintKeepsItsValue

For an architecture file holding `{"sushiai_model": 1}` and the objective `cross_entropy`, the
fingerprint of `ReplicaSource::make` is 0xd043c6822d22eede, a value computed apart from SushiAI.

## source/test_training_program.cpp, source/test_inference_program.cpp

Train the `tiny_yolo` fixture, two convolutions and one anchor head over four 16 by 16 images,
for two steps through the `anchor` run row (`run_training_file`), then report its checkpoint
through the `anchor_detection` inference row (`run_inference_file`).
`source/tiny_yolo_files.hpp` holds the training and evaluation files both share.
[`source/test_training_program.md`](source/test_training_program.md) and
[`source/test_inference_program.md`](source/test_inference_program.md) describe each case.

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

### ARunThatFailsStillWritesItsRecord

Three `train.step` marks run under the session, and `end_profiled_run` is given a `DeviceFailed`
status. It returns that status, not the result of the close, and the record file holds two
periods.

### ADroppedSessionStillWritesItsRecord

A session that goes out of scope without `close()`, as it does when an exception leaves the run,
writes a record of two periods.

### ASecondCloseSucceeds

Two `close()` calls both succeed, and the record is the one the first wrote.

## source/test_profile_arguments.cpp

Suites: `Small_ProfileArguments`, in `sushiai_programs_tests`.

### ALineWithoutTheFlagIsKept

`train.json` alone comes back as the one positional argument, with no configuration.

### ThePairIsTakenOutWhereverItStands

`--profile p.json` before, between and after three positional arguments leaves the three in order
and reads `p.json`.

### AFlagWithNoPathOrGivenTwiceIsRefused

A trailing `--profile`, a `--profile` followed by another flag and two `--profile` pairs each fail
with `InvalidDescription`, the first two with "needs the path" and the last with "given twice".
