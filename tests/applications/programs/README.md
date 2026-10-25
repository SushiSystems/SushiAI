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
