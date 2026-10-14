# tests/applications/programs

The tests of `applications/programs`. The suite prefix of a case gives its ctest label.

## source/test_training_run.cpp, source/test_inference_run.cpp, source/test_run_start.cpp

Suites: `Integration_ProgramRuns`, in `sushiai_program_tests`.

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
