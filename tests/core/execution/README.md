# tests/core/execution

The tests of the module `core/execution`, one file per header of `modules/core/execution/include/`.
The suite prefix of a case gives its ctest label.

## test_execution.cpp

The `Medium_ExecutionEngine` cases record on the engine of `SushiAI::Test::EngineTest` and fill
float32 vectors of four elements.

### RecordAndRunRunsWhatASuccessfulRecordingRecorded

A recording that sets a tensor of ones to 2.0 and returns success leaves every element at
2.0 after `record_and_run`.

### RecordAndRunDiscardsWhatAFailedRecordingRecorded

A recording that sets a tensor to 5.0 and then throws `std::invalid_argument` makes
`record_and_run` fail with `InvalidDescription` and the exception's message. A following
recording sets a second tensor to 2.0; that tensor holds 2.0 and the first still holds 1.0.

### RecordAndRunDiscardsWhenTheRecordingReturnsAFailure

A recording that sets a tensor to 5.0 and returns an `Unsupported` status makes
`record_and_run` fail with that status and "no host path". After `run_recorded` the tensor
still holds 1.0.
