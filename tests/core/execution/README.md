# tests/core/execution

The tests of the module `core/execution`, one file per header of `modules/core/execution/include/`.
The suite prefix of a case gives its ctest label.

## test_execution.cpp

The five `Small_Execution` cases run `record_and_run` over a `HostExecutionBackend` that each case
makes and fill float32 vectors of four elements. Each case records on the tasks of that backend.

### RecordAndRunRunsWhatASuccessfulRecordingRecorded

A recording that sets a tensor of ones to 2.0 and returns success leaves every element at
2.0 after `record_and_run`.

### RecordAndRunDiscardsWhatAThrowingRecordingRecorded

A recording that sets a tensor to 5.0 and then throws `std::invalid_argument` makes
`record_and_run` fail with `InvalidDescription` and the exception's message. The tensor still
holds 1.0.

### RecordAndRunRunsARecordingThatFollowsAFailedOne

After a recording that threw, a recording that sets a second tensor to 2.0 succeeds; that tensor
holds 2.0 and the tensor of the failed recording still holds 1.0.

### RecordAndRunDiscardsWhenTheRecordingReturnsAFailure

A recording that sets a tensor to 5.0 and returns an `Unsupported` status makes
`record_and_run` fail with that status and "no host path". The tensor still holds 1.0.

### RecordAndRunSucceedsForARecordingOfNoTask

A recording that records no task runs once, runs nothing and succeeds.

## test_host_execution_backend.cpp

The `Small_StepTasksContract` cases run the shared `StepTasksContract` of
`tests/support/step_tasks_contract.hpp` against `HostExecutionBackend`, the adapter that needs no
runtime. The contract holds the promises of a `StepTasks`: a recorded task writes its values on
replay, a second replay runs the tasks again, hooks run before the tasks in the order added, a
recording of no task replays as success, and `record_and_run` runs its work once. The host adapter
also keeps the origin of each task, which `StepTasks::task_origins()` returns: a task recorded in an
origin scope returns its label and one recorded outside a scope returns an empty one
(`TheTasksReportTheOriginEachWasRecordedUnder`).

## test_runtime_execution_backend.cpp

The `Medium_StepTasksContract` cases run the same `StepTasksContract` against
`RuntimeExecutionBackend`, which records on a `SushiRuntime::Graph` of the runtime of the program
and needs that runtime, which makes the file medium.
