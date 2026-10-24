# tests/core/graph/test_step_recording.cpp


Suites: `Medium_StepRecording`.

Tests that `Graph::StepRecording` hands back the task recorder and the runtime recorder it was
built with. The case builds a graph on the shared runtime of `TestEnvironment` and runs nothing.

## HandsBackTheRecordersItWasBuiltWith

`tasks()` returns the task recorder given and `graph()` the graph's recorder, both by address.
