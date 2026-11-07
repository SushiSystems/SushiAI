# tests/core/engine_fixture/test_test_graph.cpp

Suites: `Medium_TestGraph`.

Tests that `TestGraph` gives a case a recording of its own from the backend and a recorder over it, and that `run`
executes what the recorder recorded.

The recording needs the runtime, which makes the file medium.

## RunSucceedsWithNothingRecorded

`run` on a recording that holds no task returns success.

## RunExecutesWhatTheRecorderRecorded

A constant fill of 7.0 recorded through `recorder()` on a float32 tensor of four elements leaves
every element at 7.0 after `run` and `flush`.
