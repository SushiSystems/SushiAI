# tests/core/engine_fixture/test_engine_test.cpp


Suites: `Medium_EngineTest`.

Tests that `EngineTest` gives each case an engine on the shared runtime, and that its `flush`
returns.

The engine needs the runtime, which makes the file medium.

## SetUpGivesTheCaseAnEngine

The fixture's `engine` is not null inside the case.

## FlushReturnsWithNothingRecorded

`flush` returns when the engine holds no recorded work. A failed execute or wait would stop the
process, so reaching `SUCCEED` is the check.

## TheEngineAllocatesATensorAndFlushes

`allocate_tensor` on the engine yields a [2, 3] tensor of 6 elements with storage, and `flush`
returns afterwards.
