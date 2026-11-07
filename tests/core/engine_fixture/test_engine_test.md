# tests/core/engine_fixture/test_engine_test.cpp


Suites: `Medium_EngineTest`.

Tests that `EngineTest` gives each case an engine on the shared runtime and the backend over them.

The engine needs the runtime, which makes the file medium.

## SetUpGivesTheCaseAnEngine

The fixture's `engine` is not null inside the case.

## BackendRunsOnTheEngineOfTheCase

The engine of the fixture's `backend()` is the engine the case records on.
