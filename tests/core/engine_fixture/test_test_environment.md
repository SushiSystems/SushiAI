# tests/core/engine_fixture/test_test_environment.cpp


Suites: `Medium_TestEnvironment`.

Tests that `TestEnvironment` creates one runtime and hands it to every caller.

The runtime opens a device, which makes the file medium.

## EveryCallReturnsTheSameRuntime

Two calls to `runtime()` return the same address.

## TheRuntimeWaitsWithNothingSubmitted

`wait()` on the shared runtime succeeds when nothing was submitted.
