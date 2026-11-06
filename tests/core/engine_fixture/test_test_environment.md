# tests/core/engine_fixture/test_test_environment.cpp


Suites: `Medium_TestEnvironment`.

Tests that the runtime `TestEnvironment` shares accepts a wait. That `runtime()` creates one
runtime and hands it to every caller is the promise of `LazyShared`, the holder of that runtime,
which `test_lazy_shared.cpp` tests.

The runtime opens a device, which makes the file medium.

## TheRuntimeWaitsWithNothingSubmitted

`wait()` on the shared runtime succeeds when nothing was submitted.
