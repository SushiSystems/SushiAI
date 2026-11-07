# engine_fixture

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The one `SushiRuntime::Runtime` a test program shares, the `main()` of every test program,
the one `SushiBLAS::Engine` on that runtime, the one `RuntimeExecutionBackend` over the two
(`TestEnvironment::backend()`, made once per program beside them), `EngineTest`, the fixture
that gives each test that engine and `backend()`, and `TestGraph`, a `StepTasks` a test takes
from a backend, with the `SushiBLAS::TaskRecorder` of those tasks. A test never records on
the engine's own graph: it builds a `TestGraph` over `backend()`, records through its
`recorder()` and calls `run()`, which seals and replays the tasks once. `EngineTest::flush()`
only waits for the device.

`Detail::create_runtime()` starts a worker pool, probes the topology and builds SYCL queues.
Doing that per test would dominate the run and restart the thread pool while other work is
settling, so `TestEnvironment` keeps one runtime for the whole program, held in a `LazyShared`
whose factory is `Detail::create_runtime()` and which `TestEnvironment(factory)` replaces. `EngineTest::SetUp()`
rewinds the engine's mode and random stream, so no test inherits another's random offset. A
runtime that cannot be created ends the process with its message.

`source/test_main.cpp` holds `main()`, so a test program does not link `GTest::gtest_main`;
linking both is a duplicate-symbol error at link time. Both loggers are switched to synchronous
mode, because asynchronous logging interleaves with the output of GTest.

## Depends on

`execution`, for `Detail::create_runtime()`, `RuntimeExecutionBackend` and `StepTasks`, `failure`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/engine_fixture/lazy_shared.hpp` | `LazyShared` |
| `SushiAI/engine_fixture/test_environment.hpp` | `TestEnvironment` |
| `SushiAI/engine_fixture/engine_test.hpp` | `EngineTest` |
| `SushiAI/engine_fixture/test_graph.hpp` | `TestGraph` |
