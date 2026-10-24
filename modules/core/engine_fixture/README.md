# engine_fixture

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

The one `SushiRuntime::Runtime` a test program shares, the `main()` of every test program,
`EngineTest`, the fixture that gives each test a fresh `SushiBLAS::Engine` on that runtime, and
`TestGraph`, a graph a test owns with a `SushiBLAS::TaskRecorder` over it and the
`SushiBLAS::LateCountHold` the recorder keeps late-count tasks in. A test never records on
the engine's own graph: it builds a `TestGraph`, records through its `recorder()` and calls
`run()`. `EngineTest::flush()` only waits for the device.

`Detail::create_runtime()` starts a worker pool, probes the topology and builds SYCL queues.
Doing that per test would dominate the run and restart the thread pool while other work is
settling, so `TestEnvironment` keeps one runtime for the whole program and `EngineTest` gives
each test its own `Engine`, so no test inherits another's graph or random offset. `SetUp()`
fails the run with the message of the `Status` when the runtime cannot be created.

`source/test_main.cpp` holds `main()`, so a test program does not link `GTest::gtest_main`;
linking both is a duplicate-symbol error at link time. Both loggers are switched to synchronous
mode, because asynchronous logging interleaves with the output of GTest.

## Depends on

`execution`, for `Detail::create_runtime()`, `failure`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/engine_fixture/test_environment.hpp` | `TestEnvironment` |
| `SushiAI/engine_fixture/engine_test.hpp` | `EngineTest` |
| `SushiAI/engine_fixture/test_graph.hpp` | `TestGraph` |
