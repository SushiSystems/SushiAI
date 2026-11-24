# tests/core/engine_fixture

The test sources of `modules/core/engine_fixture` kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `test_engine_test.cpp` | Tests that EngineTest gives each test an engine and the backend over it. |
| `test_lazy_shared.cpp` | Tests that LazyShared creates its value once, on first use, through its factory. |
| `test_test_environment.cpp` | Tests that the runtime TestEnvironment shares accepts a wait. |
| `test_test_graph.cpp` | Tests that TestGraph records on a graph of its own and runs it once. |
