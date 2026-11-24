# tests/layers/umbrella/test_SushiAI

The test sources kept in this folder. The suite prefix of a case gives its ctest label.

| File | Tests |
| --- | --- |
| `compile_noexcept_surface.cpp` | Checks at compile time that every public function of the core and layers modules is noexcept. |
| `test_engine_smoke.cpp` | Tests that the umbrella header gives an engine and a task recorder that run. |
