# tests/layers/umbrella/test_SushiAI/test_engine_smoke.cpp


Suites: `Medium_EngineSmoke`.

Tests that the library links through the umbrella header `SushiAI/SushiAI.hpp` and gives an
engine and a task recorder that run. The value an axpy yields is SushiBLAS's own promise and is
tested there. The distribution headers the umbrella names are checked by
`compile_noexcept_surface.cpp`.

### TheUmbrellaGivesAnEngineAndATaskRecorderThatRun

This is the P6 "consumer-shaped example" for AI-0. It proves that the two-sibling D3 dependency
chain, SushiRuntime and SushiBLAS both linked through SushiAI, compiles, links and records one
operation that a graph then runs to completion. It does not stand in for the autograd operation
coverage of AI-1.
