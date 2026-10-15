# tests/layers/umbrella

The tests of the module `layers/umbrella`, one file per header of `modules/layers/umbrella/include/`. The suite prefix of a case gives its ctest label.

## test_SushiAI.cpp

Suites: `Medium_EngineSmoke`.

### AXPYThroughSushiBLAS

This is the P6 "consumer-shaped example" for AI-0. It proves that the two-sibling D3 dependency
chain, SushiRuntime and SushiBLAS both linked through SushiAI, compiles, links and runs a real
operation end to end. It does not stand in for the autograd operation coverage of AI-1.
