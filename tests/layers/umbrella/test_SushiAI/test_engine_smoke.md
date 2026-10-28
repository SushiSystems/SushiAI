# tests/layers/umbrella/test_SushiAI/test_engine_smoke.cpp


Suites: `Medium_EngineSmoke`, `Small_UmbrellaDistribution`.

Tests that the library links and runs through the umbrella header `SushiAI/SushiAI.hpp`, and that a
distributed build names every distribution header through it.

### AXPYThroughSushiBLAS

This is the P6 "consumer-shaped example" for AI-0. It proves that the two-sibling D3 dependency
chain, SushiRuntime and SushiBLAS both linked through SushiAI, compiles, links and runs a real
operation end to end. It does not stand in for the autograd operation coverage of AI-1.

### RegressionNamesEveryDistributionHeader

The case compiles only in a build that defines `SA_TEST_DISTRIBUTED_BUILD`. It names
`ClusterSection`, `GradientExchange`, `ParameterPublish`, `ReplicaPosition` and `RowPositions`
as classes and checks that `sums_exactly` accepts `UInt32`.
