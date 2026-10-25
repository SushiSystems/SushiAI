# tests/vision/training/test_distribution_expectation_graph.cpp


Suites: `Medium_DistributionExpectationGraph`, `Medium_DistributionExpectationGraphEndToEnd`.

One bin returns the box and emits nothing. Four bins emit one softmax (`LOG_SOFTMAX`, `EXP`) and
one `MATMUL`. On an engine, a `[1, 1, 2, 16]` box, one hand anchor and one seeded anchor, equals
`expect_distances` on the same float32 values to 1e-5.
