# tests/vision/training/test_distribution_expectation_graph.cpp


Suites: `Small_DistributionExpectationGraph`, `Medium_DistributionExpectationGraphEndToEnd`.

One bin returns the box and emits nothing. Four bins give a float64 `[1, 1, 2, 4]` value; the
operations that make it are not counted, and both cases build on the host. On an engine, a
`[1, 1, 2, 16]` box, one hand anchor and one seeded anchor, equals `expect_distances` on the same
float32 values to 256 float32 unit roundoffs, and the hand anchor's means 2 and 1 hold.
