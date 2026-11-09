# tests/vision/training/test_distribution_focal_loss.cpp


Suites: `Small_DistributionFocalLoss`, `Medium_DistributionFocalLossEndToEnd`.

The graph case builds on the host: the loss is one float64 element. The operations that make it
are not counted. The hand
case: three uniform sides and a top side of probabilities `1..4 / 10` with target 1.25, so the
value is `0.2 * (3 ln 4 - 0.75 ln 0.2 - 0.25 ln 0.3)`; the background anchor's large logits weigh
0. The tolerance is 32 float32 unit roundoffs, because the weights are float32. The gradient check runs
`check_full_jacobian` over a FLOAT64 box with its default options.
