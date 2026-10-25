# tests/vision/training/test_distribution_focal_loss.cpp


Suites: `Medium_DistributionFocalLoss`, `Medium_DistributionFocalLossEndToEnd`.

The loss is one value from one `LOG_SOFTMAX`, with no `GATHER_ROWS` and no `NLL_LOSS`. The hand
case: three uniform sides and a top side of probabilities `1..4 / 10` with target 1.25, so the
value is `0.2 * (3 ln 4 - 0.75 ln 0.2 - 0.25 ln 0.3)`; the background anchor's large logits weigh
0. The tolerance is 1e-6 because the weights are float32. The gradient check runs
`check_full_jacobian` over a FLOAT64 box with its default options.
