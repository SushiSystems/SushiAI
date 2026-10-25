# tests/vision/training/test_free_detection_loss.cpp

Suites: `Small_FreeDetectionLoss`, `Medium_FreeDetectionLossEndToEnd`.

The fixture is one image on a 1x2 grid with two classes. Anchor 0 is foreground with a target box
of 0.2, 0.1, 1.3, 1.4 grid units and a normalized weight of 0.7; anchor 1 is background and
carries the safe box. The graph runs in FLOAT64 and each term must equal
`evaluate_free_detection_loss` to 1e-9, at reg_max 4 with seeded logits and at reg_max 1 with an
L1 gain of 1.5.

The gradient checks run `check_full_jacobian` over the box outputs and the score logits. At
reg_max 1 the predicted box is the target moved by 0.1, so the two aspect ratios agree, CIoU's
aspect term and its derivative are 0, and holding alpha constant changes nothing. At reg_max 4
the aspect ratios differ, so that check sets the box gain to 0 and covers the class and
distribution terms.
