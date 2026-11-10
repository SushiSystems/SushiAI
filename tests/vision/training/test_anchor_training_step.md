# tests/vision/training/test_anchor_training_step.cpp


Suites: `Medium_AnchorTrainingStep` and `Small_AnchorTrainingStep`.

Tests `AnchorTrainingStep` of `anchor_training_step.hpp` on a two-head case: a model with a stem,
a biased running-mean norm and two heads of 14 channels (`head.0` finer than `head.1`), a batch of
2, an SGD optimizer and a truth of boxes. The cases that train run on the CPU SYCL device through
`EngineTest`, so they are medium. The six refusals below are decided before a task is launched, so
they run in `Small_AnchorTrainingStep` on a `HostExecutionBackend`. `test_detection_truth_check.cpp` tests why a truth is refused;
the step calls those checks.

### TheLeavesFedAreTheDeltasOfTheHeads

After one run, each delta leaf equals the delta `compute_head_delta` gives its head alone on the
forward-only output, element for element, with the assignment counts of the fixture. The stored
cost is the sum of the squared deltas and the total is that sum over the batch, each to 16777216
double unit roundoffs of the sum.

### TheShapeRatioRuleFeedsItsDeltasAndItsLoss

Under the shape-ratio rule, with squared sigmoid sides and a stretch of 2, each leaf equals the
delta that rule computes, both heads assign something, the reported cost equals the rule's total,
and its box and non-box shares add up to it.

### OneStepMovesAHeadBiasAlongItsDelta

The bias of `head.0`'s convolution moves by the learning rate over the batch times the sum of its
channel's deltas over the positions, to 256 float32 unit roundoffs, and some channel moves by more
than the visible bound.

### OneStepBlendsARunningMeanOnce

The running mean of the stem norm after one step is `(1 - momentum)` times its start plus
`momentum` times the batch mean of the stem, to the same bound. A second blend would add 0.09 of
the mean, which the case shows is visible.

### ACentreOutsideTheInputIsRefused

A valid box whose centre lies at the right edge fails with `InvalidDescription` naming "truth row 0
of image 1" and "centre". The check sits in `compute_head_delta`; this case keeps the wiring.

### MakeRefusesAPoolWithoutTheInputLeaf

`make` over a pool in which the input leaf is not bound fails with `InvalidDescription` naming
"input leaf" and "bound in the training pool".

### AnObjectiveNamingAnUnpublishedHeadIsRefused

An objective that names `head.9` fails in `make` with `InvalidDescription` naming `head.9`.

### AHostStageThatReadsAValueTheGraphLacksIsRefused, AHostStageBehindAReaderOfItsLeafIsRefused, AStepWithoutAHostStageRefusesHostWork

These three refuse in `Train::TrainingStep`, not in the anchor step. They run on the host backend
because that backend is the seam `layers/train` offers for them (REPORT.md, decision 78); no change to
`Train::TrainingStep` is made.
