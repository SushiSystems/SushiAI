# tests/layers/train/test_trainer/test_learning_curve.cpp


Suites: `Regression_LearningCurve`.

This file holds the test that catches "everything passes and the model stops training".

Every other test in the suite checks a property: this gradient matches a finite difference, that
step compiled once, this loss is finite. A model can satisfy all of them and still learn measurably
worse than it did last month. A gradient scaled by the wrong constant, a momentum buffer applied a
step late and an initialisation gain quietly changed all do this. Each leaves the loss going down
along a different path, and no property test has an opinion about the path.

So TheLossTrajectoryMatchesTheRecordedReference pins the trajectory itself against a reference
recorded from a real run of this code. The tolerance carries the design of the test:

- Too tight, and the test fails on every legitimate kernel change. Reassociating a gemm's inner
  sum moves an fp32 result by an ULP or two, and twelve steps of momentum amplify that. A tolerance
  below about 1e-3 would be noise.
- Too loose, and the test stops distinguishing training from not training. A gradient wrong by a
  constant factor, or a backward rule reading the forward output where it needed the input, moves
  this trajectory by tenths.

The head steps, 0 to 5, use the absolute bound K_LOSS_TOLERANCE, 0.02, which sits between those
two bounds. The tail steps, 6 to 15, use K_LOSS_RELATIVE_TOLERANCE, 2 per cent of each step's own
reference. From step 6 on the reference values are below 0.02, so an absolute bound there would
accept a loss ten times its reference: at step 15 the reference is 0.002215 and the bound is 0.02.
The relative bound pins each tail step to its own magnitude. The shape assertions in
TheTrajectoryHasTheShapeALearningModelHas (the exact ln(C) start, the per-epoch descent, the final
value) hold even if a future toolchain widens the noise floor.

Recording the reference again is legitimate only when the change that moved it is understood and
intended. If the trajectory moved and nobody knows why, the movement is the finding and the
tolerance stays where it is.

## TheTrajectoryHasTheShapeALearningModelHas

The shape assertions are stated independently of the recorded numbers, so they survive a legitimate
re-record. Together they say that the trajectory starts where a zeroed classifier must start, goes
down, and gets most of the way to the minimum.

The descent is not monotone step by step, because each step sees a different batch. Each epoch's
mean must beat the mean of the epoch before it.
