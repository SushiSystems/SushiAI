# tests/layers/train/test_trainer/test_learning_curve.cpp


Suites: `Medium_LearningCurve`.

This file holds the cases that read the loss along a run of 16 steps: an MLP of two linear layers
trained for 4 epochs of 4 batches by SGD with momentum 0.9 on a synthetic dataset of fixed seed.
They state what a learning model does and no number recorded from a run of this code. The first
loss of a zeroed classifier is ln(C) and the fall to a quarter of the first loss belong to
`test_convergence.cpp`, which states them once. The old case compared each step with a reference array pasted from a run on 2026-10-09 and had two typed
tolerances; a gradient scaled by the wrong constant and a legitimate kernel change both moved it,
and it could not tell them apart. It is gone. The shape assertions below hold on every toolchain.

## EachEpochImprovesOnTheOneBefore

The descent is not monotone step by step, because each step sees a different batch. The mean of
the 4 losses of epochs 1, 2 and 3 each beats the mean of the epoch before it.
