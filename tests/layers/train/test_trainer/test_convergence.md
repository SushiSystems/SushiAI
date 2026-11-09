# tests/layers/train/test_trainer/test_convergence.cpp


Suites: `Medium_Training`, `Medium_TrainingDataset` and `Small_TrainingBinding`.

This is the end-to-end gate: a real MLP on the synthetic classification dataset that has to
converge, and the single-step rules of the two optimizers. The compile-once promise is in
`test_compile_once.cpp` and is not repeated here. The binding cases run on the host backend; every
case that trains on a device stays `Medium_`.

## InitialLossIsExactlyLogCWithAZeroedClassifier

Zeroing the last layer makes every logit zero, so every class is equally likely and the initial
cross-entropy of five classes over a batch of 8 is ln(5), within 51 rounded operations of float:
the log-softmax rounds 10, the product with the target once, the sum of the 40 products 39 times at
worst and the mean once.

## TheStepAllocatesNothingInSteadyState

Replaying reuses the same buffers, so a long run's memory does not grow. After one step, twelve
more steps over the four batches leave the pool with the same tensor count and the same bytes.

## SGDDrivesTheLossDownOnASeparableProblem

An MLP of 8 features, 24 hidden units and 4 classes is fitted to a synthetic dataset of 6 batches of
16 for 30 epochs, with SGD at a rate of 0.2 and a momentum of 0.9. The last loss is below a quarter
of the first.

## SGDReachesAHighAccuracyOnASeparableProblem

The same fit ends at an accuracy above 0.9.

## AdamWDrivesTheLossDownOnASeparableProblem

The same model and data with AdamW at a rate of 0.02. The last loss is below a quarter of the first
and the accuracy is above 0.9.

## AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate

The "eps after the bias correction" ordering exists to preserve this property. On step one,
m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so every coordinate moves by almost exactly the
learning rate. Folding eps inside the correction makes it 31.6x larger and visibly damps this
first step, the only place the bug is easy to see.

All 12 weights of a `Linear` of 4 features and 3 classes are checked, none skipped. The distance of
a coordinate's move from the learning rate is bounded by eps / (|g| + eps), which the coordinate's
own gradient gives, plus 8 rounded operations of float on the parameter and the rate.

## SGDWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient

One step of SGD at a rate of 0.1 over the same `Linear`. Each of the 12 weights after the step
equals its value before minus 0.1 times its gradient, within 2 rounded operations of float over the
product and the result.

## ADisconnectedParameterIsRejectedRatherThanSilentlySkipped

The test fakes a parameter that no gradient reaches. It stands in for a layer wired into the
model list but not into the forward pass. The refusal is decided on the binding's name and leaf, so
it runs on the host backend.

## TwoDatasetsOfOneSeedHoldTheSameInputs

One row per batch index (0, 1, 2). Each dataset gets a fresh Engine on purpose. SushiBLAS's
`Engine::set_seed()` sets the seed and leaves the monotonic RNG offset counter alone, so a second
generator built on an engine that has already drawn gets a different stream from the same seed.
Reproducibility is therefore a property of a fresh engine, which is how the trainer uses it. The
inputs of the two datasets are equal, and the dataset holds 3 batches under the name
`synthetic-classification`.

## EveryBatchHoldsEachClassTwice

One row per batch index. Round-robin labels make every batch of 8 hold each of the 4 classes
exactly twice, so accuracy reads correctly without class weighting.

## RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays

The model is Unflatten, a batch norm over three channels of a 2 by 2 grid, Flatten and a Linear
layer. It is trained for two steps on two synthetic batches. After each step, the running mean of
every channel equals the `NN::updated_statistic` of that step's channel mean, with the default
momentum and a source scale of 1. The bound is the sum of 32 values and the 3 rounded operations of
the blend, applied to the magnitudes the two paths carry. After the second step, every channel's
running mean differs from its value after the first step.

## APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm

The same model is trained with the per-step optimizer placement. After one step the loss is
positive, the step count is 1, and `3.weight` differs from its value before the step. The test
notes that the classifier starts at zero, so the first gradient reaches its weight alone. That the
optimizer is outside the plan is `test_training_step.cpp`'s case and is not read from the trainer
here.

## ABatchOfAnotherShapeIsRefusedWithBothShapes

A trainer compiled for a batch of 8 samples with 6 features refuses a batch of 4 samples. The
refusal is `Incompatible`, and its message contains "[4, 6]" and "[8, 6]". The trainer compiles
and trains on a device, so the case stays `Medium_`.

## BoundSlotsDecayByTheRoleOfTheirParameter

The classifier graph of a two-layer MLP binds four slots: `0.weight` and `2.weight` decay, and
`0.bias` and `2.bias` do not. The binding runs on the host backend.
