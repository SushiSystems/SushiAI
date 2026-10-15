# tests/layers/train/test_trainer/test_convergence.cpp


Suites: `Medium_Training`.

This is the end-to-end gate: a real MLP on the synthetic classification dataset that has to
converge, with the compile-once guarantee checked across every step.

## InitialLossIsExactlyLogCWithAZeroedClassifier

Zeroing the last layer makes every logit zero, so every class is equally likely and the
initial cross-entropy is ln(C) to the last digit. The test can assert that number; a loss that
merely looks reasonable would have to be judged by eye.

## TheStepIsCompiledExactlyOnceAcrossManySteps

This is the AI-1 headline gate. A loop that recompiles per step still converges and pays graph
analysis every step. That regression hides for months unless something counts compilations.

## TheStepAllocatesNothingInSteadyState

Compile-once is half the claim. The other half is that replaying reuses the same buffers, so a
long run's memory does not grow.

## SGDDrivesTheLossDownOnASeparableProblem

An MLP of 8 features, 24 hidden units and 4 classes is fitted to a synthetic dataset of 6 batches of
16 for 30 epochs, with SGD at a rate of 0.2 and a momentum of 0.9. The first loss is ln(4) within
1e-4, the last is below a quarter of the first, the accuracy is above 0.9 and the compile count is
1.

## AdamWDrivesTheLossDownOnASeparableProblem

The same model and data with AdamW at a rate of 0.02. The last loss is below a quarter of the first,
the accuracy is above 0.9 and the compile count is 1.

## AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate

The "eps after the bias correction" ordering exists to preserve this property. On step one,
m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so every coordinate moves by almost exactly the
learning rate. Folding eps inside the correction makes it 31.6x larger and visibly damps this
first step, the only place the bug is easy to see.

The loop skips coordinates whose gradient is below 1e-4, where eps does matter. The claim is
about the ordinary case.

## SGDWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient

One step of SGD at a rate of 0.1 over a `Linear` of 4 features and 3 classes. Each element of the
first parameter after the step equals its value before minus 0.1 times its gradient, within 1e-6.

## ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes

The shared test runtime is created without `RuntimeConfig::profiling`, so the runtime records
no timing. The report has to say that. A table of zeroes would read as if the work took no
time.

## ADisconnectedParameterIsRejectedRatherThanSilentlySkipped

The test fakes a parameter that no gradient reaches. It stands in for a layer wired into the
model list but not into the forward pass.

## TheSyntheticDatasetIsDeterministicAndBalanced

Each dataset gets a fresh Engine on purpose. SushiBLAS's `Engine::set_seed()` sets the seed and
leaves the monotonic RNG offset counter alone, so a second generator built on an engine that
has already drawn gets a different stream from the same seed. Reproducibility is therefore a
property of a fresh engine, which is how the trainer uses it.

Round-robin labels make every batch exactly balanced, so accuracy reads correctly without
class weighting.

## RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays

The model is Unflatten, a batch norm over three channels of a 2 by 2 grid, Flatten and a Linear
layer. It is trained for two steps on two synthetic batches. After each step, the running mean of
every channel equals the `NN::updated_statistic` of that step's channel mean, with the default
momentum and a source scale of 1, within 1e-5. After the second step, every channel's running mean
differs from its value after the first step by more than 1e-7.

## APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm

The same model is trained with the per-step optimizer placement. After one step, the optimizer is
not in the plan, the step count is 1, and at least one element of `3.weight` has changed by more
than 1e-7. The test notes that the classifier starts at zero, so the first gradient reaches its
weight alone.

## ABatchOfAnotherShapeIsRefusedWithBothShapes

A trainer compiled for a batch of 8 samples with 6 features refuses a batch of 4 samples. The
refusal is `Incompatible`, and its message contains "[4, 6]" and "[8, 6]".

## BoundSlotsDecayByTheRoleOfTheirParameter

The classifier graph of a two-layer MLP binds four slots. The first is named `0.weight` and decays,
the second is `0.bias` and does not, and the third and fourth follow the same pattern.
