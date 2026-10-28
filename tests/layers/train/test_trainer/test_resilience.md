# tests/layers/train/test_trainer/test_resilience.cpp


Suites: `Medium_Resilience`, `Medium_ResilienceDeathTest`.

The file checks what the library does when it is handed something it cannot serve, and what it
does at the edges of what it can. The happy path belongs to `test_forward.cpp` and
`test_training.cpp`. This file makes three other claims.

1. Every device-lane precondition fails loudly and names the actual problem.
   `modules/core/failure_expectations/include/SushiAI/failure_expectations/status_expectations.hpp` asserts the message and not only the code.
2. Degenerate and boundary extents produce right answers and not off-by-one ones: one sample,
   one feature, one class, exactly the GEMM register-block edge and exactly one past it. Index
   arithmetic breaks here or nowhere.
3. Numbers that would make a naive implementation produce NaN do not. A NaN is the worst
   failure this library has. It propagates into every weight on the next update, the loss
   stops moving, and nothing reports an error. Finiteness is therefore asserted explicitly and
   not inferred from a loss that looks plausible.

The file is laid out in four groups: loud failures that name the problem, degenerate and
boundary extents, numerical extremes, and resource discipline.

## AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch

A `[4, 3]` parameter with a `[4, 2]` gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.weight", "12 elements" and "gradient has 8".

## AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType

A float32 parameter with a float64 gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.bias", "float32" and "float64".

## AnOptimizerRefusesADTypeItCannotServeAndExplainsWhy

Complex is not a training dtype at all. Half is a real float, and it is refused for a reason
the message has to carry: the refusal exists to prevent a model that compiles and trains
worse.

## AdamWNamesTheHyperparameterThatIsOutOfRange

`AdamW::make` refuses a `beta1` of 1.0 and a `beta2` of -0.5, each with its name and "[0, 1)" in the
message, and an `eps` of 0 with "eps must be positive". All three are `InvalidDescription`.

## ADisconnectedParameterNamesItselfAndSaysWhatToDo

A binding named "ghost" with no gradient value is appended to a classifier's parameters.
`bind_parameters` refuses it as `InvalidDescription`, and the message contains 'ghost', "receives no
gradient" and "no optimizer could ever move it".

## AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage

The library has no `zero_grad()`, by design. Every gradient buffer in a step is written and
not accumulated into (see `optimizer.hpp`). That leaves one window in which a gradient has
never been written: between `bind_parameters()` and the first replay. Stepping in that window
would feed whatever the allocator last had in that memory straight into the weights. Nothing
downstream catches it. The parameters become garbage or NaN, and the loss never recovers.

## TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep

A step's buffers are addresses baked into a compiled plan. A batch of the wrong size cannot be
accommodated, only reported. The refusal names both shapes, [4, 6] and [8, 6].

## FitNamesTheDatasetDimensionThatDisagrees

A trainer compiled for a batch of 8 and 4 classes refuses three datasets. One without batches is
`InvalidDescription` with "The dataset holds no batches". A batch size of 4 is `Incompatible` with
"batch size is 4" and "compiled for 8". Five classes are `Incompatible` with "5 classes" and
"compiled for 4".

## TheSyntheticDatasetNamesEveryExtentWhenOneIsNotPositive

`SyntheticClassification::make` refuses zero batches, a batch size of zero and zero classes as
`InvalidDescription`. Each message contains "positive extents" and the extent that is zero, as
"batches=0", "batch=0" or "classes=0".

## AllocationRefusesAnEmptyOrUnrepresentableTensor

Signed overflow in the extent product is undefined behaviour, and a compiler may assume it
away along with any check written after it. The overflow is detected before the multiply, so
the diagnostic is about the shape and not about what the overflow left behind.

## AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel

The request is a petabyte, which no allocator serves. What matters is what the caller is told:
the shape and element type it asked for. A bare byte count from four frames down would not
help, and neither would a null pointer for a kernel to dereference.

## ASingleSampleWithASingleFeatureTrains

One `Linear` over a batch of one sample with one feature and three classes. The first loss is ln(3)
within 1e-5, the loss stays finite over eight more steps at a rate of 0.5, and the weight and its
gradient are finite afterwards.

## ASingleClassIsDegenerateButNotUndefined

With C = 1 the softmax is identically one, so the loss is exactly zero and the gradient
(p - y) / N is exactly zero. Nothing learns, which is correct. A 0/0 anywhere in the recipe
would show up as NaN.

## MatmulIsExactAtTheRegisterBlockEdgeAndOnePastIt

The local-tiled GEMM stages a 64x64 block of float per work-group and pads the ragged
remainder with zeros. An extent of exactly 64 exercises the padding-free path. 65 leaves a
one-row, one-column remainder, where an off-by-one in the boundary test shows up as a wrong
answer and not as a fault. Both are checked against a host reference, with the transpose flag
the Linear layer emits.

The values that reached the device are float. The reference is computed from those same
floats, or the comparison would measure rounding and not indexing.

## TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt

The case runs for n = 64 and n = 65, with n as the batch size, the feature count and the hidden
width. Over four steps of SGD at a rate of 0.1 every loss is finite, and so is every parameter and
every gradient afterwards.

## LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude

The max shift is the only reason these inputs are representable. Without it exp(1e30) is inf
and exp(-1e30) is zero, and both give NaN. With it every magnitude must land on the same
answer, because a uniform row is uniform whatever its offset.

## LogSoftmaxSurvivesBothExtremesInsideOneRow

The row holds +1e30 against -1e30. The shift drives the winner to exactly zero and every loser
to -2e30, whose exponential underflows to zero without becoming a NaN. The row sum is exactly
one and log(1) is exactly zero, so the winner's log-probability is exactly zero. That is a
probability of one, the right answer for a logit gap of 2e30.

## LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange

The two rows' own maxima are 60 apart. The case once pinned the supported side of a boundary.
The shift was by the batch's global maximum, so a row far enough below it came back +inf, and
60 was inside the range that still worked.

The boundary is gone: the shift is each row's own maximum, so the gap does not reach the
arithmetic. The case is kept unchanged. It is the shape a training run produces, and a
regression that brought back a global shift would still pass every test whose rows sit on top
of each other. The gaps that break a global shift are pinned separately, with their measured
errors, in `tests/core/graph/lowering/test_loss.cpp`.

Both rows have the same internal spread, so both must produce the same log-probabilities. That
is the claim of shift invariance.

## CrossEntropyIsFiniteOnPerfectlyConfidentPredictions

The logits carry a confidence of exp(80) either way. Anything that exponentiates without
shifting first overflows here, and anything that takes the log of a rounded zero produces
-inf.

With the same logits against the opposite targets the loss is large, and large is the correct
answer. It must not be inf, because an infinite loss makes every gradient below it a NaN.

## AdamWsFirstStepWithAZeroGradientStaysFinite

m and v are both exactly zero on step one, so the update is 0 / (sqrt(0 / bc2) + eps). Only
the eps in the denominator keeps that a zero and not a NaN. eps is added after the bias
correction, so it is not scaled away. Folding it inside makes the denominator
sqrt(0 + eps) / sqrt(bc2), which is still non-zero here and wrong everywhere else. Dropping it
makes this 0/0.

A second step still has no gradient to work with, and the bias corrections have changed. A
divide that only happened to be safe on step one would fail there.

## BuildingAndDestroyingTrainersReachesASteadyState

Every `SushiBLAS::Storage` allocates through the runtime, which records each allocation, so
`runtime.advanced().live_allocation_count()` is the number of live allocations. Whether the last
iteration leaked a buffer is then a number the test can read, where otherwise somebody would have
to watch Task Manager. One leaked Storage moves the count by exactly one.

The baseline is taken after the first iteration and not before it. A one-time lazy cache
inside the runtime is tolerated; a per-iteration leak is not.

## BatchNormUpdateStatsIsNotAnIROperationAndSaysSo

CV-7 keeps a BatchNorm's running statistics out of the IR. They are recorded into the training step
beside the graph, the same way `modules/core/optim/source/` emits K_SGD_UPDATE and K_ADAMW_UPDATE
directly. K_BATCHNORM_UPDATE_STATS is therefore one of the three identities listed in `K_UNLOWERED`
(`modules/core/graph/include/SushiAI/graph/lowering/unlowered.hpp`), and it has no row in
`K_LOWERINGS`. That is deliberate.

From the outside a deliberate gap and an omission look the same, which is why the test exists.
Pinning the fatal stop records the decision. If somebody later adds a lowering, the test fails and
asks them to say why.
