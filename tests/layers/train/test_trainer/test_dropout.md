# tests/layers/train/test_trainer/test_dropout.cpp


Suites: `Medium_Dropout` and `Medium_DropoutSeeds`.

The probe runs DROPOUT on an engine over a `[64, 64]` FLOAT32 parameter at a rate of 0.25. Its loss
is `NLL_LOSS` over the masked value with every target at -64, which is the sum of the masked
elements. The model case uses `Linear, ReLU, Dropout(0.2), Linear` over 8 features, 32 hidden units
and 4 classes with a batch of 16.

The draw itself is not checked against `Graph::dropout_keeps`, since that is the code under test;
`tests/core/graph/test_dropout_mask.cpp` holds the draw and `tests/core/graph/test_compiled_step.cpp`
holds that each run takes the mask of its run count. The gradient being the mask is held by
`tests/core/autograd/rules/test_random.cpp`, and the two trace modes of the layer by
`tests/layers/nn/test_dropout.cpp`.

## TheOutputIsTheInputTimesAMaskOfZeroOrTheScale

Each mask element is exactly 0 or the float `1 / 0.75`. Each output is the input times the mask, at
a relative distance of at most one rounding of float.

## TheKeptFractionIsAboutTheKeepProbability

The kept fraction of the 4096 elements is within four standard deviations of 0.75, where one
deviation is sqrt(0.75 * 0.25 / 4096), about 0.0068.

## TwoFreshStepsWithTheSameSeedAgreeStepForStep

Over three runs, two fresh steps with one seed give the same output bytes.

## AStepWithAnotherSeedDiffersAtEveryRun

Over three runs, a step with the seed plus one gives output bytes that differ from those of the
first step.

## TheLossIsTheSumOfTheMaskedOutput

After one run the loss equals the sum of the outputs, within the roundings of a sum of 4096 values.
This is the only case that reads the loss of the masked value.

## AStepWithoutDropoutRegistersNoPrelude

A graph that only scales a `[2, 3]` input by 2 compiles to a step with zero replay preludes. What
the step computes is the compiled-step promise of `test_compiled_step.cpp`.

## AModelWithDropoutTrainsThroughTheTrainer

The training graph holds at least one DROPOUT node. `Train::Trainer` fits the model with AdamW at
1e-2 on a synthetic dataset of 8 batches for 10 epochs. The report holds 80 steps and one compile,
and the last loss is finite and below the first.
