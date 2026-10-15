# tests/layers/train/test_trainer/test_dropout.cpp


Suites: `Integration_Dropout`.

The probe runs DROPOUT on an engine over a `[64, 64]` FLOAT32 parameter at a rate of 0.25. Its loss
is `NLL_LOSS` over the masked value with every target at -64, which is the sum of the masked
elements, so the gradient at the input is the mask. A mask is checked against what `dropout_keeps`
answers for the seed and the run. The model cases use `Linear, ReLU, Dropout(0.2), Linear` over 8
features, 32 hidden units and 4 classes with a batch of 16.

## TheOutputIsTheInputTimesAMaskOfZeroOrTheScale

Each mask element is 0 or `1 / 0.75`, and each output is 0 or `x / 0.75`, within 1e-5. The kept
fraction of the 4096 elements is within 0.03 of 0.75, and the mask is the one of run 0.

## TwoStepsOfOneCompiledStepDrawDifferentMasks

The step registers one replay prelude. Two runs give different masks, those of runs 0 and 1, with a
compile count of 1 and a run count of 2.

## TwoFreshStepsWithTheSameSeedAgreeStepForStep

Over three runs, two fresh steps with one seed give the same output bytes, and a third step with the
seed plus one gives different bytes.

## TheGradientAtTheInputIsTheMaskOfThatStep

On each of two runs the gradient at the input equals the mask of that run within 1e-5, and the loss
equals the sum of the outputs to 1e-4 relative.

## AStepWithoutDropoutRegistersNoPrelude

A graph that only scales a `[2, 3]` input by 2 compiles to a step with zero replay preludes, and the
step doubles its input.

## AnEvaluationTraceOfAModelEmitsNoDropoutNode

The evaluation graph of the model holds no DROPOUT node and its probe graph holds one. That node
carries the rate 0.2 and the seed `dropout_site_seed` gives for the graph seed 7 and site 0.

## AModelWithDropoutTrainsThroughTheTrainer

The training graph holds at least one DROPOUT node. `Train::Trainer` fits the model with AdamW at
1e-2 on a synthetic dataset of 8 batches for 10 epochs. The report holds 80 steps and one compile,
and the last loss is finite and below the first.
