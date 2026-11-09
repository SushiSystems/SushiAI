# tests/layers/train/test_steps.cpp


Suites: `Small_Steps`, `Small_StepsParameter` (one row per parameter of the MLP),
`Small_StepsUpdate` (one row per buffer update), `Small_StepsState` (one row per parameter of
the batch-norm classifier), `Small_StepsForward` (one row per builder of a forward graph:
probe, evaluation, carrying) and `Small_StepsFold` (evaluation and carrying).

Tests the builders that trace a model into a training, probe or evaluation graph, and the batch
shape function.

This file needs no device. `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`,
`build_carrying_graph` and `batch_shape` are declared in
`modules/layers/train/include/SushiAI/train/steps.hpp`, and `Train::TrainingGraph` is declared in
`modules/layers/train/include/SushiAI/train/training_graph.hpp`. The constants `K_BATCH` (4),
`K_FEATURES` (6), `K_HIDDEN` (8), `K_CLASSES` (3), `K_CHANNELS` (3) and `K_SIDE` (2) set every
shape in the file. The training-graph cases read the loss, the gradients, the bindings and the
pinned values a builder returns. The probe, evaluation and carrying cases run the graph of one
`BatchNorm2d` of two channels on `Graph::HostEvaluator` and compare its elements with values
worked by hand.

## K_PROBE_OPERATIONS

17 rounded operations on the longest path to an element the probe writes: the mean of four values
rounds 3 additions and a division (4), the centred value (1), its square (1), 3 additions and a
division (4) give the variance, then the epsilon sum, the root and the inverse (3), then the
centred value, the product with the inverse, the gain and the shift (4).

## K_FOLD_OPERATIONS

6 rounded operations on the longest path to an element a folded layer writes: the scale rounds
the epsilon sum, the root and the quotient (3), the shift rounds the product of the mean with the
scale and the difference (2), and the element rounds the product with the scale and the sum with
the shift; the shift path is the longer, 3 + 2 + 1 = 6.

## parameter_states, parameter_names, parameter_leaves

Each returns, for every parameter of a model in the order of `NN::for_each_parameter`, its
`state()`, its name or its `id_in` of a graph.

## write_norm_state and run_layer

`write_norm_state` writes a gain, a shift, a stored mean and a stored variance into the leaves of
a `BatchNorm2d` that a graph holds, and skips a leaf the graph lacks. `run_layer` writes the
image whose channel 0 is {1, 3, 5, 7} and channel 1 is {2, 2, 2, 2} (epsilon 4) together with the
gain {2, 1}, the shift {0, 1}, the stored mean {-1, 0} and the stored variance {5, 0}, runs the
graph and reads its one output. A leaf the graph holds and the run does not write makes the
run fail, so a graph that reads a buffer it should not shows up there.

## build_two_output_model

`build_two_output_model` is `Test::build_two_output_model` from
`modules/layers/sample_models/include/SushiAI/sample_models/two_output_model.hpp`. It builds a Linear
named "body", a GELU named "activation" and a Linear named "head" over [6], and publishes
"features" on the GELU and "logits" on the head.

## NamedOutputObjective

`NamedOutputObjective` is a `Train::Objective` that declares one leaf, "targets", of
[batch, classes]. Its `loss()` is the cross-entropy of the output it was given by name against
that leaf. It stays a stub on purpose: no repository objective reads a model output by name,
and the case that uses it tests the builder's refusal to set the logits, not the objective.

## ATrainingGraphOfAnMLPHasAScalarLossAndTheLogits

The training graph of the MLP under cross-entropy sets `spec.loss` and `spec.logits`. The loss
has the shape [1] and the logits the shape [4, 3]. The old case compared the operation of each of
the 17 nodes with a list pasted from an earlier run; it asserted the graph's inside, and a fusion
change would have broken it with the promise intact.

## ATrainingGraphGivesEveryParameterAGradientOfItsShape

One row per parameter index 0 to 3: the binding holds a gradient, and its shape equals the
shape of the parameter's value.

## ATrainingGraphListsEveryParameterWithItsLeafInThatGraph

One row per parameter index 0 to 3: the binding carries the walk's name at that index, its value is
set, and it is the parameter's `id_in` in the built graph.

## ATrainingGraphListsEveryParameterWithItsRole

The training graph of the two-layer MLP lists four parameters with the roles `Weight`,
`Bias`, `Weight`, `Bias`. `WeightDecayAppliesToWeightsOnly` is gone: the same promise is tested
in `tests/layers/train/test_parameter_groups.cpp`, and a default `ParameterBinding` role is a
default value.

## ABatchNormModelYieldsTwoUpdates

A `BatchNorm2d`, `Flatten` and `Linear` model over [4, 2, 2, 3] yields two buffer updates.

## TheSourceOfAnUpdateIsPinned

One row per update, 0 and 1: its source is in the pinned values, since the host reads it.

## ATrainingGraphPinsTheLoss and ATrainingGraphPinsTheOutput

For the same model the loss is pinned, and the one output in `outputs` is pinned.

## AProbeGraphNormalisesWithTheStatisticsOfItsBatch

The probe of a `BatchNorm2d` of two channels, epsilon 4, over [1, 2, 2, 2]. Channel 0 =
{1, 3, 5, 7} has mean 4 and variance 5, channel 1 = {2, 2, 2, 2} has mean 2 and variance 0, so the
inverse deviations are 1/3 and 1/2. With gain {2, 1} and shift {0, 1} the result is
{-2, 1, -2/3, 1, 2/3, 1, 2, 1}. The stored statistics written beside differ from the batch's, and
the probe has no buffer leaf to receive them: the result shows that the probe uses the batch
statistics, and that it reads and updates no buffer.

## AGraphThatFoldsReadsTheStoredStatistics

Rows: evaluation and carrying. With the stored mean {-1, 0} and variance {5, 0} and epsilon 4 the
scales are 2/3 and 1/2 and the shifts 2/3 and 1, so the same image becomes
{4/3, 2, 8/3, 2, 4, 2, 16/3, 2}. These values differ from the probe's, so the result shows that
the stored statistics are read; the buffers have no update because the graph has no node that
writes one.

## AGraphPublishesTheOneOutputOfTheLayer and AGraphFeedsTheModelInputAloneAndCarriesNothing

Rows: probe, evaluation and carrying, over a `BatchNorm2d(3)`: one output; no fed leaf and no
carried value.

## OneModelTracedThreeTimesKeepsOneStateIDPerParameter

One row per parameter index 0 to 3. After the training graph, the probe graph and the evaluation
graph are built from one batch-norm classifier, the parameter has the same state identity as
after the training graph alone, and that state has a leaf in all three graphs.

## AModelGraphWithTwoOutputsLeavesLogitsUnset

`NamedOutputObjective("logits")` reads the model of `build_two_output_model`. The training graph
has `spec.logits` set to `K_NO_VALUE`, two entries in `outputs` and a set `spec.loss`.

## ARefusingLayerFailsTheBuilderWithItsStatus

`build_probe_graph` of a `BatchNorm2d(3)` over [1, 1, 1, 3] returns `StatusCode::InvalidDescription`,
with a message containing "BatchNorm2d(3)" and "more than one value per channel".

## BatchShapePrependsTheBatch

`batch_shape(16, [28, 28, 1])` is [16, 28, 28, 1].

## BatchShapeRefusesABatchThatIsNotPositive

A batch of 0 returns `StatusCode::InvalidDescription`, with a message containing "batch" and
"positive".

## AnIncrementalGraphNamesTheNodeThatHasNoIncrementalForm

`build_incremental_graph` of a model whose one node is a `Linear` named "dense" fails with
`Unsupported`, "node 'dense'" and "no incremental form".
