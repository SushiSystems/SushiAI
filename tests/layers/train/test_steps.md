# tests/layers/train/test_steps.cpp


Suites: `Unit_Steps`.

Tests the builders that trace a model into a training, probe or evaluation graph, and the batch
shape function.

This file needs no device. `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`
and `batch_shape` are declared in `modules/layers/train/include/SushiAI/train/steps.hpp`, and `Train::TrainingGraph`
is declared in `modules/layers/train/include/SushiAI/train/training_graph.hpp`. The constants `K_BATCH` (4),
`K_FEATURES` (6), `K_HIDDEN` (8), `K_CLASSES` (3), `K_CHANNELS` (3) and `K_SIDE` (2) set every
shape in the file. Each case reads the graph a builder returns, through its nodes, its leaves,
its pinned values, its buffer updates or its status.

## K_MLP_TRAINING_OPS

`K_MLP_TRAINING_OPS` lists the operation of each of the 17 nodes of the MLP's training graph
under `CrossEntropyObjective`, in node order.

## parameter_states

`parameter_states` returns the `state()` of every parameter of a model, in the order of
`NN::for_each_parameter`.

## build_two_output_model

`build_two_output_model` is `Test::build_two_output_model` from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`,
which this file brings in with a using-declaration. It builds a Linear named "body", a GELU named
"activation" and a Linear named "head" over [6], and publishes "features" on the GELU and
"logits" on the head.

## NamedOutputObjective

`NamedOutputObjective` is a `Train::Objective` that declares one leaf, "targets", of
[batch, classes]. Its `loss()` is the cross-entropy of the output it was given by name against
that leaf. `score()` returns no metric, and `name()` returns "named_output".

## ATrainingGraphOfAnMLPHasTheNodesBuildGraphGave

The training graph of the MLP under cross-entropy has 17 nodes. Their operations, in node order,
equal `K_MLP_TRAINING_OPS`.

## ATrainingGraphListsEveryParameterWithItsLeafInThatGraph

The training graph lists four parameters, in the order of the parameter walk, and each entry
carries the walk's name. Each entry's value is the parameter's `id_in` in the built graph, and
neither its value nor its gradient is `K_NO_VALUE`.

## ATrainingGraphListsEveryParameterWithItsRole

The training graph of the two-layer MLP lists four parameters with the roles `Weight`,
`Bias`, `Weight`, `Bias`.

## WeightDecayAppliesToWeightsOnly

`takes_weight_decay` is true for `Weight` and false for `Bias`, `Scale` and `Embedding`. A
default `ParameterBinding` has the role `Weight`.

## ABatchNormModelYieldsTwoUpdatesAndPinsTheirSources

A `BatchNorm2d`, `Flatten` and `Linear` model over [4, 2, 2, 3] yields two buffer updates, and
the source of each update is pinned.

## ATrainingGraphPinsTheLossAndTheOutputs

For the same model, the loss and every value in `outputs` are pinned.

## AProbeGraphHasBatchStatisticsAndNoUpdate

The probe graph of a `BatchNorm2d` over [4, 2, 2, 3] has one `BATCHNORM2D` node, no
`BATCHNORM_FOLD` node and no buffer leaf, and it publishes one output.

## AnEvaluationGraphFoldsTheStoredStatistics

The evaluation graph of the same layer has one `BATCHNORM_FOLD` node, two buffer leaves and no
`BATCHNORM2D` node.

## OneModelTracedThreeTimesKeepsOneStateIDPerParameter

After the training graph, the probe graph and the evaluation graph are built from one
batch-norm classifier, its four parameters have the same state identities as after the training
graph alone, in walk order. Each state has a leaf in all three graphs.

## AModelGraphWithTwoOutputsLeavesLogitsUnset

`NamedOutputObjective("logits")` reads the model of `build_two_output_model`. The training graph
has `spec.logits` set to `K_NO_VALUE`, two entries in `outputs` and a set `spec.loss`.

## ARefusingLayerFailsTheBuilderWithItsStatus

`build_probe_graph` of a `BatchNorm2d(3)` over [1, 1, 1, 3] returns `StatusCode::InvalidDescription`,
with a message containing "BatchNorm2d(3)" and "more than one value per channel".

## BatchShapePrependsTheBatch

`batch_shape(16, [28, 28, 1])` is [16, 28, 28, 1]. A batch of 0 returns
`StatusCode::InvalidDescription`, with a message containing "batch" and "positive".

## AnEvaluationGraphAndAProbeGraphFeedTheModelInputAlone

The steps `build_probe_graph` and `build_evaluation_graph` trace list no fed leaf and no
carried value.

## ACarryingGraphFoldsTheStoredStatisticsAndUpdatesNoBuffer

For a `BatchNorm2d`, `build_carrying_graph` gives the same operation names as
`build_evaluation_graph`: no BATCHNORM2D node, two buffer leaves, no fed leaf and no carried
value.

## AnIncrementalGraphNamesTheNodeThatHasNoIncrementalForm

`build_incremental_graph` of a model whose one node is a `Linear` named "dense" fails with
`Unsupported`, "node 'dense'" and "no incremental form".
