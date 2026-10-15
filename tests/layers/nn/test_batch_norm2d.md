# tests/layers/nn/test_batch_norm2d.cpp


BatchNorm2d's two trace forms and the running-statistic update.

Suites: `Unit_BatchNorm2d`, `Unit_BatchNorm2dDeathTest`.

Tests BatchNorm2d's two trace forms and the running-statistic update.

This file needs no device. `NN::BatchNorm2d` in `modules/layers/nn/include/SushiAI/nn/batch_norm2d.hpp` is a
modal layer. `trace_training` emits one K_BATCHNORM2D node and requests two buffer updates
through the trace. `trace_evaluation` reads the two running statistics as buffer leaves and
emits a K_BATCHNORM_FOLD and a K_CHANNEL_AFFINE. Both forms record a refusal on the trace and
return their input.

The last two cases read `tests/layers/nn/fixtures/batch_norm2d/batchnorm.sacp` through
`modules/layers/reference_files/include/SushiAI/reference_files/reference_file.hpp` and report themselves skipped when the file is absent.
They compute the updated statistics on the host from the momentum and the source scale the
trace requested. No device runs the update.

Two `static_assert` lines after the helpers hold that `BatchNorm2d` is a `ModalModule` and
neither a `Module` nor a `TraceModule`, and that `Sequential<Conv2d, BatchNorm2d>` is a
`TraceModule` and not a `Module`.

## Constants

`K_REFERENCE_EPSILON` is 1e-3 and `K_REFERENCE_MOMENTUM` is 0.03, the two values the reference
layers ran with. `K_UNIT_CASE_TOLERANCE` is 5.3e-7 and `K_MODEL_CASE_TOLERANCE` is 7.0e-7. Each
is four float32 epsilons of the case's largest running variance, 1.1086 and 1.4578.
`K_EXACT_TOLERANCE` is 1e-15, for a double computed the same way on both sides.

## count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

## only_node

`only_node` returns the first node of a graph that carries one operation. Each caller has
asserted a count of one before it.

## count_buffer_leaves

`count_buffer_leaves` counts the values of a graph whose lifetime is `BUFFER`.

## ChannelStatistics

`ChannelStatistics` holds the per-channel mean and biased variance of one NHWC activation.

## batch_statistics

`batch_statistics` computes a `ChannelStatistics` over batch, height and width from a flat
NHWC array, in double.

## updated_vector

`updated_vector` applies `NN::updated_statistic` per channel and rounds each result to
float32.

## view

`view` returns a `SushiRuntime::span<const float>` over a vector, the argument
`Test::near_reference` takes.

## expect_reference_case

`expect_reference_case` reads one case of `batchnorm.sacp`: an input and the running mean and
variance before and after one torch training step. It traces a layer of the input's channel
count with the reference epsilon and momentum, and applies the two requested updates to the
"before" statistics on the host. The results must be within the tolerance of the "after"
statistics. The variance recomputed with a source scale of 1.0 must not be, so the fixture
tells the unbiased update from the biased one.

## refused_training

`refused_training` traces the training form, expects the input id back and returns the trace's
`status()`.

## refused_evaluation

`refused_evaluation` does the same for the evaluation form.

## train_under_an_evaluation_trace

`train_under_an_evaluation_trace` calls `trace_training` with an evaluation trace. It is the
statement of a death test.

## evaluate_over_a_value_the_graph_lacks

`evaluate_over_a_value_the_graph_lacks` calls `trace_evaluation` over value 7 in a graph of
one value. It is the statement of a death test.

## ATrainingTraceEmitsOneBatchNorm

A training trace over [2, 2, 2, 3] emits one K_BATCHNORM2D node and no K_BATCHNORM_FOLD or
K_CHANNEL_AFFINE. The graph gains three nodes: the two parameter leaves and the K_BATCHNORM2D. The
node reads the input, the weight's leaf and the bias's leaf, has four outputs of which the
first is the returned value, and carries the epsilon 1e-3 in attribute 0.

## ATrainingTraceReadsNoBuffer

After a training trace the graph holds no value of lifetime `BUFFER`, and neither running
statistic has a leaf in it.

## ATrainingTraceRequestsTheTwoRunningUpdates

The list holds two updates. The first reads output 1 of the node and targets the running
mean's state, and the second reads output 3 and targets the running variance's state. The two
states are distinct and neither is `K_NO_STATE`. Both momenta are 0.03 and the mean's source
scale is 1.0.

## TheVarianceUpdateCarriesTheUnbiasedFactor

The variance update's source scale is n / (n - 1), where n is the number of values per channel.
For [2, 2, 2, 3] it is 8/7, and for [1, 20, 20, 128] it is 400/399.

## AnEvaluationTraceFoldsTheRunningStatistics

An evaluation trace over [2, 4, 4, 3] emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE and no
K_BATCHNORM2D, and the graph holds two buffer leaves. It gains six nodes: two parameter leaves,
two buffer leaves, the fold and the affine. The fold reads the weight, the bias, the running
mean and the running variance in that order and carries the epsilon 1e-3. The affine reads the
input and the fold's two outputs, and its output is the returned value of shape [2, 4, 4, 3].

## BothFormsNameTheSameState

One layer is traced in training form into one graph and in evaluation form into another. In
the evaluation graph, `state_leaf` of each update's target is the leaf of the matching running
statistic. The weight's and the bias's states are unchanged by the second trace, and each has
a leaf in the evaluation graph.

## TheWalkYieldsTwoParametersThenTwoBuffers

`NN::walk_state(layer, "bn", visitor)` over a traced five-channel layer yields "bn.weight",
"bn.bias", "bn.running_mean" and "bn.running_var" in that order, each of shape [5]. Both
parameters have a constant fill. The starting values are 1.0 for the weight, 0.0 for the bias,
0.0 for the running mean and 1.0 for the running variance.

## TrainingOverOneValuePerChannelFails

A training trace over [1, 1, 1, 3] records `InvalidDescription` with a message containing
"BatchNorm2d(3)", "more than one value per channel" and "got 1". The graph gains no value and
no node, the list holds no update, the weight is not declared, and the graph is marked refused.

## AnIntegerInputFails

Over an INT32 input, both forms record `InvalidDescription` with a message containing
"BatchNorm2d(3)", "half, float32 or float64 input" and "int32". Nothing is added to the graph
or the list, and neither the weight nor the running mean is declared.

## EvaluationOverOneValuePerChannelTraces

An evaluation trace over [1, 1, 1, 3] returns a value of that shape. The limit that
TrainingOverOneValuePerChannelFails holds belongs to the training form.

## AnInputThatDoesNotFitFails

Over a [2, 2, 2, 4] input both forms fail with a message containing "BatchNorm2d(3)" and
"[N, H, W, 3]". Over a [2, 3] input both fail with "rank-4". The graph gains no value and the
list no update.

## AnNCHWInputFails

Over a value converted to NCHW both forms fail with a message containing "BatchNorm2d(3)" and
"NHWC", and the graph gains no value.

## ALayerDescribedWrongFails

A default-constructed layer fails with "positive channel count" and "got 0". A layer with
epsilon 0.0 fails with "positive epsilon". A layer with momentum 1.5 fails with "momentum in
[0, 1]" and "1.5". The graph gains no value.

## ASecondTraceUnderAnotherElementTypeFails

A layer traced over a FLOAT32 input and then, in a second graph, over a FLOAT64 input fails
the second time with a message containing "'weight'" and both type names. The running mean's
type stays FLOAT32.

## TheScaleStartsAtOneAndTheShiftAtZero

After a trace the weight has the role `ParameterRole::Scale` and a constant fill of 1.0, and
the bias has the role `BIAS` and a constant fill of 0.0.

## TheFactoryReadsTheChannelsAndTheTwoConstants

`NN::make_batch_norm2d_node` over the shape [8, 8, 6] with `bn_epsilon` 0.001 and `bn_momentum`
0.03 returns a `ModuleNode<BatchNorm2d>` whose layer has 6 channels and those two options. The
node has one port of the input's shape and an input count of 1.

## TheFactoryRefusesARowThatDoesNotFit

The key `channels` fails with a message containing the context, "'channels'" and "bn_epsilon".
The input shape [48] fails with "[48]" and "[H, W, C]". A `bn_epsilon` of -1.0 fails with
"'bn_epsilon'" and "positive".

## TheTrainingFormUnderAnEvaluationTraceIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_training` given an evaluation
trace stops the process with a report containing "trace_training" and "was given an evaluation
trace".

## ATraceOverAValueTheGraphLacksIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_evaluation` over value 7 stops the
process with a report containing "BatchNorm2d(3)", "value 7" and "holds 1 values".

## TraceLayerPicksTheFormOfTheMode

`NN::trace_layer` under a training trace leaves one K_BATCHNORM2D and no K_BATCHNORM_FOLD in its
graph. Under an evaluation trace it leaves one K_BATCHNORM_FOLD and no K_BATCHNORM2D.

## OutputShapeIsTheInputShape

`output_shape([7, 5, 3])` is [7, 5, 3]. A three-channel layer also returns [7, 5, 4] for
[7, 5, 4]: `output_shape` does not check the channel count.

## UpdatedStatisticBlendsTheScaledBatchValue

`NN::updated_statistic(running, batch, momentum, scale)` is
(1 - momentum) * running + momentum * scale * batch. (1.0, 3.0, 0.25, 2.0) gives 2.25 and
(4.0, 8.0, 0.5, 1.0) gives 6.0. A momentum of 0 returns the running value 4.0 and a momentum
of 1 returns the batch value 8.0. Every operand and result is exact in a double.

## TheUpdatesReproduceTheUnitReferenceCase

`expect_reference_case("unit", K_UNIT_CASE_TOLERANCE)` checks the [2, 2, 2, 3] case. The
test's own comment gives the miss of the unscaled variance on this case as 8.3e-3.

## TheUpdatesReproduceTheModelReferenceCase

`expect_reference_case("model.9.cv1.bn", K_MODEL_CASE_TOLERANCE)` checks the case recorded at
that layer of the reference model. The test's own comment gives the miss of the unscaled
variance on this case as 1.4e-5.
