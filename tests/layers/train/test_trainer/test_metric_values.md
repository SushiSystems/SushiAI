# tests/layers/train/test_trainer/test_metric_values.cpp


Suites: `Small_MetricValues` and `Medium_MetricValuesTrainer`.

Tests `Objective::declare_metric_values` of `objective.hpp`, which an objective uses to publish
extra graph values to its metrics, and what the builders of `steps.hpp` and the trainer do with
them. The `Small_` cases build graphs over a Linear, GELU, Linear model of 6 features, 8 hidden
units and 3 classes at a batch of 4, and need no device. The test objective `PublishingObjective`
takes the loss and the leaves of a cross entropy and publishes the INT64 argmax of the model's
only output under a name it is given.

## TheDefaultObjectiveDeclaresNoValue

The training graph of a `CrossEntropyObjective`, which overrides nothing, holds no metric value.

## ATrainingGraphHoldsAndPinsThePublishedValue

The training graph of the publishing objective holds exactly one metric value, `classes`. Its
value is INT64 of shape [4] and is among the pinned values of the graph, so the planner does not
pack it.

## AnEvaluationGraphHoldsThePublishedValue

`build_evaluation_graph` given the publishing objective holds `classes` as INT64 of shape [4].

## AnEvaluationGraphWithoutAnObjectiveHoldsNoValue

`build_evaluation_graph` given no objective holds no metric value.

## ATrainingGraphHandsTheHookWhatTheLossKept

`TraceReadingObjective` keeps the model's only output in the `LossTrace` of `loss()` under `kept`
and publishes the argmax of what the trace holds under the same name. The training graph holds
exactly that one metric value, `kept`, so the hook received the trace `loss()` returned.

## AnEvaluationGraphHandsTheHookAnEmptyTrace

`build_evaluation_graph` given the trace-reading objective holds no metric value: an evaluation
graph has no loss, so the hook gets the default `LossTrace`, which holds no `kept`.

## TheFailureOfTheHookEndsTheBuild

An objective whose hook fails with `InvalidDescription` ends `build_training_graph` and
`build_evaluation_graph` with that status, naming "refuses to declare 'classes'".

## ASumKeepsTheValuesOfItsTermsOnceByName

A `SumObjective` of three publishing terms, named `classes`, `classes` and `scores`, holds two
metric values in the order `classes`, `scores`: of two values of one name the first term's is kept.

## TheTrainerHandsThePublishedValueToMeasure

`Medium_MetricValuesTrainer`, on the shared engine. A Linear of 3 inputs and 3 classes trains one
step of a batch of four rows at a learning rate of 0. The recording objective's `measure` reads
the value the step holds under `classes` and the logits of the step, and notes for each row the
class of the value and the column of the largest logit, the lowest on a tie, which the host finds
with `std::max_element`. The two lists are equal over the four rows.
