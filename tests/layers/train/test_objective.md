# tests/layers/train/test_objective.cpp


Suites: `Small_Objective`, `Medium_ObjectiveScore`.

Tests the objective seam: the leaves it declares, the loss it emits, the resulting IR, and what
`feed` and `measure` do with a batch.

The `Small_Objective` cases need no device. Declaring leaves and emitting a loss are IR-to-IR,
so that part of the seam is testable without an Engine. The `Medium_ObjectiveScore` fixture is
an `EngineTest`, because `feed` and `measure` read tensors.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIR. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

## BuildGraphAndBuildClassifierProduceTheSameIR

`build_classifier` and `build_training_graph` with a `CrossEntropyObjective`, each over its
own model of the same widths, give graphs with the same value count, node count and
description, and specs with the same `inputs`, `targets`, `logits` and `loss` ids.

## TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

## CrossEntropyRequiresTheLabelsRole

The training graph under `CrossEntropyObjective(4, 3)` requires one role: `K_ROLE_LABELS`
with an empty sample shape, `Int32` and 3 classes.

## CrossEntropyRefusesAnOutputOfAnotherShape

`CrossEntropyObjective(4, 3)` over a model output of shape [4, 5] returns an `InvalidDescription`
status that names both shapes, in place of throwing.

## CrossEntropyRefusesAModelWithTwoOutputs

The same objective over a model that publishes two outputs returns an `InvalidDescription`
status that says exactly one output was expected and names the second.

## LeavesKeepEveryRequiredRoleInOrder

`ObjectiveLeaves` starts with no required role. After `require` of labels, boxes and labels
again it lists the three in that order, the second with the sample shape [8, 4], and still
has no leaf.

## AccuracyIsReadFromTheLabelsRole

`measure` over the fixture's logits and the labels 1, 0, 0 reports one metric, `accuracy`,
with `is_fraction` set and the value 2/3.

## ABatchWithoutTheLabelsRoleIsRefused

`measure` over a batch without targets fails with `InvalidDescription` and "'labels'", and
adds no metric.

## AccuracyReadsOneOutput

`measure` over a step with two outputs fails with `InvalidDescription`, "one model output"
and "published 2".

## FeedWritesTheOneHotRowsOfTheLabels

`feed` of the labels 1, 0, 0 over 2 classes writes 0, 1, 1, 0, 1, 0 into the tensor bound to
the `targets` leaf.

## AnObjectiveFedByItsOwnStepRefusesABatch

`feed` of the base class fails with `Unsupported`, "'step_fed'" and "fed by its own step".
Its `measure` succeeds and adds no metric.

## ALeafWithoutATensorIsNamed

`find_leaf_tensor` fails with `InvalidDescription` for a leaf that was not declared
("no leaf 'collocation'") and for a declared leaf the pool has no tensor for ("no tensor",
"'targets'").

## TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

## LeavesRefuseADuplicateRoleAndNameAnAbsentOne

Adding the leaf `targets` a second time raises an error holding "'targets'" and "twice".
`at("collocation")` raises one holding "'collocation'" and "'targets'". `has` is true for
`targets` and false for `collocation`, and the set holds one leaf.

## CrossEntropyReadsTheOnlyOutput

The model of `MLP` with 6, 8 and 3 units publishes one output, under the name
`NN::K_SINGLE_PORT_NAME`, which is "output". The case asserts that the outputs hold that one
entry, that the loss node is an `NLL_LOSS` reading a `LOG_SOFTMAX`, and that the `LOG_SOFTMAX`
reads the value published under that name.

## CrossEntropyRefusesTwoOutputs

The model of `Test::build_two_output_model`, from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`, publishes
"features" and "logits". The case asserts that
`build_training_graph` returns `StatusCode::InvalidDescription` and that the message names both
outputs.

## AnEmptyLeafSetSaysSoRatherThanListingNothing

`at("targets")` on an empty leaf set raises an error holding "'targets'" and "(none)".

## TheFactoryBuildsCrossEntropyOverTheContext

`make_cross_entropy` for a batch of 4 and 3 classes gives an objective named "cross_entropy"
whose `targets` leaf has the shape [4, 3].

## CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

## CrossEntropyNeedsTheClassCountOfTheData

`make_cross_entropy` for a context without a class count fails with `InvalidDescription`,
"objective", "class count" and "0 classes".
