# tests/layers/train/test_objective.cpp


Suites: `Small_Objective`, `Small_ObjectiveScore`, `Small_ObjectiveLeafLookup`,
`Small_ObjectiveGraph`, `Medium_ObjectiveLeavesDeathTest`, `Medium_ObjectiveLoss`.

Tests the objective seam: the leaves it declares, the loss it emits, the resulting IR, what
`feed` and `measure` do with a batch, and the factories of the dense objectives.

The `Small_` cases need no device. Declaring leaves and emitting a loss are IR-to-IR. The
`Small_ObjectiveScore` fixture owns a `HostExecutionBackend`, whose engine the logits, the labels
and the target tensors are allocated from, because `feed` and `measure` read host tensors. The
death cases are `Medium_`, since a death test starts a child process. `Medium_ObjectiveLoss` is an
`EngineTest`, because it compiles the loss graph and runs it, and the NLL_LOSS it emits has
no host rule.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIR. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

## LeafLookupRow

`LeafLookupRow` holds the role looked up and two texts of the refusal.

## BuildGraphAndBuildClassifierProduceTheSameIR

`build_classifier` and `build_training_graph` with a `CrossEntropyObjective`, each over its
own model of the same widths, give graphs with the same value count, node count and
description, and specs with the same `inputs`, `targets`, `logits` and `loss` ids.

## TheSpecTargetsAreTheLeafTheObjectiveDeclared

The training graph has a leaf named `targets`, and the spec's `targets` is that leaf. No literal
value id is pinned.

## CrossEntropyRequiresTheLabelsRole

The training graph under `CrossEntropyObjective(4, 3)` requires one role: `K_ROLE_LABELS`
with an empty sample shape, `Int32` and 3 classes.

## CrossEntropyRefusesAnOutputOfAnotherShape

`CrossEntropyObjective(4, 3)` over a model output of shape [4, 5] returns an `InvalidDescription`
status that names both shapes, in place of throwing. What shape an output must be is the
objective's question, since a detector's is not a classifier's. The larger copy that built a
whole training graph is deleted.

## CrossEntropyRefusesAModelWithTwoOutputs

The same objective over a model that publishes two outputs returns an `InvalidDescription`
status that says exactly one output was expected and names the second. The larger copy that
built a training graph from a two-output model is deleted.

## LeavesKeepEveryRequiredRoleInOrder

`ObjectiveLeaves` starts with no required role. After `require` of labels, boxes and labels
again it lists the three in that order, the second with the sample shape [8, 4], and still
has no leaf.

## AccuracyIsReadFromTheLabelsRole

`measure` over the fixture's logits and the labels 1, 0, 0 reports one metric, `accuracy`,
with `is_fraction` set and the value 2/3, within one rounded operation of double: the counts
are exact integers and the quotient rounds once.

## ABatchWithoutTheLabelsRoleIsRefused

`measure` over a batch without targets fails with `InvalidDescription` and "'labels'", and
adds no metric.

## AccuracyReadsOneOutput

`measure` over a step with two outputs fails with `InvalidDescription`, "one model output"
and "published 2".

## FeedWritesTheOneHotRowsOfTheLabels

`feed` of the labels 1, 0, 0 over 2 classes writes 0, 1, 1, 0, 1, 0 into the tensor bound to
the `targets` leaf. The one-hot rule has its own cases in `test_one_hot.cpp`; this case pins the
wiring of the objective to the leaf.

## AnObjectiveFedByItsOwnStepRefusesABatch

`feed` of a `Train::InMemoryObjective`, which keeps the defaults of the base, fails with
`Unsupported`, "'in-memory'" and "fed by its own step", and `is_fed_by_batch()` is false.

## AnObjectiveWithoutMetricsMeasuresNothing

`measure` of the same objective succeeds and adds no metric.

## ALeafWithoutATensorIsNamed

`find_leaf_tensor` fails with `InvalidDescription`. The rows are a leaf that was not declared
("no leaf 'collocation'") and a declared leaf the pool has no tensor for ("no tensor",
"'targets'").

## LeavesReportWhatTheyHold

After one `targets` leaf is added, `has("targets")` is true, `has("collocation")` is false and
the set holds one leaf.

## ARoleAddedTwiceIsFatal

Adding the leaf `targets` a second time stops the process with a message holding "'targets'" and
"twice".

## AnAbsentRoleIsFatalAndTheRolesHeldAreListed

`at("collocation")` stops the process with a message holding "'collocation'" and "'targets'".

## AnEmptyLeafSetSaysSoRatherThanListingNothing

`at("targets")` on an empty leaf set stops the process with a message holding "'targets'" and
"(none)".

## CrossEntropyReadsTheOnlyOutput

The loss is computed from the value the model published. A [3, 4] logits leaf published as the
only output, and a one-hot targets leaf, give a loss that equals the mean over the three rows of
the cross-entropy, computed in double from the normaliser of each row. Two sets of logits are
the rows, so the loss moves when the output moves. The bound is 16 rounded operations of float32:
the four terms of a normaliser, the exponential and the logarithm, the product with the target,
the sum over four classes and the mean over three rows are 12, with slack for the device's
exponential.

## TheFactoryBuildsCrossEntropyOverTheContext

`make_cross_entropy` for a batch of 4 and 3 classes gives an objective named "cross_entropy"
whose `targets` leaf has the shape [4, 3].

## CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

## CrossEntropyNeedsTheClassCountOfTheData

`make_cross_entropy` for a context without a class count fails with `InvalidDescription`,
"objective", "class count" and "0 classes".

## BinaryMakeRefusesABatchBelowOne

`BinaryCrossEntropyObjective::make(0)` fails with `InvalidDescription`, "binary_cross_entropy"
and "at least 1".

## BinaryMakeRefusesAThresholdOnTheBound

A threshold of 1 fails with "above 0 and below 1".

## BinaryMakeRefusesAnInt32LabelType

An int32 label type fails with "float32" and "int32".

## MultiLabelMakeRefusesAWidthBelowOne

`MultiLabelObjective::make(4, 0)` fails with "multi_label" and "at least 1".

## MultiLabelMakeRefusesAThresholdOnTheBound

A threshold of 0 fails with "above 0 and below 1".

## MeanSquaredErrorMakeRefusesAWidthBelowOne

`MeanSquaredErrorObjective::make(4, 0)` fails with "mean_squared_error" and "at least 1".

## TheBinaryFactoryReadsTheOneOutput

The factory over a [4, 1] output and the key `threshold` 0.9 gives an objective with the
threshold 0.9 and the batch 4.

## TheMultiLabelFactoryReadsTheOneOutput

The factory over a [4, 5] output gives 5 classes and the default threshold 0.5.

## TheMeanSquaredErrorFactoryReadsTheOneOutput

The factory over a [4, 2] output gives the width 2.

## TheBinaryFactoryRefusesAnOutputOfTwoLogits

A [4, 2] output fails with "objective", "one logit per sample" and "[4, 2]".

## TheMultiLabelFactoryRefusesAnOutputOfRankThree

A [4, 3, 5] output fails with "objective", "[4, columns]" and "[4, 3, 5]".

## TheMeanSquaredErrorFactoryRefusesAnOutputOfAnotherBatch

An [8, 2] output for a batch of 4 fails with "objective", "[4, columns]" and "[8, 2]".

## AFactoryRefusesAModelWithTwoOutputs

A context with a second output fails with "objective" and "publishes 2".

## TheMeanSquaredErrorFactoryRefusesAStrayKey

The key `huber` fails with "objective" and "'huber'".

## TheMultiLabelFactoryRefusesAStrayKey

The key `weights` fails with "objective", "'weights'" and the key it knows, "'threshold'".
