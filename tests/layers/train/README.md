# tests/layers/train

The tests of the module `layers/train`, one file per header of `modules/layers/train/include/`. The suite prefix of a case gives its ctest label.

## test_binary_confusion.cpp

Suites: `Unit_BinaryConfusion`, `Unit_BinaryConfusionRatios`, `Unit_DecisionThreshold`.

Tests `Train::count_confusion`, the four ratios over a `BinaryConfusion`, `Train::logit_of`,
`Train::read_threshold` and `Train::require_host_reals`. The `Unit_BinaryConfusion` fixture
needs an engine to allocate the logits and the truth. The other two suites run on the host.

### EachRowLandsInOneOfTheFourCounts

Five rows of one column with the logits 2, 1, -1, -3, 0 and the truth 1, 0, 1, 0, 1, cut at
`logit_of(0.5)`, give 2 true positives, 1 false positive, 1 false negative and 1 true negative.
The accuracy is 3/5, and the precision, the recall and the F1 are each 2/3.

### AColumnIsCountedAlone

Over a [2, 2] float64 matrix of logits and a [2, 2] truth, column 0 counts 2 true positives.
Column 1 counts 0 true positives, 1 false negative and 1 false positive.

### TheThresholdMovesTheCut

One row with the logit 1 and the truth 1 is a true positive at the threshold 0.5 and a false
negative at the threshold 0.9.

### ARatioOverNothingIsZero

For a `BinaryConfusion` of four zeros the accuracy, the precision, the recall and the F1 are
each 0. With 4 true negatives and nothing else the accuracy is 1 and the F1 is 0.

### LogitOfInvertsTheSigmoid

`logit_of(0.5)` is exactly 0. For 0.1, 0.731 and 0.9 the sigmoid of `logit_of(p)` is p within
1e-12.

### TheReaderDefaultsToOneHalf

`read_threshold` over `{}` returns `K_DEFAULT_THRESHOLD`, and over `{"threshold": 0.25}` it
returns 0.25.

### TheReaderRefusesWhatIsNoProbability

The key `cut` fails with `InvalidDescription` and a message naming "objective", `'cut'` and
`'threshold'`. The value `"high"` fails with a message containing "number". The thresholds 0,
1, -0.5 and 1.5 each fail with "above 0 and below 1".

### TheHostReadRefusesWhatItCannotRead

`require_host_reals` is ok for a float64 tensor of four elements asked for 4 values, and
`host_real(reals, 2)` is 2.5. An int32 tensor fails with `InvalidDescription` and a message
containing "host read of the logits" and "int32". Asking the float64 tensor for 6 values fails
with `Incompatible`, "6 values" and "holds 4".

## test_dense_losses.cpp

Suites: `Unit_DenseLosses`, `Unit_DenseLossesDeathTest`.

Tests the nodes `Train::emit_mean_of_elements`, `Train::emit_logistic_loss` and
`Train::emit_squared_error` add to a graph. Each case reads the operations of the whole graph in
emission order.

### TheMeanIsOneColumnSummedAndScaled

Over a [4, 3] input the mean is RESHAPE, SUM_ROWS, SCALE. The result has one element and the
reshaped value has the shape [12, 1].

### TheLogisticLossReadsTheLogitInItsStableForm

Over [4, 3] logits and targets the graph is RELU, MUL, ABS, SCALE, SOFTPLUS, SUB, ADD, RESHAPE,
SUM_ROWS, SCALE, and the loss is one float32 element. SOFTPLUS reads a SCALE whose operand is
the ABS node, and ABS reads the logits.

### TheSquaredErrorSquaresTheDifference

Over [4, 2] outputs and targets the graph is SUB, MUL, RESHAPE, SUM_ROWS, SCALE and the loss
has one element. MUL reads one value twice, and SUB reads the outputs and then the targets.

### OperandsOfTwoShapesAreFatal

`emit_squared_error` over a [4, 2] and a [4, 3] value stops the process with a report
containing "[4, 2]" and "[4, 3]".

## test_dense_targets.cpp

Suites: `Unit_DenseTargets`.

Tests `Train::fill_dense_targets` and `Train::fill_binary_targets`. The fixture needs an engine.
A leaf holds 7 in every element before a fill.

### ValuesAreCopiedInOrderIntoEitherRealType

Six float32 values of shape [2, 3] fill a float32 leaf and a float64 leaf of the same shape,
and each leaf reads back the six values in order.

### AColumnLeafTakesOneLabelPerRow

`fill_binary_targets` writes the [4] labels 1, 0, 0, 1 into a [4, 1] leaf.

### AnotherCountOrTypeIsRefusedAndTheLeafKept

`fill_dense_targets` fails with `Incompatible` for six values into a leaf of four ("holds 6
values", "targets hold 4"), for int32 values ("int32", "float32") and for an int32 leaf
("targets are int32"). The float32 leaf still holds four 7s.

### ALabelThatIsNotZeroOrOneIsRefused

`fill_binary_targets` over the labels 1, 0, 0.5, 1 fails with `Incompatible` and a message
containing "element 2", "0.5" and "0 or 1". The leaf still holds four 7s.

## test_input_match.cpp

Suites: `Unit_InputMatch`.

Tests `Train::describe_input_leaf` and `Train::match_input`. The dataset is a local `Published`
class named "published" that publishes the input it is built with, no role and no batch.

### ALeafIsDescribedWithoutItsBatchAxis

A [4, 16] int32 leaf is described as [16] int32, a [2, 8, 8, 3] float32 leaf as [8, 8, 3]
float32 and a [4] float64 leaf as an empty shape and float64.

### ALeafWithNoBatchAxisIsRefused

A leaf of rank 0 fails with `InvalidDescription` and "no batch extent".

### ADatasetThatPublishesTheInputIsAccepted

`match_input` is ok for [16] int32 against a dataset that publishes [16] int32, and for [16]
float32 against one that publishes [16] float32.

### AnotherElementTypeIsRefusedWithBothTypes

A float32 input against the int32 dataset fails with `Incompatible` and a message containing
`'published'`, "as int32 [16] per sample" and "compiled for float32 [16] per sample". An int32
input against the float32 dataset fails with the two types the other way round.

### AnotherShapeIsRefusedWithBothShapes

A [32] int32 input against the [16] int32 dataset fails with `Incompatible`, "as int32 [16] per
sample" and "compiled for int32 [32] per sample".

## test_metrics.cpp

Suites: `Unit_Metrics`.

Tests `Train::Metrics`, `Train::describe_metric_value` and `Train::only_output`.

### MetricsAreKeptInOrderAndFoundByName

A new `Metrics` is empty. After `accuracy` 0.75 and `perplexity` 12.5 are added, `all()` holds
them in that order, `find("perplexity")` has the value 12.5 and `find("recall")` is null.
`value_or` returns 0.75 for `accuracy` and the fallback -1 for `recall`. After `clear()` the
value is empty and `find("accuracy")` is null.

### AnEmptyNameADuplicateAndAnOverflowAreRefused

`add` fails with `InvalidDescription` for an empty name ("non-empty name") and for a second `a`
("'a'", "twice"), and `a` keeps the value 1. After `K_METRIC_CAPACITY` metrics the next `add`
fails with "at most 8" and `'i'`, and the size stays `K_METRIC_CAPACITY`.

### AFractionPrintsAsAPercentage

`describe_metric_value` gives "92.40%" for the fraction 0.9240 and "12.5" for the plain number
12.5.

### OnlyOutputReturnsTheOneTensor

`only_output` over a view with one output returns the address of that output's tensor.

### OnlyOutputRefusesAnyOtherCount

`only_output` fails with `InvalidDescription` for no output ("published 0"), for two outputs
("published 2") and for one output whose tensor is null ("'output'", "no tensor").

## test_objective.cpp

Suites: `Unit_Objective`, `Unit_ObjectiveScore`.

Tests the objective seam: the leaves it declares, the loss it emits, the resulting IR, and what
`feed` and `measure` do with a batch.

The `Unit_Objective` cases need no device. Declaring leaves and emitting a loss are IR-to-IR,
so that part of the seam is testable without an Engine. The `Unit_ObjectiveScore` fixture is
an `EngineTest`, because `feed` and `measure` read tensors.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIR. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

### BuildGraphAndBuildClassifierProduceTheSameIR

`build_classifier` and `build_training_graph` with a `CrossEntropyObjective`, each over its
own model of the same widths, give graphs with the same value count, node count and
description, and specs with the same `inputs`, `targets`, `logits` and `loss` ids.

### TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

### CrossEntropyRequiresTheLabelsRole

The training graph under `CrossEntropyObjective(4, 3)` requires one role: `K_ROLE_LABELS`
with an empty sample shape, `Int32` and 3 classes.

### LeavesKeepEveryRequiredRoleInOrder

`ObjectiveLeaves` starts with no required role. After `require` of labels, boxes and labels
again it lists the three in that order, the second with the sample shape [8, 4], and still
has no leaf.

### AccuracyIsReadFromTheLabelsRole

`measure` over the fixture's logits and the labels 1, 0, 0 reports one metric, `accuracy`,
with `is_fraction` set and the value 2/3.

### ABatchWithoutTheLabelsRoleIsRefused

`measure` over a batch without targets fails with `InvalidDescription` and "'labels'", and
adds no metric.

### AccuracyReadsOneOutput

`measure` over a step with two outputs fails with `InvalidDescription`, "one model output"
and "published 2".

### FeedWritesTheOneHotRowsOfTheLabels

`feed` of the labels 1, 0, 0 over 2 classes writes 0, 1, 1, 0, 1, 0 into the tensor bound to
the `targets` leaf.

### AnObjectiveFedByItsOwnStepRefusesABatch

`feed` of the base class fails with `Unsupported`, "'step_fed'" and "fed by its own step".
Its `measure` succeeds and adds no metric.

### ALeafWithoutATensorIsNamed

`find_leaf_tensor` fails with `InvalidDescription` for a leaf that was not declared
("no leaf 'collocation'") and for a declared leaf the pool has no tensor for ("no tensor",
"'targets'").

### TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

### LeavesRefuseADuplicateRoleAndNameAnAbsentOne

Adding the leaf `targets` a second time raises an error holding "'targets'" and "twice".
`at("collocation")` raises one holding "'collocation'" and "'targets'". `has` is true for
`targets` and false for `collocation`, and the set holds one leaf.

### CrossEntropyReadsTheOnlyOutput

The model of `MLP` with 6, 8 and 3 units publishes one output, under the name
`NN::K_SINGLE_PORT_NAME`, which is "output". The case asserts that the outputs hold that one
entry, that the loss node is an `NLL_LOSS` reading a `LOG_SOFTMAX`, and that the `LOG_SOFTMAX`
reads the value published under that name.

### CrossEntropyRefusesTwoOutputs

The model of `Test::build_two_output_model`, from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`, publishes
"features" and "logits". The case asserts that
`build_training_graph` returns `StatusCode::InvalidDescription` and that the message names both
outputs.

### AnEmptyLeafSetSaysSoRatherThanListingNothing

`at("targets")` on an empty leaf set raises an error holding "'targets'" and "(none)".

### TheFactoryBuildsCrossEntropyOverTheContext

`make_cross_entropy` for a batch of 4 and 3 classes gives an objective named "cross_entropy"
whose `targets` leaf has the shape [4, 3].

### CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

### CrossEntropyNeedsTheClassCountOfTheData

`make_cross_entropy` for a context without a class count fails with `InvalidDescription`,
"objective", "class count" and "0 classes".

## test_objective_context.cpp

Suites: `Unit_ObjectiveContext`.

Tests `Train::list_output_shapes` and `Train::find_output_shape`. No case needs an engine.

### ListsEveryOutputWithItsExtentsInPublishedOrder

For a `ModelOutputs` holding `coarse` of shape [2, 4, 4, 14] and `fine` of shape [2, 8, 8, 14],
the list has two entries with those names and shapes, in that order.

### AnEmptyOutputSetListsNothing

An empty `ModelOutputs` over an empty graph gives an empty list.

### AProbeTraceGivesTheShapesWithoutAnEngine

For a `ModelGraph` of two `Linear` nodes, 6 to 5 and 5 to 3, that publishes both,
`Train::build_probe_graph` over [4, 6] gives a graph whose listed outputs are `hidden` of shape
[4, 5] and `logits` of shape [4, 3].

### FindsAnOutputByItsName

`find_output_shape` returns the entry named `fine` with its shape, and null for `medium`.

## test_objective_registry.cpp

Suites: `Unit_ObjectiveRegistry`.

Tests `Train::ObjectiveRegistry`, `Train::register_objectives` and `Train::make_objective`. The
file's stand-in factory returns a `CrossEntropyObjective` when the context holds a task with a
`type`, one output named `logits`, the input sample [6], a `weight` attribute and a registry
that holds the row `stand_in`. Otherwise it fails with `Unsupported` and names the field.

### TheLayersTierAddsItsFiveObjectives

After `register_objectives` the registry holds five rows in the order `cross_entropy`,
`binary_cross_entropy`, `multi_label`, `mean_squared_error`, `sum`. `find("cross_entropy")`
returns a row and `find("anchor")` is null.

### ATypeIsHeldOnce

A second `add` of `stand_in` fails with `InvalidDescription`, `'stand_in'` and "already
registered", and the registry holds one row.

### TheFactoryReceivesTheContextAndItsRegistry

`make_objective` for `stand_in` with the attribute `weight` returns an objective named
"cross_entropy".

### AnUnknownTypeListsWhatTheRegistryHolds

`make_objective` for `yolo` fails with `InvalidDescription` and a message containing `'yolo'`,
`'cross_entropy'` and `'stand_in'`.

### TheFailureOfAFactoryIsReturned

`make_objective` for `stand_in` with empty attributes and the label "the objective" fails with
`Unsupported`, "the objective" and "other attributes".

## test_one_hot.cpp

Suites: `Unit_OneHot`.

Tests `Train::fill_one_hot`. The fixture needs an engine. The labels are an int32 tensor of one
extent, and the rows hold 7 in every element before a fill.

### EveryRowHoldsOneInItsLabelsColumn

The labels 2, 0, 1, 2 fill a [4, 3] float32 tensor with 1 in each row's label column and 0
elsewhere.

### AFloat64LeafReceivesTheSameRows

The same labels fill a [4, 3] float64 tensor with the same rows.

### ASecondFillReplacesTheFirst

After a fill with the labels 0, 0 and a second with 1, 0, a [2, 2] tensor holds 0, 1, 1, 0.

### ALabelOutsideTheClassesIsRefusedAndNothingIsWritten

Over three classes the labels 0, 3 fail with `Incompatible`, "row 1", "is 3" and "3 classes",
and the labels -1, 0 fail with "row 0", "is -1" and "3 classes". The rows still hold six 7s.

### AnotherLabelCountIsRefusedWithBothShapes

Three labels for a [4, 3] tensor fail with `Incompatible`, "[3]" and "[4, 3]".

### LabelsOfAnotherElementTypeAreRefused

Float32 labels fail with `Incompatible`, "float32" and "int32".

### RowsThatAreNotRealAreRefused

An int32 [2, 3] tensor of rows fails with `Incompatible`, "int32", "float32" and "float64".

## test_role_match.cpp

Suites: `Unit_RoleMatch`.

Tests `Train::match_roles`. The dataset is a local `Published` class named "published" that
publishes the roles it is built with and holds no batch. The cases use two roles: `labels`,
one int32 per sample over a class count, and `boxes`, a float32 [n, 4] value per sample.

### ADatasetThatPublishesEveryRoleIsAccepted

A dataset that publishes `labels` over 4 classes and `boxes` of 8 matches a step that requires
`boxes` of 8 and then `labels` over 4.

### AStepThatReadsNoRoleAcceptsAnyDataset

An empty list of required roles matches a dataset that publishes no role and one that
publishes `labels`.

### AMissingRoleIsNamedWithTheDatasetAndWhatItPublishes

Requiring `boxes` of a dataset that publishes only `labels` fails with `InvalidDescription`
and a message containing `'published'`, `'boxes'`, "float32 [8, 4] per sample" and "publishes
'labels'". Requiring `labels` of a dataset that publishes nothing fails with "(none)".

### AnotherShapeIsRefusedWithBothShapes

`boxes` of 8 against a dataset that publishes `boxes` of 16 fails with `Incompatible`, "as
float32 [16, 4] per sample" and "compiled for float32 [8, 4] per sample".

### AnotherElementTypeIsRefusedWithBothTypes

Int32 `labels` against float32 `labels` fails with `Incompatible`, "as 4 classes of float32 []
per sample" and "compiled for 4 classes of int32 [] per sample".

### AnotherClassCountIsRefusedWithBothCounts

`labels` over 4 classes against `labels` over 5 fails with `Incompatible`, "as 5 classes of
int32 [] per sample" and "compiled for 4 classes of int32 [] per sample".

### ARequiredClassCountOfZeroAcceptsAnyCount

`labels` required over 0 classes matches a dataset that publishes `labels` over 5.

### TheFirstRoleThatDoesNotMatchIsReported

Requiring `boxes` and then `labels` over 4 of a dataset that publishes `labels` over 5 fails
with `InvalidDescription` and a message naming `'boxes'`.

## test_steps.cpp

Suites: `Unit_Steps`.

Tests the builders that trace a model into a training, probe or evaluation graph, and the batch
shape function.

This file needs no device. `build_training_graph`, `build_probe_graph`, `build_evaluation_graph`
and `batch_shape` are declared in `modules/layers/train/include/SushiAI/train/steps.hpp`, and `Train::TrainingGraph`
is declared in `modules/layers/train/include/SushiAI/train/training_graph.hpp`. The constants `K_BATCH` (4),
`K_FEATURES` (6), `K_HIDDEN` (8), `K_CLASSES` (3), `K_CHANNELS` (3) and `K_SIDE` (2) set every
shape in the file. Each case reads the graph a builder returns, through its nodes, its leaves,
its pinned values, its buffer updates or its status.

### K_MLP_TRAINING_OPS

`K_MLP_TRAINING_OPS` lists the operation of each of the 17 nodes of the MLP's training graph
under `CrossEntropyObjective`, in node order.

### parameter_states

`parameter_states` returns the `state()` of every parameter of a model, in the order of
`NN::for_each_parameter`.

### build_two_output_model

`build_two_output_model` is `Test::build_two_output_model` from `modules/core/engine_fixture/include/SushiAI/engine_fixture/engine_test.hpp`,
which this file brings in with a using-declaration. It builds a Linear named "body", a GELU named
"activation" and a Linear named "head" over [6], and publishes "features" on the GELU and
"logits" on the head.

### NamedOutputObjective

`NamedOutputObjective` is a `Train::Objective` that declares one leaf, "targets", of
[batch, classes]. Its `loss()` is the cross-entropy of the output it was given by name against
that leaf. `score()` returns no metric, and `name()` returns "named_output".

### ATrainingGraphOfAnMLPHasTheNodesBuildGraphGave

The training graph of the MLP under cross-entropy has 17 nodes. Their operations, in node order,
equal `K_MLP_TRAINING_OPS`.

### ATrainingGraphListsEveryParameterWithItsLeafInThatGraph

The training graph lists four parameters, in the order of the parameter walk, and each entry
carries the walk's name. Each entry's value is the parameter's `id_in` in the built graph, and
neither its value nor its gradient is `K_NO_VALUE`.

### ATrainingGraphListsEveryParameterWithItsRole

The training graph of the two-layer MLP lists four parameters with the roles `Weight`,
`Bias`, `Weight`, `Bias`.

### WeightDecayAppliesToWeightsOnly

`takes_weight_decay` is true for `Weight` and false for `Bias`, `Scale` and `Embedding`. A
default `ParameterBinding` has the role `Weight`.

### ABatchNormModelYieldsTwoUpdatesAndPinsTheirSources

A `BatchNorm2d`, `Flatten` and `Linear` model over [4, 2, 2, 3] yields two buffer updates, and
the source of each update is pinned.

### ATrainingGraphPinsTheLossAndTheOutputs

For the same model, the loss and every value in `outputs` are pinned.

### AProbeGraphHasBatchStatisticsAndNoUpdate

The probe graph of a `BatchNorm2d` over [4, 2, 2, 3] has one `BATCHNORM2D` node, no
`BATCHNORM_FOLD` node and no buffer leaf, and it publishes one output.

### AnEvaluationGraphFoldsTheStoredStatistics

The evaluation graph of the same layer has one `BATCHNORM_FOLD` node, two buffer leaves and no
`BATCHNORM2D` node.

### OneModelTracedThreeTimesKeepsOneStateIDPerParameter

After the training graph, the probe graph and the evaluation graph are built from one
batch-norm classifier, its four parameters have the same state identities as after the training
graph alone, in walk order. Each state has a leaf in all three graphs.

### AModelGraphWithTwoOutputsLeavesLogitsUnset

`NamedOutputObjective("logits")` reads the model of `build_two_output_model`. The training graph
has `spec.logits` set to `K_NO_VALUE`, two entries in `outputs` and a set `spec.loss`.

### ARefusingLayerFailsTheBuilderWithItsStatus

`build_probe_graph` of a `BatchNorm2d(3)` over [1, 1, 1, 3] returns `StatusCode::InvalidDescription`,
with a message containing "BatchNorm2d(3)" and "more than one value per channel".

### BatchShapePrependsTheBatch

`batch_shape(16, [28, 28, 1])` is [16, 28, 28, 1]. A batch of 0 returns
`StatusCode::InvalidDescription`, with a message containing "batch" and "positive".

### AnEvaluationGraphAndAProbeGraphFeedTheModelInputAlone

The steps `build_probe_graph` and `build_evaluation_graph` trace list no fed leaf and no
carried value.

### ACarryingGraphFoldsTheStoredStatisticsAndUpdatesNoBuffer

For a `BatchNorm2d`, `build_carrying_graph` gives the same operation names as
`build_evaluation_graph`: no BATCHNORM2D node, two buffer leaves, no fed leaf and no carried
value.

### AnIncrementalGraphNamesTheNodeThatHasNoIncrementalForm

`build_incremental_graph` of a model whose one node is a `Linear` named "dense" fails with
`Unsupported`, "node 'dense'" and "no incremental form".

## test_sum_objective.cpp

Suites: `Unit_SumObjective`.

Tests `Train::SumObjective` and its factory `Train::make_sum_objective`, reached through the
`sum` row of the registry. Most cases sum two `CrossEntropyObjective` terms over 4 rows of 3
classes. Two local objectives stand in for the others: `StepFedObjective`, which keeps the
base's `is_fed_by_batch`, and `ReportingObjective`, whose `measure` adds the two metrics it was
made with.

### MakeKeepsItsTermsInOrder

A sum of the weights 1 and 0.5 holds two terms with those weights in order. The second term's
objective is named "cross_entropy", the sum is named "sum" and is fed by the batch, and
`leaf_prefix(1)` is "term1.".

### MakeRefusesNoTermAndAWeightItCannotUse

`make` fails with `InvalidDescription` for no term ("at least one term"), a second weight of 0
("Term 1", "'cross_entropy'", "above zero"), a first weight of -0.5 ("Term 0"), a second weight
of infinity ("Term 1", "finite") and a first weight of NaN ("Term 0").

### MakeRefusesATermItsOwnStepFeeds

`StepFedObjective` reports false for `is_fed_by_batch` and `CrossEntropyObjective` reports
true. A sum whose second term is a `StepFedObjective` fails with `InvalidDescription`, "Term
1", `'step_fed'` and "fed by its own step".

### EachTermDeclaresItsLeavesUnderItsOwnPrefix

`declare` adds two leaves, `term0.targets` and `term1.targets`, with different ids, and no leaf
named `targets`. The second has the shape [4, 3]. Two targets are required and both have the
role `K_ROLE_LABELS`.

### TheLossAddsTheWeightedLossesInTheGraph

For the weights 1 and 0.5 the loss has the shape [1] and the graph holds two NLL_LOSS nodes,
one ADD and one SCALE whose factor is 0.5. The last node of the graph is the ADD and it writes
the loss.

### ATrainingGraphDifferentiatesTheSum

`Train::build_training_graph` over one `Linear` layer and a sum of the weights 2 and 0.5 gives
a loss of one element, `spec.targets` equal to `K_NO_VALUE` and both prefixed leaves. The two
parameters each have a gradient.

### MeasureKeepsTheFirstOfTwoMetricsOfOneName

With a first term that reports `accuracy` 0.25 and `precision`, and a second that reports
`recall` 0.75 and `accuracy` 1, `measure` gives three metrics in the order `accuracy`,
`precision`, `recall`. The accuracy is 0.25 and the recall is 0.75.

### TheFactoryBuildsEachTermThroughTheRegistry

A `terms` list of two `cross_entropy` entries, the second with the weight 0.5, gives a
`SumObjective` of two terms with the weights 1 and 0.5 whose first objective is named
"cross_entropy".

### ATermMayBeASum

A `terms` list holding a `cross_entropy` entry and a `sum` entry with one `cross_entropy` term
declares two leaves, `term0.targets` and `term1.term0.targets`.

### TheFactoryRefusesASectionItCannotRead

Nine sections fail with `InvalidDescription`: no `terms` ("the objective block", "'terms'"), a
`gamma` key, `terms` as an object ("an array"), an empty list ("at least one term"), an entry
that is a number ("'terms' entry 0", "an object"), an entry without `type`, a `weight` of
"half" ("entry 0 'weight'", "a number"), the type `yolo` ("'yolo'", "'cross_entropy'",
"'sum'") and a `smoothing` key in a term ("'terms' entry 0", "'smoothing'").

### TheFactoryNeedsTheRegistryOfItsContext

`make_sum_objective` called with a context that holds no registry fails with
`InvalidDescription`, "the objective block" and "registry".

## test_evaluation_step.cpp

Suites: `Unit_CarryingStep`, `Unit_PlainEvaluationStep`.

Tests a `Train::EvaluationStep` that is fed two leaves and carries a value between runs. Both
fixtures need an engine. The model of `Unit_CarryingStep` is a local `Accumulator` whose
`forward` feeds a leaf named `gain`, declares a carried leaf `sum` with the initial value 1 and
returns sum + x * gain. The fixture builds the step with `Train::build_carrying_graph` over a
[1, 3] row and fills x with 1, 2, 3 and the gain with 2, 2, 2. A `static_assert` holds that
`Accumulator` is a `TraceModule`.

### TheStepNamesTheModelInputThenTheFedLeaves

`input_names()` is `K_MODEL_INPUT_NAME` then `gain`, and both `carries_state()` and
`step().has_trailing_work()` are true. `input("gain")` returns a tensor whose second extent is
3. `input("absent")` fails with `InvalidDescription` and a message naming `'absent'`,
`'inputs'` and `'gain'`.

### ARunReadsWhatTheRunBeforeCarried

After one, two and three runs by name, element i of the output is 1 + (i + 1) * 2 * runs within
1e-6. The step compiled once and ran three times.

### AResetRestoresTheInitialValue

After two runs, `reset()` and one more run, the output is what one run gives. The compile count
stays 1.

### ARunInOrderFeedsTheLeavesAsTheStepListsThem

`run_in_order` with x then the gain gives the output of one run by name.

### ARunOfOneTensorIsRefusedByAStepFedTwoLeaves

`run(x)` fails with `Incompatible` and a message containing "1 tensors", "fed 2" and `'gain'`.
The run count stays 0.

### NamedFeedsThatDoNotCoverTheLeavesAreRefused

`run_by_name` fails with `InvalidDescription` three times: for a feed named `scale` ("'scale'",
"no input leaf"), for `gain` given twice ("'gain' twice") and for one feed alone ("1 named
tensors", "fed 2"). The run count stays 0.

### AFeedOfAnotherShapeIsRefusedUnderItsName

A [1, 4] tensor fed as `gain` fails with `Incompatible` and a message containing "gain",
"[1, 4]" and "[1, 3]".

### AnOrdinaryStepFeedsOneLeafAndCarriesNothing

A step built with `Train::build_evaluation_graph` over `NN::Linear{3, 3}` lists one input,
`K_MODEL_INPUT_NAME`. `carries_state()` and `has_trailing_work()` are false, `reset()` is ok and
the run count is 0.

## test_buffer_update_recorder.cpp

Suites: `Integration_BufferUpdateRecorder`.

The recorder blends each buffer's batch statistic into its running buffer, once per replay. The
cases check how many tasks it records, the value one replay leaves behind, the empty recorder,
and the two refusals that binding makes.

### RecordsTwoTasksPerSlot

Two slots record four tasks in all, two for each slot. The test asserts that `slot_count()` is 2
and that `record_into_step` returns 4.

### OneReplayBlendsTheSourceIntoTheTarget

One slot over four float64 elements. After one replay each element equals the
`NN::updated_statistic` of its running value and its batch value, with momentum 0.1 and a source
scale of 2, within 1e-9.

### NoSlotRecordsNothing

A recorder built with no slots records zero tasks, reports zero slots and runs without error.

### BindRefusesATargetTheStateLacks

The state holds one entry, `bn.running_mean`, under identity 1. The update targets identity 7.
Binding is refused as `Incompatible`, and the message contains "no entry".

### BindRefusesADTypeMismatch

The target entry is float32 and the source tensor is float64. Binding is refused as
`Incompatible`, and the message names both dtypes by their `dtype_name`.

Suites: `Integration_EvaluationStep`.

The evaluation step replays a forward-only graph over a `NN::ModelState`. The cases check that it
compiles once, that it reads the weights a training step wrote, that a probe run leaves the
batch norm buffers alone, and the two refusals it makes.

### TheStepCompilesOnce

The step is made from an MLP's evaluation graph. `compile_count()` is 1 after `make()` and still
1 after three runs.


### AnEvaluationStepSeesAWeightTheTrainingStepJustMoved

A training pool and an evaluation pool attach to one state. The evaluation output for fixed
inputs is read before and after one training step on the synthetic dataset. The two byte
sequences differ. The test notes that the default policy starts the classifier at zero.

### AProbeRunLeavesBothRunningVectorsUnchanged

The evaluation graph of a batch norm gives its buffers their leaves, and its probe graph is run
once. The bytes of the running mean and the running variance are the same before and after.

### AnInputOfAnotherShapeIsRefusedWithBothShapes

A step compiled for a [4, 6] input refuses a [2, 6] input. The refusal is `Incompatible`, and its
message names "inputs", "[2, 6]" and "[4, 6]".

### AnUnknownOutputNameListsTheNames

`output("boxes")` is refused as `InvalidDescription`. The message names "boxes" and the single
port name, `NN::K_SINGLE_PORT_NAME`.

## test_trainer/test_convergence.cpp

Suites: `Integration_Training`.

This is the end-to-end gate: a real MLP on the synthetic classification dataset that has to
converge, with the compile-once guarantee checked across every step.

### InitialLossIsExactlyLogCWithAZeroedClassifier

Zeroing the last layer makes every logit zero, so every class is equally likely and the
initial cross-entropy is ln(C) to the last digit. The test can assert that number; a loss that
merely looks reasonable would have to be judged by eye.

### TheStepIsCompiledExactlyOnceAcrossManySteps

This is the AI-1 headline gate. A loop that recompiles per step still converges and pays graph
analysis every step. That regression hides for months unless something counts compilations.

### TheStepAllocatesNothingInSteadyState

Compile-once is half the claim. The other half is that replaying reuses the same buffers, so a
long run's memory does not grow.

### SGDDrivesTheLossDownOnASeparableProblem

An MLP of 8 features, 24 hidden units and 4 classes is fitted to a synthetic dataset of 6 batches of
16 for 30 epochs, with SGD at a rate of 0.2 and a momentum of 0.9. The first loss is ln(4) within
1e-4, the last is below a quarter of the first, the accuracy is above 0.9 and the compile count is
1.

### AdamWDrivesTheLossDownOnASeparableProblem

The same model and data with AdamW at a rate of 0.02. The last loss is below a quarter of the first,
the accuracy is above 0.9 and the compile count is 1.

### AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate

The "eps after the bias correction" ordering exists to preserve this property. On step one,
m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so every coordinate moves by almost exactly the
learning rate. Folding eps inside the correction makes it 31.6x larger and visibly damps this
first step, the only place the bug is easy to see.

The loop skips coordinates whose gradient is below 1e-4, where eps does matter. The claim is
about the ordinary case.

### SGDWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient

One step of SGD at a rate of 0.1 over a `Linear` of 4 features and 3 classes. Each element of the
first parameter after the step equals its value before minus 0.1 times its gradient, within 1e-6.

### ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes

The shared test runtime is created without `RuntimeConfig::profiling`, so the runtime records
no timing. The report has to say that. A table of zeroes would read as if the work took no
time.

### ADisconnectedParameterIsRejectedRatherThanSilentlySkipped

The test fakes a parameter that no gradient reaches. It stands in for a layer wired into the
model list but not into the forward pass.

### TheSyntheticDatasetIsDeterministicAndBalanced

Each dataset gets a fresh Engine on purpose. SushiBLAS's `Engine::set_seed()` sets the seed and
leaves the monotonic RNG offset counter alone, so a second generator built on an engine that
has already drawn gets a different stream from the same seed. Reproducibility is therefore a
property of a fresh engine, which is how the trainer uses it.

Round-robin labels make every batch exactly balanced, so accuracy reads correctly without
class weighting.

### RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays

The model is Unflatten, a batch norm over three channels of a 2 by 2 grid, Flatten and a Linear
layer. It is trained for two steps on two synthetic batches. After each step, the running mean of
every channel equals the `NN::updated_statistic` of that step's channel mean, with the default
momentum and a source scale of 1, within 1e-5. After the second step, every channel's running mean
differs from its value after the first step by more than 1e-7.

### APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm

The same model is trained with the per-step optimizer placement. After one step, the optimizer is
not in the plan, the step count is 1, and at least one element of `3.weight` has changed by more
than 1e-7. The test notes that the classifier starts at zero, so the first gradient reaches its
weight alone.

### ABatchOfAnotherShapeIsRefusedWithBothShapes

A trainer compiled for a batch of 8 samples with 6 features refuses a batch of 4 samples. The
refusal is `Incompatible`, and its message contains "[4, 6]" and "[8, 6]".

### BoundSlotsDecayByTheRoleOfTheirParameter

The classifier graph of a two-layer MLP binds four slots. The first is named `0.weight` and decays,
the second is `0.bias` and does not, and the third and fourth follow the same pattern.

## test_trainer/test_dense_objectives.cpp

Suites: `Integration_DenseObjectives`.

A `Linear` over three features is trained through `Train::Trainer` on a batch of four with plain
SGD. The inputs and the weights follow fixed integer patterns. The value cases use a rate of 0, so
the logits the step computed can be read back and the loss recomputed on the host from them. A
float32 loss of order one is compared within 1e-5 and a ratio of counts within 1e-12.

### TheBinaryLossAndMetricsEqualTheHostComputation

`BinaryCrossEntropyObjective` over one output. The loss equals the mean of `max(x, 0) - x y + log(1
+ exp(-|x|))` on the host. The step reports three metrics, and `accuracy`, `precision` and `recall`
equal the ratios counted from the signs of the logits. Both signs occur among the logits.

### ALargeLogitGivesAFiniteBinaryLoss

The inputs are scaled by 400, so the largest logit exceeds 100 in magnitude. The loss is finite and
equals the host value within 1e-3.

### TheMultiLabelLossEqualsTheHostComputation

`MultiLabelObjective` over three classes. The loss over the twelve logits equals the host value. The
one metric, `f1`, equals the mean over the classes of `2 hits / (2 hits + wrong)`, with zero for a
class without a hit.

### TheSquaredErrorEqualsTheHostComputation

`MeanSquaredErrorObjective` over two outputs. The loss equals the sum of squared differences over 8,
and the one metric, `mean_absolute_error`, the sum of absolute differences over 8, both within 1e-5.

### FitLowersTheSquaredError

`fit` runs 40 epochs over a dataset of one batch at a rate of 0.05. The report holds 40 steps and
one compile, the last loss is below the first, and `mean_absolute_error` is among the last metrics
and in the report's text.

### FitRefusesLabelsOfAnotherType

A binary objective is fitted to a dataset whose `labels` role is int32 class indices. `fit` is
refused as `Incompatible`, the message names 'classes', 'labels', int32 and float32, and the trainer
has taken zero steps.

## test_trainer/test_dropout.cpp

Suites: `Integration_Dropout`.

The probe runs DROPOUT on an engine over a `[64, 64]` FLOAT32 parameter at a rate of 0.25. Its loss
is `NLL_LOSS` over the masked value with every target at -64, which is the sum of the masked
elements, so the gradient at the input is the mask. A mask is checked against what `dropout_keeps`
answers for the seed and the run. The model cases use `Linear, ReLU, Dropout(0.2), Linear` over 8
features, 32 hidden units and 4 classes with a batch of 16.

### TheOutputIsTheInputTimesAMaskOfZeroOrTheScale

Each mask element is 0 or `1 / 0.75`, and each output is 0 or `x / 0.75`, within 1e-5. The kept
fraction of the 4096 elements is within 0.03 of 0.75, and the mask is the one of run 0.

### TwoStepsOfOneCompiledStepDrawDifferentMasks

The step registers one replay prelude. Two runs give different masks, those of runs 0 and 1, with a
compile count of 1 and a run count of 2.

### TwoFreshStepsWithTheSameSeedAgreeStepForStep

Over three runs, two fresh steps with one seed give the same output bytes, and a third step with the
seed plus one gives different bytes.

### TheGradientAtTheInputIsTheMaskOfThatStep

On each of two runs the gradient at the input equals the mask of that run within 1e-5, and the loss
equals the sum of the outputs to 1e-4 relative.

### AStepWithoutDropoutRegistersNoPrelude

A graph that only scales a `[2, 3]` input by 2 compiles to a step with zero replay preludes, and the
step doubles its input.

### AnEvaluationTraceOfAModelEmitsNoDropoutNode

The evaluation graph of the model holds no DROPOUT node and its probe graph holds one. That node
carries the rate 0.2 and the seed `dropout_site_seed` gives for the graph seed 7 and site 0.

### AModelWithDropoutTrainsThroughTheTrainer

The training graph holds at least one DROPOUT node. `Train::Trainer` fits the model with AdamW at
1e-2 on a synthetic dataset of 8 batches for 10 epochs. The report holds 80 steps and one compile,
and the last loss is finite and below the first.

## test_trainer/test_token_feed.cpp

Suites: `Integration_TokenFeed`.

A materialised dataset named "tokens" holds int32 ids, two rows of five per batch, all below 7. The
cases feed it to an int32 input leaf and to a float32 one.

### AnInt32BatchOfADatasetFeedsAnEmbedding

An `Embedding` of 7 rows and width 4 is traced for a `[2, 5]` int32 input, and its state has one
entry. The input leaf describes itself as int32 and `match_input` accepts the dataset. After one
evaluation step over the batch, each output row equals the table row of its id, with a tolerance of
0.

### FitRefusesAnInt32DatasetForAFloat32Leaf

A classifier MLP whose input leaf is float32 `[5]` is fitted to the same dataset. `fit` is refused
as `Incompatible`; the message names 'tokens', "as int32 [5] per sample" and "compiled for float32
[5] per sample". The trainer has taken zero steps.

## test_trainer/test_resilience.cpp

Suites: `Integration_Resilience`, `Integration_ResilienceDeathTest`.

The file checks what the library does when it is handed something it cannot serve, and what it
does at the edges of what it can. The happy path belongs to `test_forward.cpp` and
`test_training.cpp`. This file makes three other claims.

1. Every device-lane precondition fails loudly and names the actual problem.
   `modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp` explains why the message is asserted and not only the throw.
2. Degenerate and boundary extents produce right answers and not off-by-one ones: one sample,
   one feature, one class, exactly the GEMM register-block edge and exactly one past it. Index
   arithmetic breaks here or nowhere.
3. Numbers that would make a naive implementation produce NaN do not. A NaN is the worst
   failure this library has. It propagates into every weight on the next update, the loss
   stops moving, and nothing reports an error. Finiteness is therefore asserted explicitly and
   not inferred from a loss that looks plausible.

The file is laid out in four groups: loud failures that name the problem, degenerate and
boundary extents, numerical extremes, and resource discipline.

### AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch

A `[4, 3]` parameter with a `[4, 2]` gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.weight", "12 elements" and "gradient has 8".

### AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType

A float32 parameter with a float64 gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.bias", "float32" and "float64".

### AnOptimizerRefusesADTypeItCannotServeAndExplainsWhy

Complex is not a training dtype at all. Half is a real float, and it is refused for a reason
the message has to carry: the refusal exists to prevent a model that compiles and trains
worse.

### AdamWNamesTheHyperparameterThatIsOutOfRange

`AdamW::make` refuses a `beta1` of 1.0 and a `beta2` of -0.5, each with its name and "[0, 1)" in the
message, and an `eps` of 0 with "eps must be positive". All three are `InvalidDescription`.

### ADisconnectedParameterNamesItselfAndSaysWhatToDo

A binding named "ghost" with no gradient value is appended to a classifier's parameters.
`bind_parameters` refuses it as `InvalidDescription`, and the message contains 'ghost', "receives no
gradient" and "no optimizer could ever move it".

### AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage

The library has no `zero_grad()`, by design. Every gradient buffer in a step is written and
not accumulated into (see `optimizer.hpp`). That leaves one window in which a gradient has
never been written: between `bind_parameters()` and the first replay. Stepping in that window
would feed whatever the allocator last had in that memory straight into the weights. Nothing
downstream catches it. The parameters become garbage or NaN, and the loss never recovers.

### TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep

A step's buffers are addresses baked into a compiled plan. A batch of the wrong size cannot be
accommodated, only reported. The refusal names both shapes, [4, 6] and [8, 6].

### FitNamesTheDatasetDimensionThatDisagrees

A trainer compiled for a batch of 8 and 4 classes refuses three datasets. One without batches is
`InvalidDescription` with "The dataset holds no batches". A batch size of 4 is `Incompatible` with
"batch size is 4" and "compiled for 8". Five classes are `Incompatible` with "5 classes" and
"compiled for 4".

### TheSyntheticDatasetNamesEveryExtentWhenOneIsNotPositive

`SyntheticClassification::make` refuses zero batches, a batch size of zero and zero classes as
`InvalidDescription`. Each message contains "positive extents" and the extent that is zero, as
"batches=0", "batch=0" or "classes=0".

### AllocationRefusesAnEmptyOrUnrepresentableTensor

Signed overflow in the extent product is undefined behaviour, and a compiler may assume it
away along with any check written after it. The overflow is detected before the multiply, so
the diagnostic is about the shape and not about what the overflow left behind.

### AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel

The request is a petabyte, which no allocator serves. What matters is what the caller is told:
the shape and element type it asked for. A bare byte count from four frames down would not
help, and neither would a null pointer for a kernel to dereference.

### ASingleSampleWithASingleFeatureTrains

One `Linear` over a batch of one sample with one feature and three classes. The first loss is ln(3)
within 1e-5, the loss stays finite over eight more steps at a rate of 0.5, and the weight and its
gradient are finite afterwards.

### ASingleClassIsDegenerateButNotUndefined

With C = 1 the softmax is identically one, so the loss is exactly zero and the gradient
(p - y) / N is exactly zero. Nothing learns, which is correct. A 0/0 anywhere in the recipe
would show up as NaN.

### MatmulIsExactAtTheRegisterBlockEdgeAndOnePastIt

The local-tiled GEMM stages a 64x64 block of float per work-group and pads the ragged
remainder with zeros. An extent of exactly 64 exercises the padding-free path. 65 leaves a
one-row, one-column remainder, where an off-by-one in the boundary test shows up as a wrong
answer and not as a fault. Both are checked against a host reference, with the transpose flag
the Linear layer emits.

The values that reached the device are float. The reference is computed from those same
floats, or the comparison would measure rounding and not indexing.

### TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt

The case runs for n = 64 and n = 65, with n as the batch size, the feature count and the hidden
width. Over four steps of SGD at a rate of 0.1 every loss is finite, and so is every parameter and
every gradient afterwards.

### LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude

The max shift is the only reason these inputs are representable. Without it exp(1e30) is inf
and exp(-1e30) is zero, and both give NaN. With it every magnitude must land on the same
answer, because a uniform row is uniform whatever its offset.

### LogSoftmaxSurvivesBothExtremesInsideOneRow

The row holds +1e30 against -1e30. The shift drives the winner to exactly zero and every loser
to -2e30, whose exponential underflows to zero without becoming a NaN. The row sum is exactly
one and log(1) is exactly zero, so the winner's log-probability is exactly zero. That is a
probability of one, the right answer for a logit gap of 2e30.

### LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange

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

### CrossEntropyIsFiniteOnPerfectlyConfidentPredictions

The logits carry a confidence of exp(80) either way. Anything that exponentiates without
shifting first overflows here, and anything that takes the log of a rounded zero produces
-inf.

With the same logits against the opposite targets the loss is large, and large is the correct
answer. It must not be inf, because an infinite loss makes every gradient below it a NaN.

### AdamWsFirstStepWithAZeroGradientStaysFinite

m and v are both exactly zero on step one, so the update is 0 / (sqrt(0 / bc2) + eps). Only
the eps in the denominator keeps that a zero and not a NaN. eps is added after the bias
correction, so it is not scaled away. Folding it inside makes the denominator
sqrt(0 + eps) / sqrt(bc2), which is still non-zero here and wrong everywhere else. Dropping it
makes this 0/0.

A second step still has no gradient to work with, and the bias corrections have changed. A
divide that only happened to be safe on step one would fail there.

### BuildingAndDestroyingTrainersReachesASteadyState

Every `SushiBLAS::Storage` allocates through the runtime, which records each allocation, so
`runtime.advanced().live_allocation_count()` is the number of live allocations. Whether the last
iteration leaked a buffer is then a number the test can read, where otherwise somebody would have
to watch Task Manager. One leaked Storage moves the count by exactly one.

The baseline is taken after the first iteration and not before it. A one-time lazy cache
inside the runtime is tolerated; a per-iteration leak is not.

### BatchNormUpdateStatsIsNotAnIROperationAndSaysSo

CV-7 keeps a BatchNorm's running statistics out of the IR. They are recorded into the training step
beside the graph, the same way `modules/core/optim/source/` emits K_SGD_UPDATE and K_ADAMW_UPDATE
directly. K_BATCHNORM_UPDATE_STATS is therefore one of the three identities listed in `K_UNLOWERED`
(`modules/core/graph/include/SushiAI/graph/lowering/unlowered.hpp`), and it has no row in
`K_LOWERINGS`. That is deliberate.

From the outside a deliberate gap and an omission look the same, which is why the test exists.
Pinning the fatal stop records the decision. If somebody later adds a lowering, the test fails and
asks them to say why.

## test_trainer/test_mixed_precision.cpp

Suites: `Integration_MixedPrecision`.

This is the device half of AI-4. `tests/core/autograd/test_backward.cpp` pins the graph
shape with no hardware. This file holds what only a real Engine can answer:

- `sycl::half` executes here, so the fp16 path is a computed answer and not a graph nobody ran.
- The tensors behind the graph carry the types the IR claims. A master weight is a 4-byte
  buffer, and the copy a GEMM reads is a 2-byte one.
- The trainer refuses, before recording anything, a graph whose dtypes disagree with the policy
  it was handed.

No test asserts speed. This machine has no low-precision matrix unit, and a timing gate would
pin an artifact of emulation.

### optimizer_options

The helper returns AdamW at 3e-3, the rate `sa demo mlp` uses, and not the library default.
1e-3 is the reference AdamW default and is right as a default. It does not move a hidden layer
this small far in the couple of hundred steps a test can afford, and a test whose "did it
learn" threshold has to be relaxed until it passes asserts nothing. At the demo's rate the test
and the demo answer the same question at the same setting.

### AModelWithFP16ProductsStillLearns

Correctness is the one claim the mixed-precision change can make on this hardware. It is not
faster here and cannot be.

A zeroed classifier starts at a loss of ln(4). A last loss below half of that shows a model
that learned something and did not merely run.

### TheMastersAreFourByteBuffersAndTheCopiesAreTwo

The IR says FLOAT32 and HALF, and the test checks that the tensors underneath say the same.
The two could disagree. The test reads each parameter and gradient slot and requires FLOAT32, and
it reads each cast destination the pool holds and requires HALF. A planner that sized a slot from
the wrong dtype would hand out a view that overruns its neighbour.

That the step ran at all is the loud half of the check. SushiBLAS refuses a mixed-dtype gemm
by name, so a graph with a missing conversion could not get this far.

### TheTrainerRefusesAGraphItsPolicyDisowns

A graph built without the cast pass is handed to a trainer that was told it is mixed. A caller
makes this mistake by setting one of the two knobs and forgetting the other. Without the
refusal it would not be an error; it would be an fp32 run reported as a mixed one.

### AccuracyRefusesLogitsItCannotRead

The mixed-precision change would otherwise have introduced this defect. `score_accuracy()`
reads the logits on the host. It once fell back to `data_as<float>()` for anything that was not
FLOAT64, which on a half buffer reads two elements as one and reports an accuracy that is
fiction. Nothing throws, nothing is NaN, and the run says 100%.

`build_classifier` lists the logits as observed, so the shipped path cannot reach this. A
caller running the pass themselves can, and gets an error.

Suites: `Regression_MixedPrecision`.

Every guarantee in this file fails silently, which is what puts these tests in the regression tier
and not beside the integration tests. A run that reports success cannot catch any of them.

- A master weight retyped to fp16 does not throw. The optimizer keeps working, the loss keeps
  falling, and updates smaller than the weight's own fp16 spacing are discarded. That is the
  failure master weights exist to prevent, and it looks like a learning rate that wants tuning.
- A conversion placed on the wrong side of a reduction does not throw either; the model learns
  worse.
- The control matters as much as the case. If the fp32 policy also inserted casts, every "mixed
  inserted N casts" assertion in this repository would measure the pass's unconditional behaviour
  and not the policy's decision.

### optimizer_options

1e-3 is the reference AdamW default. It is not enough to move a model this small far in the couple
of hundred steps a test can afford, and a comparison between two runs that both barely moved would
agree for the wrong reason. 3e-3 is the demo's rate, so the numbers in this file and the ones in
docs/README.md come from the same setting.

### train

Everything except the policy is fixed, so a difference between two runs is the policy and nothing
else. The initialiser and the dataset are both seeded, and the seeds are the library defaults and
not values this file picked. A change to either default therefore shows up as a changed trajectory
in both arms, not as a silent divergence in one.

The helper reads the parameter dtypes after the run and not before. Fusion and the cast pass both
rewrite nodes, and a master weight that survived the passes could still have been swapped for a
narrow copy by the optimizer wiring.

### TheFP32PolicyIsIndistinguishableFromNoPolicyAtAll

This test is the control, in the strongest form available: zero casts, the same recorded operation
count, and the same losses to the bit. If the pass ever starts inserting a conversion
unconditionally, this test fails before any mixed-precision test can look correct for the wrong
reason.

### NarrowingCostsConversionsAndNothingElseStructural

Casts are nodes, so they show up as recorded operations, and the count has to be exactly the casts.
A higher count would mean a conversion landed inside a fusable chain and cost the fusion as well.
graph/cast_insertion.hpp's ordering argument exists to avoid that failure, and no other test in
this file would notice it.

### MixedAndSinglePrecisionTrainToTheSamePlace

This test makes the numeric claim, and its tolerance is derived, not tuned until green. fp16
carries 11 bits of significand, so one rounded value is accurate to about 2^-11 = 4.9e-4 relative.
A step's loss passes through a handful of rounded values, so a few times that figure is the
expected deviation.

The measured figure on `sa demo mlp --epochs 8 --batches 16` (a wider model than this one, 128
steps) was 9.5e-4 relative, at the end of the first epoch. That is the largest loss whose six
printed decimals still carry enough digits to divide. The test's bound is 2e-2, about twenty times
the measured figure, and it is a bound and not a measurement. It was chosen so the test survives a
change of seed or schedule. Passing it says the deviation is under 2e-2 and nothing more precise.

The test first asserts that both runs learned. Two runs that both diverged would agree and the
agreement would mean nothing.

### AGELULayerIsNotFoldedUnderANarrowingPolicy

graph/precision.hpp's rules 2 and 5 meet in this test, and it asserts that they meet consistently.

Rule 2 says a fused epilogue runs inside the GEMM's kernel in the GEMM's element type, so an
activation the catalog folds has to compute in the GEMM's type or fusion would change the numbers.
Rule 5 says GELU stays wide: the tanh approximation computes `x*x*x`, which overflows fp16 at |x|
about 40.3. Since FusedForm::GEMMBiasGELU exists, the two rules would contradict each other if
nothing kept them apart.

The pass order keeps them apart, and it does so by structure. The cast pass runs before fusion, so
an activation the policy keeps wide gets a K_CAST between it and the narrowed K_BIAS_ADD, and a chain
with a conversion in the middle matches no catalog entry. Under fp32 the layer folds whole. Under a
narrowing policy it falls back to gemm+bias and the GELU runs on its own, as it did before the
entry existed.

If this test ever sees a gemm+bias+gelu node, an fp16 `x*x*x` is being computed, and nothing else
in the suite would say so.

The test also asserts that fusion did not stop altogether. The bias still folds on both layers, so
the fallback is the shorter catalog entry and not an unfused graph.

The last assertion is that the K_GELU is still in the graph and still in the master type. That is the
other half of rule 5 and the reason the fold is refused.

## test_trainer/test_compile_once.cpp

Suites: `Regression_CompileOnce`.

This file was the regression tier's first. It pins the two claims in this library whose failure is
completely silent.

"Trace once, differentiate once, compile once, replay for ever" is the premise the design is
arranged around. Both halves of it fail without a symptom:

- A trainer that recompiles per step still converges. It pays graph analysis every step, and
  nothing in the output says so.
- A replay that never dispatched still reports complete. A plan whose per-run state is not
  restored before a replay has its completion sentinel ready at once, so run() returns having
  done nothing. The loss stops moving, which reads like a bad learning rate.

SushiRuntime restores that state inside `Graph::run()`, and sushiblas cannot see a training step
at all. So both claims are pinned here, in the shape the trainer and `sa demo mlp` use.

Each test in this file spells out its own setup and shares none of it through a fixture struct.
Trainer holds raw pointers into the pool and the optimizer, so the construction order (create the
state, bind, then initialise, then optimizer, then trainer) is part of the contract, and a test that spells the order
out fails loudly if that order ever stops working. The setup matches
tests/layers/train/test_trainer/test_convergence.cpp line for line for the same reason.

### ManyFitCallsCompileTheStepExactlyOnce

`sa demo mlp` drives one fit() per epoch so it can print a per-epoch trajectory. That call pattern
differs from a single fit() with epochs=N. The integration test calls fit() once, so it could not
see a recompile hidden inside fit().

### EveryReplayDispatchesRatherThanReportingCompleteImmediately

This test looks at a replay that dispatches nothing from the outside. Such a run() returns
successfully, so the parameters never move and the loss goes flat. It shows from the second
replay onward when the first replay still runs off the state the compilation left.

### TheVeryFirstRunDispatchesRatherThanReportingCompleteOverAnEmptyBaseline

This test catches a run that dispatches nothing at the first run, where
EveryReplayDispatchesRatherThanReportingCompleteImmediately catches it only at the second replay.

That other test reads the defect off a loss that stops changing, so it needs two replays to see
it. A first run over a plan with an empty baseline fails the same way: the completion sentinel is
ready before anything is dispatched, and the run returns success having run no kernel. The only
thing that distinguishes that from a step that ran is the contents of the output buffer.

So the test first poisons the buffer with a value the graph cannot produce, and it asserts that the
poison is gone. A run that dispatched nothing leaves the poison where it was. The step reports
`compile_count()` 1 before that first run and still 1 after the four runs.

The replay loop repeats the check three times. Each pass poisons the output again, changes the
input, and requires the output to track the new input and not the old answer or the poison.

### CompileCountSurvivesEverythingTheLoopDoesBetweenSteps

ManyFitCallsCompileTheStepExactlyOnce asserts compile-once for the plain path. This test pins that
none of the things a real loop does between steps reaches back into the plan: rewriting the batch
buffers, interleaving an optimizer that records and flushes its own graph every step, stopping and
resuming, and changing a hyperparameter mid-run.

The test builds two datasets of the same shape with different contents. The compiled plan reads its
inputs by address, so swapping the data a step trains on must be a copy into those buffers and
nothing more.

The `round` loop stops and resumes: fit() returns and is entered again, which is the pattern
`sa demo mlp` uses to print a per-epoch trajectory.

The loop also halves the learning rate each round. A learning-rate schedule changes a host-side
scalar the optimizer reads per step. If that scalar ever became a compiled attribute it would force
a recompile; ir.hpp's warning on Attributes says the same.

## test_trainer/test_determinism.cpp

Suites: `Regression_Determinism`.

This file tests bit-for-bit reproducibility from end to end.

Reproducibility is the guarantee the stack is arranged around, and no test proved it before this
file. SushiRuntime promises sushiruntime_deterministic_fp. autograd/tape.cpp sums a fan-out through
an ordered K_ACCUMULATE chain and not through float atomics, and its own header says an atomic sum
"would break that quietly, showing up as a training run that cannot be reproduced rather than as a
test failure". A claim whose failure mode is "not a test failure" needs a test, and the test has to
compare bits. EXPECT_NEAR passes on a run that reorders a reduction, and reordering a reduction is
the change the claim forbids.

The file sits in the regression tier because every failure here is silent. A model that trains to
a slightly different place each run still converges and still reports good accuracy. The defect
shows on the day someone needs to reproduce a result.

Every test in this file builds a fresh Engine per run, and that is part of the contract. SushiBLAS's
set_seed() leaves the engine's monotonic RNG offset counter alone, so two generators built on one
engine draw different streams from one seed. dataset.hpp documents this.

### train_once

Everything the run depends on is either fixed in this helper or seeded: the model shape, the
initialisation seed (InitPolicy's default), the data seed and the optimizer. Two calls therefore
differ in nothing, so a difference in their output is a finding.

### expect_identical

The loss is read out of device memory as a float. Two identical runs must produce the same float,
so `==` is the right comparison. EXPECT_NEAR would admit that the two runs might differ.

### TwoIdenticalTrainingRunsAgreeByteForByte

The run is two epochs with momentum. That is enough steps for a reordered reduction to accumulate a
visible difference. With momentum, step N depends on every step before it, so a single divergent
bit cannot cancel out.

### TwoIdenticalAdamWRunsAgreeByteForByte

AdamW divides by the square root of an accumulated second moment. After that division a one-ULP
difference in the gradient is more than one ULP.

### InitialisationFromOneSeedIsBitwiseIdentical

The starting point has to be identical before anything downstream can be. Initialisation is also
the only part of the run that goes through the device RNG and not through arithmetic.

### TheSyntheticDatasetIsBitwiseIdentical

The existing dataset coverage compares through EXPECT_DOUBLE_EQ, which tolerates four ULP and
cannot see a sign difference on a zero. A dataset is either reproducible or it is not, so this test
compares the bytes.

### ReplayingOneCompiledStepOnFixedInputsIsBitwiseStable

This is the narrowest form of the claim, with the optimizer and the RNG removed: the same compiled
plan and the same buffers, run twice. Anything the runtime does non-deterministically shows up here
with nothing else to blame it on. A work-stealing reduction, an atomic accumulation and a
scheduler-dependent order are the candidates.

The optimizer's learning rate is zero. That keeps the parameters fixed, so every replay sees the
state the one before it saw.

## test_trainer/test_fused_optimizer.cpp

Suites: `Regression_FusedOptimizer`, `Regression_FusedOptimizerDeathTest`.

This file is the gate on collapsing eleven launches per parameter into one.

Fusing an optimizer update is a pure performance change, and it touches the only arithmetic in the
library that nothing downstream validates. Every failure mode it has is silent: a transposed scalar
index, an epsilon folded inside the square root, a momentum buffer read a step late, a first-step
branch that stopped being special. Each of those still trains, still converges and still reports a
healthy loss. So the fused kernel is not tested against a tolerance or a fixture. It is tested
against the separate-op sequence it replaced, byte for byte, from identical state.

The tests make five claims, in file order:

1. One kernel computes what eleven did: AdamWsFusedKernelMatchesTheSeparateOperationsExactly,
   AdamWWithoutDecayTakesTheOtherBranchAndStillMatches and
   SGDsFusedKernelMatchesTheSeparateOperationsExactly.
2. Folding the update into the plan changes no number:
   TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt and
   TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo.
3. The step compiles once, with the update inside it: TheStepCompilesOnceWithTheOptimizerInsideIt.
4. The launch count: TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve.
5. An update that cannot be replayed is refused and not downgraded:
   AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored,
   SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused and
   TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored.

The helpers from `K_BATCH` to `expect_runs_agree` belong to claims 2 and 3, which train a whole model
through the Trainer.

### Why byte for byte is the right bar

Each element's update is independent and the operation sequence is unchanged, so the only thing
that could move a bit is a difference in rounding. The one plausible source of that is a compiler
contracting `a * b + c` into an FMA inside one kernel where it could not across two. The build
rules it out: the whole stack compiles with `-ffp-contract=off` through SushiRuntime's
`sushiruntime_deterministic_fp` contract. Every expression in the fused functors is also written in
the same operand order as the kernel it replaces (`x *= alpha` is `x * alpha`; `y += alpha * x` is
`y + (alpha * x)`). Both paths read their scalars out of the same float32 device buffer, so neither
can pick up a differently rounded constant.

If a comparison in this file ever fails, the response is to find which expression moved. Loosening
the comparison is not a response.

### The second half: folding the update into the compiled plan

Fusing makes the update replay-invariant, and replay-invariance is what lets the update be recorded
into the same graph as the forward and backward passes. That is a separate change with its own

silent failure mode. A dependency edge the runtime does not see would let the update race the
backward pass that produces its gradients, and a race that usually resolves the right way looks
like training. So the folded and unfolded placements are compared byte for byte too, and
compile_count() is pinned at one with the optimizer inside the plan.

### K_ELEMENTS

`K_ELEMENTS` is 133, which is not a multiple of the 128-bit vector width. Both the wide pass and the
scalar tail of Launch::zip_inplace therefore run on every call.

### gradient_for

The first four entries of the gradient are placed by hand and not sampled: an exact zero, a
negative zero, a denormal-adjacent tiny value and a large one. Zero is where AdamW's
`0 / (sqrt(0) + eps)` lives. Negative zero distinguishes `b = staged` from `b = 0*b + 1*staged`:
the second form turns -0.0 into +0.0, and no tolerance would see that. The two extremes exercise
the square root and the division away from unity.

### drive

The helper uses no graph, no trainer and no device RNG. It has one parameter, one gradient
rewritten from the host between steps, and the optimizer's own step(). The update is then the only
thing in the measurement.

### train

The helper builds a fresh Engine, because reproducibility is a property of one engine. dataset.hpp
explains the RNG offset counter behind that.

### AdamWWithoutDecayTakesTheOtherBranchAndStillMatches

With `weight_decay = 0` the decay factor is exactly 1, and both paths must skip the multiplication
and not multiply by 1. `theta * 1.0f` happens to be bit-identical here, but the skip is a separate
branch in each path and this test is what covers it.

### SGDsFusedKernelMatchesTheSeparateOperationsExactly

The test runs every branch of the SGD recipe, because each one is a separate line of the fused
functor and a separate chance to have written it differently.

### TheUpdateInThePlanTrainsIdenticallyToTheUpdateOutsideIt

If the two runs diverge, the update is not ordered against the graph the way the separate flush
ordered it. The runtime is then missing a dependency edge, and the fix is that edge, never a
tolerance.

### TheSeparateSpellingOutsideThePlanTrainsIdenticallyToo

This is the end-to-end version of AdamWsFusedKernelMatchesTheSeparateOperationsExactly. That test
drives one optimizer over a synthetic gradient. This one runs the whole pipeline, with gradients
the backward pass produced and a trajectory long enough for a one-ULP difference in an early
denominator to become a visible one.

### TheStepCompilesOnceWithTheOptimizerInsideIt

The model has four parameters, so four fused updates joined the plan. The test states the count as
a difference and not as an absolute, so a change to the graph itself does not have to be recorded
again here.

### TheFusedUpdateIsOneTaskPerParameterWhereTheSeparateOneIsTwelve

This test counts the headline claim and does not time it. A wall clock says different things on a
loaded machine, but the number of tasks issued is the same everywhere, and it is the quantity the
change altered.

The separate AdamW update with decay issues twelve tasks: the decay scal, two for m, three for v,
four for the denominator, and two for the step. Without decay it issues eleven, which is the figure
the SushiBLAS seam's own documentation quotes.

The counting lambda runs the phases step() would run and reads the count out of the middle one. The
flush after it is required: the tasks are in the engine's builder and something has to drain them.

SGD's first step is its cheapest, because `b_1 = g_1` is one copy where later steps need a scal and
an axpy. Four is therefore the floor for SGD and not its typical cost.

### AskingToCompileAnUnfusedUpdateIsRefusedRatherThanIgnored

The separate spelling bakes the current step's bias corrections into its recorded tasks. Folded
into the plan, it would replay step one's corrections for ever and still converge. That is why the
request has to be an error and not a quiet fallback.

### SteppingAnOptimizerThatIsAlreadyInThePlanIsRefused

Before the update joined the plan, `trainer.run_step()` called `optimizer.step()`, and a caller who
also called it got a visible double update. With the update inside the replay and invisible from
the outside, the same mistake would train at twice the learning rate and produce a run that looks
fine and is wrong.

### TurningMomentumOnAfterConstructionIsRefusedRatherThanIgnored

The momentum buffer is allocated at construction or not at all, so the fused kernel's operand count
is fixed there too. Before this check existed, the separate path dereferenced a null tensor and the
fused path would have trained without momentum.

## test_trainer/test_fusion_numerics.cpp

Suites: `Regression_Fusion`.

This file is the gate on collapsing the forward and backward graph into fewer kernels.

Fusion is a pure performance change over the only arithmetic in the library that nothing downstream
validates, and every way it can be wrong is silent:

- An epilogue that adds the bias at the wrong coordinate still produces a loss that falls, because
  the model learns a different bias.
- An activation folded into a GEMM whose gradient needed the pre-activation trains towards the
  wrong place and reports nothing.
- A fused kernel reading memory the dependency tracker cannot see races its own producer, and a
  race that usually resolves the right way looks like training.

So the fused graph is not tested against a tolerance. It is tested against the unfused graph it
replaced, byte for byte, from identical state.

The tests fall into four groups, in file order. The first claims that fusing changes no number:
AReLUMLPTrainsIdenticallyFusedAndUnfused, ATanhMLPTrainsIdenticallyFusedAndUnfused and
ASigmoidMLPTrainsIdenticallyFusedAndUnfused. The second is the five GELU tests, from
AGELUMLPTrainsIdenticallyFusedAndUnfused to DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt,
which the section on the spilled activation describes. The third counts what fusion cost:
TheFusedStepIssuesTenFewerKernelsPerStep, AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias,
WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt and
DisablingOneCatalogEntryLeavesTheRestAlone. The fourth covers the dynamic selector, from
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove to EveryChoiceCarriesAReasonSomebodyCanRead.

### Why byte for byte is the right bar

The argument is the one test_fused_optimizer.cpp makes for the optimizer. Every element is computed
by the same operations in the same order. The one plausible source of a moved bit is a compiler
contracting `a * b + c` into an FMA inside one kernel where it could not across two, and
`-ffp-contract=off` rules that out; the whole stack inherits the flag through SushiRuntime's
`sushiruntime_deterministic_fp` contract. Each fused functor is written in the operand order of the
calls it replaces, and a constant that reached a kernel through a `float` parameter still does.

If a comparison fails, the response is to find which expression moved, not to widen the comparison.

### The second half: the activation that needs a spill to be fused

GELU's gradient differentiates at its input, so the epilogue that folds GELU has to write that
input to an auxiliary buffer on its way past. cuBLASLt calls this GELU_AUX_BIAS; here it is
FusedForm::GEMMBiasGELU. Every way of getting it wrong is silent, and worse than the other
activations: a forward pass that matches perfectly while the gradients are taken at the wrong point
still converges, to a different place.

A forward comparison is therefore not the test. The test is that a GELU model trained end to end
reaches byte-identical parameters after 24 optimizer steps. That is a statement about the
gradients, since nothing else moves a weight. It is taken three ways, against three different
fillings of the same buffer:

- AGELUMLPTrainsIdenticallyFusedAndUnfused compares against the graph with no fusion pass at all.
- DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt compares against the same graph with only
  this catalog row switched off, so the difference is the entry and nothing else.
- ASelectorRefusalStillFillsTheSpillAndMovesNoBit compares against a selector that declines the
  fused arm, so the aux buffer is filled by a separate copy in the lowering and not from inside
  the epilogue.

TheSpillHoldsThePreActivationItself then reads the buffer directly and compares it with a
host-computed pre-activation, because three spellings can agree while all three hold the wrong
tensor.

### train

The helper builds a fresh Engine, because reproducibility is a property of one engine; dataset.hpp
explains the RNG offset counter behind that. It trains with AdamW and not plain SGD because AdamW's
division by an accumulated square root is where a one-ULP difference stops being one ULP.

### pretend_device

The one device on the test machine is an OpenCL CPU with no dedicated scratchpad, and it supports
doubles. A selector rule that can only be exercised on hardware nobody owns is a rule nobody has
run, so the declining branches are driven from a described device.

### AReLUMLPTrainsIdenticallyFusedAndUnfused

This is the demo's own model. Both linear layers fuse, the first with its ReLU and the second
without one, and so does the cross-entropy gradient's elementwise tail. One run therefore covers
three separate collapses that reach the same arithmetic.

### ATanhMLPTrainsIdenticallyFusedAndUnfused

tanh's gradient reads its own output, so the pre-activation is dead and the whole layer folds. The
tanh epilogue is a different template instantiation from ReLU's, which is why it has its own
comparison.

### AGELUMLPTrainsIdenticallyFusedAndUnfused

This test runs the GELU hazard end to end, and its claim is about the gradients and not the forward
pass. A K_GEMM_BIAS_GELU that computed the right activation and spilled nothing would match the
unfused forward exactly and still differentiate at the wrong tensor. Nothing would throw, no value
would go non-finite, and the loss would still fall. What such a kernel could not do is land the
parameters on the same bits after 24 AdamW steps, because a weight only moves by its gradient.

The fused graph holds two fused linear layers, the first folding its GELU and the second with no
activation to fold, and the fused cross-entropy gradient tail.

The test also asserts that the loss fell. Without that assertion, "identical" could describe two
runs broken in the same way.

### TheGELUFoldIsReportedAsAFoldAndNotAsARefusal

The test asserts that the spilled value is the one the backward rule reads. It checks this on the
IR and not through a loss, because "the gradient is taken at the right tensor" has no numerical
symptom when it is wrong.

### TheSpillHoldsThePreActivationItself

The test reads the spill buffer directly and does not infer its contents from a downstream
consequence. Every other check on this catalog entry goes through gradients that agree, and two
spellings can agree while both hold the wrong tensor.

`pre` is left unobserved on purpose. The fold has to survive on the spill and not on a pin, or the
test would be pinning the wrong thing.

The reference the spill is compared against is computed in the test, on the host, from the inputs.
It is not read out of another tensor the same kernel wrote.

The test also asserts that the spill is not the post-activation, which is the value a fold without
a spill would leave for the backward pass. Where that check is made matters. gelu(x) tends to x for
large positive x, so the two tensors agree there to within a rounding error, and an assertion taken
where |x| is large and positive would be satisfied by either. On the negative side gelu tends to
zero, so a sufficiently negative element separates them by about |x|.

### ASelectorRefusalStillFillsTheSpillAndMovesNoBit

This test is the control for TheSpillHoldsThePreActivationItself. With the fused arm declined, the
aux buffer is filled by a separate copy in `record_fused_gemm` (`modules/core/graph/source/lowering/fused.cpp`) and
not by the epilogue.
That is a different code path to the same buffer, and the gradients have to come out identical.
Deleting either write makes this comparison fail.

### DisablingTheGELUFormTrainsToTheSamePlaceAsFoldingIt

This is the A/B comparison the catalog entry was written against: the same model with the catalog
row on and off. With the row off, the K_GELU is an ordinary IR node and the pre-activation is an
ordinary value. With it on, the pre-activation is a second output of a fused GEMM, written from
inside a device kernel. The parameters after 24 steps have to be the same bits, and they are the
same bits only if the gradients were.

Disabling the form costs two more kernels, as switching the ReLU form off does. The activation
becomes an out-of-place IR node again, which is a copy and a gelu. The spill costs nothing on the
unfused side, because the value the fused node spilled is the value K_BIAS_ADD writes anyway.

### TheFusedStepIssuesTenFewerKernelsPerStep

This test counts the headline claim and does not time it. A wall clock says different things on a
loaded machine, and the kernel count is the number the change controls.

The saving is ten for this model. The first layer's gemm+bias+relu replaces five calls (gemm; copy
and ger for the bias; copy and relu) with one. The second layer's gemm+bias replaces three with
one. The cross-entropy gradient's exp/sub/scale replaces five with one.

The saving has no cost beside it. Each fused GEMM needs no host-side ordering node to carry
the dependency edges execute_gemm could not declare for a bias its functor dereferences. Two layers
meant two extra nodes. SushiBLAS's seam now takes the epilogue's own access set, so
`operations_per_step` is the whole plan again and the two counts cannot drift apart.

### AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias

The fused spelling carries a hazard the unfused one does not. The epilogue dereferences the bias
from inside the GEMM kernel, and the optimizer rewrites that same bias inside the same compiled
plan. SushiBLAS cannot see into the functor, so this library has to make the declaration:
`gemm_epilogue(f).reads(bias)` in `record_fused_gemm`.

The edge is what the test is for, and a loss cannot stand in for it. With no edge the two tasks may still come out
in the right order on any given run, so a falling loss proves nothing about ordering, and this
failure mode survives an otherwise green suite. The case reads the edge from
`SushiRuntime::Graph::plan()`, as the predecessor count of the second recorded node. The node
count is asserted for a related reason:
the plan holds two nodes, and no host-side ordering node carries this edge.

The bias-writing task in the test stands in for the optimizer's update: a task in the same plan
that writes the bias the epilogue reads. It is recorded first, so an edge can only run from it to
the GEMM.

### WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt

This test covers the other half of the epilogue's access set, the half only the spilling arm has.
`.reads(bias)` orders the GEMM behind whoever writes the bias; `.writes(aux)` orders whoever reads
the spill behind the GEMM. Without the second declaration, the backward pass that differentiates at
the pre-activation races the kernel that produced it. A race that usually resolves the right way
cannot be told apart from training, so a loss cannot stand in for the edge here either. The case
asserts the predecessor count of the reader in the plan.

The test emits a reader of the pre-activation after the K_GELU, the way the differentiation pass
emits gelu_backward.

### DisablingOneCatalogEntryLeavesTheRestAlone

The per-entry switch serves two callers. TheFusedStepIssuesTenFewerKernelsPerStep needs it as its
A/B control, and a user who has measured a different result on their own model needs it as a way
out. Switching the three-op GEMM form off must fall back to the two-op form and not to nothing.

The catalog refusal costs two kernels more and not one, and the two ways of declining differ here.
Refused at the catalog, the ReLU stays an ordinary IR node. An IR activation is out of place, so it
lowers to a copy and a ReLU. Refused by the selector, as in
ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove, the node is already fused and its output
tensor is the activation's own result, so the trailing ReLU runs in place and costs one. The
numbers are the same either way; the launch counts differ.

### ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove

The seam between the two stages makes one promise. The static pass decides what could be fused and
the selector decides what is fused, so a device that wants no fusion gets an unfused plan out of a
fused graph. The numbers stay the same because every spelling of a fused node is the same
arithmetic.

### AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory

Under ForceReference the epilogue is invoked on the host by Kernels::Reference::gemm,
so any pointer it carries must be host-readable there. SushiAI's own allocations are shared USM and
would be readable. A bound allocation need not be, and nothing in the selector can ask which kind
it got.

The zip-based form runs no host arm, so it is unaffected. A blanket refusal of every form in these
modes would be a different bug from the one being prevented.

### EveryChoiceCarriesAReasonSomebodyCanRead

A selector nobody can interrogate is a selector nobody will trust. The static pass carries a report
for the same reason: both of its answers produce a plan that trains.

## test_trainer/test_learning_curve.cpp

Suites: `Regression_LearningCurve`.

This file holds the test that catches "everything passes and the model stops training".

Every other test in the suite checks a property: this gradient matches a finite difference, that
step compiled once, this loss is finite. A model can satisfy all of them and still learn measurably
worse than it did last month. A gradient scaled by the wrong constant, a momentum buffer applied a
step late and an initialisation gain quietly changed all do this. Each leaves the loss going down
along a different path, and no property test has an opinion about the path.

So TheLossTrajectoryMatchesTheRecordedReference pins the trajectory itself against a reference
recorded from a real run of this code. The tolerance carries the design of the test:

- Too tight, and the test fails on every legitimate kernel change. Reassociating a gemm's inner
  sum moves an fp32 result by an ULP or two, and twelve steps of momentum amplify that. A tolerance
  below about 1e-3 would be noise.
- Too loose, and the test stops distinguishing training from not training. A gradient wrong by a
  constant factor, or a backward rule reading the forward output where it needed the input, moves
  this trajectory by tenths.

The head steps, 0 to 5, use the absolute bound K_LOSS_TOLERANCE, 0.02, which sits between those
two bounds. The tail steps, 6 to 15, use K_LOSS_RELATIVE_TOLERANCE, 2 per cent of each step's own
reference. From step 6 on the reference values are below 0.02, so an absolute bound there would
accept a loss ten times its reference: at step 15 the reference is 0.002215 and the bound is 0.02.
The relative bound pins each tail step to its own magnitude. The shape assertions in
TheTrajectoryHasTheShapeALearningModelHas (the exact ln(C) start, the per-epoch descent, the final
value) hold even if a future toolchain widens the noise floor.

Recording the reference again is legitimate only when the change that moved it is understood and
intended. If the trajectory moved and nobody knows why, the movement is the finding and the
tolerance stays where it is.

### TheTrajectoryHasTheShapeALearningModelHas

The shape assertions are stated independently of the recorded numbers, so they survive a legitimate
re-record. Together they say that the trajectory starts where a zeroed classifier must start, goes
down, and gets most of the way to the minimum.

The descent is not monotone step by step, because each step sees a different batch. Each epoch's
mean must beat the mean of the epoch before it.

## test_trainer/test_static_memory.cpp

Suites: `Regression_StaticMemory`.

The file tests four guarantees of the static memory planner whose failure would be silent: the loop
still converges, the run still reports success, and nobody notices.

1. A training loop allocates nothing after the first step. The test counts this. Every Storage
   allocates through the runtime, so `AdvancedControls::live_allocation_count()` is the
   live-allocation count, and one extra buffer moves it by exactly one.
2. Packing shrinks the footprint. A pass that quietly stopped packing would leave every other test
   in the suite green. The footprint is measured on the unfused graph, because fusion reaches the
   same bytes by a different route: it deletes the intermediates where packing overlaps them. A
   planner test that let fusion do its work first would be measuring the catalog.
3. Packing changes no number. If it does, two values are aliasing that should not be, and the right
   response is to find them, not to record the reference again.
4. A buffer the planner packed is rewritten every replay. A K_ZERO buffer that is filled once and
   then only read holds zeros for exactly one step and then whatever overwrote it. Such a model
   trains, badly, for ever.

### train

With `packed` false, the pool has already adopted the empty plan, which covers no value and asks
for no arena. That is the behaviour the library had before the planner pass existed, which makes it
the only honest control.

The `fusion` parameter says what the fusion pass is allowed to do. Fusion is a confound in these
measurements: a fused subtree's intermediates stop being values, so fusion removes exactly the
buffers packing would overlap. Every test in this file therefore says which graph its claim is
about.

### ATrainingLoopAllocatesNothingAfterTheFirstStep

This is the headline property of the planner pass, and the test counts it. The baseline is taken
after the first step, so a one-time lazy cache inside the runtime is tolerated while a per-step
allocation is not. A per-step buffer would still train correctly and would show up only as a slowly
growing process.

### PackingShrinksTheFootprintOfTheSameModel

The footprint is measured on the unfused graph on purpose. Fusion is a different mechanism reaching
the same bytes: a fused subtree's intermediates stop being values, so they need no slot where
packing would give them a shared one. Planning what fusion already deleted would make this test a
statement about the fusion catalog and not about the planner, and it would stop testing anything
the day a new catalog entry landed.

### FusingTakesTheBuffersPackingUsedToOverlap

This test states the interaction between fusion and packing so that nobody has to rediscover it. On
the fused graph the planner's own report reads like a broken planner: every remaining activation's
live range intersects every other's, so the planner overlaps nothing and prints "0% saved". The
planner is not broken. The buffers it would overlap do not exist, because a fused subtree's
intermediates are not values any more. That is the better outcome of the two, and it is why the
arena has nothing left to do.

`unpacked_bytes` measures this directly. It is what the same tensors would cost at one allocation
each, so a drop in it means buffers disappeared and were not merely shared.

Against the packed-but-unfused graph the fused footprint is never worse. On this model it is
exactly equal, because what fusion deletes here happens to be what the planner was overlapping. The
assertion is written as `<=` and not `==` on purpose: the equality is a property of this model's
shape and not of the two passes, and pinning it would fail on the next catalog entry for a reason
nobody should have to investigate.

### PackedAndUnpackedTrainingAgreeByteForByte

If the two runs ever differ, two tensors are sharing bytes they should not. The fix is to find
which two, never to relax the comparison. The test compares bytes and not a tolerance for the
reason test_determinism.cpp gives: a tolerance cannot tell a rounding difference from a real
overlap.

### APackedZeroBufferIsRefilledOnEveryReplay

K_ZERO is the one operation whose output nothing else writes. It would work perfectly if it were
filled once outside the step, until the planner handed its bytes to a later value. The test
corrupts the buffer from the host between two replays, which is what a reused arena slot does, and
the second replay must still see zeros.

## test_binary_cross_entropy_objective.cpp, test_multi_label_objective.cpp, test_mean_squared_error_objective.cpp, test_objective.cpp

The leaf and role each of the binary, multi-label and squared-error objectives declares, what each refuses, their feeds and their metrics. Needs an engine.

Suites: `Unit_DenseObjectiveGraphs`, `Unit_DenseObjectives`.

Tests `Train::BinaryCrossEntropyObjective`, `Train::MultiLabelObjective` and
`Train::MeanSquaredErrorObjective`: the leaf and the role each declares, what `make` and the
factories refuse, what `feed` writes and what `measure` reports. The graph cases trace a
Linear, ReLU, Linear model of 6 features over a batch of 4 and need no device. The
`Unit_DenseObjectives` fixture needs an engine, and its `bind_targets` helper binds a tensor of
7s to the `targets` leaf before a feed.

### BinaryDeclaresAColumnLeafAndRealLabels

The objective is named "binary_cross_entropy" and its threshold is 0.5. In the built training
graph the `targets` leaf has the shape [4, 1] and is `spec.targets`, and the loss has one
element. The one required target has the role `K_ROLE_LABELS`, an empty sample shape, the type
float32 and 0 classes.

### MultiLabelDeclaresOneLabelPerClass

`MultiLabelObjective::make(4, 3)` is named "multi_label" and reports 3 classes. The `targets`
leaf has the shape [4, 3] and the loss has one element. The one required target has the role
`K_ROLE_LABELS`, the sample shape [3] and the type float32.

### MeanSquaredErrorDeclaresTheValuesRole

`MeanSquaredErrorObjective::make(4, 2)` is named "mean_squared_error" and reports the width 2.
The `targets` leaf has the shape [4, 2] and the loss has one element. The one required target
has the role `K_ROLE_VALUES`, the sample shape [2] and the type float32.

### MakeRefusesWhatNoLeafCanHold

Six calls fail with `InvalidDescription`. `BinaryCrossEntropyObjective::make` refuses a batch of
0 ("at least 1"), a threshold of 1 ("above 0 and below 1") and int32 ("float32", "int32").
`MultiLabelObjective::make` refuses 0 classes and a threshold of 0, and
`MeanSquaredErrorObjective::make` refuses a width of 0.

### TheFactoriesReadTheOneOutput

`make_binary_cross_entropy` over a [4, 1] output and `{"threshold": 0.9}` gives the threshold
0.9 and the batch 4. `make_multi_label` over a [4, 5] output and `{}` gives 5 classes and the
threshold 0.5. `make_mean_squared_error` over a [4, 2] output gives the width 2.

### TheFactoriesRefuseAnotherOutputAndAStrayKey

Each failure is `InvalidDescription` and names "objective". The binary factory refuses a [4, 2]
output ("one logit per sample") and a model that publishes two outputs ("publishes 2"). The
multi-label factory refuses a [4, 3, 5] output and the regression factory an [8, 2] output,
both with "[4, columns]". The key `huber` is refused by the regression factory and the key
`weights` by the multi-label factory, which also names `'threshold'`.

### BinaryFeedCopiesLabelsOfZeroAndOne

A batch whose labels are 1, 0.25, 0, 1 fails with `Incompatible`, "element 1" and "0 or 1",
and the leaf still holds four 7s. A batch of 1, 0, 0, 1 feeds ok and the leaf holds those
values.

### MeanSquaredErrorFeedReadsTheValuesRole

A batch that carries only the labels role fails with `InvalidDescription` and a message naming
`'values'`. A batch with the values 1.5, -2, 0.25, 8 under `K_ROLE_VALUES` feeds ok and the
leaf holds them.

### BinaryMeasuresAccuracyPrecisionAndRecall

Over five rows with the logits 2, 1, -1, -3, 4 and the labels 1, 0, 1, 0, 1, `measure` adds
three metrics in the order accuracy, precision, recall, with the values 3/5, 2/3 and 2/3. An
objective made with the threshold 0.95 reports the precision 1 and the recall 1/3.

### MultiLabelMeasuresTheMeanF1OfTheClasses

Over two rows of three classes `measure` adds one metric named "f1" that is a fraction and
equals 1/3.

### MeanSquaredErrorMeasuresTheMeanAbsoluteError

Over the predictions 1, 2, 3, 4 and the values 1.5, 2, 2, 6, `measure` adds one metric named
"mean_absolute_error" that is not a fraction and equals 3.5/4.

### AMetricRefusesAnOutputOfAnotherCount

A multi-label objective of 2 rows and 4 classes given a [2, 3] output fails with `Incompatible`
and a message containing "the logits", "8 values" and "holds 6". No metric is added.
