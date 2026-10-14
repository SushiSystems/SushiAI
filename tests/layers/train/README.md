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

The model of `Test::build_two_output_model`, from `tests/common/test_utils.hpp`, publishes
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

`build_two_output_model` is `Test::build_two_output_model` from `tests/common/test_utils.hpp`,
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
