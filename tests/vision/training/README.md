# tests/vision/training

The tests of the module `vision/training`, one file per header of `modules/vision/training/include/`. The suite prefix of a case gives its ctest label.

## test_anchor_objective.cpp

Suites: `Unit_AnchorObjective`.

Tests `AnchorObjective` of the vision training module over a 64-pixel network, a batch of 2
and two classes. Most cases use two heads, "head.0" on a 4x4 grid and "head.1" on a 2x2 grid,
each with two anchors and so 14 channels.

### DeclaresOneLeafPerHeadInTheShapeOfItsOutput

`declare` adds two leaves, "delta.head.0" of the shape [2, 4, 4, 14] and "delta.head.1" of
[2, 2, 2, 14]. Both are `Float32` and do not require a gradient, and `leaf_role(1)` is
"delta.head.1".

### TheRootSumsOneDeltaLossPerHeadScaledByTheBatch

`loss` adds MUL, RESHAPE, SUM_ROWS and SCALE once per head and then one ADD. Each MUL reads the
leaf of its head's name and then that head's output, although the model published "head.1"
first. Each SCALE carries -1/2 to 1e-12. The ADD reads the two SCALE results and writes the
root, whose shape is [1].

### AModelWithoutAHeadThrowsNamingTheHead

`loss` over outputs that hold "head.0" alone throws an error naming "head.1".

### AHeadOfAnotherShapeThrowsNamingBothShapes

`loss` over a "head.0" of the shape [2, 4, 4, 21] throws an error naming "[2, 4, 4, 14]" and
"[2, 4, 4, 21]".

### TheCostIsTheSumOverTheHeads

For two heads of one anchor and one class, the second with an object weight of 0.5, and deltas
set by hand, `measure_cost` returns a total of 7 + 2.5, a non-box share of 6.5 + 0.25, a box
share of 0.5 + 2.25 and a stored cost of 14 + 5, each to 1e-12.

### NamesItselfAndIsFedByItsOwnStep

`name()` is "anchor_delta" and `is_fed_by_batch()` is false. The objective holds two heads, four
anchors and a network width of 64.

### AHeadCountOtherThanTheLayoutsIsRefused

`make` with no head, and with one head, over a two-head layout fails with `InvalidDescription`.
The messages hold "0 heads" and "1 heads", each with "holds 2".

### HeadsOfDifferentBatchesAreRefused

A second head with a batch of 3 fails with `InvalidDescription`, and the message holds
"head.1", "batch of 3" and "head.0".

### AnEmptyOrARepeatedOutputNameIsRefused

A head with an empty output name fails with "head 1" and "names no model output". Two heads
that both name "head.0" fail with "both read" and "head.0". Both codes are
`InvalidDescription`.

### AnExtentOrANetworkSideBelowOneIsRefused

A head with a height of 0 fails with "head.0" and "[2, 0, 4]". A network size of 0 by 64 fails
with "0 by 64". Both codes are `InvalidDescription`.

### OptionsTheValidatorsRefuseAreRefused

A head with no class fails with `InvalidDescription` and "at least one class". A head with a
box weight of -1 fails with `InvalidDescription`.

## test_anchor_objective_factory.cpp

Suites: `Unit_AnchorObjectiveFactory`.

Tests `build_anchor_objective` and `make_anchor_objective` in
`modules/vision/training/include/SushiAI/vision/training/anchor_objective_factory.hpp`. The
context has a batch of 2, 2 classes and the input sample [8, 6, 3]. Its task states four
anchors and two heads of two anchors each: "head.0" with the coding "logit" and "head.1" with
"activated" and a `scale_xy` of 1.5. It publishes the outputs "head.0" of [2, 4, 3, 14],
"stem" of [2, 4, 3, 8] and "head.1" of [2, 2, 1, 14].

### EachHeadTakesItsExtentsFromItsPublishedOutput

With an empty section the objective has a network of height 8 and width 6, four anchors and
two heads. The first reads "head.0" with a batch of 2, a height of 4, a width of 3, 2 classes
and `HeadCoding::Logit`. The second reads "head.1" with a height of 2, a width of 1,
`HeadCoding::Activated` and a `scale_xy` of 1.5.

### TheSectionSetsTheTrainingOptionsOfEveryHead

With the section `{"type": "anchor", "box_weight": 0.5}`, every head has a box weight of 0.5
and the default object weight and box loss of `HeadTrainingOptions`.

### TheClassCountOfTheDataMustBeTheTasks

A context with 0 classes builds. One with 80 classes fails with `Incompatible`, "80 classes"
and "task states 2".

### AModelWithoutATaskIsRefused

A context whose task is a null value fails with `InvalidDescription`, "'task'" and "'anchor'".

### AHeadTheModelDoesNotPublishIsRefused

Without the output "head.1" the build fails with `InvalidDescription`, "head 1", "'head.1'",
"'head.0'" and "'stem'".

### AnOutputOfOtherExtentsIsRefused

The build fails with `Incompatible` for "head.0" with 15 channels ("'head.0'",
"[2, 4, 3, 15]", "[2, H, W, 14]"), for "head.1" with a batch of 3 ("'head.1'",
"[2, H, W, 14]") and for "head.0" of rank 2 ("'head.0'", "[2, H, W, 14]").

### AnInputOfAnotherRankIsRefused

An input sample of [192] fails with `InvalidDescription`, "[192]" and "[H, W, C]".

### AKeyTheSectionDoesNotHaveIsRefused

A section with the key `label_smoothing` fails with `InvalidDescription`, "objective" and
"'label_smoothing'".

### TheRegistryFormPrefixesARefusalWithItsContext

`make_anchor_objective` with the context "the objective block" returns an objective whose
`name` is "anchor_delta". With a null task it fails with `InvalidDescription`,
"the objective block" and "'task'".

## test_anchor_training_loop.cpp

Suites: `Unit_AnchorTrainingLoop`.

Tests `run_anchor_training` in
`modules/vision/training/include/SushiAI/vision/training/anchor_training_loop.hpp` and
`AnchorTrainingStep::restore_run_count`. The fixture is an `EngineTest`. It builds a model of
one convolution published as "head" over an [8, 8, 3] input, an `AnchorObjective` of one head
with two anchors and 2 classes on a 4 by 4 grid, an SGD made with the rate 0.2 and the
`AnchorTrainingStep` over them. Its dataset, named "truths", holds three batches of two seeded
images with the three truth roles of `describe_truth_roles(2, 2)`.

### EachUpdateTakesTheRateOfItsIteration

With a `StepSchedule` of base rate 0.1 that halves at the updates 2 and 5, a run of two epochs
calls `on_update` six times. Call i reports the epoch i / 3 + 1, the batch i modulo 3, the
iteration i + 1 and the rates 0.1, 0.05, 0.05, 0.05, 0.025, 0.025. The report states 6
updates and a last rate of 0.025, which is also the optimizer's rate, and its first and last
costs equal those of the first and last call. The optimizer's step count is 6 and the
training step's compile count is 1.

### ACosineScheduleSetsTheRateOfEachUpdate

With a `CosineSchedule` of base rate 0.1, 6 total iterations and a final rate of 0.01, each of
the six updates reports the rate `rate_at` gives its iteration, and the report's last rate is
`rate_at(6)`.

### ARunWithoutAScheduleKeepsTheOptimizersRate

A run without options reports 3 updates and a last rate of 0.2, and the optimizer's rate
stays 0.2.

### ADatasetTheStepCannotReadIsRefusedBeforeAnyUpdate

The run fails with `Incompatible` for batches of three images ("'truths'", "batch of 3",
"declared for 2"), and with `InvalidDescription` for a dataset of no batch ("'truths'",
"no batches") and for one whose only role is `labels` ("'truths'", "'boxes'"). The
optimizer's step count stays 0.

### OptionsARunCannotUseAreRefused

The run fails with `InvalidDescription` for 0 epochs ("at least one epoch", "0") and for a
schedule of two steps and one scale ("2 steps", "1 scales"). The optimizer's step count stays
0.

### ARunStartsAtTheEpochAndTheBatchItsOptionsState

With two epochs, `first_epoch` 3 and `first_batch` 2, over a dataset that records each
`begin_epoch`, the run makes 4 updates and begins the epochs 3 and 4. The first call reports
the epoch 1 and the batch 2, and the second the epoch 2 and the batch 0.

### AStartTheDatasetLacksIsRefused

`first_epoch` -1 fails with `InvalidDescription`, "epoch index" and "-1". `first_batch` 3
fails with `Incompatible`, "batch 3", "'truths'" and "3 batches". The optimizer's step count
stays 0.

### ARestoredRunCountIsTheCountOfBothParts

The step's run count starts at 0. After `restore_run_count(4)` it is 4, and so is the run
count of the part before the host stage. After a run of three batches both are 7. A second
`restore_run_count` then fails with `InvalidDescription`, "before its first run" and "run 7".

### ACallbackThatThrowsEndsTheRunAsAFailure

With an `on_update` that throws `std::runtime_error("stop here")`, the run returns a failure
whose message holds "stop here", and the optimizer's step count is 1.

## test_head_delta_loss.cpp

Suites: `Unit_HeadDeltaLoss`.

Tests the IR that `emit_head_delta_loss` of the vision training module emits over a head
parameter and a delta input of the same shape, [2, 3, 3, 7] in this file, with the scale 0.25.

### EmitsFourNodesEndingInAScaleOfOneElement

The emitter adds `sushiai.mul`, `sushiai.reshape`, `sushiai.sum_rows` and `sushiai.scale`, in
that order. MUL reads the delta and then the head, and each later node reads the one before it.
The RESHAPE result has the shape [126, 1], the loss has the shape [1], and the SCALE node's
attribute 0 is -0.25 to 1e-12.

### TheResultHasTheElementTypeOfTheHead

Over a `Float64` head and a `Float64` delta, the loss has the element type `Float64`.

### TheResultIsARootAutogradDifferentiates

`Autograd::differentiate` of the loss with respect to the head does not throw and returns one
gradient of the head's shape. The delta's `requires_grad` is false.

### ADeltaOfAnotherShapeThrowsNamingBothShapes

A delta of the shape [2, 3, 3, 6] throws an error naming "MUL needs equally shaped operands",
"[2, 3, 3, 6]" and "[2, 3, 3, 7]", and adds no node.

### ADeltaOfAnotherElementTypeThrowsNamingBothTypes

A `Float64` delta against a `Float32` head throws an error naming
"MUL needs operands of one element type", "float32" and "float64", and adds no node.

Suites: `Integration_HeadDeltaLoss`, in `sushiai_vision_tests`.

The graph holds a `[2, 3, 3, 7]` float32 parameter, the head, and an input leaf of the same shape,
the delta. `emit_head_delta_loss` joins them with a scale of 0.5, and the loss is differentiated at
the head. Both leaves are filled from seeded host noise, with every third delta set to zero. The
expectations start from the values rounded to float32.

### TheGradientAtTheHeadIsMinusScaleTimesDelta

Each element of the gradient equals `-0.5 * delta` within 1e-6.

### TheValueIsMinusScaleTimesTheSumOfProducts

The loss is one value and equals `-0.5 * sum(delta * head)`, summed in double on the host, within
1e-4.

## test_pixel_targets.cpp

Suites: `Unit_VisionTrainingPixelTargets`.

Tests `validate_pixel_labels` in
`modules/vision/training/include/SushiAI/vision/training/pixel_labels.hpp`,
`fill_pixel_targets` in
`modules/vision/training/include/SushiAI/vision/training/pixel_targets.hpp` and
`fill_dice_targets` in
`modules/vision/training/include/SushiAI/vision/training/dice_targets.hpp`. The fixture is an
`EngineTest`. The masks hold 6 pixels over 3 classes with the ignore index 255. Before a fill
every index leaf holds -7 and every real leaf -7.0, and reals are compared within 1e-7.

### TheLabelCheckNamesThePixelOutOfBounds

`validate_pixel_labels` passes a mask of classes and 255. It fails with `Incompatible` for a
3 at pixel 3 ("holds 3 at pixel 3", "below 3", "255"), a -1 at pixel 4
("holds -1 at pixel 4"), an expected count of 7 ("6 pixels", "expects 7") and a `Float32` mask
("mask is").

### AKeptPixelIndexesItsClassAndSharesTheWeight

For the mask 0, 2, 255, 1, 1, 255, `fill_pixel_targets` writes the indices 0, 5, 6, 10, 13, 15
and the weights 0.25, 0.25, 0, 0.25, 0.25, 0.

### AMaskWithNoKeptPixelGivesZeroWeights

For a mask of six 255s, the indices are 0, 3, 6, 9, 12, 15 and the six weights are 0.

### ARefusedPixelFillChangesNothing

`fill_pixel_targets` fails with `Incompatible` for a mask with a 9 at pixel 4
("holds 9 at pixel 4"), for a weights leaf of five elements ("6 pixel indices",
"5 pixel weights") and for a `Float32` indices leaf ("pixel indices are"). Afterwards the
indices still hold -7 and the weights -7.0.

### TheDiceLeavesCountTheKeptPixelsOfEachClass

For the mask 0, 2, 255, 0, 2, 0, `fill_dice_targets` writes `score_indices` 0, 5, 6, 9, 14,
15, `class_indices` 0, 2, 0, 0, 2, 0, `kept` 1, 1, 0, 1, 1, 1, `class_pixels` 3, 0, 2 and
`class_weights` 0.5, 0, 0.5.

### ARefusedDiceFillChangesNothing

`fill_dice_targets` fails with `Incompatible` for a mask with a 3 at pixel 4
("holds 3 at pixel 4"), for a `class_weights` leaf of two elements ("'class_weights'",
"2 elements") and for a `Float32` `class_indices` leaf ("'class_indices'", "int32").
Afterwards `score_indices` and `class_indices` still hold -7, and `kept` and `class_pixels`
-7.0.

## test_segmentation_metrics.cpp

Suites: `Unit_VisionTrainingSegmentationMetrics`.

Tests `measure_segmentation` in
`modules/vision/training/include/SushiAI/vision/training/segmentation_metrics.hpp`. The
fixture is an `EngineTest`. A step holds one output of `Float32` logits of [1, 2, 2, 3] and a
batch whose `mask` role is `Int32` of [1, 2, 2], with the ignore index 255.

### TheMetricsEqualTheHandCount

The logits pick the classes 0, 1, 0, 1 against the mask 0, 2, 255, 1. `measure_segmentation`
reports two metrics: `mean_iou` 0.5 and `pixel_accuracy` 2/3, each within 1e-15. The
`mean_iou` metric has `is_fraction` set.

### AStepThatDoesNotFitIsRefused

`measure_segmentation` fails with `InvalidDescription` for a step of two outputs
("published 2") and for a batch without a mask ("'mask'"). It fails with `Incompatible` for a
mask with a 3 at pixel 1 ("true class of pixel 1", "is 3") and for extents of 4 classes over
logits of 3 ("the logits"). No metric is reported by the four calls.

## test_truth_from_roles.cpp

Suites: `Unit_TruthRoles`, `Unit_TruthFromRoles`.

Tests `describe_truth_roles` and `truth_from_roles` in
`modules/vision/training/include/SushiAI/vision/training/truth_from_roles.hpp`. The
`Unit_TruthFromRoles` fixture is an `EngineTest`. Its batch holds two images of three slots
over 5 classes: the slots 0, 2 and 5 are in use with the classes 4, 2 and 1, and the padding
slots hold zeros.

### DescribesBoxesClassesAndValidOverTheSlots

`describe_truth_roles(3, 5)` gives three roles: `K_ROLE_BOXES` of [3, `K_TRUTH_BOX_VALUES`],
`Float32` and 0 classes; `K_ROLE_CLASSES` of [3], `Int32` and 5 classes; `K_ROLE_VALID` of
[3], `Float32` and 0 classes.

### DescribesAClassCountOfZero

`describe_truth_roles(1, 0)` succeeds and its classes role states 0 classes.

### RefusesNoSlotAndANegativeClassCount

`describe_truth_roles` fails with `InvalidDescription` for 0 slots ("object slot", "got 0")
and for -1 classes ("classes", "got -1").

### CopiesTheCornersWidensTheClassesAndMakesTheFlagsBytes

`truth_from_roles` gives a truth of batch 2 and 3 boxes per image whose `boxes` equal the
corners written, whose `labels` are 4, 0, 2, 0, 0, 1 and whose `valid` bytes are 1, 0, 1, 0,
0, 1.

### FindsTheRolesInAnotherOrder

With the first and third target of the batch swapped, the truth has the same boxes and valid
bytes.

### KeepsTheStorageOfATruthBetweenBatchesOfOneSize

After a second `truth_from_roles` into the same truth, `boxes`, `labels` and `valid` have the
data pointers of the first call, and the first label is the 3 written in between.

### RefusesABatchThatLacksARole

Without any one of the three roles the call fails with `InvalidDescription`, "no role '<role>'"
and "it carries". A batch with no target fails with "no role 'boxes'" and "(none)".

### RefusesARoleOfAnotherElementTypeOrRank

The call fails with `Incompatible` for `Float32` classes ("'classes'", "int32",
"float32 [2, 3]"), for `Int32` valid flags ("'valid'", "float32", "int32 [2, 3]") and for
boxes of sample shape [12] ("'boxes'", "rank 3", "[2, 12]").

### RefusesABoxOfAnotherWidthAndRolesOfOtherSlots

The call fails with `Incompatible` for boxes of three values ("'boxes'", "[2, 3, 3]",
"4 values"), for classes of four slots ("'classes'", "[2, 4]", "[2, 3, 4]") and for valid
flags of two slots ("'valid'", "[2, 2]", "[2, 3, 4]").

### RefusesAFlagThatIsNeitherZeroNorOneAndLeavesTheTruth

After a first copy, a valid flag of 0.5 fails with `InvalidDescription`, "'valid'", "0.5" and
"slot 1 of image 1", and the truth's first label stays 4 although the batch now holds 1. A
flag of NaN fails with `InvalidDescription`, "'valid'" and "slot 1 of image 1".

## test_anchor_training_step.cpp

Suites: `Integration_AnchorTrainingStep`, in `sushiai_vision_tests`.

The model reads `[8, 8, 3]` images through two normalised stride-2 convolutions of 8 channels, each
followed by a LeakyReLU of 0.1 and a 1x1 head convolution of 14 channels. It publishes `head.0` on a
4 by 4 grid, `head.1` on a 2 by 2 grid and `stem`, the first convolution before its normalisation.
The layout holds four anchors, two for each head, over two classes. The batch is two seeded images
with three truths, two assigned to head 0 and one to head 1. An `AnchorTrainingStep` runs the
training graph around its host stage, with plain SGD at a rate of 0.1.

### TheLeavesFedAreTheDeltasOfTheHeads

After one run, the delta of each head is computed on the host from the head the step published
with `compute_head_delta`. The leaf the training graph was fed equals that delta element for element,
with a tolerance of 0, and the assignments are 2 and 1. The stored cost is the sum of the squared
deltas and the total is that sum over the batch of two, both to 1e-9 relative.

### OneStepMovesAHeadBiasAlongItsDelta

Each of the 14 entries of `fine.bias` after one step equals its value before plus rate / batch times
the sum of its channel's deltas over the 32 positions, within 1e-5. At least one bias moves by more
than 1e-7.

### OneStepBlendsARunningMeanOnce

Each entry of `stem_norm.running_mean` after one step equals `(1 - momentum) * before + momentum *
mean`, within 1e-5, where the mean is taken on the host over the published `stem` output. The
largest batch mean must exceed 1e-2, so that a second blend would be visible.

### ATruthOfAnotherBatchIsRefused

A truth of three images against a step of two is refused as `Incompatible`. The message contains "3
images" and "batch of 2".

### ALabelOutsideTheClassesIsRefused

A valid row whose label is 2, with two classes, is refused as `InvalidDescription`. The message
contains "row 2", "label 2" and "head.0".

### ACentreOutsideTheInputIsRefused

Row 0 of image 1 is moved to columns 7 to 9, which puts its centre at exactly 1. The run is refused
as `InvalidDescription`, and the message contains "truth row 0 of image 1" and "centre".

### AnObjectiveNamingAnUnpublishedHeadIsRefused

An objective that names `head.9` is given to `AnchorTrainingStep::make` over a model that publishes
`head.0` and `head.1`. The step is refused as `InvalidDescription`, and the message names "head.9".

### AHostStageThatReadsAValueTheGraphLacksIsRefused

`Train::TrainingStep::make` over the training graph, with a host stage that reads the id one
behind the last value, fails with `InvalidDescription`, "The host stage reads value" and
"values".

### AHostStageBehindAReaderOfItsLeafIsRefused

The stage reads the loss and writes the delta leaf of head 0. The loss is computed from that
leaf, so `make` fails with `InvalidDescription`, "before the host stage" and "which the stage
writes".

### AStepWithoutAHostStageRefusesHostWork

A `Train::TrainingStep` made with no host stage reports `has_host_stage()` false and a null
`lead()`. `run` with host work fails with `InvalidDescription` and "without a host stage",
the work is not called and the run count stays 0.

## test_pixel_cross_entropy_objective.cpp

Suites: `Integration_SegmentOverfit`, in `sushiai_vision_tests`.

The model, `TinySegmenter`, is a 3x3 convolution of stride 2, a second 3x3 convolution and a
`SegmentHead` that upsamples by 2, all 8 channels wide, over two classes. The dataset is one batch
of two 16 by 16 images with three channels: sample 0 is split left and right, sample 1 above and
below, and two corners of each mask hold the ignore value 255. `Train::Trainer` fits it with
`PixelCrossEntropyObjective` and SGD at a rate of 0.1 for 80 epochs. The source marks the epoch
count, the rate and both thresholds as not yet confirmed by a run.

### TwoConvolutionsAndAHeadFitOneBatch

The report holds 80 steps and one compile. The last loss is below half the first, `mean_iou` of the
last step is above 0.8, and `pixel_accuracy` is at least `mean_iou`.
