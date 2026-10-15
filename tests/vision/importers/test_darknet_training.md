# tests/vision/importers/test_darknet_training.cpp


Suites: `Small_DarknetTraining`.

Tests `read_darknet_training` over texts parsed with `parse_darknet_config`. A head's options
are checked as seven values: box loss, objectness target, box weight, object weight,
`max_delta`, match threshold and ignore threshold. Floats are compared to 1e-6 and doubles to
1e-12.

## ReadsTheV3HeadSettingsWhenTheTrainingKeysAreAbsent

A head with only `ignore_thresh=0.7` and `truth_thresh=1` among the training keys reads as
`BoxLoss::SquaredError`, `ObjectnessTarget::One`, a box weight of 0.75, an object weight of 1,
a `max_delta` of 0, a match threshold of 1 and an ignore threshold of 0.7.

## ReadsTheV4TinyHeadSettings

With `iou_loss=ciou` and `iou_normalizer=0.07`, the head reads as `BoxLoss::CompleteIoU` with a
box weight of 0.07. The other five values are as in the case above.

## ReadsTheV4HeadSettings

With `iou_thresh=0.213` and `max_delta=5` added, the head reads a match threshold of 0.213 and
a `max_delta` of 5.

## ReadsTheV7HeadSettings

With `objectness_smooth=1`, `iou_normalizer=0.05`, `iou_thresh=0.2` and `max_delta=2`, the head
reads `ObjectnessTarget::BestIoU`, a box weight of 0.05, a match threshold of 0.2 and a
`max_delta` of 2. `cls_normalizer=0.5` is accepted.

## KeepsTheSolverDefaultsWhenTheKeysAreAbsent

With no solver key, the solver has a learning rate of 0.001, a momentum of 0.9, a decay of
0.0005, a `burn_in` of 0, no steps and no scales, a batch of 1, 1 subdivision and a
`max_batches` of 0.

## ReadsEverySolverKeyTheFileSpells

Batch 64, subdivisions 16, momentum 0.949, decay 0.0005, learning rate 0.00261, `burn_in` 1000,
`max_batches` 500200, the steps 400000 and 450000 and the scales 0.1 and 0.1 are each read to
the value the text states.

## KeepsTheHeadsInFileOrderAndSkipsOtherSections

A text with a convolution, a head, a route and a second head gives two heads, with the box
weights 0.07 and 0.05 in that order.

## ReadsMSEAndObjectnessSmoothZeroAsTheDefaults

`iou_loss=mse` with `objectness_smooth=0` gives the seven values of a head without training
keys.

## SelectsTheDarknetCompatibleBoxGradientForEveryHead

A head with `iou_loss=ciou` and a head with `iou_loss=mse` both read
`BoxGradient::DarknetCompatible`. A default `HeadTrainingOptions` has `BoxGradient::Analytic`.

## AcceptsTruthThreshOneAndLabelSmoothZero

`truth_thresh=1.0` with `label_smooth_eps=0` reads as one head.

## RefusesAPolicyOtherThanSteps

`policy=exp` fails with `Unsupported`, "line 2" and "policy".

## RefusesATruthThreshOtherThanOne

`truth_thresh=0.9` fails with `Unsupported`, "line 3" and "truth_thresh".

## RefusesANonZeroLabelSmooth

`label_smooth_eps=0.1` fails with `Unsupported`, "line 3" and "label_smooth_eps".

## RefusesAnIoULossOtherThanMSEAndCIoU

`iou_loss=giou` fails with `Unsupported`, "line 3" and "iou_loss".

## RefusesAClsNormalizerThatIsNotANumber

`cls_normalizer=half` fails with `InvalidDescription`, "line 3" and "cls_normalizer".

## RefusesAnObjectnessSmoothOtherThanZeroOrOne

`objectness_smooth=2` fails with `InvalidDescription`, "line 3" and "objectness_smooth".

## RefusesAHeadWhoseMatchThresholdIsZero

`iou_thresh=0` fails with `InvalidDescription`, "line 2", the head section's line, and
"match threshold".

## RefusesAHeadWhoseIgnoreThresholdIsAboveOne

`ignore_thresh=1.5` fails with `InvalidDescription`, "line 2" and "ignore threshold".

## RefusesALearningRateThatIsNotANumber

`learning_rate=abc` fails with `InvalidDescription`, "line 2", "learning_rate" and "abc".

## RefusesANonFiniteLearningRate

`learning_rate=nan` fails with `InvalidDescription`, "line 2" and "nan".

## RefusesABatchBelowOne

`batch=0` fails with `InvalidDescription` and a message holding "batch".

## RefusesAnEmptyEntryInAList

`steps=1,,2` fails with `InvalidDescription`, "line 2" and "steps".

## RefusesStepsAndScalesOfDifferentLengths

Two steps with one scale fail with `InvalidDescription`, "line 1", "'steps' holds 2" and
"'scales' holds 1".
