# tests/vision/importers/test_darknet_training/test_recipe.cpp


Suites: `Small_DarknetTraining`, `Small_DarknetTrainingLearningRate`.

Tests `read_darknet_training` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_training.hpp`. Each case
parses `.cfg` text with `parse_darknet_config` and reads the solver and the heads. Every value
asserted is copied from the text or from the defaults the header states, so each is compared
for equality.

## ReadsTheV3HeadSettingsWhenTheTrainingKeysAreAbsent

A `[yolo]` section with `ignore_thresh=0.7` and `truth_thresh=1` gives one head with the
squared-error box loss, the objectness target One, a box weight of 0.75, an objectness weight of
1, no delta bound, a match threshold of 1 and an ignore threshold of 0.7.

## ReadsTheV4TinyHeadSettings

With `iou_loss=ciou` and `iou_normalizer=0.07` the head has the CIoU box loss and a box weight of
0.07. The other values are those of the v3 head.

## ReadsTheV4HeadSettings

With `iou_thresh=0.213` and `max_delta=5` added, the head has a match threshold of 0.213 and a
delta bound of 5.

## ReadsTheV7HeadSettings

With `iou_thresh=0.2`, `objectness_smooth=1`, `max_delta=2` and `iou_normalizer=0.05` the head
has the objectness target BestIoU, a box weight of 0.05, a delta bound of 2 and a match threshold
of 0.2. `cls_normalizer=0.5` is read and reaches no option.

## KeepsTheSolverDefaultsWhenTheKeysAreAbsent

A `[net]` section with only `width` gives a learning rate of 0.001, a momentum of 0.9, a decay of
0.0005, no burn-in, no steps, no scales, a batch of 1, 1 subdivision and 0 maximum batches.

## ReadsEverySolverKeyTheFileSpells

A `[net]` section spelling every solver key reads back as a batch of 64, 16 subdivisions, a
momentum of 0.949, a decay of 0.0005, a learning rate of 0.00261, a burn-in of 1000, 500200
maximum batches, the steps 400000 and 450000 and the scales 0.1 and 0.1.

## KeepsTheHeadsInFileOrderAndSkipsOtherSections

Two `[yolo]` sections with `iou_normalizer` 0.07 and 0.05, parted by a `[convolutional]` and a
`[route]` section, give two heads whose box weights are 0.07 and 0.05, in that order.

## SelectsTheDarknetCompatibleBoxGradientUnderCIoU

A head with `iou_loss=ciou` has `BoxGradient::DarknetCompatible`.

## SelectsTheDarknetCompatibleBoxGradientUnderMSE

A head with `iou_loss=mse` has `BoxGradient::DarknetCompatible`.

## AcceptsTruthThreshOneAndLabelSmoothZero

`truth_thresh=1.0` and `label_smooth_eps=0` are accepted and the file gives one head.

## Refusals

Each case below reads one `.cfg` and expects a failure whose message contains the listed words.

| Case | Text | Status | Message holds |
| --- | --- | --- | --- |
| RefusesAPolicyOtherThanSteps | `policy=exp` | `Unsupported` | "line 2", "policy" |
| RefusesATruthThreshOtherThanOne | `truth_thresh=0.9` | `Unsupported` | "line 3", "truth_thresh" |
| RefusesANonZeroLabelSmooth | `label_smooth_eps=0.1` | `Unsupported` | "line 3", "label_smooth_eps" |
| RefusesAnIoULossOtherThanMSEAndCIoU | `iou_loss=giou` | `Unsupported` | "line 3", "iou_loss" |
| RefusesAClsNormalizerThatIsNotANumber | `cls_normalizer=half` | `InvalidDescription` | "line 3", "cls_normalizer" |
| RefusesAnObjectnessSmoothOtherThanZeroOrOne | `objectness_smooth=2` | `InvalidDescription` | "line 3", "objectness_smooth" |
| RefusesAHeadWhoseMatchThresholdIsZero | `iou_thresh=0` | `InvalidDescription` | "line 2", "match threshold" |
| RefusesAHeadWhoseIgnoreThresholdIsAboveOne | `ignore_thresh=1.5` | `InvalidDescription` | "line 2", "ignore threshold" |
| RefusesABatchBelowOne | `batch=0` | `InvalidDescription` | "batch" |
| RefusesAnEmptyEntryInAList | `steps=1,,2` | `InvalidDescription` | "line 2", "steps" |
| RefusesStepsAndScalesOfDifferentLengths | `steps=100,200` with `scales=0.1` | `InvalidDescription` | "line 1", "'steps' holds 2", "'scales' holds 1" |

## Small_DarknetTrainingLearningRate.IsRefused

A `learning_rate` of `abc` and of `nan` each fail with `InvalidDescription`, "line 2",
"learning_rate" and the value.
