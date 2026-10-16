# tests/vision/training/test_dice_targets.cpp


Suites: `Medium_VisionTrainingDiceTargets`.

Tests the five leaves `fill_dice_targets` writes for a mask with no kept pixel and for float64
leaves with an ignore index of -1.

The cases allocate tensors on the CPU SYCL device through `EngineTest`, so the suite is medium.
`tests/vision/training/test_pixel_targets.cpp` checks a mixed float32 mask with ignore index 255
and the refusals that leave the leaves unchanged. This file covers the two promises those cases
leave open.

## K_UNTOUCHED and K_TOLERANCE

Every real leaf starts at `K_UNTOUCHED`, -7, so a leaf the fill skipped shows. `K_TOLERANCE`
bounds the error of a count or of 1 / n in float32.

## Medium_VisionTrainingDiceTargets

The fixture builds INT32 and real tensors on the test's engine and reads them back.

## AMaskWithNoKeptPixelGivesZeros

Four ignored pixels give score indices 0, 3, 6 and 9, class indices 0, and zeros in `kept`,
`class_pixels` and `class_weights`.

## FloatSixtyFourLeavesHoldTheSameCounts

The mask 1, 1, 0, -1 with ignore index -1 and float64 real leaves gives score indices 1, 4, 6, 9
and class indices 1, 1, 0, 0. `kept` is 1, 1, 1, 0, the classes hold 1, 2 and 0 pixels, and the
two present classes weigh 0.5 each.
