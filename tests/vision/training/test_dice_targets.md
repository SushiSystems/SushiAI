# tests/vision/training/test_dice_targets.cpp


Suites: `Medium_VisionTrainingDiceTargets` and `Medium_VisionTrainingDiceTargetsCounts`.

Tests the five leaves `fill_dice_targets` writes for a mask with no kept pixel and for float64
leaves with an ignore index of -1.

The cases allocate tensors on the CPU SYCL device through `EngineTest`, so the suite is medium:
an `InMemoryTensorSource` tensor holds no storage, and the fill writes into it.
`tests/vision/training/test_pixel_targets.cpp` checks a mixed float32 mask with ignore index 255
and the refusals that leave the leaves unchanged. This file covers the two promises those cases
leave open.

## K_UNTOUCHED

Every real leaf starts at `K_UNTOUCHED`, -7, so a leaf the fill skipped shows. The leaves are
compared for equality: every expected value is a count, a zero or a half, exact in float32 and
float64.

## Medium_VisionTrainingDiceTargets

The fixture builds INT32 and real tensors on the test's engine and reads them back.

## AMaskWithNoKeptPixelGivesZeros

Four ignored pixels give score indices 0, 3, 6 and 9, class indices 0, and zeros in `kept`,
`class_pixels` and `class_weights`.

## TheDiceLeavesCountTheKeptPixelsOfEachClass

Rows, each a mask, an ignore index and a leaf type:

- `FloatSixtyFourIgnoreMinusOne`: the mask 1, 1, 0, -1 with ignore index -1 and float64 real
  leaves gives score indices 1, 4, 6, 9 and class indices 1, 1, 0, 0. `kept` is 1, 1, 1, 0, the
  classes hold 1, 2 and 0 pixels, and the two present classes weigh 0.5 each.
- `FloatThirtyTwoIgnoreTwoFiftyFive`: the mask 0, 2, 255, 0, 2, 0 with ignore index 255 and
  float32 leaves gives score indices 0, 5, 6, 9, 14, 15 and class indices 0, 2, 0, 0, 2, 0. `kept`
  is 1, 1, 0, 1, 1, 1, the classes hold 3, 0 and 2 pixels, and the two present classes weigh 0.5
  each. An ignored pixel points at the first score of its own row and at class 0.
