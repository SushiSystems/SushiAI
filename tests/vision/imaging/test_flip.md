# tests/vision/imaging/test_flip.cpp


Suites: `Unit_ImagingFlip`.

Tests the plan and the application of the horizontal flip to an image and its labels.

This file needs no device. `Unit_ImagingFlip` covers `modules/vision/imaging/include/SushiAI/vision/imaging/flip.hpp`.
`plan_flip` makes the one random draw and records the decision in a `FlipPlan`; `apply_flip`
then mirrors an image or a label set from the plan and draws nothing.

## Constants

`K_BOX_TOLERANCE` is 1e-4 px. `K_SEED_COUNT` is 10000, the seeds 0 to 9999 a statistical case
draws from. `K_FREQUENCY_TOLERANCE` is 0.03, six standard deviations of a frequency of 0.5 over
10000 draws.

## make_counted_image

`make_counted_image` returns an image whose byte at offset i is i modulo 256.

## expect_same_image

`expect_same_image` expects two images to have the same size, channel count and bytes.

## expect_box

`expect_box` compares the four coordinates of two boxes within `K_BOX_TOLERANCE`.

## PlanRecordsTheSizeAndMakesExactlyOneDraw

The plan holds the source size 480 by 640. Its `is_flipped` equals one `bernoulli` draw of a
second generator with the same seed, and the two generators are equal afterwards.

## ProbabilityZeroNeverFlipsAndOneAlwaysFlips

Over 10000 seeds, probability 0 never flips and probability 1 always does, and both leave the
generator in the same state.

## DefaultProbabilityFlipsAboutHalfOfTheSeeds

With the default settings the flip frequency over 10000 seeds is 0.5 within
`K_FREQUENCY_TOLERANCE`.

## PlanRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5 and -0.1 throw with a message containing "validate_flip" and
"probability", and the generator is unchanged.

## PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw

A NaN probability throws and the generator is unchanged.

## PlanRejectsASideThatIsNotPositiveBeforeAnyDraw

Sizes of 0 by 8 and 8 by -2 throw with a message containing "plan_flip", "source size" and the
size, and the generator is unchanged.

## SourceWithoutPixelsIsRejected

`apply_flip` on a default `ImageView` throws with "apply_flip" and "no pixels".

## DestinationThatIsTheSourceIsRejected

`apply_flip` with the image as both source and destination throws with "destination holds the
source's pixels", and the image is unchanged.

## ViewInsideTheDestinationIsRejected

A view of three rows of the destination is refused with the same message, and the image is
unchanged.

## FlippedImageHoldsEachRowInReverseColumnOrder

For a 2 by 3 image of three channels, pixel x of each destination row equals pixel 2 - x of the
source row, channel by channel. The first pixel of row 0 is bytes 6, 7, 8: the pixels are
reversed and the channels inside a pixel are not.

## UnflippedImageIsCopiedUnchanged

A plan with `is_flipped` false copies the source.

## FlippingAnImageTwiceRestoresIt

A 4 by 7 image of three channels and a 3 by 4 image of one channel both come back after two
flips.

## ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 3 source with a plan for 2 by 4 or for 5 by 3 throws with a message naming the plan's
size and "2x3".

## FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder

In a 640 px wide image the box (100, 50, 300, 200) becomes (340, 50, 540, 200), and a box
spanning the whole width is unchanged. Classes and order are kept, and x1 stays at or below x2.

## UnflippedBoxesAreReturnedUnchanged

An unflipped plan returns the labels as they are.

## FlippingBoxesTwiceRestoresThem

Two boxes come back after two flips within `K_BOX_TOLERANCE`.

## FlippedPolygonMirrorsEveryVertexAboutTheWidth

In a 640 px wide image the vertices at x = 100, 300 and 150.5 move to 540, 340 and 489.5, and
every y is kept. An object without a polygon still has none.

## UnflippedPolygonIsReturnedUnchanged

An unflipped plan returns the polygon's vertices as they are.

## ValidateAcceptsZeroAndOne

`validate_flip` does not throw for probability 0, 1 or the default.
