# tests/vision/imaging/test_flip.cpp


Suites: `Small_ImagingFlip`, `Small_ImagingFlipDecision`, `Medium_ImagingFlip`.

Tests the plan and the application of the horizontal flip to an image and its labels.

This file needs no device. `Small_ImagingFlip` covers `modules/vision/imaging/include/SushiAI/vision/imaging/flip.hpp`.
`plan_flip` makes the one random draw and records the decision in a `FlipPlan`; `apply_flip`
then mirrors an image or a label set from the plan and draws nothing.

## Constants and tolerance

`K_SAMPLE_COUNT` is 1024, the plans a statistical case draws from one generator seeded with
`K_STATISTICAL_SEED` (20261007). `K_FALSE_FAILURE_RATE` is 1e-9. `frequency_tolerance` derives the
allowed distance of a flip frequency from 0.5 by Hoeffding's inequality, sqrt(ln(2 / rate) / (2 n)),
which is about 0.102 for 1024 draws. Box and polygon coordinates are dyadic, so those cases compare
exactly.

## make_counted_image

`make_counted_image` returns an image whose byte at offset i is i modulo 256.

## expect_same_image

`expect_same_image` expects two images to have the same size, channel count and bytes.

## expect_bytes

`expect_bytes` expects an image to hold exactly the listed bytes, in order.

## expect_two_flips_restore

`expect_two_flips_restore` flips a source twice with a flipped plan and expects the source back.

## expect_box

`expect_box` compares the four coordinates of two boxes exactly.

## DecisionRow

`DecisionRow` holds a probability, the seed of its draw and the decision the plan must give.

## PlanRecordsTheSourceSize

The plan holds the source size 480 by 640.

## PlanMakesExactlyOneDraw

The plan's `is_flipped` equals one `bernoulli` draw of a second generator with the same seed, and
the two generators are equal afterwards.

## ProbabilityZeroNeverFlipsAndOneAlwaysFlips

In rows of probability 0 and 1 over three seeds each, probability 0 never flips and probability 1
always does, and each plan leaves the generator after one `bernoulli` draw.

## DefaultProbabilityFlipsAboutHalfOfTheDraws

With the default settings the flip frequency over 1024 plans from one seeded generator is 0.5
within `frequency_tolerance`.

## PlanRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5 and -0.1 fail with a message containing "validate_flip" and
"probability".

## PlanRejectedForItsProbabilityMakesNoDraw

After the plans refused for 1.5 and -0.1, the generator equals a fresh one.

## PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw

A NaN probability returns `InvalidDescription` and the generator is unchanged.

## PlanRejectsASideThatIsNotPositiveBeforeAnyDraw

Sizes of 0 by 8 and 8 by -2 fail with a message containing "plan_flip", "source size" and the
size, and the generator is unchanged.

## SourceWithoutPixelsIsRejected

`apply_flip` on a default `ImageView` stops the process with "apply_flip" and "no pixels". The
test is in `Medium_ImagingFlip` because a death test starts a child process.

## SourceHeldByTheDestinationIsRejected

`apply_flip` with the image as both source and destination, and with a view of three rows of the
destination as the source, stops the process with "destination holds the source's pixels".

## FlippedImageHoldsEachRowInReverseColumnOrder

For a 2 by 3 image of three channels, the destination bytes are 6, 7, 8, 3, 4, 5, 0, 1, 2 in row
0 and 15, 16, 17, 12, 13, 14, 9, 10, 11 in row 1: the pixels are reversed and the channels
inside a pixel are not.

## UnflippedImageIsCopiedUnchanged

A plan with `is_flipped` false copies the source.

## FlippingAnImageTwiceRestoresIt

A 4 by 7 image of three channels and a 3 by 4 image of one channel both come back after two
flips.

## ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 3 source with a plan for 2 by 4 or for 5 by 3 stops the process with a message naming the plan's
size and "2x3". The test is in `Medium_ImagingFlip`.

## FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder

In a 640 px wide image the box (100, 50, 300, 200) becomes (340, 50, 540, 200), and a box
spanning the whole width is unchanged. Classes and order are kept, and x1 stays at or below x2.

## UnflippedBoxesAreReturnedUnchanged

An unflipped plan returns the labels as they are.

## FlippingBoxesTwiceRestoresThem

Two boxes come back exactly after two flips.

## FlippedPolygonMirrorsEveryVertexAboutTheWidth

In a 640 px wide image the vertices at x = 100, 300 and 150.5 move to 540, 340 and 489.5, and
every y is kept. An object without a polygon still has none.

## UnflippedPolygonIsReturnedUnchanged

An unflipped plan returns the polygon's vertices as they are.

## ValidateAcceptsZeroAndOne

`validate_flip` does not throw for probability 0, 1 or the default.
