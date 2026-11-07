# tests/vision/imaging/test_mosaic.cpp


Suites: `Small_ImagingMosaic`, `Medium_ImagingMosaic` and the parameterised `Small_ImagingMosaicPlacement` and `Small_ImagingMosaicColours`.

Tests the draw, the plan and the application of the four-image mosaic.

This file needs no device. The `Medium_` suite holds the death tests, since each starts a child process. The `Small_` suites cover `modules/vision/imaging/include/SushiAI/vision/imaging/mosaic.hpp`.
`draw_mosaic` asks a `PartnerSource` for three partners, draws the gate and then the centre.
`plan_mosaic` turns a centre and four patch sizes into four placements on a canvas of twice the
image size. `apply_mosaic` copies the patches onto the canvas, or shifts, joins, clips and
prunes their labels. The two placement tables are lines 525 to 538 of the reference's
`data/augment.py`, worked by hand.

## K_SIDE and the statistical bound

`K_SIDE` is 8, the image size of the hand cases, so the canvas is 16 by 16. `K_FALSE_FAILURE_RATE` is 1e-9, and `mean_tolerance(range, n)` is range * sqrt(ln(2 / rate) / (2 n)), the Hoeffding distance of a mean of n draws within a range. Box and vertex values are small integers, so they are compared exactly.

## PatchSizes and Placements

`PatchSizes` holds four `ImageSize` values and `Placements` holds four `MosaicPlacement`
values, patch 0 first.

## The low centre

`K_LOW_CENTER_X` and `K_LOW_CENTER_Y` are 5 and 11, a centre that cuts patch 0 on the left and
patches 2 and 3 at the bottom. `K_LOW_SIZES` are the patch sizes used with it; patch 1 is
shorter and patch 2 narrower than the others. `K_LOW_PLACEMENTS` are the four expected
placements.

## The high centre

`K_HIGH_CENTER_X` and `K_HIGH_CENTER_Y` are 10 and 6, a centre that cuts patches 0 and 1 at the
top and patches 1 and 3 on the right. `K_HIGH_SIZES` are four 8 by 8 patches and
`K_HIGH_PLACEMENTS` are the four expected placements.

## make_draw

`make_draw` returns an enabled `MosaicDraw` with a given centre and the partners 0, 0, 0.

## make_constant and make_textured

`make_constant` returns a three-channel image of one colour. `make_textured` returns one whose
every byte depends on its row, column, channel and a salt, so a copy from the wrong position
shows.

## view_all

`view_all` returns the views of four images in patch order.

## count_wrong_bytes

`count_wrong_bytes` counts the canvas bytes that differ from what four placements name: the
patch byte at the placement's offset inside a placement, and the fill value elsewhere.

## expect_placements

`expect_placements` compares all ten fields of a plan's four placements with the expected ones.

## expect_box and expect_point

`expect_box` and `expect_point` compare a box and a vertex exactly.

## Draw summaries and rows

`measure_centres`, `collect_centres_of_size_five` and `count_open_gates` run `draw_mosaic` over a range of seeds and return what the cases compare. `PlacementRow` holds a hand-worked centre, its patch sizes and its placements. `MeetRow` holds a centre and patch sizes; `meet_rows` builds the fixed low centre and the centres of 32 draws. `make_colour_patches` and `make_dark_patches` return four constant-colour patches. `make_patch_labels` and `make_low_plan` build the inputs of the label cases. `skip_draws` advances a generator.

## pixel_at

`pixel_at` returns the three bytes of one canvas pixel.

## DrawMakesPartnersThenGateThenRowThenColumn

With a `DatasetPartners` of 1000 and image size 640, the draw equals a replay on a copied
generator: three `index(1000)` draws, one `bernoulli(1)`, then `uniform(320, 960)` truncated
for the centre row and again for the centre column. The generators are equal afterwards, so
the call makes six draws.

## DisabledDrawLeavesTheFieldsZero

With probability 0 the draw is disabled and its partners and centre are 0.

## DisabledDrawMakesFourDraws

With probability 0 the generator has advanced by four draws: the three partners and the gate.

## CentreStaysInTheReferenceRangeOverManySeeds

Over 5000 seeds at image size 640 both centre coordinates lie in 320 to 959. Their mean is
639.5 within `mean_tolerance(640, 10000)`, the smallest is at most 330 and the largest at least 949.

## CentreOfAnOddSizeUsesTheFlooredBorder

At image size 5 the centres of 400 seeds are exactly the values 3, 4, 5 and 6 on both axes. The
reference computes the border as -5 // 2, which is -3, so the range is `int(uniform(3, 7))`.

## GateOpensWithItsProbability

At probability 0.25 the draw is enabled for 0.25 of 4000 seeds, within `mean_tolerance(1, 4000)`.

## PlanCutsPatchesAtTheEdgesOfTheCanvas

Suite `Small_ImagingMosaicPlacement`. `plan_mosaic` for the low centre gives `K_LOW_PLACEMENTS` and for the high centre `K_HIGH_PLACEMENTS`, and records the image size.

## PlanCarriesTheFillValueOfTheSettings

The plan's `fill_value` is 9 when the settings say 9 and 114 by default.

## CanvasIsTwiceTheSizeOfTheTrainingSize

The canvas is 16 by 16 with three channels.

## UncoveredCanvasTakesTheFillValue

The canvas's two outer corners, a pixel in the row above the shorter patch 1 and a pixel in the column left of the narrower patch 2 hold the fill value.

## EachPatchLandsWhereItsPlacementPutsIt

Four patches of one colour each around the low centre give a canvas on which `count_wrong_bytes` is 0.

## EachColourLandsInItsQuadrantAroundTheCentre

Suite `Small_ImagingMosaicColours`. For the fixed low centre and for the centres of 32 draws at size 8, four patches of one colour each are planned and applied. The centre lies in 4 to 11, and the four pixels around it are red, green, blue and yellow, in patch order. A draw and a plan therefore agree on which coordinate is the row.

## CutPatchesKeepThePixelsNearestTheCentre

Four textured patches around the high centre with fill value 3 give a canvas with no wrong
byte. The texture makes the test fail for a patch copied from its far corner.

## ApplyOverwritesWhatTheDestinationHeld

A destination that was 40 by 40 and filled with 200 becomes the 16 by 16 canvas with no wrong
byte.

## LabelsAreJoinedInPatchOrderAndPrunedWhenClippedToNothing

Five boxes in patches 0, 1 and 3 are joined in patch order. The second box of patch 0 lies wholly left of the canvas, has no area after the clip and is removed, so the classes are 3, 2, 7, 5.

## LabelBoxesAreShiftedByTheirPatchOffsetAndClippedToTheCanvas

The four boxes that remain are shifted by the offsets of their placements and clipped to the canvas, and equal the boxes the case lists.

## LabelsOfFourBackgroundPatchesAreEmpty

Four empty label sets join to an empty one.

## DrawRejectsANonPositiveSize

Image sizes of 0 and -4 fail with "validate_mosaic_size" and the value.

## DrawRefusedForItsSizeMakesNoDraw

A draw refused for its image size leaves the generator unchanged.

## PlanRejectsADisabledDraw

Suite `Medium_ImagingMosaic`. `plan_mosaic` stops the process for a disabled draw ("not enabled").

## PlanRejectsAnEmptyPatch

`plan_mosaic` fails for a patch of height 0 ("patch 2").

## PlanRejectsACentreOutsideTheCanvas

Suite `Medium_ImagingMosaic`. `plan_mosaic` stops the process for a centre column of 17 on a 16 px canvas ("centre").

## ApplyRejectsAPatchWithOtherChannels

`apply_mosaic` fails for a one-channel patch among three-channel ones ("patch 1", "channels").

## ApplyRejectsAPatchOfAnotherSizeThanThePlan

Suite `Medium_ImagingMosaic`. A patch of another size than the plan's stops the process ("patch 3").

## DrawRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5, -0.25 and NaN fail with "validate_mosaic" and "probability".

## ValidateRejectsAProbabilityOutsideZeroToOne

`validate_mosaic` fails for a probability of 1.5 and passes the default settings.

## DrawRefusedForItsProbabilityMakesNoDraw

A draw refused for its probability leaves the generator unchanged.

## DrawRejectsASizeWhoseCanvasIsTooLarge

An image size of 8193, whose canvas would exceed `K_MAX_IMAGE_SIDE`, returns `InvalidDescription` with "image_size of
8193" and "[1, 8192]".

## DrawAcceptsTheLargestSizeWhoseCanvasFits

An image size of 8192 is accepted.

## DrawAsksThePartnerSourceForEverySample

With an `InMemoryPartnerSource` scripted with the partners 4, 5, 6, one draw with the gate open and one with it closed record the
indices 17 and 23. The open draw holds the partners 4, 5, 6 and the closed one 0, 0, 0.

## PolygonsAreShiftedClampedAndPrunedWithTheirObjects

Polygons move with their patch, by (-3, 3) for patch 0 and (5, 11) for patch 3, and are clamped
to the 16 by 16 canvas. The object whose box leaves the canvas is removed with its polygon, and
the object without a polygon still has none.

## PlanRejectsASizeOutOfRange

`plan_mosaic` returns `InvalidDescription` for an image size of 0 and of 8193 ("image_size").

## PlanRejectsAPatchSideAboveTheLimit

`plan_mosaic` returns `InvalidDescription` for a patch height of 16385 ("patch 1").

## ApplyToPixelsRejectsAPlanWithoutAValidSize

Suite `Medium_ImagingMosaic`. The pixel form of `apply_mosaic` stops the process with "image_size" for a default plan and for one whose image size is 8193.

## ApplyToLabelsRejectsAPlanWithoutAValidSize

Suite `Medium_ImagingMosaic`. The label form of `apply_mosaic` stops the process with "image_size" for the same two plans.

## ApplyRejectsAPatchWithoutPixels

Suite `Medium_ImagingMosaic`. A default view as patch 2 stops the process with "apply_mosaic: patch 2" and "no pixels".

## ApplyRejectsAPatchThatLiesAnywhereInTheDestination

Suite `Medium_ImagingMosaic`. A patch that views the start of the destination, and one that views its row 20, stop the process with
"destination holds the source's pixels" under the patch's number.

## ApplyRejectsAPlacementOutsideTheCanvas

Suite `Medium_ImagingMosaic`. A placement whose `canvas_x2` is 17 stops the process with "patch 1" and "outside a canvas".

## ApplyRejectsAPlacementThatResamples

Suite `Medium_ImagingMosaic`. A placement whose patch span is one pixel narrower than its canvas span stops the process with "patch 3" and "resamples nothing".
