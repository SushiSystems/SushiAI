# tests/vision/imaging/test_mosaic.cpp


Suites: `Unit_ImagingMosaic`.

Tests the draw, the plan and the application of the four-image mosaic.

This file needs no device. `Unit_ImagingMosaic` covers `modules/vision/imaging/include/SushiAI/vision/imaging/mosaic.hpp`.
`draw_mosaic` asks a `PartnerSource` for three partners, draws the gate and then the centre.
`plan_mosaic` turns a centre and four patch sizes into four placements on a canvas of twice the
image size. `apply_mosaic` copies the patches onto the canvas, or shifts, joins, clips and
prunes their labels. The two placement tables are lines 525 to 538 of the reference's
`data/augment.py`, worked by hand.

## K_SIDE and K_BOX_TOLERANCE

`K_SIDE` is 8, the image size of the hand cases, so the canvas is 16 by 16. `K_BOX_TOLERANCE`
is 1e-4 px.

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

`expect_box` and `expect_point` compare a box and a vertex within `K_BOX_TOLERANCE`.

## RecordingPartners

`RecordingPartners` is a `PartnerSource` that returns the partners 4, 5 and 6, makes no draw
and records every index it is asked for.

## pixel_at

`pixel_at` returns the three bytes of one canvas pixel.

## SettingsDefaultToTheReferenceProbabilityAndFill

Default `MosaicSettings` have probability 1 and fill value 114, and `K_MOSAIC_PATCHES` is 4.

## DrawMakesPartnersThenGateThenRowThenColumn

With a `DatasetPartners` of 1000 and image size 640, the draw equals a replay on a copied
generator: three `index(1000)` draws, one `bernoulli(1)`, then `uniform(320, 960)` truncated
for the centre row and again for the centre column. The generators are equal afterwards, so
the call makes six draws.

## DisabledDrawMakesFourDrawsAndLeavesTheFieldsZero

With probability 0 the draw is disabled, its partners and centre are 0, and the generator has
advanced by four draws: the three partners and the gate.

## CentreStaysInTheReferenceRangeOverManySeeds

Over 5000 seeds at image size 640 both centre coordinates lie in 320 to 959. Their mean is
639.5 within 10, the smallest is at most 330 and the largest at least 949.

## CentreOfAnOddSizeUsesTheFlooredBorder

At image size 5 the centre lies in 3 to 6 and all four values occur over 400 seeds. The
reference computes the border as -5 // 2, which is -3, so the range is `int(uniform(3, 7))`.

## PartnersCoverThePoolAndNeverLeaveIt

Over 300 seeds and a pool of 7, every partner is below 7 and every index occurs.

## GateOpensWithItsProbability

At probability 0.25 the draw is enabled for 0.25 of 4000 seeds, within 0.04.

## PlanCutsPatchesAtTheLeftAndBottomEdges

`plan_mosaic` for the low centre gives `K_LOW_PLACEMENTS` and records the image size.

## PlanCutsPatchesAtTheTopAndRightEdges

`plan_mosaic` for the high centre gives `K_HIGH_PLACEMENTS`.

## PlanCarriesTheFillValueOfTheSettings

The plan's `fill_value` is 9 when the settings say 9 and 114 by default.

## CanvasIsTwiceTheSizeAndFilledWhereNoPatchLands

The canvas is 16 by 16 with three channels. Its two outer corners, a pixel in the row above the
shorter patch 1 and a pixel in the column left of the narrower patch 2 hold the fill value.

## EachColourLandsInItsQuadrantAroundTheCentre

Four patches of one colour each are placed around the low centre. The four pixels that meet at
the centre are red, green, blue and yellow, in patch order, and `count_wrong_bytes` is 0.

## CutPatchesKeepThePixelsNearestTheCentre

Four textured patches around the high centre with fill value 3 give a canvas with no wrong
byte. The texture makes the test fail for a patch copied from its far corner.

## ApplyOverwritesWhatTheDestinationHeld

A destination that was 40 by 40 and filled with 200 becomes the 16 by 16 canvas with no wrong
byte.

## LabelsAreShiftedJoinedInPatchOrderClippedAndPruned

Five boxes in patches 0, 1 and 3 are shifted by their placements' offsets and joined in patch
order. The second box of patch 0 lies wholly left of the canvas, has no area after the clip and
is removed. The four that remain have the classes 3, 2, 7, 5 and the boxes the case lists.

## LabelsOfFourBackgroundPatchesAreEmpty

Four empty label sets join to an empty one.

## DrawRejectsANonPositiveSizeBeforeAnyDraw

Image sizes of 0 and -4 throw with "validate_mosaic_size" and the value, and the generator is
unchanged.

## PlanRejectsADisabledDrawAndAnEmptyPatch

`plan_mosaic` throws for a disabled draw ("not enabled"), for a patch of height 0 ("patch 2")
and for a centre column of 17 on a 16 px canvas ("centre").

## ApplyRejectsPatchesThePlanWasNotMadeFor

`apply_mosaic` throws for a one-channel patch among three-channel ones ("patch 1", "channels")
and for a patch of another size than the plan's ("patch 3").

## DrawnCentreIsWhereTheFourColoursMeet

Over 32 seeds the case draws, plans and applies a mosaic of four 8 by 8 patches of one colour
each. The centre lies in 4 to 11, and the four pixels around it are red, green, blue and
yellow. The draw and the plan therefore agree on which coordinate is the row.

## DrawRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5, -0.25 and NaN throw with "validate_mosaic" and "probability", the
default settings pass `validate_mosaic`, and the generator is unchanged.

## DrawRejectsASizeWhoseCanvasIsTooLarge

An image size of 8193, whose canvas would exceed `K_MAX_IMAGE_SIDE`, throws with "image_size of
8193" and "[1, 8192]". An image size of 8192 is accepted.

## DrawAsksThePartnerSourceForEverySample

With a `RecordingPartners`, one draw with the gate open and one with it closed record the
indices 17 and 23. The open draw holds the partners 4, 5, 6 and the closed one 0, 0, 0.

## PolygonsAreShiftedClampedAndPrunedWithTheirObjects

Polygons move with their patch, by (-3, 3) for patch 0 and (5, 11) for patch 3, and are clamped
to the 16 by 16 canvas. The object whose box leaves the canvas is removed with its polygon, and
the object without a polygon still has none.

## PlanRejectsASizeOutOfRangeAndAPatchSideAboveTheLimit

`plan_mosaic` throws for an image size of 0 and of 8193 ("image_size") and for a patch height
of 16385 ("patch 1").

## ApplyRejectsAPlanWithoutAValidSizeForPixelsAndForLabels

Both forms of `apply_mosaic` throw with "image_size" for a default plan and for one whose image
size is 8193.

## ApplyRejectsAPatchWithoutPixels

A default view as patch 2 throws with "apply_mosaic: patch 2" and "no pixels".

## ApplyRejectsAPatchThatLiesAnywhereInTheDestination

A patch that views the start of the destination, and one that views its row 20, throw with
"destination holds the source's pixels" under the patch's number. The destination keeps its
40 by 40 size and its bytes.

## ApplyRejectsAPlacementOutsideTheCanvasAndOneThatResamples

A placement whose `canvas_x2` is 17 throws with "patch 1" and "outside a canvas". A placement
whose patch span is one pixel narrower than its canvas span throws with "patch 3" and
"resamples nothing".
