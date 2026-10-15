# tests/vision/imaging/test_affine.cpp


Suites: `Unit_ImagingAffine`.

Tests the plan and the application of the scale and translation warp to an image and its
labels.

This file needs no device. `Unit_ImagingAffine` covers `modules/vision/imaging/include/SushiAI/vision/imaging/affine.hpp`.
`plan_affine` draws a scale and a translation per axis and stores the 2 by 3 matrix;
`apply_affine` warps an image with bilinear weights, or warps, clips and filters a label set.
The matrix has no rotation and no shear, and `apply_affine` refuses a plan that has one.

## K_PIXEL_TOLERANCE

`K_PIXEL_TOLERANCE` is 1e-3 px, the tolerance of a box edge or a matrix term.

## Matrix3 and multiply

`Matrix3` is a row-major 3 by 3 float matrix, the form the reference builds its factors in.
`multiply` returns the product of two of them with each term summed left to right.

## make_pattern_image

`make_pattern_image` returns an image whose byte at row y, column x, channel c is
(31 y + 7 x + 53 c + 1) modulo 256.

## make_gray_image

`make_gray_image` returns a one-channel image holding the given bytes.

## make_plan

`make_plan` returns an `AffinePlan` with a given scale and matrix translation, built without a
draw. A case uses it to state the matrix it tests.

## make_labels

`make_labels` returns a label set with one object per box and the classes 0, 1, 2 and so on.

## expect_box_near

`expect_box_near` compares the four edges of two boxes within `K_PIXEL_TOLERANCE`.

## bytes_of

`bytes_of` returns the bytes of an image.

## ZeroRangesPutTheSourceCentreOnTheOutputCentre

With the scale and translate ranges at 0, a 1200 by 1000 source and a 480 by 640 output give
scale 1, translation (320, 240) and the matrix [1 0 -180; 0 1 -360]: the source centre
(500, 600) maps to the output centre.

## MatrixEqualsTranslationTimesScaleTimesCentre

The case replays the three draws on a copy of the generator and builds the centring, scaling
and translation matrices. The plan's matrix equals translation times scaling times centring.
The seed is required to give a scale more than 0.05 from 1, because at scale 1 the product is
the same in any order.

## PlanConsumesExactlyThreeDraws

After `plan_affine` the generator equals a copy advanced by three draws.

## PlanCarriesSizesBorderValueAndFilter

The plan holds the source size, the output size, the border value 37 and the filter fields of
the settings.

## DrawnParametersStayInRangeAndCentreOnTheirMeans

Over 20000 plans from 200 seeds, the scale lies in [0.5, 1.5) and each translation in [0.4,
0.6] of its output side. The means are 1 within 0.012 and 0.5 within 0.0025, the extremes come
within 0.01 and 0.002 of the range ends, and the x and y fractions are within 1e-6 of each
other in fewer than 1 percent of the plans, so each axis has its own draw.

## PlanRejectsSettingsAndSizesOutsideTheirRanges

A scale half-range of 1, -0.1 or NaN and a translate half-range of 0.6 or -0.1 throw with
"validate_affine" or the field name. A source of 0 by 16 and an output of 16 by 0 throw with
"plan_affine" and the size. No rejected call makes a draw, and the default settings pass.

## IdentityPlanLeavesTheImageUnchanged

With both ranges at 0 and equal source and output sizes, the warped image has the source's
bytes.

## IdentityPlanLeavesBoxesUnchanged

Under the same plan three boxes keep their classes and their edges.

## TranslationMovesPixelsAndFillsTheUncoveredBorder

A 3 by 4 source shifted by (2, 1) into a 5 by 6 output puts source pixel (y - 1, x - 2) at
(y, x) and fills every other pixel with 114.

## ScaleByTwoInterpolatesAndFadesIntoTheBorder

A 2 by 2 image scaled by 2 with border value 100 gives the sixteen bytes the case lists. Odd
output coordinates fall halfway between two source pixels, and the last row and column blend
with the border.

## HalfPixelBlendRoundsToNearest

A row (10, 13) shifted by half a pixel gives 62, 12 and 64: the means 62, 11.5 and 63.5 with
the border 114, rounded half up.

## FractionRoundsToTheNearestOf2048Weights

With an offset of 6141 / 8192 the fraction times 2048 is 512.75, so the weights are 1535 and
513. The three output bytes are 90, 19 and 42, computed from those weights; the weights 1536
and 512 give other bytes.

## OutputFarFromTheSourceIsAllBorder

A translation of (1000, -1000) leaves every output byte at the border value 9.

## ImageWarpRejectsAPlanItCannotApply

`apply_affine` on an image throws for a plan made for another source size, a matrix with an
off-diagonal term, a diagonal term of 0, a matrix scale that is not the plan's scale, an
infinite translation, a source without pixels, a source of five channels, the destination as
its own source and a view of two rows of the destination. The aliased image keeps its bytes.

## TranslationMovesBoxesByTheOffset

A translation of (30, -10) maps (10, 20, 50, 60) to (40, 10, 80, 50).

## ScaleMultipliesBoxCornersBeforeTheOffset

Scale 2 with translation (10, -20) maps (5, 20, 25, 40) to (20, 20, 60, 60).

## WarpedBoxIsTheHullOfItsFourWarpedCorners

The case maps the four corners of a box through the plan's matrix and takes their hull. The
warped box equals the hull and the literal (30, 40, 80, 90).

## BoxesAreClippedThenFilteredAndSurvivorsKeepTheirOrder

Five 40 by 40 boxes are shifted right by 80 in a 100 px wide output, which leaves widths of 10,
5, 3, 2 and 0 inside it. The first two remain, in order, clipped to x2 = 100. Width 3 fails the
area ratio at 0.075, width 2 fails the minimum side and width 0 fails both.

## FilterComparesAgainstTheSourceBoxTimesTheScale

At scale 0.25 a 40 by 40 box becomes 10 by 10 and is kept: its ratio is 1 against the scaled
box and 0.0625 against the unscaled one. At scale 2 a 20 by 20 box that keeps 3 by 40 inside
the output is removed: its ratio is 0.075 against the scaled box and 0.3 against the unscaled
one.

## BoxWarpUsesThePlansFilter

A 4 px wide box is kept with the default filter and removed when the plan's `min_side` is 4.

## BoxWarpRejectsARotatedMatrixAndAScaleThatIsNotThePlans

`apply_affine` on labels throws for an off-diagonal matrix term and for a matrix scale that is
not the plan's scale.

## FiltersDefaultToTheBoxAndThePolygonAreaRatios

The settings and a plan made from them have `filter.min_area_ratio` 0.10 and
`polygon_filter.min_area_ratio` 0.01, and a default `AffinePlan` has the second value too.

## PolygonVerticesAreScaledThenTranslated

Scale 2 with translation (10, -5) maps the vertices (5, 10), (25, 12) and (15, 30) to (20, 15),
(60, 19) and (40, 55), and the box to (20, 15, 60, 55).

## PolygonObjectTakesTheBoxOfItsVisiblePart

A triangle that reaches outside a 100 by 80 output gets the box (0, 0, 36.666664, 33), the
reference's `segment2box` result. The same bounds without a polygon are clipped to
(0, 0, 70, 60).

## WarpedPolygonIsClampedToItsNewBox

The same triangle's vertices become (0, 0), (36.666664, 0) and (0, 33): each is clamped to the
box rebuilt from the polygon.

## WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside

In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at -1.86e-9. The box's x1 is exactly
0, and every vertex lies inside the output and between the box's x1 and x2.

## PolygonObjectIsFilteredAtTheLowerAreaRatio

A box that keeps an area ratio of 0.075 is kept when its object has a polygon and removed when
it has none.

## PolygonWarpUsesThePlansPolygonFilter

The same object is removed when `polygon_filter.min_area_ratio` is 0.08, with
`filter.min_area_ratio` set to 0. The polygon filter decides for an object with a polygon.

## PolygonOutsideTheOutputDropsItsObject

With a translation of 500, the object whose polygon leaves the output is removed and the one
the translation moves into it is kept, with class 1 and the box (40, 10, 80, 50).
