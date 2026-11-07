# tests/vision/imaging/test_affine.cpp


Suites: `Small_ImagingAffine`, `Small_ImagingAffinePolygonRatio`, `Medium_ImagingAffine`.

Tests the plan and the application of the scale and translation warp to an image and its
labels.

This file needs no device. `Small_ImagingAffine` covers `modules/vision/imaging/include/SushiAI/vision/imaging/affine.hpp`.
`plan_affine` draws a scale and a translation per axis and stores the 2 by 3 matrix;
`apply_affine` warps an image with bilinear weights, or warps, clips and filters a label set.
The matrix has no rotation and no shear, and `apply_affine` refuses a plan that has one.

## Tolerances and statistics

`K_PIXEL_TOLERANCE` is 32768 float unit roundoffs, about 2e-3 px, for a box edge or a matrix term
that float arithmetic rounds: the sum of three products or one interpolation of terms below 1500,
each rounding off by at most 9e-5. Every exact case compares for equality. `K_CROSSING_X` is
110 / 3, the x of the polygon crossing, 36.666664 in float. `K_DISTINCT_FRACTION_GAP` is 32 float
unit roundoffs.

`K_SAMPLE_COUNT` is 2048, the plans the statistical case draws from one generator seeded with
`K_STATISTICAL_SEED` (20261007). `mean_tolerance` derives the allowed distance of a sample mean by
Hoeffding's inequality, w sqrt(ln(2 / rate) / (2 n)) for draws in an interval of width w, with
`K_FALSE_FAILURE_RATE` 1e-9: 0.072 for the scale (w = 1) and 0.0145 for a fraction (w = 0.2).

## Matrix3 and multiply

`Matrix3` is a row-major 3 by 3 float matrix, the form the reference builds its factors in.
`multiply` returns the product of two of them with each term summed left to right.

## make_pattern_image

`make_pattern_image` returns an image whose byte at row y, column x, channel c is
(31 y + 7 x + 53 c + 1) modulo 256.

## make_shifted_image

`make_shifted_image` returns an image of a given size where the pixel at (y, x) is the source pixel
at (y - shift_y, x - shift_x) when that lies inside the source, and the border value otherwise.

## make_gray_image

`make_gray_image` returns a one-channel image holding the given bytes.

## make_plan

`make_plan` returns an `AffinePlan` with a given scale and matrix translation, built without a
draw. A case uses it to state the matrix it tests.

## make_identity_plan

`make_identity_plan` returns the identity plan of a 4 by 4 image.

## make_labels

`make_labels` returns a label set with one object per box and the classes 0, 1, 2 and so on.

## expect_box_near and expect_box

`expect_box_near` compares the four edges of two boxes within `K_PIXEL_TOLERANCE`, and `expect_box`
compares them exactly. `expect_vertex_inside` expects a vertex to lie inside the output and between
the box's left and right edges.

## bytes_of and RatioRow

`bytes_of` returns the bytes of an image. `RatioRow` holds the plain and the polygon area ratios of
a plan and whether the polygon object must survive.

## ZeroRangesPutTheSourceCentreOnTheOutputCentre

With the scale and translate ranges at 0, a 1200 by 1000 source and a 480 by 640 output give
scale 1, translation (320, 240) and the matrix [1 0 -180; 0 1 -360], compared exactly: the source
centre (500, 600) maps to the output centre.

## MatrixEqualsTranslationTimesScaleTimesCentre

From the scale and translation the plan records, the case builds the centring, scaling and
translation matrices. The plan's matrix equals translation times scaling times centring within
`K_PIXEL_TOLERANCE`. The seed is required to give a scale more than 0.05 from 1, because at scale 1
the product is the same in any order.

## PlanConsumesExactlyThreeDraws

After `plan_affine` the generator equals a copy advanced by three `next_u64` draws, written out
one by one.

## PlanCarriesSizesBorderValueAndFilter

The plan holds the source size, the output size, the border value 37 and the filter fields of
the settings, compared exactly.

## DrawnParametersStayInRangeAndCentreOnTheirMeans

Over 2048 plans from one seeded generator, the scale lies in [0.5, 1.5) and each translation in
[0.4, 0.6] of its output side. The means are 1 within 0.072 and 0.5 within 0.0145 (see
`mean_tolerance`), the extremes come within 0.05 and 0.01 of the range ends, and the x and y
fractions are within `K_DISTINCT_FRACTION_GAP` of each other in fewer than 1 percent of the plans,
so each axis has its own draw.

## PlanRejectsAScaleRangeOutsideItsBounds

A scale half-range of 1, -0.1 or NaN fails with "validate_affine" or "scale", from the plan, and
the half-range of 1 fails from `validate_affine` too.

## PlanRejectsATranslateRangeOutsideItsBounds

A translate half-range of 0.6 or -0.1 fails with "validate_affine" or "translate", from the plan,
and 0.6 fails from `validate_affine` too.

## PlanRejectsASizeThatIsNotPositive

A source of 0 by 16 and an output of 16 by 0 fail with "plan_affine" and the size.

## PlanRejectedForItsSettingsOrSizesMakesNoDraw

After four refused plans (scale range, translate range, source size, output size) the generator
equals a fresh one.

## ValidateAcceptsTheDefaultSettings

`validate_affine` accepts the default settings.

## IdentityPlanLeavesTheImageUnchanged

With both ranges at 0 and equal source and output sizes, the warped image has the source's
bytes.

## IdentityPlanLeavesBoxesUnchanged

Under the same plan three boxes keep their classes and their edges, exactly.

## TranslationMovesPixelsAndFillsTheUncoveredBorder

A 3 by 4 source shifted by (2, 1) into a 5 by 6 output puts source pixel (y - 1, x - 2) at
(y, x) and fills every other pixel with 114, as `make_shifted_image` builds it.

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

## ImageWarpStopsOnAPlanOfAnotherSourceSize

`apply_affine` on an image stops the process for a plan made for a 4 by 5 source. The tests of
this group are in `Medium_ImagingAffine` because a death test starts a child process.

## ImageWarpStopsOnAShear

A matrix with an off-diagonal term stops the process with "off-diagonal".

## ImageWarpStopsOnAZeroDiagonal

A diagonal term of 0 stops the process with "above zero".

## ImageWarpStopsOnAScaleThatIsNotThePlans

A matrix scale that is not the plan's scale stops the process with "plan's scale".

## ImageWarpStopsOnAnInfiniteTranslation

An infinite translation stops the process with "finite translation".

## ImageWarpStopsOnASourceWithoutPixels

A default view stops the process with "pixels".

## ImageWarpStopsOnASourceOfFiveChannels

A source of five channels stops the process with "channel count".

## ImageWarpStopsOnASourceTheDestinationHolds

The destination as its own source and a view of two rows of the destination stop the process with
"destination holds the source's pixels".

## TranslationMovesBoxesByTheOffset

A translation of (30, -10) maps (10, 20, 50, 60) to (40, 10, 80, 50).

## ScaleMultipliesBoxCornersBeforeTheOffset

Scale 2 with translation (10, -20) maps (5, 20, 25, 40) to (20, 20, 60, 60).

## WarpedBoxIsTheHullOfItsFourWarpedCorners

The corners (100, 20), (200, 20), (100, 120) and (200, 120) of a box map through scale 0.5 and
translation (-20, 30) to (30, 40), (80, 40), (30, 90) and (80, 90), and the warped box is their
hull (30, 40, 80, 90).

## BoxesAreClippedToTheOutput

Five 40 by 40 boxes are shifted right by 80 in a 100 px wide output, which leaves widths of 10,
5, 3, 2 and 0 inside it. The first two remain, clipped to x2 = 100.

## ClippedBoxesBelowTheFilterAreDropped

The same five boxes leave two: width 3 fails the area ratio at 0.075, width 2 fails the minimum
side and width 0 fails both.

## SurvivingBoxesKeepTheirOrder

The survivors of the same five boxes have the classes 0 and 1, in that order.

## FilterComparesAgainstTheSourceBoxTimesTheScale

At scale 0.25 a 40 by 40 box becomes 10 by 10 and is kept: its ratio is 1 against the scaled
box and 0.0625 against the unscaled one. At scale 2 a 20 by 20 box that keeps 3 by 40 inside
the output is removed: its ratio is 0.075 against the scaled box and 0.3 against the unscaled
one.

## BoxWarpUsesThePlansFilter

A 4 px wide box is kept with the default filter and removed when the plan's `min_side` is 4.

## BoxWarpStopsOnARotatedMatrix

`apply_affine` on labels stops the process with "off-diagonal" for an off-diagonal matrix term.
The box warp death tests are in `Medium_ImagingAffine`.

## BoxWarpStopsOnAScaleThatIsNotThePlans

A matrix scale that is not the plan's scale stops the process with "plan's scale".

## PolygonVerticesAreScaledThenTranslated

Scale 2 with translation (10, -5) maps the vertices (5, 10), (25, 12) and (15, 30) to (20, 15),
(60, 19) and (40, 55), and the box to (20, 15, 60, 55), all exactly.

## PolygonObjectTakesTheBoxOfItsVisiblePart

A triangle that reaches outside a 100 by 80 output gets the box (0, 0, `K_CROSSING_X`, 33) within
`K_PIXEL_TOLERANCE`, the reference's `segment2box` result. The same bounds without a polygon are clipped to
(0, 0, 70, 60).

## WarpedPolygonIsClampedToItsNewBox

The same triangle's vertices become (0, 0), (`K_CROSSING_X`, 0) and (0, 33): each is clamped to the
box rebuilt from the polygon.

## WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside

In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at -1.86e-9. The box's x1 is exactly
0, and every vertex lies inside the output and between the box's x1 and x2.

## PolygonObjectIsFilteredAtThePolygonAreaRatio

A box that keeps an area ratio of 0.075 is kept when the plan's filters are
0.10 for a box and 0.01 for a polygon, and removed when they are 0 and 0.08. The polygon filter
of the plan decides for an object with a polygon.

## BoxWithoutPolygonIsFilteredAtTheBoxAreaRatio

The same box without a polygon is removed at the default box ratio of 0.10.

## PolygonOutsideTheOutputDropsItsObject

With a translation of 500, the object whose polygon leaves the output is removed and the one
the translation moves into it is kept, with class 1 and the box (40, 10, 80, 50).
