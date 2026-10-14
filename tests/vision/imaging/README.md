# tests/vision/imaging

The tests of the module `vision/imaging`, one file per header of `modules/vision/imaging/include/`. The suite prefix of a case gives its ctest label.

## test_affine.cpp

Suites: `Unit_ImagingAffine`.

Tests the plan and the application of the scale and translation warp to an image and its
labels.

This file needs no device. `Unit_ImagingAffine` covers `modules/vision/imaging/include/SushiAI/vision/imaging/affine.hpp`.
`plan_affine` draws a scale and a translation per axis and stores the 2 by 3 matrix;
`apply_affine` warps an image with bilinear weights, or warps, clips and filters a label set.
The matrix has no rotation and no shear, and `apply_affine` refuses a plan that has one.

### K_PIXEL_TOLERANCE

`K_PIXEL_TOLERANCE` is 1e-3 px, the tolerance of a box edge or a matrix term.

### Matrix3 and multiply

`Matrix3` is a row-major 3 by 3 float matrix, the form the reference builds its factors in.
`multiply` returns the product of two of them with each term summed left to right.

### make_pattern_image

`make_pattern_image` returns an image whose byte at row y, column x, channel c is
(31 y + 7 x + 53 c + 1) modulo 256.

### make_gray_image

`make_gray_image` returns a one-channel image holding the given bytes.

### make_plan

`make_plan` returns an `AffinePlan` with a given scale and matrix translation, built without a
draw. A case uses it to state the matrix it tests.

### make_labels

`make_labels` returns a label set with one object per box and the classes 0, 1, 2 and so on.

### expect_box_near

`expect_box_near` compares the four edges of two boxes within `K_PIXEL_TOLERANCE`.

### bytes_of

`bytes_of` returns the bytes of an image.

### ZeroRangesPutTheSourceCentreOnTheOutputCentre

With the scale and translate ranges at 0, a 1200 by 1000 source and a 480 by 640 output give
scale 1, translation (320, 240) and the matrix [1 0 -180; 0 1 -360]: the source centre
(500, 600) maps to the output centre.

### MatrixEqualsTranslationTimesScaleTimesCentre

The case replays the three draws on a copy of the generator and builds the centring, scaling
and translation matrices. The plan's matrix equals translation times scaling times centring.
The seed is required to give a scale more than 0.05 from 1, because at scale 1 the product is
the same in any order.

### PlanConsumesExactlyThreeDraws

After `plan_affine` the generator equals a copy advanced by three draws.

### PlanCarriesSizesBorderValueAndFilter

The plan holds the source size, the output size, the border value 37 and the filter fields of
the settings.

### DrawnParametersStayInRangeAndCentreOnTheirMeans

Over 20000 plans from 200 seeds, the scale lies in [0.5, 1.5) and each translation in [0.4,
0.6] of its output side. The means are 1 within 0.012 and 0.5 within 0.0025, the extremes come
within 0.01 and 0.002 of the range ends, and the x and y fractions are within 1e-6 of each
other in fewer than 1 percent of the plans, so each axis has its own draw.

### PlanRejectsSettingsAndSizesOutsideTheirRanges

A scale half-range of 1, -0.1 or NaN and a translate half-range of 0.6 or -0.1 throw with
"validate_affine" or the field name. A source of 0 by 16 and an output of 16 by 0 throw with
"plan_affine" and the size. No rejected call makes a draw, and the default settings pass.

### IdentityPlanLeavesTheImageUnchanged

With both ranges at 0 and equal source and output sizes, the warped image has the source's
bytes.

### IdentityPlanLeavesBoxesUnchanged

Under the same plan three boxes keep their classes and their edges.

### TranslationMovesPixelsAndFillsTheUncoveredBorder

A 3 by 4 source shifted by (2, 1) into a 5 by 6 output puts source pixel (y - 1, x - 2) at
(y, x) and fills every other pixel with 114.

### ScaleByTwoInterpolatesAndFadesIntoTheBorder

A 2 by 2 image scaled by 2 with border value 100 gives the sixteen bytes the case lists. Odd
output coordinates fall halfway between two source pixels, and the last row and column blend
with the border.

### HalfPixelBlendRoundsToNearest

A row (10, 13) shifted by half a pixel gives 62, 12 and 64: the means 62, 11.5 and 63.5 with
the border 114, rounded half up.

### FractionRoundsToTheNearestOf2048Weights

With an offset of 6141 / 8192 the fraction times 2048 is 512.75, so the weights are 1535 and
513. The three output bytes are 90, 19 and 42, computed from those weights; the weights 1536
and 512 give other bytes.

### OutputFarFromTheSourceIsAllBorder

A translation of (1000, -1000) leaves every output byte at the border value 9.

### ImageWarpRejectsAPlanItCannotApply

`apply_affine` on an image throws for a plan made for another source size, a matrix with an
off-diagonal term, a diagonal term of 0, a matrix scale that is not the plan's scale, an
infinite translation, a source without pixels, a source of five channels, the destination as
its own source and a view of two rows of the destination. The aliased image keeps its bytes.

### TranslationMovesBoxesByTheOffset

A translation of (30, -10) maps (10, 20, 50, 60) to (40, 10, 80, 50).

### ScaleMultipliesBoxCornersBeforeTheOffset

Scale 2 with translation (10, -20) maps (5, 20, 25, 40) to (20, 20, 60, 60).

### WarpedBoxIsTheHullOfItsFourWarpedCorners

The case maps the four corners of a box through the plan's matrix and takes their hull. The
warped box equals the hull and the literal (30, 40, 80, 90).

### BoxesAreClippedThenFilteredAndSurvivorsKeepTheirOrder

Five 40 by 40 boxes are shifted right by 80 in a 100 px wide output, which leaves widths of 10,
5, 3, 2 and 0 inside it. The first two remain, in order, clipped to x2 = 100. Width 3 fails the
area ratio at 0.075, width 2 fails the minimum side and width 0 fails both.

### FilterComparesAgainstTheSourceBoxTimesTheScale

At scale 0.25 a 40 by 40 box becomes 10 by 10 and is kept: its ratio is 1 against the scaled
box and 0.0625 against the unscaled one. At scale 2 a 20 by 20 box that keeps 3 by 40 inside
the output is removed: its ratio is 0.075 against the scaled box and 0.3 against the unscaled
one.

### BoxWarpUsesThePlansFilter

A 4 px wide box is kept with the default filter and removed when the plan's `min_side` is 4.

### BoxWarpRejectsARotatedMatrixAndAScaleThatIsNotThePlans

`apply_affine` on labels throws for an off-diagonal matrix term and for a matrix scale that is
not the plan's scale.

### FiltersDefaultToTheBoxAndThePolygonAreaRatios

The settings and a plan made from them have `filter.min_area_ratio` 0.10 and
`polygon_filter.min_area_ratio` 0.01, and a default `AffinePlan` has the second value too.

### PolygonVerticesAreScaledThenTranslated

Scale 2 with translation (10, -5) maps the vertices (5, 10), (25, 12) and (15, 30) to (20, 15),
(60, 19) and (40, 55), and the box to (20, 15, 60, 55).

### PolygonObjectTakesTheBoxOfItsVisiblePart

A triangle that reaches outside a 100 by 80 output gets the box (0, 0, 36.666664, 33), the
reference's `segment2box` result. The same bounds without a polygon are clipped to
(0, 0, 70, 60).

### WarpedPolygonIsClampedToItsNewBox

The same triangle's vertices become (0, 0), (36.666664, 0) and (0, 33): each is clamped to the
box rebuilt from the polygon.

### WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside

In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at -1.86e-9. The box's x1 is exactly
0, and every vertex lies inside the output and between the box's x1 and x2.

### PolygonObjectIsFilteredAtTheLowerAreaRatio

A box that keeps an area ratio of 0.075 is kept when its object has a polygon and removed when
it has none.

### PolygonWarpUsesThePlansPolygonFilter

The same object is removed when `polygon_filter.min_area_ratio` is 0.08, with
`filter.min_area_ratio` set to 0. The polygon filter decides for an object with a polygon.

### PolygonOutsideTheOutputDropsItsObject

With a translation of 500, the object whose polygon leaves the output is removed and the one
the translation moves into it is kept, with class 1 and the box (40, 10, 80, 50).

## test_boxes.cpp

Suites: `Unit_ImagingBoxes`.

Tests the two box forms, their conversions and the candidate filter.

This file needs no device. `Unit_ImagingBoxes` covers `modules/vision/imaging/include/SushiAI/vision/imaging/boxes.hpp`: a
`NormalizedBox` is a centre and an extent in fractions of the image, a `PixelBox` is two corners
in pixels, and `is_box_candidate` decides whether a box survives an augmentation.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px and `K_FRACTION_TOLERANCE` is 1e-6 of the image.
`K_EPSILON_TOLERANCE` is 1e-22, for the filter's default epsilon of 1e-16. `K_SIZE` is 480 rows
by 640 columns, the size the hand cases use.

### expect_pixel_box

`expect_pixel_box` compares the four corners of a `PixelBox` within `K_PIXEL_TOLERANCE`.

### expect_normalized_box

`expect_normalized_box` compares the four values of a `NormalizedBox` within
`K_FRACTION_TOLERANCE`.

### box_of_sides

`box_of_sides` returns a box with its left top corner at the origin and the given width and
height.

### ToPixelBoxPlacesCornersFromCentreAndExtent

The normalized box (0.5, 0.25, 0.5, 0.25) in `K_SIZE` is the pixel box (160, 60, 480, 180).

### ToPixelBoxUsesWidthForXAndHeightForY

A box covering a 100 by 200 image is (0, 0, 200, 100). The two sides differ, so a conversion
that swapped them gives other corners.

### ToPixelBoxDoesNotClip

A box centred on the left edge converts to (-50, 75, 50, 125). The conversion leaves the
negative corner as it is.

### ToNormalizedBoxDividesCentreAndExtentByTheImageSize

The pixel box (160, 60, 480, 180) in `K_SIZE` is the normalized box (0.5, 0.25, 0.5, 0.25).

### NormalizedToPixelAndBackReturnsTheBox

Four normalized boxes, one of them all zeros, come back from the two conversions within
`K_FRACTION_TOLERANCE`.

### PixelToNormalizedAndBackReturnsTheBox

The pixel box (12.5, 7.25, 630, 471.75) comes back within `K_PIXEL_TOLERANCE`.

### ScaleBoxMultipliesEachAxisByItsOwnFactor

`scale_box` by 2 across and 0.5 down maps (10, 20, 30, 40) to (20, 10, 60, 20).

### TranslateBoxAddsEachAxisItsOwnOffset

`translate_box` by 5 across and -25 down maps (10, 20, 30, 40) to (15, -5, 35, 15).

### ClipBoxClampsEachEdgeToTheImage

`clip_box` maps (-10, 20, 700, 500) in `K_SIZE` to (0, 20, 640, 480).

### ClipBoxLeavesABoxInsideTheImageAlone

The box (0, 0, 640, 480) is unchanged.

### ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge

A box to the right of the image clips to (640, 10, 640, 20) and one above it to
(10, 0, 20, 0). Both have area 0.

### BoxAreaIsWidthTimesHeight

`box_area` of (1, 2, 4, 6) is 12.

### PolygonFilterLowersOnlyTheAreaRatioToOneHundredth

`make_polygon_filter` returns a `CandidateFilter` whose `min_area_ratio` is 0.01, which is
`K_POLYGON_MIN_AREA_RATIO`. Its `min_side`, `max_aspect_ratio` and `epsilon` equal those of a
default filter.

### PolygonFilterKeepsABoxThePlainFilterRejects

A box that goes from 100 by 100 to 50 by 10 keeps 5 percent of its area. The default filter
rejects it and the polygon filter keeps it.

### CandidateFilterDefaultsAreTheReferenceThresholds

A default `CandidateFilter` has `min_side` 2, `max_aspect_ratio` 100, `min_area_ratio` 0.10 and
`epsilon` 1e-16.

### CandidateNeedsAWidthAboveTheMinimumSide

A width of 2 is rejected and a width of 2.25 is kept. The comparison is strict.

### CandidateNeedsAHeightAboveTheMinimumSide

A height of 2 is rejected and a height of 2.25 is kept.

### CandidateNeedsMoreThanTheMinimumAreaRatio

Against a 100 by 100 box, a 10 by 100 box has a ratio of 0.10 in float and is rejected. A width
of 10.5 is kept and a width of 9.5 is rejected.

### CandidateAreaRatioFollowsTheFilterField

With `min_area_ratio` set to 0.01, the 10 by 100 box is kept and a 10 by 10 box, whose ratio is
0.01, is rejected.

### CandidateNeedsAnAspectRatioBelowTheMaximum

A 300 by 3 box and a 3 by 300 box have an aspect ratio of 100 in float and are rejected. Boxes
of 297 by 3 and 3 by 297 are kept. The filter takes the larger of the two ratios.

### CandidateMeasuresSidesNotPositions

A 50 by 40 box moved to another position is kept. The filter reads widths and heights and no
coordinate.

### CandidateEpsilonIsAddedToTheAreaBefore

With `epsilon` 100 and `min_area_ratio` 0.6, a 10 by 10 box against itself has the ratio
100 / (100 + 100), which is 0.5, and is rejected. Without the epsilon in the denominator the
ratio is 1.

### CandidateEpsilonIsAddedToTheHeightOfAWideBox

With `epsilon` 1, a 300 by 3 box has the ratio 300 / (3 + 1), which is 75, and is kept.

### CandidateEpsilonIsAddedToTheWidthOfATallBox

With `epsilon` 1, a 3 by 300 box has the ratio 75 and is kept.

### CandidateWithAnEmptyBeforeBoxIsKeptAndAnEmptyAfterBoxIsNot

An empty box before and a 10 by 10 box after is kept. A 10 by 10 box before and an empty box
after is rejected.

## test_flip.cpp

Suites: `Unit_ImagingFlip`.

Tests the plan and the application of the horizontal flip to an image and its labels.

This file needs no device. `Unit_ImagingFlip` covers `modules/vision/imaging/include/SushiAI/vision/imaging/flip.hpp`.
`plan_flip` makes the one random draw and records the decision in a `FlipPlan`; `apply_flip`
then mirrors an image or a label set from the plan and draws nothing.

### Constants

`K_BOX_TOLERANCE` is 1e-4 px. `K_SEED_COUNT` is 10000, the seeds 0 to 9999 a statistical case
draws from. `K_FREQUENCY_TOLERANCE` is 0.03, six standard deviations of a frequency of 0.5 over
10000 draws.

### make_counted_image

`make_counted_image` returns an image whose byte at offset i is i modulo 256.

### expect_same_image

`expect_same_image` expects two images to have the same size, channel count and bytes.

### expect_box

`expect_box` compares the four coordinates of two boxes within `K_BOX_TOLERANCE`.

### PlanRecordsTheSizeAndMakesExactlyOneDraw

The plan holds the source size 480 by 640. Its `is_flipped` equals one `bernoulli` draw of a
second generator with the same seed, and the two generators are equal afterwards.

### ProbabilityZeroNeverFlipsAndOneAlwaysFlips

Over 10000 seeds, probability 0 never flips and probability 1 always does, and both leave the
generator in the same state.

### DefaultProbabilityFlipsAboutHalfOfTheSeeds

With the default settings the flip frequency over 10000 seeds is 0.5 within
`K_FREQUENCY_TOLERANCE`.

### PlanRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5 and -0.1 throw with a message containing "validate_flip" and
"probability", and the generator is unchanged.

### PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw

A NaN probability throws and the generator is unchanged.

### PlanRejectsASideThatIsNotPositiveBeforeAnyDraw

Sizes of 0 by 8 and 8 by -2 throw with a message containing "plan_flip", "source size" and the
size, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_flip` on a default `ImageView` throws with "apply_flip" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_flip` with the image as both source and destination throws with "destination holds the
source's pixels", and the image is unchanged.

### ViewInsideTheDestinationIsRejected

A view of three rows of the destination is refused with the same message, and the image is
unchanged.

### FlippedImageHoldsEachRowInReverseColumnOrder

For a 2 by 3 image of three channels, pixel x of each destination row equals pixel 2 - x of the
source row, channel by channel. The first pixel of row 0 is bytes 6, 7, 8: the pixels are
reversed and the channels inside a pixel are not.

### UnflippedImageIsCopiedUnchanged

A plan with `is_flipped` false copies the source.

### FlippingAnImageTwiceRestoresIt

A 4 by 7 image of three channels and a 3 by 4 image of one channel both come back after two
flips.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 3 source with a plan for 2 by 4 or for 5 by 3 throws with a message naming the plan's
size and "2x3".

### FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder

In a 640 px wide image the box (100, 50, 300, 200) becomes (340, 50, 540, 200), and a box
spanning the whole width is unchanged. Classes and order are kept, and x1 stays at or below x2.

### UnflippedBoxesAreReturnedUnchanged

An unflipped plan returns the labels as they are.

### FlippingBoxesTwiceRestoresThem

Two boxes come back after two flips within `K_BOX_TOLERANCE`.

### FlippedPolygonMirrorsEveryVertexAboutTheWidth

In a 640 px wide image the vertices at x = 100, 300 and 150.5 move to 540, 340 and 489.5, and
every y is kept. An object without a polygon still has none.

### UnflippedPolygonIsReturnedUnchanged

An unflipped plan returns the polygon's vertices as they are.

### ValidateAcceptsZeroAndOne

`validate_flip` does not throw for probability 0, 1 or the default.

## test_gray.cpp

Suites: `Unit_ImagingGray`.

Tests the expansion of a one-channel image to three equal channels.

This file needs no device. `Unit_ImagingGray` covers `expand_gray` of
`modules/vision/imaging/include/SushiAI/vision/imaging/gray.hpp`.

### make_counted

`make_counted` returns an image whose byte at offset i is 17 * i + 3 modulo 256.

### ExpandCopiesEachLevelIntoThreeChannels

A 3 by 5 grey image expands to a 3 by 5 image of `K_RGB_CHANNELS` channels, and all three
channels of every pixel hold the pixel's grey level.

### ExpandRewritesAReusedDestination

A destination that was 9 by 9 with four channels and filled with 255 becomes 2 by 2 with three
channels, and every byte holds its pixel's grey level.

### ExpandRejectsASourceThatIsNotGrey

Sources of three and of four channels throw with "expand_gray" and the channel count.
`K_GRAY_CHANNELS` is 1.

### ExpandRejectsASourceWithoutPixelsAndOneInTheDestination

An empty view throws with "no pixels". The destination's own view, and a view of two of its
rows, throw with "destination holds the source's pixels". The image keeps its size, channel
count and bytes.

## test_hsv.cpp

Suites: `Unit_ImagingHsv`.

Tests the hue, saturation and value jitter: its plan, its tables and both colour conversions.

This file needs no device. `Unit_ImagingHsv` covers `modules/vision/imaging/include/SushiAI/vision/imaging/hsv.hpp`.
`plan_hsv` draws three gains, `make_hsv_tables` turns a plan into three 256-entry tables, and
`apply_hsv` converts each pixel with `to_hsv`, looks it up and converts back with `to_rgb`. Hue
runs over `K_HUE_STEPS` (180) steps and every level is an integer, so the expected colours are
exact and a comment in the case derives them.

### Constants

`K_SEED_COUNT` is 10000, the seeds a statistical case draws from. `K_ROUND_TRIP_LEVELS` is 4,
the largest level error of `to_hsv` followed by `to_rgb`. `K_SWEEP_STEP` is 5, the step between
the levels of the colour sweep.

### make_row and pixel_at

`make_row` returns a one-row RGB image of the given pixels, and `pixel_at` returns the pixel at
a column of such an image.

### jitter

`jitter` applies a plan to a single pixel through `apply_hsv` and returns the result.

### expect_rgb and expect_hsv

`expect_rgb` and `expect_hsv` compare the three levels of a pixel for equality.

### level_distance

`level_distance` returns the absolute difference of two levels.

### PlanDrawsHueThenSaturationThenValueAndNothingMore

The plan's three gains equal three `uniform(-1, 1)` draws of a second generator, each times its
setting, in the order hue, saturation, value. The comparison is exact, and the two generators
are equal afterwards.

### DrawnGainsStayInsideTheConfiguredRangesOverManySeeds

Over 10000 seeds every drawn gain lies in [-gain, gain). The smallest of each is below -0.9
times its gain and the largest above 0.9 times it, so the draws cover the range.

### ZeroGainsDrawThreeTimesAndGiveAZeroPlan

With three gains of 0 the plan is three zeros and the generator has advanced by three draws.
The draw count does not depend on the settings.

### PlanRejectsANegativeGain

A saturation gain of -0.5 throws with "validate_hsv" and "gain", and the generator is
unchanged.

### PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw

An infinite hue gain and a NaN value gain throw, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_hsv` on a default view throws with "apply_hsv" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_hsv` with the image as source and destination throws with "destination holds the
source's pixels", and both pixels are unchanged.

### ViewInsideTheDestinationIsRejected

A view of the destination's second pixel is refused with the same message, and the image keeps
its size and pixels.

### ZeroPlanGivesIdentityTables

For a zero plan the saturation and value tables map every level to itself. The hue table maps
x to x below 180 and to x - 180 from 180 up.

### HueTableShiftsByAFractionOfATurnAndWrapsBothWays

A hue gain of 0.25 shifts by 45 steps: 0 maps to 45, 134 to 179 and 135 to 0. A gain of -0.25
maps 0 to 135 and 45 to 0. Every entry of both tables is below 180.

### SaturationAndValueTablesScaleTruncateAndClamp

A saturation gain of 0.5 multiplies by 1.5: 3 maps to 4, which is 4.5 truncated, and 170 and
above map to 255. A value gain of 0.25 multiplies by 1.25: 203 maps to 253 and 204 to 255. A
saturation gain of -0.5 maps 255 to 127, and a value gain of -2 maps every level to 0.

### PrimaryColoursHaveHuesSixtyStepsApart

`to_hsv` gives red, green and blue the hues 0, 60 and 120 at saturation 255 and value 255.

### GreyHasHueZeroAndBlackHasSaturationZero

Black, a mid grey and white have hue 0 and saturation 0, and their value is their level.

### MixedColourConvertsToTheHandComputedLevels

(100, 50, 50) converts to hue 0, saturation 128, value 100. (200, 100, 50) converts to hue 10,
saturation 191, value 200. The comments give the integer arithmetic.

### EachSectorConvertsBackToTheHandComputedColour

`to_rgb` is checked at eight HSV values, one or more in each of the six hue sectors, among them
hue 178 in the last sector and a colour that is not fully saturated.

### PureHueShiftOnRedGivesTheHandComputedColours

Red with a hue gain of 0.25 becomes (128, 255, 0), with -0.25 it becomes (128, 0, 255) and with
0.5 it becomes (0, 255, 255).

### ValueGainClampsAtFullAndAtZero

(100, 50, 50) with a value gain of 2 becomes (255, 127, 127), and with -2 it becomes black.

### SaturationGainClampsAtFullAndAtZero

(100, 50, 50) with a saturation gain of 2 becomes (100, 0, 0), and with -2 it becomes the grey
(100, 100, 100).

### WhiteStaysWhiteUnderASaturationGain

White with a hue gain of 0.3 and a saturation gain of 0.7 stays white. Its saturation is 0, and
0 times any factor is 0.

### ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries

A zero plan returns nine colours unchanged: three greys, the three primaries and the three
secondaries.

### ZeroPlanCopiesColoursTheRoundTripWouldChange

`to_rgb(to_hsv(.))` changes the blue level of (12, 200, 99) to 100 and of (250, 3, 180) to 176,
which the case asserts first. A zero plan returns both colours unchanged, so `apply_hsv` copies
the image for an identity plan and does not send it through the conversion.

### PlanOfZeroGainsCopiesTheImage

Over 64 seeds a plan drawn from three zero gains satisfies `is_hsv_identity`, and the same two
colours come back unchanged.

### PlanWithOneGainAwayFromZeroIsNotAnIdentity

`is_hsv_identity` is true for a zero plan and for one with negative zeros, and false when any
one gain is 1e-6 or -1e-6. A value gain of 1e-6 changes no table entry, and (12, 200, 99) still
comes back as (12, 200, 100): the image went through the conversion.

### ValidateAcceptsZeroGainsAndTheDefaults

`validate_hsv` does not throw for three zero gains or for the defaults.

### RoundTripMovesNoLevelByMoreThanItsBound

Over a sweep of every fifth level on each channel, `to_hsv` gives a hue below 180 and `to_rgb`
of it differs from the original by at most `K_ROUND_TRIP_LEVELS` on any channel.

### ImageWithoutThreeChannelsIsCopiedUnchanged

A one-channel image with a plan that is not the identity comes back with the same size, one
channel and the same bytes.

### SamePlanGivesTheSameImageTwice

Two applications of one plan to one image give the same bytes.

## test_image.cpp

Suites: `Unit_ImagingImage`.

Tests the owned image, the view over foreign pixels and the checks both share.

This file needs no device. `Unit_ImagingImage` covers `modules/vision/imaging/include/SushiAI/vision/imaging/image.hpp`:
`ImageSize`, `ImageView`, `Image`, `overlaps` and `check_source`. Pixels are bytes in
height, width, channel order.

Eleven `static_assert` lines at file scope hold the parts that are constant expressions:
equality of `ImageSize`, `is_valid_side`, `is_valid_size` and `is_valid_channel_count` at both
ends of their ranges and one past each, and `row_bytes()` and `byte_count()` of a view.

### write_offsets

`write_offsets` writes each byte's own offset, modulo 251, into an image.

### ImageSizeComparesHeightAndWidth

Two sizes are equal when both sides are, and 3 by 4 differs from 4 by 3.

### ADefaultImageHasNoPixels

A default `Image` has height, width, channel count, row bytes, byte count and allocated bytes
of 0.

### AConstructedImageReportsItsExtents

An image of 4 by 6 with three channels reports those extents, 18 row bytes and 72 bytes, and
its data pointer is not null.

### AConstructedImageIsZero

Every byte of a constructed image is 0.

### TheLargestSideAndChannelCountAreAccepted

An image of `K_MAX_IMAGE_SIDE` by 1 and one of 1 by `K_MAX_IMAGE_SIDE` with `K_MAX_CHANNELS`
channels are constructed and report their byte counts.

### ConstructionRejectsAHeightOutsideItsBounds

Heights of 0, -2 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAWidthOutsideItsBounds

Widths of 0, -7 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAChannelCountOutsideItsBounds

Channel counts of 0, -1 and 5 throw with a message containing "Image:" and the count; the
first also contains "[1, 4]".

### RowsStartOneRowOfBytesApart

`row(y)` of the image, of a const reference to it and of its view all equal the data pointer
plus 15 * y for a 5 px wide image of three channels.

### AByteWrittenThroughARowIsReadAtItsHwcOffset

A byte written at row 2, column 1, channel 2 of a 3 by 4 image is byte 29 of the data, and
every other byte is 0.

### FillSetsEveryByte

`fill(114)` sets every byte to 114.

### FillOnAnEmptyImageDoesNothing

`fill` on a default image leaves the byte count at 0.

### ResetGivesTheImageNewExtents

`reset` to 5 by 3 with four channels gives 12 row bytes and 60 bytes, and every byte of the new
extent can be written and read back.

### ResetOnADefaultImageAllocates

`reset` on a default image gives a data pointer that is not null and 18 bytes.

### ResetKeepsTheAllocationWhenItIsLargeEnough

An 8 by 8 image of three channels keeps its data pointer through a reset to 6 bytes and through
one to 192 bytes, the count it was constructed with.

### ResetRejectsTheBoundsTheConstructorRejects

`reset` throws for a height of 0, a width of 16385 and five channels, with a message
containing "Image::reset".

### ARejectedResetLeavesTheImageAsItWas

After a `reset` that throws, the image has its size, channel count, data pointer and bytes.

### AViewOfAnImageDescribesTheSamePixels

`view()` has the image's data pointer, size, channel count, row bytes and byte count.

### ADefaultViewIsEmpty

A default `ImageView` has a null pointer, size 0 by 0 and no bytes.

### AViewOverForeignBytesAddressesThemInHwcOrder

A view over a twelve-byte array as 2 by 3 with two channels has 6 row bytes, and row 1,
column 2, channel 1 is byte 11.

### ACopyOwnsItsOwnPixels

A copy has another data pointer, and filling the copy leaves the original's bytes as they
were.

### AllocatedBytesKeepsTheLargestSizeTheImageHad

After a reset from 48 bytes to 4, `byte_count()` is 4 and `allocated_bytes()` is 48.

### OverlapsFindsTheWholeImageAndAViewInsideIt

`overlaps` is true for the image's own view, for a view of two of its rows and for a view of
its last byte.

### OverlapsIsFalseForBytesOfAnotherContainer

A view over an array the image does not own does not overlap it.

### OverlapsCountsTheAllocationBeyondTheCurrentSize

A view of the last row of a 4 by 4 image still overlaps it after the image is reset to one
byte. The test is against the allocation and not the current size.

### OverlapsIsFalseForSeparateOrEmptyOperands

`overlaps` is false for two separate images, for a default view, for an unallocated image, and
for views with a height of 0 or no channels.

### CheckSourceAcceptsAViewOfAnotherImage

`check_source` does not throw for a view of another image, with an allocated or an unallocated
destination.

### CheckSourceRejectsEachFaultUnderTheCallersName

`check_source` throws with the caller's name as prefix for a view without pixels, a height of
0, a width of 16385, five channels, the destination's own view and a view of two of its rows.

## test_label_set.cpp

Suites: `Unit_ImagingLabelSet`.

Tests the label set that holds classes, boxes and polygons in step, and its conversions.

This file needs no device. `Unit_ImagingLabelSet` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/label_set.hpp`. `Labels` holds pixel boxes and `FileLabels` holds
normalized boxes; both keep one class, one box and one polygon per object, and a polygon may be
empty.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px, `K_FRACTION_TOLERANCE` is 1e-6 of the image, and `K_SIZE` is
480 rows by 640 columns.

### expect_pixel_box, expect_normalized_box and expect_point

The three helpers compare a `PixelBox`, a `NormalizedBox` and a `Point` with expected values.
`expect_point` takes its tolerance as an argument.

### make_diagonal

`make_diagonal` returns a polygon of a given number of vertices spread along the diagonal from
(0, 0) towards (1, 1).

### DefaultSetHoldsNoObject

A default `Labels` is empty, has size 0 and has no class, box or polygon.

### ConstructorsRejectListsOfDifferentLengths

Two classes with one box, one class with two boxes, two objects with one polygon, and no class
with one box all throw with a message containing "LabelSet" and the two counts.

### SetWithoutPolygonsHoldsOneEmptyPolygonPerObject

A set built from two classes and two boxes has two polygons, both empty.

### AddAppendsOneEntryToEachList

`add` with and without a polygon grows the classes, the boxes and the polygons together. The
object added without one has an empty polygon.

### KeepIfRemovesAnObjectFromAllThreeLists

`keep_if` with a predicate that keeps even classes leaves classes 2 and 4 with their own boxes
and polygons.

### ToPixelLabelsKeepsClassesAndOrder

`to_pixel_labels` converts three normalized boxes in `K_SIZE` and keeps the classes 7, 2, 7 in
order.

### ToPixelLabelsScalesVerticesByWidthAndHeight

A vertex at (0.25, 0.25) becomes (160, 120): x is multiplied by the width and y by the height.
An empty polygon stays empty.

### ToNormalizedLabelsKeepsClassesAndOrder

`to_normalized_labels` converts two pixel boxes and the vertices of one polygon, and keeps
classes and order.

### LabelConversionsOfABackgroundImageAreEmpty

Both conversions of an empty set are empty.

### ClipLabelsClipsBoxesAndVerticesAndRemovesNone

`clip_labels` clips each box to the image and clamps each vertex onto it. A box wholly outside
collapses to zero width and stays in the set.

### RemoveEmptyBoxesDropsZeroAreaAndKeepsOrderClassesAndPolygons

Of five objects, the two with zero width or zero height are removed. The classes 10, 12 and 14
remain in order with their boxes and polygons.

### RemoveEmptyBoxesDropsABoxOfNegativeArea

A box whose x1 is above its x2 is removed.

### ResamplePolygonsGivesEachPolygonAThousandPoints

`resample_polygons` gives a triangle and a polygon of 40 vertices `K_POLYGON_POINTS` points
each. An empty polygon stays empty, and classes and boxes are unchanged.

### ResamplePolygonsUsesTheLongestPolygonPlusOneAboveAThousand

With one polygon of 1200 vertices in the set, both polygons are resampled to 1201 points.

### ResamplePolygonsLeavesASetWithoutPolygonsAlone

A set whose one polygon is empty comes back with the polygon empty and the box unchanged.

## test_labels.cpp

Suites: `Unit_ImagingLabels`.

Tests the parser of a label file's text.

This file needs no device and reads no file. `Unit_ImagingLabels` passes strings to
`parse_label_text` of `modules/vision/imaging/include/SushiAI/vision/imaging/labels.hpp`. A row is a class and four
normalized box values, or a class and the x, y pairs of a polygon. The refusal cases check that
the message names the line and quotes the field.

### K_CLASS_COUNT and K_FRACTION_TOLERANCE

`K_CLASS_COUNT` is 80, the class count the cases parse with. `K_FRACTION_TOLERANCE` is 1e-7,
the tolerance of a parsed value.

### expect_box

`expect_box` compares the four values of a `NormalizedBox` within `K_FRACTION_TOLERANCE`.

### ParsesBoxRowsInFileOrder

Two rows give two objects, with classes 0 and 3 and the four values of each row.

### EmptyTextIsABackgroundImage

An empty string gives an empty set.

### TextOfBlankLinesIsABackgroundImage

Text of newlines, spaces, a tab and carriage returns gives an empty set.

### AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows

Rows with leading spaces, tabs, repeated spaces, a blank line between them and no final newline
parse to two objects.

### AcceptsCarriageReturnLineEndings

Rows ended by "\r\n" and by a lone "\r" parse to three objects.

### AcceptsVerticalTabAndFormFeedLineEndings

A vertical tab and a form feed each end a row.

### AcceptsTheSeparatorBytesAsLineEndings

The bytes 0x1c, 0x1d and 0x1e each end a row: four rows separated by them parse to four
objects.

### AcceptsTheUnitSeparatorByteBetweenFields

The byte 0x1f separates fields, alone or repeated.

### AcceptsExponentsBareDecimalPointsAndAPlusSign

"+7.0 5e-1 .5 2.5E-1 +0.125" parses to class 7 and the box (0.5, 0.5, 0.25, 0.125).

### RejectsARowOfFourColumns

A row of four columns throws with "parse_label_text: line 1", "5 columns" and "this row has 4".

### RejectsARowOfSixColumns

A row of six columns throws with "5 columns" and "this row has 6".

### RejectsARowOfOneColumn

A row of one column throws with "this row has 1".

### NamesTheLineOfTheRejectedRow

A short row after one row and one blank line is reported as "line 3". Blank lines count.

### RejectsAFieldThatIsNotANumber

The field "abc" throws with "'abc'" and "not a number".

### RejectsAFieldWithCharactersAfterTheNumber

The field "0.5x" throws with "'0.5x'".

### RejectsADecimalComma

The field "0,5" throws with "'0,5'".

### RejectsADoubledSign

The field "+-0.5" throws with "'+-0.5'".

### RejectsNotANumberAndInfinity

The fields "nan", "inf" and "1e999" each throw with the field quoted.

### RejectsAFiniteValueAboveTheFloatRange

The field "1e39", finite as a double and not as a float, throws with "not a finite number".

### RejectsANonzeroValueBelowTheSmallestNormalDouble

The fields "1e-310" and "1e-999" throw with the field quoted.

### AcceptsZeroAndTheSmallestNormalDouble

"0.0" and "2.2250738585072014e-308" are accepted and both store 0 within the tolerance.

### RejectsACoordinateBelowTheTolerance

Values of -0.02 and -0.5 throw with "negative".

### RejectsACoordinateAboveTheTolerance

Values of 1.02 and 640 throw with "above". The second is a pixel value in a file that must hold
fractions.

### StoresACoordinateInsideTheToleranceUnclipped

Values of 1.005 and -0.005 are stored as they are.

### AcceptsACoordinateOnEitherToleranceBound

Values of 1.01 and -0.01 are accepted and stored as they are.

### RejectsANegativeClass

Class -1 throws with "class" and "negative".

### RejectsAClassWithAFraction

Class 1.5 throws with "class" and "whole number".

### RejectsAClassAtOrAboveTheClassCount

Class 80 of 80 throws with "out of range" and "80 classes", and class 1 of 1 with "1 classes".

### AcceptsTheLastClass

Class 79 of 80 is accepted.

### RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder

Five rows of which two repeat earlier ones give three objects, with classes 5, 2, 9 in file
order.

### RemovesARowThatRepeatsAnotherInADifferentSpelling

"5.0 5e-1 0.50 .25 0.250" repeats "5 0.5 0.5 0.25 0.25" and is removed. Rows are compared by
value and not by text.

### KeepsRowsThatDifferInOneValue

Three rows that differ in the class or in one box value are all kept.

### ReducesAPolygonRowToTheBoxAroundItsPoints

A row of three points gives the box (0.5, 0.5, 0.5, 0.5) around them.

### KeepsThePointsOfAPolygonRowInFileOrder

A row of five points gives a polygon of those five points in file order.

### GivesABoxRowNoPolygon

Each object of two box rows has an empty polygon.

### ReducesAPolygonOfManyPointsToItsBounds

The five points of the earlier case give the box (0.5, 0.375, 0.75, 0.5).

### RejectsABoxRowInAPolygonFile

A box row after a polygon row throws with "line 2" and "polygon".

### RejectsAPolygonRowWithAHalfPoint

A polygon row of eight columns throws with "polygon" and "this row has 8".

### RejectsAPolygonRowOfFewerThanThreePoints

A row of three columns in a polygon file throws with "line 2", "three points" and "this row
has 3".

### RejectsAPolygonWhoseBoxLeavesTheImage

A polygon with a vertex at x = 1.5 throws with "above".

### RemovesARepeatedPolygonAndKeepsOneWithTheSameBox

Of three polygon rows the second repeats the first and is removed. The third has other points
and the same box, and is kept: both remaining boxes are (0.5, 0.5, 0.5, 0.5) and their second
vertices differ.

## test_letterbox.cpp

Suites: `Unit_ImagingLetterbox`.

Tests the plan and the application of the letterbox to an image and its labels.

This file needs no device. `Unit_ImagingLetterbox` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/letterbox.hpp`. `plan_letterbox` computes the ratio, the resized size
and the four paddings, and draws no random number. `apply_letterbox` resizes with
`resize_bilinear` of `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`, pads, and moves boxes and polygons
by the same plan.

### K_BOX_TOLERANCE

`K_BOX_TOLERANCE` is 1e-4 px.

### make_counted_image and make_flat_image

`make_counted_image` returns an image whose byte at offset i is a first value plus i, modulo
256. `make_flat_image` returns an image with every byte at one level.

### is_content

`is_content` reports whether a pixel of the target lies inside the plan's resized image.

### expect_two_levels

`expect_two_levels` expects every content byte of an image to be one level and every padding
byte another.

### WideImageIsPaddedAboveAndBelowWithTheOddRowBelow

A 50 by 200 source in a 100 by 100 target has ratio 0.5 and resizes to 25 by 100. The 75 rows
of padding split into 37 above and 38 below, and the plan records the source and target sizes.

### TallImageIsPaddedLeftAndRightWithTheOddColumnRight

A 200 by 50 source resizes to 100 by 25 with 37 columns left and 38 right.

### ExactFitHasRatioOneAndNoPadding

A 64 by 64 source in a 64 by 64 target has ratio 1 and four paddings of 0.

### SmallImageIsOnlyPaddedUnlessScaleUpIsSet

A 20 by 40 source in an 80 by 80 target keeps its size by default, with paddings 30, 30, 20,
20. With `scale_up` set the ratio is 2, the resized size is 40 by 80 and the paddings are 20
above and below.

### ResizedSideRoundsAHalfToTheEvenInteger

At ratio 0.5 a height of 5 resizes to 2 and a height of 7 to 4: 2.5 and 3.5 round to the even
integer.

### RectangularStrideTargetPadsOnlyTheShortAxis

A 480 by 640 source in a 512 by 640 target has ratio 1 and 16 rows above and below.

### PlanCopiesThePaddingValueOfTheSettings

The plan's `padding_value` is that of the settings, at the default and at 7.

### PlanRejectsASideThatIsNotPositive

A source of 0 by 8 and a target of 8 by -1 throw with "plan_letterbox" and the size.

### PlanRejectsASideAboveTheImageBound

A source height and a target width of 16385 throw with the size and "16384".

### ValidateAcceptsEverySetting

`validate_letterbox` does not throw with `scale_up` set and a padding value of 255, or for the
defaults.

### PlanRejectsAResizedSideOfZeroPixels

A 1 by 1000 source in an 8 by 8 target resizes to a height of 0, and the plan throws with "zero
pixels".

### ImageWithoutResizeKeepsItsPixelsInsideThePadding

A 2 by 4 source in a 4 by 6 target is placed at row 1, column 1. Every content byte equals the
source byte and every other byte is the padding value.

### ExactFitCopiesEveryPixel

A plan without resize or padding copies the source.

### WideImageIsResizedThenPaddedWithTheFillValue

A flat 50 by 200 image at level 200 gives a 100 by 100 result that is 200 inside the content
rectangle and 114, the default padding value, outside it.

### TallImageIsResizedThenPaddedWithACustomFillValue

A flat one-channel 200 by 50 image at level 33 with a padding value of 7 gives 33 inside and 7
outside.

### ResizedContentEqualsTheBilinearResizeOfTheSource

For a counted 50 by 200 image, every content byte equals the byte `resize_bilinear` gives for
the plan's resized size, and every padding byte is 114. The case also expects the content not
to be flat, since a flat rectangle cannot tell a resize from a fill.

### SourceWithoutPixelsIsRejected

`apply_letterbox` on a default view throws with "apply_letterbox" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_letterbox` with the image as source and destination throws with "destination holds the
source's pixels", and the image keeps its size.

### ViewInsideTheDestinationIsRejected

A view of two rows of the destination is refused with the same message, and the image keeps its
size and its bytes.

### ResizedContentBesideSidePaddingEqualsTheBilinearResize

A counted 200 by 50 image in a 100 by 101 target has 38 columns of padding on each side. The
destination starts as a 120 by 120 image filled with 255. After the call no byte differs from
the bilinear resize placed between two bands of 114.

### DestinationIsFullyRewrittenWhenItIsReused

A destination that was 9 by 9 and filled with 255 becomes 4 by 6 with content 50 and padding
114.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 4 source with a plan made for 4 by 4 throws with "plan was made for".

### PlanWhosePaddingsDoNotTileTheTargetIsRejected

A plan whose right padding is raised by 1 throws with "tile".

### BoxesMoveByTheRatioThenTheLeftAndTopPadding

In the tall plan a box maps by x * 0.5 + 37 and y * 0.5, and in the wide plan by x * 0.5 and
y * 0.5 + 37. Classes and order are kept.

### BoxesOutsideTheImageAreNeitherClippedNorRemoved

A box that extends past the source and a box of zero size are both moved by the plan and kept.

### ExactFitLeavesBoxesWhereTheyWere

With ratio 1 and no padding a box keeps its four coordinates.

### PolygonMovesByTheRatioThenTheLeftAndTopPadding

In the tall plan the vertices (10, 20), (30, 40) and (-4, 210) move to (42, 10), (52, 20) and
(35, 105). An object without a polygon still has none.

## test_mask_pipeline.cpp

Suites: `Unit_ImagingAffineNearest`, `Unit_ImagingMaskPipeline`.

Tests `apply_affine_nearest` in
`modules/vision/imaging/include/SushiAI/vision/imaging/affine_nearest.hpp` and
`TrainingMaskPipeline` in
`modules/vision/imaging/include/SushiAI/vision/imaging/mask_pipeline.hpp`. Every plan scales by
one factor, translates and has the border value 255. The pipeline cases take grey RGB images
whose one box of the image's index covers the whole image, masks held in memory, and the seed
20261011.

### AnIdentityPlanCopiesTheSource

A plan of scale 1 and no translation over a mask of 3 rows and 4 columns gives an image of
that size and one channel holding the source's bytes.

### ATranslationFillsWhatItUncoversWithTheBorderValue

A translation of 2 columns and 1 row over a mask of 2 rows and 4 columns gives a first row of
255. In the second row the first two columns are 255 and column x holds the source's row 0,
column x - 2.

### AScaleRepeatsPixelsAndBlendsNone

The row 3, 50, 7, 120 warped at scale 2 into 8 columns is 3, 50, 50, 7, 7, 120, 120, 255.

### EveryChannelOfAPixelMovesTogether

An RGB row of the pixels (1, 2, 3) and (4, 5, 6) translated by one column is (255, 255, 255),
(1, 2, 3).

### RefusesWhatItCannotWarp

`apply_affine_nearest` fails with `InvalidDescription` for an empty view ("holds no image"),
for an output that is the source image ("share pixels") and for a matrix with a shear term of
0.5 ("off-diagonal"). A plan whose source size is 3 by 3 over a 2 by 2 image fails with
`Incompatible`, "3x3" and "2x2".

### AFlippedSampleHasItsMaskMirrored

With no mosaic, an identity warp and a flip probability of 1 over an 8 by 8 image, the trace
of sample 0 is flipped and has no mosaic. `make_mask` over that trace gives an 8 by 8 mask
whose pixel (y, x) holds the source mask's pixel (y, 7 - x).

### AMosaicMaskFollowsTheBoxesOfItsSample

Four images of four sizes have masks filled with their own index, and the training size is 32
with the default settings. For each sample of the epochs 0, 1 and 2 the trace has the mosaic
enabled and the mask is 32 by 32 with one channel. Every mask pixel at least 2 pixels inside a
box of the sample holds that box's class, and at least one pixel is compared.

### RefusesAMaskItCannotReplay

`make_mask` fails with `Incompatible` for a mask of three channels ("3 channels") and for a
mask of 8 rows and 4 columns under an 8 by 8 image ("8x4", "8x8"). When the mask source fails
it returns that failure: `InvalidFile` and "mask 0 is unreadable".

## test_mosaic.cpp

Suites: `Unit_ImagingMosaic`.

Tests the draw, the plan and the application of the four-image mosaic.

This file needs no device. `Unit_ImagingMosaic` covers `modules/vision/imaging/include/SushiAI/vision/imaging/mosaic.hpp`.
`draw_mosaic` asks a `PartnerSource` for three partners, draws the gate and then the centre.
`plan_mosaic` turns a centre and four patch sizes into four placements on a canvas of twice the
image size. `apply_mosaic` copies the patches onto the canvas, or shifts, joins, clips and
prunes their labels. The two placement tables are lines 525 to 538 of the reference's
`data/augment.py`, worked by hand.

### K_SIDE and K_BOX_TOLERANCE

`K_SIDE` is 8, the image size of the hand cases, so the canvas is 16 by 16. `K_BOX_TOLERANCE`
is 1e-4 px.

### PatchSizes and Placements

`PatchSizes` holds four `ImageSize` values and `Placements` holds four `MosaicPlacement`
values, patch 0 first.

### The low centre

`K_LOW_CENTER_X` and `K_LOW_CENTER_Y` are 5 and 11, a centre that cuts patch 0 on the left and
patches 2 and 3 at the bottom. `K_LOW_SIZES` are the patch sizes used with it; patch 1 is
shorter and patch 2 narrower than the others. `K_LOW_PLACEMENTS` are the four expected
placements.

### The high centre

`K_HIGH_CENTER_X` and `K_HIGH_CENTER_Y` are 10 and 6, a centre that cuts patches 0 and 1 at the
top and patches 1 and 3 on the right. `K_HIGH_SIZES` are four 8 by 8 patches and
`K_HIGH_PLACEMENTS` are the four expected placements.

### make_draw

`make_draw` returns an enabled `MosaicDraw` with a given centre and the partners 0, 0, 0.

### make_constant and make_textured

`make_constant` returns a three-channel image of one colour. `make_textured` returns one whose
every byte depends on its row, column, channel and a salt, so a copy from the wrong position
shows.

### view_all

`view_all` returns the views of four images in patch order.

### count_wrong_bytes

`count_wrong_bytes` counts the canvas bytes that differ from what four placements name: the
patch byte at the placement's offset inside a placement, and the fill value elsewhere.

### expect_placements

`expect_placements` compares all ten fields of a plan's four placements with the expected ones.

### expect_box and expect_point

`expect_box` and `expect_point` compare a box and a vertex within `K_BOX_TOLERANCE`.

### RecordingPartners

`RecordingPartners` is a `PartnerSource` that returns the partners 4, 5 and 6, makes no draw
and records every index it is asked for.

### pixel_at

`pixel_at` returns the three bytes of one canvas pixel.

### SettingsDefaultToTheReferenceProbabilityAndFill

Default `MosaicSettings` have probability 1 and fill value 114, and `K_MOSAIC_PATCHES` is 4.

### DrawMakesPartnersThenGateThenRowThenColumn

With a `DatasetPartners` of 1000 and image size 640, the draw equals a replay on a copied
generator: three `index(1000)` draws, one `bernoulli(1)`, then `uniform(320, 960)` truncated
for the centre row and again for the centre column. The generators are equal afterwards, so
the call makes six draws.

### DisabledDrawMakesFourDrawsAndLeavesTheFieldsZero

With probability 0 the draw is disabled, its partners and centre are 0, and the generator has
advanced by four draws: the three partners and the gate.

### CentreStaysInTheReferenceRangeOverManySeeds

Over 5000 seeds at image size 640 both centre coordinates lie in 320 to 959. Their mean is
639.5 within 10, the smallest is at most 330 and the largest at least 949.

### CentreOfAnOddSizeUsesTheFlooredBorder

At image size 5 the centre lies in 3 to 6 and all four values occur over 400 seeds. The
reference computes the border as -5 // 2, which is -3, so the range is `int(uniform(3, 7))`.

### PartnersCoverThePoolAndNeverLeaveIt

Over 300 seeds and a pool of 7, every partner is below 7 and every index occurs.

### GateOpensWithItsProbability

At probability 0.25 the draw is enabled for 0.25 of 4000 seeds, within 0.04.

### PlanCutsPatchesAtTheLeftAndBottomEdges

`plan_mosaic` for the low centre gives `K_LOW_PLACEMENTS` and records the image size.

### PlanCutsPatchesAtTheTopAndRightEdges

`plan_mosaic` for the high centre gives `K_HIGH_PLACEMENTS`.

### PlanCarriesTheFillValueOfTheSettings

The plan's `fill_value` is 9 when the settings say 9 and 114 by default.

### CanvasIsTwiceTheSizeAndFilledWhereNoPatchLands

The canvas is 16 by 16 with three channels. Its two outer corners, a pixel in the row above the
shorter patch 1 and a pixel in the column left of the narrower patch 2 hold the fill value.

### EachColourLandsInItsQuadrantAroundTheCentre

Four patches of one colour each are placed around the low centre. The four pixels that meet at
the centre are red, green, blue and yellow, in patch order, and `count_wrong_bytes` is 0.

### CutPatchesKeepThePixelsNearestTheCentre

Four textured patches around the high centre with fill value 3 give a canvas with no wrong
byte. The texture makes the test fail for a patch copied from its far corner.

### ApplyOverwritesWhatTheDestinationHeld

A destination that was 40 by 40 and filled with 200 becomes the 16 by 16 canvas with no wrong
byte.

### LabelsAreShiftedJoinedInPatchOrderClippedAndPruned

Five boxes in patches 0, 1 and 3 are shifted by their placements' offsets and joined in patch
order. The second box of patch 0 lies wholly left of the canvas, has no area after the clip and
is removed. The four that remain have the classes 3, 2, 7, 5 and the boxes the case lists.

### LabelsOfFourBackgroundPatchesAreEmpty

Four empty label sets join to an empty one.

### DrawRejectsANonPositiveSizeBeforeAnyDraw

Image sizes of 0 and -4 throw with "validate_mosaic_size" and the value, and the generator is
unchanged.

### PlanRejectsADisabledDrawAndAnEmptyPatch

`plan_mosaic` throws for a disabled draw ("not enabled"), for a patch of height 0 ("patch 2")
and for a centre column of 17 on a 16 px canvas ("centre").

### ApplyRejectsPatchesThePlanWasNotMadeFor

`apply_mosaic` throws for a one-channel patch among three-channel ones ("patch 1", "channels")
and for a patch of another size than the plan's ("patch 3").

### DrawnCentreIsWhereTheFourColoursMeet

Over 32 seeds the case draws, plans and applies a mosaic of four 8 by 8 patches of one colour
each. The centre lies in 4 to 11, and the four pixels around it are red, green, blue and
yellow. The draw and the plan therefore agree on which coordinate is the row.

### DrawRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5, -0.25 and NaN throw with "validate_mosaic" and "probability", the
default settings pass `validate_mosaic`, and the generator is unchanged.

### DrawRejectsASizeWhoseCanvasIsTooLarge

An image size of 8193, whose canvas would exceed `K_MAX_IMAGE_SIDE`, throws with "image_size of
8193" and "[1, 8192]". An image size of 8192 is accepted.

### DrawAsksThePartnerSourceForEverySample

With a `RecordingPartners`, one draw with the gate open and one with it closed record the
indices 17 and 23. The open draw holds the partners 4, 5, 6 and the closed one 0, 0, 0.

### PolygonsAreShiftedClampedAndPrunedWithTheirObjects

Polygons move with their patch, by (-3, 3) for patch 0 and (5, 11) for patch 3, and are clamped
to the 16 by 16 canvas. The object whose box leaves the canvas is removed with its polygon, and
the object without a polygon still has none.

### PlanRejectsASizeOutOfRangeAndAPatchSideAboveTheLimit

`plan_mosaic` throws for an image size of 0 and of 8193 ("image_size") and for a patch height
of 16385 ("patch 1").

### ApplyRejectsAPlanWithoutAValidSizeForPixelsAndForLabels

Both forms of `apply_mosaic` throw with "image_size" for a default plan and for one whose image
size is 8193.

### ApplyRejectsAPatchWithoutPixels

A default view as patch 2 throws with "apply_mosaic: patch 2" and "no pixels".

### ApplyRejectsAPatchThatLiesAnywhereInTheDestination

A patch that views the start of the destination, and one that views its row 20, throw with
"destination holds the source's pixels" under the patch's number. The destination keeps its
40 by 40 size and its bytes.

### ApplyRejectsAPlacementOutsideTheCanvasAndOneThatResamples

A placement whose `canvas_x2` is 17 throws with "patch 1" and "outside a canvas". A placement
whose patch span is one pixel narrower than its canvas span throws with "patch 3" and
"resamples nothing".

## test_pipeline.cpp

Suites: `Unit_ImagingPipeline`.

Tests the training and the validation pipeline that compose the imaging transforms into a
sample.

This file needs no device and reads no file. `Unit_ImagingPipeline` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/pipeline.hpp`. A `TrainingPipeline` loads an image from a
`SampleSource`, draws and applies mosaic, affine, colour jitter and flip, and returns a
`TrainingTrace` of the plans it used. A `ValidationPipeline` resizes the long side and
letterboxes. The cases replay a sample's draws on a copy of its generator and rebuild the
sample from the trace with the transform functions, so the pipeline is compared with its parts
and not with stored images.

### Constants

`K_SIDE` is 32, the training size of the cases, so a mosaic canvas is 64 by 64.
`K_REPLAY_TOLERANCE` is 0: a replayed draw repeats the same arithmetic. `K_BOX_TOLERANCE` is
1e-4 px. `K_MIN_SCALE_GAP` is 2^-24, one step of the 24-bit uniform over the default scale
range of width 1.

### MemorySource

`MemorySource` is a `SampleSource` that holds images and labels in memory and records the index
of every `load`. `loads` returns the recorded indices and `clear_loads` forgets them.

### FixedPartners

`FixedPartners` is a `PartnerSource` that returns the three partners it was constructed with
and makes no draw.

### make_textured, make_constant and make_flat

`make_textured` returns a three-channel image whose every byte depends on its row, column,
channel and a salt. `make_constant` returns a three-channel image of one colour. `make_flat`
returns an image of a given channel count with every byte at one level.

### one_object

`one_object` returns the `FileLabels` of one object of a given class and box.

### fill_source

`fill_source` adds five textured images to a source: 32 by 64, 64 by 64, 64 by 32, 16 by 32 and
32 by 32. The first four carry one object each and the last carries none.

### make_settings

`make_settings` returns default `TrainingSettings` with the image size `K_SIDE`.

### make_still_settings

`make_still_settings` returns settings under which nothing moves: one epoch with
`close_mosaic` 1, so epoch 0 has no mosaic, affine ranges of 0, HSV gains of 0 and a flip
probability of 0.

### whole_dataset

`whole_dataset` returns a `DatasetPartners` over every image of a source.

### images_match

`images_match` returns a GoogleTest assertion result that says whether two images have the same
size, channel count and bytes, and how they differ when they do not.

### expect_same_labels

`expect_same_labels` expects two label sets to hold the same classes, box edges and polygon
vertices at `K_REPLAY_TOLERANCE`.

### expect_same_affine and expect_same_hsv

`expect_same_affine` compares the sizes and the three drawn values of two `AffinePlan` objects.
`expect_same_hsv` compares the three gains of two `HSVPlan` objects.

### load_patch

`load_patch` loads one image the way a pipeline does: `resize_long_side` to `K_SIDE`,
`expand_gray` when the image has one channel, and `to_pixel_labels` of the resampled polygons.

### compose_from_trace

`compose_from_trace` builds a sample from a `TrainingTrace` by calling the transforms in the
reference's order, mosaic, affine, colour jitter, flip (`data/augment.py`, lines 2870 to 2912),
each with the plan the trace holds.

### SettingsDefaultToTheReferenceRun

Default `TrainingSettings` have image size 640, 100 epochs, `close_mosaic` 10, batch size 16
and seed 0. Default `ValidationSettings` have image size 640 and a letterbox that does not
scale up.

### MosaicClosesForTheLastTenEpochsOfAHundred

`is_mosaic_closed` is false for epochs 0 and 89 of 100 with `close_mosaic` 10 and true for 90
and 99. With `close_mosaic` 0 it is false for epoch 99, and with 5 epochs and `close_mosaic`
10 it is true for epoch 0.

### SampleRandomDerivesFromTheSeedTheEpochAndTheIndex

`make_sample_random(2026, 3, 17)` equals `Random(2026).derive(3).derive(17)`. Another seed,
epoch or index gives another generator, and so do the epoch and the index swapped.

### TrainingDrawsMosaicThenAffineThenHSVThenFlip

The trace of sample 2 in epoch 5 equals a replay on `make_sample_random(41, 5, 2)`:
`draw_mosaic`, then `plan_affine` from the 64 by 64 canvas to 32 by 32, then `plan_hsv`, then
`plan_flip`.

### TrainingLoadsTheSampleThenItsPartnersInDrawOrder

The source records four loads: index 3, then the trace's three partners in order.

### TrainingAppliesItsPlansInTheReferenceOrder

For every index of the source in epochs 0 and 1, the sample's image and labels equal what
`compose_from_trace` builds from the trace.

### TrainingAppliesAForcedFlipAndSkipsAForbiddenOne

Two pipelines with one seed and flip probabilities of 1 and 0 give images that are mirrors of
each other, pixel for pixel. The flipped sample also equals `compose_from_trace` of its trace.

### TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage

The sample is 32 by 32 with `K_RGB_CHANNELS` channels. `original_size` is 32 by 64, the stored
size of image 0, and `loaded_size` is 16 by 32, its size after the long-side resize. The affine
plan maps 64 by 64 to 32 by 32.

### MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize

Four 32 by 32 images each carry a box over the whole image, and the affine ranges are 0, so
the warp shows canvas rows and columns 16 to 48. Over eight seeds every box of the sample lies
inside the 32 by 32 output with positive width and height, and exactly one box equals the
quadrant that is at least 16 by 16, computed from the trace's centre.

### ClosedEpochSkipsTheMosaicAndItsDraws

With 10 epochs and `close_mosaic` 3, epoch 7 is closed. The trace says so, the mosaic is not
enabled, the source records one load, and the affine, HSV and flip plans equal a replay that
starts with `plan_affine` on the sample's generator. No mosaic draw precedes them. The sample
equals `compose_from_trace`.

### SampleWithoutMosaicKeepsTheBoxOfItsOwnImage

In a closed epoch with affine ranges of 0, the box (8, 4, 24, 12) of the 16 by 32 loaded image
becomes (8, 12, 24, 20) in the 32 by 32 output: the image is centred, which adds 8 to y. The
box is symmetric about the centre column, so a flip leaves it where it is.

### EpochBeforeTheClosedOnesStillMakesAMosaic

Epoch 6 of the same settings is open: the mosaic is enabled and the source records four loads.

### ClosedGateInAnOpenEpochWarpsTheLoadedImage

With a mosaic probability of 0 in an open epoch, the trace is not closed and the mosaic is not
enabled. The affine plan's source is the 32 by 16 loaded image, the source records one load,
and the sample equals `compose_from_trace`.

### SampleDependsOnTheSeedTheEpochAndTheIndexAlone

The same seed, epoch and index give the same image, labels, affine plan and partners a second
time. Another seed and another epoch each give another image and an affine scale that differs
by more than `K_MIN_SCALE_GAP`.

### EarlierSamplesDoNotShiftTheDrawsOfALaterOne

A pipeline that already made a mosaic sample and a closed-epoch sample, which differ by six
draws, gives the same sample for index 0 of epoch 0 as a new pipeline does.

### DefaultPartnersComeFromTheImagesThePipelineLoaded

A pipeline built without a partner source gives the first sample, index 3, the partners
3, 3, 3. The second sample's partners are each 3 or 1, the two indices loaded so far.

### GivenPartnerSourceChoosesThePartners

With `FixedPartners` of 4, 4, 0 the trace holds those partners and the source records the loads
2, 4, 4, 0.

### TrainingRejectsEverySettingOutOfRangeAtConstruction

Constructing a `TrainingPipeline` throws for an image size of 0 or 8193, 0 epochs, a
`close_mosaic` of -1, a batch size of 0, a mosaic probability of 1.5, an affine scale range of
1, a negative HSV gain and a NaN flip probability, each under the name of the validator that
found it. No rejected construction loads an image, and `validate_training` accepts the test
settings.

### TrainingRejectsAnEmptySourceAndANullPartnerSource

An empty source throws with "TrainingPipeline" and "no image", with or without a partner
source. A null partner source throws with "partner source is null".

### TrainingRejectsIndicesAndEpochsOutsideTheirRange

`make_sample` throws for index 5 of five images and for epochs -1 and 100 of 100.

### TrainingExpandsAGreyImageToRGB

A one-channel image at level 77 under the still settings gives a 32 by 32 sample of three
channels whose centre pixel is (77, 77, 77).

### TrainingComposesAMosaicOfGreyAndColourImages

A source of two one-channel images and one colour image, with fixed partners, gives a
three-channel 32 by 32 sample that equals `compose_from_trace`.

### TrainingRejectsAnImageOfFourChannels

A four-channel image throws from `make_sample` with "4 channels" and "image 0".

### TrainingCarriesAPolygonAndTakesTheBoxFromIt

Under the still settings a triangle in a 32 by 32 image keeps its place. The sample's one
object has class 5, the box (8, 8, 24, 24) and a polygon of `K_POLYGON_POINTS` points that
starts at (8, 8).

### ValidationResizesTheLongSideThenOnlyPads

An 8 by 16 image of one colour is loaded as 16 by 32 and letterboxed into 32 by 32 with ratio 1
and 8 rows above and below. The sample's rows 8 to 23 hold the colour and the other rows hold
114. The source records one load.

### ValidationMovesBoxesByThePaddingWithoutClipping

The box (8, 4, 24, 12) of the loaded image becomes (8, 12, 24, 20): the top padding adds 8 to
y.

### ValidationIntoARectangularTargetAddsNoPadding

With a 16 by 32 target for a 16 by 32 image the four paddings are 0 and the sample's image
equals the source image.

### ValidationRepeatsItsSampleAfterAnotherIndex

Sample 0 made again after sample 2 has the same image and labels as the first time, and sample
2 has another image.

### ValidationExpandsAGreyImageToRGB

A one-channel 8 by 16 image at level 61 gives a three-channel 32 by 32 sample with 61 in the
content rows and 114 in the padding.

### ValidationCarriesAPolygonThroughTheLetterbox

A triangle's polygon is resampled to `K_POLYGON_POINTS` points, and its first vertex, (8, 4) on
the loaded image, becomes (8, 12).

### ValidationRejectsSettingsAndIndicesOutsideTheirRange

Constructing a `ValidationPipeline` throws for an image size of 0 ("validate_validation") and
for an empty source ("no image"). `make_sample` throws for index 9 of five images.

## test_polygon.cpp

Suites: `Unit_ImagingPolygon`.

Tests polygon resampling, mapping, clamping, the point test and the visible box.

This file needs no device. `Unit_ImagingPolygon` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/polygon.hpp`. The expected vertices and boxes are the outputs of the
reference's `resample_segments` and `segment2box` on the same inputs, written as literals; a
comment in each case names the call.

### Tolerances and K_IMAGE

`K_POINT_TOLERANCE` is 1e-6, for a resampled coordinate of magnitude up to 4.
`K_BOX_TOLERANCE` is 1e-4 px, for a box edge up to 100. `K_IMAGE` is 80 rows by 100 columns.

### expect_polygon

`expect_polygon` compares a polygon with a flat list of expected x, y values.

### expect_box

`expect_box` compares the four edges of a `PixelBox` within `K_BOX_TOLERANCE`.

### make_square, make_triangle and make_hexagon

`make_square` returns the square from (0, 0) to (4, 4). `make_triangle` returns a triangle in
fractions of an image. `make_hexagon` returns six vertices with integer coordinates.

### PointCountIsAThousandOrTheLongestPlusOne

`K_POLYGON_POINTS` is 1000. `resampled_point_count` is 1000 for a longest polygon of 0, 3 or
1000 vertices and 1002 for 1001.

### ResampleMergesTheVerticesWithEvenPositions

The square resampled to 8 and to 11 points keeps its four vertices and gains the evenly spaced
positions between them.

### ResampleToOneMoreThanTheClosedLengthRepeatsTheFirstVertex

The square resampled to 6 points gains one point, at the first vertex.

### ResampleInterpolatesAlongTheEdges

The triangle resampled to 9 points holds interpolated points such as (0.55, 0.275) on its first
edge.

### ResampleOfALongPolygonTakesEvenPositionsOnly

The hexagon, closed, has seven points. Resampled to 4 it gives every second vertex and the
first again; resampled to 7 it gives the six vertices and the first again.

### ResampleLeavesAPolygonOfTheCountAndAnEmptyOneAlone

The triangle resampled to 3 points is unchanged, and an empty polygon stays empty.

### ResampleToAThousandPointsKeepsTheOutline

The triangle resampled to 1000 points has 1000 points, starts and ends at x = 0.1, and has the
bounds (0.1, 0.2, 0.7, 0.9).

### ResampleRejectsACountOfZero

A count of 0 throws with "resample_polygon" and "point count of 0".

### MapScalesThenTranslatesEachAxis

`map_polygon` with scales 2 and 0.5 and offsets 10 and -1 maps the square to the corners
(10, -1), (18, -1), (18, 1), (10, 1).

### ClampMovesVerticesOntoTheBounds

`clamp_polygon` to the box (0, 0, 4, 5) moves each coordinate outside onto the nearest bound
and leaves the vertex inside alone.

### PointInsideOrOnTheOutlineIsInside

`is_point_inside` is true for the centre of the square, a point on an edge and two vertices.

### PointOutsideIsOutside

`is_point_inside` is false for four points outside the square, for a point in the notch of a
concave polygon and for any point against an empty polygon. Two other points of the concave
polygon are inside.

### VisibleBoxOfAPolygonInsideTheImageIsItsBounds

A triangle inside `K_IMAGE` has the visible box (10, 20, 50, 60).

### VisibleBoxOfAPolygonOutsideTheImageIsZero

A triangle above and left of the image, one to its right and an empty polygon each give four
zeros.

### VisibleBoxUsesTheCrossingsOfTheImageLines

A triangle crossing the top and left edges gives (0, 0, 60, 50), and one crossing the right
and bottom edges gives (90, 70, 100, 80).

### VisibleBoxIncludesTheImageCornersThePolygonContains

A triangle that contains the image's origin gives (0, 0, 36.666664, 33), and a square around
the whole image gives the image, (0, 0, 100, 80).

## test_random.cpp

Suites: `Unit_ImagingRandom`.

Tests the generator every augmentation draws from.

This file needs no device. `Unit_ImagingRandom` covers `Random` of
`modules/vision/imaging/include/SushiAI/vision/imaging/random.hpp`: a 64-bit counter advanced by `K_INCREMENT` and passed
through `mix`. The first cases follow the arithmetic by hand, with each intermediate value as a
literal, so the sequence is fixed by this file and not by a second copy of the code. The
statistical cases use one seed and state their bound as a multiple of the standard error.

### Constants

`K_SEED` is 2026, the seed of every statistical case. `K_SAMPLES` is 65536. `K_UNIFORM_VALUES`
is 2^24, the number of values `uniform()` can return.

### MixFollowsTheFinaliserStepByStep

For the value 0x9E3779B97F4A7C15 the case asserts each of the five steps of `mix`: the shift by
30 and xor, the product with `K_FIRST_MULTIPLIER`, the shift by 27 and xor, the product with
`K_SECOND_MULTIPLIER` and the shift by 31 and xor. `mix` returns the last literal.

### MixOfTheSecondCounterFollowsTheFinaliserStepByStep

The same five steps for 0x3C6EF372FE94F82A, the second counter of seed 0.

### SeedZeroReproducesTheHandComputedSequence

The first four `next_u64` values of seed 0 are the literals the case lists, and the counter
after draw n is n times `K_INCREMENT` modulo 2^64.

### TheCounterStartsAtTheSeed

`state()` of a new generator is its seed. The first draw of seed 42 is
`mix(42 + K_INCREMENT)`, and the counter is then `42 + K_INCREMENT`.

### NextU32IsTheHighHalfOfOneDraw

The first two `next_u32` values of seed 0 are the high 32 bits of its first two draws, and two
calls advance the counter twice.

### UniformIsTheTop24BitsScaledByTwoToTheMinus24

The first three `uniform()` values of seed 0 are 14819496, 7239838 and 443485 over 2^24, the
top 24 bits of each draw.

### UniformStaysInTheHalfOpenUnitInterval

65536 draws are all at or above 0 and below 1.

### UniformHasTheMeanAndVarianceOfAUniformVariable

Over 65536 draws the mean is 0.5 within 0.006 and the variance 1/12 within 0.0015, five
standard errors each.

### RangedUniformMapsTheUnitDrawLinearly

`uniform(-1, 1)` and `uniform(0.5, 1.5)` of seed 0 return `low + (high - low) * u` for the
first two unit draws, as exact floats, and make one draw each.

### RangedUniformStaysInItsHalfOpenInterval

65536 draws of `uniform(0.4, 0.6)` are all at or above 0.4 and below 0.6.

### RangedUniformNeverReturnsItsUpperBound

With `high` the float after `low`, 1024 draws all return `low`. A sum that rounds up to `high`
is brought back.

### RangedUniformOverAnEmptyRangeStillDraws

`uniform(0.25, 0.25)` returns 0.25 and advances the generator by one draw.

### IndexIsTheHighWordOfTheProduct

`index(10)` of seed 0 returns 8, 4 and 0: the high 32 bits of each 32-bit draw times 10.

### IndexOfOneIsAlwaysZeroAndStillDraws

64 calls of `index(1)` return 0 and advance the generator 64 times.

### IndexStaysBelowTheCountAndFillsEveryBucketEvenly

70000 draws of `index(7)` are all below 7, and each bucket holds 10000 within 500, above five
standard deviations of 93. `K_BUCKETS` and `K_DRAWS` are local to the case.

### IndexReachesTheLargestCount

`index(0xFFFFFFFF)` of seed 0 returns 0xE220A838.

### BernoulliComparesOneUniformDrawWithTheProbability

`bernoulli(0.5)` of seed 0 returns false, true, true for the unit draws 0.8833, 0.4315 and
0.0264, one draw per call.

### BernoulliNeverSucceedsAtZeroAndAlwaysSucceedsAtOne

Over 65536 draws, probability 0 never succeeds and probability 1 always does, and both make
one draw per call.

### AZeroDrawIsTheLowerBoundAndFailsAZeroProbability

`mix(0)` is 0, so a seed of minus `K_INCREMENT` makes the first draw 64 zero bits. `uniform()`
returns 0, `bernoulli(0)` returns false and `uniform(0.4, 0.6)` returns 0.4. A draw of exactly
0 does not pass a probability of 0.

### BernoulliSucceedsAtTheRateOfItsProbability

Over 65536 draws `bernoulli(0.25)` succeeds at a rate of 0.25 within 0.01.

### DeriveStartsAtTheMixOfTheStateAndTheKeyedMix

`Random(0).derive(0).state()` is 0x48218226FF3CD4BF. The case follows the five steps of `mix`
from the keyed value 0xE220A8397B1DCDAF.

### DeriveDependsOnTheKeyAndOnTheParentState

`Random(0).derive(1)` and `Random(7).derive(5)` have the two states the case lists. Two keys
of one parent give different generators, and so does one key of two parents.

### DeriveLeavesTheParentUntouched

After `derive(3)` the parent's state is its seed, the child differs from the parent, and a
second `derive(3)` equals the first.

### ACopyReplaysTheSequenceFromThePointOfTheCopy

A copy taken after one draw equals the original, and their next 16 draws are equal.

### EqualityComparesTheCounter

Two generators of one seed are equal, and are unequal after one of them draws.

## test_raw_cache.cpp

Suites: `Unit_ImagingRawCache`.

Tests the raw image file: its bytes, its header checks and its file reader and writer.

This file needs no device. `Unit_ImagingRawCache` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/raw_cache.hpp`. A file is a 24-byte header of six little-endian
32-bit words (magic, version, height, width, channel count, channel order) followed by the
pixels. Most cases read and write string streams. The cases from ImageRoundTripsThroughAFile on
write files in a directory of their own under the system's temporary directory, which
`ScratchDirectory` creates empty and removes when the case ends.

### K_RGB

`K_RGB` is 1, the channel order word of an RGB file.

### append_u32

`append_u32` appends a value to a byte string as four little-endian bytes.

### header_bytes and layout_bytes

`header_bytes` returns the 24 header bytes for six given fields. `layout_bytes` returns them
for `K_RAW_IMAGE_MAGIC`, `K_RAW_IMAGE_VERSION` and a given height, width, channel count and
order.

### make_image

`make_image` returns an image whose byte at offset i is 7 * i + 3 modulo 256.

### bytes_of

`bytes_of` returns a copy of an image's bytes.

### written_bytes

`written_bytes` returns the bytes `write_raw_image` produces for an image.

### expect_untouched

`expect_untouched` expects an image to be 1 by 1 with one channel still, the destination a
failed read must not replace.

### ScratchDirectory

`ScratchDirectory` owns one directory under `std::filesystem::temp_directory_path()`. It
removes what the path held and creates the directory in its constructor, and removes it in its
destructor. `write` puts bytes in a file of the directory and `file` returns a path inside it.
The class cannot be copied or moved.

### WriterEmitsTheDocumentedBytes

A 1 by 2 RGB image is written as "SAIM", the words 1, 1, 2, 3, 1 and the six pixel bytes. The
case lists all thirty bytes.

### HeaderSizeAndMagicMatchTheDocumentedLayout

`K_RAW_IMAGE_HEADER_BYTES` is 24 and the first four header bytes are "SAIM".

### RGBImageRoundTripsThroughAStream

A 5 by 7 RGB image written and read back has the header fields 5, 7, 3 and
`ChannelOrder::RGB`, `pixel_bytes()` of 105 and the same bytes.

### GrayImageRoundTripsThroughAStream

A 4 by 3 one-channel image round-trips with `ChannelOrder::Gray`.

### BGRFileIsWrittenAsGivenAndReadAsRGB

Written with `ChannelOrder::BGR`, the pixel bytes 1 to 6 are stored as given. Read back they
are 3, 2, 1, 6, 5, 4: the reader swaps the first and third channel so the image is RGB.

### ReadReusesADestinationOfAnotherSize

A destination that was 9 by 9 with one channel becomes 2 by 2 with three channels and holds the
file's bytes.

### HeaderRejectsAWrongMagic

A header that starts with the PNG signature bytes throws with "magic" and "not a raw image
file".

### HeaderRejectsANewerVersionAndNamesBoth

Version 2 throws with "format version 2" and "reads version 1".

### HeaderRejectsVersionZero

Version 0 throws with "format version 0" and "reads version 1".

### HeaderRejectsATruncatedHeader

A stream of 23 bytes throws with "header ended" and "23 of 24".

### HeaderRejectsAnEmptyStream

An empty stream throws with "header ended" and "0 of 24".

### HeaderRejectsAZeroHeight

A height of 0 throws with "height 0" and "16384".

### HeaderRejectsAZeroWidth

A width of 0 throws with "width 0" and "16384".

### HeaderRejectsAHeightAboveTheLimit

A height of 16385 throws with "height 16385".

### HeaderRejectsAWidthAboveTheLimit

A width of 16385 throws with "width 16385".

### HeaderAcceptsBothSidesAtTheLimit

A 16384 by 16384 RGB header is accepted, and `pixel_bytes()` is 805306368.

### HeaderRejectsExtentsWhoseProductWouldOverflow

Extents of 0xFFFFFFFF throw with "height 4294967295". The extent is checked before any product
is formed.

### HeaderRejectsAnExtentWithTheSignBitSet

A width of 0x80000001 throws with "width 2147483649". The word is read as unsigned and is not
taken for a negative number.

### HeaderRejectsAnUnknownChannelOrder

A channel order of 3 throws with "channel order 3" and "unknown".

### HeaderRejectsThreeChannelsInAGrayFile

Three channels with order 0 throw with "channel count 3" and "channel order 0".

### HeaderRejectsOneChannelInAnRGBFile

One channel with order 1 throws with "channel count 1" and "channel order 1".

### HeaderRejectsFourChannels

Four channels throw with "channel count 4".

### HeaderRejectsZeroChannels

Zero channels with order 2 throw with "channel count 0" and "channel order 2".

### PixelsRejectAStreamThatEndsEarly

A stream one byte short of its pixels throws with "pixels ended" and "11 of 12".

### PixelsRejectAHeaderTheReaderDidNotValidate

`read_raw_image_pixels` given a hand-built header of height 0 throws with "height 0" and leaves
the destination as it was.

### WriterRejectsAnEmptyView

A default view throws with "no pixels" and nothing is written.

### WriterRejectsAChannelCountThatDoesNotFitTheOrder

One channel with `RGB` or `BGR` and three channels with `Gray` throw with the channel count,
and nothing is written.

### WriterRejectsASideAboveTheLimit

A view 16385 px wide throws with "width 16385" and nothing is written.

### WriterReportsAFailedStream

Writing to a stream whose bad bit is set throws with "stream failed".

### ImageRoundTripsThroughAFile

A 6 by 4 RGB image written with `write_raw_image_file` gives a file of 24 + 72 bytes, and
`read_raw_image_file` returns the same extents and bytes.

### FileWriterReplacesALongerFile

Writing a 1 by 1 image over a file that held an 8 by 8 one leaves a file of 24 + 3 bytes that
reads back as the 1 by 1 image.

### FileWriterLeavesTheFileAloneWhenItRejectsTheImage

A write that throws for its channel count leaves an existing four-byte file at four bytes. The
image is checked before the file is opened.

### FileWriterRejectsAPathItCannotOpen

A path inside a directory that does not exist throws with "cannot open".

### FileReaderRejectsAMissingFile

A missing file throws with "cannot open" and the file name, and the destination is untouched.

### FileReaderRejectsATruncatedHeader

A file of 10 bytes throws with "header ended" and "10 of 24", and the destination is untouched.

### FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination

A file one byte short throws with "holds 35 bytes" and "declares 36", and the destination is
untouched.

### FileReaderRejectsTrailingBytesBeforeResettingTheDestination

A file with one byte after its pixels throws with "holds 37 bytes" and "declares 36", and the
destination is untouched.

### FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold

A file that is only a header declaring 16384 by 16384 pixels throws with "holds 24 bytes" and
"declares 805306392". The destination is untouched, so the reader compares the file size with
the header before it allocates.

### FileReaderRejectsAMalformedHeaderBeforeResettingTheDestination

A file whose header has extents of 0xFFFFFFFF throws with "height 4294967295", and the
destination is untouched.

## test_rectangle.cpp

Suites: `Unit_ImagingRectangle`.

Tests the plan that orders a dataset by aspect ratio and gives each batch one shape.

This file needs no device. `Unit_ImagingRectangle` covers
`modules/vision/imaging/include/SushiAI/vision/imaging/rectangle.hpp`. The expected shapes are what lines 405 to 427 of the
reference's `data/base.py` give for the same sizes.

### make_sizes

`make_sizes` returns seven image sizes with height over width below, at and above 1. Images 0
and 5 share the ratio 0.75.

### make_settings

`make_settings` returns `RectangleSettings` with a batch size, an image size and a pad, at the
default stride of 32.

### SettingsDefaultToTheReferenceValidationRun

Default settings are image size 640, batch size 32, stride 32 and pad 0.5.

### OrderRisesWithTheAspectRatioAndKeepsTies

`plan_rectangular_batches` orders the seven images as 6, 3, 0, 5, 2, 1, 4. Images 0 and 5 keep
their dataset order.

### ShapesMatchTheReferenceArithmetic

With batch 3 and pad 0.5 the shapes are 512 by 672, 672 by 672 and 672 by 448. With batch 2 and
pad 0 they are 448 by 640, 480 by 640, 640 by 640 and 640 by 448. With one batch of 7 at image
size 320 the shape is 352 by 352.

### ShapeAtNamesTheBatchOfAPosition

`shape_at` returns the first batch's shape for positions 0 and 2, the second's for position 3
and the third's for position 6.

### EverySideIsAMultipleOfTheStride

With stride 64 and image size 600, both sides of all four shapes are multiples of 64 and at
least 64.

### OneWideImageGivesOneWideShape

One 480 by 640 image gives the order {0} and the shape 512 by 672:
ceil(0.75 * 640 / 32 + 0.5) * 32 and ceil(20.5) * 32.

### RejectsSettingsAndSizesOutsideTheirRanges

`validate_rectangle` throws for a batch size, a stride or an image size of 0 and for a pad that
is negative or NaN. `plan_rectangular_batches` throws for an empty size list ("no image") and
for a size of 0 by 500 ("image 2", "0x500"). Default settings pass.

### RejectsAShapeAboveTheImageBound

An image size of `K_MAX_IMAGE_SIDE` with pad 0.5 gives a side above the bound, and the call
throws with "batch side" and "16384".

## test_resize.cpp

Suites: `Unit_ImagingResize`.

Tests the bilinear resize and the fit of an image's long side.

This file needs no device. `Unit_ImagingResize` covers `modules/vision/imaging/include/SushiAI/vision/imaging/resize.hpp`:
`resize_bilinear`, `fit_long_side` and `resize_long_side`. The kernel samples at pixel centres
with weights rounded to `K_BILINEAR_WEIGHT_BITS` bits and rounds its result half up. The first
cases compare whole images with values a comment derives.

### Bytes

`Bytes` is a vector of bytes, an image's contents in height, width, channel order.

### make_image and bytes_of

`make_image` returns an image holding the given bytes, and `bytes_of` returns an image's bytes.

### resized_bytes

`resized_bytes` resizes a one-channel source to a size and returns the bytes. It also expects
the destination to have that size and one channel.

### WeightBitsGiveAWeightSumOf2048

`1 << K_BILINEAR_WEIGHT_BITS` is 2048.

### TwoByTwoToFourByFourMatchesTheHandComputedImage

Along one axis the four outputs of a pair (a, b) are a, (3a + b) / 4, (a + 3b) / 4 and b. The
case lists all sixteen output bytes, among them 58.75 rounded to 59 and 76.25 rounded to 76.

### FourByFourToTwoByTwoAveragesEachTwoByTwoBlock

Each output reads position 0.5 or 2.5 on both axes, so it is the mean of a 2 by 2 block. The
sums 142, 220, 460 and 541 give 36, 55, 115 and 135: 35.5 rounds up and 135.25 rounds down.

### ResizeToTheSameSizeCopiesEveryPixel

A 3 by 3 image resized to 3 by 3 is unchanged.

### AnEnlargedRowClampsAtBothEdges

A row (10, 250) enlarged to four columns starts at 10 and ends at 250. Its outer columns read
positions outside the two pixel centres.

### AReducedRowUsesWeightsRoundedToElevenBits

Eight columns reduced to three read positions 0.8333, 3.5 and 6.1667, with upper weights 1707,
1024 and 341 of 2048. The case computes the three bytes from those weights, checks they are 3,
41 and 209, and compares the resize with them.

### EachChannelIsInterpolatedOnItsOwn

A two-pixel row of three channels enlarged to four pixels gives each channel the values of its
own pair. Channel 2 falls from 255 to 3 while the others rise.

### AConstantImageStaysConstantWhenEnlarged

A 5 by 7 image of the colour (17, 114, 255) enlarged to 13 by 23 holds that colour in every
pixel.

### AConstantImageStaysConstantWhenReduced

A 31 by 17 image at level 201 reduced to 4 by 5 holds 201 in every byte.

### AHorizontalGradientStaysMonotoneAtEveryWidth

A rising row of eight levels resized to every width from 1 to 40 never falls from one column to
the next.

### AnEnlargedGradientKeepsItsEndValues

The same row enlarged to 20 columns starts at 0 and ends at 255.

### AVerticalGradientStaysMonotoneAndRowsStayUniform

A 4 by 3 image whose rows are each one level is resized to every height from 1 to 17 at width
5. Every output row is one level, and the levels never fall from one row to the next.

### TheDestinationIsResetToTheRequestedSizeAndChannels

A destination that was 9 by 9 with one channel becomes 2 by 2 with the source's three channels.

### ResizeRejectsASourceWithoutPixels

`resize_bilinear` on a default view throws with "resize_bilinear", "source" and "no pixels".

### ResizeRejectsASizeOutsideTheImageBounds

Destination sizes of 0 by 2, 2 by -1 and 2 by 16385 throw with "destination size" and the size.

### ResizeRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2, of 2 by 16385 and of five channels throws with "source size" or
"source channel count".

### ResizeRejectsADestinationThatHoldsTheSource

`resize_bilinear` with the image as source and destination throws with "destination holds the
source's pixels". The image is still 2 by 2 with its bytes, so the refusal comes before the
reset.

### FitLongSideLeavesAnImageWhoseLongSideAlreadyFits

`fit_long_side` at 640 returns 480 by 640, 640 by 427 and 640 by 640 unchanged.

### FitLongSideReducesALargeImage

1080 by 1920 becomes 360 by 640, and 1920 by 1080 becomes 640 by 360.

### FitLongSideEnlargesASmallImage

100 by 50 becomes 640 by 320, and 500 by 375 becomes 640 by 480.

### FitLongSideRoundsTheShortSideUp

333 by 500 becomes 427 by 640, 7 by 3 becomes 640 by 275 and 3 by 1000 becomes 2 by 640: the
short side is the ceiling of 426.24, 274.29 and 1.92.

### FitLongSideCapsBothSidesAtTheImageSize

1281 by 1280 becomes 640 by 640. The short side would be the ceiling of 639.5.

### FitLongSideAlwaysPutsTheLongSideAtTheImageSize

Over heights and widths from 1 to 1499 in steps of 7 and 11, the long side of the result is 640
and the short side lies in [1, 640].

### ResizeLongSideCopiesAnImageThatAlreadyFits

A 2 by 4 image with image size 4 comes out as a 2 by 4 image with the same bytes under another
data pointer. The destination, which was 3 by 3 with three channels, takes the source's one
channel.

### ResizeLongSideEnlargesToTheFittedSize

A 1 by 2 image with image size 4 becomes 2 by 4, and both rows are 0, 25, 75, 100.

### ResizeLongSideReducesToTheFittedSize

The 4 by 4 image of the block-average case with image size 2 becomes the 2 by 2 image 36, 55,
115, 135.

### ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds

Image sizes of 0 and 16385 throw with "resize_long_side" and the value.

### ResizeLongSideRejectsASourceWithoutPixels

A default view throws with "resize_long_side" and "no pixels".

### ResizeLongSideRejectsADestinationThatHoldsTheSource

The image as source and destination throws with "destination holds the source's pixels".

### ResizesRejectAViewInsideTheDestination

`resize_bilinear` and `resize_long_side` both refuse a view of two rows of the destination, and
the image keeps its size and its bytes.

### ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2 and one of five channels throw with "source size" and "source channel
count".

### NearestRepeatsEachPixelWhenEnlarged

`resize_nearest` of the 2 by 2 image 1, 2, 3, 4 to 4 by 4 gives one channel with the rows
1 1 2 2, 1 1 2 2, 3 3 4 4, 3 3 4 4.

### NearestReadsThePixelUnderEachCentreWhenReduced

The row 10, 11, 12, 13, 14 resized to two columns is 11, 13.

### NearestKeepsClassIndicesUnblended

The row 0, 255 resized to seven columns is 0, 0, 0, 255, 255, 255, 255.

### NearestCopiesEveryChannelOfAPixel

An RGB row of the pixels (1, 2, 3) and (4, 5, 6), resized to 2 by 2 into a destination that
was 9 by 9 with one channel, gives a 2 by 2 image of three channels holding that row twice.

### NearestRejectsABadSourceABadSizeAndAnAliasedDestination

`resize_nearest` raises an error for an empty view ("resize_nearest", "source", "no pixels"),
for a size of 0 by 2 ("Image::reset", "0x2") and for a destination that is the source image
("resize_nearest", "destination holds the source's pixels"). The source still holds its four
bytes of 9.
