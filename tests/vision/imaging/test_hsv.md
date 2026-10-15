# tests/vision/imaging/test_hsv.cpp


Suites: `Small_ImagingHsv`.

Tests the hue, saturation and value jitter: its plan, its tables and both colour conversions.

This file needs no device. `Small_ImagingHsv` covers `modules/vision/imaging/include/SushiAI/vision/imaging/hsv.hpp`.
`plan_hsv` draws three gains, `make_hsv_tables` turns a plan into three 256-entry tables, and
`apply_hsv` converts each pixel with `to_hsv`, looks it up and converts back with `to_rgb`. Hue
runs over `K_HUE_STEPS` (180) steps and every level is an integer, so the expected colours are
exact and a comment in the case derives them.

## Constants

`K_SEED_COUNT` is 10000, the seeds a statistical case draws from. `K_ROUND_TRIP_LEVELS` is 4,
the largest level error of `to_hsv` followed by `to_rgb`. `K_SWEEP_STEP` is 5, the step between
the levels of the colour sweep.

## make_row and pixel_at

`make_row` returns a one-row RGB image of the given pixels, and `pixel_at` returns the pixel at
a column of such an image.

## jitter

`jitter` applies a plan to a single pixel through `apply_hsv` and returns the result.

## expect_rgb and expect_hsv

`expect_rgb` and `expect_hsv` compare the three levels of a pixel for equality.

## level_distance

`level_distance` returns the absolute difference of two levels.

## PlanDrawsHueThenSaturationThenValueAndNothingMore

The plan's three gains equal three `uniform(-1, 1)` draws of a second generator, each times its
setting, in the order hue, saturation, value. The comparison is exact, and the two generators
are equal afterwards.

## DrawnGainsStayInsideTheConfiguredRangesOverManySeeds

Over 10000 seeds every drawn gain lies in [-gain, gain). The smallest of each is below -0.9
times its gain and the largest above 0.9 times it, so the draws cover the range.

## ZeroGainsDrawThreeTimesAndGiveAZeroPlan

With three gains of 0 the plan is three zeros and the generator has advanced by three draws.
The draw count does not depend on the settings.

## PlanRejectsANegativeGain

A saturation gain of -0.5 throws with "validate_hsv" and "gain", and the generator is
unchanged.

## PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw

An infinite hue gain and a NaN value gain throw, and the generator is unchanged.

## SourceWithoutPixelsIsRejected

`apply_hsv` on a default view throws with "apply_hsv" and "no pixels".

## DestinationThatIsTheSourceIsRejected

`apply_hsv` with the image as source and destination throws with "destination holds the
source's pixels", and both pixels are unchanged.

## ViewInsideTheDestinationIsRejected

A view of the destination's second pixel is refused with the same message, and the image keeps
its size and pixels.

## ZeroPlanGivesIdentityTables

For a zero plan the saturation and value tables map every level to itself. The hue table maps
x to x below 180 and to x - 180 from 180 up.

## HueTableShiftsByAFractionOfATurnAndWrapsBothWays

A hue gain of 0.25 shifts by 45 steps: 0 maps to 45, 134 to 179 and 135 to 0. A gain of -0.25
maps 0 to 135 and 45 to 0. Every entry of both tables is below 180.

## SaturationAndValueTablesScaleTruncateAndClamp

A saturation gain of 0.5 multiplies by 1.5: 3 maps to 4, which is 4.5 truncated, and 170 and
above map to 255. A value gain of 0.25 multiplies by 1.25: 203 maps to 253 and 204 to 255. A
saturation gain of -0.5 maps 255 to 127, and a value gain of -2 maps every level to 0.

## PrimaryColoursHaveHuesSixtyStepsApart

`to_hsv` gives red, green and blue the hues 0, 60 and 120 at saturation 255 and value 255.

## GreyHasHueZeroAndBlackHasSaturationZero

Black, a mid grey and white have hue 0 and saturation 0, and their value is their level.

## MixedColourConvertsToTheHandComputedLevels

(100, 50, 50) converts to hue 0, saturation 128, value 100. (200, 100, 50) converts to hue 10,
saturation 191, value 200. The comments give the integer arithmetic.

## EachSectorConvertsBackToTheHandComputedColour

`to_rgb` is checked at eight HSV values, one or more in each of the six hue sectors, among them
hue 178 in the last sector and a colour that is not fully saturated.

## PureHueShiftOnRedGivesTheHandComputedColours

Red with a hue gain of 0.25 becomes (128, 255, 0), with -0.25 it becomes (128, 0, 255) and with
0.5 it becomes (0, 255, 255).

## ValueGainClampsAtFullAndAtZero

(100, 50, 50) with a value gain of 2 becomes (255, 127, 127), and with -2 it becomes black.

## SaturationGainClampsAtFullAndAtZero

(100, 50, 50) with a saturation gain of 2 becomes (100, 0, 0), and with -2 it becomes the grey
(100, 100, 100).

## WhiteStaysWhiteUnderASaturationGain

White with a hue gain of 0.3 and a saturation gain of 0.7 stays white. Its saturation is 0, and
0 times any factor is 0.

## ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries

A zero plan returns nine colours unchanged: three greys, the three primaries and the three
secondaries.

## ZeroPlanCopiesColoursTheRoundTripWouldChange

`to_rgb(to_hsv(.))` changes the blue level of (12, 200, 99) to 100 and of (250, 3, 180) to 176,
which the case asserts first. A zero plan returns both colours unchanged, so `apply_hsv` copies
the image for an identity plan and does not send it through the conversion.

## PlanOfZeroGainsCopiesTheImage

Over 64 seeds a plan drawn from three zero gains satisfies `is_hsv_identity`, and the same two
colours come back unchanged.

## PlanWithOneGainAwayFromZeroIsNotAnIdentity

`is_hsv_identity` is true for a zero plan and for one with negative zeros, and false when any
one gain is 1e-6 or -1e-6. A value gain of 1e-6 changes no table entry, and (12, 200, 99) still
comes back as (12, 200, 100): the image went through the conversion.

## ValidateAcceptsZeroGainsAndTheDefaults

`validate_hsv` does not throw for three zero gains or for the defaults.

## RoundTripMovesNoLevelByMoreThanItsBound

Over a sweep of every fifth level on each channel, `to_hsv` gives a hue below 180 and `to_rgb`
of it differs from the original by at most `K_ROUND_TRIP_LEVELS` on any channel.

## ImageWithoutThreeChannelsIsCopiedUnchanged

A one-channel image with a plan that is not the identity comes back with the same size, one
channel and the same bytes.

## SamePlanGivesTheSameImageTwice

Two applications of one plan to one image give the same bytes.
