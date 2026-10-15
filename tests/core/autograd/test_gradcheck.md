# tests/core/autograd/test_gradcheck.cpp


Suites: `Unit_GradCheck`.

Tests the gradient checker against closed-form functions and a wrong gradient.

This file needs no device. The gradient checker is the instrument the integration tier measures
with, so this file calibrates it against closed-form functions whose right answer is known. One
of them is a deliberately wrong gradient, which proves the check can fail.

## AcceptsAnExactGradient

`check_full_jacobian` of the cubic and its gradient at six coordinates passes with a largest
relative error below 1e-7 and six probes.

## RejectsAGradientThatIsWrongByATenthOfAPercent

This case is the reason for insisting on FLOAT64. A 1e-3 relative error is far above the noise
floor of about 1e-9 in FLOAT64. It would sit inside the floor of about 2.4e-5 that an fp32
difference quotient bottoms out at, and would pass unnoticed.

## LeavesTheProbePointUnchanged

The probe point holds the same values after `check_full_jacobian` and after
`check_directional`.

## DirectionalModeAgreesWithTheFullJacobian

At eight coordinates both `check_full_jacobian` and `check_directional` with 12 directions
pass, and the directional result reports 12 probes.

## DirectionalModeCatchesAWrongGradientToo

`check_directional` with 16 directions over `slightly_wrong_gradient` does not pass.

## DirectionalModeIsReproducibleForAGivenSeed

Two `check_directional` calls with the seed 12345 report the same largest relative error and
the same worst index.

## PushOffKinksMovesCoordinatesAwayFromZeroAndKeepsTheirSign

`push_off_kinks` over 0.0, 1e-12, -1e-12, 5.0, -5.0 leaves the first non-zero, the second
positive, the third negative and the last two unchanged. Every coordinate is then at least
`kink_margin * min_step * 0.999` from zero.

## AReLUFamilyFunctionNeedsToBeSampledOffTheKink

`options.min_step` is large enough that the difference survives the cancellation of summing it
against the O(1) terms. A step of 1e-4 keeps the pushed coordinates at 1e-3, where a central
difference of x^2 is still exact to 1e-9.

The `unshifted` point straddles the kink. The quotient taken across it reports the average of
the two one-sided derivatives, which no correct rule can match.

## AnEmptyProbePointIsRejected

`check_full_jacobian` over an empty point throws `Error`, and so does `check_directional` with
0 directions.
