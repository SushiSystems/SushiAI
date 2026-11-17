# tests/language/training/test_calibration_error.cpp

Suites: `Small_LanguageTrainingCalibrationError`, `Small_LanguageTrainingCalibrationErrorRefusal`.

Tests `expected_calibration_error` in
`modules/language/training/include/SushiAI/language/training/calibration_error.hpp`, over 15 bins and
the sets of `calibration_test_support.hpp` (see `test_temperature_scaling.md`).

## TheErrorOfTheHandSetIsTheHandValue

Four two-class rows: two with the margin `2 ln 3` labelled 0 and 1, two with `ln 3` labelled 0 and 0.
The first pair has confidence 9/10, which is bin 13 (13.5 of 15), and one of the two is right, so its
gap is `|1 - 1.8| = 0.8`. The second pair has confidence 3/4, bin 11 (11.25), and both are right, so
`|2 - 1.5| = 0.5`. The other bins are empty. The error is `(0.8 + 0.5) / 4 = 0.325`. The bound is the
float roundoff of the widest margin times the largest slope of the logistic function, 1/4, which
bounds how far a confidence moves, plus eight rounded operations of the sums.

## ACalibratedSetHasNoError

Eight rows at 3/4 with six right fill bin 11 with an accuracy equal to its mean confidence; the
error is 0 within the float roundoff of the margin times 1/4.

## AConfidenceOfOneIsInTheLastBin

The logits `[100, 0]` labelled 1 give a confidence of exactly 1, since `exp(-100)` is under half a
unit of 1. The row is wrong and lands in bin 14, so the error is exactly 1.

## ATemperatureDividesTheLogits

The error of the `f = 2` set at `T = 2` equals that of the `f = 1` set at `T = 1` bit for bit.

## ScalingByTheFittedTemperatureLowersTheError

At `f = 2` the confidence is 9/10 and the accuracy 3/4, so the error before scaling is
`0.9 - 0.75 = 0.15` within the bound of the hand case. The error at the fitted temperature is below
it.

## TheMetricRefusesWhatDoesNotFit

One promise over four rows: `expected_calibration_error` fails with `InvalidDescription` naming
`expected_calibration_error` for no row, a label past the classes, a temperature of 0 and a NaN
temperature.
