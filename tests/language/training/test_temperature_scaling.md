# tests/language/training/test_temperature_scaling.cpp

Suites: `Small_LanguageTrainingTemperatureScaling`, `Small_LanguageTrainingTemperatureScalingFactor`,
`Small_LanguageTrainingTemperatureScalingSet`, `Small_LanguageTrainingTemperatureScalingTemperature`.

Tests `fit_temperature` and `mean_negative_log_likelihood` in
`modules/language/training/include/SushiAI/language/training/temperature_scaling.hpp`. The sets come
from `calibration_test_support.hpp`: groups of four two-class rows with the logits `[f * ln 3, 0]`
and the labels 0, 0, 0, 1, where `f` is the factor of overconfidence. At `f = 1` three of four rows
are right and the logits say 3/4, so the loss is least at `T = 1`: at inverse temperature 1 each
right row adds `-m (1 - p)` to the slope of the loss and the wrong one adds `m p`, with `p = 3/4`,
and the three and the one cancel. Scaling the logits by `f` moves the least point to `T = f`.

## TheFittedTemperatureEqualsTheFactor

One promise over the factors 1, 1.5, 2 and 4: the fitted `ln T` is `ln f` within the derived bound.
The bound is the sum of three terms. The search stops at `K_TEMPERATURE_LOG_TOLERANCE`. The margin is
a float, so it is off by at most the float unit roundoff, relative, which moves the least point of
`ln T` by at most that. The loss is evaluated to `rows + 8` rounded operations in double, an error
`e = (rows + 8) u L` of a loss `L`, so two points whose losses differ by under `2e` cannot be told
apart; with the second derivative of the loss in `ln T` at the minimum, `m^2 p (1 - p)`, that is
`sqrt(4 e / (m^2 p (1 - p)))`. The least loss is `(3 ln(4/3) + ln 4) / 4`.

## TheLossOfAnOverconfidentSetIsTheHandValue

At `f = 2` a row is right with probability 9/10, so the mean loss at `T = 1` is
`(3 ln(10/9) + ln 10) / 4`. The bound is the float roundoff of the margin `2 ln 3`, since the loss
moves by at most the change of the margin, plus `rows + 8` rounded operations of the loss.

## ATemperatureDividesTheLogits

The loss of the `f = 2` set at `T = 2` equals the loss of the `f = 1` set at `T = 1` bit for bit,
since doubling a float is exact.

## TheFittedTemperatureLowersTheLoss

On the `f = 2` set the loss at the fitted temperature is below the loss at `T = 1`.

## TheSameSetGivesTheSameTemperature

Two fits of the `f = 3` set return the same double.

## ASetEveryRowOfWhichIsRightFitsTheLowestTemperature

Two rows of `[5, 0]` labelled 0 have no finite minimum; `ln T` is `ln K_MIN_TEMPERATURE` within the
bracket width `K_TEMPERATURE_LOG_TOLERANCE`.

## ASetEveryRowOfWhichIsWrongFitsTheHighestTemperature

The same rows labelled 1 give `ln K_MAX_TEMPERATURE` within the same width.

## TheFitRefusesASetThatDoesNotFit

One promise over seven rows: `fit_temperature` fails with `InvalidDescription` naming
`fit_temperature` for no row, one class, a logit count that is no rows of the classes, a label past
the classes (row 1) or below 0 (row 0), a NaN logit and an infinite logit.

## TheLossRefusesATemperatureOutOfRange

One promise over the temperatures 0, -1, NaN and infinity: `mean_negative_log_likelihood` fails
with `InvalidDescription`, "finite and positive".
