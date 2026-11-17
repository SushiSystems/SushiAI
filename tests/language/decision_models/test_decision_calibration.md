# tests/language/decision_models/test_decision_calibration.cpp

Suites: `Small_DecisionCalibration`.

Tests the temperatures of `modules/language/decision_models/include/SushiAI/language/decision_models/decision_calibration.hpp`: the fit over predictions and the place of the
two temperatures in a model file. The cases use values only; the model-file cases store into the
one-row model text `K_ONE_ROW_MODEL`. The hand-built predictions come from
`decision_prediction_rows.hpp`, which sits beside this file.

## TheTemperaturesSurviveTheModelFile

`store_calibration` into the parsed `K_ONE_ROW_MODEL`, `write_model`, a parse of the text and
`read_calibration` return the stored 1.25 and 0.5. The values are copied through a JSON number, and
the writer reads back an equal tree, so the comparison is exact.

## AFileWithNoTemperaturesReadsOne

A `"decision"` task with no `calibration` key reads as 1 and 1.

## ATaskOfAnotherTypeIsRefused

A file whose task is `next_token_generation` is refused by `store_calibration` and by
`read_calibration` with `InvalidDescription`.

## ATemperatureOutsideTheBoundsIsRefused

A temperature of 0 is outside `[K_MIN_TEMPERATURE, K_MAX_TEMPERATURE]` and is refused.

## TheFitHeatsAnOverconfidentModel

Eight choices and eight yes-or-no rows, each answered at a logit of 6 and right in 6 of 8. The
likelihood is greatest where the confidence is 3/4, so the fitted temperature of each head is above 1
and the expected calibration error of each is lower at the fitted temperature than at 1. This is a
metamorphic check; the fit itself is pinned in `tests/language/training/test_temperature_scaling.md`.

## ASplitWithNoYesNoRowIsRefused

Two choices and no yes-or-no row: `fit_decision_calibration` returns `InvalidDescription`.
