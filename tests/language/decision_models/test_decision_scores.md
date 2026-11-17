# tests/language/decision_models/test_decision_scores.cpp

Suites: `Small_DecisionScores`.

Tests `score_decisions` of `modules/language/decision_models/include/SushiAI/language/decision_models/decision_scores.hpp` over predictions built by hand with
`decision_prediction_rows.hpp`. The cases use values only and need no device.

## TheScoresEqualTheHandCounts

Two choices (one right), three yes-or-no rows (two right) and two numbers with errors 0.25 and 0.5.
The accuracies are 1/2 and 2/3 and the mean error is 0.375, which is exact in float32.

## ATemperatureChangesNoAccuracy

The scores of one set at the temperatures 1 and (5, 3) have equal accuracies: a positive scale of
the logits changes no argmax and no sign.
