# tests/language/decision_models/test_calibrate_decision_model.cpp

Suites: `Large_DecisionModel`.

Tests `calibrate_decision_model` of `modules/language/decision_models/include/SushiAI/language/decision_models/calibrate_decision_model.hpp` on a trained model, and what the
model answers. The suite builds a session from `model.json` and `train.json` of the example with the
rows of `Language::register_nodes` and `Train::register_objectives` (vocabulary 32, width 16, two
heads, one block, windows of 8 ids, 4 options, batches of 16), trains it for the 12 epochs the
training file states over 1 536 records, fits the temperatures on a split of 960 records and scores a
held-out split of 1 200. The splits are drawn from the seeds 11, 12 and 13. A question has few ids to
draw, so two splits can ask the same question, and the rule gives it the same answer in both.

`SetUpTestSuite` trains the session once for the cases of the suite. `Large_` runs one case in a
process, so each case trains once more. Nothing here has been run. Each bound below is unmeasured: it
is derived from the chance level of its head, written beside the constant in the test, and a first
run decides whether it stays.

## AnswersEachTypeOfQuestionOfTheHeldOutSplit

After training and calibration, over the held-out split: choice accuracy at least
`K_CHOICE_ACCURACY_FLOOR` (0.85), yes-or-no accuracy at least `K_YES_NO_ACCURACY_FLOOR` (0.90) and
the mean absolute error of the numbers at most `K_NUMBER_ERROR_BOUND` (0.15). Unmeasured: the choice
floor is 0.52 above the chance level of 1/3 over 2 to 4 options, the yes-or-no floor 0.40 above 1/2,
and the number bound about half of the 0.29 a model that answers 0 makes.

## TheCalibrationErrorAfterCalibrationIsAtMostTheBound

The expected calibration error over 15 bins of the choice head and of the yes-or-no head on the
held-out split, at the temperatures fitted on the calibration split, is at most `K_ECE_BOUND`
(0.05), the bound of the ruling LANGUAGE 6. Unmeasured: a split of 400 rows of a type puts about 0.03
of sampling noise into the error of a model that is calibrated, so the bound has little room.

## TheModelFileHoldsTheFittedTemperatures

`calibrate_decision_model` over a copy of `model.json` returns the temperatures it fitted, and the file
read back, with its extra inputs and its `from` strings, holds those two numbers. The values are copied through the writer, so the comparison is
exact.
