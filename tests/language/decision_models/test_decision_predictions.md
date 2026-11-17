# tests/language/decision_models/test_decision_predictions.cpp

Suites: `Small_DecisionPredictions`.

Tests `DecisionPredictions`, `choice_rows` and `yes_no_rows` of `modules/language/decision_models/include/SushiAI/language/decision_models/decision_predictions.hpp`. The
cases use predictions built by hand with `decision_prediction_rows.hpp` and need no device.

## TheRowsAreTheRecordsTheyHold

Empty predictions hold 0 rows; five hand-built rows (choice, yes-or-no, choice, number, yes-or-no)
hold 5.

## ChoiceRowsHoldTheLogitsAndLabelsOfTheChoicesInOrder

Over those five rows `choice_rows` returns the two choice rows in order: the logits 3, 1, 1, 3, the
labels 0 and 1, and 2 classes.

## YesNoRowsHoldZeroAndTheLogitWithTheAnswerAsLabel

`yes_no_rows` returns the rows {0, 2} and {0, -2} with the labels 1 and 0, and 2 classes.

## AValueAboveOneHalfIsAYes

A yes-or-no value of 0.75 is the label 1 and a value of 0.5 is the label 0.

## AHeadWithNoRowOfItsTypeHoldsNoRow

Predictions of one number row give no choice row, which still states its 3 classes, and no yes-or-no
row.

## PredictionsWhoseVectorsDisagreeInCountAreRefused

One `numbers` entry fewer than the rows makes both calls fail with `InvalidDescription`, naming the
`numbers`.

## PredictionsThatStateNoOptionAreRefused

Predictions with 0 options make `choice_rows` fail with `InvalidDescription`, naming the options.
