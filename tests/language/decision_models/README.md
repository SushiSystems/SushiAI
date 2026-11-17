# tests/language/decision_models

The tests of the module `language/decision_models`. The suite prefix of a case gives its ctest label.

## Test files

| Test file | Headers | Suites | Page |
| --- | --- | --- | --- |
| `test_decision_model.cpp` | `decision_model.hpp` | `Medium_DecisionExampleFiles` | [`test_decision_model.md`](test_decision_model.md) |
| `test_decision_set.cpp` | `decision_set.hpp` | `Small_DecisionSet`, `Small_DecisionSetRefusals` | [`test_decision_set.md`](test_decision_set.md) |
| `test_decision_calibration.cpp` | `decision_calibration.hpp` | `Small_DecisionCalibration` | [`test_decision_calibration.md`](test_decision_calibration.md) |
| `test_decision_scores.cpp` | `decision_scores.hpp` | `Small_DecisionScores` | [`test_decision_scores.md`](test_decision_scores.md) |
| `test_decision_predictions.cpp` | `decision_predictions.hpp` | `Small_DecisionPredictions` | [`test_decision_predictions.md`](test_decision_predictions.md) |
| `test_decision_session.cpp` | `decision_session.hpp` | `Medium_DecisionSession` | [`test_decision_session.md`](test_decision_session.md) |
| `test_calibrate_decision_model.cpp` | `calibrate_decision_model.hpp` | `Large_DecisionModel` | [`test_calibrate_decision_model.md`](test_calibrate_decision_model.md) |

## What the sizes use

The `Small_` suites use values only: the records of a set, hand-built predictions and the parsed
text of a one-row model file. The `Medium_` suites read the two example files of
`applications/examples/decision_model/` from `SA_REPOSITORY_ROOT`, writes a small decision file under
a scratch directory and compares the files with each other and with the roles the dataset publishes.
The `Large_` suite trains the model of those two files on the engine of the program: it writes three
decision files under a scratch directory, builds a `DecisionSession` from `model.json` and
`train.json` with the rows of `language/registration`, trains it, calibrates it on one split and scores
another. It is `Large_` because one case
trains a model for about a thousand steps; `Large_` runs one case in a process, so the three cases
of the suite train three times. No case has been run, and the bounds of the suite are unmeasured.

`SaTest::EngineTest` gives each case the one engine of the program, as in the other test folders. The
suite trains in `SetUpTestSuite`, which makes no device, queue or runtime: it asks the environment
for the engine it already holds.

`decision_prediction_rows.hpp` builds the hand-made predictions the calibration, score and prediction
cases share, and `decision_fixture.hpp` holds the example files, the splits and the trained session the
medium and large cases share. Both sit in this folder and no other test includes them.
