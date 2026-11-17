# tests/language/decision_models/test_decision_session.cpp

Suites: `Medium_DecisionSession`.

Tests `DecisionSession` of `modules/language/decision_models/include/SushiAI/language/decision_models/decision_session.hpp`. The case reads the example files of
`applications/examples/decision_model/` from `SA_REPOSITORY_ROOT`, writes a small decision file
under a scratch directory and hands the session registries that hold no row. The session trained
end to end is checked in `test_calibrate_decision_model.cpp`.

## ASessionRefusesAFileThatStatesAScheduleOrAStart

`Medium_`. A training file with a `schedule` and one with a `resume` path each fail
`DecisionSession::make` with `Unsupported`, naming the key, before any row of the model file is
looked up; the registries the cases pass are empty.
