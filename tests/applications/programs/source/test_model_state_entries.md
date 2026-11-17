# tests/applications/programs/source/test_model_state_entries.cpp

Suites: `Small_ModelStateEntries`.

Tests `model_state_entries` of `applications/programs/source/model_state_entries.hpp`, the listing
of the state records a model file states, built without an engine. The model is the file of the tiny
run, `tests/applications/programs/fixtures/tiny_run/tiny_model.json`: a linear row of 8 units over 6
features, a gelu row and a linear row of 4 units. The cases allocate no tensor.

## EachLinearRowListsItsWeightAndItsBiasInRowOrder

The listing holds four records, since the gelu row holds none: the weight [8, 6], the bias [8], the
weight [4, 8] and the bias [4], in the order of the rows.

## EveryRecordIsAFloat32ParameterUnderItsOwnName

Each record is a parameter of element type float32, and no two records share a name.

## ARegistryWithoutTheRowsOfTheFileRefusesTheModel

With a registry that holds no row, the build of the model fails and no listing is returned.
