# tests/language/decision_models/test_decision_model.cpp

Suites: `Medium_DecisionExampleFiles`.

Tests that the example files of `applications/examples/decision_model/` agree with
`decision_model.hpp` (`K_TEST_EXTENTS`) and with the roles the `decisions` dataset publishes. The
cases read `model.json` and `train.json` from `SA_REPOSITORY_ROOT` and write a small decision file
under a scratch directory, on the engine of the program.

## TheModelFileNamesTheRolesTheDatasetPublishes

`Medium_`. `model.json` lists the extra inputs `lengths`, `options`, `option_lengths` and
`option_valid`, in that order. For each, the dataset a split writes publishes a role of that name with
the shape and the element type the input states. The model input of `model.json` has the dataset's
sample shape and element type, and the one `decision_heads` row reads, in order, the model input and
those four inputs by name.

## TheTrainFileSumsOneTermForEachTypeOfQuestion

`Medium_`. The objective of `train.json` is a `sum` of three terms: `cross_entropy`,
`binary_cross_entropy` and `mean_squared_error`. Each term reads the output of its own name,
`choice`, `yes_no` or `number`, takes its targets from the role of the same name, and keeps column 0, 1
or 2 of the type mask.

## TheDataBlockStatesTheExtentsOfTheTests

`Medium_`. The data block of `train.json` is of the type `decisions` and states the window, the options
and the batch size of `K_TEST_EXTENTS`; the file states 12 epochs, the count the tests pass over the
training split.
