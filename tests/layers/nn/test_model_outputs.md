# tests/layers/nn/test_model_outputs.cpp


Suites: `Unit_ModelOutputs`, `Unit_ModelOutputsDeathTest`.

Tests ModelOutputs: publishing named values and reading them back.

This file needs no device. `NN::ModelOutputs` in `modules/layers/nn/include/SushiAI/nn/model_outputs.hpp` is
the set of named `ValueID`s one trace of a model publishes. `add()` returns a `Status`, and
`at()` and `only()` return a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelOutputs` is final,
copy-constructible and nothrow move-constructible.

## boxes_and_scores

`boxes_and_scores` returns a set holding "boxes" as value 4 and "scores" as value 9, added in
that order.

## add_without_value

`add_without_value` adds "boxes" with `Graph::K_NO_VALUE`. It is the statement of the death
test.

## StartsEmpty

A new set's `all()` is empty.

## AllKeepsTheOrderOfAdd

After adding "scores", "boxes" and "features", `all()` holds those three entries in that order
with their ids 9, 4 and 0. The order is the order of the calls, not of the names or the ids.

## AddRejectsADuplicateName

A second `add("boxes", ...)` returns `InvalidDescription` with a message containing "'boxes'"
and "boxes, scores". The set still holds two entries and `at("boxes")` is still 4.

## AddRejectsAnEmptyName

`add("", ...)` returns `InvalidDescription` with "needs a name" and adds nothing.

## AddStopsOnNoValue

The case is in the suite `Unit_ModelOutputsDeathTest`. `add("boxes", K_NO_VALUE)` stops the
process with a report containing "'boxes'" and "no value".

## AtReturnsTheNamedValue

`at("boxes")` is 4 and `at("scores")` is 9.

## AtDoesNotMatchAPrefix

`at("box")` and `at("boxes.")` both fail with `InvalidDescription` and the name that was asked
for.

## AtOnAMissingNameListsTheNames

`at("masks")` fails with a message containing "'masks'" and "boxes, scores".

## AtOnAnEmptySetSaysNone

`at("boxes")` on an empty set fails with a message containing "'boxes'" and "none".

## OnlyReturnsTheSingleValue

`only()` on a set of one entry returns that entry's id, 12.

## OnlyFailsOnZero

`only()` on an empty set fails with a message containing "exactly one", "has 0" and "none".

## OnlyFailsOnTwo

`only()` on a set of two fails with a message containing "exactly one", "has 2" and "boxes,
scores".
