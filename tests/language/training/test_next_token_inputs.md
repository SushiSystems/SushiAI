# tests/language/training/test_next_token_inputs.cpp

Suites: `Small_LanguageTrainingNextTokenInputs`, `Small_LanguageTrainingNextTokenInputsAccepted`.

Tests `Language::Training::check_next_token_inputs` in
`modules/language/training/include/SushiAI/language/training/next_token_inputs.hpp` (RW-98). The
span cases need no tensor and no engine. The table has five rows.

## EveryRowOfTheTableIsAnId

The ids 0, 4, 2, 1, 3, 0 pass, and so does an empty list.

## AnIdOutsideTheTableIsRefused

The check fails with `Incompatible` for an id of 5 at position 4 ("position 4 is 5", "holds 5
rows") and for an id of -1 at position 2 ("position 2 is -1"). Without the check the ids reach the
gather kernel.

## ATensorOfAnotherTypeIsRefused

A shape-only float32 tensor of six elements fails with `Incompatible` ("inputs are", "read as"),
before any element is read.
