# tests/language/generation/test_window_generator.cpp


Suites: `Medium_Generation`, `Small_Generation`.

Tests `WindowGenerator` in `modules/language/generation/include/SushiAI/language/generation/window_generator.hpp`. The `Medium_` case runs a decoder built from a
model file on the device; the `Small_` case runs over an `InMemoryWindowLogits`. The extents of the
step are read by the other cases and are not tested alone, and the refusal case moved to the
`Small_GeneratorContract` suite of `test_generator.cpp`.

## ARunEqualsOneTokenAtATimeOverTheGrownPrompt

Twelve greedy tokens after the prompt 1, 2, 3 equal twelve runs of one token, each over the prompt grown by the tokens before it, and every token is in the vocabulary. The loop is in a fixture helper that stops at the first failed run. The count of compilations is not checked.

## ASeedFixesADrawnContinuation

Over an `InMemoryWindowLogits` of a window of 8 and a vocabulary of 32, two runs of 12 temperature-sampled tokens under one seed give the same tokens.
