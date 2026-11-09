# tests/language/generation/test_in_memory_window_logits.cpp


Suites: `Small_InMemoryWindowLogitsHot`, `Small_InMemoryWindowLogits`.

Tests `InMemoryWindowLogits` in `modules/language/generation/include/SushiAI/language/generation/in_memory_window_logits.hpp`, the adapter of the `WindowLogits`
port.

## APositionScoresTheSuccessorOfItsId

Rows over the window 1, 4, 0 of five tokens: the positions score 1 at 2, 0 and 1.

## TheWindowsItWasRunOnAreKept

Two runs leave the two windows in order.

## AWindowOfAnotherLengthIsRefused

A window of two ids fails with `InvalidDescription`.
