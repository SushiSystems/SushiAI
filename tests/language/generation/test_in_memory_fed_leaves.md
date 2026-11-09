# tests/language/generation/test_in_memory_fed_leaves.cpp


Suites: `Small_InMemoryFedLeaves`.

Tests `InMemoryFedLeaves` in `modules/language/generation/include/SushiAI/language/generation/in_memory_fed_leaves.hpp`, the adapter of the `FedLeaves` port.

## TheNamesComeBackInTheOrderGiven

Two leaves list "first" then "second".

## ALeafHoldsTheFactsItWasGiven

"second" reads back as [1, 3] float32.

## AnAbsentLeafIsRefused

An absent name fails with `InvalidDescription` naming it.
