# tests/language/generation/test_in_memory_one_position_step.cpp


Suites: `Small_InMemoryOnePositionStep`.

Tests `InMemoryOnePositionStep` in `modules/language/generation/include/SushiAI/language/generation/in_memory_one_position_step.hpp`, the adapter of the
`OnePositionStep` port, on a host engine.

## ARunScoresTheSuccessorOfTheFedId

Fed the id 4 over five tokens, the step scores 1 at token 0 and 0 at token 1.

## TheFedIdsAndTheResetsAreKept

After a reset and two runs of the ids 2 and 0, the step lists those ids and one reset.

## ARunWithoutAModelInputIsRefused

A run fed nothing fails with `InvalidDescription`.

## TheOutputIsAnsweredByItsNameAlone

The output reads as float32 of [1, 1, 5]; another name fails naming it.

## TheStepReportsTheStateFlagItWasGiven

A step built stateless reports that it carries no state.
