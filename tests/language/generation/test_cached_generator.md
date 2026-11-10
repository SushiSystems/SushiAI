# tests/language/generation/test_cached_generator.cpp


Suites: `Medium_CachedGeneration`, `Medium_CachedRefusal`, `Medium_CachedRing`, `Medium_CachedSlide`, `Small_CachedGeneration`.

Tests `CachedGenerator` in `modules/language/generation/include/SushiAI/language/generation/cached_generator.hpp` against `WindowGenerator`. The `Medium_` cases
run decoders built from model files on the device: a learned table (two blocks), a rotary ring of two
blocks and a rotary ring of one block, as rows. The `Small_` case runs over an
`InMemoryOnePositionStep`.

## AStepIsAcceptedWithTheLeavesItIsFed

Rows over the three decoders: `make` over the one-position step succeeds with the leaves of the blocks. It replaces the two cases that read the extents and the leaves of the step (TheGeneratorReadsTheExtentsOfTheStep, TheStepIsFedTheRingAndNoPosition).

## TheStateHoldsParametersOnly

Rows over the three decoders: every entry of a fresh state is a parameter and none is named for a rotary table. It replaces TheCachesAreNoEntryOfTheModelState and the rotary case of the same promise; the count against a constant is gone.

## APromptThatFitsGivesTheTokensOfTheWindowGenerator

Rows over the three decoders: the prompt 1, 2, 3 continued by four greedy tokens equals the window generator's.

## TheLogitsEqualThoseOfTheWindowAtTheLastPosition

Rows over the three decoders: after the prompt 4, 7, 1, 30, 2 the logits of the cached step equal the window's at position 4, through `expect_logits_near`.

## AContextThatSlidesGivesTheTokensOfTheWindowGenerator

Suite `Medium_CachedSlide`, rows over the table (12 tokens) and the one-block ring (26 tokens): the tokens equal the window generator's across the slides. The two-block ring is not a row: its second block reads keys and values cached from the first block's outputs, which attended to ids the window has since dropped, so past the first slide it differs from the window by design: the tokens agree up to the sixth. Its oracle is the wide cache of `ARingEqualsAWideCacheReadThroughItsLastPositions`. The counts of runs and compilations are not checked.

## ASecondPromptMatchesAFreshGenerator

Rows over the three decoders: a second prompt after a first one gives what a fresh generator gives and what the window generator gives.

Size (RW-166): the rows ran 12 and 26 tokens, and the two slowest took 7.7 s and 5.1 s on the debug build, over the 5 s of a `medium` test. The case takes its own token count, not that of the sliding run. The first prompt holds 4 ids and the window is 8, so the first run wraps the ring once, which is what leaves a state a second prompt could wrongly inherit, when it continues for window - 4 + 1 = 5 tokens. The second run holds 2 + 5 = 7 ids and stays inside the window, where every decoder, the two-block ring too, equals the window generator, so the case needs no slide. The three comparisons are unchanged.

## AStepOrARingThatDoesNotFitIsRefused

Four rows: a step that carries nothing, a misnamed mask leaf, a ring without its angle function, and a step fed both the position and the slot. Each fails with `InvalidDescription`.

## TheLogitsOfAOneBlockRingEqualTheWindowsAtEveryPosition

A one-block ring fed 30 ids gives logits equal to the window generator's over the same ids so far, at every position. The window is fed by the generator, not rebuilt by hand; the loop is in `expect_ring_logits_near_window`.

## ARingEqualsAWideCacheReadThroughItsLastPositions

A two-block ring generates the tokens of a 64-row cache fed by hand and read through its last 8 positions, over 26 tokens. The loop is in `wide_cache_tokens`.

## ASlotWrittenTwiceHoldsTheSecondRow

After eight positions and one more at position 8 (row 0 written twice), read through a band of one position, the logits equal those of a step that wrote row 0 once. The neighbour row is not read through the pool's tensors; its promise rests on the every-position case above.

## APromptLongerThanTheWindowKeepsItsLastIDs

Over an `InMemoryOnePositionStep` of a window of 8, the prompt 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11 feeds the last eight ids after one reset, and the one token generated is 12. It replaces the device case.
