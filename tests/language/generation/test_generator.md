# tests/language/generation/test_generator.cpp


Suites: `Small_LanguageGenerationGenerator`.

Tests that a prompt is continued through the interface `Generator` of
`modules/language/generation/include/SushiAI/language/generation/generator.hpp` alone. The cases
use a fake and need no device. The two generators of the module are tested in
`test_window_generator.cpp` and `test_cached_generator.cpp`.

## LargestSampler

A sampler that picks the token of the largest logit and draws nothing.

## SuccessorGenerator

A fake of a window of 4 and a vocabulary of 5. Each step scores the token after the last one
highest, modulo 5, and appends the sampled token. It refuses an empty prompt with
`InvalidDescription` and sets a flag when it is destroyed.

## ACallerContinuesAPromptThroughTheInterface

The prompt 1, 3 continued by three tokens appends 4, 0, 1 after the 9 the vector already held.
`window()` and `vocabulary()` read 4 and 5 through the base.

## ARefusalLeavesTheGeneratedTokens

An empty prompt fails with `InvalidDescription` naming "at least one id", and the vector still
holds only 9.

## DestroyingThroughTheBaseRunsTheDerivedDestructor

A `std::unique_ptr<Generator>` that owns the fake runs the fake's destructor when it is reset.
