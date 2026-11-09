# tests/language/generation/test_generator.cpp


Suites: `Small_LanguageGenerationGenerator`, `Small_GeneratorContract`.

Tests the interface `Generator` of `modules/language/generation/include/SushiAI/language/generation/generator.hpp` through `InMemoryGenerator`, and the
contract of the three generators of the module (`InMemoryGenerator`, `WindowGenerator` over an
`InMemoryWindowLogits` and `CachedGenerator` over an `InMemoryOnePositionStep`). No case needs a
device. The device cases of the two loops are in `test_window_generator.cpp` and
`test_cached_generator.cpp`.

## ACallerContinuesAPromptThroughTheInterface

Through a `Generator&`, the prompt 1, 3 continued by three greedy tokens appends 4, 0, 1 after the 9 the vector already held.

## ARefusalLeavesTheGeneratedTokens

An empty prompt fails with `InvalidDescription` naming "no id", and the vector still holds only 9.

## DestroyingThroughTheBaseRunsTheDerivedDestructor

A `std::unique_ptr<Generator>` that owns an `InMemoryGenerator` runs its destructor when it is reset; the generator's witness reads true afterwards.

## APromptOrACountThatDoesNotFitIsRefused

The merge of the window and the cached generator's refusal cases. Every generator against every row: an empty prompt ("no id"), an id outside the vocabulary ("id 1", "0 to 31") and a count of 0 ("0 tokens"). Each fails with `InvalidDescription` and leaves `generated` empty.
