# tests/language/generation/test_sampler.cpp


Suites: `Small_LanguageGenerationSampler`, `Small_LanguageGenerationSamplerContract`.

Tests `modules/language/generation/include/SushiAI/language/generation/sampler.hpp`: the generator `RandomGenerator` a sampler draws from, and the
`Sampler` contract over the four samplers of the module, each held behind its interface. The
test-local `LargestSampler` and `DrawingSampler` are gone: the real samplers are the adapters.

## TheGeneratorYieldsTheValueTheStandardFixes

The 10000th draw of a default-seeded `RandomGenerator` is 9981545732273789042, the value the C++ standard states for `std::mt19937_64`.

## APickTakesAtMostOneDrawFromTheGenerator

Rows over greedy, temperature, top-k and nucleus: after one pick through a `const Sampler&` the generator equals a copy advanced by the draws the row states (0 for greedy, 1 for the others). It replaces ASamplerThatDrawsNothingLeavesTheGenerator.

## OneSeedGivesOneTokenAndOneGeneratorState

Rows over the same four samplers: two generators seeded 1234 lead the sampler to the same token over seven logits and are equal afterwards. It replaces OneSeedRepeatsTheTokensOfADrawingSampler.
