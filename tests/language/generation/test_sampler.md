# tests/language/generation/test_sampler.cpp


Suites: `Small_LanguageGenerationSampler`.

Tests `modules/language/generation/include/SushiAI/language/generation/sampler.hpp`: the
generator `RandomGenerator` a sampler draws from and the interface `Sampler`. The cases use two
fakes and need no device. The four samplers of the module have their own test files.

## LargestSampler

Picks the token of the largest logit and draws nothing.

## DrawingSampler

Picks the token at one draw of the generator modulo the vocabulary.

## TheGeneratorYieldsTheValueTheStandardFixes

The 10000th draw of a default-seeded `RandomGenerator` is 9981545732273789042, the value the C++
standard states for `std::mt19937_64`. A run therefore repeats on every platform.

## ASamplerThatDrawsNothingLeavesTheGenerator

Through a `const Sampler&`, the fake picks token 1 of the logits 0.5, 2.0, -1.0, its name reads
"largest", and the generator seeded 42 equals its copy taken before the call.

## OneSeedRepeatsTheTokensOfADrawingSampler

Two generators seeded 1234 lead the drawing fake to the same token on each of eight draws over
seven logits, and they are equal afterwards.
