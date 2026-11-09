# tests/language/generation/test_in_memory_generator.cpp


Suites: `Small_InMemoryGenerator`.

Tests `InMemoryGenerator` in `modules/language/generation/include/SushiAI/language/generation/in_memory_generator.hpp`, the `Generator` that runs no model.

## ThePromptIsContinuedWithSuccessorsModuloTheVocabulary

The prompt 1, 3 over five tokens is continued by 4, 0, 1 under a greedy sampler.

## TheWitnessReadsTrueOnceTheGeneratorIsDestroyed

The witness reads false while the generator lives and true after it is reset.
