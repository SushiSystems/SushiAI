# tests/language/generation/test_nucleus_sampler.cpp


Suites: `Small_GenerationNucleusMass`, `Small_GenerationNucleusSampler`, `Small_GenerationNucleusRefusal`.

Tests `NucleusSampler` in `modules/language/generation/include/SushiAI/language/generation/nucleus_sampler.hpp` against hand-worked distributions over the logits
ln 2, 0, 0 (probabilities 1/2, 1/4, 1/4) and under a seed. Shares are counted over 20000 draws and
compared through `Support::expect_shares`.

## TheFewestTokensThatReachTheMassAreDrawn

The merge of three cases as rows: a mass of 0.6 draws tokens 0 and 1 in the shares 2/3 and 1/3, a mass of 0.4 draws token 0 alone, and a mass of 1 keeps all three in the shares 1/2, 1/4, 1/4.

## ASeedFixesTheSequence

Two runs of 64 draws under one seed give the same tokens.

## MakeRefusesAMassOutsideItsRangeAndABadTemperature

Four rows: a mass of 0, of 1.5, of NaN, and a temperature of -2 each fail with `InvalidDescription`.
