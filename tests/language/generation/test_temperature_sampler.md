# tests/language/generation/test_temperature_sampler.cpp


Suites: `Small_GenerationTemperatureShares`, `Small_GenerationTemperatureSampler`, `Small_GenerationTemperatureRefusal`.

Tests `TemperatureSampler` in `modules/language/generation/include/SushiAI/language/generation/temperature_sampler.hpp` against hand-worked distributions and
under a seed.

## TheSharesAreTheSoftmax

The merge of two cases as rows: logits 0, ln 2, 0 at temperature 1 give 1/4, 1/2, 1/4; logits 0, ln 2 at temperature 1/2 give 1/5 and 4/5.

## ASeedFixesTheSequence

Two runs of 64 draws under one seed give the same tokens. The claim that another seed gives another sequence has no promise row in the header and is not tested.

## MakeRefusesATemperatureThatIsNotPositiveAndFinite

Four rows: 0, -1, infinity and NaN each fail with `InvalidDescription`.
