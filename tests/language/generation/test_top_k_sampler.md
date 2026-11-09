# tests/language/generation/test_top_k_sampler.cpp


Suites: `Small_GenerationTopKSampler`, `Small_GenerationTopKRefusal`.

Tests `TopKSampler` in `modules/language/generation/include/SushiAI/language/generation/top_k_sampler.hpp` against hand-worked distributions and under a seed.

## OnlyTheKLargestAreDrawnAndTheLowerIndexWinsATie

Logits 0, ln 2, 0, -5 with k = 2 draw tokens 0 and 1 in the shares 1/3 and 2/3, and tokens 2 and 3 never.

## KeepingOneTokenIsTheGreedyPick

k = 1 over 0.3, 0.2, 0.9, 0.9 always picks token 2, the lower index of the tie.

## AKAboveTheRowKeepsEveryToken

k = 50 over 0 and ln 2 keeps both tokens in the shares 1/3 and 2/3.

## ASeedFixesTheSequence

Two runs of 64 draws under one seed give the same tokens.

## MakeRefusesAKBelowOneAndABadTemperature

Two rows: k = 0 and a temperature of 0 each fail with `InvalidDescription`.
