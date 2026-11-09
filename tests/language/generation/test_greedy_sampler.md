# tests/language/generation/test_greedy_sampler.cpp


Suites: `Small_GenerationGreedyPick`, `Small_GenerationGreedySampler`, `Medium_GenerationGreedySamplerDeathTest`.

Tests `GreedySampler` in `modules/language/generation/include/SushiAI/language/generation/greedy_sampler.hpp`. Every generator is seeded with 20261011.

## ThePickIsTheLargestLogitAndTheFirstOfEquals

Three rows: 0.5, 2, -1 picks 1; 0.5, 2, 2, -1 picks 1; the single logit -3 picks 0.

## APickDrawsNothing

After one `sample`, the generator compares equal to a second one built from the same seed and never used.

## NoLogitIsFatal

`sample` over an empty span stops the process with a report containing "no logit". It is `Medium_` because a death test starts a child process.
