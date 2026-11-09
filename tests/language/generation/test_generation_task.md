# tests/language/generation/test_generation_task.cpp


Suites: `Small_GenerationTask`, `Small_GenerationTaskRefusal`, `Small_GenerationTaskCache`.

Tests `read_generation_task` in `modules/language/generation/include/SushiAI/language/generation/generation_task.hpp`, the reader of the
`next_token_generation` task section. Every call passes the context "the task section". The
refusals and the cache flag run as rows.

## EveryKeyIsRead

A section with every key reads back with those values and a sampler named `K_TOP_K_SAMPLER`.

## TheOptionalKeysHaveDefaults

A section with `vocabulary` and `window` only reads 64 tokens and seed 0, and its sampler picks the largest logit of 0.1, 0.9, 0.5 and draws nothing. The checks are against stated values and behaviour, not against the header constants.

## AKeyThatDoesNotFitIsRefused

Eight rows, each a section with one wrong key (the type, an absent `vocabulary` or `window`, a `window`, `tokens` or `seed` out of range, a `prompt` key, a sampler of the type "beam"). Each fails with `InvalidDescription` and two substrings.

## TheCacheIsReadAndDefaultsToTrue

Three rows: no `cache` key reads true, `false` reads false and `true` reads true.

## ACacheThatIsNoFlagIsRefused

`"cache": 1` fails with `InvalidDescription` naming the section, `'cache'` and "a boolean".
