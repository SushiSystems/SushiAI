# tests/language/generation/test_sampler_section.cpp


Suites: `Small_GenerationSamplerBuild`, `Small_GenerationSamplerAbsent`, `Small_GenerationSamplerUnread`.

Tests `make_sampler` in `modules/language/generation/include/SushiAI/language/generation/sampler_section.hpp`. Every call passes the context "the task section
'sampler'". All three cases run as rows.

## EachTypeBuildsItsSampler

Five rows: greedy, temperature with and without its key, top-k and nucleus build a sampler whose name is the one of its class.

## AnUnknownTypeAndAnAbsentKeyAreRefused

Five rows: an unknown type (listing the four), a top-k without `k`, a nucleus without `p`, a section without `type`, and a section that is no object.

## AKeyTheTypeDoesNotReadIsRefused

Five rows: a key the type does not read (two), a `k` of 0, a `k` that is no number, and a `p` of 2.
