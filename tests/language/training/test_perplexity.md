# tests/language/training/test_perplexity.cpp


Suites: `Small_LanguageTrainingPerplexity`.

Tests `Language::Training::perplexity` and `Language::Training::next_token_metrics` in
`modules/language/training/include/SushiAI/language/training/perplexity.hpp`. The values at 0,
ln 8 and ln 64 and the names of the two metrics are pinned by the cases of the same suite in
`test_next_token_targets.cpp`. The cases here pin the product rule and a loss out of range.

## ASumOfLossesMultipliesThePerplexities

`perplexity(ln 2 + ln 3)` equals `perplexity(ln 2) * perplexity(ln 3)` within 6e-12, and a larger
loss has a larger perplexity.

## AnOverflowingLossKeepsItsValue

`next_token_metrics(1e6)` reports the loss as exactly 1e6 and the perplexity as positive
infinity.
