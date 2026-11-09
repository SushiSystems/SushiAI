# tests/language/training/test_perplexity.cpp


Suites: `Small_LanguageTrainingPerplexity`, `Small_LanguageTrainingPerplexityOverflow`.

Tests `Language::Training::perplexity` and `Language::Training::next_token_metrics` in
`modules/language/training/include/SushiAI/language/training/perplexity.hpp`. The bounds come from `tools/testing/support/tolerance.hpp`: the
relative error of the perplexity against its exact value, measured in rounded operations of a
double. No bound is typed.

## ALossOfZeroHasAPerplexityOfOne

`perplexity(0)` is exactly 1.

## ThePerplexityIsTheExponentialOfTheLoss

`perplexity(ln 64)` is 64 within six rounded operations. The case moved here from
`test_next_token_targets.cpp`; its check of an infinite perplexity for a loss of 1e6 is the
overflow rows below.

## TheMetricsAreTheLossAndThePerplexity

`next_token_metrics(ln 8)` returns "loss" with exactly ln 8 and "perplexity" with 8 within six
rounded operations, neither marked as a fraction. The case moved here from
`test_next_token_targets.cpp`.

## ASumOfLossesMultipliesThePerplexities

`perplexity(ln 2 + ln 3)` equals `perplexity(ln 2) * perplexity(ln 3)` within six rounded
operations: the sum, two exponentials and the product.

## ALargerLossHasALargerPerplexity

`perplexity(ln 2)` is below `perplexity(ln 3)`. It is a second promise of the product rule case
and has its own test.

## AnOverflowingLossKeepsItsValue

For the losses 710, 1e6 and 1e300, `next_token_metrics` reports the loss as exactly that value
and the perplexity as positive infinity. The three losses are rows.
