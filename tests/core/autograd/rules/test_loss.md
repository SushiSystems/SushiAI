# tests/core/autograd/rules/test_loss.cpp


Suites: `Small_LossRules`.

Tests the table `K_LOSS_RULES` and `starts_from_implicit_seed`. How the two rules differentiate,
fused and unfused, is covered by `test_backward.cpp`.

## TheTablePairsEachLossOperationWithItsRule

The table holds K_LOG_SOFTMAX with `log_softmax_backward`, then K_NLL_LOSS with
`nll_loss_backward`, and `Autograd::has_backward_rule` is true for both.

## OnlyTheNLLLossStartsFromTheImplicitSeed

`starts_from_implicit_seed` is true for K_NLL_LOSS and false for K_LOG_SOFTMAX, K_ADD and
K_SCALE. A `static_assert` pins that it is usable at compile time.
