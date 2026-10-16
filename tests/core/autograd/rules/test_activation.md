# tests/core/autograd/rules/test_activation.cpp


Suites: `Small_ActivationRules`.

Tests the table `K_ACTIVATION_RULES`. What the rule computes for each activation is covered by
`rules/test_all.cpp` and `test_backward.cpp`; this file pins the table itself.

## TheTableListsTheSevenActivations

The table holds K_RELU, K_GELU, K_SIGMOID, K_TANH, K_LEAKY_RELU, K_SILU and K_SOFTPLUS in that
order, and each is declared differentiable in `Ops`.

## OneRuleServesEveryActivation

Every row points to `activation_backward`, and `Autograd::has_backward_rule` is true for every
operation in the table.
