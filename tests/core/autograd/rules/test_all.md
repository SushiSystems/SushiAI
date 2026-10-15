# tests/core/autograd/rules/test_all.cpp


Suites: `Medium_RuleTable`.

Tests that every domain's backward rules reach the combined table exactly once.

## EveryDomainsRulesReachTheCombinedTable

`modules/core/autograd/include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

## NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

## TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

## NoGradientBarrierHasARule

For every row of `Ops::K_ALL` with `blocks_gradient` set, `Autograd::has_rule` and
`Autograd::has_backward_rule` are both false. The loop counts the rows it checked and expects
4, so it cannot pass by checking none.

## TheComparisonIsABarrierWithoutARule

`Ops::K_LESS_EQUAL` has `differentiable` false and `blocks_gradient` true, and
`Autograd::has_rule` is false for it.

## EveryActivationRowHasARuleForItsForwardIdentityOnly

For every row of `Ops::K_ACTIVATIONS`, `Autograd::has_rule` is true for the `forward` identity
and false for the `backward` one. `Autograd::K_ACTIVATION_RULES` has as many entries as
`Ops::K_ACTIVATIONS` has rows.

## BatchNormFoldHasNoRuleAndBatchNormKeepsIts

`Autograd::has_rule` is true for K_BATCHNORM2D and false for K_BATCHNORM_FOLD, `has_backward_rule`
is false for K_BATCHNORM_FOLD, and `Autograd::K_NORM_RULES` has two entries.

## LayerNormHasItsRuleAndItsGradientHasNone

`has_rule` is true for LAYER_NORM. LAYER_NORM_BACKWARD is not differentiable and has no rule.

## LogSoftmaxHasItsRuleAndItsGradientHasNone

`K_LOSS_RULES` has two rows. LOG_SOFTMAX is differentiable and has a rule.
LOG_SOFTMAX_BACKWARD is not differentiable and has none.

## TheCausalMaskHasItsRule

`K_LINALG_RULES` has six rows and `has_rule` is true for CAUSAL_MASK.

## DropoutHasTheRuleOfTheRandomDomain

`K_RANDOM_RULES` has one row. DROPOUT is differentiable and has a rule.

## TheGatherAndTheScatterAddDifferentiateIntoEachOther

GATHER_ROWS and SCATTER_ADD_ROWS are both differentiable and both have a rule.

## TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone

`K_SPATIAL_RULES` has nine rows and `has_rule` is true for DEPTHWISE_CONV2D. Its two backward
identities are not differentiable and have no rule.
