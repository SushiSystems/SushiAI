# tests/core/autograd/rules/test_all.cpp


Suites: `Small_RuleTable`, `Small_RuleTableEntry`, `Small_RuleTableLegacyOperation`,
`Small_RuleTableBarrier` and `Small_RuleTableActivation`.

Tests that every domain's backward rules reach the combined table exactly once. What each rule computes is checked in the `test_gradients.cpp` files of the `test_<domain>/` folders beside it.

## EveryDomainsRulesReachTheCombinedTable

`modules/core/autograd/include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.
There are nine domains: elementwise, linear algebra, activation, loss, shape, spatial, norm,
random and broadcast.

## NoOperationHasTwoRules

One case per row of the combined table. Two entries for one identity is not a compile error, and
the first one silently wins. Which gradient runs would then depend on the order in which two
domains happened to be concatenated, so the operation of each row must appear exactly once.

## EveryOperationIsDifferentiableAndHasARule

One case per row of the combined table, so every domain is covered. The operation is declared
differentiable in `Ops::K_ALL`, and `Autograd::has_backward_rule` is true for it.

## TheRulesThatExistedBeforeTheSplitStillDo

One case per operation. The split is a move, not a rewrite. These fourteen rules were the table
before it, and every one must still resolve.

## NoGradientBarrierHasARule

One case per gradient barrier: ZERO, CONSTANT, LESS_EQUAL, STOP_GRADIENT and ARGMAX_ROWS. Each has
`blocks_gradient` set, is not differentiable, and both `Autograd::has_rule` and
`Autograd::has_backward_rule` are false for it. The comparison LESS_EQUAL is one of the rows.

## TheBarrierRowsNameEveryDeclaredBarrier

`Ops::K_ALL` holds five rows with `blocks_gradient` set, so the rows above cannot pass by naming
a different set from the one the operation table declares.

## EveryActivationRowHasARuleForItsForwardIdentityOnly

One case per row of `Ops::K_ACTIVATIONS`. `Autograd::has_rule` is true for the `forward` identity
and false for the `backward` one.

## TheActivationRuleTableHasOneRowPerActivation

`Autograd::K_ACTIVATION_RULES` has as many entries as `Ops::K_ACTIVATIONS` has rows.

## BatchNormFoldHasNoRuleAndBatchNormKeepsIts

`Autograd::has_rule` is true for K_BATCHNORM2D and false for K_BATCHNORM_FOLD, `has_backward_rule`
is false for K_BATCHNORM_FOLD, and `Autograd::K_NORM_RULES` has three entries.

## LogSoftmaxHasItsRuleAndItsGradientHasNone

`K_LOSS_RULES` has two rows. LOG_SOFTMAX is differentiable and has a rule.
LOG_SOFTMAX_BACKWARD is not differentiable and has none.

## TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone

`K_SPATIAL_RULES` has ten rows and `has_rule` is true for DEPTHWISE_CONV2D. Its two backward
identities are not differentiable and have no rule.

## TheStaticProofsOfTheFlagsHoldAtRunTimeToo

`every_differentiable_op_has_a_rule` and `no_gradient_barrier_has_a_rule`, which `rules/all.hpp`
proves with a `static_assert`, return true when called over `Ops::K_ALL` at run time. The build
already fails when either does not hold; this case names the two proofs in the test list.

## TheRootTheLogarithmAndThePowerHaveRulesAndTheArgmaxHasNone

SQRT, LOG and POW are declared differentiable and `Autograd::has_rule` is true for each.
ARGMAX_ROWS has `blocks_gradient` set and no rule.

## TheBilinearForwardHasItsRuleAndItsGradientHasNone

UPSAMPLE2D_BILINEAR is differentiable and has a rule; its gradient identity is neither.

## TheBroadcastsTheSumAlongAndTheAlongRowsHaveRules

`K_BROADCAST_RULES` has five rows. The four broadcasts, SUM_ALONG, TAKE_ALONG and PUT_ALONG are each
differentiable and have a rule.
