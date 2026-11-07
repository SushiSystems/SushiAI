# tests/core/autograd/rules/test_elementwise.cpp


Suites: `Small_ElementwiseRules`, `Small_ElementwiseRuleRow` (one case per expected operation).

Tests the table `K_ELEMENTWISE_RULES`. The arithmetic of each rule is covered by
`rules/test_all.cpp` and `test_backward.cpp`; this file pins that each operation reaches the rule
`modules/core/autograd/include/SushiAI/autograd/rules/elementwise.hpp` declares for it.

## EveryRowPairsItsOperationWithItsOwnRule

One case per expected operation. The table holds the operation in exactly one row, and that row
carries the rule declared for it. The twelve pairs are K_ADD with `add_backward`, K_SUB with `sub_backward`,
K_MUL with `mul_backward`, K_SCALE with `scale_backward`, K_EXP with `exp_backward`,
K_ACCUMULATE with `accumulate_backward`, K_DIV with `div_backward`, K_MINIMUM with
`minimum_backward`, K_MAXIMUM with `maximum_backward`, K_ABS with `abs_backward`, K_ATAN with
`atan_backward` and K_SELECT with `select_backward`. A swap of two rules fails two cases.

## TheTableHoldsOneRowPerExpectedOperation

The table has twelve rows, as many as the expected pairs, so a row with no expected pair fails.

The differentiability of every operation of every rule table, and its backward rule, are checked
by `EveryOperationIsDifferentiableAndHasARule` in `rules/test_all.cpp`.
