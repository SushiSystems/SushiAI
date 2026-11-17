# tests/core/autograd/rules/test_shape/test_rules.cpp


Suites: `Small_ShapeRules`.

Tests the table `K_SHAPE_RULES`. What the three rules compute is covered by `rules/test_shape/test_gradients.cpp`.

## TheTablePairsEachStructuralOperationWithItsRule

The table holds K_RESHAPE with `reshape_backward`, K_TRANSPOSE with `transpose_backward`, then
K_LAYOUT_CONVERT with `layout_convert_backward`.

The differentiability of the three operations and their backward rules are checked by
`EveryOperationIsDifferentiableAndHasARule` in `rules/test_all.cpp`.
