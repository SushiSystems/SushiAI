# tests/core/autograd/rules/test_shape.cpp


Suites: `Small_ShapeRules`.

Tests the table `K_SHAPE_RULES`. What the two rules compute is covered by `rules/test_all.cpp`.

## TheTablePairsEachStructuralOperationWithItsRule

The table holds K_RESHAPE with `reshape_backward`, then K_LAYOUT_CONVERT with
`layout_convert_backward`.

The differentiability of both operations and their backward rules are checked by
`EveryOperationIsDifferentiableAndHasARule` in `rules/test_all.cpp`.
