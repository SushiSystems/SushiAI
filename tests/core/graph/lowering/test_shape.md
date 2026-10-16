# tests/core/graph/lowering/test_shape.cpp


Suites: `Small_ShapeLowerings`.

Tests `K_SHAPE_LOWERINGS`. The values of RESHAPE, LAYOUT_CONVERT and STOP_GRADIENT and the
refusal of an unbound leaf are covered in `tests/core/graph/test_lowering/test_hand_values.cpp` and
`tests/core/graph/test_lowering/test_host_values.cpp`, so this file pins the table alone.

## EachRowPairsItsOperationWithItsLowering

The table has four rows, in the order the header writes them. Each row names its operation, the
function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.
