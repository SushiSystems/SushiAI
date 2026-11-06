# tests/core/graph/lowering/test_activation.cpp


Suites: `Small_ActivationLowerings`, `Small_ActivationRow`, `Medium_ActivationLowerings`.

Tests `K_ACTIVATION_LOWERINGS` and two of its gradient rows. The forward activations and the SiLU
and softplus gradients are compared with host values in
`tests/core/graph/test_lowering/test_hand_values.cpp` and
`tests/core/graph/test_lowering/test_host_values.cpp`, so the medium case here covers the two
gradients that read the forward output instead of the input. Device results are FLOAT32 and are
compared with a host computation in double within `K_ABSOLUTE_TOLERANCE`, 32 unit roundoffs of float.

## TheTableHoldsExactlyTheRowsTheHeaderNames

The table has as many rows as the header names (fourteen).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## TanhAndSigmoidGradientsReadTheForwardOutput

`activation_backward` emits TANH_BACKWARD and SIGMOID_BACKWARD over the same upstream gradient
and the same forward output y, with y between 0 and 1. The step writes dy * (1 - y * y) for tanh
and dy * y * (1 - y) for the sigmoid. A row that read y as the input would give other values.
