# tests/layers/nn/test_softmax.cpp

That the softmax row normalises over the axis it names, last by default, moves another axis last and back through two `K_TRANSPOSE` nodes, refuses an axis outside the sample, and has the gradient central differences give.

Suites: `Small_SoftmaxValues`, `Small_Softmax`, `Small_SoftmaxRefusal`, `Medium_SoftmaxGradient`.

Tests `NN::Softmax` and `NN::make_softmax_node` in
`modules/layers/nn/include/SushiAI/nn/softmax.hpp`. The values cases evaluate the traced graph on
the host in double.

## Small_SoftmaxValues.TheValuesAreTheExponentialsOverTheirGroup

Each element is its exponential over the sum of the exponentials of the elements that share every
index but the one along the axis. The oracle is that formula over `std::exp`, and the bound is 12
rounded operations of double: the shift by the maximum, the exponential, a sum of at most three,
the logarithm, the difference and the final exponential. The rows are the last axis of a
[2, 3] value, the first axis of a [2, 3, 2] value spelled 0 and -2, its last axis, and the first
and the middle axis of a [2, 2, 3, 2] value.

## Small_Softmax.TheLastAxisEmitsNoTranspose

Over the last axis the graph holds one `K_LOG_SOFTMAX` and no `K_TRANSPOSE`, and the result has
the shape [2, 3, 4].

## Small_Softmax.AnotherAxisIsMovedLastAndBack

Over axis 0 of a [2, 3, 4] value the graph holds one `K_LOG_SOFTMAX`, two `K_TRANSPOSE` nodes and
no `K_MATMUL`, and the result has the shape [2, 3, 4].

## Small_Softmax.AnAxisOutsideTheRankFailsTheTrace

Axis 2 over a [2, 3, 4] value returns the input and records `InvalidDescription` naming "axis 2".

## Small_SoftmaxRefusal.TheFactoryRefusesTheRow

`InvalidDescription` with the context "layer 2 'softmax'" and a fragment per row: axis 2 and axis
-3 over a rank-2 sample, an axis that is no integer, an unknown key ("'dim'") and a rank-0 input.

## Small_Softmax.TheDefaultAxisIsTheLastAndThePortKeepsTheShape

A row with no key over [4, 3] gives one port named "output" of the shape [4, 3] and one input.

## Small_Softmax.TheOutputShapeIsTheInputShapeForAnyAxisInRange

`output_shape` of axis 0 and of axis -1 over [5, 3] returns [5, 3].

## Medium_SoftmaxGradient.TheGradientOverAnInnerAxisMatchesCentralDifferences

The gradient of a loss over the softmax of axis 0 of a [2, 3, 2] input, reshaped to [2, 6],
matches central differences in FLOAT64.
