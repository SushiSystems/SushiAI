# tests/core/autograd/rules/test_norm.cpp


Suites: `Small_LayerNormBackward`.

Tests the nodes `Autograd::differentiate` appends for LAYER_NORM. The graph normalises a
[5, 3] value with a [3] gain and a [3] bias at the epsilon 1e-5 and takes the `nll_loss` of
the `log_softmax` of the result.

## TheForwardHasARuleAndItsGradientHasNone

`Ops::K_LAYER_NORM` is differentiable and has a backward rule. `Ops::K_LAYER_NORM_BACKWARD` is
not differentiable and has none.

## OneGradientNodeServesTheThreeOperands

With x as a parameter, differentiating at x, the gain and the bias adds one
LAYER_NORM_BACKWARD node of five inputs. Inputs 1 to 4 are x, the gain, the saved mean and the
saved inverse deviation, and its outputs are the three returned gradients, of the shapes
[5, 3], [3] and [3].

## AnInputThatNeedsNoGradientReceivesNone

With x as an input, the gradient of x is `K_NO_VALUE`, the gain has a gradient and the graph
holds one LAYER_NORM_BACKWARD node.
