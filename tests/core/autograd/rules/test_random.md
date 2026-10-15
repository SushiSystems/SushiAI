# tests/core/autograd/rules/test_random.cpp


Suites: `Unit_DropoutBackward`.

Tests the backward rule of DROPOUT. The graph cases apply `GraphBuilder::dropout` with the rate
0.5 and the seed 11 to a [3, 4] value and take the `nll_loss` of the `log_softmax` of the
result.

## DropoutHasARule

`Ops::K_DROPOUT` is differentiable, does not block the gradient and has a backward rule.
`Autograd::K_RANDOM_RULES` has one row, for DROPOUT, and its rule is
`Autograd::dropout_backward`.

## TheInputGradientIsTheGradientTimesTheSavedMask

With x as a parameter, the gradient of x is written by a MUL node of two inputs whose second
input is the mask the forward node saved, and it has the shape [3, 4]. One node reads the mask
and the graph holds one DROPOUT node.

## AnInputThatNeedsNoGradientAddsNoNode

With x as an input, the gradient is `K_NO_VALUE` and no node reads the saved mask.
