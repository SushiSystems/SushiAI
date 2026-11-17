# tests/core/autograd/rules/test_shape/test_gradients.cpp


Suites: `Medium_Autograd`.

Tests the backward rules of reshape, transpose and layout convert against central differences in FLOAT64, through the `GradientProbe` of `tests/support/gradient_probe.hpp`. The method is described in [`tests/core/autograd/README.md`](../../README.md).

## ReshapeMovesItsGradientBackUnchanged

A reshape moves no data, so its gradient is its incoming gradient in the original shape. The
numeric check is still worth having: reshaping back to the input shape and reshaping back to
the output shape are both one line, and only one is right.

The fused K_LOG_SOFTMAX + K_NLL_LOSS rule's closed form assumes a one-hot target row
(`nll_loss_backward` says it is "valid because the targets are one-hot"). Noise in the targets
would make the analytic gradient disagree with the numeric one for reasons that have nothing
to do with K_RESHAPE.

## TransposeMovesItsGradientBackThroughTheInversePermutation

A permutation of three per-sample axes, (1, 2, 0), is not its own inverse, which is (2, 0, 1). A
rule that reused the forward permutation on the gradient would return a gradient of the wrong
shape here, and one that copied the gradient back without permuting it would have the right
shape in the wrong places. The loss is a fused log-softmax over the flattened result with a
one-hot target, so it depends on where each element sits, and the central differences find any
misplaced element. The input shape [2, 3, 4, 5] has no two equal per-sample extents, so every
misplacement changes a shape or a value.

## LayoutConvertsGradientIsThePermutedGradient

The case uses a single `layout_convert`, so the loss depends on the permuted element order. A
round trip would not do. A rule that reshaped dy back to the operand's shape without permuting
it has the right magnitude in the wrong place, and it still satisfies every shape check
`Differentiator::contribute` makes. A round trip cannot tell the two apart, because any
involutive, shape-correct pair of backward hops composes to the identity.

CV-4 forces every node that is neither a K_LEAF nor a K_LAYOUT_CONVERT to agree with its operand
on layout. A variable therefore cannot reach `log_softmax` by converting one way and stopping:
`graph::finalise()` would refuse the mismatched `log_softmax` output before any gradient ran.
CV-4 exempts a leaf, which has no operands to disagree with, so its layout tag is free. This
is also how the operation is used in practice. K_LAYOUT_CONVERT exists for boundary interop,
such as loading an NCHW checkpoint, where the source data carries the other tag and no
mid-graph value does. The case tags the variable NCHW before its one K_LAYOUT_CONVERT node and
reshapes straight into `log_softmax`. That reaches a CV-4-legal NHWC consumer with no round
trip, so the permutation is what the loss depends on.

## LayoutConvertRoundTripComposesBothDirections

The two directions composing to the identity is fair to check. It cannot tell a correct
permutation from a same-shape non-permutation, for the reason given under the single-hop case,
so it is a secondary check and not the proof.
