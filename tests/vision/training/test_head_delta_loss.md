# tests/vision/training/test_head_delta_loss.cpp


Suites: `Small_HeadDeltaLoss` and `Medium_HeadDeltaLossEndToEnd`.

Tests `emit_head_delta_loss` of `head_delta_loss.hpp`, which turns a head and a delta of the head's
shape into the root `-scale * sum(delta * head)`. The head is two samples on a 3 by 3 grid of 7
channels, 126 elements. The delta is an input leaf, so the gradient at the head is `-scale * delta`.

The `Small_` cases build the graph and read it on the host; none needs a device.

### TheResultIsOneElementScaledByMinusTheFactor

The result has one element, the head's element type, and is the output of the last node, a scale
by `-0.25`. The operation names and wiring of the four nodes are not asserted.

### TheValueOnTheHostIsMinusTheFactorTimesTheSumOfProducts

On `HostEvaluator`, a head of 2 and a delta of 3 over 126 elements give `-0.25 * 756 = -189`,
exactly.

### TheResultHasTheElementTypeOfTheHead

With a float64 head and delta, the result is float64.

### TheResultIsARootAutogradDifferentiates

`Autograd::differentiate` takes the result as root and returns one gradient of the head's shape;
the delta does not require a gradient.

### ADeltaOfAnotherShapeIsRefusedNamingBothShapes

A delta of `[2, 3, 3, 6]` fails with `InvalidDescription` naming "MUL needs equally shaped
operands", `[2, 3, 3, 6]` and `[2, 3, 3, 7]`, and the graph gains no node.

### ADeltaOfAnotherElementTypeIsRefusedNamingBothTypes

A float64 delta against a float32 head fails with `InvalidDescription` naming "MUL needs operands
of one element type", `float32` and `float64`, and the graph gains no node.

### TheGradientAtTheHeadIsMinusScaleTimesDelta

On an engine, the compiled step with a factor of 0.5 gives a gradient whose every element is
`-0.5` times the delta's, to 32 float32 unit roundoffs. Every third delta element is zero, as most
of a real head's are.

### TheValueIsMinusScaleTimesTheSumOfProducts

On an engine the root equals `-0.5` times the sum of the 126 products, summed in double, to 2048
float32 unit roundoffs.
