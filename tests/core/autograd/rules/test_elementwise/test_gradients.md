# tests/core/autograd/rules/test_elementwise/test_gradients.cpp


Suites: `Medium_Autograd`, `Medium_AutogradBinary` and `Medium_AutogradPow`.

Tests the backward rules of add, sub, mul, scale, exp, accumulate, sqrt, log and pow against central differences in FLOAT64, through the `GradientProbe` of `tests/support/gradient_probe.hpp`. The method is described in [`tests/core/autograd/README.md`](../../README.md).

## ElementwiseAddAndSubtract

One case per operation over a `[4, 3]` variable and a `[4, 3]` input, `add` and `sub` with the
variable on the left. Each gradient is checked against central differences.

## ElementwiseSubtractFromTheRight

The variable is the right operand of `sub`, so the case covers dB = -dC.

## ElementwiseMultiply

The variable is the left operand of `mul` against a `[4, 3]` input.

## ElementwiseMultiplyFromTheRight

The case checks dB = dC * A, the half the left-hand check cannot reach. The rule is written
symmetrically, so a copy-paste slip that reads `inputs[1]` on both sides would leave the
left-hand test green.

## AccumulateInTheForwardGraph

K_ACCUMULATE has a backward rule that nothing else exercises. The two fan-out tests build an
K_ACCUMULATE chain in the backward graph, and `differentiate()` walks only the forward nodes, so
those chains are never differentiated themselves. Reaching the rule needs an K_ACCUMULATE the
caller put in the forward pass.

The lowering aliases the result onto the accumulator's own buffer, so running the step leaves
the variable holding `x + other`. Every probe rewrites the whole variable first, which makes
that harmless.

## Scale

A `[4, 3]` variable is scaled by -2.5.

## Exp

The gradient of `exp` over a `[4, 3]` variable.

## GradientAccumulationAtAFanOut

The variable feeds `tanh` and `sigmoid`, and their sum is the logits. The two branches make the
gradient of the variable an ACCUMULATE chain.

## SquareRootReadsItsForwardResult

A [4, 3] FLOAT64 value of 0.5 to 1.7 in steps of 0.1 goes through `sqrt`, and the full Jacobian of
the loss is compared with central differences. The inputs sit above 0.1, so the root is smooth over
a whole difference step. The rule reads the forward result: `scale(div(dy, y), 0.5)`.

## LogarithmDividesByItsInput

The same point through `log`; the rule is `div(dy, x)`.

## ThePowerDifferentiatesAtEachExponent

The same point through `pow` at the exponents 2, 0.5 and -0.5, one case each. The rule is
`scale(mul(dy, pow(x, e - 1)), e)`; the exponent -0.5 also exercises a negative `e - 1`.
