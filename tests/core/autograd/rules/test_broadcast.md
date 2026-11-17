# tests/core/autograd/rules/test_broadcast.cpp


Suites: `Medium_AutogradBroadcast` and `Medium_AutogradSumAlong`.

Tests the backward rules of the four broadcast operations and the sum along against central differences in FLOAT64, through the `GradientProbe` of `tests/support/gradient_probe.hpp`. The method is described in [`tests/core/autograd/README.md`](../README.md).

## TheBroadcastDifferentiatesAtEachOperand

One case per operation (add, sub, mul, div), per side (leading, trailing), per rank (2 and 3) and
per differentiated operand (the tensor and the vector): thirty-two cases. The tensor is [4, 3] or
[2, 4, 3]; the vector is a prefix ([4] or [2, 4]) or a suffix ([3] or [4, 3]) one axis short of it.
The result is read as rows of three against random targets. Where the vector is the divisor it holds
0.5 to 1.7, so the quotient stays away from zero.

## TheSumAlongDifferentiatesAtEachSideAndRank

The sum of a [4, 3] or [2, 4, 3] value down to a prefix or a suffix, on each side at each rank,
read as one column against random targets. The rule is a `broadcast_add` of the gradient to zeros.
