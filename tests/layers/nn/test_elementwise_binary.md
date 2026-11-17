# tests/layers/nn/test_elementwise_binary.cpp

That the mul, maximum, sub and div rows take two inputs of one shape or a leading prefix, emit the node of their operation, give the values a hand calculation gives, and have the gradient central differences give.

Suites: `Small_ElementwiseBinary`, `Small_ElementwiseBinaryBroadcast`, `Small_ElementwiseBinaryNoncommuting`, `Small_ElementwiseBinaryMul`, `Small_ElementwiseBinaryMaximum`, `Small_ElementwiseBinaryValues`, `Small_ElementwiseBinaryPrefixValues`, `Medium_ElementwiseBinaryGradient`.

Tests `NN::ElementwiseBinary` and `NN::make_binary_node` in
`modules/layers/nn/include/SushiAI/nn/elementwise_binary.hpp`. The four operations are the rows of
one table, `Mul`, `Maximum`, `Sub` and `Div`, and each parametrised case runs once per row unless
it says otherwise.

## Small_ElementwiseBinary.CreateKeepsTheSharedShapeAsThePort

Over two [8, 6, 3] shapes the node has one port, named "output", of the shape [8, 6, 3], and
`input_count()` is 2.

## Small_ElementwiseBinary.CreateRejectsAnInputCountOtherThanTwo

One shape and three shapes fail with `InvalidDescription`; the message holds the row type,
"takes 2 inputs" and the given count.

## Small_ElementwiseBinary.CreateRejectsAnInputOfRankZero

An empty shape as input 1 fails with "input 1" and "rank 1 or more".

## Small_ElementwiseBinary.CreateRejectsShapesThatShareNoPrefix

[8, 6, 3] with [6, 3] fails with both shapes and "leading prefix" in the message.

## Small_ElementwiseBinary.ForwardEmitsOneNodeReadingBothInputsInOrder

Over two [2, 8, 6, 3] inputs given as b then a, `forward` adds one node. Its operation is the
row's (`K_MUL`, `K_MAXIMUM`, `K_SUB` or `K_DIV`), its inputs are b then a, its one output is the
value `forward` returned, and that value has the shape [2, 8, 6, 3].

## Small_ElementwiseBinary.ForwardRejectsAnInputCountItWasNotBuiltOver

`forward` over one value fails with "input count is 2" and "given is 1", returns no output and
adds no node.

## Small_ElementwiseBinary.TheFactoryTakesNoKey

`make_binary_node` over two [4, 3] shapes and an empty row returns an `ElementwiseBinary`. A row
with the key "alpha" fails with the context, "'alpha'" and "no key".

## Small_ElementwiseBinary.TheRowNameIsTheModelFileType

`binary_row_name` returns "mul", "maximum", "sub" and "div" for the four operations.

## Small_ElementwiseBinaryBroadcast.TheSecondInputMayBeALeadingPrefix

Rows Mul, Sub and Div. [8, 6, 3] with [8, 6] gives a port of [8, 6, 3].

## Small_ElementwiseBinaryBroadcast.ForwardEmitsOneLeadingBroadcastNode

Rows Mul, Sub and Div. Over a [2, 8, 6, 3] value and a [2, 8] value, `forward` adds one node of
`K_BROADCAST_MUL`, `K_BROADCAST_SUB` or `K_BROADCAST_DIV`, whose inputs are the whole value then
the prefix value and whose side attribute is `Leading`. The result has the shape [2, 8, 6, 3].

## Small_ElementwiseBinaryNoncommuting.ThePrefixFirstIsRefused

Rows Sub and Div. [8] with [8, 6, 3] fails with "leading prefix", since a difference and a
quotient depend on the order.

## Small_ElementwiseBinaryMul.ThePrefixMayComeFirst

A product over [8] and [8, 6, 3] has the port [8, 6, 3].

## Small_ElementwiseBinaryMul.APrefixFirstIsEmittedWithTheWholeInputFirst

`forward` of that product over the values vector then whole adds a `K_BROADCAST_MUL` whose inputs
are whole then vector.

## Small_ElementwiseBinaryMaximum.AShorterSecondInputIsRefused

A maximum over [8, 6, 3] and [8] fails with "maximum takes equal shapes only", since the graph has
no broadcast maximum.

## Small_ElementwiseBinaryMul.TheStateWalkVisitsNothing

`walk_state` calls neither the parameter visitor nor the buffer visitor.

## Small_ElementwiseBinaryValues.TheValuesOverEqualShapesMatchTheHandResults

The graph is evaluated on the host in double. With a = {7, -2, 5, 8, 3, -6} and b = {2, 4, -5, 8,
-1, 3} over [2, 3], the results worked by hand are

| Row | Result |
| --- | --- |
| Mul | 14, -8, -25, 64, -3, -18 |
| Maximum | 7, 4, 5, 8, 3, 3 |
| Sub | 5, -6, 10, 0, 4, -9 |
| Div | 3.5, -0.5, -1, 1, -3, -2 |

Every result is a dyadic rational, so the comparison bound is zero and each element is exact.

## Small_ElementwiseBinaryPrefixValues.TheValuesOverAPrefixMatchTheHandResults

Rows Mul, Sub and Div. With a = {1, 2, 3, 4, 5, 6, 7, 8} over [2, 2, 2] and v = {2, 4, 1, -2} over
[2, 2], element [n, i, j] meets v[n, i]. The results worked by hand are 2, 4, 12, 16, 5, 6, -14,
-16 for Mul, -1, 0, -1, 0, 4, 5, 9, 10 for Sub and 0.5, 1, 0.75, 1, 5, 6, -3.5, -4 for Div,
each exact.

## Medium_ElementwiseBinaryGradient.TheFirstInputGradientMatchesCentralDifferences

The gradient of the loss with respect to the first input of a [2, 3] pair, at the point 7.3, -2.2,
5.1, 8.4, 3.2, -6.1 against the fixed second operand 2, 4, -5, 8, -1.5, 3, matches central
differences in FLOAT64. Every first element is at least 0.3 from its partner, so the maximum never
sits on a tie.

## Medium_ElementwiseBinaryGradient.TheSecondInputGradientMatchesCentralDifferences

The same check for the second input, at the second operand, against the fixed first operand.
