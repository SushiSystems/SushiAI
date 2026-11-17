# tests/core/graph/lowering/test_matmul.cpp

Suites: `Medium_AttentionOperations`, `Small_MatmulLowerings` and `Small_MatmulRow`.

The file runs the rank-3 matrix product, the softmax over the last axis and the product's gradient on
an engine in FLOAT32. The softmax is built from `log_softmax` and lowered by the loss table; it sits
here because attention is its caller. The inputs follow a fixed pattern, `scale * ((i * stride) % 11 - 5)`.
Every result is compared with a host computation in double within `K_ABSOLUTE_TOLERANCE`, 32 unit
roundoffs of float.

## BatchedProductWithTransposedRightOperandMatchesHost

`matmul` of a `[2, 3, 4]` value with a `[2, 5, 4]` value, the right operand transposed, gives the
`[2, 3, 5]` product a host loop computes.

## SoftmaxOverTheLastAxisMatchesHost

The softmax of a `[2, 3, 3]` value equals the host softmax of each row of three.

## BatchedProductGradientsHaveTheOperandShapesAndMatchTheHost

The loss is the sum of every element of the same product, differentiated at both operands. The
gradients have the shapes `[2, 3, 4]` and `[2, 5, 4]`. The values equal the host sums: for the left
operand the sum of the right over j, for the right operand the sum of the left over i.

## TheTableHoldsExactlyTheRowsTheHeaderNames

`K_MATMUL_LOWERINGS` holds as many rows as `matmul.hpp` names (three).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.
