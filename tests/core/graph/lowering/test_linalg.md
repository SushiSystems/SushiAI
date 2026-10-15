# tests/core/graph/lowering/test_linalg.cpp


Suites: `Integration_AttentionOperations`.

The file runs the rank-3 matrix product, the softmax and the product's gradient on an engine in
FLOAT32. The inputs follow a fixed pattern, `scale * ((i * stride) % 11 - 5)`. Every result is
compared with a host computation in double within 1e-6.

## BatchedProductWithTransposedRightOperandMatchesHost

`matmul` of a `[2, 3, 4]` value with a `[2, 5, 4]` value, the right operand transposed, gives the
`[2, 3, 5]` product a host loop computes.

## SoftmaxOverTheLastAxisMatchesHost

The softmax of a `[2, 3, 3]` value equals the host softmax of each row of three.

## BatchedProductGradientEmitsRankThreeProducts

The loss is the sum of every element of the same product, differentiated at both operands. The
gradients have the shapes `[2, 3, 4]` and `[2, 5, 4]`, and the graph holds three MATMUL nodes, each
with a rank-3 output. The values equal the host sums: for the left operand the sum of the right over
j, for the right operand the sum of the left over i.
