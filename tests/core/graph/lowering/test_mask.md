# tests/core/graph/lowering/test_mask.cpp

Suites: `Medium_CausalMaskOperations`, `Small_MaskLowerings` and `Small_MaskRow`.

The file runs the causal mask on an engine in FLOAT32 and HALF. The input follows a fixed pattern,
`scale * ((i * stride) % 11 - 5)`.

## HalfCausalMaskFillsTheUpperTriangleWithTheLowestFiniteValue

A FLOAT32 `[2, 4, 4]` input of the fixed pattern is cast to HALF, masked with the fill -1e9 and cast
back to FLOAT32. Every entry above the diagonal equals -65504, the lowest finite HALF, and no entry
is infinite. Every entry on and below the diagonal equals the input; the pattern values are
multiples of 0.25 within [-1.25, 1.25], which HALF holds exactly.

## TheTableHoldsExactlyTheRowsTheHeaderNames

`K_MASK_LOWERINGS` holds as many rows as `mask.hpp` names (one).

## EachRowPairsItsOperationWithItsLowering

The row names `K_CAUSAL_MASK`, the function the header declares for it and no scratch rule, and
`find_lowering` returns that function for the operation.
