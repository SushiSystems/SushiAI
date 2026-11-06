# tests/core/graph/lowering/test_fused.cpp


Suites: `Small_FusedLowerings`, `Small_FusedRow`, `Medium_FusedLowerings`.

Tests `K_FUSED_LOWERINGS` and the GEMM_BIAS_RELU row. The fusion pass itself is tested in
`tests/core/graph/test_fusion.cpp`; here a fused node is run. The medium case passes the file-local
`TakeTheMatch` selector, which takes every match whole, so the fused row records whatever the
device is. Results are FLOAT32, compared with a host computation in double within `K_ABSOLUTE_TOLERANCE`, 256 unit
roundoffs of float.

## TheTableHoldsExactlyTheRowsTheHeaderNames

The table has as many rows as the header names (six).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation, the
function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## TheGEMMWithBiasAndReLUMatchesTheHost

A [2, 3] input times a [4, 3] weight read transposed, plus a [4] bias, then a ReLU, is fused into
one GEMM_BIAS_RELU node. The step records one selection, which chose that form, and writes
relu(x * w^T + b). The inputs put sums on both sides of zero, so the ReLU clips some entries.
