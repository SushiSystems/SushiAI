# tests/core/graph/lowering/test_row_index.cpp

Suites: `Medium_RowIndexOperations`, `Small_RowIndexLowerings` and `Small_RowIndexRow`.

The cases run the row operations on an engine. Their values are small integers, so every
result is exact in FLOAT32 and is compared with `EXPECT_EQ`. The `[4, 3]` table holds 1 to 12, so
row r is (3r + 1, 3r + 2, 3r + 3).

## TheTableHoldsExactlyTheRowsTheHeaderNames

`K_ROW_INDEX_LOWERINGS` holds as many rows as `row_index.hpp` names (five).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## GatherRowsCopiesTheRowsTheIndexNames

The INT32 index 2, 0, 2, 3, 1 over the table gives its rows 2, 0, 2, 3 and 1, in that order.

## GatherRowsReadsAnInt64Index

The same call with an INT64 index gives the same fifteen values.

## GatherRowsOverARankTwoIndexKeepsItsShape

A [2, 2] index 3, 3, 0, 1 gives a [2, 2, 3] value: rows 3, 3, 0 and 1.

## GatherRowsReadsZerosForAnIndexOutsideTheTable

The index -1, 4, 1 over the four-row table gives a row of zeros for -1, a row of zeros for the
bound 4 and row 1 for 1.

## ScatterAddRowsSumsRepeatedIndexesAndLeavesTheOthersZero

Five rows (1 to 15) with the index 1, 1, 3, 0, 1 into four rows give row 0 = the fourth input row
(10, 11, 12), row 1 = rows 0, 1 and 4 added (18, 21, 24), row 2 = zeros, which no index names, and
row 3 = the third input row (7, 8, 9).

## ScatterAddRowsReadsAnInt64Index

The same rows and an INT64 index give the same twelve values.

## ScatterAddRowsSkipsAnIndexOutsideTheTable

The index -1, 0, 4, 2, 0 skips input rows 0 and 2, whose indices are -1 and the bound 4. Row 0 of
the result is rows 1 and 4 added (17, 19, 21), row 2 is the input row 3 (10, 11, 12), and rows 1
and 3 are zeros, so nothing of a skipped row reaches the result. A write outside the result is
SushiBLAS's contract and is tested in its own gather tests; this file cannot see outside a tensor.

## TakeAlongPicksOneElementPerRow

Over the [3, 4] values 1 to 12 the index 2, 0, 3 gives 3, 5 and 12.

## TakeAlongReadsZeroForAnIndexOutsideTheRow

The INT64 index -1, 4, 1 gives 0, 0 and 10: -1 and the bound 4 each read zero.

## PutAlongWritesOneElementPerRowAndZeroElsewhere

The picked values 7, 8, 9 at the columns 2, 0, 3 of a [3, 4] result give a result that holds each
value at its column and zeros elsewhere.

## PutAlongSkipsAnIndexOutsideTheRow

The INT64 index -1, 4, 1 leaves the first two rows zero and writes 9 at column 1 of the third.

## ArgmaxRowsEqualsTheHostArgmaxIncludingATie

A [4, 5] value whose rows are (1, 5, 2, 5, 0), (-3, -1, -2, -9, -1), (4, 4, 4, 4, 4) and
(0, 1, 2, 3, 9) gives the INT64 columns 1, 1, 0 and 4, which `std::max_element` gives on the
host. Row 0 ties at columns 1 and 3, row 1 at 1 and 4, and row 2 at every column; each tie gives
its lowest column.

## TheArgmaxStaysInTheGraphsPrecisionUnderAMixedPolicy

`narrows_under_mixed_precision` is false for `K_ARGMAX_ROWS`, and a `MixedPrecisionPolicy` answers
the asked type for HALF, FLOAT32 and FLOAT64.
