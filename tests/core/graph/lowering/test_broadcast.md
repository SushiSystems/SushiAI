# tests/core/graph/lowering/test_broadcast.cpp


Suites: `Small_BroadcastLowerings`, `Small_BroadcastRow`, `Medium_BroadcastHandValues` and
`Medium_BroadcastLowerings`.

Tests `K_BROADCAST_LOWERINGS`, the four broadcast operations and `K_SUM_ALONG` on an engine. The
values are small integers, or quotients and products of them by powers of two, so every result is
exact in FLOAT32 and is compared with `EXPECT_EQ`. The gradient checks of the five operations are
in `tests/core/autograd/rules/test_broadcast.cpp`.

## TheTableHoldsExactlyTheRowsTheHeaderNames

The table has as many rows as the header names (five).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## EachOperationOnEachSideWritesTheHandValues

One case per operation and side over the [2, 3] tensor 1 to 6. The leading vector is (2, 4) and
the trailing vector is (1, 2, 4). For the leading side, add gives (3, 4, 5, 8, 9, 10), sub
(-1, 0, 1, 0, 1, 2), mul (2, 4, 6, 16, 20, 24) and div (0.5, 1, 1.5, 1, 1.25, 1.5). For the
trailing side, add gives (2, 4, 7, 5, 7, 10), sub (0, 0, -1, 3, 3, 2), mul (1, 4, 12, 4, 10, 24)
and div (1, 1, 0.75, 4, 2.5, 1.5).

## ARankThreeTensorTakesALeadingAndATrailingVector

Over a [2, 2, 3] tensor of 1 to 12, `broadcast_mul` with the [2, 2] leading vector (1, 2, 3, 4)
scales each of the four rows of three by one factor, and `broadcast_add` with the [2, 3] trailing
vector (10, ..., 60) adds the table to each of the two matrices.

## SumAlongSumsDownToAPrefixAndToASuffix

Over the [2, 3] matrix 1 to 6, the sum to [2] on the leading side gives (6, 15) and to [3] on the
trailing side gives (5, 7, 9). Over the [2, 2, 3] cube 1 to 12 the sum to [2, 2] leading gives
(6, 15, 24, 33), to [2, 3] trailing gives (8, 10, 12, 14, 16, 18), to [2] leading gives (21, 57)
and to [3] trailing gives (22, 26, 30).

## APositionTableThroughBroadcastAddEqualsTheReshapeForm

A [2, 3, 4] value and a [3, 4] position table are added by `broadcast_add` on the trailing side, and
again by the reshape form: the value as [2, 12] rows plus the table as a [12] bias. The inputs are
not exact in FLOAT32 (0.1 i + 1/3 and 0.7 / (i + 1)), and the two results are compared byte for
byte, so a different order or width of addition would show.

## ABroadcastRefusesAVectorThatIsNotThatSideOfTheTensor

A [2] vector declared trailing over a [2, 3] tensor is refused with "a suffix", a [3] vector
declared leading with "a prefix", and a sum to [3] declared leading with "a prefix". No node is
left behind.
