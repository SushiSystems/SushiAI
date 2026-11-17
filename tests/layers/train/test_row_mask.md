# tests/layers/train/test_row_mask.cpp

Suites: `Small_MaskedMetrics`.

Tests that `measure()` of `cross_entropy`, `binary_cross_entropy`, `multi_label` and
`mean_squared_error`, built with `type_mask: true`, counts the rows the mask keeps and no others.
Every case runs on the engine of a `HostExecutionBackend` and needs no device. The mask of the
cases is 1, 0, 1, 1: row 1 is a row of another question type, built so that counting it moves each
metric.

## CrossEntropyAccuracyCountsTheKeptRowsAlone

The labels 0, 2, 2, 1 against the best logits 0, 1, 2, 0 are right on rows 0 and 2. Over the kept
rows 0, 2 and 3 accuracy is 2 of 3; counting row 1 would give 2 of 4. The bound is one rounding,
the quotient of two exact counts.

## CrossEntropyAccuracyOfABatchThatKeepsNoRowIsZero

With the mask 0, 0, 0, 0 accuracy is exactly 0, the value the ratios of `binary_confusion.hpp` take
for no prediction, and the call succeeds.

## AMaskOfAnotherRowCountIsRefusedByMeasure

A mask of 5 values for an objective of 4 rows fails with `Incompatible`, "5 rows" and "reads 4",
and adds no metric.

## BinaryMetricsCountTheKeptRowsAlone

The logits 2, 1, -1, -3 and the labels 1, 0, 1, 0 give a hit, a false alarm (row 1), a miss and a
correct rejection at the cut 0. Over the kept rows accuracy is 2 of 3, precision 1 of 1 and recall
1 of 2; row 1 counted would give 2 of 4, 1 of 2 and 1 of 2.

## MultiLabelF1CountsTheKeptRowsAlone

Four rows of two columns. Over the kept rows, column 0 has 2 hits, 1 miss and no false alarm
(F1 4 / 5) and column 1 has 1 hit, 1 miss and a correct rejection (F1 2 / 3), so the mean is
11 / 15. Row 1 counted would give 2 / 3 and 1 / 2. The bound is four roundings: two quotients, the
sum and the division by 2.

## MeanAbsoluteErrorCountsTheKeptRowsAlone

The absolute differences of rows 0, 2 and 3 sum to 5.5 over 6 elements, so the metric is 5.5 / 6.
Row 1 differs by 10 in each element and would give 25.5 / 8.

## MeanAbsoluteErrorOfABatchThatKeepsNoRowIsZero

With the mask 0, 0, 0, 0 the metric is exactly 0 and the call succeeds.
