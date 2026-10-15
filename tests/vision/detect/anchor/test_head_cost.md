# tests/vision/detect/anchor/test_head_cost.cpp


Suites: `Unit_DetectAnchorCost`.

Tests `measure_head_cost`
over a head with two classes whose delta the file builds by hand. Its box part sums to 5.25 and
its non-box part to 26, over a batch of 2. Every number is compared to 1e-12.

## SquaredErrorSplitsBoxFromNonBoxAndDividesByBatch

With the squared-error box loss and an objectness weight of 1, the total is 15.625, the non-box
share 13, the box share 2.625 and the stored cost 31.25.

## ObjectWeightScalesTheNonBoxShare

With an objectness weight of 2 the non-box share is 26 and the box share -10.375. The total
stays 15.625 and the stored cost 31.25.

## CompleteIoUStoredAddsTheIoUTermAndWeightedNonBoxSum

With the CompleteIoU box loss, a box weight of 0.75, an objectness weight of 2 and two
assignments whose IoU sums to 1, the stored cost is 52.375. The total is 15.625, the non-box
share 26 and the box share -10.375.

## CompleteIoUWithoutAssignmentsOmitsTheIoUTerm

A CompleteIoU head with no assignment and an objectness weight of 1 stores 26.

## ZeroDeltaReportsZeroCost

A delta of 28 zeros reports 0 in the total, the box share, the non-box share and the stored
cost.
