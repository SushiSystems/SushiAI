# tests/vision/detect/anchor/test_head_truth.cpp


Suites: `Small_DetectAnchorHeadTruth`.

Tests `validate_head_truth` over a batch of one image with two truth rows, three classes and a
64 by 64 network. The row under test is row 1, given by its pixel corners and its class. Every
refusal has the code `InvalidDescription`.

## AcceptsRowsWhoseCentreLiesInsideTheInput

A row with the corners -8 and 8 on both axes, whose centre is exactly 0, and the class 2 is
accepted.

## ARefusedRowIsNamedWithItsValue

One row per refused truth row, in three groups; each fails with `InvalidDescription` and a
message that holds "truth row 1 of image 0" and the words of its row. `CentreOutside`: a centre
of exactly 1 along the columns ("centre", "(1, 0.5)"), of -0.25 along the columns
("(-0.25, 0.5)"), of 1.25 along the rows ("(0.5, 1.25)") and of -0.25 along the rows
("(0.5, -0.25)"). `SideNotPositive`: a row of zero width ("size 0 by 0.5") and a row whose
bottom lies above its top ("size 0.5 by -0.25"). `ClassOutOfRange`: the class 3 with three
classes ("class 3", "3 classes") and the class -1 ("class -1").

## RefusesACentreThatIsNotANumber

A NaN corner is refused with a message holding "truth row 1 of image 0" and "centre".

## LeavesAPaddingRowUnread

A row with an out-of-range centre and class is accepted when its `valid` flag is 0.

## RefusesArraysThatDoNotHoldTheirShape

A batch count of 2 against a truth of one image is refused with "2 images" and "batch of 1". A
truth one box value short is refused with "8 box values" and "7".

## RefusesANetworkSideBelowOne

A network size of 64 by 0 is refused with "network size" and "64 by 0".

## TheDeltaRefusesATruthOutsideTheGrid

`compute_head_delta` given the row with a centre of exactly 1 fails with the same message
parts, and leaves `delta` and `assigned_row` of its result empty.
