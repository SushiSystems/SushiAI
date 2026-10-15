# tests/vision/detect/anchor/test_head_truth.cpp


Suites: `Unit_DetectAnchorHeadTruth`.

Tests `validate_head_truth` over a batch of one image with two truth rows, three classes and a
64 by 64 network. The row under test is row 1, given by its pixel corners and its class. Every
refusal has the code `InvalidDescription`.

## AcceptsRowsWhoseCentreLiesInsideTheInput

A row with the corners -8 and 8 on both axes, whose centre is exactly 0, and the class 2 is
accepted.

## RefusesACentreAtTheRightEdge

A centre of exactly 1 along the columns is refused; the message holds "truth row 1 of image 0",
"centre" and "(1, 0.5)".

## RefusesACentreLeftOfTheInput

A centre of -0.25 along the columns is refused, and the message holds "(-0.25, 0.5)".

## RefusesACentreBelowTheInput

A centre of 1.25 along the rows is refused, and the message holds "(0.5, 1.25)".

## RefusesACentreAboveTheInput

A centre of -0.25 along the rows is refused, and the message holds "(0.5, -0.25)".

## RefusesACentreThatIsNotANumber

A NaN corner is refused with a message holding "truth row 1 of image 0" and "centre".

## RefusesAWidthThatIsNotPositive

A row of zero width is refused, and the message holds "size 0 by 0.5".

## RefusesAHeightThatIsNotPositive

A row whose bottom lies above its top is refused, and the message holds "size 0.5 by -0.25".

## RefusesAClassAtTheClassCount

The class 3 with three classes is refused, and the message holds "class 3" and "3 classes".

## RefusesANegativeClass

The class -1 is refused, and the message holds "class -1".

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
