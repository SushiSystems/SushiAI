# tests/vision/detect/anchor/test_head_assignment.cpp


Suites: `Unit_DetectAnchorHeadAssignment`.

Tests `assign_anchors_to_heads`, which gives each head, in the order of its stride, a run of
anchor indices in ascending area and returns an `AnchorLayout`.

## NineAnchorsOverThreeHeadsGiveThreeRunsOfThree

Nine anchors over the strides 32, 16 and 8 give the masks {6, 7, 8}, {3, 4, 5} and {0, 1, 2}.

## SixAnchorsOverTwoHeadsGiveTwoRunsOfThree

Six anchors of rising area over the strides 32 and 16 give {3, 4, 5} and {0, 1, 2}.

## ReproducesTheMasksOfFiveCfgs

For five named cases the file holds, each an anchor list in pixels, an input side, the strides
and the expected masks, the computed masks equal the expected ones.

## DiffersFromTheSecondMaskOfYolov4TinyByOneAnchor

For six rising anchors over the strides 32 and 16, the first computed mask is {3, 4, 5}. The
second is {0, 1, 2}, which is not {1, 2, 3}.

## TheRemainderGoesToTheHeadOfSmallestStride

Seven anchors over the strides 32 and 16 give {4, 5, 6} and {0, 1, 2, 3}. Over the strides 16
and 32 they give {0, 1, 2, 3} and {4, 5, 6}.

## ARemainderOfTwoGoesToTheTwoSmallestStrides

Eleven anchors over the strides 16, 32 and 8 give {4, 5, 6, 7}, {8, 9, 10} and {0, 1, 2, 3}.

## UnsortedAnchorsKeepTheirOrderAndAreNamedByArea

Four anchors whose ascending area is 3, 1, 2, 0 give the masks {2, 0} and {3, 1} over the
strides 32 and 16. The layout's anchors are the inputs in their given order, compared exactly.

## AnchorsOfEqualAreaGoInIndexOrder

Two anchors of one area over the strides 8 and 16 give {0} and {1}.

## TheResultIsAValidLayout

`validate_anchor_layout` is ok for the layout of a nine-anchor, three-head assignment.

## PerHeadSetsTheCountOfEachHead

With `per_head` {2, 4} over the strides 32 and 16, six anchors give {4, 5} and {0, 1, 2, 3}.

## RefusesAPerHeadThatDoesNotSumToTheCount

For six anchors, `per_head` {2, 3}, {3, 4} and two counts of the largest `int64_t` each fail
with `Incompatible`.

## RefusesAPerHeadOfAnotherLengthThanTheStrides

For two strides, `per_head` {6} and {2, 2, 2} each fail with `Incompatible`.

## RefusesAPerHeadWithAnEmptyHead

`per_head` {6, 0} and {7, -1} each fail with `InvalidDescription`.

## RefusesFewerAnchorsThanHeads

Two anchors over three strides fail with `Incompatible`.

## RefusesMissingAnchorsAndBadStrides

No anchor, no stride, a stride of 0 and a stride of -8 each fail with `InvalidDescription`.

## RefusesAnAnchorWithoutAFinitePositiveSide

An anchor with a zero height and an anchor with a NaN width each fail with
`InvalidDescription`.
