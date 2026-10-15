# tests/vision/detect: anchor/test_head_options.cpp, anchor/test_head_decode.cpp


The boxes and scores an anchor head decodes to under the Logit and the Activated coding and under `scale_xy`, the order of the decoded rows, the channel and row counts, and what `validate_head` refuses.

Suites: `Unit_DetectAnchorDecode`.

Tests the decoding of one anchor head, with `count_head_channels`, `count_head_rows` and
`validate_head`. A head entry holds x, y, width, height, objectness and then the classes. A
saturated logit in this file is 100 or -100, and decoded values are compared to 1e-6.

## ZeroLogitsGiveTheCellCentreTheAnchorAndQuarterScores

A one-cell head of zero logits over a 0.25 x 0.5 anchor decodes to the centre (0.5, 0.5), the
size (0.25, 0.5) and a score of 0.25 for each of two classes.

## SaturatedLogitsReachTheCellEdgesAndTheScoreBounds

With x at +100, y at -100 and width at -100, the box has x 1, y 0, width 0 and height 0.5. With
objectness and the first class at +100 and the second class at -100, the scores are 1 and 0.

## ZeroObjectnessLogitHalvesASaturatedClass

A zero objectness logit with a saturated class logit gives the score 0.5.

## ScaleTwoMovesASaturatedOffsetHalfACellPastTheEdge

With `scale_xy` 2 on a grid of one cell, x at +100 decodes to 1.5 and y at -100 to -0.5.

## ScaleTwoKeepsAZeroLogitAtTheCellCentre

With `scale_xy` 2, zero logits decode to x 0.5 and y 0.5.

## ActivatedSizeAtZeroHalfAndOneIsZeroTheAnchorAndFourAnchors

Under the Activated coding with a 0.125 x 0.25 anchor, size inputs of 0, 0.5 and 1 decode to
the widths 0, 0.125 and 0.5 and the heights 0, 0.25 and 1.

## ActivatedOffsetsAndScoresTakeNoLogistic

Under the Activated coding, x 0.25 and y 1 decode to 0.25 and 1. Objectness 0.5 with the
classes 0.5 and 1 gives the scores 0.25 and 0.5.

## ActivatedScaleTwoMapsZeroAndOnePastTheCellEdges

Under the Activated coding with `scale_xy` 2, the offsets 0 and 1 decode to -0.5 and 1.5.

## RowsRunCellByCellThenByAnchorSlotInEveryImage

On a 2x3 grid with two slots and two images, decoded row `(row * 3 + col) * 2 + slot` holds the
centre of that cell and the size of that slot's anchor. The head's slots are anchors 2 and 0 of
the layout. Every score is 0.25 except row 11 of image 1, which the case saturated and which
is 1.

## OneLayoutGivesTheSameFractionalSizeOnTwoGrids

One 0.3 x 0.45 anchor decodes to that size on a 13x13 grid and to the identical floats on a
19x19 grid. The last cell's x is 12.5 / 13 on the first grid and 18.5 / 19 on the second.

## CountsChannelsAndRowsFromTheHeadsOwnAnchors

With 80 classes, a head of three anchors has 255 channels and a head of two has 170. A batch of
4 on a 13x13 grid with three anchors has 507 rows, and on a 26x26 grid with two anchors 1352.

## ValidatesAHeadTheLayoutHolds

`validate_head` is ok for 80 classes, the Activated coding, `scale_xy` 2 and head 0 of a
one-head layout.

## RefusesAHeadIndexOutsideTheLayout

Head index 1 and head index -1 of a one-head layout each fail with `InvalidDescription`.

## RefusesAHeadWithoutClasses

A class count of 0 fails with `InvalidDescription`.

## RefusesAScaleThatIsNotFiniteAndPositive

A `scale_xy` of 0 and a NaN `scale_xy` each fail with `InvalidDescription`.

## RefusesAHeadWhoseLayoutDoesNotValidate

A layout whose head names anchor 3 of one anchor fails with `InvalidDescription`.
