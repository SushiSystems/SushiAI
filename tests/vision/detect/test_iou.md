# tests/vision/detect/test_iou.cpp


Suites: `Medium_DetectBoxes` and `Small_DetectBoxes`.

Tests `intersection_over_union`, `complete_iou` and the elementwise and pairwise forms in
`modules/vision/detect/include/SushiAI/vision/detect/iou.hpp` against hand cases. `K_IOU_TOLERANCE`
is 32 times the float unit roundoff, the smallest power of two not tighter than the earlier 1e-6.

## IdenticalBoxesScoreOne

A box against itself has IoU 1 and complete IoU 1: the intersection and the union are 4, and the
centre and the aspect gaps are 0.

## DisjointBoxesScoreZeroAndPayForDistance

Two unit squares with a gap of 1 have IoU 0. Their complete IoU is lowered by the centre distance
over the diagonal of the hull; the case holds the only check of this term.

## BoxesTouchingAtAnEdgeScoreZero

Two unit squares that share the edge x = 1 have an IoU of exactly 0, since the overlap has no
area.

## OverlappingSquaresLoseTheCentreTerm

Two squares of side 2 shifted by 1 on both axes give 1/7, and the complete IoU loses the centre
term only.

## DifferentAspectRatiosLoseTheAspectTerm

A 4 by 2 box and a 2 by 2 box give IoU 1/2, and the complete IoU loses the centre and the aspect
terms.

## ADegenerateBoxHasAnIouOfZero

A point against a box has an IoU of 0: the overlap has no area.

## ADegenerateBoxPaysTheAspectTerm

A point against a box has a complete IoU of minus the aspect term; the point's width over its
height is 0, so the angle gap is pi / 4.

## ADegenerateBoxAgainstItselfScoresZero

A point against itself gives a finite complete IoU of 0.

## ElementwiseFormsPairRows

The elementwise forms pair row i of one array with row i of the other, for the IoU and the
complete IoU.

## PairwiseFormFillsEveryPair

The pairwise form fills row n, column m with the IoU of box n of the first array and box m of the
second.

## RejectsElementwiseInputsOfDifferentLength

Arrays of 8 and 4 values stop the process, naming 8, 4 and 2. It is a death test in a medium
suite, since the stop can only be seen in a child process.
