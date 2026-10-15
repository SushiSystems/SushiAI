# tests/vision/detect: free/test_box_coding.cpp, test_box_format.cpp, test_iou.cpp


Box decoding and encoding, the two box formats and the IoU forms.

Suites: `Unit_DetectBoxes`.

Tests box decoding and encoding, the two box formats and the IoU forms.

This file needs no device. `Unit_DetectBoxes` covers `modules/vision/detect/include/SushiAI/vision/detect/free/box_coding.hpp`,
`modules/vision/detect/include/SushiAI/vision/detect/box_format.hpp` and `modules/vision/detect/include/SushiAI/vision/detect/iou.hpp`. The hand cases
write the expected value as the arithmetic that produces it. Three cases read `forward.sacp`
and report themselves skipped when the file is absent.

## Tolerances

`K_EXACT` is 0, for values float32 holds exactly. `K_DECODE_TOLERANCE` is 1.3e-4, two float32
steps at 640, for a decoded box in pixels. `K_ROUND_TRIP_TOLERANCE` is 1.6e-5, two steps at 80,
for a decode followed by an encode in grid units. `K_IOU_TOLERANCE` is 1e-6, eight float32
steps at 1.

## K_REFERENCE_ANCHORS and K_REFERENCE_LEVELS

`K_REFERENCE_ANCHORS` is 8400 and `K_REFERENCE_LEVELS` holds the three levels of a 640 px
input, both from "levels" in `tests/vision/registration/fixtures/yolo26/MANIFEST.json`.

## ReferenceGrid

`ReferenceGrid` holds the `[8400, 2]` points and the `[8400]` strides of the reference levels
in grid units, and `make_reference_grid` fills one with `make_anchors`.

## join_raw_boxes

`join_raw_boxes` reads a branch's three per-level tensors, `<branch>.level<i>.boxes`, and
copies each into its place in an `[8400, 4]` array with `join_level`. It returns an empty
string, or the reason a level could not be read.

## DecodesDistancesToCorners

`decode_xyxy` of the distances (0.5, 1, 1.5, 2) at the point (1.5, 2.5) gives
(1, 1.5, 3, 4.5): left and top are subtracted from the point, right and bottom added.

## DecodesDistancesToCentreAndSize

`decode_xywh` of the same input gives the midpoint (2, 3) and the size (2, 3) of those corners.

## RepeatsTheAnchorPointsForEachImage

Two anchor points and four rows of distances decode as two images: rows 2 and 3 use points 0
and 1 again.

## EncodeInvertsDecode

`encode_distances` of the boxes `decode_xyxy` writes returns the three rows of distances at
tolerance 0.

## ScalesEachBoxByItsAnchorStride

`scale_boxes` multiplies all four values of a row by its anchor's stride: the row (1, 2, 3, 4)
becomes (8, 16, 24, 32) at stride 8 and (16, 32, 48, 64) at stride 16.

## ConvertsBetweenCornersAndCentreSize

`xyxy_to_xywh` maps (1, 2, 5, 8) to (3, 5, 4, 6), and `xywh_to_xyxy` maps that back to the
corners.

## RejectsRowsThatDoNotCoverTheAnchors

Three rows of distances over two anchor points throw with a message containing "2 anchors" and
"12".

## ReproducesTheReferenceOneToManyDecode

The raw one-to-many boxes, decoded with `decode_xywh` and scaled by the strides, equal
`one2many.decoded.boxes` within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp`
is absent.

## ReproducesTheReferenceOneToOneDecode

The raw one-to-one boxes, decoded with `decode_xyxy` and scaled, equal `one2one.decoded.boxes`
within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp` is absent.

## EncodeInvertsTheDecodeOfTheReferenceBoxes

Encoding the decode of the raw one-to-one boxes returns them within `K_ROUND_TRIP_TOLERANCE`.
The case is skipped when `forward.sacp` is absent.

## IdenticalBoxesScoreOne

A 2 by 2 box has IoU 1 and complete IoU 1 with itself.

## DisjointBoxesScoreZeroAndPayForDistance

Two unit squares two units apart have IoU 0. Their complete IoU is minus the squared centre
distance, 4, over the squared hull diagonal, 10.

## OverlappingSquaresLoseTheCentreTerm

Two 2 by 2 squares offset by 1 on both axes have IoU 1 / 7, and their complete IoU is that
minus 2 / 18. The aspect ratios are equal, so the aspect term is 0.

## DifferentAspectRatiosLoseTheAspectTerm

A 4 by 2 box and a 2 by 2 box at one corner have IoU 0.5. The expected complete IoU is computed
in double from the centre term, the angle gap `atan(1) - atan(2)` and the trade-off factor, and
compared within `K_IOU_TOLERANCE`.

## ADegenerateBoxStaysFinite

A box of zero size inside a 2 by 2 box has IoU 0. Its width over its height is 0 over eps, so
the angle gap is pi / 4, the aspect term is 0.25, and the complete IoU is -0.25 * 0.25 / 1.25,
which is -0.05. The complete IoU of the zero-size box with itself is finite and 0.

## ElementwiseFormsPairRows

`elementwise_iou` and `elementwise_complete_iou` over two rows give 1 / 7 and 0, and
1 / 7 - 2 / 18 and -4 / 10: row i of the first array is paired with row i of the second.

## PairwiseFormFillsEveryPair

`pairwise_iou` of two boxes against three writes six values, row n and column m holding the
IoU of box n with box m: 1, 1 / 7, 0, then 1 / 4, 0, 0.

## RejectsElementwiseInputsOfDifferentLength

Arrays of 8 and 4 values with an output of 2 throw with a message containing "got 8, 4 and 2".
