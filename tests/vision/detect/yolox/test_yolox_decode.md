# tests/vision/detect/yolox/test_yolox_decode.cpp

Suites: `Small_DetectYoloxDecode`. The cases run on plain arrays, so no case needs a device.

Tests `decode_yolox`, `make_yolox_nms_config` and `select_yolox_detections` against boxes and
scores worked by hand. Where a value is exact in float32 it is compared exactly; the one value
that passes through `exp` is compared to the double reference within four float roundings of
its magnitude.

## TheCentreIsTheCellPlusTheOffsetTimesTheStride

On a 2 by 2 level of stride 8, the cell at row 1, column 0 with the offsets 0.25 and 0.5 has the
centre (0.25 * 8, 1.5 * 8) = (2, 12).

## TheSizeIsTheExponentialTimesTheStride

On a level of stride 16, the raw sizes 0 and 1 decode to 16 and exp(1) * 16.

## AScoreIsTheObjectnessTimesTheClassProbability

With objectness logit 0 and class logits 0 and 100, the scores are 0.25 and 0.5. With objectness
logit 100 and class logits -100 and 100, the scores are 0 and 1.

## AnchorsRunLevelByLevel

A 1 by 2 level of stride 8 and a 1 by 1 level of stride 16 give 3 anchors. The second anchor has
centre x 8; the third, with offset 0.5 and stride 16, has centre x 8 and width 16.

## EachImageReadsItsOwnSlice

With a batch of 2, the second image's box centre is 4 and its score 1, while the first image's
centre is 0 and its score 0.25.

## RefusesNoLevel

An empty level list fails with "at least one level".

## RefusesASpanThatDoesNotFitItsLevel

A class span of 5 values over a 2 by 2 level of 3 classes fails with "level 0", "5 class
values" and "need 12".

## RefusesAStrideThatIsNotPositive

A stride of 0 fails with "level 0" and "stride".

## TheNmsConfigFollowsTheTask

The config takes the task's score threshold, suppression threshold, criterion and exponent, keeps
the best class of each anchor and keeps classes apart.

## ADuplicateBoxIsSuppressedAndAnIsolatedOneStays

Three cells of one row at stride 8. Cell 0 is the box (0, 0, 8, 8) with score 0.5. Cell 1 shifts
left by half a cell to the same box with score 0.25 and is suppressed, since their IoU is 1.
Cell 2 is (16, 0, 24, 8) with score 0.25 and stays. The detections of the image are
`0 0 8 8 0.5 0` then `16 0 24 8 0.25 0`.
