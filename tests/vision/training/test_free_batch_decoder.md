# tests/vision/training/test_free_batch_decoder.cpp

Suite: `Small_FreeBatchDecoder`. No case needs a device.

The fixture is one image over two levels: 1 by 2 cells at stride 8, then 1 by 1 at stride 16,
so three anchors with centres (0.5, 0.5), (1.5, 0.5) and (0.5, 0.5) in grid units, and two
classes. The class logits 0, ln 3 and -ln 3 decode to 0.5, 0.75 and 0.25.

## ReadsSingleBinLogitsAsDistances

With reg_max 1 the box channels are the distances. Left, top, right, bottom of
(0.5 0.5 0.5 0.5), (1 0 1 0) and (0.5 0.5 1.5 1.5) give the centre-size boxes
(4 4 8 8), (12 4 16 0) and (16 16 32 32) in pixels.

## TakesTheExpectationOfFourBins

With reg_max 4, uniform bins on level 0 expect 1.5 per side, so (4 4 24 24) and
(12 4 24 24). Bin 2 at logit 50 on level 1 expects 2 per side, so (8 8 64 64).

## RejectsAMissingLevel

One level against facts of two throws `Error`.
