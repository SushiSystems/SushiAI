# tests/core/graph/lowering/test_spatial.cpp


Suites: `Small_SpatialLowerings`, `Medium_SpatialLowerings`.

Tests `K_SPATIAL_LOWERINGS` and two gradient rows no other file compares with host values. The
forward spatial rows are covered in `tests/core/graph/test_lowering/test_hand_values.cpp`. The
file-local `run_one` writes one leaf, compiles and runs the graph once and reads one result. Results
are FLOAT32, compared within 1e-6.

## EachRowPairsItsOperationWithItsLowering

The table has fifteen rows, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## UpsampleGradientSumsEachBlockOntoItsPixel

`upsample2d_backward` of a [1, 4, 4, 1] gradient holding 0 to 15, onto a [1, 2, 2, 1] input with
both scales 2, writes at each pixel the sum of its 2 x 2 block.

## SpatialMeanGradientSpreadsEvenlyOverThePlane

`spatial_mean_backward` of the [1, 2] gradient (4, -2) onto a [1, 2, 2, 2] input writes 1 and
-0.5 at every one of the four pixels, which is each gradient over H * W.
