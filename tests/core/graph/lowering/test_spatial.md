# tests/core/graph/lowering/test_spatial.cpp


Suites: `Small_SpatialLowerings`, `Small_SpatialRow`, `Medium_SpatialLowerings`.

Tests `K_SPATIAL_LOWERINGS`, two gradient rows and the two bilinear rows, none of which another file compares with host values. The
forward spatial rows are covered in `tests/core/graph/test_lowering/test_hand_values.cpp`. The
file-local `run_one` writes one leaf, compiles and runs the graph once and reads one result. Results
are FLOAT32, compared within `K_ABSOLUTE_TOLERANCE`, 32 unit roundoffs of float.

## TheTableHoldsExactlyTheRowsTheHeaderNames

The table has as many rows as the header names (seventeen).

## EachRowPairsItsOperationWithItsLowering

One row per entry of the table, in the order the header writes them. Each row names its operation,
the function the header declares for it and no scratch rule, and `find_lowering` returns that
function for the operation.

## UpsampleGradientSumsEachBlockOntoItsPixel

`upsample2d_backward` of a [1, 4, 4, 1] gradient holding 0 to 15, onto a [1, 2, 2, 1] input with
both scales 2, writes at each pixel the sum of its 2 x 2 block.

## SpatialMeanGradientSpreadsEvenlyOverThePlane

`spatial_mean_backward` of the [1, 2] gradient (4, -2) onto a [1, 2, 2, 2] input writes 1 and
-0.5 at every one of the four pixels, which is each gradient over H * W.

## BilinearUpsampleGivesTheHandValuesOfSushiBLAS

`upsample2d_bilinear` of the [1, 2, 2, 1] image 0, 8, 16, 40 at scale 2 gives the sixteen values
SushiBLAS's own test holds: (0, 2, 6, 8), (4, 7, 13, 16), (12, 17, 27, 32) and (16, 22, 34, 40).
Along one axis the output pixels read 1.0 of x0; 0.75 of x0 and 0.25 of x1; 0.25 of x0 and 0.75 of
x1; and 1.0 of x1. The gradient check at scales 2 and 3 is in `tests/core/autograd/rules/test_spatial/test_gradients.cpp`.

## BilinearGradientOfOnesGivesEachPixelTheScaleProduct

`upsample2d_bilinear_backward` of a [1, 6, 4, 1] gradient of ones, onto a [1, 2, 2, 1] input with
the scales 3 and 2, gives 6 at every pixel: the weights of each output pixel sum to one, so each
input pixel of a two-pixel axis gathers the scale of that axis, and 3 * 2 = 6.
