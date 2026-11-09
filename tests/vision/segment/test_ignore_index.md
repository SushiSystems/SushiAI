# tests/vision/segment/test_ignore_index.cpp


Suites: `Small_SegmentIgnoreIndex`.

Tests `K_DEFAULT_IGNORE_INDEX`, through `ConfusionMatrix::add`.

This file needs no device. The header defines one constant and nothing else, so the single case
runs `add` with it.

## TheDefaultIsTheLargestValueOfAMaskByte

Of the two pixels with the predictions 0, 1, the second has the truth 255, the largest value an
8-bit mask pixel can hold, so a PNG mask can carry it. `add` with `K_DEFAULT_IGNORE_INDEX` counts
one pixel, at (0, 0), and `predicted_pixels(1)` is 0.
