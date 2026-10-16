# tests/vision/segment/test_ignore_index.cpp


Suites: `Small_SegmentIgnoreIndex`.

Tests the value of `K_DEFAULT_IGNORE_INDEX`.

This file needs no device. The header defines one constant and nothing else, so the single case
pins its value.

## TheDefaultIsTheLargestValueOfAMaskByte

`K_DEFAULT_IGNORE_INDEX` is 255, the largest value an 8-bit mask pixel can hold, so a PNG mask
can carry it.
