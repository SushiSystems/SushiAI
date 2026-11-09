# tests/vision/detect/test_top_k.cpp


Suites: `Small_DetectTopK` and `Medium_DetectTopKDeathTest`.

Tests `select_top_k` in `modules/vision/detect/include/SushiAI/vision/detect/top_k.hpp`, which
returns the indices of the k largest values, larger value first and lower index first among equals.

## OrdersByValueThenIndex

For the values 0.5, 0.9, 0.5, 0.1, 0.9 and k of 4, the indices are 1, 4, 0, 2: the two 0.9 in
index order, then the two 0.5 in index order.

## RanksNaNBelowEveryNumber

A NaN ranks below every number, and NaNs keep ascending index order. For NaN, 0.5, NaN, -1, 0.5
and k of 5, the indices are 1, 4, 3, 0, 2.

## RefusesAKAboveTheValueCount

A k of 6 over 5 values stops the process, and the message names both numbers. It is a death
test, so its suite is medium.
