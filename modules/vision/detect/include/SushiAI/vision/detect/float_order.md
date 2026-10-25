# modules/vision/detect/include/SushiAI/vision/detect/float_order.hpp


## first_non_finite

Returns the index of the first NaN or infinity of a span, or -1. The host stages call it on the
values they receive and throw with the index it returns.

## ranks_above

Compares two floats for a descending sort. A NaN ranks below every number and equal to another
NaN, so a comparator built on it, with the index as the last tie-break, is a strict weak ordering
whatever the data holds. `select_top_k`, the suppression's row maximum, the shape-ratio
assignment and the average-precision ranking use it.
