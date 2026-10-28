# tests/vision/training/test_anchor_head.cpp

Suites: `Small_AnchorHead`.

Tests `AnchorHead` of `anchor_head.hpp`, a struct with no behaviour of its own. `head_delta_rule_fixture.hpp` holds the two-head batch that this file and the four head delta rule files share: a layout of four anchors in two heads, heads on grids of 4 and 2 over a 64 pixel input, three classes, and a truth of two boxes in one image.

### ADefaultHeadIsEmpty

A default head names no output, has an empty extent, no classes, the logit coding, a stretch of 1, exponential sides, the squared-error box loss and the objectness target 1.

### ACopyHoldsEveryPartAndChangesApart

A copy holds the output, extent, class count and weights of its source, and changing the copy leaves the source as it was.

### ARuleNamesAHeadByItsOutput

A head named "stem.detect" with squared sigmoid sides is refused by the best-shape rule `check`, and the message holds "head 'stem.detect'".
