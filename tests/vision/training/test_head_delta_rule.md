# tests/vision/training/test_head_delta_rule.cpp

Suites: `Small_HeadDeltaRule` and `Small_HeadDeltaRuleContract`.

Tests the `HeadDeltaRule` interface of `head_delta_rule.hpp`. The interface has two implementations, and its cases use each one only through a `HeadDeltaRule` reference. `HeadDeltaBatch` is plain data: its default values and the fixture's view of its own members are not promises and have no case.

### TheRuleServesTheInterface

Rows: `BestShape` over exponential sides and `ShapeRatio` over squared sigmoid sides. The rule accepts the heads of the fixture in `check`. `compute` returns a cost above 0 and a delta of 4 x 4 x 16 values for the first head and 2 x 2 x 16 for the second.

### EachRuleRefusesThroughCheckTheHeadsOfTheOther

The best-shape rule refuses heads of squared sigmoid sides and the shape-ratio rule refuses heads of exponential sides.
