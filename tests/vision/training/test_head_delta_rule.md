# tests/vision/training/test_head_delta_rule.cpp

Suites: `Small_HeadDeltaRule`.

Tests `HeadDeltaBatch` and the `HeadDeltaRule` interface of `head_delta_rule.hpp`. The interface has two implementations, and its cases use each one only through a `HeadDeltaRule` reference.

### ADefaultBatchHoldsNothing

A default batch has a null layout and truth, no heads, no outputs and a network size of 0 by 0.

### ABatchViewsTheHeadsOutputsLayoutAndTruthItIsGiven

The batch of the fixture points at the fixture layout and truth, holds two heads and two outputs, names the second head "head.1", and has a network size of 64 by 64.

### TheBestShapeRuleServesTheInterface, TheShapeRatioRuleServesTheInterface

Each rule accepts the heads of the fixture in `check`. `compute` returns a cost above 0 and a delta of 4 x 4 x 16 values for the first head and 2 x 2 x 16 for the second.

### EachRuleRefusesThroughCheckTheHeadsOfTheOther

The best-shape rule refuses heads of squared sigmoid sides and the shape-ratio rule refuses heads of exponential sides.
