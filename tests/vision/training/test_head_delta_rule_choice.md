# tests/vision/training/test_head_delta_rule_choice.cpp

Suites: `Small_HeadDeltaRuleChoice` and `Small_HeadDeltaRuleChoiceBestShape`.

Tests `choose_head_delta_rule` of `head_delta_rule_choice.hpp`.

### BestShapeBuildsTheBestShapeRule

Rows: the default choice, and a best-shape choice that also holds a shape-ratio limit of 1, which the best-shape rule does not read. Each builds a `BestShapeHeadDeltaRule`; the pointer is never null.

### ShapeRatioBuildsTheShapeRatioRule

The shape-ratio choice builds a `ShapeRatioHeadDeltaRule`; the pointer is never null.

### TheShapeRatioSettingsReachTheRule

A rule built with a neighbour offset of 0.5 assigns more slots over the fixture than one built with an offset of 0, because the 16 pixel box at (20, 20) lies 1.25 cells into the first grid and takes its left and top neighbours.

### ShapeRatioSettingsThatAreRefusedNameTheirKey

A ratio limit of 1 fails with `InvalidDescription` naming "'ratio_limit'".
