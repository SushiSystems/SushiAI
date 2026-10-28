# tests/vision/training/test_head_delta_rule_choice.cpp

Suites: `Small_HeadDeltaRuleChoice`.

Tests `choose_head_delta_rule` of `head_delta_rule_choice.hpp`.

### BestShapeBuildsTheBestShapeRule, ShapeRatioBuildsTheShapeRatioRule

The default choice builds a `BestShapeHeadDeltaRule` and the shape-ratio choice builds a `ShapeRatioHeadDeltaRule`; the pointer is never null.

### TheShapeRatioSettingsReachTheRule

A rule built with a neighbour offset of 0.5 assigns more slots over the fixture than one built with an offset of 0, because the 16 pixel box at (20, 20) lies 1.25 cells into the first grid and takes its left and top neighbours.

### ShapeRatioSettingsThatAreRefusedNameTheirKey

A ratio limit of 1 fails with `InvalidDescription` naming "'ratio_limit'".

### BestShapeIgnoresTheShapeRatioSettings

A best-shape choice that holds a ratio limit of 1 still builds its rule.
