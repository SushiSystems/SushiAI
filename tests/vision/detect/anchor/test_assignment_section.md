# tests/vision/detect/anchor/test_assignment_section.cpp

Suites: `Small_DetectAnchorAssignmentSection`.

Tests the `"assignment"` member through `read_anchor_task` and `write_anchor_task`, with
`modules/vision/detect/include/SushiAI/vision/detect/anchor/assignment_section.hpp` behind them.
Every case starts from a valid section of two heads over four anchors.

## ASectionWithoutTheMemberSelectsBestShape

Without the member the task reads `BestShape`, and the written section leaves the member out.

## ShapeRatioReadsItsDefaults

`{"rule": "shape_ratio"}` reads a `ratio_limit` of 4 and a `neighbour_offset` of 0.5.

## AShapeRatioSectionRoundTrips

A section with a `ratio_limit` of 3.5 and a `neighbour_offset` of 0.25 reads both values and is
written back with the same JSON text.

## RefusesAnUnknownRule

The rule "nearest" is refused naming `'rule'`, `'nearest'` and `'shape_ratio'`.

## RefusesAnUnknownKey

The key `g` under `shape_ratio` is refused naming it.

## RefusesARatioLimitOutOfRange

A `ratio_limit` of 0.5 is refused naming the key.

## RefusesANeighbourOffsetOutOfRange

A `neighbour_offset` of 1 is refused naming the key.

## RefusesAShapeRatioKeyUnderBestShape

A `ratio_limit` stated under `best_shape` is refused naming the key and the rule.
