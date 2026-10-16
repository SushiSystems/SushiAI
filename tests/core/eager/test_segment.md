# tests/core/eager/test_segment.cpp


Suites: `Small_Segment`.

Tests the default value of `Segment` and the half-open node range it names.

This file needs no device. `Segment` is a plain struct; `test_scope_trace.md` covers the segments
a trace yields, so this file pins only what the declaration itself promises.

## ADefaultSegmentIsUnnamedAndEmpty

A default segment has an empty name, occurrence 0, and `begin` equal to `end`, so it holds no
node.

## TheRangeExcludesItsEnd

A segment from node 4 to node 9 holds five nodes: `end` is one past the last node.
