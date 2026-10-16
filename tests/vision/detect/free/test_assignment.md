# tests/vision/detect/free/test_assignment.cpp


Suites: `Small_DetectAssignment`.

Tests that a default `Assignment` has no extents and holds no targets.

This file needs no device. The header declares a struct and no function. The assigner that
fills it is tested in `tests/vision/detect/free/test_assigner.cpp`, and the leaves read from it
in `tests/vision/detect/free/test_target_leaves.cpp`, so this file pins the empty state only.

## ADefaultAssignmentHasNoExtentsAndNoTargets

A default `Assignment` has a batch, an anchor count and a class count of 0, and its five target
arrays are empty.
