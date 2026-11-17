# tests/layers/nn/test_input_leaves.cpp

Suites: `Small_InputLeafList`.

Tests `InputLeafList` of `input_leaves.hpp`, the leaves a step declares for the extra inputs of a
model. The file needs no device and builds no graph; a leaf is a name and a value id.

## AnEmptyListFindsNoName

A list with no leaf returns `K_NO_VALUE` for any name and holds no leaf.

## FindReturnsTheValueOfTheNamedLeaf

With the leaves "lengths" (7) and "options" (9), `find` returns 9 for "options" and 7 for
"lengths".

## FindOfAnUndeclaredNameReturnsNoValue

A name that is only a prefix of a declared name, or the declared name with a trailing space, finds
nothing.

## LeavesComeBackInTheOrderTheyWereAdded

Three leaves added as b, a, c come back as b, a, c with their values.
