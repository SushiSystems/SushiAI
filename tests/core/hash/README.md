# tests/core/hash

The tests of the module `core/hash`, one file per header of `modules/core/hash/include/`. The
suite prefix of a case gives its ctest label.

## test_fnv1a.cpp

Suites: `Small_FNV1a`.

Tests the folds of `SushiAI/hash/fnv1a.hpp` against values computed apart from it.

### TheTextFoldGivesThePublishedValues

`fnv1a_text` from `K_FNV1A_OFFSET_BASIS` gives 0xcbf29ce484222325 for the empty text,
0xaf63dc4c8601ec8c for "a" and 0x85944171f73967e8 for "foobar".

### AWordFoldsAsItsBytesLowFirst

`fnv1a_word` of 0x0123456789abcdef equals `fnv1a_text` of its eight bytes, low byte first.

### ThreeWordsKeepTheFingerprintOfTheReplicaStep

Folding the words 1, 2 and 0x0123456789abcdef gives 0x78165499fd352566.

### TheFoldIsAConstantExpression

`fnv1a_text` of "a" is a `constexpr` value that a `static_assert` compares with 0xaf63dc4c8601ec8c.
