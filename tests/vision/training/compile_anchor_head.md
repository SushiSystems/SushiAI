# tests/vision/training/compile_anchor_head.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`AnchorHead` in `modules/vision/training/include/SushiAI/vision/training/anchor_head.hpp` is
copy constructible and nothrow move constructible, since an anchor objective keeps a copy of every
head and a list of heads grows by moving.

`AnchorHead` is plain data with no behaviour of its own. The two cases that stated its default
values and its copy were deleted: a default value and a compiler-generated copy are not promises.
`ARuleNamesAHeadByItsOutput` was deleted because the best-shape rule's own file refuses a head by
its output with the same kind of refusal (`test_best_shape_head_delta_rule.cpp`).

The file needs no device and runs nothing. Decision 48 of the report allows a compile-only unit
for a header that declares no function, so that `check_test_mirror.py` still finds a unit for it.
