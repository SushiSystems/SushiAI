# tests/vision/datasets/compile_sample.cpp


A compile-only unit: it holds no test and the runner never lists it.

Tests the shape of `Sample` and `SampleList` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample.hpp`. The header declares two
structs and no function, so no code of the unit runs. The default-value cases and the indexing
case of the earlier `test_sample.cpp` are deleted: they pinned the defaults of a struct and the
indexing of `std::vector`. What each reader puts in a list is tested in the reader's own file,
such as `test_yolo_txt.cpp` and `test_mask_png.cpp`.

The file asserts at namespace scope that both structs are aggregates, that `Sample::file` and
`Sample::mask` are indices into `SampleList::files` of type `std::uint32_t`, and that a list holds
its files as paths and its samples as `Sample` records. It needs no device and runs nothing; the
program compiles it.
