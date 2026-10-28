# tests/vision/umbrella/test_vision/test_noexcept_surface.cpp


Suites: `Small_NoexceptSurface`.

Tests that no public function of the vision modules lets an exception out. The file includes the
public headers of the modules the vision test program reaches and holds one `static_assert` per
function: 418 assertions over 162 headers. Each asserts `noexcept` of a call whose arguments are
`std::declval` of the declared parameter types, so overload resolution picks the declaration the
caller gets. A function whose declaration loses its `noexcept` stops the build of this file.

This file needs no device. The one case, `EveryPublicFunctionIsNoexcept`, runs nothing: the
assertions are the test, and the case only gives the program a case of its size.

What the file asserts is every non-virtual, non-template public function that carries `noexcept`:
the free functions, static functions, constructors and member functions of the headers under
`modules/vision/*/include/`. The core and layers modules have the same test in
`tests/layers/umbrella/test_SushiAI/test_noexcept_surface.cpp`.

What it leaves out:

- Virtual functions and destructors. Their `noexcept` is part of the interface they override.
- Function templates and members of class templates. A call needs types the file cannot name
  for every instantiation.
- `use_sample_cache` of `datasets`, which calls `std::filesystem` without an error code and so
  can still throw.
- `resample_polygons`, `to_pixel_labels`, `to_normalized_labels` and `remove_empty_boxes` of
  `imaging`, whose declarations the wave-8 script could not pair with one definition. They stay
  as they were.
- The headers of `darknet_forward_cases`, `darknet_import_cases` and `darknet_training_cases`
  (41 functions). They are test-support modules, and the include path of this test program does
  not reach them.
