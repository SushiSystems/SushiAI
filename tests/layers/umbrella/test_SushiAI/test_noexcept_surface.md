# tests/layers/umbrella/test_SushiAI/test_noexcept_surface.cpp


Suites: `Small_NoexceptSurface`.

Tests that no public function of the core and layers modules lets an exception out. The file
includes the public headers of every module the library archives and holds one `static_assert`
per function. Each asserts `noexcept` of a call whose arguments are `std::declval` of the
declared parameter types, so overload resolution picks the declaration the caller gets. A
function whose declaration loses its `noexcept` stops the build of this file.

This file needs no device. The one case, `EveryPublicFunctionIsNoexcept`, runs nothing: the
assertions are the test, and the case only gives the program a case of its size.

What the file asserts is every non-virtual, non-template public function that carries `noexcept`:
the free functions, static functions, constructors and member functions of the headers under
`modules/core/*/include/` and `modules/layers/*/include/` that the `sushiai` library exposes. The
distribution headers sit under `SA_TEST_DISTRIBUTED_BUILD`, which
`tests/layers/umbrella/CMakeLists.txt` defines in a distributed build.

What it leaves out:

- Virtual functions and destructors. Their `noexcept` is part of the interface they override.
- Function templates and members of class templates. A call needs types the file cannot name
  for every instantiation.
- Public functions that stay without `noexcept` because their body can still throw a standard
  exception, such as one that calls `std::filesystem` without an error code.
- The headers of the vision and language libraries and of the test-support modules other than
  `engine_fixture`. This test program links only the `sushiai` library, so their include
  directories are not on its path.
