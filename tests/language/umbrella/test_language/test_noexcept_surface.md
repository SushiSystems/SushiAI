# tests/language/umbrella/test_language/test_noexcept_surface.cpp


Suites: `Small_NoexceptSurface`.

Tests that no public function of the language modules lets an exception out. The file includes the
public headers of the modules the language test program reaches and holds one `static_assert` per
function: 110 assertions over 33 headers. Each asserts `noexcept` of a call whose arguments are
`std::declval` of the declared parameter types, so overload resolution picks the declaration the
caller gets. A function whose declaration loses its `noexcept` stops the build of this file.

This file needs no device. The one case, `EveryPublicFunctionIsNoexcept`, runs nothing: the
assertions are the test, and the case only gives the program a case of its size.

What the file asserts is every non-virtual, non-template public function that carries `noexcept`:
the free functions, static functions, constructors and member functions of the headers under
`modules/language/*/include/`. The core and layers modules have the same test in
`tests/layers/umbrella/test_SushiAI/test_noexcept_surface.cpp`.

What it leaves out:

- Virtual functions and destructors. Their `noexcept` is part of the interface they override.
- Function templates and members of class templates. A call needs types the file cannot name
  for every instantiation.
- The header of `decoder_models` (2 functions). The include path of this test program does not
  reach that module.
