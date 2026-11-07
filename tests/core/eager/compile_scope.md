# tests/core/eager/compile_scope.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
a `Scope` guard is neither copy- nor move-constructible nor assignable, so no second object can
close the scope a guard opened.

The file needs no device and runs nothing. `test_scope.md` covers when a guard opens and closes
its scope.
