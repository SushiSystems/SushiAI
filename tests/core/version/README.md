# tests/core/version

The tests of the module `core/version`, one file per header of `modules/core/version/include/`. The suite prefix of a case gives its ctest label.

## test_version.cpp

Suites: `Small_Version`.

Tests the runtime version string.

### TheVersionStringIsNeverNull

`version_string()` returns a pointer that is not null.
