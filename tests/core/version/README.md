# tests/core/version

The tests of the module `core/version`, one file per header of `modules/core/version/include/`. The suite prefix of a case gives its ctest label.

## test_version.cpp

Suites: `Unit_Version`.

Tests that the runtime version agrees with the compile-time macros.

### MatchesEncodedMacro

`SUSHIAI_VERSION` equals `SUSHIAI_VERSION_ENCODE` of the major, minor and patch macros.

### RuntimeStringMatchesCompileTimeMacro

`version_string()` equals `SUSHIAI_VERSION_STRING`.
