# tests/package

An out-of-tree consumer of the installed SushiAI package. `sa test --label package` installs
the project to a prefix under `build/`, configures this folder against that prefix and runs
the executable; [`cli/README.md`](../../cli/README.md) describes that suite.

## CMakeLists.txt

Configures this folder as a consumer of the installed package. It calls `find_package(SushiAI
REQUIRED)` and nothing else, so it cannot reach this source tree. It also fails when the exported
interface of `SushiAI::sushiai` carries a compiler-policy flag, such as a warning level, `-march`
or a sanitizer, because a consumer would inherit that flag.

## main.cpp

The program is the smallest thing that proves the package works. It runs one real operation,
an `axpy`, through both siblings that the package resolves transitively (SushiBLAS and
SushiRuntime), against an installed SushiAI, from a project with no access to this source
tree.
