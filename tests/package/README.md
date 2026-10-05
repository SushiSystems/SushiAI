# tests/package

An out-of-tree consumer of the installed SushiAI package. `sa test --suite package` installs
the project to a prefix under `build/`, configures this folder against that prefix and runs
the executable; [`cli/README.md`](../../cli/README.md) describes that suite.

## main.cpp

The program is the smallest thing that proves the package works. It runs one real operation,
an `axpy`, through both siblings that the package resolves transitively (SushiBLAS and
SushiRuntime), against an installed SushiAI, from a project with no access to this source
tree.
