# tests/core/data/compile_roles.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that `SushiAI/data/roles.hpp` compiles on its own.

The file needs no device and runs nothing. It keeps `roles.hpp` under a file of its own in this
folder, which holds one test file per header.
