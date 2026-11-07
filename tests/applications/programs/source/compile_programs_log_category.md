# tests/applications/programs/source/compile_programs_log_category.cpp

A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`programs_log_category.hpp` names the project `sushiai` and the module `programs`.

The file needs no device and runs nothing. It keeps the header under a file of its own in this
folder, which holds one test file per unit.
