# tests/language/generation/compile_fed_leaves.cpp


A compile-only unit: it holds no test and the runner never lists it. It compiles
`modules/language/generation/include/SushiAI/language/generation/fed_leaves.hpp` on its own and checks at build time that FedLeaves is a port, and that InMemoryFedLeaves implements it.

The file needs no device and runs nothing.
