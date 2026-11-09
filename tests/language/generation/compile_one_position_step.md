# tests/language/generation/compile_one_position_step.cpp


A compile-only unit: it holds no test and the runner never lists it. It compiles
`modules/language/generation/include/SushiAI/language/generation/one_position_step.hpp` on its own and checks at build time that OnePositionStep is a port that lists its fed leaves, and that InMemoryOnePositionStep implements it.

The file needs no device and runs nothing.
