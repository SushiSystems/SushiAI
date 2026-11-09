# tests/language/generation/compile_window_logits.cpp


A compile-only unit: it holds no test and the runner never lists it. It compiles
`modules/language/generation/include/SushiAI/language/generation/window_logits.hpp` on its own and checks at build time that WindowLogits is a port, and that InMemoryWindowLogits implements it.

The file needs no device and runs nothing.
