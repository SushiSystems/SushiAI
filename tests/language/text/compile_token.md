# tests/language/text/compile_token.cpp


A compile-only unit: it holds no test and the runner never lists it. It compiles
`modules/language/text/include/SushiAI/language/text/token.hpp` on its own and checks at build time
that `TokenID` is `std::int32_t`, the type every id tensor and every token file holds.

The file needs no device and runs nothing.
