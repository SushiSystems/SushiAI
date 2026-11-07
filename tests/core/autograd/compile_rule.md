# tests/core/autograd/compile_rule.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`BackwardRule` in `modules/core/autograd/include/SushiAI/autograd/rule.hpp` is a pointer to a
function of a `const GradientTape&`.

The file needs no device and runs nothing.
