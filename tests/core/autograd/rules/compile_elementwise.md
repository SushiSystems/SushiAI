# tests/core/autograd/rules/compile_elementwise.cpp

A compile-only unit: it holds no test and the runner never lists it. It checks at build time that no
two entries of `K_ELEMENTWISE_RULES` in
`modules/core/autograd/include/SushiAI/autograd/rules/elementwise.hpp` name the same operation, so
no operation has two backward rules.

The file needs no device and runs nothing.
