# tests/core/autograd/compile_tape.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`GradientTape` in `modules/core/autograd/include/SushiAI/autograd/tape.hpp` is not
default-constructible, so a tape always belongs to a differentiation run.

The file needs no device and runs nothing.
