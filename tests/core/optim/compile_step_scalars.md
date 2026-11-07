# tests/core/optim/compile_step_scalars.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
a `StepScalars` buffer is neither copy-constructible nor copy-assignable, because a kernel captures
its address.

The file needs no device and runs nothing. `test_step_scalars.md` covers the move that keeps the
address.
