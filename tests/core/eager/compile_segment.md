# tests/core/eager/compile_segment.cpp


A compile-only unit: it holds no test and the runner never lists it. It checks at build time that
`SushiAI/eager/segment.hpp` compiles on its own.

The file needs no device and runs nothing. `Segment` has no behaviour of its own; `test_scope_trace.md`
covers the segments a trace yields.
