# tests/common

The pieces every test binary in this repository shares: the `main()`, the GTest environment,
the per-test fixture and the assertion helpers.

| File | Holds |
| --- | --- |
| `test_main.cpp` | The test binary's `main()` |
| `test_environment.hpp` | `TestEnvironment`, owner of the shared `RuntimeContext` |
| `test_utils.hpp` | `EngineTest`, the tensor read and write helpers, `HostNoise` |
| `test_throws.hpp` | `SA_EXPECT_ERROR` and the functions under it |

The sections below hold the reasoning that the source files cite.

## test_environment.hpp

A `RuntimeContext` starts a worker pool, probes the topology and builds SYCL queues. Doing
that per test would dominate the suite's runtime, and it would repeatedly tear down and
restart the thread pool while other work is settling. The suite therefore keeps one
process-global context in a GTest `Environment` and gives each test a fresh `Engine`, so no
test inherits another's task graph or RNG offset. SushiBLAS's own suite uses the same split.

## test_main.cpp

The custom `main()` exists so the `RuntimeContext` is built once for the whole binary and
not once per test. `tests/CMakeLists.txt` does not link `GTest::gtest_main`. Linking both
would be a duplicate-symbol error at link time, which is the good failure mode.

Both loggers are switched to synchronous mode. Asynchronous logging interleaves with GTest's
own output and makes a failure report unreadable. The sibling suites make the same choice.

## test_throws.hpp

`EXPECT_THROW` proves only that something went wrong, which is the weaker half of the
contract. Every `SA_THROW_IF` in the library names the actual problem: the two shapes that
disagree, the operation with no rule, the parameter nothing can move. That naming is what a
reader depends on at three in the morning, and it is what a refactor silently destroys.
Collapsing five specific diagnostics into one generic "invalid argument" keeps every
`EXPECT_THROW` passing.

The assertions in this header therefore take the substrings the message must contain. A
diagnostic that stops naming the offending value fails the test that covers it, and nobody
has to hit it a year later to notice.

The header is host-only. It includes no SushiBLAS and no SYCL, so the device-free unit tier
can use it too.

`capture_error` records a foreign exception and does not propagate it. The assertion can
then say "threw std::bad_alloc instead of SushiAI::Error", which is a more useful failure
than an aborted test.

`SA_EXPECT_ERROR` needs its statement parenthesised when the statement contains a top-level
comma. The preprocessor honours parentheses but not braces, so
`SA_EXPECT_ERROR((Shape{1, 2}), "...")` is required where
`SA_EXPECT_ERROR(Shape{1, 2}, "...")` would split into three arguments.

## test_utils.hpp

`write_tensor` and `read_tensor` go straight through the shared-USM pointer and not through
a queue memcpy. Every tensor the pool hands out is Shared-strategy USM, and the fixture
waits before it reads, so the pointer is valid and coherent. Reading this way also keeps a
test from queueing work into the engine graph it is in the middle of inspecting.

`read_tensor_bytes` exists because `read_tensor` widens everything to double. Widening is
lossless for float and so it is fine for an equality check, but it cannot distinguish -0.0
from 0.0 and it says nothing about a payload difference between two NaNs. A determinism
claim is about bits, so the check that guards it reads bits.

`HostNoise` is a 64-bit splitmix step mapped to (-1, 1). It is not a good PRNG and is not
meant to be one. It has to be reproducible without bringing a device RNG into a test that is
checking something else.
