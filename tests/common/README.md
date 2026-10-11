# tests/common

The pieces every test binary in this repository shares: the `main()`, the GTest environment,
the per-test fixture and the assertion helpers.

| File | Holds |
| --- | --- |
| `test_main.cpp` | The test binary's `main()` |
| `test_environment.hpp` | `TestEnvironment`, owner of the shared `SushiRuntime::Runtime` |
| `test_utils.hpp` | `EngineTest`, the tensor read and write helpers, `HostNoise` |
| `test_status.hpp` | `SA_EXPECT_OK`, `SA_EXPECT_FAILURE`, `SA_ASSERT_OK_AND_ASSIGN`, `SA_EXPECT_FATAL` and the functions under them |
| `test_throws.hpp` | `SA_EXPECT_ERROR` and the functions under it |
| `reference_fixture.hpp` | `ReferenceFile`, the YOLO26 reference readers and the comparisons against them |

## Running the suite

`sushiai_tests` holds 1803 cases. `sa run sushiai_tests` runs all of them in one process, so
the `Runtime` that `TestEnvironment` owns is created once. `sa test` runs the same binary
through CTest, which starts it once for each case it discovered, and assigns each case the label
of its suite prefix: `Unit_`, `Integration_` or `Regression_`.

SushiBLAS's own tests do not enter this tree. Its `SB_BUILD_TESTS` option defaults to
`SB_IS_TOP_LEVEL` (`CMakeLists.txt`, line 137 of the SushiBLAS repository), so the option is off
when SushiBLAS is built as a dependency of SushiAI.

The sections below hold the reasoning that the source files cite.

## test_environment.hpp

`Runtime::create()` starts a worker pool, probes the topology and builds SYCL queues. Doing
that per test would dominate the suite's runtime, and it would repeatedly tear down and
restart the thread pool while other work is settling. The suite therefore keeps one
process-global runtime in a GTest `Environment` and gives each test a fresh `Engine`, so no
test inherits another's graph or RNG offset. `SetUp()` fails the run when the runtime
cannot be created. SushiBLAS's own suite, in its
`tests/common/test_environment.hpp`, uses the same split.

## test_main.cpp

The custom `main()` exists so the `Runtime` is created once for the whole binary and
not once per test. `tests/CMakeLists.txt` does not link `GTest::gtest_main`. Linking both
would be a duplicate-symbol error at link time, which is the good failure mode.

Both loggers are switched to synchronous mode. Asynchronous logging interleaves with GTest's
own output and makes a failure report unreadable. The sibling suites make the same choice.

## test_status.hpp

The status assertions check a `Status` by its code and by the words its message must contain.
`SA_EXPECT_FAILURE` takes the expected `StatusCode` and, optionally, the needles. A failure with
another code, or a message without a needle, fails with the code and the message that it did
carry. `SA_EXPECT_OK` checks a success the same way, and `SA_ASSERT_OK_AND_ASSIGN` binds the value
of a `Result` to a declaration or fails the test with the message.

The needles do the work that `SA_EXPECT_ERROR` does for a thrown `Error`: a status that stops
naming its cause fails the test that covers it.

`SA_EXPECT_FATAL` runs a statement that stops the process and matches the report against the
needles. `death_pattern` joins the needles in order and escapes the regular-expression characters
in each one. The `*DeathTest` suites use it.

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

The header is host-only. It includes no SushiBLAS engine header and no SYCL header. Through
`SushiAI/core/common.hpp` it reaches two SushiBLAS core headers, which use only the standard
library, so the device-free unit tier can use it too.

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

## reference_fixture.hpp

This header reads the YOLO26 reference set under `tests/fixtures/yolo26/`: the `.sacp` files,
which use the checkpoint format, and `MANIFEST.json`. `ReferenceFile` opens one file and indexes
its records by tensor name. `SA_OPEN_REFERENCE_OR_SKIP` opens a file or skips the test and names
the file that is missing. `SA_READ_REFERENCE_OR_FAIL` reads one named tensor or fails the test.
`near_reference` compares float results within a tolerance, and `equal_reference` compares
integer results element by element.

The fixture directory comes from the `SA_TEST_FIXTURE_DIR` definition that `tests/CMakeLists.txt`
sets.
