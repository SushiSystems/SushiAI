# failure_expectations

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

Assertions over a failure: a `Status` checked by its code and by the words its message
must contain, and a thrown `Error` checked by the words of its message.

`SA_EXPECT_FAILURE` takes the expected `StatusCode` and any number of needles. A failure with
another code, or a message without a needle, fails with the code and the message it did carry.
`SA_EXPECT_OK` checks a success the same way, and `SA_ASSERT_OK_AND_ASSIGN` binds the value of a
`Result` to a declaration or fails the test with the message. `SA_EXPECT_FATAL` runs a statement
that stops the process and matches the report against the needles; `death_pattern` joins the
needles in order and escapes the regular-expression characters in each.

`SA_EXPECT_ERROR` does the same for a thrown `Error`. `EXPECT_THROW` proves only that something
went wrong; every `SA_THROW_IF` in the library names the actual problem, and a refactor that
collapses those messages into one keeps every `EXPECT_THROW` passing. `capture_error` records a
foreign exception without propagating it, so the assertion can say what was thrown instead.
`SA_EXPECT_ERROR` needs its statement parenthesised when the statement holds a top-level comma:
`SA_EXPECT_ERROR((Shape{1, 2}), "...")`. `throw_expectations.hpp` includes no SushiBLAS engine
header and no SYCL header, so a device-free test can use it.

## Depends on

`failure`, for `Status`, `Result` and `Error`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/failure_expectations/status_expectations.hpp` | `status_is_ok`, `status_names`, `failure_names`, `death_pattern`, `SA_EXPECT_FAILURE`, `SA_EXPECT_OK`, `SA_ASSERT_OK_AND_ASSIGN`, `SA_EXPECT_FATAL` |
| `SushiAI/failure_expectations/throw_expectations.hpp` | `ThrowOutcome`, `capture_error`, `error_names`, `SA_EXPECT_ERROR` |
