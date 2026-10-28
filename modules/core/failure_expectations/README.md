# failure_expectations

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

Assertions over a failure: a `Status` checked by its code and by the words its message
must contain.

`SA_EXPECT_FAILURE` takes the expected `StatusCode` and any number of needles. A failure with
another code, or a message without a needle, fails with the code and the message it did carry.
`SA_EXPECT_OK` checks a success the same way, and `SA_ASSERT_OK_AND_ASSIGN` binds the value of a
`Result` to a declaration or fails the test with the message. `SA_EXPECT_FATAL` runs a statement
that stops the process and matches the report against the needles; `death_pattern` joins the
needles in order and escapes the regular-expression characters in each.
`require_value` returns the value of a `Result`, or stops the process with the failure it holds: a
test helper that returns a value uses it for a fixture the test wrote, where an assertion cannot
be used.

## Depends on

`failure`, for `Status` and `Result`, and GTest.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/failure_expectations/status_expectations.hpp` | `status_is_ok`, `status_names`, `failure_names`, `death_pattern`, `require_value`, `SA_EXPECT_FAILURE`, `SA_EXPECT_OK`, `SA_ASSERT_OK_AND_ASSIGN`, `SA_EXPECT_FATAL` |
