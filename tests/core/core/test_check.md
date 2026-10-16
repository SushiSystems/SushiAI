# tests/core/core/test_check.cpp


Suites: `Small_Check`.

Tests the shape of the fatal checks of `modules/core/core/include/SushiAI/core/check.hpp`. The
fatal family itself, `set_fatal_handler`, `Detail::report_fatal`, `SA_FATAL_IF`, `SA_FATAL`
and `SA_ASSERT` stopping the process, is covered by `test_status.cpp` in this folder, so this
file pins only what those cases leave out: each macro expands to exactly one statement.

## branch_around_fatal_if

`branch_around_fatal_if` holds `SA_FATAL_IF` with a false condition as the body of an unbraced
`if` followed by an `else`. It returns 1 when the `if` branch ran and 2 when the `else` branch
ran. A macro that expanded to more than one statement would not compile there, or would bind the
`else` to its own `if`.

## branch_around_assert

`branch_around_assert` does the same with `SA_ASSERT` on a condition that holds.

## FatalIfIsOneStatementInAnUnbracedIfElse

The helper returns 1 for a true flag and 2 for a false one.

## AssertIsOneStatementInAnUnbracedIfElse

The helper returns 1 for a true flag and 2 for a false one, in a build with or without `NDEBUG`.

## TheUncomposedReportSaysAFatalCheckFailed

`Detail::K_UNCOMPOSED_FATAL_MESSAGE`, the text a fatal check reports when composing its own
message threw, contains "fatal check failed".
