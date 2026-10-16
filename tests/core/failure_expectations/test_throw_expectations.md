# tests/core/failure_expectations/test_throw_expectations.cpp


Suites: `Small_ThrowExpectations`.

Tests `capture_error`, `error_names` and `SA_EXPECT_ERROR`, which check what a thrown
`SushiAI::Error` names.

This file needs no device. Three file-local functions are the statements the cases run:
`throw_rank_error` throws an `Error` naming "conv1" and "rank 4", `throw_out_of_range` throws a
standard exception, and `do_nothing` returns.

## CaptureErrorRecordsAnErrorAndItsMessage

The outcome of `throw_rank_error` has `threw_error` set, `threw_other` clear and the whole
message.

## CaptureErrorTellsAForeignExceptionFromNoThrow

The outcome of `throw_out_of_range` has `threw_other` set and its text in `other_what`. The
outcome of `do_nothing` has neither flag set.

## ErrorNamesAcceptsEveryNeedleAndRefusesTheRest

`error_names` accepts the `Error` with the needles "conv1" and "rank 4" and refuses it with
"conv2". It refuses an outcome that did not throw and one that threw a standard exception.

## TheMacroPassesForANamingError

`SA_EXPECT_ERROR` passes for `throw_rank_error` with both needles.
