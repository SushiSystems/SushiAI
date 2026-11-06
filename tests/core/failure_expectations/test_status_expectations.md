# tests/core/failure_expectations/test_status_expectations.cpp


Suites: `Small_StatusExpectations`, `Medium_StatusExpectationsFatal`.

Tests the assertions every test program uses to check a `Status`: `status_is_ok`,
`status_names`, `death_pattern`, `require_value` and the macros built on them.

This file needs no device. `short_header` returns an `InvalidFile` failure whose message names
"a.bin" and "short", and every failing case reads that one status.

## StatusIsOkAcceptsASuccessAndQuotesAFailure

`status_is_ok` accepts a default `Status`. For the failure it returns a false result whose
message quotes the failure's message.

## StatusNamesAcceptsTheCodeAndEveryNeedle

`status_names` accepts the failure under `InvalidFile` with the needles "a.bin" and "short", and
with no needle at all.

## StatusNamesRefusesASuccessAnotherCodeAndAMissingNeedle

`status_names` refuses a success, refuses the failure under `FileNotFound`, and refuses the
needle "b.bin", which the message does not hold; the refusal names "b.bin".

## DeathPatternJoinsAndEscapesItsNeedles

`death_pattern` joins its needles with ".*" and puts a backslash before every regular-expression
character, so "[2, 3]" and "(c)*" match literally. No needle gives an empty pattern. One plain
needle is returned as it is, and each of `\ { } | ^ $ + ?` is escaped with a backslash.

## ExpectOkPassesASuccess

`SA_EXPECT_OK` passes for a default `Status`.

## ExpectFailurePassesAFailureHeldAsAStatusAndAsAResult

`SA_EXPECT_FAILURE` passes for the failure of `short_header` as a `Status` and wrapped in a
`Result<int>`.

## AssertOkAndAssignBindsTheValueOfASuccess

`SA_ASSERT_OK_AND_ASSIGN` binds 7 from a successful `Result<int>`.

## RequireValueReturnsTheValueOfAResult

`require_value` returns 7 for a `Result<int>` holding 7.

## RequireValueReturnsTheReferenceOfAReferenceResult

`require_value` returns the same object for a `Result<int&>` over a local.

## RequireValueStopsTheProcessOnAFailure

Suite `Medium_StatusExpectationsFatal`: the case is a death test, which starts a child process.
`require_value` over the failed `Result` of `short_header` stops the process with a report that
holds "fixture is refused" and the failure's message.
