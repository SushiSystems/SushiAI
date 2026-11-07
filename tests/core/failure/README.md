# tests/core/failure

The tests of the module `core/failure`, one file per header of `modules/core/failure/include/`. The
suite prefix of a case gives its ctest label.

## test_result.cpp

Suites: `Small_Result`, `Medium_ResultFatal`.

Tests Result, its reference form and SA_TRY_ASSIGN.

This file needs no device. `Result<T>` and `SA_TRY_ASSIGN` are in
`modules/core/failure/include/SushiAI/failure/result.hpp`. A `Result<T>` holds a value or a failed
`Status`, and
`Result<T&>` holds a reference to the caller's object or a failed `Status`.

Nine `static_assert` lines at the top of the file hold the type's shape. Neither form is
default-constructible. Construction of a `Result<int>` from an `int` and from a `Status`, and
its move, are `noexcept`. A `Result<int>` is at most two pointers and a `Result<int&>` is
exactly two. Neither `Result<int&>` nor `Result<const int&>` is constructible from an rvalue
`int`, so a reference form cannot refer to a temporary.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### checked_extent

`checked_extent` returns its extent as a `Result<int>`, or an `InvalidDescription` failure from
`SA_FAIL_IF` when the extent is not positive.

### doubled_extent

`doubled_extent` binds `checked_extent`'s value through `SA_TRY_ASSIGN` and returns twice it.

### boxed_extent

`boxed_extent` returns a `std::unique_ptr<int>` holding the checked extent, a move-only value.

### unboxed_extent

`unboxed_extent` binds `boxed_extent`'s pointer through `SA_TRY_ASSIGN` and returns the
integer it owns.

### forward_temporary_status

`forward_temporary_status` runs `SA_TRY(checked_extent(extent).status())`, which reads a
`Status` reference into a temporary `Result`, and sets a flag after it.

### slot_at

`slot_at` returns a `Result<int&>` to one of two slots, or an `InvalidDescription` failure for
an index outside 0 and 1.

### bump_slot

`bump_slot` binds `int& slot` from `slot_at` through `SA_TRY_ASSIGN` and increments it.

### HoldsAValueWithASuccessStatus

`Result<int>(7)` has a value, an ok status and the value 7.

### AWriteThroughValueIsReadBackThroughTheConstView

Writing 9 through `value()` of a `Result<int>` is read back through a const reference.

### HoldsAFailureAndNoValue

A result built from the failure "rank must be 2" has no value. `SA_EXPECT_FAILURE` accepts
both the result and its `status()`. A `Result<int&>` from `slot_at(slots, 2)` has no value and the
`InvalidDescription` failure "slot 2 is not one of 2".

### AReturnStatementConvertsFromAValue

`checked_extent(5)` yields 5 through `SA_ASSERT_OK_AND_ASSIGN`: the function returns an `int`
where a `Result<int>` is declared.

### AReturnStatementConvertsFromAFailedStatus

`checked_extent(-1)` is an `InvalidDescription` failure containing "extent must be positive" and
"got -1": the function returns the `Status` of `SA_FAIL_IF` where a `Result<int>` is declared.

### SuccessPathCreatesNoFailureBlock

A default `Status` has the code `Ok` and an empty message. The case constructs, copies, moves,
copy-assigns and move-assigns successful statuses and calls `annotated` on one. The annotation's
composer never runs, both results are ok, and an `InMemoryFailureBlockCounter` installed before
the case counts no live block.

### FailIfFormatsOnlyWhenTheConditionHolds

`check_extent(4)` succeeds with zero calls to `counted`. `check_extent(-2)` fails with
`InvalidDescription` and a message containing "extent must be positive" and "got -2", after one
call to `counted`.

### CopiesShareOneBlockAndTheLastOneFreesIt

A failure, a copy of it and a status assigned from the copy leave an
`InMemoryFailureBlockCounter` at one live block, and the third status's `message().data()` is
the first one's pointer. After the scope closes the counter reads zero.

### MovedFromStatusKeepsItsFailure

A failed status is moved from twice, once by construction and once by assignment. The source
and both targets still report `Unsupported` and "no kernel for int64".

### AssignmentReplacesTheHeldFailure

Assigning a second failure to a status that holds one leaves one live block on an
`InMemoryFailureBlockCounter`, and the second failure's code and message. A self-assignment through a const reference keeps the failure.
Assigning `Status()` makes the status ok and frees the block.

### AnnotatedPrefixesTheMessageAndKeepsTheCode

`SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16)` over the failure "stride must be positive"
returns a status with the same code and the message "Conv2d(3, 16): stride must be positive".
The inner status's message is unchanged.

### ThrowingComposerYieldsTheStaticOutOfMemoryFailure

`Status::from_message` with a composer that throws returns a failure of code `OutOfMemory`
whose message contains "message could not be built".

### ThrowingAnnotationKeepsTheOriginalFailure

`annotated` with a composer that throws returns the code `InvalidFile` and the message "wrong
magic" of the status it was called on.

### TryForwardsAFailureAndPassesASuccess

`forward_extent(0)` returns `check_extent`'s failure, with "got 0" in the message, and leaves
its flag false. `forward_extent(5)` succeeds and sets the flag.

### ReportFatalDeliversFileLineAndMessageOnceToTheInstalledSink

With an `InMemoryFatalSink` installed, one call of
`Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused")` leaves one report in the
sink, with that file, line and message.

### NullFatalHandlerInstallsTheDefaultOneThatLogsAtFatal

With an `InMemoryFatalSink` installed, `set_fatal_handler(nullptr)` replaces it with the default
handler. Calling that handler with "layer.cpp", 42 and "a sealed trace was reused" under a
`SushiLog::LogCapture` leaves one record at `Fatal` in `sushiai.failure`, with that file, line
and message.

### TheHandlerRunsBeforeTheStop

`report_then_stop` with an `InMemoryFatalSink` installed and a stop function that
returns: the stop function sees one report in the sink, and the sink holds the message "stop 3".

### FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse

`stop_when(false, ...)` returns and never calls `counted`.

### AssertDoesNotEvaluateItsConditionInAnNdebugBuild

Compiled only with `NDEBUG` defined: `SA_ASSERT(counted(evaluations, 0) == 1)` does not evaluate
its condition, and the counter stays at zero.

### FromMessageWithTheOkCodeIsFatal

Suite `Medium_StatusFatal`: the cases are death tests, each of which starts a child process.
`Status::from_message(StatusCode::Ok, ...)` stops the process with a report containing
"Status::from_message needs a failure code" and "got ok".

### FatalIfReportsTheMessageAtTheFileAndLineOfTheCheckAndStops

`stop_when(true, ...)` stops the process with a FATAL record of `[sushiai.failure]` whose call
site is `[test_status.cpp:<line>]`, the line `stop_when` returned, and whose message is "extent 7
is refused". The case calls `stop_when(false, ...)` first to learn that line.

### FatalCheckWhoseMessageCannotBeFormattedReportsTheFixedMessage

`stop_with_unformattable_message` stops the process with a FATAL record of `[sushiai.failure]`
at a call site in `test_status.cpp`, with the text "a fatal check failed" and "its message could
not be composed".

### AssertStopsOnAFalseConditionInADebugBuild

Compiled only without `NDEBUG`: `assert_positive(1)` returns and `assert_positive(0)` stops the
process with "assertion failed: value > 0".
