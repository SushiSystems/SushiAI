# tests/core/failure

The tests of the module `core/failure`, one file per header of `modules/core/failure/include/`. The
suite prefix of a case gives its ctest label.

## test_error.cpp

The cases are described in [`test_error.md`](test_error.md).

## test_result.cpp

Suites: `Small_Result`.

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

`Result<int>(7)` has a value, an ok status and the value 7. Writing 9 through `value()` is
read back through a const reference.

### HoldsAFailureAndNoValue

A result built from the failure "rank must be 2" has no value. `SA_EXPECT_FAILURE` accepts
both the result and its `status()`.

### ReturnStatementsConvertFromAValueAndFromAStatus

`checked_extent(5)` yields 5 through `SA_ASSERT_OK_AND_ASSIGN`, and `checked_extent(-1)` is an
`InvalidDescription` failure containing "extent must be positive" and "got -1". The function
returns an `int` on one path and the `Status` of `SA_FAIL_IF` on the other.

### SuccessPathCreatesNoFailureBlock

A made, a copied and a moved `Result<int>` and one `SA_TRY_ASSIGN` through `doubled_extent`
sum to 13 and leave `live_failure_count()` where it was.

### RvalueValueMovesAMoveOnlyValueOut

`std::move(result).value()` on a `Result<std::unique_ptr<int>>` returns a pointer that is not
null and owns 11.

### MovedFromValueResultStillHoldsAMovedFromValue

After a `Result<std::unique_ptr<int>>` is moved from, the target owns 2. The source still
reports a value and an ok status, and its pointer is null.

### MovedFromFailureResultKeepsItsFailure

After a failed `Result<int>` is moved from, the source and the target both report
`InvalidDescription` and "layout must be NHWC", and the source has no value.

### CopyAndAssignmentCarryTheHeldState

Assigning a failed result to one that holds 1 leaves no value and the failure "no such key".
Assigning `Result<int>(8)` to it afterwards leaves a value, an ok status and 8.

### TryAssignBindsTheValueOrReturnsTheFailure

`doubled_extent(21)` is 42. `doubled_extent(0)` is the failure of `checked_extent`, with
"got 0" in its message.

### TryAssignMovesFromATemporaryResult

`unboxed_extent(6)` is 6, which requires `SA_TRY_ASSIGN` to move the `unique_ptr` out of the
temporary result. `unboxed_extent(-6)` is the failure with "got -6".

### ReferenceFormRefersToTheCallersObject

`slot_at(slots, 1)` has a value whose address is `&slots[1]`, and writing 25 through it
changes the array. A `Result<const int&>` built from a const `int` returns that object's
address.

### TryReadsAStatusReferenceIntoATemporaryResult

`forward_temporary_status(-4)` returns the failure with "extent must be positive" and
"got -4" and leaves its flag false. `forward_temporary_status(4)` succeeds and sets it.
`live_failure_count()` is at its start after each call.

### MovedFromReferenceResultKeepsItsState

A `Result<int&>` to slot 0 is copied and moved from. The source, the copy and the target all
refer to `&slots[0]`. A failed reference result moved from keeps the failure "slot 2 is not
one of 2" in both source and target. Assigning the failed one over a result to slot 1 gives
the failure, and assigning the successful one over that gives `&slots[0]` and an ok status.

### ReferenceFormHoldsAFailure

`slot_at(slots, 2)` has no value and the `InvalidDescription` failure "slot 2 is not one of 2".

### TryAssignBindsAReferenceFromTheReferenceForm

`bump_slot(slots, 0)` succeeds and slot 0 goes from 10 to 11, so the reference `SA_TRY_ASSIGN`
bound is the array's element. `bump_slot(slots, 5)` fails with "slot 5" and slot 1 stays 20.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Result<int>` and on a failed `Result<int&>` stops the process. The
report contains "Result::value() was called on a failure" and the failure's own message.

### BuildingFromASuccessStatusIsFatal

`Result<int>(Status())` and `Result<int&>(Status())` stop the process with "a Result was built
from a success Status".

## test_status.cpp

Suites: `Small_Status`.

Tests Status, its macros, the fatal family and the assertions of test_status.hpp.

This file needs no device. `Status` and the `SA_FAIL_IF`, `SA_TRY` and `SA_ANNOTATE` macros are
in `modules/core/failure/include/SushiAI/failure/status.hpp`, and the fatal checks are in
`modules/core/failure/include/SushiAI/failure/check.hpp`. The assertions `SA_EXPECT_OK`,
`SA_EXPECT_FAILURE` and
`SA_EXPECT_FATAL` come from
`modules/core/failure_expectations/include/SushiAI/failure_expectations/status_expectations.hpp`.
`SA_EXPECT_FATAL` is a death test:
the statement runs in a child process and the needles are matched against what it wrote to
stderr.

Six `static_assert` lines at the top of the file hold that a `Status` is the size of one
pointer and that its default construction, copy, move and both assignments are `noexcept`.

Several cases read `Detail::live_failure_count()`. A failed `Status` owns one failure block and
a successful one owns none, so a count that did not move means the code under test allocated
nothing for a failure.

### counted

`counted` returns its value and increments a counter. A case passes it as a format argument to
find out whether a macro evaluated its arguments.

### failure_of

`failure_of` builds a failure of a given code and message through `Status::from_message`.

### check_extent

`check_extent` fails with `StatusCode::InvalidDescription` through `SA_FAIL_IF` when its extent
is not positive. The message's argument goes through `counted`.

### forward_extent

`forward_extent` calls `check_extent` under `SA_TRY` and sets a flag after it, so a case can
tell whether control went past the macro.

### FatalRecord

`FatalRecord` holds the call count, file, line and message of the last report the recording
handler received.

### fatal_record

`fatal_record` returns the one static `FatalRecord` of the file.

### record_fatal

`record_fatal` is a `FatalHandler` that stores its report in `fatal_record()`.

### announce_fatal

`announce_fatal` is a `FatalHandler` that writes "recorded: " and the message to stderr, where
a death test reads it.

### stop_when

`stop_when` runs `SA_FATAL_IF` on its condition with the message "extent {} is refused" and
returns the line number of that check. The extent goes through `counted`.

### stop_with_throwing_composer

`stop_with_throwing_composer` calls `Detail::fail_fast_composed` with a composer that throws
`std::runtime_error`.

### stop_under_announcer

`stop_under_announcer` installs `announce_fatal` and then runs `SA_FATAL("stop {}", 3)`.

### assert_positive

`assert_positive` runs `SA_ASSERT(value > 0)`. It is compiled only when `NDEBUG` is not
defined.

### DefaultIsSuccessWithNoMessage

A default-constructed `Status` reports `is_ok()`, the code `StatusCode::Ok` and an empty
message, and `SA_EXPECT_OK` accepts it.

### EveryCodeHasALowerCaseName

`status_code_name` returns "ok", "invalid_description", "file_not_found", "invalid_file",
"incompatible", "unsupported", "out_of_memory" and "device_failed" for the eight codes, and
"unknown" for the value 200, which is no enumerator.

### FromMessageCarriesTheCodeAndTheText

A status built from `StatusCode::InvalidFile` and "the header is 3 bytes short" is not ok and
returns that code and that text.

### SuccessPathCreatesNoFailureBlock

The case constructs, copies, moves, copy-assigns and move-assigns successful statuses and
calls `annotated` on one. The annotation's composer never runs, both results are ok, and
`live_failure_count()` is what it was before.

### FailIfFormatsOnlyWhenTheConditionHolds

`check_extent(4)` succeeds with zero calls to `counted` and no failure block. `check_extent(-2)`
fails with `InvalidDescription` and a message containing "extent must be positive" and
"got -2", after one call to `counted`. The count of live blocks is back at its start once the
temporary is gone.

### CopiesShareOneBlockAndTheLastOneFreesIt

A failure, a copy of it and a status assigned from the copy leave `live_failure_count()` one
above its start, and the third status's `message().data()` is the first one's pointer. After
the scope closes the count is back at its start.

### MovedFromStatusKeepsItsFailure

A failed status is moved from twice, once by construction and once by assignment. The source
and both targets still report `Unsupported` and "no kernel for int64".

### AssignmentReplacesTheHeldFailure

Assigning a second failure to a status that holds one leaves one live block and the second
failure's code and message. A self-assignment through a const reference keeps the failure.
Assigning `Status()` makes the status ok and frees the block.

### AnnotatedPrefixesTheMessageAndKeepsTheCode

`SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16)` over the failure "stride must be positive"
returns a status with the same code and the message "Conv2d(3, 16): stride must be positive".
The inner status's message is unchanged.

### ThrowingComposerYieldsTheStaticOutOfMemoryFailure

`Status::from_message` with a composer that throws returns a failure of code `OutOfMemory`
whose message contains "message could not be built". `live_failure_count()` does not move, so
that failure is not a heap block.

### ThrowingAnnotationKeepsTheOriginalFailure

`annotated` with a composer that throws returns the code `InvalidFile` and the message "wrong
magic" of the status it was called on.

### TryForwardsAFailureAndPassesASuccess

`forward_extent(0)` returns `check_extent`'s failure, with "got 0" in the message, and leaves
its flag false. `forward_extent(5)` succeeds and sets the flag.

### FromMessageWithTheOkCodeIsFatal

`Status::from_message(StatusCode::Ok, ...)` stops the process with a report containing
"Status::from_message needs a failure code" and "got ok".

### StatusNamesRejectsSuccessAWrongCodeAndAMissingNeedle

`Test::status_names` is called directly. Over the failure `InvalidFile` "wrong magic 0x1F" it
is true for the right code with both needles and with no needle. It is false for a successful
status, for the code `FileNotFound` and for the needle "wrong version". `Test::status_is_ok` is
false for the failure.

### DeathPatternEscapesAndJoinsItsNeedles

`Test::death_pattern` is called directly. One plain needle is returned as it is. Two needles
are joined with `.*`, and each of the characters `. [ ] ( ) * { } | ^ $ + ? \` is escaped with
a backslash.

### FatalHandlerIsReplacedByARecorderAndRestored

`set_fatal_handler(&record_fatal)` returns a handler that is not null. One call of
`Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused")` reaches the recorder
once, with that file, line and message. Restoring the original returns `&record_fatal`, and
setting the original again returns the original.

### NullFatalHandlerInstallsTheDefaultOne

`set_fatal_handler(nullptr)` returns the handler that was installed, `&record_fatal`. The
handler in place after it is not null and is not the recorder.

### FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse

`stop_when(false, ...)` returns and never calls `counted`.

### FatalIfReportsTheMessageAtTheFileAndLineOfTheCheckAndStops

`stop_when(true, ...)` stops the process with a FATAL record of `[sushiai.failure]` whose call
site is `[test_status.cpp:<line>]`, the line `stop_when` returned, and whose message is "extent 7
is refused". The case calls `stop_when(false, ...)` first to learn that line.

### FatalCheckWhoseComposerThrowsReportsTheFixedMessage

`stop_with_throwing_composer` stops the process with a FATAL record of `[sushiai.failure]` at a
call site in `test_status.cpp`, with the message "its message could not be composed".

### ReplacedFatalHandlerRunsBeforeTheStop

`stop_under_announcer` stops the process and its stderr contains "recorded: stop 3", the line
`announce_fatal` writes.

### AssertChecksInDebugBuildsOnly

With `NDEBUG` defined, `SA_ASSERT(counted(evaluations, 0) == 1)` does not evaluate its
condition: the counter stays at zero. Without it, `assert_positive(1)` returns and
`assert_positive(0)` stops the process with "assertion failed: value > 0".
