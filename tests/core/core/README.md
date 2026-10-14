# tests/core/core

The tests of the module `core/core`, one file per header of `modules/core/core/include/`. The suite prefix of a case gives its ctest label.

## test_ops.cpp

Suites: `Unit_Ops`.

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`modules/core/core/include/SushiAI/core/ops.hpp`. These tests cover what that assert cannot: that the table is
complete, and that the reverse lookup behaves.

### EveryIdentityHashesDistinctly

No two entries of `Ops::K_ALL` have the same `OpID` value.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### NameOfRoundTripsEveryIdentity

`Ops::name_of` returns the name of every entry of `Ops::K_ALL` for its identity.

### NameOfReportsUnknownForAForeignIdentity

`Ops::name_of(OpID(0xDEADBEEF))` is "sushiai.unknown".

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of K_ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

K_ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of K_ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is K_IM2COL + K_MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReLUDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for K_RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

### EveryElementwiseAndSelectionIdentityIsListedInAll

`Ops::name_of` resolves K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_LESS_EQUAL, K_SELECT, K_SILU,
K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD to a name other than "sushiai.unknown". The
reason is the one EverySpatialIdentityIsListedInAll gives.

### TheComparisonAndTheActivationGradientsAreNotFlaggedDifferentiable

`differentiable` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU, and
false for K_LESS_EQUAL, K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD.

### TheGradientBarriersAreTheFourIdentitiesThatPassNoGradient

`Ops::is_gradient_barrier` is true for K_ZERO, K_CONSTANT, K_LESS_EQUAL and K_STOP_GRADIENT. Over every
row of `Ops::K_ALL` it equals the row's `blocks_gradient` flag, and exactly four rows carry the
flag. A fifth barrier therefore fails the count until the list in this test names it.

### NoGradientBarrierIsFlaggedDifferentiable

No row of `Ops::K_ALL` has both `differentiable` and `blocks_gradient` set.

### LeafCastAndTheGradientOperationsAreNotGradientBarriers

`is_gradient_barrier` is false for K_LEAF, K_CAST, K_DYNAMIC_SCALE, K_LOG_SOFTMAX, K_RELU_BACKWARD,
K_SILU_BACKWARD, K_SOFTPLUS_BACKWARD, K_MAXPOOL2D_BACKWARD, K_BATCHNORM2D_BACKWARD,
K_BATCHNORM_UPDATE_STATS, K_BATCHNORM_FOLD, K_GEMM_BIAS, K_EXP_SUB_SCALE and K_ADAMW_UPDATE, and for the
foreign identity 0xDEADBEEF. These either pass a gradient or throw when one reaches them.
Marking one as a barrier would turn that throw into a gradient that is dropped without a
message.

### SiLUDifferentiatesAtItsInput

The `Ops::K_ACTIVATIONS` row of K_SILU names K_SILU_BACKWARD as its gradient and
`GradientSource::Input` as its source, and `activation_survives_fusion(K_SILU)` is false.

### SoftplusDifferentiatesAtItsInput

The row of K_SOFTPLUS names K_SOFTPLUS_BACKWARD and `GradientSource::Input`, and
`activation_survives_fusion(K_SOFTPLUS)` is false.

### IsActivationBackwardAcceptsOnlyTheTablesGradientColumn

For every row of `Ops::K_ACTIVATIONS`, `is_activation_backward` is true for the row's `backward`
identity and false for its `forward` one. Over `Ops::K_ALL` it accepts as many identities as the
table has rows.

### BatchNormFoldIsListedAndClaimsNoRule

`Ops::name_of(K_BATCHNORM_FOLD)` is "sushiai.batchnorm_fold", and the row has `differentiable`
false and `blocks_gradient` false. `test_backward.cpp` holds what differentiation does when a
gradient reaches the node.

## test_result.cpp

Suites: `Unit_Result`.

Tests Result, its reference form and SA_TRY_ASSIGN.

This file needs no device. `Result<T>` and `SA_TRY_ASSIGN` are in
`modules/core/core/include/SushiAI/core/result.hpp`. A `Result<T>` holds a value or a failed `Status`, and
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

## test_shape.cpp

Suites: `Unit_Shape`.

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
runtime, which is what keeps the unit tier fast.

### DefaultIsAScalarWithOneElement

A default `Shape` has rank 0 and one element, and prints as "[]".

### ExtentsAreStoredOutermostFirst

`Shape{64, 784}` has rank 2, `dim(0)` 64, `dim(1)` 784 and 64 * 784 elements, and prints as
"[64, 784]".

### NegativeIndicesCountFromTheInnermost

On `Shape{2, 3, 4}`, `dim(-1)` is 4, `dim(-2)` is 3 and `dim(-3)` is 2.

### OutOfRangeDimensionThrows

On `Shape{2, 3}`, `dim(2)` and `dim(-3)` throw `Error`.

### NonPositiveExtentIsRejectedAtConstruction

`Shape({2, 0})` and `Shape({-1})` throw `Error`.

### RankBeyondTheMaximumIsRejected

A list of nine extents, more than `K_MAX_RANK`, throws `Error`.

### BuildsFromExtentsKnownOnlyAtRunTime

A `Shape` built from a span holding 64 and 784 equals `Shape{64, 784}` and prints as
"[64, 784]".

### BuildsTheScalarFromAnEmptyExtentSpan

A `Shape` built from an empty span has rank 0 and one element.

### RuntimeExtentsGetTheSameValidationAsAListDoes

A span holding 4 and 0 throws `Error`, and so does a span of `K_MAX_RANK` + 1 ones.

### EqualityComparesRankAndEveryExtent

[4, 5] equals [4, 5] and differs from [5, 4], from [4, 5, 1], and [4] differs from the
scalar.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

### WorksAsAnUnorderedMapKey

An `unordered_map` keyed by `Shape`, written twice under [2, 2] and once under [4], holds two
entries with the values 2 and 7.

## test_status.cpp

Suites: `Unit_Status`.

Tests Status, its macros, the fatal family and the assertions of test_status.hpp.

This file needs no device. `Status` and the `SA_FAIL_IF`, `SA_TRY` and `SA_ANNOTATE` macros are
in `modules/core/core/include/SushiAI/core/status.hpp`, and the fatal checks are in
`modules/core/core/include/SushiAI/core/check.hpp`. The assertions `SA_EXPECT_OK`, `SA_EXPECT_FAILURE` and
`SA_EXPECT_FATAL` come from `modules/core/failure_expectations/include/SushiAI/failure_expectations/status_expectations.hpp`. `SA_EXPECT_FATAL` is a death test:
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

### FatalIfPrintsTheMessageThenTheFileAndTheLineAndStops

`stop_when(true, ...)` stops the process with a report containing "sushiai: fatal: extent 7 is
refused" followed by "test_status.cpp:", the line `stop_when` returned and ")". The case calls
`stop_when(false, ...)` first to learn that line.

### FatalCheckWhoseComposerThrowsReportsTheFixedMessage

`stop_with_throwing_composer` stops the process with a report containing "sushiai: fatal:",
"its message could not be composed" and "test_status.cpp:".

### ReplacedFatalHandlerRunsBeforeTheStop

`stop_under_announcer` stops the process and its stderr contains "recorded: stop 3", the line
`announce_fatal` writes.

### AssertChecksInDebugBuildsOnly

With `NDEBUG` defined, `SA_ASSERT(counted(evaluations, 0) == 1)` does not evaluate its
condition: the counter stays at zero. Without it, `assert_positive(1)` returns and
`assert_positive(0)` stops the process with "assertion failed: value > 0".

## test_seam.cpp

Suites: `Unit_Seam`, `Unit_SeamEngine`.

Tests the seam in `modules/core/core/include/SushiAI/core/seam.hpp`, the one place where an exception of SushiBLAS
or of a standard type, or an `Error` SushiRuntime returned, becomes a `Status`. `guard` and
`guard_value` run a call and return its Status or its value. `status_of_current_exception`
chooses the code from the type of the exception and from the `SeamPhase` the call was in:
`Record`, `Allocate` or `Execute`. `status_of_error` chooses it from the `ErrorCode`.

The `Unit_Seam` cases need no device and build each exception and each `Error` they map by hand. The
`Unit_SeamEngine` fixture is an `EngineTest`: its cases record work on an engine through
`record_and_run`.

### GuardReturnsTheStatusOfACallThatDoesNotThrow

A success passes through `guard`, and a failure Status returns with its code and its message.

### GuardValueReturnsTheValueOfACallThatDoesNotThrow

`guard_value` returns the value of a call that does not throw, as an ok `Result`.

### GuardValueReturnsTheStatusOfWhatTheCallThrew

A `std::bad_alloc` thrown by the call becomes a failed `Result` of code `OutOfMemory`.

### UnsupportedExecutionIsUnsupportedThoughItIsAnInvalidArgument

`SushiBLAS::UnsupportedExecution` derives from `std::invalid_argument`, and it maps to
`Unsupported`, not `InvalidDescription`. The message is kept.

### InvalidArgumentIsInvalidDescription

A plain `std::invalid_argument` maps to `InvalidDescription`, and its text is kept.

### PlainRuntimeErrorTakesItsCodeFromThePhase

A plain `std::runtime_error` maps to `OutOfMemory` when the call was allocating, and to
`DeviceFailed` when it was recording or executing.

### RuntimeFailureThatTheUserCanActOnMapsByItsCode

A SushiRuntime `Error` that carries a code the user can act on keeps it: `OutOfMemory` for a
missing USM allocation, `DeviceFailed` for a device fault or an unplugged device, and `Unsupported`
for a missing fp16 path.

### RuntimeInvalidGraphIsInvalidDescription

`InvalidGraph` maps to `InvalidDescription` and keeps its text: the runtime refuses a graph for
what the model asked of it, such as buffers that span two device contexts.

### RuntimeInvalidAccessIsUnsupported

`InvalidAccess` maps to `Unsupported` and keeps its text: the machine lacks the device a
placement names, or the runtime refuses an external handle.

### RuntimeInvalidStateIsUnsupported

`InvalidState` maps to `Unsupported` and keeps its text: one runtime runs one graph at a time,
and a second trainer on the same runtime asked for another.

### RuntimeProtocolMismatchIsIncompatible

`ProtocolMismatch` maps to `Incompatible` and keeps its text: two builds of the runtime speak
different versions of the wire protocol.

### RuntimeCompileFailedTakesItsCodeFromThePhase

`CompileFailed` names no cause, so it takes its code from the phase, as a plain
`std::runtime_error` does: `OutOfMemory` while allocating and `DeviceFailed` while executing.

### RuntimeDanglingResourceIsFatal

`DanglingResource` stops the process and returns no Status, because a tensor freed before the
step that names it breaks the contract of the pool. The case checks the report with
`SA_EXPECT_FATAL`.

### ErrorOfAHostUnitIsFatalWithItsText

A `SushiAI::Error` thrown while recording stops the process, and the report carries its text.

### BadAllocIsOutOfMemoryInEveryPhase

`std::bad_alloc` maps to `OutOfMemory` in each of the three phases.

### AnyOtherStandardExceptionIsDeviceFailed

Any other standard exception maps to `DeviceFailed` and keeps its text.

### ExceptionOfAnUnknownTypeIsDeviceFailed

An exception that derives from no standard type maps to `DeviceFailed`.

### RecordAndRunRunsWhatASuccessfulRecordingRecorded

A recording that sets a tensor of ones to 2.0 and returns success leaves every element at
2.0 after `record_and_run`.

### RecordAndRunDiscardsWhatAFailedRecordingRecorded

A recording that sets a tensor to 5.0 and then throws `std::invalid_argument` makes
`record_and_run` fail with `InvalidDescription` and the exception's message. A following
recording sets a second tensor to 2.0; that tensor holds 2.0 and the first still holds 1.0.

### RecordAndRunDiscardsWhenTheRecordingReturnsAFailure

A recording that sets a tensor to 5.0 and returns an `Unsupported` status makes
`record_and_run` fail with that status and "no host path". After `run_recorded` the tensor
still holds 1.0.

## test_shape.cpp, tests/core/graph/test_builder.cpp, tests/core/graph/test_ir.cpp, tests/layers/nn/test_layers.cpp, tests/core/autograd/test_backward.cpp

Suites: `Unit_Diagnostics`.

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`modules/core/failure_expectations/include/SushiAI/failure_expectations/throw_expectations.hpp` defines SA_EXPECT_ERROR.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

K_MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### ShapeDimensionOutOfRangeNamesTheIndexAndTheRank

On a [4, 3] shape, `dim(2)` raises an error holding "Dimension 2" and "rank-2 shape", and
`dim(-3)` one holding "Dimension -3" and "rank-2 shape".

### MatmulContractionMismatchNamesBothInnerExtents

`matmul` of [4, 5] and [7, 3] raises an error holding "MATMUL contracts 5 against 7",
"[4, 5]" and "[7, 3]".

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### MatmulNamesBothDtypesWhenTheyDisagree

`matmul` of a float32 and a float64 operand raises an error holding
"MATMUL needs operands of one element type", "float32" and "float64".

### BiasAddNamesTheAxisThatDoesNotLineUp

`bias_add` raises an error for a bias of 5 entries over [4, 3]
("BIAS_ADD bias has 5 entries", "[4, 3]", "3 columns"), for a [1, 3] bias
("BIAS_ADD needs a rank-1 bias", "[1, 3]") and for a rank-1 left operand
("BIAS_ADD needs a rank-2 left operand", "[3]").

### SumRowsAndLogSoftmaxRejectARank1Operand

Over a [6] operand, `sum_rows` raises an error holding "SUM_ROWS needs a rank-2 operand" and
"[6]", and `log_softmax` one holding "LOG_SOFTMAX needs a rank-2 operand" and "[6]".

### ElementwiseOperationsNameBothShapesAndBothDtypes

Over [4, 3] and [4, 2], `add` raises an error holding "ADD needs equally shaped operands" and
both shapes, and `sub`, `mul` and `accumulate` raise the same sentence under their own
operation names. `mul` of a float32 and a float64 operand raises
"MUL needs operands of one element type" with "float32" and "float64".

### NLLLossNamesTheShapeItCannotServe

`nll_loss` raises an error for [4, 3] against [4, 5]
("NLL_LOSS needs equally shaped operands", "[4, 3]", "[4, 5]") and for rank-1 operands
("NLL_LOSS needs rank-2 operands", "[4]").

### ActivationBackwardRefusesAnOperationThatIsNotOne

`activation_backward` with the identity of RELU raises an error holding "sushiai.relu" and
"is not an activation gradient".

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### AnUnknownOperandNamesTheOperationThatAskedForIt

`add_node` of a RELU that reads the value 9999 raises an error holding "sushiai.relu",
"reads value 9999" and "does not exist". One that writes the value 9999 raises
"writes value 9999" and "does not exist".

### LinearNamesItsOwnDimensionsWhenTheInputDoesNotFit

`Linear{6, 4}` over [8, 5] raises an error holding "Linear(6, 4)", "[8, 5]" and
"inner extent must be 6". `Linear{5, 4}` over a [5] input raises
"Linear needs a rank-2 input" with "[5]", and `Linear{0, 4}` raises
"Linear needs positive feature counts" with "in=0" and "out=4".

### ANonScalarRootNamesTheValueAndItsShape

Differentiating from a [4, 3] ReLU output raises an error holding
"needs a scalar to differentiate" and "[4, 3]".

### AOneElementRootOfADifferentiableOperationIsSeeded

The root is a [1, 1] K_MATMUL of a parameter `x` and an input `w`. Differentiation does not
throw and appends two nodes. The first is a K_CONSTANT in the root's shape, the seed. The second
is a K_MATMUL that reads the seed and `w`, and its result is the gradient of `x`, in the shape of
`x`. `test_loss_roots.cpp` covers the seed in full.

### ARootThatRequiresNoGradientNamesItsProducerAndTheCause

Two roots are refused. A K_MATMUL of two inputs is reached by no parameter, and the message
contains "value 3", "sushiai.matmul" and "requires no gradient". A K_MATMUL that reads the
parameter through `stop_gradient` is refused with a message containing "sushiai.matmul",
"requires no gradient" and "K_STOP_GRADIENT", so the message names the barrier as a cause the
caller can look for.

### ARootWhoseProducerHasNoRuleNamesTheProducer

The root is a K_DYNAMIC_SCALE of a parameter, an operation with no backward rule. The message
contains "sushiai.dynamic_scale", "has no backward rule" and "autograd/rules/", the folder a
rule is added in.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose K_SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.
