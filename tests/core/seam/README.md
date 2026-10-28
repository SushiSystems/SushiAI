# tests/core/seam

The tests of the module `core/seam`, one file per header of `modules/core/seam/include/`. The suite
prefix of a case gives its ctest label.

## test_seam.cpp

Suites: `Small_Seam`.

Tests the seam in `modules/core/seam/include/SushiAI/seam/seam.hpp`, the one place where an
exception of SushiBLAS
or of a standard type, or an `Error` SushiRuntime returned, becomes a `Status`. `guard` and
`guard_value` run a call and return its Status or its value. `status_of_current_exception`
chooses the code from the type of the exception and from the `SeamPhase` the call was in:
`Record`, `Allocate` or `Execute`. `status_of_error` chooses it from the `ErrorCode`.

The `Small_Seam` cases need no device and build each exception and each `Error` they map by hand.

### GuardReturnsTheStatusOfACallThatDoesNotThrow

A success passes through `guard`, and a failure Status returns with its code and its message.

### GuardValueReturnsTheValueOfACallThatDoesNotThrow

`guard_value` returns the value of a call that does not throw, as an ok `Result`.

### GuardValueReturnsTheStatusOfWhatTheCallThrew

A `std::bad_alloc` thrown by the call becomes a failed `Result` of code `OutOfMemory`.

### GuardValueFlattensAResultTheCallReturns

A call that returns `Result<int>` gives `Result<int>`, not a nested `Result`: a held 7 comes back
as 7, and a returned `InvalidDescription` comes back with its message.

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

### BadAllocIsOutOfMemoryInEveryPhase

`std::bad_alloc` maps to `OutOfMemory` in each of the three phases.

### AnyOtherStandardExceptionIsDeviceFailed

Any other standard exception maps to `DeviceFailed` and keeps its text.

### ExceptionOfAnUnknownTypeIsDeviceFailed

An exception that derives from no standard type maps to `DeviceFailed`.
