# tests/core/optim/test_step_scalars.cpp


Suites: `Small_StepScalars`, `Medium_StepScalarsFatal`.

Tests that `StepScalars` holds a zeroed buffer at one address and rewrites it in place.

The buffer is allocated from the engine of a `HostExecutionBackend`, so the file is small. The
buffer is host-readable shared memory, so every case reads it through `device_pointer()` without a
copy. `compile_step_scalars.cpp` checks that the buffer cannot be copied.

## AZeroCountIsRefused

`make()` with a count of 0 fails with `InvalidDescription` and "at least one slot".

## ANewBufferHoldsZeros

A buffer of three slots reports `count()` 3, a non-null address and three zeros.

## AWriteReplacesEveryValue

`write()` of three values leaves the buffer holding exactly those values.

## AWriteKeepsTheAddress

`write()` leaves `device_pointer()` where it was.

## AMoveKeepsTheAddress

A buffer moved into a new object keeps the address taken before the move and its count.

## AWriteOfAnotherLengthIsFatal

Size (RW-166): the death check runs in a child process and took more than the 0.1 s of a
`small` test on the debug build, so the case is in `Medium_StepScalarsFatal`; a death test is `medium` by its
size rule.

`write()` of two values into a buffer of three stops the process and names "holds 3 slots" and
"2 values".
