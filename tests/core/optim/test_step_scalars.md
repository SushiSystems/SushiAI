# tests/core/optim/test_step_scalars.cpp


Suites: `Medium_StepScalars`.

Tests that `StepScalars` holds a zeroed buffer at one address and rewrites it in place.

The buffer is allocated through an engine, so the file is medium. The buffer is host-readable
shared memory, so every case reads it through `device_pointer()` without a copy.

## span_of

`span_of` views a vector of floats as the span `write()` takes.

## AZeroCountIsRefused

`make()` with a count of 0 fails with `InvalidDescription` and "at least one slot".

## ANewBufferHoldsZeros

A buffer of three slots reports `count()` 3, a non-null address and three zeros.

## AWriteKeepsTheAddress

`write()` of three values leaves `device_pointer()` where it was and the buffer holds exactly
those values.

## AMoveKeepsTheAddress

`StepScalars` cannot be copied. A buffer moved into a new object keeps the address taken before
the move and its count, which is what lets a kernel that captured the address replay.

## AWriteOfAnotherLengthIsFatal

Writing two values into a buffer of three stops the process with a report naming "holds 3
slots" and "2 values".
