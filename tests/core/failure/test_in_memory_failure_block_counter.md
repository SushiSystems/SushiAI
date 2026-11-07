# tests/core/failure/test_in_memory_failure_block_counter.cpp


Suites: `Small_InMemoryFailureBlockCounter`.

Tests that `InMemoryFailureBlockCounter` counts the failure blocks of its own scope.

## ReadsTheBlocksMadeUnderItAndNotYetReleased

Two failures made under the counter read as two live blocks, and as none once both are gone.

## DoesNotCountABlockMadeBeforeIt

A failure made before the counter leaves it at zero.

## TheProcessCounterCountsAgainOnceItIsGone

A failure made under the counter leaves `live_failure_count()` where it was. Once the
counter is gone, a failure raises it by one.
