# tests/layers/nn/test_throwing_bridge.cpp


Suites: `Small_ThrowingBridge`, `Small_ThrowingBridgeDeathTest`.

Tests bridge_throwing: a value passes through and a thrown Error becomes a Status.

This file needs no device. `NN::Detail::bridge_throwing` in
`modules/layers/nn/include/SushiAI/nn/throwing_bridge.hpp` runs a callable that may throw and returns its value
as a `Result`.

## refuse_a_shape

`refuse_a_shape` throws an `Error` through `SA_THROW_IF` with the message "Linear(6, 8) was
given [5]".

## refuse_an_allocation

`refuse_an_allocation` throws `std::bad_alloc`.

## throw_a_logic_error

`throw_a_logic_error` throws `std::logic_error("an index was out of range")`, which is neither
an `Error` nor `std::bad_alloc`.

## bridge_a_logic_error

`bridge_a_logic_error` passes `throw_a_logic_error` to `bridge_throwing`, as the statement of
a death test.

## AValuePassesThrough

A callable that returns 7 gives an ok `Result<int>` holding 7.

## AMoveOnlyValuePassesThrough

A callable that returns `std::make_unique<int>(7)` gives an ok result whose pointer is not
null and owns 7.

## AnErrorBecomesInvalidDescriptionWithItsMessage

Bridging `refuse_a_shape` gives an `InvalidDescription` failure whose message contains
"Linear(6, 8)" and "[5]".

## ARefusedAllocationBecomesOutOfMemory

Bridging `refuse_an_allocation` gives an `OutOfMemory` failure whose message contains
"allocation was refused".

## AnotherExceptionIsFatal

The case is in the suite `Small_ThrowingBridgeDeathTest`. `bridge_a_logic_error` stops the
process with a report containing "other than Error" and "an index was out of range".
