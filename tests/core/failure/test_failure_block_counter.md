# tests/core/failure/test_failure_block_counter.cpp


Suites: `Small_FailureBlockCounter`.

Tests `set_failure_block_counter` of `failure_block_counter.hpp`, with the in-memory counter
as the counter to install and `live_failure_count()` as the process counter's reading.

## InstallingACounterReturnsTheOneItReplaced

With two in-memory counters made one after the other, installing the first returns the second.

## ANullCounterInstallsTheProcessCounter

After `set_failure_block_counter(nullptr)`, a failure made raises `live_failure_count()`
by one and leaves the in-memory counter at zero.
