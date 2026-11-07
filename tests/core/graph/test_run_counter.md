# tests/core/graph/test_run_counter.cpp


Suites: `Small_RunCounter`.

Tests `Graph::RunCounter` in `modules/core/graph/include/SushiAI/graph/run_counter.hpp`, the count
of runs a `CompiledStep` holds. The file needs no device. That a step counts its runs through the
counter is tested in `test_compiled_step.cpp`.

## ANewCounterHasCountedNoRun

A new counter holds 0.

## EachAdvanceCountsOneRun

Two calls of `advance` leave the count at 2.

## ACountAboveZeroIsRestoredOnce

A counter restored to 5 refuses a second restore to 7 with `InvalidDescription` and "run 5", and
its count stays 5.

## RestoringZeroLeavesTheCounterNew

`restore(0)` and then `restore(5)` both succeed on a new counter, and its count is 5.

## ACounterThatHasRunRefusesARestore

A counter that advanced once refuses `restore(4)` with `InvalidDescription`, "before its first
run" and "run 1", and its count stays 1.
