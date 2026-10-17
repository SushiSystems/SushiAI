# tests/core/log

The tests of the module `core/log`, one file per header of `modules/core/log/include/`. The suite
prefix of a case gives its ctest label.

## test_log.cpp

Each case sets a sink that keeps every record's `project.module` and message, runs synchronously,
and restores the sink and the `sushiai` level before it checks.

### AWarnRecordCarriesItsProjectAndModule

With the `sushiai` level at `Warn`, `SA_LOG_WARN(K_LOG_TEST_CATEGORY, "extent {} is refused", 7)`
reaches the sink once, as `sushiai.log_test` with the message "extent 7 is refused".

### TurningSushiAIOffLeavesSushiRuntimeAtItsLevel

With the `sushiai` level at `Off`, an `SA_LOG_ERROR` reaches no sink, and the level of the
`sushiruntime` project reads the same before and after.

## test_sushiai_log_project.cpp

### NamesSushiAIAndItsLevelVariable

At compile time, `K_SUSHIAI_LOG_PROJECT` names the project `sushiai` and the variable
`SA_LOG_LEVEL`.
