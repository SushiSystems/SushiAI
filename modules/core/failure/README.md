# failure

## Owns

The failure contract. `Status` carries a `StatusCode` and a message, and `Result<T>` holds either
a value or the `Status` that explains its absence. The fatal checks stop the process for a broken
precondition.

The default fatal handler writes the message through `SushiLog::log_at` at `FATAL` in
`K_FAILURE_LOG_CATEGORY`, so the record reads `[sushiai.failure]` and carries the file and line
of the failed check as its call site. The record passes every level gate, so `SA_LOG_LEVEL=off`
does not silence it, and the call formats nothing. SushiLog delivers it after every queued record
and flushes before `fail_fast` aborts.

`Status` counts the heap blocks that hold its failures through a `FailureBlockCounter`. The
process counter is the default, and `Detail::live_failure_count()` reads it. A test installs an
`InMemoryFailureBlockCounter` with `set_failure_block_counter` to read the blocks of its own
statuses. `Detail::report_then_stop` delivers a report and runs a stop function the caller
gives; `fail_fast` passes it the one that flushes and aborts.

## Depends on

It links `SushiLog::text` as public, for `SushiLog::format_text`, which builds every message,
and depends on `log` as private, for the project the fatal record's category names. It links
`SushiLog::SushiLog` through `log`, for `SushiLog::log_at`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/failure/check.hpp` | `FatalHandler`, `set_fatal_handler`, `Detail::report_then_stop`, `SA_FATAL_IF`, `SA_FATAL`, `SA_ASSERT` |
| `SushiAI/failure/failure_block_counter.hpp` | `FailureBlockCounter`, `set_failure_block_counter` |
| `SushiAI/failure/in_memory_failure_block_counter.hpp` | `InMemoryFailureBlockCounter` |
| `SushiAI/failure/in_memory_fatal_sink.hpp` | `InMemoryFatalSink` |
| `SushiAI/failure/result.hpp` | `Result<T>`, `Result<T&>`, `Detail::status_of`, `SA_TRY_ASSIGN` |
| `SushiAI/failure/status.hpp` | `StatusCode`, `Status`, `SA_FAIL_IF`, `SA_ANNOTATE`, `SA_TRY` |
