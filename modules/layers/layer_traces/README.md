# layer_traces

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` is on.

## Owns

`TraceSummary`, what one training trace of a model graph leaves: the operation of every
node, every published output with its extents, and every piece of state. `trace_summary` traces
over a batch of `K_TRACE_BATCH` samples.

## Depends on

`failure`, `shape`, `graph` and `nn`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/layer_traces/trace_summary.hpp` | `OpList`, `K_TRACE_BATCH`, `TraceSummary`, `state_record`, `state_records`, `trace_summary` |
