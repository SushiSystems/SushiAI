# execution

## Owns

The calls that run work and return a `Status`: `create_runtime` creates the runtime an engine is
built on, `run_recorded` runs what an engine recorded, `run_step` runs one step of a graph, and
`record_and_run` records and runs in one call, discarding what a failed recording left through
`abandon_recording` and `discard_recorded`. Each maps what SushiRuntime or SushiBLAS reports
through `seam`.

## Depends on

`failure` and `seam`, public. It links `SushiBLAS::sushiblas` and `SushiRuntime::api` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/execution/execution.hpp` | `Detail::create_runtime`, `Detail::run_recorded`, `Detail::run_step`, `Detail::discard_recorded`, `Detail::abandon_recording`, `Detail::record_and_run`, the alias `sr` |
