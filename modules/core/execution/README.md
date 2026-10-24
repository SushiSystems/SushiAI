# execution

## Owns

The calls that run work and return a `Status`: `create_runtime` creates the runtime an engine is
built on, `create_graph` creates an empty graph on the engine's runtime, `run_step` runs one step
of a graph, `run_once` runs every task recorded on a graph, and `record_and_run` creates a graph
of its own, calls the caller's recording with a `SushiBLAS::TaskRecorder` over it, and runs it. A
failed recording leaves nothing behind, because the graph dies with its tasks. The module never
touches the graph of an engine. Each call maps what SushiRuntime reports through `seam`.

## Depends on

`failure` and `seam`, public. It links `SushiBLAS::sushiblas` and `SushiRuntime::api` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/execution/execution.hpp` | `Detail::create_runtime`, `Detail::create_graph`, `Detail::run_step`, `Detail::run_once`, `Detail::record_and_run`, the alias `sr` |

## Designs

<!-- designs:begin -->
- [LAYERED_EXECUTION](../../../docs/design/LAYERED_EXECUTION/README.md): SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and it trains one model over the owner's five machines through SushiRuntime's distributed...
<!-- designs:end -->
