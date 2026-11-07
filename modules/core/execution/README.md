# execution

## Owns

Two ports and their adapters, and the calls that run work and return a `Status`.

`StepTasks` is one recording of tasks that replays: its recorder, its task count, the refusal kept
while recording, the hooks that run before each replay, a label, `seal()` and `replay()`.
`ExecutionBackend` gives a caller the engine, a new `StepTasks` from `make_tasks()` and
`supports_fp64()`. `HostExecutionBackend` owns a host `SushiBLAS::Engine` and keeps the tasks of
each `StepTasks` in a `SushiBLAS::HostTaskSink`; a task that fails on replay is a `DeviceFailed`
naming its index and name. `RuntimeExecutionBackend` borrows a `SushiRuntime::Runtime` and an
engine built on it, records each `StepTasks` on a `SushiRuntime::Graph` with a
`SushiBLAS::LateCountHold` declared before it, and gives out its runtime through `RuntimeAccess`, the
port the regions, the cluster packs and the placement scope ask for; a host backend has none.
`RuntimeBackend` is the port that is both an `ExecutionBackend` and a `RuntimeAccess`; a class that needs the
engine and the runtime takes one `RuntimeBackend&`, so the two never name different objects, and
`RuntimeExecutionBackend` implements it. The two `StepTasks`
adapters are private to the sources of the backends. The caller that makes the engine picks the
backend; no class asks an engine its kind.

`create_runtime` creates the runtime an engine is built on. `record_and_run(backend, record)` makes
tasks on the backend, calls the caller's recording with their `SushiBLAS::TaskRecorder` under the
Record guard, returns the refusal the recording kept, seals the tasks and replays them once. A
recording of no task runs nothing and succeeds. A failed recording leaves nothing behind, because
the tasks die with the call. Each call maps what SushiRuntime reports through `seam`.

## Depends on

`failure` and `seam`, public. It links `SushiBLAS::sushiblas` and `SushiRuntime::api` as public.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/execution/step_tasks.hpp` | `StepTasks` |
| `SushiAI/execution/execution_backend.hpp` | `ExecutionBackend` |
| `SushiAI/execution/runtime_access.hpp` | `RuntimeAccess` |
| `SushiAI/execution/runtime_backend.hpp` | `RuntimeBackend` |
| `SushiAI/execution/host_execution_backend.hpp` | `HostExecutionBackend` |
| `SushiAI/execution/runtime_execution_backend.hpp` | `RuntimeExecutionBackend` |
| `SushiAI/execution/execution.hpp` | `Detail::create_runtime`, `Detail::record_and_run`, the alias `sr` |

## Designs

<!-- designs:begin -->
- [LAYERED_EXECUTION](../../../docs/design/LAYERED_EXECUTION/README.md): SushiAI's share of the stack's layered execution: it stops using SushiBLAS's engine graph, and it trains one model over the owner's five machines through SushiRuntime's distributed...
<!-- designs:end -->
