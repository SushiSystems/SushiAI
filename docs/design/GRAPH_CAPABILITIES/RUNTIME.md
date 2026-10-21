# SushiRuntime items

Part of the [graph capabilities plan](README.md).

## R1. State and test that a replayed plan reads memory the host wrote between executions

**Limit and evidence.** SushiAI writes Shared USM from the host with no task and expects the
kernels of the next execution to read it:

- the row offsets of a row selection, once at lowering
  (`modules/core/graph/source/lowering/linalg.cpp:97-99`);
- the two `[T * T]` vectors of the causal mask, once at lowering
  (`modules/core/graph/source/lowering/linalg.cpp:148-158`);
- the optimizer's scalars, before every replay (`modules/core/optim/source/optimizer.cpp:174-177`,
  `modules/core/graph/source/compiled_step.cpp:116-120`).

The memory is Shared: `TensorFactory::create` defaults to `Residency::Shared`
(`sushiblas/modules/data/tensor/include/SushiBLAS/tensor/tensor_factory.hpp:103-107`) and
`TensorPool::create` passes no residency (`modules/core/graph/source/tensor_pool.cpp:63-66`).
`sushiruntime/docs/design/GRAPH_REPLAY.md` section D designs the same pattern for late-bound
counts: a kernel reads a host-USM word the step machinery writes before each submission. No
header states it as a contract of `SushiRuntime::Graph` and its plan, and no test replays a plan
around a host write.

**Design.** No code changes. One contract sentence and one test.

The sentence, on `CompiledGraph::rearm` in
`sushiruntime/modules/engine/graph/include/SushiRuntime/graph/compiled_graph.hpp` and in
`sushiruntime/modules/engine/graph/README.md`:

> Between two executions of one compiled plan the caller may write a Shared or Host USM
> allocation that no node of the plan declares as written. Every node of the next execution
> observes the bytes written. The caller writes only while no execution of the plan is in
> flight.

**Semantics.** The rule holds on the scheduler path and on layers 1 and 2 of `GRAPH_REPLAY.md`.
Layer 2 is not built; this item binds it: a recorded command graph fixes a kernel's arguments,
which are addresses, and not the bytes behind them. A Device allocation is outside the rule,
since the host cannot address it. The rule adds no ordering: the host write is not a node, so
the caller orders it by writing between executions.

**Adds.** `sushiruntime/modules/engine/execution/tests/test_replay_host_write.cpp`.

**Must not change.** Every signature of `SushiRuntime::Graph`, its plan and the execution engine.

**Tests.** One kernel node reads a Shared word `w` and writes `out = w + 1`. The plan is
compiled once. For `k` in 0 to 7 the host writes `w = k`, re-arms and executes; `out` is `k + 1`
each time. The case runs on every device queue the context has.

**SushiAI then.** Nothing changes; `modules/core/graph/README.md` cites the contract where it
describes the host-filled scratch.

## R2. Several machines

The design is [`LAYERED_EXECUTION.md`](../LAYERED_EXECUTION/README.md); its "One step on five machines" holds what the
runtime has to provide.
