# Memory planning and lowering

How the compiled step plans memory, lowers the IR to SushiBLAS through a table, and carries the optimizer. Part of the architecture chapter; [`ARCHITECTURE.md`](ARCHITECTURE.md) holds §1 to §4.

## 5. Memory planning, and lowering

`graph/memory_plan.hpp` decides where every non-persistent value's bytes go,
before anything is lowered. It is a pure function of the IR: no engine, no
allocator, no SYCL, which is why the whole pass is unit-tested over
synthetic graphs in the device-free tier.

Two decisions in it are load-bearing:

- **One arena per lifetime class, and only the non-persistent classes.**
  SushiRuntime's dependency tracker shards its access history on an
  allocation's base pointer, so a single global arena would collapse every
  shard into one chain and make hazard resolution quadratic. Parameters,
  gradients, optimizer state and inputs live for the whole run, so packing
  them would save nothing while merging exactly the shards that matter most;
  they keep their own `Storage`.
- **Liveness is the reachability partial order, not an interval overlap.**
  Value `b1` may hand its bytes to `b2` only when *every* use of `b1` is a
  strict ancestor of `b2`'s first write. A planner that instead sorts nodes
  into a linear schedule and overlaps disjoint `[first_use, last_use]`
  intervals passes every obvious test and quietly serialises branches the
  runtime was free to run in parallel: memory reuse *injects* WAR and WAW
  anti-dependencies, and only the reachability form guarantees the injected
  edge was already implied. The transitive closure costs O(V²/64) memory, so
  above `PlanOptions::reuse_node_limit` nodes the planner declines to overlap
  anything and says so through `reuse_note()` rather than substituting the
  linear approximation.

`graph/arena.hpp`'s `ArenaSet` is the device-lane other half:
`ArenaSet::make` allocates one `Storage` per arena, or returns the
`OutOfMemory` failure of the first arena the allocator refuses, and the set
never allocates again. It hands out views through `view_tensor()`
(`allocation/allocation.hpp`), which stops the process on a byte offset that is not
a whole number of the view's elements. `Tensor::storage_offset` counts
elements, so such an offset has no representation.

This is only safe because SushiBLAS now honours `storage_offset` everywhere
and `Tensor::region()` reports a view's own byte interval rather than the
bare allocation base; two disjoint tensors in one arena therefore produce
non-overlapping `SushiRuntime::BufferRegion`s and do not falsely serialise. That claim is
checked directly in `tests/core/graph/test_arena.cpp`.

The two reduction operands of `K_LOG_SOFTMAX` and `K_NLL_LOSS` are planned like
any other value. SushiBLAS builds the regions of `execute_reduction` and
`execute_folding_axis_reduction` from `Tensor::region()`, so a reduction
operand at a non-zero offset gets its dependency edge; SushiBLAS pins that in
`Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation` and
`Unit_AxisReductions.AxisReductionsOrderAgainstViewsOfOneAllocation`.
`PlanOptions::pinned` remains for a caller that wants a value kept out of the
arenas.

`graph/tensor_pool.hpp`'s `TensorPool` holds the tensor of each value. It
adopts the plan and asks it where each value goes: packed values become arena
views, everything else gets its own `Storage`. Every call that allocates
(`create`, `materialise`, `alias`, `scratch`, `ones`, `request_constant`)
returns a `Status` or a `Result`. The planner assumes **every node writes all
of its outputs on every replay**, which is why `K_ZERO` records its fill into
the step rather than into the one-time constant queue.

### 5.0 The lowering is a table, one row per operation

`Lowering::record` (`graph/lowering.hpp`) walks the graph in node order and
records each node through its row of `K_LOWERINGS`. A row is a
`LoweringEntry`, an `OpID` paired with a `LoweringRule`:

```cpp
using LoweringRule = Status (*)(const LoweringContext&);
```
Source: `modules/core/graph/include/SushiAI/graph/lowering/rule.hpp`

The rows live in eight domain tables, each a header under
`modules/core/graph/include/SushiAI/graph/lowering/` with its functions in the `.cpp` of the
same name under `modules/core/graph/source/lowering/`: `elementwise`, `activation`, `linalg`,
`loss`, `shape`, `spatial`, `norm` and `fused`. `all.hpp` joins them with
`concat_lowerings` into the 62 rows of `K_LOWERINGS`, and `find_lowering`
searches that array. No rule calls the next one.

A rule sees one `LoweringContext` (`graph/lowering/context.hpp`), built for
its node and gone when it returns. The context gives the rule the recorder,
the RNG stream, the node, its operand and result tensors, scratch from the
pool, and the fusion selector. A rule builds a SushiBLAS view from
`context.recorder()` and records on it:

```cpp
Status lower_zero(const LoweringContext& context)
{
    context.require_arity(0, 1);
    SA_TRY_ASSIGN(SushiBLAS::Tensor& zeros, context.output(0));

    SushiBLAS::RandomOps random(context.recorder(), context.stream());
    random.constant(zeros, 0.0);
    context.count(1);
    return Status();
}
```
Source: not compiled

The driver runs each rule inside `Detail::guard(SeamPhase::Record, ...)`, so
an exception SushiBLAS throws while recording comes back as a status, and it
annotates a refusal with the operation's name and the node id.

Two `static_assert`s in `all.hpp` close the table against `Ops::K_ALL`: every
identity has exactly one row or is listed in `K_UNLOWERED`, never both, and
the two lists together are as long as `K_ALL`. Adding an operation is a
function in its domain's `.cpp` and a row in that domain's header. Adding a
domain is one include and one `concat_lowerings` argument in `all.hpp`.

### 5.0.1 The compiled step

`graph/compiled_step.hpp` wraps the compile-once/replay contract in a type,
because the ordering has a silent failure mode: `rearm()` before
`capture_base_state()` leaves the completion sentinel armed at in-degree
zero, so the run reports complete before any kernel has been dispatched and
the step quietly does nothing. `CompiledStep::compile` is the only way to
obtain a step, and it captures the base state before it returns; `run()`
re-arms and executes. That makes the state unrepresentable.

A `CompiledStep` owns its `SushiRuntime::Graph`, the
`SushiBLAS::TaskRecorder` bound to that graph's recorder, and the
`SushiRuntime::RunReport` each run refills. The graph holds the plan.
The engine and the pool are borrowed and must outlive it. When the recording
fails, `compile` discards the recorded nodes and returns the refusal.

It also takes an optional `StepRecorder` (`graph/step_recorder.hpp`): work
that belongs to the step but has no IR node behind it, recorded after the
lowering and before the one compilation, and driven around every replay. The
optimizer's parameter update is what it exists for; see §5.1.

### 5.1 The optimizer is part of the compiled step

Spelled as separate SushiBLAS calls, an AdamW update is about eleven launches
per parameter per step: for an eight-tensor model, ~88 launches against the
~31 the whole forward and backward graph costs. Both optimizers now issue
**one** `Dispatch::execute_zip_inplace` per parameter instead: that seam hands
a functor lvalue references to every operand's element, so the entire recipe
runs in one kernel with `g²`, the denominator and `m/denom` living in
registers. The scratch buffers those temporaries needed are not allocated at
all on the fused path.

Fusing is also what let the update *join the plan*. The recorded obstacle was
real: a per-step scalar baked into a recorded node is a compile-time constant,
and AdamW's bias corrections change every step, but a single functor can
capture a **device pointer** instead, and read the step's numbers from the
small `Optim::StepScalars` buffer behind it. A pointer is capturable by value
and its contents may change between replays, so the recorded work is
launch-invariant and goes into the same `SushiRuntime::Graph` as forward and backward.
Ordering needs no new machinery: every call declares the bytes it reads and
writes, so the tracker already sees that the update reads the gradients the
backward pass wrote and writes the parameters the forward pass read.

Two guarantees hold this down, both in the regression tier
(`tests/layers/train/test_trainer/test_fused_optimizer.cpp`): the fused kernel and
the separate operations agree **byte for byte** (they can, because the whole
stack builds with `-ffp-contract=off` and every expression keeps the operand
order of the kernel it replaced), and folding the update into the plan changes
no number either. `UpdateKernel::Separate` keeps the reference spelling
reachable. `Trainer::make` returns `InvalidDescription` when
`OptimizerPlacement::InPlan` is asked of an optimizer whose
`is_replay_invariant()` is false, never a quiet downgrade. Calling `step()` on
an optimizer a plan already owns stops the process: the update is invisible
inside a replay, so a stray call would train at double the learning rate and
report nothing.

What it is worth, measured rather than assumed: on `sa demo mlp`'s four
parameters, a step goes from 81 device tasks in three dispatch rounds to 37 in
two, and a whole run from 278 `SushiRuntime::Graph::compile()` calls to 150. The wall
clock moves much less: a median 8.80 to 7.74 ms/step over ten runs each on a
quiet CPU/OpenCL box, because this model's largest tensor is 4096 elements
and most of a step is neither the optimizer nor arithmetic. The launch counts
are the honest number; the microbenchmark's 16-22x is what the same change is
worth when the optimizer *is* the step.
