# Eager front end

**Status:** Open — written, not built; RW-15 to RW-22.

SushiAI gains a second way to run a model. Today a model is traced once, compiled once and
replayed by `Train::Trainer`. The eager front end traces the model again on every step, so plain
C++ control flow decides what runs, and it recompiles only the part of the step whose structure
changed. Both front ends share one IR, one rule table, one set of passes and one lowering.

## 1. Decisions already taken

Each was made by the owner on 2026-10-07.

| Decision | Choice |
| --- | --- |
| Meaning of eager | Define by run, execute on read. Calls append IR; work runs when a value is read, on `backward()` or on `step()` |
| Substrate | SushiRuntime's `DynamicGraph`, one region per unit of structure, not a cache of whole compiled plans |
| Region boundary | A policy behind an interface. The default cuts only where the caller opens a scope; with no scope the forward pass is one region |
| SushiBLAS seam | `TaskRecorder` is a concrete type constructed over a `TaskGraph&` the caller supplies. `Engine::get_graph()` leaves the public surface |
| SushiRuntime seam | `Region::recorder()` hands out the `SushiRuntime::Recorder` of a region; a `SushiBLAS::TaskRecorder` is built over it |
| Python front end | Out of scope |

`Region::recorder()` is on SushiRuntime's `main`. The `TaskRecorder` is designed in sushiblas under `docs/agent/2026_10_07_V010/` and its signatures are
not fixed. Section 9 lists what this spec assumes about it.

## 2. What the caller writes

The names below are proposals; the public shape is the owner's to change.

```cpp
SushiAI::Eager::RegionExecutor executor(engine);
SushiAI::Eager::ScopePartitioner partitioner;
SushiAI::Eager::Session session(engine, executor, partitioner, options);

for (const Data::Batch& batch : batches)
{
    Eager::Step step = session.begin_step();
    Graph::ValueId x = step.input(batch.inputs);

    Graph::ValueId h = encoder.forward(step.builder(), x);
    if (use_wide_head)
    {
        Eager::Scope scope(step.scopes(), "head");
        h = wide_head.forward(step.builder(), h);
    }
    else
    {
        Eager::Scope scope(step.scopes(), "head");
        h = narrow_head.forward(step.builder(), h);
    }

    Graph::ValueId loss = objective.loss(step.builder(), h, step.input(batch.targets));
    step.backward(loss);
    step.apply(optimizer);

    const double value = step.read_scalar(loss);
}
```

A layer is traced through the `forward(const GraphBuilder&, ValueId)` it already has. No layer,
rule or builder method changes for the eager path.

When `use_wide_head` flips, the region named `head` and its backward region are rebuilt. The
encoder's regions keep their compiled nodes.

## 3. Units

Each unit has one job and reaches its neighbours through the type named in the last column.

| Unit | Job | Depends on |
| --- | --- | --- |
| `Eager::Session` | Owns what lives across steps: the dynamic graph, the region table, the storage of every live region | `StepExecutor`, `RegionPartitioner` |
| `Eager::Step` | Owns one step's trace: a `Graph`, its `GraphBuilder`, the open scopes, the flush points | `Session` |
| `Eager::Scope` | Marks the nodes traced during its lifetime as one named unit of structure | `Step` |
| `RegionPartitioner` | Assigns every node of a differentiated graph a region key | `Graph` only |
| `ScopePartitioner` | The default partitioner: one region per scope and direction | `RegionPartitioner` |
| `StructureKey` | Hashes a region's sub-graph so two steps can be compared | `Graph` only |
| `StepExecutor` | Runs a partitioned graph and makes the requested values readable | `Graph`, `TensorPool` |
| `RegionExecutor` | The `StepExecutor` over `DynamicGraph` | SushiRuntime, SushiBLAS |
| `RegionEmitter` | Builds the closure a region stores: constructs a `TaskRecorder` over the graph it is handed and lowers the region's nodes into it | `Lowering`, SushiBLAS |

`RegionPartitioner`, `StructureKey`, `Scope` and `Step`'s tracing half are host lane: no engine and
no SYCL, so they are unit tested without a device, as `fusion.hpp` and `memory_plan.hpp` are.

`Trainer` and `CompiledStep` stay as they are. A later change may express `CompiledStep` as a
`StepExecutor`; this spec does not require it.

## 4. One step, in order

1. **Trace.** The caller's code emits forward IR into the step's graph. Each node records the scope
   that was open when it was emitted.
2. **Differentiate.** `step.backward(loss)` runs `Autograd::differentiate` over the whole graph,
   once, as the compiled path does.
3. **Partition.** The partitioner gives every node a region key (section 5).
4. **Key.** `StructureKey` is computed per region (section 6) and compared with the key the session
   holds for that region.
5. **Rebuild what changed.** For a region whose key differs, or that is new: the cast pass, fusion,
   rematerialisation and the memory planner run over that region's sub-graph; the session drops the
   old region and records a new one through `Region::recorder()`. For a region whose key matches,
   nothing runs and the session does not call `DynamicGraph::region()`, which would mark it dirty.
6. **Run.** `DynamicGraph::run()` composes the changed regions with their boundaries and executes
   the step.
7. **Read.** A requested value is copied to the host after the run has finished.

Steps 3 to 5 are host work on every step, including a step whose structure did not change. Its
cost has not been measured. Section 10 makes that measurement an acceptance criterion.

## 5. Region keys

SushiRuntime orders two regions that share bytes by ascending key: the lower key is the producer.
The partitioner must therefore number regions in an order that respects every data dependency.

With `n` forward scopes in trace order:

| Region | Key |
| --- | --- |
| Forward scope `k`, `k = 1..n` | `k` |
| Loss | `n + 1` |
| Backward of scope `k` | `2n + 2 - k` |
| Optimizer update | `2n + 2` |

The backward region of scope `k` reads activations of forward scope `k` and gradients produced by
the backward region of scope `k + 1`. Both have lower keys. The optimizer reads every gradient and
writes every parameter, and has the highest key.

Nodes traced outside any scope belong to an implicit scope that runs from the previous scope's end
to the next scope's start, so trace order is preserved. With no explicit scope the whole forward
pass is one region and the step has four regions.

`n` changes when a scope appears or disappears, and with it every backward key. The session keeps
its table by scope name and direction, not by numeric key, and renumbers before recording. A
renumbered region whose structure did not change is still re-recorded, because its key is its
identity in the runtime. Section 10 lists this as a cost to measure.

## 6. Structure key

Two steps produce the same region when the key agrees. The key covers, for the region's nodes in
trace order: the op identity, the attributes, the shape, dtype and layout of every result, and the
position of every operand. An operand the region produces is positioned by local index. Every
other operand is external, a leaf or a value another region produces, and is positioned by the
identity the session supplies for its storage.

External operands need care, because a recorded kernel holds the address of its operands:

- A parameter keeps its storage across steps, so its identity is stable and is part of the key.
- A batch is a different tensor every step. `step.input()` copies it into a buffer the session
  owns, as `Trainer` does today, so the region sees one stable address. The two copies per step
  that `REMAINING_WORK.md` records for the compiled path apply here too.
- A boundary value lives in storage the session owns, not in its producer's arena (section 7). The
  session keeps that storage, and so its identity, when the producer region is rebuilt and the
  value's shape and element type did not change. That is what lets the readers of a changed region
  keep their compiled nodes. How the session matches a boundary value of one step with the storage
  of the previous step is part of the `Session` design and is not settled here.

A key match is trusted. A hash collision would replay the wrong kernels with no error, so the key
is a full 64-bit hash and the session also compares node counts and op identities before it
accepts a match.

## 7. Memory, fusion and boundaries

A value read by a node of another region is a boundary value. Boundary values, the loss, and any
value the caller reads are passed to the passes as observed, through `FusionOptions::observed` and
the planner's equivalent. They get their own `Storage` and are never folded away or overlapped.

Inside a region the passes work as they do today. A `matmul`, `bias_add`, `relu` chain folds into
one kernel when all three nodes are in one region. A chain cut by a scope boundary does not fold.
That is the price of a boundary, and it is the reason the default partitioner cuts only where the
caller asks.

A region owns its arenas and its `TensorPool`. They live as long as the region does and are freed
when it is dropped. An unchanged region therefore allocates nothing on a step, and the steady-state
allocation guarantee of the compiled path carries over to a loop whose structure has settled.

## 8. Flush points and phases

Work runs at three points: a host read, `backward()` followed by a read or `apply()`, and the end
of the step. A step with one read at the end runs once.

A read in the middle of a step, followed by more tracing, splits the step into phases. The runtime
runs every live region of a `DynamicGraph` on each `run()`, and has no way to run a subset. So the
session keeps one `DynamicGraph` per phase: phase `i` holds the regions traced between flush
`i - 1` and flush `i`. Phases run in order on the host, and a loop that reads at the same points on
every step reuses the same phase graphs.

A debug option flushes after every node. Each node is then its own region and its own phase, no
fusion applies, and every value is readable as soon as it is traced.

Differentiation across a mid-step read is not supported in the first version: `backward()` must be
called in the phase that produced the loss, over values of that phase and of earlier phases treated
as constants. A read that feeds a branch is the supported use.

This section is the least verified part of the design. It was derived from reading
`dynamic_graph.hpp`; no prototype exists.

## 9. What this spec assumes about SushiBLAS

Sent to the sushiblas session and recorded there as `SUSHIAI_CONSTRAINTS.md`:

1. A `TaskRecorder` can be constructed over any `TaskGraph&` and holds no other graph.
2. A recorded task can be emitted again. A region replays its emitter at every rebuild.
3. The value a record call returns does not name a node of one graph. SushiAI does not keep it.
4. Recording, compiling and executing are separate acts. On this path the runtime owns the graphs.
5. Every record declares its accesses as byte intervals from `Tensor::region()`.
6. Device work waits on the dependency events it is handed.

If the recorder that lands differs on any of these, this spec is revised before a plan is written.

## 10. Acceptance

The first version is done when these hold, each as a test or a recorded measurement.

- A model traced through the eager front end with no scope produces, step for step, the same loss
  bytes as `Trainer` on the same model, seed and data.
- A loop whose structure does not change reports one composition of the dynamic graph, and live
  allocations stop moving after the first step.
- Flipping a branch inside a scope rebuilds that scope's two regions and no other, counted by the
  number of emitter calls per region.
- A two-layer model with a scope between a `Linear` and its `ReLU` trains to the same loss as the
  unscoped model within a stated tolerance, and the fusion report shows the fold was refused at the
  boundary.
- The host cost of steps 3 to 5 of section 4 is measured on a step whose structure did not change,
  on the demo model and on a model of at least 50 layers, and reported beside the step time.
- The cost of renumbering (section 5) is measured for a scope that appears and disappears.
- The suite runs on the CUDA build. A test that needs the GPU is labelled and reported as skipped
  without one.

## 11. Not in this version

- Differentiation across a mid-step read (section 8).
- Higher-order derivatives through the eager path. The IR supports them; nothing here tests them.
- Mixed precision and loss scaling on the eager path. The cast pass runs per region as written, but
  the loss scaler's device-resident state is wired to `Trainer` and needs its own design.
- A partitioner that cuts per layer automatically.
- Any change to `Trainer`, `CompiledStep` or the JSON-configured applications.

## 12. Order of work

1. SushiBLAS lands the recorder and removes the engine facades.
2. SushiAI migrates its 131 facade calls to the new signatures and splits `lowering.cpp` into
   per-domain tables in the same pass. `Lowering` must be able to lower a subset of a graph's nodes
   into a recorder it is given; that is the one capability the eager path adds to it.
3. The host-lane units: `Scope`, `RegionPartitioner`, `ScopePartitioner`, `StructureKey`.
4. `RegionEmitter` and `RegionExecutor`.
5. `Session` and `Step`, then the acceptance tests.

Step 3 depends on nothing outside SushiAI and can start before step 1 finishes.
