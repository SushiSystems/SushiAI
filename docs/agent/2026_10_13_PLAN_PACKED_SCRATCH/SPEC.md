# Plan-packed scratch: specification

**Status:** Shipped — 2026-10-13. Narrower than sections 4 and 5 propose; see `REPORT.md`.

## 1. The problem

One training step of YOLOv4 at 608 by 608 with a batch of 2 commits 11.1 GB (measured
2026-10-12). Under the default limit of `sa`, half the machine's memory, the step returns
`OutOfMemory`, and 12 cases of `Integration_DarknetYolov4Training` and
`Integration_DarknetYolov7Training` fail.

`MemoryPlan::build` packs the values of the graph into arenas and overlaps the ones whose live
ranges do not meet. The buffers a lowering row asks for through `LoweringContext::scratch()` are
not values of the graph. `TensorPool::scratch()` gives each one a `Storage` of its own, keyed by
node and slot, and keeps it for the life of the pool.

Batch norm in training mode shows the cost. Its forward row asks for one buffer of the operand's
size (`centred`), its backward row for two (`x_hat`, `correction`). A model with a hundred batch
norms holds three hundred buffers of activation size, and each is read and written only while
its own node runs.

How much of the 11.1 GB these buffers are is not measured. Section 7 makes that the first task.

## 2. What changes

A lowering row states the scratch it needs before it is lowered, in a form the planner reads.
The planner places each stated buffer in the arena of lifetime `Scratch`, beside the values that
already live there, and overlaps it with every buffer whose node is ordered before or after its
own. The pool hands the row a view of the arena where the plan placed the buffer.

Nothing changes for a caller of `CompiledStep`, `Trainer` or `Eager::Session`.

## 3. Why the overlap is safe

The planner overlaps two buffers only when every node that uses the first reaches the first
writer of the second in the graph. A scratch buffer has one user, its node. Two scratch buffers
therefore share bytes only when one node reaches the other, and a scratch buffer shares bytes
with a value only under the same rule the values already follow.

SushiRuntime orders tasks by the byte ranges they read and write (`Tensor::region()`), so two
tasks on the same bytes of an arena are ordered even when no value links them. The reachability
rule makes sure that order is one the graph already implies, so packing adds no edge between two
branches that could run side by side.

## 4. Units

Each is one responsibility. The names are proposals.

| Unit | Where | What it does |
| --- | --- | --- |
| `ScratchRequest` | `graph/lowering/scratch.hpp` | Holds one buffer a node asks for: slot, shape, element type, and whether it is `Transient` or `Held` |
| `ScratchRule` | `graph/lowering/rule.hpp` | A function `(const Graph&, NodeID) -> std::vector<ScratchRequest>`; `LoweringEntry` gains it beside `rule`, null for an operation with no scratch |
| `scratch_of` | `graph/lowering.hpp` | Collects the requests of every node of a graph through the table; host lane, no engine |
| `MemoryPlan::build` | `graph/memory_plan.hpp` | Takes the collected requests; gives each `Transient` one a buffer of lifetime `Scratch` whose one user is its node; answers `scratch_placement(node, slot)` |
| `TensorPool::scratch` | `graph/tensor_pool.hpp` | Returns the arena view the plan placed, or a `Storage` of its own when the plan holds none |
| `LoweringContext::scratch` | `graph/lowering/context.hpp` | Takes the slot alone and reads shape and type from the node's request, so a shape is written once |

`Held` marks a buffer that must keep its bytes between runs: one the host fills when the node is
lowered (the row offsets of `linalg.cpp`, the triangle factors, the dropout draw) or one a replay
prelude writes. A `Held` buffer stays a `Storage` of its own, as today. Every other request is
`Transient`: the node's own tasks write it before they read it, on every run.

The shared `ones()` tensors stay as they are. They are filled once and read by many nodes.

## 5. What the owner decides

1. **`LoweringEntry` gains a second function.** The alternative is a second table keyed by
   operation. One entry keeps a row and its scratch in one place; recommended.
2. **`MemoryPlan::build` gains a parameter**, a span of the collected requests, with an empty
   default. The alternative is to turn each request into a value of the graph before planning,
   which would put values in the graph that no node produces in the IR's own sense.
   The parameter is recommended.
3. **`LoweringContext::scratch(slot, shape, dtype)` becomes `scratch(slot)`.** Every lowering row
   that asks for scratch changes: 39 call sites in four files. Recommended, because the request
   and the row cannot then disagree about a shape.

## 6. Out of scope

- The activations and gradients a training step keeps for the backward pass. They are values of
  the graph and the planner already packs them.
- Releasing an arena between steps.
- The scratch pool of SushiBLAS (`Engine::scratch_pool()`).

## 7. Acceptance

1. **The share is known first.** `TensorPool::bytes()` reports arenas, own values, ones and
   scratch separately, and one step of YOLOv4-tiny prints them under the default memory limit.
   If scratch is under a tenth of the total, the work stops here and the owner is told.
2. Every existing test passes with unchanged bounds.
3. `Unit_MemoryPlan` gains cases: two scratch buffers of ordered nodes share bytes; two of
   unordered nodes do not; a `Held` request gets no placement.
4. After one step of YOLOv4-tiny, `TensorPool::bytes().scratch` holds only `Held` buffers, and
   the pool's total is reported beside the total before the change.
5. A full-size model is run only under the default memory limit. Whether YOLOv4 then fits is
   reported as measured; it is not a promise of this work.
