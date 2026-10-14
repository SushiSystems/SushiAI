# graph

## Owns

The intermediate representation of a computation and the passes that act on it. `Graph` holds
`Value`s and `Node`s, addressed by `ValueID` and `NodeID`, and `GraphBuilder` emits
shape-checked nodes into one. The module also owns the passes over a graph (cast insertion,
static subtree fusion, the dynamic fusion selector, precision policies and im2col
rematerialisation), the memory plan with its `ArenaSet`, and the `TensorPool` that holds the
tensors behind the values. Lowering turns a graph into SushiBLAS tasks through the tables under
`lowering/`, and `CompiledStep` records one step on a `SushiRuntime::Graph`, compiles it in `compile()` and replays it, so `compile_count()` is 1 before the first run. `StateID` names model state
across graphs, and `Traced` carries a traced value or the failure that stopped the trace.

## Depends on

The module declares `core` as a public dependency. It declares no private dependency.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/graph/arena.hpp` | `ArenaSet` |
| `SushiAI/graph/builder.hpp` | `ConvGeometry`, `output_extent`, `SpatialExtents`, `PoolResult`, `BatchNormResult`, `BatchNormFold`, `BatchNormGrads`, `LayerNormResult`, `LayerNormGrads`, `DropoutResult`, `K_DROPOUT_RATE_SLOT`, `dropout_seed`, `GraphBuilder` |
| `SushiAI/graph/cast_insertion.hpp` | `CastOptions`, `InsertedCast`, `CastReport`, `insert_casts`, `DTypeMismatch`, `find_dtype_mismatches`, `describe` |
| `SushiAI/graph/compiled_step.hpp` | `CompiledStep`, with its `compile` and `run` |
| `SushiAI/graph/dropout_mask.hpp` | `splitmix64_at`, `dropout_site_seed`, `dropout_bits`, `dropout_keeps`, `fill_keep_mask` |
| `SushiAI/graph/fusion.hpp` | `FusedForm`, `fused_form_name`, `fused_form_from_name`, `fused_form_op`, `fused_form_of`, `FusionDecline`, `FusionOptions`, `FusedSubtree`, `DeclinedFusion`, `FusionReport`, `fuse` |
| `SushiAI/graph/fusion_selector.hpp` | `FusionQuery`, `FusionChoice`, `FusionSelector`, `DeviceFusionSelector`, `FusionSelection` |
| `SushiAI/graph/ir.hpp` | `ValueID`, `NodeID`, `Lifetime`, `lifetime_name`, `is_persistent`, `Attributes`, `TensorLayout`, `Value`, `Node`, `Graph` |
| `SushiAI/graph/lowering.hpp` | `Lowering`, with its `record` |
| `SushiAI/graph/node_range.hpp` | `NodeRange` |
| `SushiAI/graph/memory_plan.hpp` | `ArenaID`, `K_NO_ARENA`, `Arena`, `Placement`, `PlanOptions`, `MemoryPlan` |
| `SushiAI/graph/precision.hpp` | `PrecisionQuery`, `PrecisionChoice`, `PrecisionPolicy`, `UniformPrecisionPolicy`, `MixedPrecisionPolicy`, `narrows_under_mixed_precision` |
| `SushiAI/graph/rematerialise.hpp` | `rematerialise_im2col` |
| `SushiAI/graph/replay_prelude.hpp` | `ReplayPrelude` |
| `SushiAI/graph/state_id.hpp` | `StateID`, `K_NO_STATE`, `StateIdentity` |
| `SushiAI/graph/step_recorder.hpp` | `StepRecorder`, `StepRecorderList` |
| `SushiAI/graph/tensor_pool.hpp` | `TensorPool` |
| `SushiAI/graph/traced.hpp` | `Traced` |
| `SushiAI/graph/lowering/activation.hpp` | `K_ACTIVATION_LOWERINGS` |
| `SushiAI/graph/lowering/all.hpp` | `K_LOWERINGS` |
| `SushiAI/graph/lowering/context.hpp` | `LoweringContext` |
| `SushiAI/graph/lowering/elementwise.hpp` | `K_ELEMENTWISE_LOWERINGS` |
| `SushiAI/graph/lowering/fused.hpp` | `K_FUSED_LOWERINGS` |
| `SushiAI/graph/lowering/linalg.hpp` | `K_LINALG_LOWERINGS` |
| `SushiAI/graph/lowering/loss.hpp` | `K_LOSS_LOWERINGS` |
| `SushiAI/graph/lowering/norm.hpp` | `K_NORM_LOWERINGS` |
| `SushiAI/graph/lowering/random.hpp` | `K_RANDOM_LOWERINGS` |
| `SushiAI/graph/lowering/rule.hpp` | `LoweringRule`, `LoweringEntry`, `concat_lowerings` |
| `SushiAI/graph/lowering/shape.hpp` | `K_SHAPE_LOWERINGS` |
| `SushiAI/graph/lowering/spatial.hpp` | `K_SPATIAL_LOWERINGS` |
| `SushiAI/graph/lowering/unlowered.hpp` | `K_UNLOWERED` |

## Memory plan

`MemoryPlan::build` packs the Activation and the Scratch values into one arena each and decides
which buffers of an arena may share bytes.

Two Activation buffers share bytes only when every use of one reaches the first write of the
other in the graph. Two branches nothing orders therefore never share, and the runtime can run
them at the same time.

Two Scratch buffers share bytes when the last use of one comes before the first write of the
other in the order of the nodes. SushiRuntime orders two accesses whose byte ranges overlap in
the order they were recorded, and a step records its nodes in graph order, so the plan is
correct; the nodes that share scratch bytes run one after the other. Scratch holds what a
node needs for a short time, such as a column matrix `rematerialise_im2col` recomputes, and
nothing in the graph orders one such buffer after another, so the rule of the Activation class
would give each its own bytes.

A `RESHAPE` result is a view of its operand: the planner puts the two in one buffer, the pool
hands the result the operand's bytes under its own extents, and the row records no copy. Two
cases keep the copy. A row that writes an operand in place (`ACCUMULATE`, `EXP_SUB_SCALE`)
would change what the other side of the view reads, so a reshape whose operand or result such
a row touches is left alone; `LoweringContext::mutable_input` stops the process if a row
writes a viewed value all the same. A pinned result must keep bytes of its own.

A lowering row that writes a buffer on every run before it reads it states the buffer in a
scratch rule beside its row (`lowering/rule.hpp`, `ScratchRule`). `scratch_of` collects the
requests of a graph, `MemoryPlan::build` places them in the Scratch arena with their node as
the one user, and `TensorPool::scratch` hands the row a view of the arena. A buffer no rule
states keeps a `Storage` of its own, which is what a buffer the host fills once needs. The
pool stops the process when a row asks for other bytes than its rule stated.

A pool may borrow the arenas of another pool (`TensorPool::adopt`, `lender`). The two then
write the same bytes, so no step of one may run while a step of the other runs, and neither
may read a packed value after the other has run. The anchor training step does this: its
forward-only pass has finished, and the host has read its pinned outputs, before the replay
starts.

## Row gathers and integer leaves

`GraphBuilder::gather_rows` and `scatter_add_rows` return a `Traced`: a refusal marks the
graph and comes back as an `InvalidDescription`, where the older emitters throw. Both lower to
`SushiBLAS::SparseOps::spmm` over a CSR matrix that holds a single one in each row, at the
column the index names. The gather is the plain product, which copies rows. The scatter-add is
the transposed product, which adds the rows in index order inside one work-item per column, so
a repeated index gives the same bytes on every run. The row offsets of that matrix are an INT32
scratch of the node, filled on the host when the node is lowered.

A leaf may be INT32: `GraphBuilder::input` takes the element type. The cast pass leaves a leaf
and any operand that is not a real float as declared, and `Graph::finalise` gives an input no
gradient, so an index needs no special case in either. Neither row operation narrows under
`MixedPrecisionPolicy`. SushiBLAS checks an index against the table only in the reference kernel
of a build without `NDEBUG`; elsewhere an index outside `[0, V)` reads or writes outside it.

## Layer normalisation and the causal mask

`GraphBuilder::layer_norm` emits one `K_LAYER_NORM` over `[x, gamma, beta]` with the epsilon in
attribute slot 0. It normalises each `[D]` row of a `[..., D]` value by the row's own mean and
biased variance, and writes three results: the value, and the `[rows]` mean and inverse
deviation that `layer_norm_backward` reads. Both return a `Result` and refuse as the row
gathers do. The operation is its own node because the IR has no square root and its
elementwise operations take equal shapes: a composition would spell each broadcast as a product
with a vector of ones and still lack the root.

The lowering views the operand as `[rows, D]` and records 14 SushiBLAS tasks; the gradient
records 19. A value per row is broadcast by `SpatialOps::channel_affine` over a transposed
`[1, 1, D, rows]` view, whose channel axis is the row, and a value per column by the same kernel
over `[rows, 1, 1, D]`. Row sums are `ReductionOps::sum_along` with axis 1. SushiBLAS needed no
new primitive.

`GraphBuilder::causal_mask` emits one `K_CAUSAL_MASK` over `[..., T, T]` scores, with the fill
in attribute slot 0: an entry above the diagonal of the last two axes becomes the fill, and
every other entry is copied. It is its own node because `K_CONSTANT` fills a tensor with one
value and nothing adds a `[T, T]` value to each matrix of a batch. The lowering is one
`channel_affine` over a `[matrices, 1, 1, T * T]` view, with a scale of one and a shift of zero
where an entry is kept and a scale of zero and a shift of the fill where it is not. The two
`[T * T]` vectors are FLOAT32 scratch of the node, filled on the host when the node is lowered,
and converted once when the operand is not FLOAT32. Its backward rule emits the same node over
the gradient with a fill of zero. Neither operation narrows under `MixedPrecisionPolicy`.

## A step over part of a graph

`Lowering::record` and `CompiledStep::compile` take a `NodeRange`, the nodes from `first` up to
and without `end`; an `end` of `K_NO_NODE` stands for the node count, and the default range is
the whole graph. A range that ends before it starts or behind the last node is
`InvalidDescription`. Two steps compiled over two ranges of one graph share its `TensorPool`,
so the second reads what the first left in the pool and what the host wrote there between
the two runs. Each step has its own run count and its own preludes. `Train::TrainingStep`
uses this to stop a step for work on the host.

## Host work before each replay

A lowering row calls `LoweringContext::before_each_replay` with a `ReplayPrelude`, a
`std::function<void(std::uint64_t run)>`. `Lowering` collects the preludes of a pass in node
order and hands them over through `take_preludes`. `CompiledStep` adds each one to its graph
with `before_each_step`, in that order, and the hook of the trailing recorder's `before_replay`
behind them. The graph calls the hooks on the thread that called `run()`, before it submits the
step, and each prelude gets `run_count()`. `run_count()` is 0 before the first replay and counts the replays that
finished. `restore_run_count(count)` sets it to what the step of a saved run had reached, so
the preludes of a resumed run get the indices the saved run would have given next. It returns
`InvalidDescription` with the count once the step has run or has been restored to a count
above 0. `replay_prelude_count()` returns how many a step
holds. A step whose graph registers none and has no trailing recorder adds no hook.

A prelude writes host-addressable memory that no task of the step writes, while no replay is in
flight. The kernels of the next replay read those bytes. SushiRuntime states this in the
`before_each_step` section of
`modules/surface/api/include/SushiRuntime/api/graph/plumbing/README.md`. A caller that

uses `Lowering` without `CompiledStep` and replays the recorded work itself calls the preludes
it took. A `LoweringContext` built by the constructor without a list answers `Unsupported`.

## Dropout

`GraphBuilder::dropout` emits one `K_DROPOUT` over `[x]` with the rate in attribute slot 0 and
a 64-bit site seed in slots 1 and 2, 32 bits each; `dropout_seed` reads it back. The node
writes two results shaped and typed like `x`: the value, and the mask of the run, whose
elements are 0 or `1 / (1 - rate)`. The backward rule in `autograd` multiplies the incoming
gradient by that saved mask, so the forward and the gradient of one replay use one draw. The
rate lies in `(0, 1)`; the layer above emits no node at a rate of 0.

The row is in `lowering/random.hpp`. It records three tasks: a copy of the node's FLOAT32 draw
scratch into the mask (a converting copy when the operand is HALF or FLOAT64), a scale of the
mask by `1 / (1 - rate)`, and the product with `x`. A replay prelude fills the scratch with ones
and zeros before each replay. The mask is a planned value written by a task, so the memory plan
may pack it; only the scratch, which the pool never packs, is written from the host. The
operation does not narrow under `MixedPrecisionPolicy`.

The draws come from `dropout_mask.hpp`. `splitmix64_at(state, n)` is output `n` of the
SplitMix64 sequence seeded with `state`: the finaliser applied to
`state + (n + 1) * 0x9E3779B97F4A7C15`. A site seed is `splitmix64_at(trace_seed, site)`, the
key of a run is `splitmix64_at(site_seed, run)`, and the bits of an element are
`splitmix64_at(run_key, element)`. An element is kept when the top 53 bits, read as a number in
`[0, 1)`, are below `1 - rate`. A draw is therefore a function of the site seed, the run index
and the element index: it does not depend on a thread, on the engine's `RNGStream`, on the node
id or on what other sites exist, and a node copied with its attributes draws the same mask.
`core/data` and `vision` hold the same three SplitMix64 constants in units `core/graph` may not
depend on, so they are repeated here.

The host writes one FLOAT32 per element of `x` per replay. For a `[32, 256, 384]` activation
that is 3 145 728 elements and 12 582 912 bytes per site per step, and the copy moves the same
bytes once more on the device. The fill is a single-threaded loop of one finaliser, one
comparison and one store per element; at an estimated 1.5 to 2.5 ns per element it costs 5 to
8 ms per site per step. That figure is an estimate and no benchmark has measured it.

`record_keep_draw` in `source/lowering/random.cpp` is the one function that knows where the
ones and zeros come from. When SushiBLAS gains the positioned Bernoulli fill
(`docs/design/GRAPH_CAPABILITIES.md`, B1), its body becomes an INT64 `[2]` position scratch, a
prelude that writes the position, and one `PositionedRandomOps::bernoulli` into the mask. Its
signature, the row, the operation, the emitter and the backward rule stay as they are; the
masks themselves change, because Philox replaces SplitMix64.

## The gradient of a log-softmax

`GraphBuilder::log_softmax_backward` emits one `K_LOG_SOFTMAX_BACKWARD` over `[dy, log_probs]`,
two `[N, C]` values, and returns a `Traced`. Its result is `dy - exp(log_probs) * rowsum(dy)`.
`log_softmax_backward` in `autograd` emits it for a `K_LOG_SOFTMAX` that no fused cross-entropy
absorbed, which is how `softmax` (`RESHAPE`, `LOG_SOFTMAX`, `EXP`, `RESHAPE`) is differentiated.

It is one node and not a composition. Composed from `K_CONSTANT`, `K_MATMUL`, `K_EXP`, `K_MUL`
and `K_SUB`, the rule would add seven nodes and three `[N, C]` values beside the result: the row
sums spread back over the columns, the exponential and their product. In an attention `[N, C]`
is `[batch * T, T]` per head, the largest values of the block. The lowering records 7 SushiBLAS
tasks and keeps one `[N, C]` scratch and one `[N]` scratch: `gemv` with a vector of ones sums the
rows, `ger` writes minus that sum into every column of the zeroed scratch, the result takes
`exp(log_probs)`, and an in-place `mul` and an `axpy` finish it. SushiBLAS has no kernel that
scales each row of a matrix by its own scalar in place; with one the `[N, C]` scratch would go.
The operation does not narrow under `MixedPrecisionPolicy`.
