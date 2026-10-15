# eager

## Owns

The eager front end: a model is traced again on every step, and only the regions whose structure
changed are compiled and recorded again.

The units fall into two lanes. The host lane needs no engine. `ScopeTrace` splits a trace into
`Segment`s as scopes open and close, and `Scope` opens and closes one scope, so that an early
return or an exception cannot leave it open. `ScopePartitioner` turns the segments and the
`Autograd::Provenance` of a differentiated graph into a `Partition`, which assigns each node a
region key, or `K_NO_REGION` for a leaf; `RegionPartitioner` is the interface a partitioning
policy implements. `RegionSignature` and `signature_of` describe the structure of a region as a
comparable value. `find_boundaries` lists the values each region shares with the others,
`SlotLedger` matches each shared buffer with its storage of the step before, `RegionTable`
decides which regions a step keeps, records again, builds and drops, and `compile_region` cuts
one region out of the step's graph and runs the passes over it.

The device lane runs a planned step. `StorageBank` holds the tensor behind each `StorageID`.
`RegionEmitter` lowers one region's `RegionProgram` into a recorder, `StepExecutor` is the
interface that runs a `StepWork`, and `RegionExecutor` implements it over a SushiRuntime
`DynamicGraph`. `Session` owns what lives across steps, and `Step` traces one step and runs it
when a value is read, when an update is applied or when `finish` is called.

## Depends on

The module declares `core`, `graph` and `autograd` as public dependencies. It declares no
private dependency. SushiBLAS and SushiRuntime reach it through `core`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/eager/partition.hpp` | `RegionKey`, `K_NO_REGION`, `Direction`, `RegionInfo`, `Partition`, with its `optimizer_key` |
| `SushiAI/eager/region_boundary.hpp` | `ValuePosition`, `RegionBoundary`, `BoundaryMap`, `find_boundaries` |
| `SushiAI/eager/region_emitter.hpp` | `RegionEmitter`, with its `emit` |
| `SushiAI/eager/region_executor.hpp` | `RegionExecutor` |
| `SushiAI/eager/region_names.hpp` | `RegionID`, `K_NO_REGION_ID`, `RegionNames` |
| `SushiAI/eager/region_partitioner.hpp` | `RegionPartitioner`, with its `assign` |
| `SushiAI/eager/region_program.hpp` | `ProgramOptions`, `RegionProgram`, `compile_region` |
| `SushiAI/eager/region_signature.hpp` | `RegionSignature`, `signature_of` |
| `SushiAI/eager/region_table.hpp` | `RegionAction`, `RegionOrder`, `StepPlan`, `RegionTable`, with its `plan` and `forget` |
| `SushiAI/eager/scope.hpp` | `Scope` |
| `SushiAI/eager/scope_partitioner.hpp` | `ScopePartitioner`, with its `assign` |
| `SushiAI/eager/scope_trace.hpp` | `ScopeTrace`, with its `open`, `close`, `seal`, `depth`, `end_segment` and `current_path` |
| `SushiAI/eager/segment.hpp` | `Segment` |
| `SushiAI/eager/session.hpp` | `SessionOptions`, `Feed`, `StateGradient`, `StepTrace`, `StepCost`, `Session` |
| `SushiAI/eager/slot_ledger.hpp` | `ReaderSlot`, `SlotRecord`, `SlotLedger`, with its `keep`, `match_readers` and `place` |
| `SushiAI/eager/step.hpp` | `Step` |
| `SushiAI/eager/step_executor.hpp` | `RegionWork`, `StepWork`, `StepExecutor`, with its `execute` and `discard` |
| `SushiAI/eager/storage_bank.hpp` | `StorageBank` |
| `SushiAI/eager/storage_id.hpp` | `StorageID`, `K_NO_STORAGE`, `StorageIDSource` |

## What the caller writes

A loop builds one `ScopePartitioner`, one `RegionExecutor` on its engine and one `Session` over
both, and binds each model state once with `Session::bind_state`. Per step it calls
`Session::begin_step`, declares each batch with `Step::input`, takes the leaf of each state from
`Step::state`, emits nodes through `Step::builder`, and opens `Scope` objects on `Step::scopes`
where it wants a region boundary. `Step::backward` appends the gradients, `Step::apply` runs
the step with an update behind it, and `Step::read_scalar` returns the loss.
`Support::Loop` in `tests/core/eager/eager_run_support.hpp` is that loop in full.

The update is a `Graph::StepRecorder`. An `Optim::Optimizer` is one, so an optimizer built over
the tensors `Session::gradient` returns is applied without this module depending on `optim`.

## One step, in order

1. The caller traces forward nodes. `Step::backward` seals the scopes and differentiates.
2. `Session::run` asks the partitioner for a `Partition`.
3. It writes the storage of every state leaf, feed and parameter gradient into one list, and
   `find_boundaries` lists what each region imports and exports.
4. `RegionTable::plan` compares each region with the last step and returns a `StepPlan`.
5. `compile_region` runs for each region whose action is `RegionAction::Build`.
6. The session allocates the created slots, copies the feeds, and hands a `StepWork` to the
   executor, which drops, records and builds regions and runs the dynamic graph once.

`StepCost` holds the host time of stages 2, 4 and 5 and of the whole flush for the last step.
`tests/core/eager/test_session.cpp` prints them for a settled loop and for a
renumbering.

## Notes on the units

The sections below record why the code is the way it is. Each is named after a header and the
symbol it explains.

### segment.hpp

#### Segment

A segment is a maximal run of consecutively traced nodes that share one innermost open scope.
Opening or closing a scope ends the running segment, and a run with no node is dropped. Segments
are therefore contiguous, ordered by node id, and together cover every traced node exactly once.
That is what lets a partitioner number them without inspecting a single dependency: a node can only
read values traced before it.

The name is the path of open scopes joined by `/`, and it is empty outside any scope. `occurrence`
counts earlier segments of the same name in the same trace, so a scope opened in a loop, or an
outer scope that resumes after an inner one closes, yields distinct segments with one name.

### scope_trace.hpp

#### ScopeTrace

`ScopeTrace` does not observe node emission. It reads the graph's node count when a scope opens,
when one closes and at `seal()`, and that is enough to place every boundary. The builder and the
layers therefore need no change to be traced under scopes.

A scope name may not be empty and may not contain `/` or `#`, because the first joins a path and
the second separates a name from its occurrence in a region name.

### scope.hpp

#### Scope

`Scope` exists so that a scope cannot be left open by an early return or an exception between
`open()` and `close()`.

### autograd/provenance.hpp

#### Provenance

The partitioner must put a backward node in the backward region of the segment whose forward node
it differentiates. Operands do not say which that is. The weight gradient of a linear layer reads
the layer's input, which an earlier segment produced, so a rule based on operands would file it
under the wrong segment. The ordering would still be valid; the locality would be lost, and a
change in one scope would rebuild its neighbour.

Only the reverse walk knows which forward node a rule is running for, so `differentiate()` records
it. Every node appended while the walk is at forward node `f` has origin `f`: the nodes `f`'s rule
emits, and the `K_ACCUMULATE` nodes emitted when the gradients of `f`'s outputs are summed. An
`K_ACCUMULATE` emitted by the closing read of a requested gradient has the producer of that value as
its origin, which for a parameter is its `K_LEAF` node.

### partition.hpp

#### RegionKey

SushiRuntime orders two regions that share bytes by ascending key: the lower key is the producer.
A partition is valid when no node reads a value that a higher key produces.

#### Partition::optimizer_key

The parameter update reads every gradient and writes every parameter, so it needs a key above
every region. The partition reserves one and records nothing under it.

### scope_partitioner.hpp

#### ScopePartitioner

With `m` segments in trace order, a forward node of segment `k` (counted from 1) gets key `k`, and
a backward node whose origin lies in segment `k` gets key `2m - k`. Later segments get lower
backward keys, which matches the direction gradients flow: the backward region of segment `k` reads
gradients that the backward region of segment `k + 1` produced.

For `k = m` the two keys coincide, so the last segment's forward and backward nodes share a region.
When the caller scopes the loss, that region is the loss and the seed of its gradient.

A backward node whose origin is a `K_LEAF` sums contributions from several segments. It takes the
highest key among the regions of its operands' producers, so it runs after all of them.

`K_LEAF` nodes get `K_NO_REGION`. The lowering records nothing for a leaf.

The optimizer key is `2m`.

### region_signature.hpp

#### RegionSignature

Two steps produce the same region when their signatures compare equal. A match is trusted, and a
recorded kernel is replayed without being looked at again, so a false match would run the wrong
kernels with no error. The signature therefore carries the operation list beside the hash, and
equality compares both.

#### signature_of

The hash covers, for the region's nodes in order: the operation, the attributes, and for each
result its shape, element type and layout; then each operand's position.

An operand's position is never its value id or its producer's node id. Those shift whenever an
earlier region grows, and a region whose own structure did not change must keep its signature.

An operand produced inside the region is hashed by the index of its producer among the region's
nodes and its index among that producer's results.

Every other operand is external to the region: a leaf, or a value another region produces. It is
hashed by the identity the caller supplies for it, and by nothing else. A recorded kernel holds the
address of each operand, so what a region needs to know about an external operand is whether it is
still the same storage, not where in the graph it came from. The caller owns that storage and is
the only party that can say. A parameter keeps its storage for the run. A value that crosses a
region boundary keeps the storage the session bound it to for as long as the session keeps it, and
that is what lets a region survive the rebuild of the region that feeds it.

The function knows nothing about other regions for the same reason. Hashing a boundary operand by
its producer's region and the producer's index inside it does not work: the index moves whenever
the producer region grows before its last node, so every reader of a changed region would change
with it.

### storage_id.hpp

#### StorageID

A `StorageID` names one tensor the session keeps across steps: the tensor of a bound state, the
gradient tensor of a trainable state, a feed buffer, or a boundary slot. It is the identity
`signature_of` hashes for an external operand. An id is a number and not an address, so the table
and the ledger run on the host with no tensor behind them, and a freed address that the allocator
hands out again cannot pass for the storage a kernel was recorded with.

#### StorageIDSource

One source per session hands out every id, so an id of a state never equals an id of a slot.

### region_boundary.hpp

#### BoundaryMap

`BoundaryMap::buffer_of` gives each value the lowest value id that shares its bytes. Only
`K_ACCUMULATE` makes two values share: its result is written in place into its first operand.
This is the rule `Graph::MemoryPlan` applies inside one graph, applied here across regions,
because a gradient that several regions add into is one buffer.

A buffer is shared when two regions touch a value of it, or when a value of it is observed. A
region exports every value it produces into a shared buffer, not only the last one of a chain. The
lowering gives the first value of a chain its own tensor unless one is bound, and it then refuses
to alias the next value onto it, so every member has to be bound to the shared storage up front.

An accumulate whose first operand is a leaf would add into storage the session does not own, and
`find_boundaries` returns `StatusCode::InvalidDescription` for it.

#### RegionBoundary

Imports are listed once each, in the order the region's nodes first read them. That order depends
on the region's own nodes alone, which is what lets a reader name a value it imports by its place
in the list; see `SlotLedger`.

### region_names.hpp

#### RegionNames

A region's key changes when a scope appears or disappears, so the key cannot name the region
across steps. Its name and direction can: `"head#0"` with `Direction::Backward`. `RegionNames`
turns that pair into a `RegionID` once, and the table and the ledger compare integers afterwards.

### slot_ledger.hpp

#### SlotLedger

A slot is the storage of one shared buffer that the session, not a region, owns. The ledger
answers one question per shared buffer and step: which slot of the step before is this?

It asks in three ways, and the table chooses the order.

`keep` is for a producer whose signature and export positions did not change. The buffer sits at
the same position in the same region, so it is the same slot.

`match_readers` is for a producer that changed. A reader names the buffer by the reader's own
id and the place of the value among its imports. If a trusted reader read a slot of the same
producer, extents and element type under that name in the step before, the buffer takes that
slot. The reader's external identity is then unchanged, its signature is unchanged, and it keeps
its recorded kernels although the region that feeds it was rebuilt.

`place` is the fallback: the slot the producer wrote at the same position if it fits, or a new one.

A slot is taken only by the region that produced it before. Without that rule a changed region
could take the slot an unchanged region is about to keep.

A wrong match cannot run wrong kernels. The storage id enters the reader's signature, so a
mismatch rebuilds the reader. The rule decides how many regions are rebuilt, and nothing else.

### region_table.hpp

#### RegionTable

The table walks the regions in ascending key order, so every producer is placed before its
readers. For each region it gives the imports their storage, computes the signature, places the
exports through the ledger and compares the result with the entry the last step left under the
region's name.

A region is kept when its signature, its export positions and the storage of its exports all
agree with the entry. The export positions are part of the comparison because the passes treat an
exported value as observed: a value that another region starts to read has to stop being folded
away or packed, and that changes the compiled region although its nodes did not change.

#### RegionAction

`Record` is for a region that agrees with its entry and has another key. SushiRuntime knows a
region by its key, so the region has to be recorded again, but its compiled program is reused and
no pass runs.

#### Trusted readers

`match_readers` is given only the readers whose structure is the one the last step recorded. The
table computes that structure with `signature_of`, passing each import's place among the imports
in place of its storage. A reader whose own nodes changed gives its imports other places, so the
name it had for a slot means nothing in this step. Without this rule a rebuilt backward region
claims, under one of its own import places, the slot that the loss region is about to read the
head's result from.

The structure is computed when a region is built and when a region is asked about as a reader of
a changed producer. A step that keeps every region computes one signature per region.

#### RegionTable::forget

A step that fails after `plan()` leaves the table describing regions the executor does not hold.
`forget()` empties it, and the session drops the executor's regions with `StepExecutor::discard`,
so the next step builds everything.

### region_program.hpp

#### RegionProgram

`compile_region` copies the step's values with their ids and adds one `K_LEAF` node per import,
then the region's nodes. The lowering, the fusion pass and the memory planner therefore see a
graph of the shape they already handle, and `Graph::Lowering::record` lowers it whole. The eager
path uses `graph` as the compiled path does.

An import that another region computes gets `Lifetime::Input`: like a batch, it is written
before the region runs and is rewritten on every step, so the planner must not pack it.

The cast pass and the fusion pass get the imports and exports as observed, and the planner gets
the exports as pinned. `ProgramOptions::fusion` and `ProgramOptions::plan` carry the caller's
other settings; their `observed` and `pinned` lists name values of some other graph and are
replaced.

A chain of `K_MATMUL`, `K_BIAS_ADD` and `K_RELU` folds into one node only when all three are in
one region. A scope between the linear layer and its activation leaves a `GEMMBias` in one region
and a `K_RELU` in the next.

### storage_bank.hpp

#### StorageBank

The bank is the one place a `StorageID` turns into a tensor. `adopt` shares the storage of a
tensor the caller owns, which is how a state is bound; `allocate` creates a tensor the session
owns. `allocation_count()` counts the second kind and is what a test reads to show that a
settled loop allocates nothing.

### region_emitter.hpp

#### RegionEmitter

`emit` builds a `SushiBLAS::TaskRecorder` over the recorder it is handed and lowers the program
into it. The task recorder lives for the call: a recorded task holds its operands' storage and
its body, and nothing of the recorder. The emitter keeps the replay preludes and the fusion
selections of the last call, and counts its calls, which is how a test shows that a region was
recorded once.

SushiRuntime's `Region` hands out its `Recorder` through `Region::recorder`, and the emitter
records on that.

### step_executor.hpp

#### StepExecutor

The session decides and the executor acts. `StepWork` carries the decisions: per region a
`RegionOrder` and, for a built region, its program; the keys to drop; the storage of each shared
value; and the trailing recorder with its key. An executor that only counts what it is handed is
enough to test the session's decisions.

### region_executor.hpp

#### RegionExecutor

`execute` works in two passes, because a renumbering can give one region the key another held.
It first drops every key that leaves: the regions the step does not have, the old key of every
region it records or builds, and the trailing region when its key or its recorder changed. It
then installs the recorded and the built regions in ascending key order.

Each live region owns its program, a `Graph::TensorPool` and its emitter. A region recorded
under a new key gets a new pool and keeps its program. The pool of a dropped region is destroyed
at once; its tasks hold their operands' storage until the runtime composes the graph without
them.

The executor calls `DynamicGraph::region` only for a region it records. A kept region is not
touched, so a step that keeps everything composes nothing.

Replay preludes are called on the calling thread before each run, in key order, with the number
of steps the session has finished. They are not added to the dynamic graph with
`before_each_step`, because a hook stays for the life of the graph and a region does not.

The trailing recorder's `before_replay` runs after the preludes and its `after_replay` after the
run, the order `Graph::CompiledStep` gives them.

### session.hpp

#### Session

The session owns the `StorageBank`, the `RegionTable` and the feed buffers. It holds the executor
and the partitioner by reference.

A trainable state gets one gradient tensor when it is bound, and every step writes that state's
gradient into it. The tensor has to exist before the first step, because an optimizer takes its
gradient tensors when it is constructed. The session places the gradient value of each step in
that tensor directly, so no copy runs.

Two states whose gradient is one value in the graph would need two tensors for one buffer, and
`Session::run` returns `Unsupported` for that step.

A trainable state that a step does not trace, or that no gradient reaches, has its gradient
tensor filled with zeros before an update runs. Without the fill the update would apply the
gradient of an earlier step to a parameter the current step never used.

A feed is copied into a buffer the session keeps per feed position, so a region reads a batch
from one address on every step. A feed of another shape or element type gets a new buffer, and
its readers are rebuilt.

#### StepTrace

`StepTrace` is what a `Step` hands the session: the graph, the segments, the provenance, the
feeds, the observed values, the gradient of each traced state and the trailing recorder.

### step.hpp

#### Step

A step runs once. The run happens at the first of: `read` or `read_scalar` of a value, `apply`,
or `finish`. A step that is destroyed before any of them runs nothing.

`backward` seals the scopes before it differentiates, because the segments have to cover the
forward nodes and no other.

A value read after the run has to be one the step kept: the loss passed to `backward`, a value
passed to `keep`, or the value whose read started the run. Any other value may have been folded
into a fused node or overwritten in an arena.

Tracing a node after the step ran, or after `backward`, makes the next read return `Unsupported`.
The design's phases, one dynamic graph per stretch between two reads, are not built; see
`docs/design/REMAINING_WORK.md`.
