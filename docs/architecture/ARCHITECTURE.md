# SushiAI Architecture

This document describes how SushiAI is built today, and names only what
actually exists under `include/SushiAI/`, `src/`, and `cmake/`. **AI-1** has
landed: the graph IR, the differentiation transform, the lowering to
SushiBLAS, the layer/optimizer/data/training layers, and an MLP that trains
end to end. It will keep growing section by section — see
[`ROADMAP.md`](../design/ROADMAP.md)
for what's coming and in what order.

> If you only want to *use* the library, the top-level [README.md](../../README.md)
> is enough.

---

## 1. What SushiAI is

SushiAI is the AI/ML head of the SushiStack: a graph-native autograd core
built on [SushiBLAS](https://github.com/SushiSystems/SushiBLAS)'s tensor and
op library, which is in turn scheduled by
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime).

**SushiAI writes no SYCL.** No file here names a `sycl::` type, submits to a
queue or writes a `parallel_for`; every compute call goes through
`SushiBLAS::Engine`. Exactly one file — `src/graph/lowering.cpp` — turns *IR*
into those calls, which is what keeps the differentiation, fusion and
memory-planning passes device-free and testable without hardware.

Two places outside the lowering call the engine directly, and both are honest
about it because neither has an IR node to lower: `nn/init.cpp` and `data/`
fill buffers through the engine's RNG, and `optim/` records the parameter
update. `optim/` additionally hands SushiBLAS a **device functor** —
`Dispatch::execute_zip_inplace`'s generic elementwise seam inlines it into a
kernel SushiBLAS owns. That is plain C++ arithmetic on `auto&` parameters, not
a kernel this repository defines, but it is device code and §5.1 says what
holds it correct.

The pipeline for one training step is:

```
trace a model  ->  forward IR
differentiate  ->  forward + backward IR      (IR -> IR, no device work)
fuse           ->  fewer, bigger nodes        (IR -> IR, no device work)
lower          ->  a SushiRuntime TaskGraph   (the only SushiBLAS calls)
compile        ->  ONCE, per model
replay         ->  per batch, for ever
```

Two properties are load-bearing and both are asserted by the test suite:
the step is compiled exactly once no matter how many batches run
(`CompiledStep::compile_count()`), and the steady state allocates nothing
(`TensorPool::allocated_bytes()` stops moving after the first step). The
first of those is additionally pinned in the `regression` tier
(`tests/functional/regression/test_compile_once.cpp`) and in the demo's own
CTest smoke test, because a recompile-per-step still converges — it just
quietly pays graph analysis every step, and nothing else would notice.

Alongside the core there is the scaffolding a real library needs from day
one: a `find_package(SushiAI)`-able target verified by an out-of-tree
consumer, a three-tier test suite with CTest labels (`unit` / `integration` /
`regression`), and the `sa` CLI. `sa demo mlp` runs the showcase —
`examples/demo_mlp.cpp`, a flag-driven training program, deliberately a
separate binary from the `examples/example.cpp` teaching file.

### The whole stack is C++17, and that is not a style choice

SushiAI is built at **C++17** — the standard SushiRuntime and SushiBLAS are
built to. The three repositories have to agree on one standard, and the reason
is an ABI, not a preference.

`SushiRuntime::span` (`SushiRuntime/core/span.hpp`) *is* `std::span` when
`__cplusplus >= 202002L` and SushiRuntime's own C++17 shim otherwise. That
branch is taken per translation unit and it reaches the mangled name of every
function that takes one. SushiAI compiled at C++20 over siblings compiled at
C++17 put the two halves of one build on opposite sides of it — the same
mismatch that moved SushiBLAS down first
([its ARCHITECTURE §1](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/ARCHITECTURE.md#1-what-sushiblas-is)).
SushiAI was the last repository still above the line, so the stack now agrees.

The library's own API did not change: the same functions take the same
arguments spelled the same way, and a *header-only* use compiles under either
standard. What did change is the linkage of the out-of-line entry points whose
parameter type is a span — `Autograd::differentiate`,
`Core::allocate_tensor`, `Optim::StepScalars::write`, the `Autograd`
gradcheck entry points. Those now export the shim's mangling, so a consumer
must be at C++17 too; `tests/package/` says so in a comment, with the two
symbols side by side. See [INTEGRATION.md §3](../guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply).

The pieces C++20 had been supplying are C++17 spellings now:

| was | is | where |
|---|---|---|
| `std::span` | `SushiRuntime::span` | 105 sites across 31 files — the same type wherever `std::span` exists |
| `std::format` | `SushiBLAS::Core::format` | `SA_THROW_IF` and every report/diagnostic string |
| `concept Module` / `ModuleContainer` | `constexpr bool` variable templates over `std::void_t` detectors | `nn/module.hpp` |
| `template<Module... Ls>` | `template<typename... Ls>` + a `static_assert` fold | `nn/layers.hpp`'s `Sequential` |
| `consteval` | `constexpr` (called only from `static_assert`, so still proven at compile time) | `core/ops.hpp`, `src/graph/fusion.cpp` |

`SushiBLAS::Core::format` is **shared, not copied**: SushiBLAS carries the
stand-in as an installed public header and SushiAI links SushiBLAS
unconditionally, so a third copy of the same code would only have meant a
third place to delete when the stack returns to C++20. One guarantee is
genuinely lost with it — `std::format`'s `consteval` check on a literal format
string — so a placeholder/argument mismatch is now a slightly worse message
rather than a compile error. Diagnostics never fail the program; that is the
trade.

---

## 2. The sibling dependency chain (D3)

SushiAI depends on two siblings, resolved independently through the same
three-mode, layered CMake pattern used stack-wide (`cmake/Runtime.cmake`,
`cmake/BLAS.cmake`):

1. **TARGET-guard** — reuse an already-defined `SushiRuntime::SushiRuntime` /
   `SushiBLAS::sushiblas` if a superproject already brought it in.
2. **`find_package(... QUIET)`** — an installed package.
3. **Sibling `add_subdirectory`** — the common case: `../sushiruntime` /
   `../sushiblas` next to this checkout.

**Order matters.** `Runtime.cmake` runs before `BLAS.cmake` in the root
`CMakeLists.txt` specifically so that when SushiBLAS's own internal
`Runtime.cmake` runs (SushiBLAS `add_subdirectory`'d as SushiAI's dependency
also depends on SushiRuntime), its own TARGET-guard already finds
`SushiRuntime::SushiRuntime` defined and skips re-adding it — one runtime in
the whole build tree, not two competing copies.

Every consumer downstream — SushiAI's own targets, the CLI's `-D` flags, an
out-of-tree `find_package(SushiAI)` consumer — links the same two namespaced
aliases regardless of which mode actually resolved them.

---

## 3. The library target

`src/CMakeLists.txt` defines one target, built from one source list:

```cmake
set(SUSHIAI_SOURCES version.cpp graph/... autograd/... nn/... optim/... data/... train/...)

add_library(sushiai STATIC ${SUSHIAI_SOURCES})
add_library(SushiAI::sushiai ALIAS sushiai)

target_link_libraries(sushiai PUBLIC
    SushiRuntime::SushiRuntime
    SushiBLAS::sushiblas
)

sushiai_apply_sycl(sushiai SOURCES ${SUSHIAI_SOURCES})
```

`SUSHIAI_SOURCES` feeds both `add_library()` and `sushiai_apply_sycl()`, so
a source added directly to `add_library()` would compile without `-fsycl` on
the intel-llvm path and fail to link the moment it touches a SushiBLAS
kernel. One variable, both consumers.

Both dependencies are linked `PUBLIC` — the lowering needs SushiBLAS, and
every consumer needs it to construct an `Engine` anyway. SushiAI therefore
inherits SushiRuntime's `sushiruntime_deterministic_fp` floating-point
contract transitively rather than owning a copy of that guarantee, which is
also why gradient accumulation is ordered through the graph's edges rather
than through atomics (see `autograd/backward.hpp`).

`include/SushiAI/SushiAI.h` is the master header and includes every public
header. Two lanes live under it:

- The **host lane** — `core/` (except `core/allocation.hpp`),
  `graph/ir.hpp`, `graph/builder.hpp`, `graph/memory_plan.hpp`,
  `graph/fusion.hpp`, `graph/precision.hpp`, `graph/cast_insertion.hpp`,
  `autograd/`, `nn/module.hpp`, `nn/layers.hpp`, `nn/loss.hpp` — reaches no SushiBLAS
  *engine* and no SYCL. The `Unit_*` tests include these directly and run
  without a device. What it does reach of its siblings is vocabulary only, and
  every one of those headers is `<cstddef>`-and-friends deep:
  `<SushiBLAS/core/common.hpp>` for the dtype/layout enums,
  `<SushiBLAS/core/format.hpp>` for the message formatter, and
  `<SushiRuntime/core/span.hpp>` and `<SushiRuntime/graph/task_types.hpp>` for
  the span and the `OpID`.
- The **device lane** — `core/allocation.hpp`, `graph/arena.hpp`,
  `graph/lowering.hpp`, `graph/compiled_step.hpp`, `nn/init.hpp`, `optim/`,
  `data/`, `train/` — reaches SushiBLAS.

SYCL device-compilation policy (`sushiai_apply_sycl()`,
`cmake/SushiAISyclTarget.cmake`) is applied `PUBLIC` on the intel-llvm path
so examples and tests inherit `-fsycl` from linking `sushiai` rather than
reapplying it.

---

## 4. The IR, and why differentiation writes IR

`include/SushiAI/graph/ir.hpp` defines a single-assignment DAG of
`Value`s and `Node`s. `graph/builder.hpp` is the only thing that creates
nodes, so every operation's shape inference is stated exactly once.

A backward rule reads that IR and writes more of it. It never calls
SushiBLAS and never touches a `TaskGraph`. The full argument is in
`ir.hpp`'s file comment; the short version is that a `TaskGraph` node
carries byte ranges that cannot be known until the memory planner has seen
the whole forward *and* backward graph, that fusion needs a DAG to
partition, and that `TaskGraph::compile()` drains its builder and so cannot
be held and rewritten.

`autograd/backward.hpp` declares the reverse walk — `differentiate()` and
`has_backward_rule()` — and nothing else.

Gradient accumulation at a fan-out is tiered: zero contributors means no
gradient value at all (a symbolic zero, not a materialised buffer), one
means the gradient *is* that value, and two or more emit `K_ACCUMULATE` nodes
in contributor order.

### 4.1 One translation unit per domain

There is no `backward.cpp`. The implementation is three kinds of file:

| File | What it owns |
|---|---|
| `src/autograd/differentiator.cpp` | `differentiate()`, the reverse traversal, and the seed |
| `src/autograd/tape.cpp` | `GradientTape` — how a rule reads its node and contributes a gradient |
| `src/autograd/rules/<domain>.cpp` | The rules themselves, one file per domain |

The domains are `elementwise`, `activation`, `linalg`, `loss`, `shape`,
`spatial` and `norm`, each with a header beside it in
`include/SushiAI/autograd/rules/`. `rules/all.hpp` is the one place that joins
them.

The property this buys is a rule about *diffs*, and it is the reason the split
is worth having at all: **adding an operation to an existing domain changes
only that domain's `.cpp` and `.hpp`. Adding a whole domain is one line in
`all.hpp`.** Before the split, every new gradient — a pooling rule, a
normalisation rule, a loss rule — edited the same 600-line file, which is how
four unrelated reasons to change end up sharing a merge conflict.

A rule sees `GradientTape` and nothing else. `Differentiator`'s definition
lives in `src/autograd/differentiator.hpp`, outside the public include tree,
reachable from exactly the two translation units that construct or call back
into it.

**Nothing self-registers**, and that is not a stylistic preference. `sushiai`
is a **static** library. A translation unit whose only job is to run a
registration constructor resolves no symbol, so the linker drops it — the rule
table comes up short, `differentiate()` reports no rule for an operation that
plainly has one, and it happens *only in a downstream consumer's build*, not in
this repository's own tests. So each domain's rules reach the table by being
named in a file-scope `constexpr` array of `{OpID, function pointer}` that
`all.hpp` concatenates. Missing one is a compile error in this repository, not
a link-time absence in somebody else's. `core/ops.hpp`'s identity table and
`graph/fusion.cpp`'s `K_CATALOG` are the same shape for the same reason.

### 4.2 Three invariants the IR proves rather than assumes

**`TensorLayout`, proved by `Graph::prove_layout_uniformity()` (CV-4).** Every
`Value` carries a `TensorLayout`, and `K_LAYOUT_CONVERT` is the only node allowed
to have operands and results that disagree about it. `finalise()` closes the
forward graph — it propagates `requires_grad` and then proves uniformity — so a
builder-emitted graph that mixes NCHW into an NHWC chain fails there, naming
the node, instead of reaching a kernel that reads the right number of elements
in the wrong order.

That is a claim about the graph the builder emits, and it is worth being exact
about how far it carries. `finalise()` is called once, before differentiation,
and the stages after it rebuild the graph: `fuse()` and `insert_casts()` both
copy the value table entry by entry, and `Graph::add_value` defaults the layout
to NHWC, so a field one of them forgets to copy is not lost but *changed* —
silently, because the shapes still agree. Both passes therefore re-prove
uniformity over the graph they produce, which is why the proof is public rather
than a private step of `finalise()`. Backward-emitted nodes are still only
covered by whichever pass runs after them; a graph that is differentiated and
then handed straight to lowering is not re-proved anywhere.

`GraphBuilder::declare_like()` is how an emitter keeps the invariant true
without restating it: a result takes its operand's layout by construction.
`declare()` is left for the values that have no operand to inherit from: the
leaves (`input()`, `parameter()`), `layout_convert()` — which sets the tag
itself, being the one node allowed to change it — and `zeros()`, whose node has
no inputs and is therefore skipped by the per-node proof. A backward rule
emitting zeros beside a real operand calls `zeros_like()` instead, since the
exposure there is the *consumer*'s node, not `K_ZERO`'s.

`Lowering::record()` proves it once more before recording anything. That is the
one gate every path passes through, so the pass-level proofs are
belt-and-braces rather than load-bearing — and it is where the tag stops being
bookkeeping, since `record_layout_convert()` reads it to decide whether to
relabel a side into `[N,H,W,C]`.

**The `differentiable` flag is split from its proof.** `core/ops.hpp` carries
the flag on each `OpName` — the *claim* that an operation has a gradient.
`autograd/rules/all.hpp` carries the `static_assert` that every operation
making that claim is in the rule table — the *proof*. They are apart because
they must be: the assert needs to see the rule table, and `core/` cannot depend
on `autograd/` without inverting the layering the whole tree is built on. The
flag stays where the operation is declared; the check lives where the evidence
is. `K_BATCHNORM_UPDATE_STATS` claims nothing and has no rule, because CV-7 keeps
a BatchNorm's running statistics out of the differentiated graph entirely — it
is recorded into the training step beside the graph, the way `K_SGD_UPDATE` and
`K_ADAMW_UPDATE` are, and it is correspondingly one of the three declared
identities with no case in `Lowering::record_node` — the other two being
`K_SGD_UPDATE` and `K_ADAMW_UPDATE` themselves, for the same reason. All three
throw "No lowering for operation" if a node ever carries them, which is the
correct answer rather than a missing case.

**`graph/rematerialise.hpp` is deliberately not general (CV-10).**
`rematerialise_im2col()` recomputes exactly one operation: it clones the
`K_IM2COL` node in front of the backward consumer that reads its column matrix,
demotes the stored value to `SCRATCH`, and stops. It is not a cost model, not a
checkpointing framework, and not parameterised over which operations are cheap
to recompute. A convolution's column matrix is `kernel_h*kernel_w` times its
input and is read once in the backward pass, which makes it both the largest
activation in a CNN and the easiest thing in the graph to rebuild; a general
pass would need a cost model nobody has measured, to serve cases nobody has.
The clone copies the original's `Attributes` rather than rederiving them, for
the reason the next paragraph gives.

**`Attributes::K_CAPACITY` is 10, and the last two slots are not redundant.**
Slots 0–7 are a 2-D window's geometry — `kernel_h`, `kernel_w`, `stride_h`,
`stride_w`, `dilation_h`, `dilation_w`, `pad_top`, `pad_left` — mirroring
`Graph::ConvGeometry` field for field. Slots 8 and 9 are the output extents
`out_h` and `out_w`, **stored rather than derived**. That looks like
duplication and is not: CV-5 gives the geometry no `pad_bottom` or `pad_right`,
so deriving the extents back from it silently assumes symmetric padding — which
YOLOv3-tiny's 13→13 pool violates. A wrong derivation there yields a window
grid that is the *transpose* of the real one, with the same element count, so
it passes every shape check in the tree and gathers the wrong pixels. Storing
what the host already computed is the fix, and A-8 ("never re-derive an output
extent from geometry") is the rule it generalises to.

`K_CAPACITY` is bounded by nothing outside SushiAI, and deliberately so.
`Lowering` reads slots with `get()` and hands the values straight to a
SushiBLAS call, so no attribute is ever copied into SushiRuntime's fixed-width
`TaskMetadata::params` array and `K_MAX_PARAMS` does not constrain it. The
capacity is what the operations need, and nothing else.

---

## 5. Memory planning, and lowering

`graph/memory_plan.hpp` decides where every non-persistent value's bytes go,
before anything is lowered. It is a pure function of the IR — no engine, no
allocator, no SYCL — which is why the whole pass is unit-tested over
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
  runtime was free to run in parallel — memory reuse *injects* WAR and WAW
  anti-dependencies, and only the reachability form guarantees the injected
  edge was already implied. The transitive closure costs O(V²/64) memory, so
  above `PlanOptions::reuse_node_limit` nodes the planner declines to overlap
  anything and says so through `reuse_note()` rather than substituting the
  linear approximation.

`graph/arena.hpp`'s `ArenaSet` is the device-lane other half: it allocates
one `Storage` per arena in its constructor and never allocates again, and
hands out views through `view_tensor()` (`core/allocation.hpp`), which
refuses a byte offset that is not a whole number of the view's elements —
`Tensor::storage_offset` counts elements, so such an offset has no
representation.

This is only safe because SushiBLAS now honours `storage_offset` everywhere
and `Tensor::region()` reports a view's own byte interval rather than the
bare allocation base; two disjoint tensors in one arena therefore produce
non-overlapping `ResourceRegion`s and do not falsely serialise. That claim is
checked directly in `tests/functional/integration/test_arena.cpp`.

`Lowering::values_needing_own_allocation()` keeps the two reduction operands of
`K_LOG_SOFTMAX` and `K_NLL_LOSS` out of the arena, and it is a **workaround whose
premise no longer holds**. It was written when `Dispatch::execute_reduction`
built its `ResourceRegion`s from raw `data()` pointers, so a reduction operand
at a non-zero offset got *no* dependency edge at all. SushiBLAS fixed that —
both `execute_reduction` and the newer `execute_folding_axis_reduction` build
their regions from `Tensor::region()`, and SushiBLAS pins it in
`Regression_Sentinel.ReductionsOrderAgainstViewsOfOneAllocation` and
`Unit_AxisReductions.AxisReductionsOrderAgainstViewsOfOneAllocation`. Deleting
this function is therefore unblocked; it has not been done, because it changes
the memory plan and deserves its own change and its own arena verification
rather than riding along with an unrelated one.

`src/graph/lowering.cpp` is the only file that calls `SushiBLAS::Engine`. Its
`TensorPool` asks the plan where each value goes: packed values become arena
views, everything else gets its own `Storage`. The planner assumes **every
node writes all of its outputs on every replay**, which is why `K_ZERO` records
its fill into the step rather than into the one-time constant queue.

`graph/compiled_step.hpp` wraps the compile-once/replay contract in a type,
because the ordering has a silent failure mode: `rearm()` before
`capture_base_state()` leaves the completion sentinel armed at in-degree
zero, so the run reports complete before any kernel has been dispatched and
the step quietly does nothing. Doing both in the constructor makes that
state unrepresentable.

It also takes an optional `StepRecorder` (`graph/step_recorder.hpp`) — work
that belongs to the step but has no IR node behind it, recorded after the
lowering and before the one `compile()`, and driven around every replay. The
optimizer's parameter update is what it exists for; see §5.1.

### 5.1 The optimizer is part of the compiled step

Spelled as separate SushiBLAS calls, an AdamW update is about eleven launches
per parameter per step — for an eight-tensor model, ~88 launches against the
~31 the whole forward and backward graph costs. Both optimizers now issue
**one** `Dispatch::execute_zip_inplace` per parameter instead: that seam hands
a functor lvalue references to every operand's element, so the entire recipe
runs in one kernel with `g²`, the denominator and `m/denom` living in
registers. The scratch buffers those temporaries needed are not allocated at
all on the fused path.

Fusing is also what let the update *join the plan*. The recorded obstacle was
real — a per-step scalar baked into a recorded node is a compile-time constant,
and AdamW's bias corrections change every step — but a single functor can
capture a **device pointer** instead, and read the step's numbers from the
small `Optim::StepScalars` buffer behind it. A pointer is capturable by value
and its contents may change between replays, so the recorded work is
launch-invariant and goes into the same `TaskGraph` as forward and backward.
Ordering needs no new machinery: every call declares the bytes it reads and
writes, so the tracker already sees that the update reads the gradients the
backward pass wrote and writes the parameters the forward pass read.

Two guarantees hold this down, both in the regression tier
(`tests/functional/regression/test_fused_optimizer.cpp`): the fused kernel and
the separate operations agree **byte for byte** (they can, because the whole
stack builds with `-ffp-contract=off` and every expression keeps the operand
order of the kernel it replaced), and folding the update into the plan changes
no number either. `UpdateKernel::Separate` keeps the reference spelling
reachable; asking a trainer to compile one in is a thrown error, never a quiet
downgrade. So is calling `step()` on an optimizer a plan already owns: the
update is invisible inside a replay, so a stray call would train at double the
learning rate and report nothing.

What it is worth, measured rather than assumed: on `sa demo mlp`'s four
parameters, a step goes from 81 device tasks in three dispatch rounds to 37 in
two, and a whole run from 278 `TaskGraph::compile()` calls to 150. The wall
clock moves much less — a median 8.80 to 7.74 ms/step over ten runs each on a
quiet CPU/OpenCL box — because this model's largest tensor is 4096 elements
and most of a step is neither the optimizer nor arithmetic. The launch counts
are the honest number; the microbenchmark's 16-22x is what the same change is
worth when the optimizer *is* the step.

---

## 5.2 Fusion is a selector over a catalog, not a code generator

`sycl_ext_codeplay_kernel_fusion` has been **removed** from intel-llvm, so
nothing in this stack fuses two kernels at run time. A fused kernel is a C++
template instantiation of a functor composition, which means *which
compositions exist* is decided when SushiAI is compiled and a pass running at
graph-build time can only choose among them. Everything about the shape of this
work follows from that one fact:

- **`core/ops.hpp` holds the catalog's identities** — `gemm_bias`,
  `gemm_bias_relu`, `gemm_bias_sigmoid`, `gemm_bias_tanh`, `exp_sub_scale` —
  each a distinct kernel in the binary rather than a flag on a generic one.
- **`graph/fusion.hpp` is the static pass**: host lane, pure IR to IR, no
  engine and no SYCL, unit-tested over synthetic graphs with no device. It
  matches subtrees longest-first and rewrites each accepted match into one
  node. It runs *after* differentiation, on the joint forward-and-backward
  graph, because a forward node's fusability depends on what the backward pass
  still reads.
- **`graph/fusion_selector.hpp` is the dynamic half**: given the device's
  `SushiBLAS::Device::Profile` and the engine's `ExecutionMode`, it chooses
  which matched entry to actually spell. Device facts live only here; the IR is
  the same on every machine and only one node's spelling changes.
- **`graph/lowering.cpp` holds the functors**, because a functor inlined into a
  SushiBLAS kernel is device code.

Adding a form is four mechanical edits: an identity, a `FusedForm`, a `K_CATALOG`
row, and a recorder. Nothing self-registers, for the same reason
`autograd/rules/all.hpp`'s rule table does not (§4.1).

### What is allowed to be fused away, and what is not

Three refusals, each of which produces a graph that still trains if you get it
wrong, and each therefore a checked rule:

1. **A value with a second consumer.** A fused kernel never writes the chain's
   interior values, so fusing one that something else reads means recomputing
   it or losing it. The pass counts uses; it does not assume a chain.
2. **A value the caller reads by id.** The loss and the logits are ordinary
   activations to the pass and are read by the training loop after every
   replay. `FusionOptions::observed` is how `build_classifier` says so;
   parameters, gradients, optimizer state and inputs are protected by their
   lifetime class without being listed.
3. **A value the backward pass differentiates at.** This is rule 1 again,
   deliberately: by the time the pass runs the backward nodes are in the graph,
   so a pre-activation the gradient reads is simply a second consumer.

That third point is where the per-activation decision lives, and it is a
property of the op rather than a case inside the pass. `Ops::K_ACTIVATIONS` states
where each activation's gradient reads, once, and both the differentiation pass
and the catalog's `static_assert` consult it:

| activation | gradient reads | foldable | entry it needs |
|---|---|---|---|
| ReLU | its output | yes | any; the pre-activation was dead already |
| sigmoid | its output | yes | any |
| tanh | its output | yes | any |
| GELU | its **input** | yes, **with a spill** | one that writes the pre-activation to a second output |

ReLU moved from reading its input to reading its output as part of this work,
and that is a substitution rather than an approximation: SushiBLAS's
`relu_backward` is `dy * (primal > 0 ? 1 : 0)` and `relu(x) > 0` exactly when
`x > 0`, so the two spellings are **bit-identical**. Making it unconditional -
not a rewrite the fusion pass applies - is what leaves the pre-activation with
a single consumer in the plain graph too, so an ordinary use count says yes for
ReLU with no activation-specific code in the matcher at all, and would say no
for GELU. The three entries in `docs/design`'s §25 note about GELU/SiLU/sigmoid/
tanh being non-recoverable is narrower than it looked: sigmoid and tanh **are**
exactly recoverable from their outputs (`y(1-y)`, `1-y²`) and SushiBLAS's rules
already take the output, so GELU was the only one the use count genuinely
refused.

**The auxiliary-buffer GELU epilogue — cuBLASLt's `GELU_AUX_BIAS` shape — is
implemented.** `FusedForm::GemmBiasGelu`'s epilogue stores the fully formed
pre-activation to a second buffer on its way past and then returns `gelu` of it,
so the tensor `gelu_backward` differentiates at is written after all and the
fold is exact rather than approximate.

Four things make that work, and each is a place it could have gone silently
wrong:

- **The second output is the same value.** `Graph::Node::outputs` was already a
  vector; the fused node's `outputs[1]` is the very `ValueId` the absorbed
  `K_BIAS_ADD` used to produce. So `gelu_backward` reads what it always read, no
  backward rule changed, and no pass had to learn about a new tensor.
- **Rule 1 is relaxed for exactly that value and nothing else.** A catalog entry
  names one `spill` step; the interior value at that step may have any number of
  consumers and may be observed, because it still reaches memory. Every other
  interior value is governed by the use count as before.
- **Order, not just liveness.** The fused node is emitted at the chain's root,
  later than the absorbed producer, so a consumer of the spilled value that sat
  between the two would read a value written after it.
  `FusionDecline::SpillReadTooEarly` refuses that case; the training graph
  never hits it, because `gelu_backward` is appended after the forward pass.
- **The write is declared.** `Dispatch::gemm_epilogue(f).reads(bias).writes(aux)`
  puts the spill in the GEMM task's own write set, so whatever reads it is
  ordered behind the kernel that filled it. Without that declaration nothing
  fails and nothing is slower — the reader simply races the write, which is the
  hazard SushiBLAS's seam exists to close.

The `static_assert` in `fusion.cpp` learned the difference the previous version
of this section predicted it would have to. It no longer refuses every entry
folding an `INPUT`-gradient activation; it refuses one that folds such an
activation **without** a spill, or that spills the wrong step of the chain. It
still fails the build for a hypothetical `GEMM_BIAS_SILU` added without an
auxiliary write.

Under a narrowing precision policy the entry is unreachable, and deliberately
so — see §6.2.

### The cost the epilogue carried

The same gap bit the bias. A fused GEMM's epilogue dereferences a bias vector
the tracker cannot see, and that bias is a parameter the optimizer rewrites
*inside the same compiled plan* - so an undeclared read is a write-after-read
hazard nothing orders. A transitive path through the gradient happens to exist
for a plain feed-forward model, and relying on it is exactly the accident that
survives every test and breaks on the first second writer.

This used to be paid for with a `Lowering::declare_epilogue_operand()` that
emitted a `TaskGraph::add_native` node running nothing, present only for its two
access declarations - one node per fused GEMM, no kernel launch, counted apart by
a `RunReport::ordering_nodes_per_step`. All of that is gone.
`Lowering::record_fused_gemm` now hands SushiBLAS the epilogue and its operands
together, `gemm_epilogue(BiasReluEpilogue{...}).reads(bias)`, once per catalog
arm - per arm rather than once above the `switch`, because the access set belongs
to the spelling and an arm that spilled a pre-activation would declare a write
the others do not. The edge lands on the GEMM task itself, so
`operations_per_step` is the plan's node count again.
`Regression_Fusion.AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias` asserts
that edge on `in_degree` rather than on a loss: with no edge the tasks may still
come out in the right order on any given run, so a loss that falls proves
nothing.

### What it is worth, measured — and what it costs

On `sa demo mlp`'s model a step goes from **35 recorded SushiBLAS calls to 25**,
all of it out of the forward and backward
graph: the first layer's `gemm+bias+relu` replaces five calls with one, the
second layer's `gemm+bias` replaces three with one, and the cross-entropy
gradient's `exp/sub/scale` tail replaces five with one. Every fused form is
compared byte for byte against the unfused spelling in
`tests/functional/regression/test_fusion_numerics.cpp`, and
`FusionOptions::enabled = false` keeps that reference reachable, exactly as
`Optim::UpdateKernel::Separate` does for the optimizer. The demo's loss
trajectory is identical to the last printed digit with the pass on and off.

The wall clock has **two terms, and only one of them is an improvement**:

- **Steady state: 0.57 → 0.40 ms/step**, the slope of total time against step
  count taken between 1 024 and 4 096 steps (median of three replicates,
  spread about ±0.06). A 30% saving against a 29% reduction in kernels, which
  is what one expects when the tensors are small enough that a step is mostly
  launch overhead.
- **First step: 0.97 → 1.88 s.** Each fused form is a distinct template
  instantiation and therefore a distinct SYCL kernel, which this OpenCL backend
  compiles at first launch. The cost is one-time, per process, and independent
  of model size, batch and step count.

Break-even is therefore around **5 000 steps** on this machine, and the demo's
default 128-step run sits far below it — with fusion on it is about 1.9x slower
in wall clock while issuing 29% fewer kernels. AOT device targeting
(`SB_SYCL_TARGETS`) is what moves that term to build time. This is recorded
here rather than quietly omitted because a fusion pass is exactly the kind of
change whose headline number invites reporting only the favourable half.

---

## 5.3 Mixed precision is structure in the IR, not a runtime switch

`graph/precision.hpp` and `graph/cast_insertion.hpp` are the dtype half of
AI-4. They are split along the same seam `fusion.hpp` and
`fusion_selector.hpp` are split along, and for the same reason: the **pass**
is a mechanical rewrite anybody can verify, and the **policy** is a set of
judgements about numerics that has to be argued. Putting them in one file
would mean the argument and the machinery move together.

- `PrecisionPolicy` answers, per operation, which element type it computes
  in. `UniformPrecisionPolicy` — the default — answers "the graph's own
  type" to everything, so a graph built without a policy is byte-for-byte what
  it was before this existed. `MixedPrecisionPolicy` narrows two families and
  nothing else: `K_MATMUL` and the fused GEMM forms, and the `K_BIAS_ADD` plus
  foldable activation those forms absorb.
- `insert_casts()` reads the policy and rewrites the graph so every node's
  operands and results carry its computed type, emitting an `Ops::K_CAST` node
  at each boundary. `find_dtype_mismatches()` states the resulting invariant
  as a function anyone can call — including after fusion, which is where it
  earns its keep.

**Why the epilogue's precision is forced rather than chosen.** A
`K_GEMM_BIAS_RELU` node applies its epilogue inside the GEMM's kernel, in the
GEMM's element type. If the IR said `K_BIAS_ADD` computes in fp32, fusing it
into an fp16 GEMM would change the numbers — and §5.2 promises fusion changes
none. So the policy does not get to pick for anything a catalog entry **will**
absorb; it has to agree with the kernel that absorbs it.

**`GELU` is the one place that rule meets a limit, and the resolution is the
pass order.** Since `K_GEMM_BIAS_GELU` exists, a catalog entry *can* absorb GELU —
so "nothing folds it, therefore nothing forces its precision" is no longer an
argument for anything. The reason it still stays wide is arithmetic and nothing
else: the tanh approximation computes `x*x*x`, fp16's largest finite value is
65504, so the cube overflows at |x| ≈ 40.3 and `tanh(±inf)` then returns ±1 —
`0.5*x*(1+tanh)` yields `x` or `0` rather than a NaN, a wrong number with no
symptom. (That last step is IEEE-754 arithmetic, not a figure measured here.)
Loss scaling cannot reach it either, because it is a *forward* activation and
the scale is applied to the backward seed.

Keeping it wide makes `K_GEMM_BIAS_GELU` an **fp32-only catalog entry**, and that
falls out of the pass order rather than needing a special case: the cast pass
runs before fusion, so an activation the policy keeps wide gets a `K_CAST` between
it and the narrowed `K_BIAS_ADD`, and a chain with a conversion in the middle
matches no entry. Under a narrowing policy a GELU layer therefore folds to
`gemm+bias` and the activation runs on its own — exactly as it did before the
entry existed — and `narrows_under_mixed_precision` answers false for both
`K_GELU` and `K_GEMM_BIAS_GELU` so the two halves cannot drift.
`Regression_MixedPrecision.AGeluLayerIsNotFoldedUnderANarrowingPolicy` pins it.

**Where the pass runs, and why exactly there.** Between `differentiate()` and
`fuse()`. After differentiation, because a backward node needs casting as much
as a forward one and there are none to cast before the transform runs; before
fusion, because a conversion dropped into the middle of a `K_MATMUL` →
`K_BIAS_ADD` → `K_RELU` chain would break the chain the matcher is looking for.

**Master weights.** A parameter keeps its id, its `FLOAT32` dtype and its
`Lifetime::Parameter`; so does its gradient, and the optimizer is untouched.
The fp16 a GEMM reads is a *derived* value with an activation lifetime,
produced by a cast and rewritten every replay because the parameter it reads
changed. The rule the pass applies is one sentence — a value's element type
follows its producer's computed type unless something outside the pass depends
on it — and "outside" means a persistent lifetime class or a value the caller
named as observed. The backward direction falls out of it: a gradient value is
`Lifetime::Gradient`, so it is pinned, so an fp16 matmul writes a fresh narrow
result and a cast widens it into the buffer the optimizer accumulates in.

**What it is worth here: correctness, and nothing else.** This machine has no
fp16 matrix unit, so half arithmetic is emulated and `--precision mixed` is
several times *slower* than fp32. No timing gate exists for it and none should
until the hardware does. What is real and tested is that the graph carries the
right types, that the masters stay fp32, and that the model still learns — see
[README.md § Mixed precision](../../README.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here)
for the measured loss curves both ways.

**Loss scaling is here too, and it is `optim/loss_scaler.hpp`.** The backward
seed is multiplied by a device-resident power of two (`Ops::K_DYNAMIC_SCALE`,
emitted by `differentiate()`'s seed rule so it lands in topological position
and before anything narrows); `reductions().count_nonfinite()` screens every
gradient inside the same compiled plan; the fused update divides the scale back
out before the decay and the moments, and returns without touching anything
when its parameter's screen came back non-zero. The scale halves on overflow
and doubles after N clean steps, and it is device-resident so that neither
move recompiles the step.

Two things about it are worth knowing before relying on it. **On this workload
it buys nothing** — fp16 without it neither stalls nor diverges here, so it is
justified by theory and by larger models. And **the skip is per parameter**:
the ordering that makes an update wait for its own screen is the
write-after-read hazard on that gradient, and there is no such hazard between
parameter A's update and parameter B's screen, so a step where only some
gradients overflowed is a partial update rather than an empty one. Nothing
unsafe results — no non-finite value reaches a parameter or a moment — and the
header states what a whole-step skip would cost.

---

## 5.4 A model can be a file, and the driver that reads one is written once

`NN::Sequential` decides its layer list at compile time: it holds a
`std::tuple<Ls...>`, its `fields()` is `static constexpr`, and the parameter
walk over it is a template walk. A JSON file decides that list at run time. The
two cannot be reconciled — one is a type, the other is data — so there is a
second, runtime-polymorphic path: `nn/layer.hpp`'s `Layer` and
`nn/layer_stack.hpp`'s `LayerStack`.

**In this architecture that path costs nothing.** `forward()` runs *once*, at
trace time, to emit IR; the graph is compiled once and replayed (§1). A
`std::vector<std::unique_ptr<Layer>>` therefore adds no per-step work at all.
The same design would be indefensible in an eager framework, where it would
mean a virtual call per layer per step — the graph-native architecture is what
pays for this feature.

### One walk, three cases

`for_each_parameter` (`nn/module.hpp`) dispatches on three cases now, tried in
this order: a type that enumerates itself through `walk_parameters`; a
container that recurses through `visit_children`; a leaf that applies its
`fields()` list. The first arm is what a runtime layer list needs, because
`fields()` is static and returns member pointers of one concrete type, and a
container walk over children that are all `Layer&` would find no parameters
*while reporting no error*.

The arm is strictly additive — `layers.hpp` asserts that neither `Linear` nor
`Sequential` takes it — and everything downstream is a template over that one
walk, so `NN::initialize`, `Train::build_graph`, `NN::parameter_elements` and
the checkpoint I/O all consume a `LayerStack` without knowing a second kind of
model exists. A virtual cannot be a template, so the visitor is type-erased
through `NN::ParameterVisitor`, which *refers* to its callable rather than
owning it and therefore allocates nothing.

### Four registries, and why they are constant tables

```
apps/train.cpp  (no type names; never edited again)
   |
   +-- NN::K_LAYER_REGISTRY       "linear", "gelu", "relu", "tanh", "sigmoid",
   |                            "leaky_relu", "unflatten", "conv2d",
   |                            "maxpool2d", "avgpool2d", "global_avgpool",
   |                            "flatten", "upsample"
   +-- Data::K_DATASET_REGISTRY   "rings", "mnist"
   +-- Optim::K_OPTIMIZER_REGISTRY "adamw", "sgd"
   +-- Train::K_OBJECTIVE_REGISTRY "cross_entropy"
```

Each is a file-scope `constexpr` array of `{name, factory function pointer}`,
in the shape of `Ops::K_ALL` and the fusion `K_CATALOG`, and `"type"` is the
discriminator key in all four — so `Config::find_entry` is written once and
serves all of them. **Nothing self-registers**, for the reason §4 gives for the
backward-rule table.

Two rules make growth local. A **layer's own factory** is the only thing that
knows which keys it accepts, so adding a layer type never edits the parser; and an
unknown `type` or key is an **error listing what is available**, so a typed
`"unitss": 256` cannot train silently against a default.

The layer seam carries a per-sample `Shape`, batch axis excluded.
`Layer::output_shape()` in `include/SushiAI/nn/layer.hpp` is the virtual, and a
`LayerFactory` in `include/SushiAI/nn/layer_registry.hpp` receives the shape the
layer before it reports, so a convolution, a pool and a flatten chain through
`build_layer_stack` with no width restated in the file. `GraphBuilder::conv2d`
in `include/SushiAI/graph/builder.hpp` owns the convolution's node sequence, a
reshape of the rank-4 weight, K_IM2COL, K_MATMUL, an optional K_BIAS_ADD and a reshape
back to NHWC, and `NN::Conv2d` in `include/SushiAI/nn/conv2d.hpp` calls it and
does not restate it.

`Train::Objective` is the seam that breaks the classifier lock-in:
`build_classifier` hardcoded `NN::CrossEntropyLoss`, and it is now a wrapper
over `build_graph(module, objective, ...)`. `CrossEntropyObjective` emits the
same operations in the same order as the code it replaced, which is why the
whole existing suite is the regression test for that change.

### Checkpoints, split along the host/device lane

`io/checkpoint_format.hpp` is the `SACP` byte stream alone — a magic, a
version, and per tensor a name, a dtype, a `Shape` and a payload, every scalar
encoded a byte at a time so the little-endian order is stated rather than
inherited. No tensors, no engine, so its whole malformed-file surface is a
device-free unit test. `io/checkpoint.hpp` is the device half: the
`template<typename M>` save/load that walks `for_each_parameter` and binds
records to `TensorPool` tensors.

Three properties are structural rather than checked beside the code:

- **The reader allocates nothing from a file-declared size.** `next()` returns
  a header, the caller compares it against a destination it already owns, and
  `read_payload()` copies into that destination.
- **Filled slots, not a record count.** A file with one duplicated name and one
  missing name has the right count, and the unfilled parameter would then read
  uninitialised device memory — `allocate_tensor` does not zero, so a forward
  pass over it produces numbers rather than crashing.
- **A `Shape`, not an element count.** `[256, 6]` cannot load into `[6, 256]`.

`LayerStack` names its parameters exactly as `Sequential` does, gaps included,
because `LayerAdapter` delegates to the same `for_each_parameter` walk. That is
what lets a checkpoint written by either load into the other, and
`Integration_Checkpoint.ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack`
proves the two paths describe one model **by the checkpoint itself**, rather
than through a second mechanism whose own correctness would then be the open
question.

---

## 6. Packaging

SushiAI is `find_package`-able from its very first commit, mirroring
SushiBLAS's SB-6 packaging shape exactly: `SushiAI::sushiai` (a `STATIC`
library, aliased in-tree and exported out-of-tree under the same name),
`install(EXPORT)` / `export(EXPORT)`, a generated `SushiAIConfig.cmake`
(`find_dependency(SushiRuntime)` and `find_dependency(SushiBLAS)`, each
guarded by a TARGET-guard so a consumer that already resolved either
dependency itself doesn't get a duplicate), `SameMinorVersion` package
version compatibility (pre-1.0), and a version-drift check between
`include/SushiAI/version.hpp` and the CMake project version at configure
time.

`sushiai_apply_build_policy()` (`cmake/SushiAIBuildPolicy.cmake`) composes
warnings/architecture/sanitizer policy targets and pulls their flag *values*
onto the real target via `target_compile_options`/
`target_compile_definitions`/`target_link_options` +
`$<TARGET_PROPERTY:...>`, never `target_link_libraries` — the STATIC-library
export-set-safe pattern SushiBLAS's own SB-6 work established (see
SushiBLAS's `ARCHITECTURE.md §10`): linking a PRIVATE policy target, even
`BUILD_INTERFACE`-wrapped, records its name in the exported
`INTERFACE_LINK_LIBRARIES` and breaks `install(EXPORT)`/`export(EXPORT)`.

`tests/package/` is a real out-of-tree consumer — `find_package(SushiAI)`
against an install prefix, no access to this source tree — that runs a real
op through both transitively-resolved siblings.

It is built at **C++17**, which is the one place this repository deliberately
stops mirroring sushiblas's `tests/package/`. SushiBLAS keeps its consumer at
C++20 as a live assertion that a C++20 consumer still links against a C++17
library, and that assertion holds there because all of its span-taking entry
points are header-inline. SushiAI's are not (§1), so the same assertion here
would be false — and would pass only because `main.cpp` happens to call
nothing that takes a span. The file records the two mangled symbols that
prove it.

`SA_BUILD_TESTS` (default `ON`) guards `enable_testing()` and
`add_subdirectory(tests)` together, matching SushiRuntime's `SR_BUILD_TESTS`
shape exactly. It exists for the other consumption mode: a superproject that
`add_subdirectory`s SushiAI would otherwise be made to compile the suite,
have `gtest_discover_tests` enumerate it, and register 376 tests it never
asked for into its own CTest run. The default is `ON` rather than
SushiRuntime's `OFF` because `sa test` expects a test binary to exist and a
direct `cmake` build that silently produces none is the wart SushiRuntime's
own docs record against that choice.

CMake module files that define SushiAI-specific functions
(`SushiAIBuildPolicy.cmake`, `SushiAICompilerWarnings.cmake`,
`SushiAIOptimization.cmake`, `SushiAISanitizers.cmake`, `SushiAISyclTarget.cmake`)
are prefixed with the program's full name, because D3's sibling
`add_subdirectory` puts every checked-out repo's `cmake/` directory on one
shared `CMAKE_MODULE_PATH` — an unprefixed `SyclTarget.cmake` here would
shadow (or be shadowed by) SushiBLAS's own file the moment both are
`add_subdirectory`'d together, exactly the collision SushiBLAS's own SB-6
work found and fixed. `Vcpkg.cmake` and `Toolchain.cmake` stay unprefixed
because their logic is idempotent regardless of which sibling's copy loads
first.

---

## 7. What's not here yet

Open work is listed in one place,
[`REMAINING_WORK.md`](../design/REMAINING_WORK.md): packing the lowering's scratch, the
log-softmax and `K_NLL_LOSS` fusions, the two batch copies, the standalone `K_LOG_SOFTMAX` rule, the
whole-step overflow skip, bfloat16, the `Dataset` and `Trainer` seams, and the demo's reference
loss curve and timing table. What behaves wrongly or narrowly today, including why `gradcheck`
does not verify a mixed-precision graph, is in
[`KNOWN_ISSUES.md`](../reference/KNOWN_ISSUES.md).

The retired 2025-06 prototype (`core/tensor.h`, eager autograd, CUDA-only)
that used to occupy this repository is **not** a migration source for any of
this — it predates the SushiStack integration entirely and was deleted, not
ported, at AI-0. See the master plan for the actual sequencing.

## Project layout

See the top-level [README.md § Project layout](../../README.md#project-layout)
for the full directory tree.
