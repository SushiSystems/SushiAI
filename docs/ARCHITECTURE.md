# SushiAI Architecture

This document describes how SushiAI is built today, and names only what
actually exists under `include/SushiAI/`, `src/`, and `cmake/`. **AI-1** has
landed: the graph IR, the differentiation transform, the lowering to
SushiBLAS, the layer/optimizer/data/training layers, and an MLP that trains
end to end. It will keep growing section by section — see
[`docs/slop/make_sushiai_great_again.md`](slop/make_sushiai_great_again.md)
for what's coming and in what order.

> If you only want to *use* the library, the top-level [README.md](../README.md)
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
symbols side by side. See [INTEGRATION.md §3](INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply).

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
  `graph/ir.hpp`, `graph/builder.hpp`, `graph/memory_plan.hpp`, `autograd/`,
  `nn/module.hpp`, `nn/layers.hpp`, `nn/loss.hpp` — reaches no SushiBLAS
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

`autograd/backward.hpp` implements the reverse walk. Its rule registry is a
file-scope `constexpr` array of `{OpID, function pointer}`, deliberately not
a self-registering map: `sushiai` is a **static** library, so a translation
unit whose only job is to run a registration constructor resolves no symbol
and is dropped by the linker — silently, and only in a downstream consumer's
build.

Gradient accumulation at a fan-out is tiered: zero contributors means no
gradient value at all (a symbolic zero, not a materialised buffer), one
means the gradient *is* that value, and two or more emit `ACCUMULATE` nodes
in contributor order.

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
checked directly in `tests/functional/integration/test_arena.cpp`. One hole
remains on the SushiBLAS side — `Dispatch::execute_reduction` still builds
its regions from raw `data()` pointers, so a reduction operand at a non-zero
offset gets *no* dependency edge at all. `Lowering::values_needing_own
_allocation()` keeps those few values out of the arena and names the fix that
retires the workaround.

`src/graph/lowering.cpp` is the only file that calls `SushiBLAS::Engine`. Its
`TensorPool` asks the plan where each value goes: packed values become arena
views, everything else gets its own `Storage`. The planner assumes **every
node writes all of its outputs on every replay**, which is why `ZERO` records
its fill into the step rather than into the one-time constant queue.

`graph/compiled_step.hpp` wraps the compile-once/replay contract in a type,
because the ordering has a silent failure mode: `rearm()` before
`capture_base_state()` leaves the completion sentinel armed at in-degree
zero, so the run reports complete before any kernel has been dispatched and
the step quietly does nothing. Doing both in the constructor makes that
state unrepresentable.

It also takes an optional `IStepRecorder` (`graph/step_recorder.hpp`) — work
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
no number either. `UpdateKernel::SEPARATE` keeps the reference spelling
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

Adding a form is four mechanical edits: an identity, a `FusedForm`, a `CATALOG`
row, and a recorder. Nothing self-registers, for the same reason
`autograd/backward.cpp`'s rule table does not.

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
property of the op rather than a case inside the pass. `Ops::ACTIVATIONS` states
where each activation's gradient reads, once, and both the differentiation pass
and the catalog's `static_assert` consult it:

| activation | gradient reads | foldable |
|---|---|---|
| ReLU | its output | yes |
| sigmoid | its output | yes |
| tanh | its output | yes |
| GELU | its **input** | no |

ReLU moved from reading its input to reading its output as part of this work,
and that is a substitution rather than an approximation: SushiBLAS's
`relu_backward` is `dy * (primal > 0 ? 1 : 0)` and `relu(x) > 0` exactly when
`x > 0`, so the two spellings are **bit-identical**. Making it unconditional -
not a rewrite the fusion pass applies - is what leaves the pre-activation with
a single consumer in the plain graph too, so an ordinary use count says yes for
ReLU and no for GELU with no activation-specific code in the matcher at all.
The three entries in `docs/slop`'s §25 note about GELU/SiLU/sigmoid/tanh being
non-recoverable is narrower than it looked: sigmoid and tanh **are** exactly
recoverable from their outputs (`y(1-y)`, `1-y²`) and SushiBLAS's rules already
take the output, so only GELU is genuinely blocked here.

An auxiliary-buffer GELU epilogue - cuBLASLt's `GELU_AUX_BIAS` shape - is
**not implementable through today's seam** and is declined rather than
approximated. `Dispatch::execute_gemm` declares A, B and C to the dependency
tracker and nothing the epilogue touches, so an aux buffer the functor wrote
would get no edge at all and its reader would race the write. That is a
SushiBLAS-side change (an epilogue read/write set on the seam), not something
this side can work around.

### The cost the epilogue carries

The same gap bites the bias. A fused GEMM's epilogue dereferences a bias vector
the tracker cannot see, and that bias is a parameter the optimizer rewrites
*inside the same compiled plan* - so an undeclared read is a write-after-read
hazard nothing orders. A transitive path through the gradient happens to exist
for a plain feed-forward model, and relying on it is exactly the accident that
survives every test and breaks on the first second writer. So
`Lowering::declare_epilogue_operand()` emits a `TaskGraph::add_native` node that
runs nothing and exists only for its two access declarations: it reads the bias
and writes the result, which pushes the optimizer's update out behind it and
pulls the GEMM in front of it. One node per fused GEMM, no kernel launch.
`RunReport::ordering_nodes_per_step` counts them separately from
`operations_per_step` so a fusion change cannot hide behind them.

### What it is worth, measured — and what it costs

On `sa demo mlp`'s model a step goes from **35 recorded SushiBLAS calls to 25**
plus 2 host-side ordering nodes, all of it out of the forward and backward
graph: the first layer's `gemm+bias+relu` replaces five calls with one, the
second layer's `gemm+bias` replaces three with one, and the cross-entropy
gradient's `exp/sub/scale` tail replaces five with one. Every fused form is
compared byte for byte against the unfused spelling in
`tests/functional/regression/test_fusion_numerics.cpp`, and
`FusionOptions::enabled = false` keeps that reference reachable, exactly as
`Optim::UpdateKernel::SEPARATE` does for the optimizer. The demo's loss
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
have `gtest_discover_tests` enumerate it, and register 153 tests it never
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

Named honestly, because the gap between what a library claims and what it
does is the expensive kind of documentation debt:

- **The memory planner packs IR values only.** The lowering's own temporaries
  — `TensorPool::scratch()` and the shared `ones()` constants — are not IR
  values, so they still get one `Storage` each. Packing them needs a
  host-lane description of each operation's scratch requirements, shared
  between the planner and the lowering; today those shapes are written inline
  in `record_log_softmax()` and `record_nll_loss()`. On the demo they are
  about 2 KiB of 130 KiB.
- **The fusion catalog is five entries, and the biggest node in the graph is
  not one of them.** `LOG_SOFTMAX` still lowers to nine calls, which is now the
  single largest term in a step, and collapsing it needs a row-wise reduction
  SushiBLAS does not expose rather than a catalog row. `NLL_LOSS`'s three calls
  are blocked the same way: the reduction seam takes no transform functor, so
  its elementwise `mul` cannot ride into the sum. Neither is a fusion-pass
  problem; both are one SushiBLAS operation away.
- **Feeding a batch still costs a flush of its own.** The optimizer no longer
  does (§5.1), but the two `blas().copy()` calls that move a batch into the
  plan's fixed input buffers do: they read a *different source tensor* every
  step, and a recorded SushiBLAS call captures its operand addresses by value.
  Fusing cannot help — the thing that changes is a pointer, not a scalar — and
  neither can SushiRuntime's `LateBindings`, which re-points a node's iteration
  count, base index and enabled flag but not the addresses its kernel captured.
  `TaskGraph::add_offload_dynamic()` *does* carry re-pointable live read/write
  lists, but only for offloadable tasks with a host fallback and no device
  kernel, so it does not reach a `copy`. What would retire this is either an
  operand-address late binding that ordinary device tasks can use, or a
  `Dataset` that writes each batch straight into the two buffers the plan
  already owns.
- **`LOG_SOFTMAX` is differentiable only as part of the fused
  `LOG_SOFTMAX` + `NLL_LOSS` pair.** The standalone rule needs a row-wise
  reduction SushiBLAS does not expose; it throws and says so.
- **No mixed precision, no checkpointing, no serialization.**
- **AI-7's metric gates are only partly met.** `sa demo mlp` exists and
  reports `compiles: 1`, a zero-allocation steady state and finite-difference
  checks are green in the suite, but there is no pinned reference loss curve
  to compare a run against, and the per-node timing table needs a
  `RuntimeSettings::profiling` context the demo does not build.

The retired 2025-06 prototype (`core/tensor.h`, eager autograd, CUDA-only)
that used to occupy this repository is **not** a migration source for any of
this — it predates the SushiStack integration entirely and was deleted, not
ported, at AI-0. See the master plan for the actual sequencing.

## Project layout

See the top-level [README.md § Project layout](../README.md#project-layout)
for the full directory tree.
