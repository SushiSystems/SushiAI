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

**SushiAI writes no SYCL and defines no kernels.** Every compute call in the
library goes through `SushiBLAS::Engine`, and exactly one file —
`src/graph/lowering.cpp` — makes those calls. Everything else deals in the
IR described below, which is what keeps the differentiation, fusion and
memory-planning passes device-free and testable without hardware.

The pipeline for one training step is:

```
trace a model  ->  forward IR
differentiate  ->  forward + backward IR      (IR -> IR, no device work)
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

- The **host lane** — `core/`, `graph/ir.hpp`, `graph/builder.hpp`,
  `autograd/`, `nn/module.hpp`, `nn/layers.hpp`, `nn/loss.hpp` — pulls in no
  SushiBLAS and no SYCL. The `Unit_*` tests include these directly and run
  without a device.
- The **device lane** — `graph/lowering.hpp`, `graph/compiled_step.hpp`,
  `nn/init.hpp`, `optim/`, `data/`, `train/` — reaches SushiBLAS.

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

## 5. Lowering, and one Storage per value

`src/graph/lowering.cpp` is the only file that calls `SushiBLAS::Engine`.
Its `TensorPool` gives every `ValueId` its own `SushiBLAS::Storage` rather
than sub-allocating one arena, which is a correctness requirement today
rather than a missed optimisation: almost every SushiBLAS kernel captures
`tensor.storage->data_ptr` and ignores `tensor.storage_offset`, so an offset
view handed to gemm, axpy or an elementwise op reads from element zero of
the arena and produces a silently wrong answer. Distinct base pointers also
keep SushiRuntime's dependency tracker sharded.

`graph/compiled_step.hpp` wraps the compile-once/replay contract in a type,
because the ordering has a silent failure mode: `rearm()` before
`capture_base_state()` leaves the completion sentinel armed at in-degree
zero, so the run reports complete before any kernel has been dispatched and
the step quietly does nothing. Doing both in the constructor makes that
state unrepresentable.

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

`SA_BUILD_TESTS` (default `ON`) guards `enable_testing()` and
`add_subdirectory(tests)` together, matching SushiRuntime's `SR_BUILD_TESTS`
shape exactly. It exists for the other consumption mode: a superproject that
`add_subdirectory`s SushiAI would otherwise be made to compile the suite,
have `gtest_discover_tests` enumerate it, and register 100 tests it never
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

- **No memory planner.** `Lifetime` is recorded on every value and nothing
  reads it yet; every value gets its own permanent buffer. The planner that
  packs the transient classes needs SushiBLAS to honour `storage_offset`
  first (see §5).
- **No fusion pass.** The IR is shaped for one (a DAG that survives
  differentiation), but nothing partitions it.
- **The optimizer update is not part of the compiled step.** AdamW's bias
  corrections are per-step scalars and SushiBLAS takes its alphas by value,
  so the update is recorded and flushed separately each step. Feeding a
  batch costs a second flush for the same reason: a compiled call captures
  raw pointers, so the batch is copied into fixed buffers rather than bound
  to them. Both collapse into the compiled plan once SushiRuntime's
  late-binding block reaches the BLAS layer.
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
