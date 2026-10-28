# SushiAI Architecture

This document describes how SushiAI is built today, and names only what
actually exists under `modules/`, `applications/`, and `cmake/`. **AI-1** has
landed: the graph IR, the differentiation transform, the lowering to
SushiBLAS, the layer/optimizer/data/training layers, and an MLP that trains
end to end. It will keep growing section by section: see
[`PROGRAMME.md`](../design/PROGRAMME/README.md)
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
queue or writes a `parallel_for`. Every compute call is recorded through a
SushiBLAS operation view (`BLASOps`, `ElementwiseOps`, `RandomOps`, `SpatialOps`
and their siblings) built from a `SushiBLAS::TaskRecorder`. The lowering is the
one place that turns *IR* into those calls: the driver in
`modules/core/graph/source/lowering.cpp` and the eight domain files under
`modules/core/graph/source/lowering/`
(§5.0). That keeps the differentiation, fusion and memory-planning passes
device-free and testable without hardware.

Two places outside the lowering record SushiBLAS work, because neither has an
IR node to lower: `modules/layers/nn/source/init.cpp` and `data/` build their views from the engine,
which binds them to the engine's own recorder and RNG, and `optim/` records the
parameter update on the recorder of the compiled step. `optim/` additionally
hands SushiBLAS a **device functor**:
`Dispatch::execute_zip_inplace`'s generic elementwise seam inlines it into a
kernel SushiBLAS owns. That is plain C++ arithmetic on `auto&` parameters, not
a kernel this repository defines, but it is device code and §5.1 says what
holds it correct.

The pipeline for one training step is:

```
trace a model  ->  forward IR
differentiate  ->  forward + backward IR      (IR -> IR, no device work)
fuse           ->  fewer, bigger nodes        (IR -> IR, no device work)
lower          ->  a SushiRuntime Graph       (the only SushiBLAS calls)
compile        ->  ONCE, per model
replay         ->  per batch, for ever
```

Two properties are load-bearing and both are asserted by the test suite:
the step is compiled exactly once no matter how many batches run
(`CompiledStep::compile_count()`), and the steady state allocates nothing
(`TensorPool::allocated_bytes()` stops moving after the first step). The
first of those is additionally pinned in the `regression` tier
(`tests/layers/train/test_trainer/test_compile_once.cpp`) and in the demo's own
CTest smoke test, because a recompile-per-step still converges: it just
quietly pays graph analysis every step, and nothing else would notice.

Alongside the core there is the scaffolding a real library needs from day
one: a `find_package(SushiAI)`-able target verified by an out-of-tree
consumer, a test suite labelled by size (`small` / `medium` /
`large`), and the `sa` CLI. Each module builds one test program,
`sushiai_<tier>_<module>_tests`: `sa run` runs one in one process and `sa test`
runs every case through CTest. `sa demo mlp` runs the showcase:
`applications/examples/source/demo_mlp.cpp`, a flag-driven training program, deliberately a
separate binary from the `applications/examples/source/example.cpp` teaching file.

### The whole stack is C++17, and that is not a style choice

SushiAI is built at **C++17**: the standard SushiRuntime and SushiBLAS are
built to. The three repositories have to agree on one standard, and the reason
is an ABI, not a preference.

`SushiRuntime::span` (`SushiRuntime/core/span.hpp`) *is* `std::span` when
`__cplusplus >= 202002L` and SushiRuntime's own C++17 shim otherwise. That
branch is taken per translation unit and it reaches the mangled name of every
function that takes one. SushiAI compiled at C++20 over siblings compiled at
C++17 put the two halves of one build on opposite sides of it: the same
mismatch that moved SushiBLAS down first
([its ARCHITECTURE §1](https://github.com/SushiSystems/SushiBLAS/blob/main/docs/ARCHITECTURE.md#1-what-sushiblas-is)).
SushiAI was the last repository still above the line, so the stack now agrees.

The library's own API did not change: the same functions take the same
arguments spelled the same way, and a *header-only* use compiles under either
standard. What did change is the linkage of the out-of-line entry points whose
parameter type is a span: `Autograd::differentiate`,
`allocate_tensor`, `Optim::StepScalars::write`, the `Autograd`
gradcheck entry points. Those now export the shim's mangling, so a consumer
must be at C++17 too; `tests/package/` says so in a comment, with the two
symbols side by side. See [INTEGRATION.md §3](../guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply).

The pieces C++20 had been supplying are C++17 spellings now:

| was | is | where |
|---|---|---|
| `std::span` | `SushiRuntime::span` | 105 sites across 31 files: the same type wherever `std::span` exists |
| `std::format` | `SushiLog::format_text` | `SA_FAIL_IF`, `SA_FATAL_IF`, `SA_THROW_IF` and every report/diagnostic string |
| `concept Module` / `ModuleContainer` | `constexpr bool` variable templates over `std::void_t` detectors | `nn/module.hpp` |
| `template<Module... Ls>` | `template<typename... Ls>` + a `static_assert` fold | `nn/layers.hpp`'s `Sequential` |
| `consteval` | `constexpr` (called only from `static_assert`, so still proven at compile time) | `ops/ops.hpp`, `modules/core/graph/source/fusion.cpp` |

`SushiLog::format_text` is **shared, not copied**: the stack keeps one general
formatter, in SushiLog's `text` module, and SushiAI links `SushiLog::text` from
the module `failure`, whose macros build every message with it. Each `{}` takes
the next argument through `operator<<`, and there are no format specs: a value
that needs a width, a precision or hex goes through one of SushiLog's value
helpers first. `std::format`'s `consteval` check on a literal format string is
lost, so a placeholder/argument mismatch is a worse message rather than a
compile error.

---

## 2. The sibling dependency chain (D3)

SushiAI depends on two siblings, SushiRuntime and SushiBLAS, and on SushiLog and SushiProfiler.
Each is a built package: `cmake/Runtime.cmake`, `cmake/BLAS.cmake`, `cmake/Log.cmake` and
`cmake/Profiler.cmake` each call `find_package(<Name> CONFIG REQUIRED)` and nothing else. No
library of another repository is compiled in this tree.

`sa build` resolves the four packages before it configures. A package missing from the store is
built once through the library's own CLI, and the folder of each is handed to CMake as
`-D<Package>_DIR`. One SushiRuntime package serves SushiBLAS and SushiAI both, so the build
holds one runtime and not two competing copies. The SYCL toolchain of the build comes from
`SushiRuntime_SYCL_TOOLCHAIN`, which the installed SushiRuntime package sets.

Every consumer downstream (SushiAI's own targets, the CLI's `-D` flags, an
out-of-tree `find_package(SushiAI)` consumer) links the same namespaced targets.

The two siblings have separate roles. SushiRuntime holds the graphs and the
memory and decides where each node runs; it computes nothing. SushiBLAS
computes every operation and records it into the recorder it is handed. SushiAI
owns the graph of a step: `CompiledStep` records on a `SushiRuntime::Graph` of
its own through a `SushiBLAS::TaskRecorder` built over that graph's recorder,
and every other recording of the library does the same: `record_and_run()` of
`execution/execution.hpp` builds a graph, records on it and runs it. SushiAI
records nothing on the graph SushiBLAS's `Engine` keeps.

---

## 3. The library target

Each module is declared by `sushiai_add_module()`, called from its own `CMakeLists.txt`.
`sushiai_add_library()` archives the modules of a tier into one library. Both functions are in
`cmake/SushiAIModule.cmake`, and `cmake/SushiAILayers.cmake` holds the tier table: the tier of each
module, and the library each tier is archived into. The `core` and `layers` tiers form the
`sushiai` library, and the `vision` domain forms `sushiai_vision`, which links `sushiai`.

```cmake
sushiai_add_module(NAME graph TIER core
    SOURCES source/builder.cpp source/fusion.cpp ...
    PUBLIC_DEPENDS allocation dtype failure ops seam shape
    TESTS tests/core/graph/test_ir.cpp ...)

sushiai_add_library(NAME sushiai EXPORT_NAME sushiai
    UMBRELLA "${CMAKE_CURRENT_SOURCE_DIR}/layers/umbrella/include/SushiAI/SushiAI.hpp")
```

A module's or test program's `DEVICE_SOURCES` name the `SOURCES` that instantiate a device
kernel, today the two elementwise lowerings of `graph`, the three optimizer steps of `optim` and
one test of `eager`. On the intel-llvm path only those compile with `-fsycl` and the options of
`SushiBLAS::device_code`; every other unit compiles for the host with the SYCL headers that
`SushiBLAS::sushiblas` publishes. A unit that submits a kernel and is left off the list fails at
compile time or at run time, so `tools/layering/check_device_sources.py` checks the list.
`sushiai_add_library()` links each library `PUBLIC` to the external targets its modules name in
`EXTERNAL_PUBLIC`, such as `SushiRuntime::core` and `SushiBLAS::sushiblas`, and the vision library
adds `DEPENDS sushiai`.

A module only the tests use passes `INTERNAL` to `sushiai_add_module()`. It joins no library and
installs nothing. `tools/layering/check_layering.py` compares every include of a module with
the `PUBLIC_DEPENDS` and `PRIVATE_DEPENDS` its `CMakeLists.txt` lists.

Both dependencies are linked `PUBLIC`: the lowering needs SushiBLAS, and
every consumer needs it to construct an `Engine` anyway. SushiAI therefore
inherits SushiRuntime's `sushiruntime_deterministic_fp` floating-point
contract transitively rather than owning a copy of that guarantee, which is
also why gradient accumulation is ordered through the graph's edges rather
than through atomics (see `autograd/backward.hpp`).

`modules/layers/umbrella/include/SushiAI/SushiAI.hpp` is the master header and includes every
public header of the `sushiai` library. Two lanes live under it. Each header name below is
relative to `SushiAI/` under the `include/` of the module that owns it:

- The **host lane** (`core/` (except `allocation/allocation.hpp` and
  `seam/seam.hpp`), `graph/ir.hpp`, `graph/builder.hpp`, `graph/memory_plan.hpp`,
  `graph/fusion.hpp`, `graph/precision.hpp`, `graph/cast_insertion.hpp`,
  `autograd/`, `nn/module.hpp`, `nn/layers.hpp`, `nn/loss.hpp`) reaches no SushiBLAS
  *engine* and no SYCL. The `Small_*` tests include these directly and run
  without a device. What it does reach of its siblings is vocabulary only, and
  every one of those headers is `<cstddef>`-and-friends deep:
  `<SushiBLAS/core/common.hpp>` for the dtype/layout enums,
  `<SushiLog/text/text_format.hpp>` for the message formatter, and
  `<SushiRuntime/core/span.hpp>` and `<SushiRuntime/core/operation_id.hpp>` for
  the span and the `OpID`.
- The **device lane** (`allocation/allocation.hpp`, `seam/seam.hpp`,
  `graph/arena.hpp`, `graph/tensor_pool.hpp`, `graph/step_recorder.hpp`,
  `graph/lowering.hpp`, `graph/lowering/`, `graph/compiled_step.hpp`,
  `nn/init.hpp`, `optim/`, `data/`, `train/`, `io/checkpoint.hpp`) reaches
  SushiBLAS.

The SYCL policy lives in `cmake/SushiAISyclTarget.cmake`. The device link
options come with `SushiBLAS::sushiblas`, so an example or a test that links a
SushiAI library pulls the device images out at its link without compiling its
own units for the device.

### 3.1 How a failure leaves the library

A content error, such as a file that does not parse, a shape that does not
fit or an allocation the device refuses, is a `Status` (`failure/status.hpp`) or
a `Result<T>` (`failure/result.hpp`). `SA_FAIL_IF` builds one with a `StatusCode`,
`SA_TRY` and `SA_TRY_ASSIGN` pass one up, and `SA_ANNOTATE` prefixes the
caller's context. A programmer's misuse is `SA_FATAL_IF` (`failure/check.hpp`),
which reports and aborts the process.

SushiBLAS and SushiRuntime report by throwing. `seam/seam.hpp` is the one place
such an exception becomes a status: `Detail::guard` and `Detail::guard_value`
run a call and return what it threw as a `Status`. `execution/execution.hpp`
holds the runtime calls the library makes, `create_runtime`, `create_graph`,
`run_step`, `run_once` and `record_and_run`, each mapping its failure through the seam. The `SeamPhase` argument (`Record`, `Allocate`, `Execute`)
decides the code of a failure that carries no type of its own.

A type whose construction can fail has a private constructor and a factory
that returns a `Result`: `CompiledStep::compile`, `Trainer::make`,
`AdamW::make`, `SGD::make`, `LossScaler::make`, `StepScalars::make`,
`ArenaSet::make`, `Rings::make`, `SyntheticClassification::make`,
`CheckpointWriter::create` and `CheckpointReader::open`. `NN::initialize`,
`save_checkpoint` and `load_checkpoint` return a `Status`.

```cpp
SA_TRY_ASSIGN(std::unique_ptr<Graph::CompiledStep> step,
              Graph::CompiledStep::compile(engine, graph, pool, &optimizer));
SA_TRY(step->run());
```
Source: not compiled

`SA_THROW_IF` (`failure/error.hpp`) still throws `SushiAI::Error` from the host
lane: `shape/shape.hpp`, `graph/ir.hpp` and `config/registry.hpp` use it, and
`docs/reference/KNOWN_ISSUES.md` holds the count. `NN::Detail::bridge_throwing` (`nn/throwing_bridge.hpp`)
turns that exception into `InvalidDescription` where a layer is built from a
file.

---

## 4. The IR, and why differentiation writes IR

`modules/core/graph/include/SushiAI/graph/ir.hpp` defines a single-assignment DAG of
`Value`s and `Node`s. `graph/builder.hpp` is the only thing that creates
nodes, so every operation's shape inference is stated exactly once.

A backward rule reads that IR and writes more of it. It never calls
SushiBLAS and never touches a `SushiRuntime::Graph`. The full argument is in
`ir.hpp`'s file comment; the short version is that a `SushiRuntime::Graph` node
carries byte ranges that cannot be known until the memory planner has seen
the whole forward *and* backward graph, that fusion needs a DAG to
partition, and that `SushiRuntime::Graph::compile()` drains its builder and so
cannot be held and rewritten.

`autograd/backward.hpp` declares the reverse walk: `differentiate()` and
`has_backward_rule()`, and nothing else.

Gradient accumulation at a fan-out is tiered: zero contributors means no
gradient value at all (a symbolic zero, not a materialised buffer), one
means the gradient *is* that value, and two or more emit `K_ACCUMULATE` nodes
in contributor order. `K_ACCUMULATE` writes into the buffer of its first
operand, so when the first contribution is also read elsewhere, as the one
gradient `K_ADD` hands both operands is, the first two contributions are
summed by a `K_ADD` into a new buffer and the chain accumulates into that.

### 4.1 One translation unit per domain

There is no `backward.cpp`. The implementation is three kinds of file:

| File | What it owns |
|---|---|
| `modules/core/autograd/source/differentiator.cpp` | `differentiate()`, the reverse traversal, and the seed |
| `modules/core/autograd/source/tape.cpp` | `GradientTape`: how a rule reads its node and contributes a gradient |
| `modules/core/autograd/source/rules/<domain>.cpp` | The rules themselves, one file per domain |

The domains are `elementwise`, `activation`, `linalg`, `loss`, `shape`,
`spatial` and `norm`, each with a header beside it in
`modules/core/autograd/include/SushiAI/autograd/rules/`. `rules/all.hpp` is the one place that joins
them.

The property this buys is a rule about *diffs*, and it is the reason the split
is worth having at all: **adding an operation to an existing domain changes
only that domain's `.cpp` and `.hpp`. Adding a whole domain is one line in
`all.hpp`.** Before the split, every new gradient (a pooling rule, a
normalisation rule, a loss rule) edited the same 600-line file, which is how
four unrelated reasons to change end up sharing a merge conflict.

A rule sees `GradientTape` and nothing else. `Differentiator`'s definition
lives in `modules/core/autograd/source/differentiator.hpp`, outside the public include tree,
reachable from exactly the two translation units that construct or call back
into it.

**Nothing self-registers**, and that is not a stylistic preference. `sushiai`
is a **static** library. A translation unit whose only job is to run a
registration constructor resolves no symbol, so the linker drops it: the rule
table comes up short, `differentiate()` reports no rule for an operation that
plainly has one, and it happens *only in a downstream consumer's build*, not in
this repository's own tests. So each domain's rules reach the table by being
named in a file-scope `constexpr` array of `{OpID, function pointer}` that
`all.hpp` concatenates. Missing one is a compile error in this repository, not
a link-time absence in somebody else's. `ops/ops.hpp`'s identity table,
`modules/core/graph/source/fusion.cpp`'s `K_CATALOG` and the lowering tables of §5.0 are the same
shape for the same reason.

### 4.2 Three invariants the IR proves rather than assumes

**`TensorLayout`, proved by `Graph::prove_layout_uniformity()` (CV-4).** Every
`Value` carries a `TensorLayout`, and `K_LAYOUT_CONVERT` is the only node allowed
to have operands and results that disagree about it. `finalise()` closes the
forward graph (it propagates `requires_grad` and then proves uniformity), so a
builder-emitted graph that mixes NCHW into an NHWC chain fails there with
`InvalidDescription`, naming the node, instead of reaching a kernel that reads
the right number of elements in the wrong order. No emitter declares a result
in another layout than its operands, so a node that does was built by hand, and
the proof stops the process on it.

That is a claim about the graph the builder emits, and it is worth being exact
about how far it carries. `finalise()` is called once, before differentiation,
and the stages after it rebuild the graph: `fuse()` and `insert_casts()` both
copy the value table entry by entry, and `Graph::add_value` defaults the layout
to NHWC, so a field one of them forgets to copy is not lost but *changed*:
silently, because the shapes still agree. Both passes therefore re-prove
uniformity over the graph they produce, which is why the proof is public rather
than a private step of `finalise()`. `insert_casts()` proves its input first and
returns the failure, so a graph that reaches it unproven is refused unchanged;
`fuse()` takes a proven graph as a precondition and stops the process on one
that is not. A rebuilt graph that fails the proof is a defect in the pass, and
both stop the process on it. `Lowering::record()` proves the graph it records
and returns the failure. Backward-emitted nodes are still only
covered by whichever pass runs after them; a graph that is differentiated and
then handed straight to lowering is not re-proved anywhere.

`GraphBuilder::declare_like()` is how an emitter keeps the invariant true
without restating it: a result takes its operand's layout by construction.
`declare()` is left for the values that have no operand to inherit from: the
leaves (`input()`, `parameter()`), `layout_convert()`, which sets the tag
itself, being the one node allowed to change it, and `zeros()`, whose node has
no inputs and is therefore skipped by the per-node proof. A backward rule
emitting zeros beside a real operand calls `zeros_like()` instead, since the
exposure there is the *consumer*'s node, not `K_ZERO`'s.

`Lowering::record()` proves it once more before recording anything. That is the
one gate every path passes through, so the pass-level proofs are
belt-and-braces rather than load-bearing, and it is where the tag stops being
bookkeeping, since `lower_layout_convert()` (`modules/core/graph/source/lowering/shape.cpp`)
reads it to decide whether to relabel a side into `[N,H,W,C]`.

**The `differentiable` flag is split from its proof.** `ops/ops.hpp` carries
the flag on each `OpName`: the *claim* that an operation has a gradient.
`autograd/rules/all.hpp` carries the `static_assert` that every operation
making that claim is in the rule table: the *proof*. They are apart because
they must be: the assert needs to see the rule table, and `core/` cannot depend
on `autograd/` without inverting the layering the whole tree is built on. The
flag stays where the operation is declared; the check lives where the evidence
is. `K_BATCHNORM_UPDATE_STATS` claims nothing and has no rule, because CV-7 keeps
a BatchNorm's running statistics out of the differentiated graph entirely: it
is recorded into the training step beside the graph, the way `K_SGD_UPDATE` and
`K_ADAMW_UPDATE` are. Those three identities are the whole of `K_UNLOWERED`
(`graph/lowering/unlowered.hpp`), the list of operations with no lowering row.
A node that carried one would stop the process in `Lowering::record` with "No
lowering for operation".

**`graph/rematerialise.hpp` is deliberately not general (CV-10).**
`rematerialise_im2col()` recomputes exactly one operation: it clones the
`K_IM2COL` node in front of the backward consumer that reads its column matrix,
demotes the stored value to `Lifetime::Scratch`, and stops. It is not a cost model, not a
checkpointing framework, and not parameterised over which operations are cheap
to recompute. A convolution's column matrix is `kernel_h*kernel_w` times its
input and is read once in the backward pass, which makes it both the largest
activation in a CNN and the easiest thing in the graph to rebuild; a general
pass would need a cost model nobody has measured, to serve cases nobody has.
The clone copies the original's `Attributes` rather than rederiving them, for
the reason the next paragraph gives.

**`Attributes::K_CAPACITY` is 10, and the last two slots are not redundant.**
Slots 0–7 are a 2-D window's geometry: `kernel_h`, `kernel_w`, `stride_h`,
`stride_w`, `dilation_h`, `dilation_w`, `pad_top`, `pad_left`, mirroring
`Graph::ConvGeometry` field for field. Slots 8 and 9 are the output extents
`out_h` and `out_w`, **stored rather than derived**. That looks like
duplication and is not: CV-5 gives the geometry no `pad_bottom` or `pad_right`,
so deriving the extents back from it silently assumes symmetric padding, which
YOLOv3-tiny's 13→13 pool violates. A wrong derivation there yields a window
grid that is the *transpose* of the real one, with the same element count, so
it passes every shape check in the tree and gathers the wrong pixels. Storing
what the host already computed is the fix, and A-8 ("never re-derive an output
extent from geometry") is the rule it generalises to.

`K_CAPACITY` is bounded by nothing outside SushiAI, and deliberately so.
A lowering row reads slots with `Attributes::get()` and hands the values
straight to a SushiBLAS call, so no attribute is ever copied into SushiRuntime's fixed-width
`TaskMetadata::params` array and `K_MAX_PARAMS` does not constrain it. The
capacity is what the operations need, and nothing else.

---

## 5. The compiled step, fusion, models and packaging

These sections have pages of their own:

- [`LOWERING.md`](LOWERING.md): §5 memory planning and lowering, §5.0 the lowering table, §5.0.1 the compiled step, §5.1 the optimizer inside it.
- [`FUSION_AND_PRECISION.md`](FUSION_AND_PRECISION.md): §5.2 fusion over a catalog, §5.3 mixed precision.
- [`MODELS_AND_PACKAGING.md`](MODELS_AND_PACKAGING.md): §5.4 the model file and its driver, §6 packaging.

---

## 7. What's not here yet

Open work is listed in one place,
[`REMAINING_WORK.md`](../design/REMAINING_WORK.md): the scratch rules of the rows that have none, the
log-softmax and `K_NLL_LOSS` fusions, the two batch copies, the standalone `K_LOG_SOFTMAX` rule, the
whole-step overflow skip, bfloat16, the `Dataset` and `Trainer` seams, and the demo's reference
loss curve and timing table. What behaves wrongly or narrowly today, including why `gradcheck`
does not verify a mixed-precision graph, is in
[`KNOWN_ISSUES.md`](../reference/KNOWN_ISSUES.md).

The retired 2025-06 prototype (`core/tensor.h`, eager autograd, CUDA-only)
is **not** a migration source for any of
this: it predates the SushiStack integration entirely and was deleted, not
ported, at AI-0. See the master plan for the actual sequencing.

## Project layout


```
CMakeLists.txt          Project, options and add_subdirectory; no logic
cmake/
  Vcpkg.cmake            (Windows) locate vcpkg + set the toolchain file
  Toolchain.cmake        pick the default SYCL compiler before project()'s probe
  Runtime.cmake          D3 resolution -> SushiRuntime::SushiRuntime
  BLAS.cmake             D3 resolution -> SushiBLAS::sushiblas
  SushiAIModule.cmake    sushiai_add_module(), sushiai_add_library()
  SushiAILayers.cmake    the tier table: each module's tier and library
  SushiAISyclTarget.cmake        the SYCL policy
  SushiAIBuildPolicy.cmake       composes warnings/perf/sanitizers
  SushiAICompilerWarnings.cmake
  SushiAIOptimization.cmake
  SushiAISanitizers.cmake
  SushiAIConfig.cmake.in         find_package(SushiAI) package config
modules/                 One folder per module, grouped by tier
  core/                  The tier that knows no layer type and no domain
    core/                Vocabulary (common, ops, shape, allocation) and the failure contract
    version/             The SUSHIAI_VERSION_* macros
    graph/               The IR, its builder, the passes, TensorPool, the lowering to SushiBLAS
    autograd/            Differentiation and gradient checking
    config/              Config::Value, the registry lookup, the model and training schemas
    optim/               SGD and AdamW, their registry and the loss scaler
    data/                The Dataset seam, its registry and the materialised dataset
    eager/               The eager front end: scope tracing, partitioning and the session that runs a step
  layers/                The layers, and what trains, saves and loads a model of them
    nn/                  Parameters, the Module contract, layers, losses, init, the runtime path
    io/                  The SACP checkpoint: the byte format and the binding
    train/               The training loop, its run report and the objective
    umbrella/            SushiAI.hpp, the master header of the sushiai library
  vision/                The vision domain, library sushiai_vision; it reads core and layers only
    imaging/  detect/  blocks/  datasets/  registration/  umbrella/
  <tier>/<module>/       Each module has CMakeLists.txt, include/, source/ and tests/, and a
                         README.md where one is written
applications/
  programs/source/       train.cpp (sushiai train), inference.cpp (sushiai inference), registries
  examples/source/       example.cpp, a teaching file; demo_mlp.cpp, the program behind sa demo mlp
  examples/heavy_mlp/  mnist_cnn/  yolo26/    One folder per example: its README and its JSON
tests/                   Tests that cross module boundaries
  common/                Process-global RuntimeContext and per-test Engine support
  unit/  integration/  regression/    The three suite tiers
  package/               Out-of-tree find_package(SushiAI) consumer
  fixtures/              Model, evaluation and training fixtures; the YOLO26 reference tensors
cli/                     Python `sa` / `sushiai` CLI package (Typer + rich)
docs/
  README.md                  Documentation manual index
  CONTRIBUTING.md            Contribution guidelines and coding standards
  DOCUMENTATION_STYLE_GUIDE.md  Documentation style and placement rules
  getting_started/           Tutorials and onboarding guides
  architecture/              System architecture and compilation pipeline
  guides/                    CLI guide and downstream integration guide
  reference/                 API reference, glossary, known issues, changelog
  design/                    Active design documents, roadmaps, single backlog
  agent/                     One dated folder per piece of agent work
  archive/                   Finished material, frozen
```
