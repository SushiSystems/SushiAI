# SushiBLAS v0.1.0 migration
**Status:** Open — wave 1 of 9

SushiAI calls the SushiBLAS surface that v0.1.0 removed and does not compile against the new one.
This document specifies the migration: every call site, the units that replace
`src/graph/lowering.cpp`, the status conversion of the files the migration rewrites, and the order
of work. It was written by reading SushiAI at `7be22fa`, SushiBLAS at `a977291` and SushiRuntime
at `778db81`. Nothing here was compiled or run. Section 9 lists what was not verified.

An implementer works from one unit of section 3 and the sections it names. Paths without a prefix
are in this repository. `sushiblas:` and `sushiruntime:` paths are relative to those repositories.

## 1. What binds the design

### 1.1 Rulings

| Subject | Ruling | Where |
| --- | --- | --- |
| Pipeline | trace, differentiate, cast insertion, fuse, memory plan, lower, compile once, replay. The migration changes the last three stages only | `docs/architecture/ARCHITECTURE.md:38-47` |
| Error contract | A content error is a `Status`; a programmer's misuse is fatal; no exception crosses the public surface. Gradient rules stay `void` | `docs/design/PROGRAMME.md:28` |
| Seam | Exceptions thrown by SushiBLAS are caught in one unit, `core/seam.hpp`, in the order `UnsupportedExecution`, `std::invalid_argument`, `std::runtime_error` | owner, 2026-10-08 |
| Lowering | Per-domain tables. A new operation adds a row in its domain's file and edits no other file. No switch or if-chain over operation kinds | owner, 2026-10-08 |
| `TensorPool` | Its own header | `docs/design/PROGRAMME.md:45` |
| `CompiledStep` | Owns its `TaskGraph` | owner, 2026-10-08 |
| Builds | None. No process that runs `sa`, `sr`, `sb`, `cmake`, `ninja`, `ctest` or a test binary. The owner builds once at the end | `docs/design/PROGRAMME.md:30` |

### 1.2 The SushiBLAS v0.1.0 surface, as read in its headers

| Fact | Where |
| --- | --- |
| `Engine` has no view accessor and no `get_graph()`. It offers `get_recorder()`, `get_tensor_factory()`, `get_rng()`, `get_scratch_pool()`, `get_context()`, `get_mode()`, `set_mode()`, `get_layout()`, `execute()`, `create_tensor()` | `sushiblas: include/SushiBLAS/engine.hpp:52-125` |
| `Engine::execute()` compiles the engine's own graph and runs it; it returns `void` and returns early on an empty graph | `sushiblas: src/engine.cpp:43-51` |
| A view is built from `Engine&` or `TaskRecorder&`: `BLASOps`, `ElementwiseOps`, `NonLinearOps`, `ReductionOps`, `SpatialOps`, `LogicOps`. `RandomOps` is built from `Engine&` or from `(TaskRecorder&, RngStream&)` | `sushiblas: include/SushiBLAS/engine/blas.hpp:42-58`, `engine/math/elementwise.hpp`, `nonlinear.hpp`, `reductions.hpp`, `random.hpp`, `engine/spatial.hpp`, `engine/logic/logic.hpp` |
| `TaskRecorder(TaskGraph&, ScratchPool&, ExecutionMode = Auto)`; it is neither copyable nor movable. Its only recording call is `record_pair` | `sushiblas: include/SushiBLAS/engine/task_recorder.hpp:99-170` |
| Every operation returns `TaskHandle` | the view headers above |
| `gemm(A, B, C, Core::Transpose transA, Core::Transpose transB, BlasScalar alpha, BlasScalar beta)`; `gemv(A, x, y, Core::Transpose, BlasScalar, BlasScalar)`; `ger(x, y, A, double alpha)`; `axpy(BlasScalar, x, y)`; `scal(BlasScalar, x)` | `sushiblas: include/SushiBLAS/engine/blas/level1.hpp`, `level2.hpp`, `level3.hpp` |
| `Core::Transpose` is an `enum class` with `NoTrans`, `Trans`, `ConjTrans`; a `bool` does not convert to it | `sushiblas: include/SushiBLAS/core/blas_types.hpp:40-45` |
| `BlasScalar` has implicit constructors from `double` and `std::complex<double>` | `sushiblas: include/SushiBLAS/engine/blas/scalar.hpp:45-52` |
| `Dispatch::execute_gemm(TaskRecorder&, A, B, C, transA, transB, alpha, beta, name, op_id, epilogue)`; `Dispatch::gemm_epilogue(f).reads(t).writes(t)` | `sushiblas: include/SushiBLAS/engine/blas/dispatch.hpp:62-152`, `:195-200` |
| `Dispatch::execute_unary_inplace(TaskRecorder&, Tensor&, name, op_id, fn)` and `execute_zip_inplace(TaskRecorder&, const std::array<Tensor*, N>&, name, op_id, fn)` | `sushiblas: include/SushiBLAS/engine/math/elementwise/dispatch.hpp:63-66`, `:154-157` |
| `Dispatch::bind_reads(fn, reads...)`, `read_as_element(tensor)`, `read_as<U>(tensor)`. The functor takes the elements first, then one pointer per read | `sushiblas: include/SushiBLAS/engine/math/elementwise/read_binding.hpp:138-151`, `:318-324` |
| `SpatialOps::window_dot`, `window_dot_backward`, `window_dot_weights_backward` exist. No SushiAI operation uses them yet | `sushiblas: include/SushiBLAS/engine/spatial.hpp` |
| `SushiBLAS::UnsupportedExecution final : public std::invalid_argument` | `sushiblas: include/SushiBLAS/core/errors.hpp:37-42` |
| Record-time operand refusals throw `std::invalid_argument`; tensor, storage and factory rules throw `std::runtime_error`; a run rethrows what the first failing task threw | `sushiblas: docs/agent/2026_10_07_V010/REPORT.md:1642-1697` |
| Enumerators: `DataType::Half`, `Float32`, `Float64`, `Complex32`, `Complex64`, `Int32`, `Int64`; `Layout::RowMajor`, `ColumnMajor`; `ExecutionMode::Auto`, `ForceSycl`, `ForceReference`; `Core::K_MAX_TENSOR_RANK` | `sushiblas: include/SushiBLAS/core/common.hpp:38-62`, `engine/execution_mode.hpp:38-43` |
| Every family runs under `ForceReference` on a host reference path. Only a device-only USM operand is refused, by `TaskRecorder::require_host_reachable` | `sushiblas: docs/agent/2026_10_07_V010/REPORT.md:170`, `task_recorder.hpp:138-148` |
| A recorded task retains the `Storage` of each declared operand for as long as its node lives | `sushiblas: include/SushiBLAS/engine/task_recorder.hpp:64-91` |

SushiRuntime, as used here: `Graph::TaskGraph(RuntimeContext&)`, `compile()`, `take_nodes()`
(`sushiruntime: include/SushiRuntime/graph/task_graph.hpp:82`, `:231`, `:238`);
`Execution::CompiledGraph` with `roots`, `capture_base_state()`, `rearm()`
(`sushiruntime: include/SushiRuntime/execution/execution_engine.hpp:68-99`);
`IExecutionEngine::execute(CompiledGraph&)`, which blocks until completion (`:177`);
`RuntimeContext::wait_all()` (`sushiruntime: include/SushiRuntime/execution/runtime_context.hpp:351`);
`Core::SushiException : std::runtime_error` with `error().code` of type `Core::Errc`
(`sushiruntime: include/SushiRuntime/core/error.hpp:46-57`, `:100-121`).

### 1.3 Rules every unit follows

1. **A unit records through a `SushiBLAS::TaskRecorder&`.** It builds each view it needs once, as
   a local at the top of the function: `SushiBLAS::BLASOps blas(recorder);`. No unit stores a
   view. No unit names a facade accessor.
2. **A unit that runs work at once** records on `engine.get_recorder()` and then calls
   `Detail::run_recorded(engine)` (unit M1). No unit calls `Engine::execute()` or
   `RuntimeContext::wait_all()` itself.
3. **Under `include/` and `src/`, this migration writes `try`, `catch` and `throw` in M1's two
   files only.** A unit calls SushiBLAS inside
   `Detail::guard` or `Detail::guard_value`, or inside a function its caller guards. Each unit
   says which.
4. **Failure sites.** A site whose row in `tools/contract/site_classes.tsv` has mechanism `status`
   becomes `SA_FAIL_IF(condition, code, same format string and arguments)`. A site with mechanism
   `fatal` becomes `SA_FATAL_IF(condition, same format string and arguments)`. The code is
   `StatusCode::InvalidDescription` unless section 5 names another. No migrated file keeps an
   `SA_THROW_IF`.
5. **Host units not yet converted.** `Graph::value`, `Graph::node`, `Graph::prove_layout_uniformity`,
   `MemoryPlan::build`, `Shape`'s constructors, `Autograd::differentiate` and `Config::*` still
   throw `SushiAI::Error`. A migrated function is not declared `noexcept`, and an `Error` thrown
   by one of those passes through it unchanged. The contract's waves 3 to 5 remove this rule
   (`docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md:1056-1058`).
6. **Scalars.** A scalar passed as `float` today stays a `float` expression at the call, so the
   value that reaches the kernel is the one of today: `scal(static_cast<float>(alpha), y)`.
7. **Flags.** A `bool` transpose flag passes through `Graph::transpose_of(flag)` (unit M10).
8. **Style.** `cpp-code-style` and `source-comments` hold: Allman braces, 4 spaces, 100 columns,
   `[[nodiscard]]` on every function that returns `Status` or `Result`, the licence block and the
   `@file` block of a neighbouring file, Doxygen comments that say what a thing does.
9. **Check before reporting.** Each `.cpp` a unit wrote passes `clang++ -fsyntax-only` with the
   flags of its entry in `build/compile_commands.json`. A new `.cpp` and a header take the flags
   of a neighbouring file of the same folder. A unit's files pass at the end of its own wave;
   files of later waves do not pass until their wave, and that is expected.

10. **Host allocation.** Ruled by the owner on 2026-10-08: a `std::bad_alloc` from a host
    allocation (`new`, `reserve`, `push_back`, `resize`, `emplace`) in a function that returns a
    `Status` or a `Result` becomes `StatusCode::OutOfMemory`. The function runs the allocating
    statements inside `Detail::guard` or `Detail::guard_value` under `SeamPhase::Allocate`, as
    `LossScaler::make` does.

## 2. Inventory

Measured with `grep` over `include/`, `src/`, `apps/`, `examples/` and `tests/` on 2026-10-08.
155 files name a SushiBLAS or SushiRuntime symbol. 131 lines call a facade accessor
(`.blas()`, `.elementwise()`, `.nonlinear()`, `.reductions()`, `.spatial()`, `.random()`), 10 call
a `Dispatch::execute_*` seam, 7 call `Engine::execute()`, and 3 call `get_graph()`.

### 2.1 The spellings

| Old | New |
| --- | --- |
| `engine.blas().f(...)` | `SushiBLAS::BLASOps blas(recorder); blas.f(...)` |
| `engine.elementwise().f(...)` | `SushiBLAS::ElementwiseOps elementwise(recorder); elementwise.f(...)` |
| `engine.nonlinear().f(...)` | `SushiBLAS::NonLinearOps nonlinear(recorder); nonlinear.f(...)` |
| `engine.reductions().f(...)` | `SushiBLAS::ReductionOps reductions(recorder); reductions.f(...)` |
| `engine.spatial().f(...)` | `SushiBLAS::SpatialOps spatial(recorder); spatial.f(...)` |
| `engine.random().f(...)` | `SushiBLAS::RandomOps random(engine); random.f(...)`, or `SushiBLAS::RandomOps random(recorder, stream);` inside a lowering row |
| `gemm(a, b, c, ta, tb, 1.0, 0.0)` with `bool` flags | `gemm(a, b, c, Graph::transpose_of(ta), Graph::transpose_of(tb), 1.0, 0.0)` |
| `gemv(A, x, y, false, 1.0f, 0.0f)` | `gemv(A, x, y, SushiBLAS::Core::Transpose::NoTrans, 1.0f, 0.0f)` |
| `gemv(A, x, y, true, 1.0f, 0.0f)` | `gemv(A, x, y, SushiBLAS::Core::Transpose::Trans, 1.0f, 0.0f)` |
| `Dispatch::execute_gemm(*engine_, ...)` | `Dispatch::execute_gemm(recorder, ..., Graph::transpose_of(ta), Graph::transpose_of(tb), ...)` |
| `Dispatch::execute_zip_inplace(engine, ...)`, `execute_unary_inplace(engine, ...)` | the same call with `recorder` first |
| `engine.execute(); engine.get_context().wait_all();` | `Detail::run_recorded(engine)` |
| `engine.get_graph().compile()` | `Detail::compile_graph(graph_, plan_)` on the `TaskGraph` the step owns |
| `context.get_execution_engine().execute(plan); context.wait_all();` | `Detail::run_plan(context, plan)` |
| `#include <SushiBLAS/engine.hpp>` in a file that calls a family | add the family header: `engine/blas.hpp`, `engine/math/elementwise.hpp`, `engine/math/nonlinear.hpp`, `engine/math/reductions.hpp`, `engine/math/random.hpp`, `engine/spatial.hpp`, `engine/logic/logic.hpp` |

`scal`, `axpy`, `copy`, `ger` and every spatial, nonlinear and reduction call keep their argument
lists. `leaky_relu` and `leaky_relu_backward` take the slope as `double`
(`sushiblas: include/SushiBLAS/engine/math/nonlinear.hpp`); rule 6 keeps the `float` cast.

### 2.2 `graph`

| File | Lines | Old call | New form | Unit |
| --- | --- | --- | --- | --- |
| `src/graph/lowering.cpp` | 94 facade, 7 seam, `:196-197` | `blas` 33 (`copy` 13, `scal` 9, `ger` 4, `axpy` 3, `gemm` 2, `gemv` 2); `elementwise` 19 (`mul` 9, `exp` 3, `sub` 2, `add`, `convert`, `log`, `reciprocal`, `sqrt`); `nonlinear` 14; `reductions` 7 (`sum_along` 4, `max_along`, `mean_spatial`, `sum`); `spatial` 18; `random().constant` 3; `execute_gemm` 5, `execute_zip_inplace` 1, `execute_unary_inplace` 1; `execute()` and `wait_all()` in `flush_constants` | Split over M6, M10, M14 to M21 and M24. The calls are in section 4.3 | M6, M10, M14 to M21, M24 |
| `include/SushiAI/graph/lowering.hpp` | `:36`, `:56-256`, `:264-363` | Holds `TensorPool` and `Lowering` with 35 private `record_*` members | `TensorPool` moves to `graph/tensor_pool.hpp`; `Lowering` keeps five public members | M6, M24 |
| `src/graph/compiled_step.cpp` | `:39-40`, `:47`, `:50`, `:55`, `:58`, `:68-70` | `Lowering(engine, pool, selector)`; `trailing_->record_into_step()`; `engine.get_graph().compile()`; `get_execution_engine().execute(plan_)`; `wait_all()` | The step's own `TaskGraph` and `TaskRecorder`; `Detail::compile_graph`; `Detail::run_plan` | M26 |
| `include/SushiAI/graph/compiled_step.hpp` | `:32-34`, `:60-62`, `:72`, `:111-118` | Constructor that lowers and compiles; `void run()` | `static compile(...)`; `Status run()`; members `graph_`, `recorder_` | M26 |
| `include/SushiAI/graph/step_recorder.hpp` | `:57`, `:86-88`, `:107-113` | `std::size_t record_into_step()` | `Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder&)` | M2 |
| `include/SushiAI/graph/arena.hpp`, `src/graph/arena.cpp` | `:32`, `:35-40`, `:59` | Constructor that allocates through `allocate_arena` | `static Result<ArenaSet> make(...)` | M6 |
| `include/SushiAI/graph/fusion_selector.hpp`, `src/graph/fusion_selector.cpp` | `:117-138`, `:38-39`, `:57` | `Device::Profile`, `Device::profile_for(queue)`, `ExecutionMode::ForceReference`, `engine.get_mode()` | No edit. Every name exists (`sushiblas: include/SushiBLAS/support/device_policy.hpp:115-222`) | none |

### 2.3 `core`, `optim`, `nn`, `io`, `data`, `train`

| File | Lines | Old call | New form | Unit |
| --- | --- | --- | --- | --- |
| `src/core/allocation.cpp` | `:90-102`, `:156-168` | `try` around `make_sushi<Storage>(engine.get_context(), bytes, Residency::Shared)`, rethrown as `Error` | The same `make_sushi` call inside `Detail::guard_value(SeamPhase::Allocate, ...)`. The `Storage` constructor keeps this argument list (`sushiblas: include/SushiBLAS/storage.hpp:65-68`) | M5 |
| `src/optim/optimizer.cpp` | `:149-150` | `engine_->execute(); wait_all()` | `Detail::run_recorded(*engine_)` | M11 |
| `src/optim/adamw.cpp` | `:102-111`, `:185`, `:246-267` | `engine.random().constant` 3; `execute_zip_inplace(engine(), ...)`; `engine().blas()` 9, `engine().elementwise()` 3 | `RandomOps(engine)`; `execute_zip_inplace(recorder, ...)`; `BLASOps(recorder)`, `ElementwiseOps(recorder)` | M22 |
| `src/optim/sgd.cpp` | `:145`, `:169`, `:222-253` | `execute_zip_inplace(engine(), ...)` 2; `engine().blas()` 8 | `execute_zip_inplace(recorder, ...)`; `BLASOps(recorder)` | M23 |
| `src/optim/loss_scaler.cpp` | `:168` | `engine_->reductions().count_nonfinite(...)` | `ReductionOps(recorder).count_nonfinite(...)` | M8 |
| `src/optim/step_scalars.cpp` | `:44` | `allocate_tensor(...)` returning a tensor | `Result<Tensor>` from M5 | M7 |
| `src/optim/optimizer_registry.cpp` | `:91-146` | `std::make_unique<AdamW>(engine, ...)`, `std::make_unique<SGD>(...)` | `AdamW::make(...)`, `SGD::make(...)` | M25 |
| `src/nn/init.cpp` | `:52`, `:62`, `:73`, `:83`, `:88` | `engine.random().normal`, `uniform`, `constant` | `RandomOps(engine)` | M12 |
| `include/SushiAI/nn/init.hpp` | `:175`, `:194-195` | `engine.random().set_seed(seed)`; `engine.execute(); wait_all()` | `RandomOps(engine).set_seed(seed)`; `Detail::run_recorded(engine)` | M12 |
| `include/SushiAI/io/checkpoint.hpp` | `:113-114` | `engine.execute(); wait_all()` | `Detail::run_recorded(engine)` | M13 |
| `src/data/dataset.cpp` | `:216`, `:225-227`, `:251-256` | `engine.random().set_seed`, `normal` 2; `engine.blas().gemm(t, m, x, false, false, 1.0, 0.0)`; `engine.blas().axpy(1.0f, n, x)`; `execute(); wait_all()` 2 | `RandomOps(engine)`; `BLASOps(engine).gemm(t, m, x)` with the defaults, which are `NoTrans`, `NoTrans`, 1.0, 0.0; `Detail::run_recorded(engine)` | M9 |
| `src/train/trainer.cpp` | `:259-262`, `:288-291`, `:319` | `engine_->blas().copy` 2; `execute(); wait_all()`; `get_execution_engine()` for profiling | `BLASOps(*engine_).copy`; `Detail::run_recorded`; the profiling calls keep their form | M27 |

### 2.4 `apps`, `examples`, `tests`

| File | Lines | Old call | New form | Unit |
| --- | --- | --- | --- | --- |
| `apps/train.cpp` | `:96-109`, `:128`, `:150` | `bind_parameters`, `initialize`, `make_optimizer`, `Trainer(...)`, `run_step`, `save_checkpoint` as throwing calls | The `Result` and `Status` forms of section 5 | M28 |
| `apps/inference.cpp` | `:110-114`, `:140` | `materialise_leaves`, `load_checkpoint`, `CompiledStep step(...)`, `step.run()` | `CompiledStep::compile`, `Status run()` | M28 |
| `examples/example.cpp`, `examples/demo_mlp.cpp` | `:53`, `:86-97`; `:549-583` | `try_load_mnist` returning null; `Trainer(...)`; optimizer constructors | The `Result` forms | M29 |
| `tests/package/main.cpp` | `:43-45` | `engine.blas().axpy(2.0f, x, y); engine.execute(); ctx.wait_all();` | `sb::BLASOps(engine).axpy(2.0f, x, y); engine.execute(); ctx.wait_all();` | M30 |
| `tests/functional/integration/test_engine_smoke.cpp` | `:51` | `engine.blas().axpy` | `SushiBLAS::BLASOps(engine).axpy` | M30 |
| `tests/functional/regression/test_fusion_numerics.cpp` | `:500`, `:502`, `:508`, `:554` | `engine.elementwise().mul`; `Lowering(engine, pool)`; `engine.get_graph().take_nodes()` | A `TaskGraph` and a `TaskRecorder` the test owns; `graph.take_nodes()` | M30 |
| `tests/functional/regression/test_fused_optimizer.cpp` | `:382` | `optimizer->record_into_step()` | `record_into_step(engine.get_recorder())` | M30 |
| `tests/functional/unit/test_step_recorder.cpp` | 13 lines | Fakes that override `record_into_step()` | The new signature | M30 |
| 16 more files under `tests/functional/integration` and `regression` | section 3, M30 | Constructors and throwing calls of the migrated SushiAI units | Factories and statuses | M30 |
| `tests/common/test_utils.hpp` | `:52`, `:64-65` | `Engine(context)`, `engine->execute()`, `wait_all()` | No edit; all three exist | none |

### 2.5 Files that name a sibling symbol and do not change

The other files of the 155 name only vocabulary that v0.1.0 keeps: `SushiRuntime::span`
(`SushiRuntime/core/span.hpp`), `SushiBLAS::Core::format`, `SushiBLAS::Core::DataType` and
`Layout` through the aliases at `include/SushiAI/core/common.hpp:46-52`, `SushiBLAS::Tensor` as a
value type, `SushiRuntime::Graph::OpID` (`include/SushiAI/core/ops.hpp:35`), and
`SushiRuntime::Core::NodeTiming` (`include/SushiAI/train/trainer.hpp:319`). They are all of
`autograd`, `config`, `detect`, `eager`, `imaging`, the host units of `graph`, and `nn` apart
from `init`.

## 3. Units

Bricks come before their callers. Each unit names the row of the dispatch matrix an implementer
needs: **one or two files** (spec-complete), **many files** (spec-complete), or **judgment**
(integration judgment).

A listing shows declarations. In the file they sit in nested `namespace` blocks, `SushiAI` then
the module's, as every header of the tree has them; a unit's includes are given in prose. A
member a listing does not show keeps its declaration of today.

### M1. The seam

**Files (new):** `include/SushiAI/core/seam.hpp`, `src/core/seam.cpp`.
**Waits on:** nothing. **Matrix row:** judgment.

```cpp
// include/SushiAI/core/seam.hpp, namespace SushiAI::Detail
enum class SeamPhase : std::uint8_t
{
    Record,   /**< Recording an operation. */
    Allocate, /**< Allocating storage. */
    Execute,  /**< Compiling or running a graph. */
};

[[nodiscard]] Status status_of_current_exception(SeamPhase phase) noexcept;

template<typename Call>
[[nodiscard]] Status guard(SeamPhase phase, Call&& call) noexcept;

template<typename Call>
[[nodiscard]] auto guard_value(SeamPhase phase, Call&& call) noexcept
    -> Result<std::decay_t<decltype(std::forward<Call>(call)())>>;

[[nodiscard]] Status run_recorded(SushiBLAS::Engine& engine) noexcept;
[[nodiscard]] Status compile_graph(SushiRuntime::Graph::TaskGraph& graph,
                                   SushiRuntime::Execution::CompiledGraph& plan) noexcept;
[[nodiscard]] Status run_plan(SushiRuntime::Execution::RuntimeContext& context,
                              SushiRuntime::Execution::CompiledGraph& plan) noexcept;
[[nodiscard]] Status discard_recorded(SushiRuntime::Graph::TaskGraph& graph) noexcept;
```

- The header includes `<exception>`, `<type_traits>`, `<utility>`, `<SushiBLAS/engine.hpp>`,
  `<SushiRuntime/execution/execution_engine.hpp>`, `<SushiRuntime/graph/task_graph.hpp>`,
  `core/result.hpp` and `core/status.hpp`. `seam.cpp` adds `<SushiBLAS/core/errors.hpp>`,
  `<SushiRuntime/core/error.hpp>`, `<new>`, `<stdexcept>` and `core/common.hpp`.
- `SeamPhase` names what a guarded call was doing, which decides the code of an untyped failure.
- `guard` runs a callable that returns `Status`; `guard_value` runs one that returns a value.
  Both are defined in the header: `try { return std::forward<Call>(call)(); }
  catch (...) { return status_of_current_exception(phase); }`.
- `status_of_current_exception` is called inside a `catch` block only and is defined in
  `seam.cpp`. It rethrows with `throw;` inside a `try` whose handlers are, in this order, the
  rows of the table below. It is the only function in the library that names a sibling exception
  type.
- `run_recorded` is `guard(SeamPhase::Execute, ...)` around `engine.execute();
  engine.get_context().wait_all();`. `compile_graph` guards `plan = graph.compile();`. `run_plan`
  guards `context.get_execution_engine().execute(plan); context.wait_all();`. `discard_recorded`
  guards `graph.take_nodes();` and drops the nodes it returns.
- A failure's message is the exception's `what()`. Callers add context with `SA_ANNOTATE`.

| Order | Caught | Becomes |
| --- | --- | --- |
| 1 | `const SushiBLAS::UnsupportedExecution&` | `Unsupported` |
| 2 | `const std::invalid_argument&` | `InvalidDescription` |
| 3 | `const std::runtime_error&`, tested inside the handler with `dynamic_cast` | 3a: a `SushiAI::Error` is `SA_FATAL` with its text; rule 5 explains why one can arrive, and the branch goes when `Error` is deleted. 3b: a `SushiRuntime::Core::SushiException` maps by `error().code`: `allocation_failed` to `OutOfMemory`; `kernel_failed` and `device_lost` to `DeviceFailed`; `unsupported` to `Unsupported`; `invalid_graph`, `compile_failed`, `invalid_access` and `dangling_resource` to `SA_FATAL`. 3c: anything else is `OutOfMemory` under `SeamPhase::Allocate` and `DeviceFailed` otherwise |
| 4 | `const std::bad_alloc&` | `OutOfMemory` |
| 5 | `const std::exception&` | `DeviceFailed`; this is where a `sycl::exception` lands |
| 6 | `...` | `DeviceFailed`, message `an exception of an unknown type` |

Rows 2 and 3c are question Q3. Rows 3b and 4 to 6 follow
`docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/SPEC.md:1000-1011`.

**Acceptance:** `seam.cpp` holds the six handlers in the order of the table and no other `catch`;
`seam.hpp` holds exactly two `catch (...)`, one per template; `git grep -nE "\b(try|catch)\b"`
over the files this migration writes under `include/` and `src/` finds only these two.

### M2. `StepRecorder`

**File:** `include/SushiAI/graph/step_recorder.hpp`. **Waits on:** nothing.
**Matrix row:** one or two files.

```cpp
// StepRecorder's one pure virtual; before_replay() and after_replay() stay as they are
/** @brief Records this recorder's work into @p recorder and returns the task count. */
[[nodiscard]] virtual Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder& recorder) = 0;
```

- Add `#include <SushiAI/core/result.hpp>` and `#include <SushiAI/core/check.hpp>`, and the
  forward declaration `namespace SushiBLAS { class TaskRecorder; }` in the house brace form. The
  header includes no SushiBLAS header.
- `StepRecorderList::record_into_step(SushiBLAS::TaskRecorder& recorder)` passes `recorder` to
  each member in the order added, sums the counts, and returns the first failure:
  `SA_TRY_ASSIGN(const std::size_t count, member->record_into_step(recorder)); recorded += count;`
  inside the braces of the loop.
- The two checks of `StepRecorderList::add` (`:86`, `:88`) become `SA_FATAL_IF` with their text.

**Acceptance:** the file holds no `SA_THROW_IF`; `StepRecorderList` has the members it has today
and no other.

### M3. The lowering row type

**Files (new):** `include/SushiAI/graph/lowering/rule.hpp`,
`include/SushiAI/graph/lowering/unlowered.hpp`. **Waits on:** nothing.
**Matrix row:** one or two files. The text of both files is in section 4.1.

**Acceptance:** `rule.hpp` includes `<array>`, `<cstddef>`, `core/ops.hpp` and `core/status.hpp`
and nothing from SushiBLAS's `engine/`; `unlowered.hpp` holds one array of three `OpID`s.

### M4. The checkpoint byte stream

**Files:** `include/SushiAI/io/checkpoint_format.hpp`, `src/io/checkpoint_format.cpp`.
**Waits on:** nothing. **Matrix row:** one or two files.

The file calls no engine. It is converted here because M13 cannot return a status over a reader
that throws.

```cpp
class CheckpointWriter
{
    public:
        [[nodiscard]] static Result<CheckpointWriter> create(std::ostream& out,
                                                             std::uint32_t count);
        [[nodiscard]] Status write(const TensorRecord& record, const void* payload);
        void finish() const;
    private:
        CheckpointWriter(std::ostream& out, std::uint32_t count) noexcept;
};

class CheckpointReader
{
    public:
        [[nodiscard]] static Result<CheckpointReader> open(std::istream& in);
        [[nodiscard]] Result<TensorRecord> next();
        [[nodiscard]] Status read_payload(const TensorRecord& record, void* destination);
    private:
        CheckpointReader(std::istream& in, std::uint32_t count) noexcept;
};
```

- The file-local `get_u32` and `get_i64` return `Result<std::uint32_t>` and
  `Result<std::int64_t>`; their callers use `SA_TRY_ASSIGN`.
- Each of the 21 sites follows rule 4 with the mechanism of its row. The status sites take
  `StatusCode::InvalidFile`, apart from `:124` and `:141`, which take `InvalidDescription`.
- `create` and `open` run the checks the constructors run today, then construct.

**Acceptance:** no `SA_THROW_IF` in either file; 6 `SA_FATAL_IF` and 15 `SA_FAIL_IF`, the counts
of the table's rows for `src/io/checkpoint_format.cpp`.

### M5. Allocation

**Files:** `include/SushiAI/core/allocation.hpp`, `src/core/allocation.cpp`. **Waits on:** M1.
**Matrix row:** one or two files.

The five functions keep their names and parameter lists
(`include/SushiAI/core/allocation.hpp:47-93`). Four return types change, and all five gain
`[[nodiscard]]`:

| Function | Returns |
| --- | --- |
| `allocate_tensor`, both overloads; `allocate_like` | `Result<SushiBLAS::Tensor>` |
| `allocate_arena` | `Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>>` |
| `view_tensor` | `SushiBLAS::Tensor`, as today |

- `checked_element_count` returns `Result<int64_t>`. `make_storage` returns
  `Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>>`: the `make_sushi` call of today inside
  `Detail::guard_value(SeamPhase::Allocate, ...)`, and on a failure
  `SA_ANNOTATE(status, "Could not allocate the {} tensor {} ({} bytes)", ...)` with today's
  arguments. `allocate_arena` does the same with `"Could not allocate a {} byte arena"`.
- The `Tensor` constructor calls at `:119`, `:143` and `:191` also run inside the guard, since
  `Tensor`'s constructor throws `std::runtime_error`.
- Both `try` blocks and both `throw Error` go. Sites `:50`, `:51`, `:113` are `SA_FAIL_IF`; sites
  `:134`, `:152`, `:176`, `:180`, `:189` are `SA_FATAL_IF`.

**Acceptance:** no `try`, `catch`, `throw` or `SA_THROW_IF` in `allocation.cpp`; the return
types of the table above.

### M6. `ArenaSet` and `TensorPool`

**Files:** `include/SushiAI/graph/arena.hpp`, `src/graph/arena.cpp`;
`include/SushiAI/graph/tensor_pool.hpp` (new), `src/graph/tensor_pool.cpp` (new).
**Waits on:** M1, M5. **Matrix row:** many files.

`ArenaSet`: the constructor becomes private and `noexcept` and takes the allocated storages;
`[[nodiscard]] static Result<ArenaSet> make(SushiBLAS::Engine& engine, MemoryPlan plan);` allocates
one `Storage` per arena with `SA_TRY_ASSIGN` over `allocate_arena`. The two checks of
`ArenaSet::view` (`src/graph/arena.cpp:47`, `:54`) become `SA_FATAL_IF`. `view` keeps its
signature.

`TensorPool` moves out of `lowering.hpp` with its private types `ShapeKey`, `ShapeKeyHash`,
`ScratchKey`, `ScratchKeyHash` and its nine data members, unchanged. Seven members change type:

```cpp
// include/SushiAI/graph/tensor_pool.hpp, namespace SushiAI::Graph
class TensorPool
{
    public:
        [[nodiscard]] Status adopt(MemoryPlan plan);
        [[nodiscard]] Result<SushiBLAS::Tensor> create(const Shape& shape, DType dtype) const;
        [[nodiscard]] Result<SushiBLAS::Tensor&> materialise(ValueId id, const Value& value);
        [[nodiscard]] Status materialise_leaves(const Graph& graph);
        [[nodiscard]] Result<SushiBLAS::Tensor&> ones(const Shape& shape, DType dtype);
        [[nodiscard]] Result<SushiBLAS::Tensor&> scratch(NodeId node, int slot, const Shape& shape,
                                                         DType dtype);
        [[nodiscard]] Status flush_constants();
        // the constructor, has_plan(), arenas(), bind(), is_bound(), both tensor(), alias(),
        // request_constant(), tensor_count(), allocated_bytes(), unpacked_bytes(), engine():
        // as today (include/SushiAI/graph/lowering.hpp:63-201)
};
```

- The bodies are those of `src/graph/lowering.cpp:43-203`, with these changes. The seven checks
  at `:50`, `:60`, `:67`, `:75`, `:83`, `:138`, `:145` become `SA_FATAL_IF`. `adopt` builds the
  arenas with `SA_TRY_ASSIGN(ArenaSet arenas, ArenaSet::make(*engine_, std::move(plan)));` and
  then `arenas_.emplace(std::move(arenas));`. `create`, `materialise`, `ones` and `scratch` pass
  an allocation failure up with `SA_TRY_ASSIGN`. `ArenaSet::view` is called inside
  `Detail::guard_value(SeamPhase::Allocate, ...)`.
- `materialise_leaves` tests `Ops::K_LEAF.id`, as today. A pool must tell a leaf from a node to
  allocate leaves before lowering; it dispatches nothing on the result.
- `flush_constants`: returns `Status()` when nothing is queued. Otherwise it records every fill
  with `SushiBLAS::RandomOps random(*engine_); random.constant(*tensor, value);` inside one
  `Detail::guard(SeamPhase::Record, ...)`, clears the queue, and returns
  `Detail::run_recorded(*engine_)`.
- `tensor_pool.hpp` includes what `lowering.hpp` includes today (`:30-44`) apart from
  `graph/fusion.hpp` and `graph/fusion_selector.hpp`, and adds `core/result.hpp` and
  `core/status.hpp`. `tensor_pool.cpp` includes `<SushiBLAS/engine/math/random.hpp>`,
  `core/allocation.hpp`, `core/ops.hpp` and `core/seam.hpp`.

**Acceptance:** `tensor_pool.hpp` does not include `graph/lowering.hpp`, `fusion.hpp` or
`fusion_selector.hpp`; `tensor_pool.cpp` holds no `SA_THROW_IF` and no call of
`Engine::execute()`.

### M7. `StepScalars`

**Files:** `include/SushiAI/optim/step_scalars.hpp`, `src/optim/step_scalars.cpp`.
**Waits on:** M5. **Matrix row:** one or two files.

```cpp
class StepScalars final
{
    public:
        [[nodiscard]] static Result<StepScalars> make(SushiBLAS::Engine& engine, std::size_t count);
        StepScalars(StepScalars&&) noexcept = default;
        StepScalars& operator=(StepScalars&&) noexcept = default;
        ~StepScalars() = default;
    private:
        StepScalars(SushiBLAS::Tensor buffer, std::size_t count) noexcept;
};
```

The two deleted copy members stay, so all five special members are declared. `make` fails with
`InvalidDescription` on a zero count (`:40`), allocates through
`allocate_tensor`, zeroes the buffer on the host as today, and constructs. The check of `write`
(`:52`) is `SA_FATAL_IF`.

**Acceptance:** no public constructor that takes an `Engine&`; no `SA_THROW_IF`.

### M8. `LossScaleState` and `LossScaler`

**Files:** `include/SushiAI/optim/loss_scaler.hpp`, `src/optim/loss_scaler.cpp`.
**Waits on:** M1, M5. **Matrix row:** one or two files.

```cpp
class LossScaleState
{
    public:
        [[nodiscard]] static Result<LossScaleState> make(LossScaleOptions options = {});
    private:
        explicit LossScaleState(LossScaleOptions options) noexcept;
};

class LossScaler
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<LossScaler>> make(
            SushiBLAS::Engine& engine, SushiBLAS::Tensor factor, std::size_t parameter_count,
            LossScaleOptions options = {});
        [[nodiscard]] Result<std::size_t> record_screens(
            SushiBLAS::TaskRecorder& recorder, const std::vector<SushiBLAS::Tensor*>& gradients);
    private:
        LossScaler(SushiBLAS::Tensor factor, SushiRuntime::sushi_ptr<SushiBLAS::Storage> arena,
                   std::vector<SushiBLAS::Tensor> count_views, LossScaleState state) noexcept;
};
```

- `LossScaleState::make` holds the seven option checks (`:58-79`) as `SA_FAIL_IF`.
  `LossScaler::make` holds the three checks (`:124-133`) as `SA_FAIL_IF`, then
  `LossScaleState::make`, `allocate_arena`, and the views, each `view_tensor` call inside
  `Detail::guard_value(SeamPhase::Allocate, ...)`. The host writes of `:147-150` follow.
- The member `engine_` goes; nothing reads it after this change.
- `record_screens` keeps its two checks as `SA_FATAL_IF`, then records
  `SushiBLAS::ReductionOps reductions(recorder); reductions.count_nonfinite(*gradients[i],
  count_views_[i]);` for every `i` inside one `Detail::guard_value(SeamPhase::Record, ...)` that
  returns `count_`.
- Include `<SushiBLAS/engine/math/reductions.hpp>`, `<memory>` and `core/seam.hpp`.

**Acceptance:** no `SA_THROW_IF`; no member of type `SushiBLAS::Engine*`.

### M9. Datasets

**Files:** `include/SushiAI/data/dataset.hpp`, `src/data/dataset.cpp`. **Waits on:** M1, M5.
**Matrix row:** judgment (file parsing, 18 sites, two factories).

```cpp
class SyntheticClassification final : public Dataset
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<SyntheticClassification>> make(
            SushiBLAS::Engine& engine, SyntheticOptions options = {});
    private:
        SyntheticClassification(SyntheticOptions options, std::vector<Batch> batches) noexcept;
};
// Rings: the same pair, over RingsOptions.

[[nodiscard]] Result<std::unique_ptr<Dataset>> try_load_mnist(SushiBLAS::Engine& engine,
                                                              const MnistOptions& options = {});
using DatasetFactory = Result<std::unique_ptr<Dataset>> (*)(SushiBLAS::Engine& engine,
                                                            const Config::Value& attributes,
                                                            std::string_view context);
[[nodiscard]] Result<std::unique_ptr<Dataset>> make_rings(...);    // same parameters as today
[[nodiscard]] Result<std::unique_ptr<Dataset>> make_mnist(...);
[[nodiscard]] Result<std::unique_ptr<Dataset>> make_dataset(...);
```

- `Dataset::batch(index)` keeps returning `const Batch&`; its three range checks (`:191`, `:264`,
  `:440`) are `SA_FATAL_IF`.
- `try_load_mnist` returns `FileNotFound` where it returns null today, with the text of
  `mnist_expected_paths(options)`, and `InvalidFile` at the sites its rows mark `outside` or
  `user-content` after a file was opened (`:293-358`). `:287` is `InvalidDescription`.
- The recording blocks at `:216-227` and `:251-256` take the forms of section 2.3, each inside
  `Detail::guard(SeamPhase::Record, ...)` and followed by `SA_TRY(Detail::run_recorded(engine))`.
- A function that returns `Result<std::unique_ptr<Dataset>>` from a
  `std::unique_ptr<SyntheticClassification>` names a `std::unique_ptr<Dataset>` local first and
  returns it with `return std::move(local);`.

**Acceptance:** no `SA_THROW_IF`; no `return nullptr`; 15 `SA_FAIL_IF` or annotated seam
failures and 3 `SA_FATAL_IF`, the counts of the table's rows for `src/data/dataset.cpp`.

### M10. `LoweringContext`

**Files (new):** `include/SushiAI/graph/lowering/context.hpp`,
`src/graph/lowering/context.cpp`. **Waits on:** M3, M6. **Matrix row:** one or two files.

A row sees a `LoweringContext` and nothing else, as a gradient rule sees a `GradientTape`
(`include/SushiAI/autograd/tape.hpp:47`).

```cpp
// include/SushiAI/graph/lowering/context.hpp, namespace SushiAI::Graph
[[nodiscard]] constexpr SushiBLAS::Core::Transpose transpose_of(bool transposed) noexcept
{
    return transposed ? SushiBLAS::Core::Transpose::Trans : SushiBLAS::Core::Transpose::NoTrans;
}

class LoweringContext final
{
    public:
        LoweringContext(SushiBLAS::TaskRecorder& recorder, SushiBLAS::RngStream& stream,
                        TensorPool& pool, const Graph& graph, NodeId id,
                        const FusionSelector& selector, std::vector<FusionSelection>& selections,
                        std::size_t& emitted) noexcept;

        [[nodiscard]] SushiBLAS::TaskRecorder& recorder() const noexcept;
        [[nodiscard]] SushiBLAS::RngStream& stream() const noexcept;
        [[nodiscard]] const Graph& graph() const noexcept;
        [[nodiscard]] NodeId id() const noexcept;
        [[nodiscard]] const Node& node() const;
        [[nodiscard]] std::string_view name() const;

        void require_arity(std::size_t inputs, std::size_t outputs) const;
        void require_arity_at_least(std::size_t inputs, std::size_t outputs) const;

        [[nodiscard]] const Value& input_value(std::size_t index) const;
        [[nodiscard]] const Value& output_value(std::size_t index) const;
        [[nodiscard]] const SushiBLAS::Tensor& input(std::size_t index) const;
        [[nodiscard]] SushiBLAS::Tensor& mutable_input(std::size_t index) const;
        [[nodiscard]] Result<SushiBLAS::Tensor&> output(std::size_t index) const;
        [[nodiscard]] bool output_is_bound(std::size_t index) const;
        void alias_output(std::size_t output, std::size_t input) const;

        [[nodiscard]] Result<SushiBLAS::Tensor&> scratch(int slot, const Shape& shape,
                                                         DType dtype) const;
        [[nodiscard]] Result<SushiBLAS::Tensor&> ones(const Shape& shape, DType dtype) const;

        void count(std::size_t tasks) const noexcept;
        [[nodiscard]] std::size_t emitted() const noexcept;
        [[nodiscard]] FusionChoice select(const FusionQuery& query) const;
        void close_selection(std::size_t emitted_before) const noexcept;
};
```

- The header includes `<cstddef>`, `<string_view>`, `<vector>`,
  `<SushiBLAS/core/blas_types.hpp>`, `<SushiBLAS/engine/rng_stream.hpp>`,
  `<SushiBLAS/engine/task_recorder.hpp>`, `core/result.hpp`, `graph/fusion_selector.hpp`,
  `graph/ir.hpp` and `graph/tensor_pool.hpp`.
- The class holds the seven reference arguments as references and the id by value. Its copy and
  move members are deleted and its destructor is defaulted, all five declared.
- `name()` is `Ops::name_of(node().op)`. `require_arity` is `SA_FATAL_IF(node.inputs.size() !=
  inputs || node.outputs.size() != outputs, "{} (node {}) takes {} operand(s) and {} result(s);
  it has {} and {}", ...)`. `require_arity_at_least` tests `<` and says `at least`.
- `input_value`, `output_value`, `input`, `mutable_input`, `output`, `output_is_bound` and
  `alias_output` are fatal on an index past the node's list, with the text of
  `src/graph/lowering.cpp:306` and `:314`. `input` and `mutable_input` return
  `pool.tensor(node.inputs[index])`. `output` returns `pool.materialise(id, graph.value(id))`
  for `id = node.outputs[index]`. `alias_output` is `pool.alias(node.outputs[output],
  node.inputs[input])`. `scratch` is `pool.scratch(id(), slot, shape, dtype)`.
- `count` adds to the `emitted` reference. `select` is the body of
  `src/graph/lowering.cpp:351-372` with `selector.choose(query)` in place of the null test, the
  check at `:363` as `SA_FATAL_IF`, and the push onto `selections`. `close_selection` is
  `selections.back().tasks = emitted - emitted_before`.

**Acceptance:** `context.hpp` includes no family header of SushiBLAS; the class has no data
member other than the eight above.

### M11. `Optimizer`

**Files:** `include/SushiAI/optim/optimizer.hpp`, `src/optim/optimizer.cpp`.
**Waits on:** M1, M2, M5, M8. **Matrix row:** one or two files.

```cpp
/** @brief Reports the first slot whose parameter and gradient disagree, or success. */
[[nodiscard]] Status validate_slots(SushiRuntime::span<const ParameterSlot> slots);

class Optimizer : public Graph::StepRecorder
{
    public:
        [[nodiscard]] Status step();
        [[nodiscard]] Status set_loss_scaler(LossScaler* scaler);
        [[nodiscard]] Result<std::size_t> record_into_step(
            SushiBLAS::TaskRecorder& recorder) final;
    protected:
        Optimizer(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters) noexcept;
        virtual std::size_t record_update(SushiBLAS::TaskRecorder& recorder) = 0;
        [[nodiscard]] Result<std::size_t> record_all(SushiBLAS::TaskRecorder& recorder);
        [[nodiscard]] Status flush();
        [[nodiscard]] Result<SushiBLAS::Tensor> allocate_state(const SushiBLAS::Tensor& like) const;
};
```

- `validate_slots` holds the four checks of the constructor (`:42-52`) as `SA_FAIL_IF`. The
  constructor checks nothing; every `make` of a derived class calls `validate_slots` first.
- `set_loss_scaler` holds its two checks (`:62`, `:69`) as `SA_FAIL_IF`.
- `record_update` may let a SushiBLAS exception escape. `record_all` is its only caller and
  guards it: `SA_TRY_ASSIGN(const std::size_t screens, scaler_->record_screens(recorder,
  gradients));` when a scaler is set, then `SA_TRY_ASSIGN(const std::size_t updates,
  Detail::guard_value(Detail::SeamPhase::Record, [&]() { return record_update(recorder); }));`,
  and returns the sum.
- `record_into_step` sets `in_compiled_step_` and returns `record_all(recorder)`.
- `step`: the check at `:133` as `SA_FATAL_IF`; `before_replay();
  SA_TRY(record_all(engine_->get_recorder())); SA_TRY(flush()); after_replay(); return Status();`.
- `flush` returns `Detail::run_recorded(*engine_)`. `allocate_state` returns
  `allocate_like(*engine_, like)`.

**Acceptance:** no `SA_THROW_IF`, no `execute(` and no `wait_all` in `optimizer.cpp`.

### M12. Initialisation

**Files:** `include/SushiAI/nn/init.hpp`, `src/nn/init.cpp`. **Waits on:** M1, M6.
**Matrix row:** one or two files.

`kaiming_normal`, `kaiming_uniform`, `xavier_normal`, `xavier_uniform`, `zeros`,
`Detail::apply_initializer` and the template `initialize` keep their names and parameter lists
and return `[[nodiscard]] Status` in place of `void`. One function is new:

```cpp
/** @brief Records a fill of @p tensor with @p value. */
[[nodiscard]] Status constant(SushiBLAS::Engine& engine, SushiBLAS::Tensor& tensor, double value);
```

- `init.hpp` includes `graph/tensor_pool.hpp` in place of `graph/lowering.hpp`, and
  `core/status.hpp`.
- Each of the six fill functions keeps its fan check as `SA_FAIL_IF` (`:51`, `:58`, `:68`, `:79`)
  and records one call of `SushiBLAS::RandomOps random(engine);` inside
  `SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, ...)`. `zeros` returns
  `constant(engine, tensor, 0.0)`.
- `apply_initializer` reads `parameter.init()`
  (`include/SushiAI/nn/module.hpp:65-91`, `:196`), in this order: a `Kind::Constant` parameter is
  filled with `constant(engine, tensor, parameter.init().constant)`; otherwise a final layer is
  zeroed; otherwise the fan-scaled fill of today. The test on `parameter.role()` at `:100` goes.
  This is items 1 and 2 of U14 in `docs/agent/2026_10_07_MODEL_GRAPH/PLAN.md:97`.
- `initialize` calls `SushiBLAS::RandomOps(engine).set_seed(policy.seed)`, keeps the search for
  the last `ParameterRole::Weight`, applies each initialiser with `SA_TRY`, and returns
  `SushiAI::Detail::run_recorded(engine)`.
- `init.cpp` includes `<SushiBLAS/engine/math/random.hpp>` and `core/seam.hpp`.

**Acceptance:** `init.cpp` holds no `SA_THROW_IF`; `apply_initializer` does not name
`ParameterRole`.

### M13. Checkpoints

**File:** `include/SushiAI/io/checkpoint.hpp`. **Waits on:** M1, M4, M6.
**Matrix row:** one or two files.

```cpp
template<typename M>
[[nodiscard]] Status save_checkpoint(const std::filesystem::path& path, M& module,
                                     Graph::TensorPool& pool, SushiBLAS::Engine& engine);
template<typename M>
[[nodiscard]] Status load_checkpoint(const std::filesystem::path& path, M& module,
                                     Graph::TensorPool& pool);
```

- Include `graph/tensor_pool.hpp` in place of `graph/lowering.hpp`, and `core/seam.hpp`.
- `save_checkpoint` opens with `SA_TRY(SushiAI::Detail::run_recorded(engine));`, then uses
  `CheckpointWriter::create`, `SA_TRY(writer.write(...))` and `writer.finish()`.
- `load_checkpoint` uses `CheckpointReader::open`, `SA_TRY_ASSIGN(const TensorRecord record,
  reader.next());` and `SA_TRY(reader.read_payload(...))`.
- `Detail::bind` keeps its signature; its check (`:76`) is `SA_FATAL_IF`. The eight other sites
  are `SA_FAIL_IF` with the codes of section 5.

**Acceptance:** no `SA_THROW_IF`; both templates return `Status`.

### M14 to M21. The eight domain files

**Files (new), one unit each:** `include/SushiAI/graph/lowering/<domain>.hpp` and
`src/graph/lowering/<domain>.cpp`, for `elementwise` (M14), `activation` (M15), `linalg` (M16),
`loss` (M17), `shape` (M18), `spatial` (M19), `norm` (M20) and `fused` (M21).
**Wait on:** M3, M10. **Matrix row:** one or two files each; M20 and M21 are judgment.

The header of a domain has the shape of `include/SushiAI/autograd/rules/norm.hpp`:

```cpp
// include/SushiAI/graph/lowering/norm.hpp, namespace SushiAI::Graph
// includes: <array>, <SushiAI/graph/lowering/rule.hpp>
[[nodiscard]] Status lower_batchnorm2d(const LoweringContext& context);
[[nodiscard]] Status lower_batchnorm2d_backward(const LoweringContext& context);
[[nodiscard]] Status lower_channel_affine(const LoweringContext& context);
[[nodiscard]] Status lower_batchnorm_fold(const LoweringContext& context);

inline constexpr std::array<LoweringEntry, 4> K_NORM_LOWERINGS = {{
    {Ops::K_BATCHNORM2D.id, &lower_batchnorm2d},
    {Ops::K_BATCHNORM2D_BACKWARD.id, &lower_batchnorm2d_backward},
    {Ops::K_CHANNEL_AFFINE.id, &lower_channel_affine},
    {Ops::K_BATCHNORM_FOLD.id, &lower_batchnorm_fold},
}};
```

A row function is named `lower_` plus the operation's name in lower case. A table is named
`K_<DOMAIN>_LOWERINGS`. The rows of each table, their order and their calls are in section 4.3.

Every row function has this shape:

1. `context.require_arity(inputs, outputs);`, with the counts of section 4.3.
2. The other checks of today's function, each as `SA_FATAL_IF` with today's text, in today's
   order. The one status site is in `lower_dynamic_scale`.
3. Operands: `const SushiBLAS::Tensor& x = context.input(0);`. Results:
   `SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));`. Scratch and ones the same way:
   `SA_TRY_ASSIGN(SushiBLAS::Tensor& rows, context.ones(Shape{n}, dtype));`.
4. One view per family used, built from `context.recorder()`.
5. The calls of section 4.3, in that order.
6. `context.count(tasks); return Status();`.

A row holds no `try`, no `SA_THROW_IF` and no test of `node.op`. The driver guards it (M24).
Where several rows differ only in the kernel, the `.cpp` holds one file-local helper that takes
the kernel as a pointer to member, and each row is one call of it:

```cpp
using InPlaceKernel = SushiBLAS::TaskHandle (SushiBLAS::NonLinearOps::*)(SushiBLAS::Tensor&);

Status record_copy_then(const LoweringContext& context, InPlaceKernel kernel);   // file-local

Status lower_relu(const LoweringContext& context)
{
    return record_copy_then(context, &SushiBLAS::NonLinearOps::relu);
}
```

The helper shapes: `BinaryKernel` over `ElementwiseOps` and over `LogicOps`
(`(const Tensor&, const Tensor&, Tensor&)`), `InPlaceKernel` over `ElementwiseOps` and over
`NonLinearOps`, `BackwardKernel` over `NonLinearOps` (`(const Tensor&, const Tensor&, Tensor&)`).

Notes per domain, beyond section 4.3:

- **M14 `elementwise`.** `lower_dynamic_scale` keeps `SA_FAIL_IF(factor.num_elements != 1,
  StatusCode::InvalidDescription, ...)` and the `static_assert` on the task name
  (`src/graph/lowering.cpp:734-736`). Its functor no longer captures a pointer:
  `Dispatch::bind_reads([](auto v, auto f) { return v * f[0]; },
  Dispatch::read_as_element(factor))`. This closes the backlog item "Move the loss-scale factor
  to a declared operand" (`docs/design/REMAINING_WORK.md:35`). `lower_zero` and `lower_constant`
  build `SushiBLAS::RandomOps random(context.recorder(), context.stream());`. `lower_accumulate`
  calls `context.alias_output(0, 0)` before it reads `context.mutable_input(0)`.
- **M16 `linalg`.** `lower_matmul` tests no rank. `gemm` takes its batch count from `C`
  (`sushiblas: include/SushiBLAS/engine/blas/dispatch.hpp:235-240`), so a rank-3 `MATMUL` lowers
  through the same row once the builder emits one.
- **M17 `loss`.** No softmax-backward row exists, because no such identity is in
  `include/SushiAI/core/ops.hpp:292-313`. It is one row here when the identity lands.
- **M18 `shape`.** `lower_leaf` records nothing: it is fatal when `!context.output_is_bound(i)`
  for any output, with the text of `src/graph/lowering.cpp:382-385`. `lower_stop_gradient` is a
  copy, as `RESHAPE` is; the memory plan knows no alias between the two values.
- **M19 `spatial`.** The file-local `blas_geometry_of` and `windowed_extents_of`
  (`src/graph/lowering.cpp:979-1011`) move here, the second with `SA_FATAL_IF`.
- **M20 `norm`.** The file-local `channel_rows` (`:1216-1222`) moves here.
  `lower_batchnorm2d` takes four results and writes the biased variance to result 3; this is U13
  of `docs/agent/2026_10_07_MODEL_GRAPH/PLAN.md:96`. `lower_batchnorm_fold` is new.
- **M21 `fused`.** The five epilogue functors (`:227-301`) move here unchanged. One file-local
  template records every GEMM form:

  ```cpp
  using ActivationKernel = SushiBLAS::TaskHandle (SushiBLAS::NonLinearOps::*)(SushiBLAS::Tensor&);

  template<typename MakeEpilogue>
  Status record_fused_gemm(const LoweringContext& context, FusedForm form,
                           const Ops::OpName& identity, ActivationKernel tail,
                           MakeEpilogue&& make_epilogue);
  ```

  `make_epilogue(bias, aux)` returns the `Dispatch::GemmEpilogue` of the form, with its `reads`
  and `writes` declared, as `:554-596` spell them. The template is the body of `:482-633` with
  the `switch` over `choice.form` replaced by three outcomes of the selector's answer: `form`
  (call `execute_gemm` with `make_epilogue`), `FusedForm::GemmBias` (call it with
  `gemm_epilogue(BiasEpilogue{values}).reads(bias)` under `Ops::K_GEMM_BIAS`), and
  `FusedForm::None` (`gemm`, then `ger` over ones). Any other answer was refused by
  `context.select`. The trailing activation, when the chosen form did not fold it, is
  `(nonlinear.*tail)(c)`; `tail` is null for `GEMM_BIAS`. Each of the five rows is one call of
  the template. `lower_exp_sub_scale` is the body of `:635-687`.

**Acceptance, each unit:** the table has the rows of section 4.3 in that order; the `.cpp` holds
no `SA_THROW_IF`, no `engine_`, no `node.op ==`; every row ends in `context.count(n)` with the
task count of section 4.3.

### M22 and M23. `AdamW` and `SGD`

**Files:** `include/SushiAI/optim/adamw.hpp`, `src/optim/adamw.cpp` (M22);
`include/SushiAI/optim/sgd.hpp`, `src/optim/sgd.cpp` (M23). **Wait on:** M7, M11.
**Matrix row:** one or two files each.

```cpp
class AdamW final : public Optimizer
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<AdamW>> make(
            SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
            AdamWOptions options = {});
    protected:
        std::size_t record_update(SushiBLAS::TaskRecorder& recorder) override;
        void write_step_scalars() override;
    private:
        AdamW(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters,
              AdamWOptions options, std::vector<State> state, StepScalars scalars) noexcept;
        std::size_t record_fused(SushiBLAS::TaskRecorder& recorder);
        std::size_t record_separate(SushiBLAS::TaskRecorder& recorder);
};
// SGD: the same members over SGDOptions; its private constructor also takes bool has_momentum.
```

- `make`: `SA_TRY(validate_slots(...))`; the option checks as `SA_FAIL_IF` (`adamw.cpp:86-90`,
  `sgd.cpp:72-75`); `StepScalars::make`; one `allocate_like` per state tensor with
  `SA_TRY_ASSIGN`; for `AdamW`, the constant fills through `SushiBLAS::RandomOps random(engine);`
  inside `Detail::guard(Detail::SeamPhase::Record, ...)` and then
  `SA_TRY(Detail::run_recorded(engine))`; then `std::unique_ptr<AdamW> made(new AdamW(...));
  return std::move(made);`.
- `record_fused` and `record_separate` replace `engine()` by `recorder` in the seam calls and
  build `SushiBLAS::BLASOps blas(recorder);` and `SushiBLAS::ElementwiseOps
  elementwise(recorder);` once. The functors and their two captured pointers do not change.
- The checks of `first_moment`, `second_moment` (`adamw.cpp:148`, `:156`) and of
  `SGD::write_step_scalars` (`sgd.cpp:98`, `:103`) are `SA_FATAL_IF`.

**Acceptance:** no public constructor; no `engine()` call inside `record_fused` or
`record_separate`; the lambda bodies are byte for byte those of today.

### M24. The table and the driver

**Files:** `include/SushiAI/graph/lowering/all.hpp` (new); `include/SushiAI/graph/lowering.hpp`
and `src/graph/lowering.cpp` (both rewritten). **Waits on:** M14 to M21.
**Matrix row:** one or two files. The text of `all.hpp` is in section 4.2.

```cpp
// include/SushiAI/graph/lowering.hpp
class Lowering final
{
    public:
        Lowering(SushiBLAS::TaskRecorder& recorder, SushiBLAS::RngStream& stream,
                 TensorPool& pool, const FusionSelector& selector) noexcept;
        [[nodiscard]] static std::vector<ValueId> values_needing_own_allocation(const Graph& graph);
        [[nodiscard]] Status record(const Graph& graph);
        [[nodiscard]] std::size_t emitted_operations() const noexcept;
        [[nodiscard]] SushiRuntime::span<const FusionSelection> selections() const noexcept;
};
```

- `lowering.hpp` no longer defines `TensorPool`; it includes `graph/tensor_pool.hpp`,
  `graph/fusion_selector.hpp` and `graph/lowering/context.hpp`. `lowering.cpp` includes
  `graph/lowering/all.hpp` and `core/seam.hpp`.
- `record`: reset `emitted_` and `selections_`; `graph.prove_layout_uniformity();`; then for
  every node id in order:

  ```cpp
  const Node& node = graph.node(id);
  const LoweringRule rule = find_lowering(node.op);
  SA_FATAL_IF(rule == nullptr,
              "No lowering for operation {} (node {}). Add a row to its domain's table under "
              "include/SushiAI/graph/lowering/, or list it in unlowered.hpp.",
              Ops::name_of(node.op), id);

  const LoweringContext context(*recorder_, *stream_, *pool_, graph, id, *selector_,
                                selections_, emitted_);
  const Status recorded = Detail::guard(Detail::SeamPhase::Record,
                                        [&]() { return rule(context); });
  if (!recorded.is_ok())
      return SA_ANNOTATE(recorded, "{} (node {})", Ops::name_of(node.op), id);
  ```

- `values_needing_own_allocation` keeps its body (`src/graph/lowering.cpp:322-337`); see Q6.
- Nothing else of today's `lowering.cpp` stays.

**Acceptance:** `lowering.cpp` is under 120 lines and holds no `record_` function;
`git grep -n "node.op ==" src/graph/lowering` finds nothing.

### M25. The optimizer registry

**Files:** `include/SushiAI/optim/optimizer_registry.hpp`, `src/optim/optimizer_registry.cpp`.
**Waits on:** M22, M23. **Matrix row:** one or two files.

`OptimizerFactory`, `make_adamw`, `make_sgd` and `make_optimizer` return
`Result<std::unique_ptr<Optimizer>>` with the parameters they have today. `make_adamw` is
`SA_TRY_ASSIGN(std::unique_ptr<AdamW> made, AdamW::make(engine, std::move(parameters), options));
std::unique_ptr<Optimizer> base = std::move(made); return std::move(base);`. The two sites
(`:108`, `:132`) are `SA_FAIL_IF`.

**Acceptance:** no `std::make_unique` in the file; no `SA_THROW_IF`.

### M26. `CompiledStep`

**Files:** `include/SushiAI/graph/compiled_step.hpp`, `src/graph/compiled_step.cpp`.
**Waits on:** M1, M2, M24. **Matrix row:** judgment.

```cpp
class CompiledStep final
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<CompiledStep>> compile(
            SushiBLAS::Engine& engine, const Graph& graph, TensorPool& pool,
            StepRecorder* trailing = nullptr, const FusionSelector* selector = nullptr);
        [[nodiscard]] Status run();
    private:
        CompiledStep(SushiBLAS::Engine& engine, StepRecorder* trailing);

        SushiBLAS::Engine* engine_;
        StepRecorder* trailing_;
        SushiRuntime::Graph::TaskGraph graph_;      // the step's own graph
        SushiBLAS::TaskRecorder recorder_;          // bound to graph_
        SushiRuntime::Execution::CompiledGraph plan_;
        // selections_, compile_count_, run_count_, operations_, empty_ as today
};
```

- The four deleted copy and move members stay and `~CompiledStep() = default;` joins them.
- The members are declared in this order, so `graph_` is built before `recorder_` and destroyed
  after `plan_`. The constructor is `engine_(&engine), trailing_(trailing),
  graph_(engine.get_context()), recorder_(graph_, engine.get_scratch_pool(), engine.get_mode())`.
- The header adds `<memory>`, `<SushiBLAS/engine/task_recorder.hpp>`,
  `<SushiRuntime/graph/task_graph.hpp>`, `core/result.hpp` and `core/status.hpp`.
- `compile`, in order:
  1. `SA_FAIL_IF(graph.has_refusal(), StatusCode::InvalidDescription, "This graph holds a refused
     operation and cannot be compiled; the builder call that refused it returned the reason");`
     (`include/SushiAI/graph/ir.hpp:455`).
  2. `std::unique_ptr<CompiledStep> step(new CompiledStep(engine, trailing));`
  3. `const DeviceFusionSelector device = DeviceFusionSelector::for_engine(engine);` and
     `Lowering lowering(step->recorder_, engine.get_rng(), pool, selector ? *selector : device);`
  4. `Status recorded = lowering.record(graph);` then, when it is ok and `trailing` is set,
     `trailing->record_into_step(step->recorder_)`, whose count is added to `operations_`.
  5. On a failure of step 4: `const Status dropped = Detail::discard_recorded(step->graph_);`,
     `SA_FATAL_IF(!dropped.is_ok(), "A failed recording could not be dropped: {}",
     dropped.message());`, and return the failure of step 4.
  6. `SA_TRY(Detail::compile_graph(step->graph_, step->plan_));`, then `++compile_count_` and
     `empty_ = plan_.roots.empty()`.
  7. `SA_TRY(pool.flush_constants());`
  8. `SA_TRY(Detail::guard(Detail::SeamPhase::Execute, [&]() { step->plan_.capture_base_state();
     return Status(); }));`
  9. `return std::move(step);`
- `run`: `if (empty_) return Status();`, `before_replay`, `plan_.rearm();`,
  `SA_TRY(Detail::run_plan(engine_->get_context(), plan_));`, `++run_count_`, `after_replay`.
- The order of steps 6 to 8 is today's (`src/graph/compiled_step.cpp:50-58`). The constant fills
  run on the engine's own graph, so they no longer depend on that order; it is kept so that this
  unit changes one thing.

**Acceptance:** the file calls `compile()` on one object, `graph_`, in one place; it names
neither `get_graph` nor `Engine::execute`.

### M27. `Trainer` and the umbrella header

**Files:** `include/SushiAI/train/trainer.hpp`, `src/train/trainer.cpp`,
`include/SushiAI/SushiAI.h`. **Waits on:** M11, M26. **Matrix row:** judgment.

```cpp
[[nodiscard]] Result<std::vector<Optim::ParameterSlot>> bind_parameters(
    const TrainingGraph& built, Graph::TensorPool& pool, Graph::PlanOptions options = {});

class Trainer final
{
    public:
        [[nodiscard]] static Result<std::unique_ptr<Trainer>> make(
            SushiBLAS::Engine& engine, const Graph::Graph& graph, Graph::TensorPool& pool,
            StepSpec spec, Optim::Optimizer& optimizer, TrainerOptions options = {});
        [[nodiscard]] Result<StepMetrics> run_step(const Data::Batch& batch);
        [[nodiscard]] Result<RunReport> fit(const Data::Dataset& dataset);
    private:
        Trainer(SushiBLAS::Engine& engine, Graph::TensorPool& pool, StepSpec spec,
                Optim::Optimizer& optimizer, TrainerOptions options,
                std::unique_ptr<Graph::CompiledStep> step, int64_t rows, int64_t classes) noexcept;
        [[nodiscard]] Result<double> score_accuracy(const Data::Batch& batch) const;
        std::unique_ptr<Graph::CompiledStep> step_;
};
```

- The accessors that read `step_` dereference the pointer; `step()` keeps returning
  `const Graph::CompiledStep&`.

- The file-local `placement_recorder` returns `Result<Graph::StepRecorder*>`; `verified` becomes
  `[[nodiscard]] Status verify_precision(const Graph::Graph&, const TrainerOptions&)`.
- `make` runs, in order: the four-ids check (`:184`), the loss element count (`:192`), a check
  that the loss value's dtype is `Float32` or `Float64` (the text of `:206`),
  `verify_precision`, `placement_recorder`, `Graph::CompiledStep::compile`, then constructs.
- `current_loss` keeps `double`; its default branch is `SA_FATAL`, since `make` checked the
  dtype. `score_accuracy` keeps its dtype check as a status, because `options()` can switch
  accuracy tracking on after `make`.
- `run_step`: the two element-count checks as `SA_FAIL_IF` with `StatusCode::Incompatible`; the
  two copies through `SushiBLAS::BLASOps blas(*engine_);` inside
  `Detail::guard(Detail::SeamPhase::Record, ...)`; `SA_TRY(Detail::run_recorded(*engine_));
  SA_TRY(step_->run());`; the optimizer's `step()` under `SA_TRY` when the step holds no
  trailing work.
- `fit`: three checks as `SA_FAIL_IF` (`:280` `InvalidDescription`; `:281`, `:284`
  `Incompatible`); `SA_TRY_ASSIGN(const StepMetrics metrics, run_step(dataset.batch(index)));`.
- `bind_parameters`: `SA_TRY(pool.adopt(...))`, `SA_TRY(pool.materialise_leaves(...))`,
  `SA_TRY_ASSIGN(SushiBLAS::Tensor& gradient, pool.materialise(...))`,
  `SA_TRY(pool.flush_constants())`; the site at `:63` is `SA_FAIL_IF`.
- `build_graph` and `build_classifier` do not change.
- `SushiAI.h` gains `#include "core/seam.hpp"`, `"graph/tensor_pool.hpp"` and
  `"graph/lowering/all.hpp"`, beside the lines of their folders.

**Acceptance:** no `SA_THROW_IF` in `trainer.cpp`; `Trainer` has no public constructor.

### M28 and M29. Applications and examples

**Files:** `apps/train.cpp`, `apps/inference.cpp` (M28); `examples/example.cpp`,
`examples/demo_mlp.cpp` (M29). **Wait on:** M27 and every unit before it.
**Matrix row:** many files.

- The body of each `main` moves into `[[nodiscard]] sa::Status run(int argc, char** argv)` in
  the same file. Each migrated call takes its `Result` or `Status` form with `SA_TRY` or
  `SA_TRY_ASSIGN`. `main` calls `run` inside the `try` it has today, prints
  `"<program> failed: " << status.message()` on a failed status, and returns `EXIT_FAILURE`. The
  `catch` stays until `Config` and the builder stop throwing (rule 5).
- An application or example is not library surface: its `SA_THROW_IF` sites become
  `SA_FAIL_IF(..., sa::StatusCode::InvalidDescription, ...)` inside `run` or a helper that
  returns `Status` or `Result`. `examples/demo_mlp.cpp`'s three `catch (const std::logic_error&)`
  blocks around `std::stoll` and its relatives stay; they catch the standard library.
- The first line an application prints on a failure does not change;
  `cli/tests/test_train.py:81-103` matches it.

**Acceptance:** no `SA_THROW_IF` in the four files; each program still prints `compiles: 1` on
the path that prints it today.

### M30. Tests

**Files:** the test files below. **Waits on:** every library unit. **Matrix row:** judgment for
the four files marked so; many files for the rest. Each file is one implementer's.

An expectation changes only when the site it provokes is in a file this migration converted. A
status-class expectation becomes `SA_EXPECT_FAILURE(expression, StatusCode::X, needles...)`, a
fatal-class one `SA_EXPECT_FATAL(statement, needles...)`, and a passing call that now returns a
`Result` becomes `SA_ASSERT_OK_AND_ASSIGN(declaration, expression)`
(`tests/common/test_status.hpp`). The class of a site is its row in
`tools/contract/site_classes.tsv`. An expectation whose site is in `Config`, the builder or the
differentiator keeps `SA_EXPECT_ERROR`.

| File | Edit | Row |
| --- | --- | --- |
| `tests/package/main.cpp`, `tests/functional/integration/test_engine_smoke.cpp` | The `axpy` spelling of section 2.4; include `<SushiBLAS/engine/blas.hpp>` where the umbrella is not included | many |
| `tests/functional/unit/test_seam.cpp` (new) | One case per row of M1's table: a callable that throws each type under each phase yields the mapped code and the `what()` text. `SushiBLAS::UnsupportedExecution` is caught before `std::invalid_argument`. The two fatal rows are `SA_EXPECT_FATAL` | many |
| `tests/functional/unit/test_lowering_table.cpp` (new) | Device-free. `find_lowering` returns non-null for every identity of `Ops::K_ALL` outside `K_UNLOWERED` and null for the three inside; no identity has two rows; each domain table has the size of section 4.3 | many |
| `tests/functional/unit/test_step_recorder.cpp` | The fakes take and ignore the recorder. A fixture builds one `SushiRuntime::Graph::TaskGraph`, one `SushiBLAS::ScratchPool` and one `SushiBLAS::TaskRecorder` over `SaTest::TestEnvironment::context()` and passes it. The two `add` expectations become `SA_EXPECT_FATAL`. The suite constructs runtime objects from now on | many |
| `tests/functional/regression/test_fusion_numerics.cpp` | The two ordering tests (`:481-515`, `:521-565`) build their own `TaskGraph` and `TaskRecorder`, record the bias write with `SushiBLAS::ElementwiseOps(recorder).mul`, lower with `Lowering(recorder, engine.get_rng(), pool, selector)`, and read `graph.take_nodes()`. `Lowering` takes a selector reference where it took a pointer | judgment |
| `tests/functional/regression/test_fused_optimizer.cpp` | `AdamW::make`, `SGD::make`; `record_into_step(engine.get_recorder())` at `:382`; `step()` under `SA_EXPECT_OK` | judgment |
| `tests/functional/integration/test_resilience.cpp` | 19 expectations; the `Runner` helper holds a `std::unique_ptr<CompiledStep>` | judgment |
| `tests/functional/integration/test_forward.cpp` | `OneShot` holds a `std::unique_ptr<CompiledStep>`; 6 expectations. Its batch-normalisation cases now expect four results | judgment |
| `tests/functional/integration/test_arena.cpp`, `test_autograd_numeric.cpp`, `test_checkpoint.cpp`, `test_mixed_precision.cpp`, `test_registries.cpp`, `test_rematerialise.cpp`, `test_training.cpp` | Factories for `CompiledStep`, `Trainer`, the optimizers, the datasets and `LossScaler`; `SA_ASSERT_OK_AND_ASSIGN` or `SA_EXPECT_OK` on every call that returns a `Result` or a `Status` | many |
| `tests/functional/regression/test_compile_once.cpp`, `test_determinism.cpp`, `test_learning_curve.cpp`, `test_log_softmax_stability.cpp`, `test_loss_scaling.cpp`, `test_mixed_precision.cpp`, `test_static_memory.cpp` | The same | many |
| `tests/functional/unit/test_checkpoint_format.cpp` | `CheckpointWriter::create`, `CheckpointReader::open`, `next()` and `read_payload` as results | many |

**Acceptance, each file:** it names no facade accessor, no `get_graph` and no public constructor
of `CompiledStep`, `Trainer`, `AdamW`, `SGD`, `Rings`, `SyntheticClassification`, `LossScaler`,
`StepScalars`, `CheckpointReader` or `CheckpointWriter`.

### M31 to M34. The model graph's engine units

`docs/agent/2026_10_07_MODEL_GRAPH/PLAN.md:94-100` lists five units that wait for this migration.
U13 is M20. Items 1 and 2 of U14 are in M12. The rest need `NN::ModelState`, a new public type
that `docs/agent/2026_10_07_MODEL_GRAPH/SPEC.md:932-1049` designs. They are written from that
design, against the surface this document leaves. Whether they run as the last wave of this work
is question Q8.

| Unit | Model-graph unit | Files | What this migration changes in its design |
| --- | --- | --- | --- |
| M31 | U14, item 3 | `include/SushiAI/nn/model_state.hpp`, `src/nn/model_state.cpp` (new); `nn/init.hpp`, `src/nn/init.cpp` | `ModelState::create` allocates through M5. `initialize(engine, state, policy)` returns `Status` and fills a buffer through `NN::constant` |
| M32 | U15 | `include/SushiAI/io/checkpoint.hpp`; `include/SushiAI/io/import_weights.hpp`, `src/io/import_weights.cpp` (new) | Both functions return `Status` over M4's reader and writer |
| M33 | U16 | `train/objective.hpp`, `src/train/objective.cpp`, `train/trainer.hpp`, `src/train/trainer.cpp`; `train/steps.hpp`, `train/buffer_update_recorder.hpp` and sources (new) | `BufferUpdateRecorder` derives from `Graph::StepRecorder` and implements `Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder& recorder)` with `SushiBLAS::BLASOps blas(recorder);` (`scal`, then `axpy`). It takes no `Engine&` |
| M34 | U17 | `data/dataset.hpp` and sources, `apps/train.cpp`, `apps/inference.cpp`, `examples/mnist_cnn/model.json`, `nn/module.hpp`, `config/model_config.hpp` | The applications already have the `run` function of M28 |

All four are judgment units.

## 4. The lowering table

### 4.1 The row type

```cpp
// include/SushiAI/graph/lowering/rule.hpp, namespace SushiAI::Graph
// includes: <array>, <cstddef>, <SushiAI/core/ops.hpp>, <SushiAI/core/status.hpp>
class LoweringContext;

/** @brief Records the SushiBLAS work of one node and reports a refusal as a status. */
using LoweringRule = Status (*)(const LoweringContext&);

/** @brief Pairs an operation with the function that lowers it. */
struct LoweringEntry
{
    OpID op;            /**< @brief The operation. */
    LoweringRule rule;  /**< @brief What records it. */
};

/** @brief Joins two lowering tables into one, @p a first. */
template<std::size_t N, std::size_t M>
constexpr std::array<LoweringEntry, N + M> concat_lowerings(
    const std::array<LoweringEntry, N>& a, const std::array<LoweringEntry, M>& b)
{
    std::array<LoweringEntry, N + M> out{};
    for (std::size_t i = 0; i < N; ++i) out[i] = a[i];
    for (std::size_t i = 0; i < M; ++i) out[N + i] = b[i];
    return out;
}

// include/SushiAI/graph/lowering/unlowered.hpp, namespace SushiAI::Graph
// includes: <array>, <SushiAI/core/ops.hpp>
/** @brief Lists the identities recorded beside the graph, which no node carries. */
inline constexpr std::array<OpID, 3> K_UNLOWERED = {{
    Ops::K_ADAMW_UPDATE.id,
    Ops::K_SGD_UPDATE.id,
    Ops::K_BATCHNORM_UPDATE_STATS.id,
}};
```

The row type is the gradient rule's (`include/SushiAI/autograd/rule.hpp:40-65`) with one
difference: a row returns `Status`. A gradient rule writes IR and marks a refusal in the graph; a
lowering row has no graph to mark and its caller is the function that reports.

### 4.2 How a row reaches the table

Nothing registers itself. `sushiai` is a static library, and a translation unit whose only work
is a registration constructor resolves no symbol, so a consumer's linker drops it
(`docs/architecture/ARCHITECTURE.md:239-248`). A row reaches the table by being named in its
domain's `constexpr` array, and `all.hpp` joins the arrays:

```cpp
// include/SushiAI/graph/lowering/all.hpp, namespace SushiAI::Graph
// includes: <cstddef>, <SushiRuntime/core/span.hpp>, <SushiAI/core/ops.hpp>, rule.hpp,
// unlowered.hpp and the eight domain headers, in the order of the concatenation below
inline constexpr auto K_LOWERINGS = concat_lowerings(
    concat_lowerings(
        concat_lowerings(K_ELEMENTWISE_LOWERINGS, K_ACTIVATION_LOWERINGS),
        concat_lowerings(K_LINALG_LOWERINGS, K_LOSS_LOWERINGS)),
    concat_lowerings(
        concat_lowerings(K_SHAPE_LOWERINGS, K_SPATIAL_LOWERINGS),
        concat_lowerings(K_NORM_LOWERINGS, K_FUSED_LOWERINGS)));

[[nodiscard]] constexpr LoweringRule find_lowering(OpID op) noexcept
{
    for (const LoweringEntry& entry : K_LOWERINGS)
        if (entry.op == op) return entry.rule;
    return nullptr;
}

[[nodiscard]] constexpr bool is_unlowered(OpID op) noexcept
{
    for (const OpID listed : K_UNLOWERED)
        if (listed == op) return true;
    return false;
}

[[nodiscard]] constexpr std::size_t lowering_row_count(OpID op) noexcept
{
    std::size_t rows = 0;
    for (const LoweringEntry& entry : K_LOWERINGS)
        if (entry.op == op) ++rows;
    return rows;
}

[[nodiscard]] constexpr bool every_identity_is_accounted_for(
    SushiRuntime::span<const Ops::OpName> table) noexcept
{
    for (const Ops::OpName& entry : table)
        if (lowering_row_count(entry.id) + (is_unlowered(entry.id) ? 1u : 0u) != 1u)
            return false;
    return true;
}

static_assert(every_identity_is_accounted_for(
                  SushiRuntime::span<const Ops::OpName>(Ops::K_ALL.data(), Ops::K_ALL.size())),
              "An identity of core/ops.hpp has no lowering row, has two, or has a row and is "
              "also listed in unlowered.hpp. A graph that held it would stop at "
              "Lowering::record with no hint at build time.");

static_assert(K_LOWERINGS.size() + K_UNLOWERED.size() == Ops::K_ALL.size(),
              "The lowering tables name an operation that core/ops.hpp does not declare.");
```

Who edits what:

| Change | Files edited |
| --- | --- |
| A new operation in an existing domain | That domain's `.hpp` (one declaration, one row, the array's size) and `.cpp` (one function). No other file |
| A new operation recorded beside the graph | `unlowered.hpp`, one line |
| A new domain | Its two new files, and `all.hpp`: one include and one `concat_lowerings` argument. `all.hpp` is the only file a new domain edits |
| Forgetting either | The first `static_assert` fails in this repository's build |

`find_lowering` is a linear search over 62 rows, run once per node at compile time of the step,
never during a replay.

### 4.3 Operation, domain, calls

`Ops::K_ALL` holds 65 identities (`include/SushiAI/core/ops.hpp:292-313`). 62 have a row and 3
are in `K_UNLOWERED`. 48 are lowered today; the 14 marked **new** have no lowering today.
"In/out" is the argument of `require_arity`. `aN` is `node.attributes.get(N)`, `fN` is
`node.attributes.flag(N)`. Views: `blas` is `BLASOps`, `ew` is `ElementwiseOps`, `nl` is
`NonLinearOps`, `red` is `ReductionOps`, `sp` is `SpatialOps`, `lg` is `LogicOps`, `rnd` is
`RandomOps`. Operands are named in node order: inputs `x0, x1, ...`, results `y0, y1, ...`.

**`elementwise`, `K_ELEMENTWISE_LOWERINGS`, 17 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `ADD` | 2/1 | `ew.add(x0, x1, y0)` | 1 |
| `SUB` | 2/1 | `ew.sub(x0, x1, y0)` | 1 |
| `MUL` | 2/1 | `ew.mul(x0, x1, y0)` | 1 |
| `DIV` **new** | 2/1 | `ew.div(x0, x1, y0)` | 1 |
| `MINIMUM` **new** | 2/1 | `ew.min(x0, x1, y0)` | 1 |
| `MAXIMUM` **new** | 2/1 | `ew.max(x0, x1, y0)` | 1 |
| `SCALE` | 1/1 | `blas.copy(x0, y0)`; `blas.scal(static_cast<float>(a0), y0)` | 2 |
| `DYNAMIC_SCALE` | 2/1 | `blas.copy(x0, y0)`; `Dispatch::execute_unary_inplace(recorder, y0, "sushiai.dynamic_scale", Ops::K_DYNAMIC_SCALE.id, bind_reads(fn, read_as_element(x1)))` | 2 |
| `EXP` | 1/1 | `blas.copy(x0, y0)`; `ew.exp(y0)` | 2 |
| `ABS` **new** | 1/1 | `blas.copy(x0, y0)`; `ew.abs(y0)` | 2 |
| `ATAN` **new** | 1/1 | `blas.copy(x0, y0)`; `ew.atan(y0)` | 2 |
| `LESS_EQUAL` **new** | 2/1 | `lg.less_equal(x0, x1, y0)` | 1 |
| `SELECT` **new** | 3/1 | `lg.where(x0, x1, x2, y0)` | 1 |
| `ZERO` | 0/1 | `rnd.constant(y0, 0.0)` | 1 |
| `CONSTANT` **new** | 0/1 | `rnd.constant(y0, a0)` | 1 |
| `ACCUMULATE` | 2/1 | alias `y0` onto `x0`; `blas.axpy(1.0f, x1, x0)` | 1 |
| `CAST` | 1/1 | `ew.convert(x0, y0)` | 1 |

**`activation`, `K_ACTIVATION_LOWERINGS`, 14 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `RELU`, `GELU`, `SIGMOID`, `TANH` | 1/1 | `blas.copy(x0, y0)`; `nl.relu(y0)`, `nl.gelu(y0)`, `nl.sigmoid(y0)`, `nl.tanh(y0)` | 2 |
| `SILU` **new**, `SOFTPLUS` **new** | 1/1 | `blas.copy(x0, y0)`; `nl.silu(y0)`, `nl.softplus(y0)` | 2 |
| `LEAKY_RELU` | 1/1 | `blas.copy(x0, y0)`; `nl.leaky_relu(y0, static_cast<float>(a0))` | 2 |
| `RELU_BACKWARD`, `GELU_BACKWARD`, `SIGMOID_BACKWARD`, `TANH_BACKWARD` | 2/1 | `nl.relu_backward(x0, x1, y0)` and its three siblings | 1 |
| `SILU_BACKWARD` **new**, `SOFTPLUS_BACKWARD` **new** | 2/1 | `nl.silu_backward(x0, x1, y0)`, `nl.softplus_backward(x0, x1, y0)` | 1 |
| `LEAKY_RELU_BACKWARD` | 2/1 | `nl.leaky_relu_backward(x0, x1, y0, static_cast<float>(a0))` | 1 |

Row order in the table: the seven forward operations in the order of the first three table rows
above, then the seven gradients in the order of the last three.

**`linalg`, `K_LINALG_LOWERINGS`, 3 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `MATMUL` | 2/1 | `blas.gemm(x0, x1, y0, transpose_of(f0), transpose_of(f1), 1.0, 0.0)` | 1 |
| `BIAS_ADD` | 2/1 | fatal unless `x0` is rank 2; `ones([rows])`; `blas.copy(x0, y0)`; `blas.ger(ones, x1, y0, 1.0f)` | 2 |
| `SUM_ROWS` | 1/1 | fatal unless `x0` is rank 2; `ones([rows])`; `blas.gemv(x0, ones, y0, Transpose::Trans, 1.0f, 0.0f)` | 1 |

**`loss`, `K_LOSS_LOWERINGS`, 2 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `LOG_SOFTMAX` | 1/1 | fatal unless rank 2. Scratch 0 `row_max [rows]`, 1 `shifted`, 2 `exponents`, 3 `row_sums [rows]`; `ones([cols])`. `red.max_along(x0, 1, row_max)`; `blas.copy(x0, shifted)`; `blas.ger(row_max, ones, shifted, -1.0f)`; `blas.copy(shifted, exponents)`; `ew.exp(exponents)`; `blas.gemv(exponents, ones, row_sums, Transpose::NoTrans, 1.0f, 0.0f)`; `ew.log(row_sums)`; `blas.copy(shifted, y0)`; `blas.ger(row_sums, ones, y0, -1.0f)` | 9 |
| `NLL_LOSS` | 2/1 | fatal unless rank 2. Scratch 0 `picked`. `ew.mul(x0, x1, picked)`; `red.sum(picked, y0)`; `blas.scal(-1.0f / static_cast<float>(rows), y0)` | 3 |

**`shape`, `K_SHAPE_LOWERINGS`, 4 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `LEAF` | 0/any | none; fatal when a result is not bound | 0 |
| `RESHAPE` | 1/1 | fatal unless `x0.is_contiguous()`; a `Tensor` over `x0.storage` with the result's extents, as `src/graph/lowering.cpp:944-949`; `blas.copy(view, y0)` | 1 |
| `LAYOUT_CONVERT` | 1/1 | the two transposed views of `:967-970`; `sp.layout_convert(source_view, destination_view)` | 1 |
| `STOP_GRADIENT` **new** | 1/1 | `blas.copy(x0, y0)` | 1 |

**`spatial`, `K_SPATIAL_LOWERINGS`, 12 rows.** `g` is `blas_geometry_of(node)`; `oh`, `ow` come
from `windowed_extents_of`.

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `IM2COL` | 1/1 | `sp.im2col(x0, y0, g, oh, ow)` | 1 |
| `COL2IM` | 1/1 | `sp.col2im(x0, y0, g, oh, ow)` | 1 |
| `MAXPOOL2D` | 1/2 | `sp.pool_max(x0, y0, y1, g)` | 1 |
| `MAXPOOL2D_BACKWARD` | 2/1 | `sp.pool_max_backward(x0, x1, y0, g)` | 1 |
| `AVGPOOL2D` | 1/1 | `sp.pool_avg(x0, y0, g)` | 1 |
| `AVGPOOL2D_BACKWARD` | 1/1 | `sp.pool_avg_backward(x0, y0, g)` | 1 |
| `UPSAMPLE2D` | 1/1 | `sp.upsample_nearest(x0, y0, int64(a0), int64(a1))` | 1 |
| `UPSAMPLE2D_BACKWARD` | 1/1 | `sp.upsample_nearest_backward(x0, y0, int64(a0), int64(a1))` | 1 |
| `SPATIAL_MEAN` | 1/1 | `red.mean_spatial(x0, y0)` | 1 |
| `SPATIAL_MEAN_BACKWARD` | 1/1 | fatal unless the result is rank 4; `sp.broadcast_spatial(x0, y0, 1.0 / (H * W))` | 1 |
| `CONCAT_CHANNELS` | at least 1/1 | for each operand `i`: `sp.concat_channels(xi, y0, offset)`; `offset += shape(xi)[3]` | operands |
| `SLICE_CHANNELS` | 1/1 | `sp.slice_channels(x0, int64(a0), y0)` | 1 |

**`norm`, `K_NORM_LOWERINGS`, 4 rows.** `rows` is `N*H*W`; `inv` is `1.0 / rows`; `C` is the
channel count; `x_rows` is `channel_rows(x, rows, C)`.

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `BATCHNORM2D` | 3/4 | Operands `x`, `gamma`, `beta`; results `y`, `mean`, `inv_std`, `variance`. Fatal unless `x` is rank 4 and contiguous. `ones([C])`; scratch 0 `negated_mean [C]`, 1 `centred`, 2 `squared`, 3 `scale [C]`. `red.sum_along(x_rows, 0, mean)`; `blas.scal(inv, mean)`; `blas.copy(mean, negated_mean)`; `blas.scal(-1.0, negated_mean)`; `sp.channel_affine(x, ones, negated_mean, centred)`; `ew.mul(centred, centred, squared)`; `red.sum_along(squared_rows, 0, variance)`; `blas.scal(inv, variance)`; `blas.copy(variance, inv_std)`; `blas.axpy(a0, ones, inv_std)`; `ew.sqrt(inv_std)`; `ew.reciprocal(inv_std)`; `ew.mul(gamma, inv_std, scale)`; `sp.channel_affine(centred, scale, beta, y)` | 14 |
| `BATCHNORM2D_BACKWARD` | 5/3 | The 15 calls of `src/graph/lowering.cpp:1348-1372`, unchanged, with scratch slots 0 to 7 as today | 15 |
| `CHANNEL_AFFINE` | 3/1 | `sp.channel_affine(x0, x1, x2, y0)` | 1 |
| `BATCHNORM_FOLD` **new** | 4/2 | Operands `gamma`, `beta`, `mean`, `variance`; results `scale`, `shift`; all `[C]`. `ones([C])`; scratch 0 `inv_std [C]`, 1 `product [C]`. `blas.copy(variance, inv_std)`; `blas.axpy(a0, ones, inv_std)`; `ew.sqrt(inv_std)`; `ew.reciprocal(inv_std)`; `ew.mul(gamma, inv_std, scale)`; `ew.mul(mean, scale, product)`; `ew.sub(beta, product, shift)` | 7 |

`BATCHNORM2D` differs from today in three lines: it takes four results, the variance is result 3
and no longer a scratch, and epsilon is added to the copy in `inv_std`. `inv_std` receives the
same sum as today, so results 0 to 2 keep their values. `BATCHNORM_FOLD` writes
`scale = gamma / sqrt(variance + epsilon)` and `shift = beta - mean * scale`
(`include/SushiAI/graph/builder.hpp:135-140`).

**`fused`, `K_FUSED_LOWERINGS`, 6 rows.**

| Operation | In/out | Calls, in order | Tasks |
| --- | --- | --- | --- |
| `GEMM_BIAS`, `GEMM_BIAS_RELU`, `GEMM_BIAS_SIGMOID`, `GEMM_BIAS_TANH` | 3/1 | `record_fused_gemm` with the form's epilogue and tail (`nullptr`, `&NonLinearOps::relu`, `&NonLinearOps::sigmoid`, `&NonLinearOps::tanh`) | 1 to 3 |
| `GEMM_BIAS_GELU` | 3/2 | the same, with `BiasGeluSpillEpilogue`, `.reads(bias).writes(aux)` and tail `&NonLinearOps::gelu`; a form that did not spill adds `blas.copy(c, aux)` | 1 to 4 |
| `EXP_SUB_SCALE` | 2/1 | fused: `Dispatch::execute_zip_inplace(recorder, {&y0, &x0, &x1}, ...)`; unfused: `blas.copy(x0, y0)`; `ew.exp(y0)`; `ew.sub(y0, x1, y0)`; `blas.scal(alpha, y0)` | 1 or 4 |

**`K_UNLOWERED`, 3 identities:** `ADAMW_UPDATE` and `SGD_UPDATE`, recorded by the optimizer
(M22, M23), and `BATCHNORM_UPDATE_STATS`, recorded by the buffer-update recorder (M33).

Three checks behind the new rows, read in SushiBLAS: `less_equal` writes its mask in the
operands' element type, and `where` requires its condition, both branches and its result to
share one element type (`sushiblas: include/SushiBLAS/engine/logic/dispatch.hpp:118-123`,
`:173-181`), which is what `SELECT`'s shape rule gives it; `constant` has a `Half` arm
(`sushiblas: src/engine/math/random/constant.cpp:93-95`).

## 5. Status conversion

The migrated files hold 167 of the table's sites: 84 become statuses and 83 become fatal checks.
`apps/` and `examples/` hold 23 more, all unclassified in the table and all statuses here.

| File | Sites | Status | Fatal |
| --- | --- | --- | --- |
| `src/graph/lowering.cpp` (to M6, M10, M14 to M21, M24) | 57 | 1 | 56 |
| `src/optim/*.cpp` | 32 | 24 | 8 |
| `src/io/checkpoint_format.cpp` | 21 | 15 | 6 |
| `src/data/dataset.cpp` | 18 | 15 | 3 |
| `src/train/trainer.cpp` | 12 | 12 | 0 |
| `src/core/allocation.cpp` | 10 | 5 | 5 |
| `include/SushiAI/io/checkpoint.hpp` | 9 | 8 | 1 |
| `src/nn/init.cpp` | 4 | 4 | 0 |
| `src/graph/arena.cpp`, `include/SushiAI/graph/step_recorder.hpp` | 4 | 0 | 4 |

Functions whose return type changes:

| Function | Today | After | Failure codes |
| --- | --- | --- | --- |
| `allocate_tensor` (two overloads), `allocate_like` | `Tensor` | `Result<Tensor>` | `InvalidDescription` for extents, `OutOfMemory` |
| `allocate_arena` | `sushi_ptr<Storage>` | `Result<sushi_ptr<Storage>>` | `OutOfMemory` |
| `ArenaSet` construction | constructor | `Result<ArenaSet> make` | `OutOfMemory` |
| `TensorPool::adopt`, `materialise_leaves`, `flush_constants` | `void` | `Status` | `OutOfMemory`; the seam's codes |
| `TensorPool::create` | `Tensor` | `Result<Tensor>` | `OutOfMemory` |
| `TensorPool::materialise`, `ones`, `scratch` | `Tensor&` | `Result<Tensor&>` | `OutOfMemory` |
| `StepRecorder::record_into_step` and its overriders | `std::size_t` | `Result<std::size_t>` | the seam's codes |
| `Lowering::record` | `void` | `Status` | the seam's codes; `InvalidDescription` from `lower_dynamic_scale` |
| `CompiledStep` construction, `run` | constructor, `void` | `Result<std::unique_ptr<CompiledStep>> compile`, `Status` | `InvalidDescription` for a refused graph; the seam's codes |
| `StepScalars`, `LossScaleState`, `LossScaler`, `AdamW`, `SGD` construction | constructors | `make`, returning `Result<T>` for the first two and `Result<std::unique_ptr<T>>` for the others | `InvalidDescription` for options and slots; `OutOfMemory` |
| `LossScaler::record_screens` | `std::size_t` | `Result<std::size_t>` | the seam's codes |
| `Optimizer::step`, `set_loss_scaler` | `void` | `Status` | the seam's codes; `InvalidDescription` |
| `make_adamw`, `make_sgd`, `make_optimizer` | `unique_ptr<Optimizer>` | `Result<unique_ptr<Optimizer>>` | as `make` |
| `kaiming_normal`, `kaiming_uniform`, `xavier_normal`, `xavier_uniform`, `zeros`, `Detail::apply_initializer`, `initialize` | `void` | `Status` | `InvalidDescription` for fans; the seam's codes |
| `CheckpointWriter`, `CheckpointReader` construction | constructors | `Result<T> create`, `Result<T> open` | `InvalidFile`; `InvalidDescription` for the tensor count |
| `CheckpointWriter::write`, `CheckpointReader::read_payload` | `void` | `Status` | `InvalidFile`; `InvalidDescription` for a name |
| `CheckpointReader::next` | `TensorRecord` | `Result<TensorRecord>` | `InvalidFile` |
| `save_checkpoint` | `void` | `Status` | `InvalidFile` for a path that does not open or a failed write |
| `load_checkpoint` | `void` | `Status` | `FileNotFound` for a path that does not open; `Incompatible` for an unknown, repeated, mistyped, misshapen or missing parameter (`:157-181`) |
| `SyntheticClassification`, `Rings` construction | constructors | `Result<unique_ptr<T>> make` | `InvalidDescription`; `OutOfMemory` |
| `try_load_mnist`, `make_rings`, `make_mnist`, `make_dataset` | `unique_ptr<Dataset>` | `Result<unique_ptr<Dataset>>` | `FileNotFound`, `InvalidFile`, `InvalidDescription` |
| `bind_parameters` | `vector<ParameterSlot>` | `Result<vector<ParameterSlot>>` | `InvalidDescription`; `OutOfMemory` |
| `Trainer` construction | constructor | `Result<unique_ptr<Trainer>> make` | `InvalidDescription` |
| `Trainer::run_step`, `fit` | `StepMetrics`, `RunReport` | `Result<StepMetrics>`, `Result<RunReport>` | `Incompatible` for a batch or a dataset that does not fit the step; the seam's codes |

Functions that keep their type because every site in them is fatal: `TensorPool::tensor`,
`bind`, `alias`, `arenas`; `ArenaSet::view`; `view_tensor`; `StepRecorderList::add`;
`StepScalars::write`; `AdamW::first_moment`, `second_moment`; `Dataset::batch`;
`CheckpointWriter::finish`; `IO::Detail::bind`; `Trainer::current_loss`.

Sixty-seven of these rows carry an `open_question` key in the table
(`docs/agent/2026_10_07_ERROR_CONTRACT_AND_NAMING/PLAN.md:542-596`). This design converts each to
the mechanism its row states and settles the five keys that decide a signature:

| Key | Settled as |
| --- | --- |
| `slot_validation_home` | The free function `Optim::validate_slots`, called first by every `make` (M11) |
| `make_by_spec_2_6` | `StepScalars::make` refuses a zero count; `Trainer::make` refuses an unset `StepSpec` field |
| `separate_kernel_source` | `set_loss_scaler` returns `Status` |
| `dtype_check_once` | `Trainer::make` checks the loss dtype once; `current_loss` returns `double`. The logits check stays per call |
| `init_fans` | The four public fill functions return `Status` |

When a unit's wave lands, the orchestrator deletes the rows of its sites from
`tools/contract/site_classes.tsv` in the same commit and runs
`python tools/contract/check_contract.py`, as `tools/contract/README.md:77-79` asks. That checker
reads source text and builds nothing.

## 6. Waves

The units of one wave have disjoint files and wait on nothing in their own wave.

| Wave | Units | Files |
| --- | --- | --- |
| 1 | M1, M2, M3, M4 | `core/seam.hpp`, `src/core/seam.cpp`; `graph/step_recorder.hpp`; `graph/lowering/rule.hpp`, `unlowered.hpp`; `io/checkpoint_format.hpp`, `src/io/checkpoint_format.cpp` |
| 2 | M5 | `core/allocation.hpp`, `src/core/allocation.cpp` |
| 3 | M6, M7, M8, M9 | `graph/arena.hpp`, `src/graph/arena.cpp`, `graph/tensor_pool.hpp`, `src/graph/tensor_pool.cpp`; `optim/step_scalars.*`; `optim/loss_scaler.*`; `data/dataset.hpp`, `src/data/dataset.cpp` |
| 4 | M10, M11, M12, M13 | `graph/lowering/context.hpp`, `src/graph/lowering/context.cpp`; `optim/optimizer.*`; `nn/init.hpp`, `src/nn/init.cpp`; `io/checkpoint.hpp` |
| 5 | M14 to M21, M22, M23 | the eight pairs under `graph/lowering/` and `src/graph/lowering/`; `optim/adamw.*`; `optim/sgd.*` |
| 6 | M24, M25 | `graph/lowering/all.hpp`, `graph/lowering.hpp`, `src/graph/lowering.cpp`; `optim/optimizer_registry.*` |
| 7 | M26 | `graph/compiled_step.hpp`, `src/graph/compiled_step.cpp` |
| 8 | M27 | `train/trainer.hpp`, `src/train/trainer.cpp`, `SushiAI.h` |
| 9 | M28, M29, M30 | `apps/*.cpp`; `examples/*.cpp`; each test file of M30 |
| 10 | M31 to M34, in that order | section 3; M33 and M34 touch files of M27 and M28 |

Between waves 3 and 6 two headers define `TensorPool`: the new `graph/tensor_pool.hpp` and the
old `graph/lowering.hpp`. No file a unit of those waves writes includes both. `trainer.cpp`,
`compiled_step.cpp`, the applications and the tests do not pass a syntax check until their wave.
Wave 5 is the widest, with ten implementers.

## 7. What the owner's build must confirm

**Lines for the owner to add, since build files are the owner's.** `src/CMakeLists.txt`,
`SUSHIAI_SOURCES`: `core/seam.cpp`, `graph/tensor_pool.cpp`, `graph/lowering/context.cpp`,
`graph/lowering/elementwise.cpp`, `activation.cpp`, `linalg.cpp`, `loss.cpp`, `shape.cpp`,
`spatial.cpp`, `norm.cpp`, `fused.cpp`. `tests/CMakeLists.txt`:
`functional/unit/test_seam.cpp`, `functional/unit/test_lowering_table.cpp`.

1. **The tree compiles and links against SushiBLAS `a977291`.** Every signature here is unchecked
   text. The constructs most likely to need a correction: a pointer to a member function of a
   view class bound to a `using` alias; `SA_TRY_ASSIGN` over a `Result<SushiBLAS::Tensor&>`;
   `return std::move(local);` into a `Result<std::unique_ptr<Base>>`; the lambda handed to
   `bind_reads` under `execute_unary_inplace`.
2. **The two `static_assert`s of `all.hpp` hold**, with 62 rows and 3 unlowered identities.
3. **Compile-once and steady state.** `CompiledStep::compile_count()` is 1 and
   `TensorPool::allocated_bytes()` stops moving after the first step
   (`tests/functional/regression/test_compile_once.cpp`, `test_static_memory.cpp`).
4. **The task count of a step does not change.** No row's count differs from today's, so
   `operations_per_step` of `sa demo mlp` stays what it is.
5. **A plan outlives nothing it needs.** `CompiledStep` destroys `plan_`, then `recorder_`, then
   `graph_`. Whether a `CompiledGraph` reads its `TaskGraph` after `compile()` was not traced.
6. **A recorded task retains its operands' storage**, so a kept step pins its arenas and its
   scratch. That is new; the byte counters should not move, and the first build shows whether
   they do.
7. **The ordering test on in-degree** passes with a `TaskGraph` the test owns
   (`Regression_Fusion.AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias`).
8. **Fourteen lowerings that have never run**: `DIV`, `MINIMUM`, `MAXIMUM`, `ABS`, `ATAN`,
   `LESS_EQUAL`, `SELECT`, `CONSTANT`, `SILU`, `SOFTPLUS`, their two gradients, `STOP_GRADIENT`
   and `BATCHNORM_FOLD`; and the fourth result of `BATCHNORM2D` in the batch-normalisation cases
   of `test_forward.cpp` and `test_autograd_numeric.cpp`.
9. **The declared loss-scale factor.** `lower_dynamic_scale` reads the factor as the node's own
   element type, and SushiBLAS now refuses a factor of another type. The loss-scaling suite shows
   whether any graph relied on the old unchecked read.
10. **Pinned numbers may move for reasons outside this migration.** `max_along` now propagates a
    NaN (`sushiblas: docs/agent/2026_10_07_V010/SPEC.md:462-466`); SushiBLAS reseeds by rewinding
    its counter (`docs/design/REMAINING_WORK.md:19-21`); the reference GEMM sums in double under
    `ForceReference`. `test_learning_curve.cpp`, `test_determinism.cpp` and
    `test_log_softmax_stability.cpp` are the suites to read first.
11. **`ForceReference`.** Every family now runs on the host under it. A HALF value computed that
    way and compared with the device's allows one HALF unit in the last place below `2^-13` and
    bit equality above it (`docs/design/REMAINING_WORK.md:99-103`).
12. **`DeviceFusionSelector` under `ForceReference`** still declines every GEMM epilogue
    (`src/graph/fusion_selector.cpp:57-60`). See Q7.
13. **Death tests.** The fatal-class expectations of M30 start one child process each on Windows.
    Their start-up time is the condition of decision D7 and has not been measured.
14. **`Unit_StepRecorder`** builds a `TaskGraph` and a `ScratchPool` and so needs the runtime
    context. The seam keeps the `wait_all()` after each `execute()` that the tree calls today,
    though `IExecutionEngine::execute` is documented as blocking.
15. **A failed recording leaves nothing behind in a step's graph.** `test_seam.cpp` can pin the
    mapping only; that `take_nodes()` releases what the recorder retained needs a run.
16. **`python tools/contract/check_contract.py`** passes after the rows of the converted sites
    are deleted and the `K_FROZEN` list no longer names the migrated files.

## 8. Questions for the owner

Each is a choice of module boundary, public interface or data ownership. The units above are
written under the recommendation.

**Decided 2026-10-08**, by the engineer the owner delegated these to: every question takes its
recommendation. Q8 puts M31 to M34 in the next work, so this work ends with wave 9. The build
lists are edited by the orchestrator, not by a unit.

| # | Question | Options | Recommendation |
| --- | --- | --- | --- |
| Q1 | Where do the lowering headers live? | (a) Public, under `include/SushiAI/graph/lowering/`, as the gradient rules are under `include/SushiAI/autograd/rules/`. (b) Private, under `src/graph/lowering/`, with only `lowering.hpp` public | (a). A consumer that adds an operation needs the row type, and the table test needs the tables. It makes `LoweringContext`, `LoweringEntry` and 62 row functions public names |
| Q2 | Where is an identity with no lowering declared? | (a) `graph/lowering/unlowered.hpp`, a list of three. (b) A third flag on `Ops::OpName` in `core/ops.hpp` | (a). `core/ops.hpp` then states nothing about a later stage, and the proof stays beside the table, as the differentiable flag's proof is beside the rules |
| Q3 | Which code does the seam give a plain `std::invalid_argument`, and a `std::runtime_error` at record time? | (a) `InvalidDescription` and `DeviceFailed`. (b) Both fatal, as defects of the lowering. (c) A new `StatusCode` | (a). The ruling asks for a status. A tensor bound through `TensorPool::bind` that disagrees with its value reaches SushiBLAS's operand checks with no defect in SushiAI. Known cost: a refused scratch lease reads `DeviceFailed`, since SushiBLAS throws one type for it and for a tensor rule |
| Q4 | Work that runs at once (constant fills, initialisation, a per-step optimizer, the batch copy, dataset synthesis) records on the engine's own graph. After a failed recording that graph keeps the tasks recorded before the failure, and v0.1.0 gives no way to drop them | (a) Accept it: a failed status from such a call means the `Engine` is discarded; ask SushiBLAS for `Engine::discard_recorded()`. (b) A SushiAI unit that owns a `TaskGraph` and a `TaskRecorder` for immediate work, used by all five callers | (a) for this migration. (b) is a new module and a `TaskGraph` per call or a shared one with its own lifetime rules |
| Q5 | `StepRecorder::record_into_step` takes a `SushiBLAS::TaskRecorder&` and returns `Result<std::size_t>` | (a) As written. (b) A SushiAI type that wraps the recorder, so `step_recorder.hpp` names no SushiBLAS type | (a). The wrapper would forward one reference. The cost is that a fake recorder's test needs a real `TaskRecorder` |
| Q6 | `Lowering::values_needing_own_allocation` tests two operation ids and its premise no longer holds (`docs/architecture/ARCHITECTURE.md:376-387`) | (a) Keep it as it is in this migration and delete it in its own change after the build. (b) Delete it now. (c) Make it a column of the row | (a). Deleting it changes the memory plan, and a build that fails after this migration should have one cause fewer |
| Q7 | `DeviceFusionSelector` declines GEMM epilogues under `ForceReference` because the host could not read the bias. v0.1.0's reference GEMM runs the epilogue on the host, and SushiBLAS itself refuses a device-only operand | (a) Leave the selector alone. (b) Drop the refusal | (a) until the build shows the fused forms agree with the plain ones under `ForceReference` |
| Q8 | Are the model graph's engine units (M31 to M34) the last wave of this work, or the next work? | (a) Wave 10 here. (b) Their own work, started from `MODEL_GRAPH/SPEC.md` after this one closes | (b). They add `ModelState`, three step builders and two step types; this migration's exit criterion (`docs/design/PROGRAMME.md:45`) does not need them, and M20 and M12 already close the two gaps that break existing tests |
| Q9 | An `Error` from a host unit not yet converted can reach the seam, and `Error` derives from `std::runtime_error` | (a) Fatal inside the third handler, until `Error` is deleted. (b) A status | (a). Inside a guarded call only `Graph::value`, `Graph::node` and `Shape`'s constructors can throw one, and the class table marks all of those fatal |

## 9. Unverified

- **Nothing was compiled.** Every declaration in this document is unchecked text.
- **`CompiledGraph` and its `TaskGraph`.** Whether the plan reads the graph after `compile()` was
  not traced in SushiRuntime. The member order of M26 is safe either way.
- **`take_nodes()` after a failed recording.** Read in the header only
  (`sushiruntime: include/SushiRuntime/graph/task_graph.hpp:234-238`). Whether dropping the nodes
  releases the retained storages and scratch leases was not traced.
- **`TaskGraph::add_task` and `reserve_registration`** may throw; SushiBLAS's own exception table
  marks them "not read" (`sushiblas: docs/agent/2026_10_07_V010/REPORT.md:1660`). The seam's
  third and fifth handlers cover whatever they throw.
- **`RandomOps::constant` and the random stream.** Whether recording a constant fill reserves a
  stream offset was not read. If it does, the `ZERO` and `CONSTANT` rows move the stream that
  initialisation and the synthetic dataset draw from; item 10 of section 7 would show it.
- **The move constructors of `MemoryPlan` and `ArenaSet`** were not read; `ArenaSet::make`
  returns an `ArenaSet` by value and needs one.
- **The message texts tests match.** `require_arity` replaces about forty hand-written arity
  messages with one. A search of `tests/functional` for the lowering's and the pool's texts
  found two expectations, neither on an arity text: `test_arena.cpp:213` (`already adopted`) and
  `test_resilience.cpp:715` (`No lowering for operation`). Both texts are kept and both sites
  are fatal now.
- **`tools/contract/check_contract.py`'s `K_FROZEN` list** was not opened.
- **The test files of M30** were counted and searched, not read in full. The edit list of each
  is what a search for the migrated names returns.
