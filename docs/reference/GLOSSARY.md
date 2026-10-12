# Glossary

The words this repository's documents use, each with the header or file that defines it. Header
paths are relative to `include/SushiAI/` of the module that owns them, `modules/<tier>/<module>/`;
`graph/ir.hpp`, for example, is `modules/core/graph/include/SushiAI/graph/ir.hpp`.

## The graph

| Word | Meaning |
| --- | --- |
| IR | SushiAI's own graph: a single-assignment DAG of tensor operations, built on the host and touching no device. Namespace `SushiAI::Graph` (`graph/ir.hpp`) |
| Node | One operation in the IR: an identity, its operands and its results (`Graph::Node`) |
| Value | One single-assignment, tensor-shaped result in the IR (`Graph::Value`) |
| Op identity | The name an operation carries, such as `sushiai.log_softmax`; all of them are declared in `core/ops.hpp` |
| Lifetime | The class a value belongs to: `Parameter`, `Gradient`, `OptimizerState`, `Input`, `Buffer`, `Activation` or `Scratch` (`Graph::Lifetime`, `graph/ir.hpp`). Only values of one class share an arena |
| Layout | Where the channel axis sits. `NHWC` is the native layout; `NCHW` appears at a boundary only (`Graph::TensorLayout`) |
| Backward rule | A function that writes the IR for one operation's gradient. Autograd is an IR-to-IR transform (`autograd/backward.hpp`, `modules/core/autograd/source/rules/`) |
| Gradient barrier | An operation declared with `gradient_barrier_op`: its result lets no gradient reach its operands, so it has no backward rule (`core/ops.hpp`) |
| gradcheck | The finite-difference check of a backward rule, run in FLOAT64 (`autograd/gradcheck.hpp`) |

## From graph to device

| Word | Meaning |
| --- | --- |
| Lowering | Records the SushiBLAS work of an IR graph, node by node, each node through its row of the lowering table (`Graph::Lowering`, `graph/lowering.hpp`) |
| Lowering table | One of the eight per-domain tables under `graph/lowering/`, each a list of rows; `graph/lowering/all.hpp` joins them and checks at compile time that every operation identity has exactly one row, or is listed in `K_UNLOWERED` |
| Lowering row | One entry of a table: an operation identity and the function that records it (`LoweringEntry`, `graph/lowering/rule.hpp`) |
| Lowering context | What one row reads for its node: the recorder, the pool, the operands and the results (`LoweringContext`, `graph/lowering/context.hpp`) |
| Task recorder | The object a SushiBLAS call records its tasks on. A lowering row builds its SushiBLAS view from one (`SushiBLAS::TaskRecorder`) |
| Task graph | The graph of recorded tasks a compiled step owns, compiled into a plan once (`SushiRuntime::Graph`) |
| Compiled step | A graph lowered, compiled once and re-armed for every batch (`Graph::CompiledStep`, `graph/compiled_step.hpp`) |
| Compile once | The guarantee that `compile_count()` stays 1 however many batches run; a regression test pins it |
| Replay | Running the compiled step again for the next batch without tracing or compiling |
| Memory plan | One placement per value plus the arena sizes, computed from value lifetimes before any allocation (`Graph::MemoryPlan`, `graph/memory_plan.hpp`) |
| Arena | One allocation that holds several values whose live ranges do not overlap (`Graph::Arena`, `Graph::ArenaSet`) |
| Tensor pool | The owner of the device buffers behind graph values across steps, including per-node scratch and shared all-ones constants (`Graph::TensorPool`, `graph/tensor_pool.hpp`) |
| Fusion catalog | The closed list of fused kernels this build contains (`Graph::FusedForm`, `graph/fusion.hpp`) |
| Epilogue | Work folded into the end of a GEMM: the bias add and, for some entries, the activation |
| Spill | The second value a fused GELU node writes, holding the pre-activation its gradient reads |
| Rematerialisation | The pass that recomputes a stored `K_IM2COL` result at its later consumers (`graph/rematerialise.hpp`) |
| Step scalars | A small device-resident float vector the host rewrites each step, for numbers that change between replays (`Optim::StepScalars`) |

## Training

| Word | Meaning |
| --- | --- |
| Mixed precision | Running products in fp16 while the weights the optimizer updates stay in fp32 (`sa demo mlp --precision mixed`) |
| Master weights | The fp32 copy of each parameter in a mixed-precision run |
| Precision policy | The object that picks each operation's element type (`Graph::PrecisionPolicy`, `graph/precision.hpp`) |
| Cast insertion | The pass that adds type conversions where a policy changes the element type (`graph/cast_insertion.hpp`) |
| Loss scale | The factor the loss is multiplied by before the backward pass, so fp16 gradients do not underflow (`sa demo mlp --loss-scale`) |
| Loss scaler | The state machine that holds the scale, screens gradients for overflow and backs the scale off (`Optim::LossScaler`) |
| Objective | What a model is trained to minimise (`Train::Objective`) |
| Objective leaves | The graph values an objective declared, by role (`Train::ObjectiveLeaves`) |
| Layer | The run-time interface of one layer of a model read from a file. Its `output_shape()` is the virtual; it replaces the former `ILayer` (`NN::Layer`, `nn/layer.hpp`) |
| Layer stack | A sequential list of layers chosen at run time; it enumerates parameters as `Sequential` does (`NN::LayerStack`) |
| SACP | The first four bytes of every checkpoint file, ASCII "SACP", and the name of the format; version 2, which adds the run-state section, and version 1 still reads (`io/checkpoint_format.hpp`) |

## Failures and construction

| Word | Meaning |
| --- | --- |
| Status | A success, or a failure with a `StatusCode`, a message and no exception. A content error returns one (`core/status.hpp`) |
| Result | A value or a `Status`: what a function returns when a content error can make it fail (`core/result.hpp`) |
| `SA_FAIL_IF`, `SA_TRY` | Macros that make a `Status` from a condition and pass a failed one on (`core/status.hpp`). `SA_TRY_ASSIGN` does the second for a `Result` (`core/result.hpp`) |
| Fatal check | `SA_FATAL_IF`: stops the process on a programmer's misuse, in every build (`core/check.hpp`) |
| Seam | The guards that turn an exception of SushiBLAS or SushiRuntime into a `Status` (`Detail::guard`, `Detail::guard_value` in `core/seam.hpp`). The contract places them in that header alone; KNOWN_ISSUES records where the library does not follow this yet |
| Factory | A static function that builds an object and returns a `Result` instead of throwing: `Graph::CompiledStep::compile`, `Train::Trainer::make`, `Optim::AdamW::make`, `Optim::SGD::make`, `Optim::LossScaler::make`, `Optim::StepScalars::make`, the datasets' `make`, `IO::CheckpointWriter::create` and `IO::CheckpointReader::open` |
| Execution mode | `SushiBLAS::ExecutionMode`: `Auto` lets the library choose the SYCL kernel, `ForceSycl` runs it and `ForceReference` runs the host reference path. The fusion selector reads it (`graph/fusion_selector.hpp`) |

## Stack and tools

| Word | Meaning |
| --- | --- |
| SushiStack | The set of repositories: SushiRuntime, SushiBLAS, SushiEngine, SushiAI and the hub |
| Sibling | A checkout that sits beside this one, such as `../sushiruntime` or `../sushiblas` |
| D3 | The order a dependency is resolved in: an in-tree CMake target, then an installed package through `find_package`, then a sibling checkout |
| `sa` | This repository's CLI, also installed as `sushiai` (`cli/`) |
| Tier | A rank in the module tree: `core`, `layers` or `vision`, lowest first. A module may depend on modules of its own tier or a lower one (`cmake/SushiAILayers.cmake`) |
| Domain | A tier that is a field of use built on the layers. `vision` is the only one, and no domain may depend on another (`docs/design/DOMAIN_MODULES.md`) |
| `hub` | The SushiStack workspace CLI, which installs module CLIs and provisions several checkouts |
| ST, SB, AI | The three tracks of the 2026-08 roadmap, now superseded by `PROGRAMME.md`: stack and CLI wiring, SushiBLAS repair, SushiAI construction. `AI-0` to `AI-7` are the steps of the third (`ROADMAP.md`) |
| Stage | A numbered step of `PROGRAMME.md`, 0 to 6. Stage 1 is the SushiBLAS v0.1.0 migration; stage 2 is the error contract and naming |
| Checker | A script under `tools/` that reads the repository and reports where it breaks a skill |
| Work folder | `docs/agent/<YYYY_MM_DD>_<WORK_NAME>/`, holding the `SPEC.md`, `PLAN.md` and `REPORT.md` of one piece of agent work |
