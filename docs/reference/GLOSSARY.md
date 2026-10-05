# Glossary

The words this repository's documents use, each with the header or file that defines it. Header
paths are relative to `include/SushiAI/`.

## The graph

| Word | Meaning |
| --- | --- |
| IR | SushiAI's own graph: a single-assignment DAG of tensor operations, built on the host and touching no device. Namespace `SushiAI::Graph` (`graph/ir.hpp`) |
| Node | One operation in the IR: an identity, its operands and its results (`Graph::Node`) |
| Value | One single-assignment, tensor-shaped result in the IR (`Graph::Value`) |
| Op identity | The name an operation carries, such as `sushiai.log_softmax`; all of them are declared in `core/ops.hpp` |
| Lifetime | The class a value belongs to: `PARAMETER`, `GRADIENT`, `OPTIMIZER_STATE`, `INPUT`, `ACTIVATION` or `SCRATCH`. Only values of one class share an arena |
| Layout | Where the channel axis sits. `NHWC` is the native layout; `NCHW` appears at a boundary only (`Graph::TensorLayout`) |
| Backward rule | A function that writes the IR for one operation's gradient. Autograd is an IR-to-IR transform (`autograd/backward.hpp`, `src/autograd/rules/`) |
| gradcheck | The finite-difference check of a backward rule, run in FLOAT64 (`autograd/gradcheck.hpp`) |

## From graph to device

| Word | Meaning |
| --- | --- |
| Lowering | Translating IR operations into recorded SushiBLAS work (`Graph::Lowering`, `graph/lowering.hpp`) |
| Compiled step | A graph lowered, compiled once and re-armed for every batch (`Graph::CompiledStep`) |
| Compile once | The guarantee that `compile_count()` stays 1 however many batches run; a regression test pins it |
| Replay | Running the compiled step again for the next batch without tracing or compiling |
| Memory plan | One placement per value plus the arena sizes, computed from value lifetimes before any allocation (`Graph::MemoryPlan`, `graph/memory_plan.hpp`) |
| Arena | One allocation that holds several values whose live ranges do not overlap (`Graph::Arena`, `Graph::ArenaSet`) |
| Tensor pool | The owner of the device buffers behind graph values across steps, including per-node scratch and shared all-ones constants (`Graph::TensorPool`) |
| Fusion catalog | The closed list of fused kernels this build contains (`Graph::FusedForm`, `graph/fusion.hpp`) |
| Epilogue | Work folded into the end of a GEMM: the bias add and, for some entries, the activation |
| Spill | The second value a fused GELU node writes, holding the pre-activation its gradient reads |
| Rematerialisation | The pass that recomputes a stored `IM2COL` result at its later consumers (`graph/rematerialise.hpp`) |
| Step scalars | A small device-resident float vector the host rewrites each step, for numbers that change between replays (`Optim::StepScalars`) |

## Training

| Word | Meaning |
| --- | --- |
| Mixed precision | Running products in fp16 while the weights the optimizer updates stay in fp32 (`--precision mixed`) |
| Master weights | The fp32 copy of each parameter in a mixed-precision run |
| Precision policy | The object that picks each operation's element type (`Graph::IPrecisionPolicy`, `graph/precision.hpp`) |
| Cast insertion | The pass that adds type conversions where a policy changes the element type (`graph/cast_insertion.hpp`) |
| Loss scale | The factor the loss is multiplied by before the backward pass, so fp16 gradients do not underflow (`--loss-scale`) |
| Loss scaler | The state machine that holds the scale, screens gradients for overflow and backs the scale off (`Optim::LossScaler`) |
| Objective | What a model is trained to minimise (`Train::IObjective`) |
| Objective leaves | The graph values an objective declared, by role (`Train::ObjectiveLeaves`) |
| Layer stack | A sequential list of layers chosen at run time; it enumerates parameters as `Sequential` does (`NN::LayerStack`) |
| SACP | The first four bytes of every checkpoint file, ASCII "SACP", and the name of the format; version 1 (`io/checkpoint_format.hpp`) |

## Stack and tools

| Word | Meaning |
| --- | --- |
| SushiStack | The set of repositories: SushiRuntime, SushiBLAS, SushiEngine, SushiAI and the hub |
| Sibling | A checkout that sits beside this one, such as `../sushiruntime` or `../sushiblas` |
| D3 | The order a dependency is resolved in: an in-tree CMake target, then an installed package through `find_package`, then a sibling checkout |
| `sa` | This repository's CLI, also installed as `sushiai` (`cli/`) |
| `hub` | The SushiStack workspace CLI, which installs module CLIs and provisions several checkouts |
| ST, SB, AI | The three tracks of the roadmap: stack and CLI wiring, SushiBLAS repair, SushiAI construction. `AI-0` to `AI-7` are the steps of the third |
| Checker | A script under `tools/` that reads the repository and reports where it breaks a skill |
| Work folder | `docs/agent/<YYYY_MM_DD>_<WORK_NAME>/`, holding the `SPEC.md`, `PLAN.md` and `REPORT.md` of one piece of agent work |
