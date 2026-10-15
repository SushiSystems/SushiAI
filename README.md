# SushiAI

SushiAI is the AI/ML head of the SushiStack: a C++17 library with a
graph-native autograd core, a memory planner, a fusion pass and
mixed-precision dtype plumbing, built on
[SushiBLAS](https://github.com/SushiSystems/SushiBLAS)'s tensor and BLAS
layer, which itself runs on
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph.

**AI-1 has landed: an MLP trains end to end.** `sa demo mlp` traces a
two-layer model into SushiAI's own IR, differentiates that IR, lowers it to
SushiBLAS once, compiles it once, and replays it per batch until the loss
falls from `ln(C)` to convergence. Memory planning and kernel fusion have
landed since, and so has **mixed precision**: fp16 products with fp32
master weights (`--precision mixed`) and gradient loss scaling with overflow
backoff (`--loss-scale S`): see
[`docs/design/ROADMAP.md`](docs/design/ROADMAP.md)
for the roadmap and [`REMAINING_WORK.md`](docs/design/REMAINING_WORK.md)
for the list of what this tree does not do yet, including the limits the
planner and the fusion catalog still have. Nothing in this document promises a
feature that plan hasn't reached.

> **Start here:** [`TUTORIAL.md`](docs/getting_started/TUTORIAL.md) walks from an empty checkout to a
> model that trains, and then explains what each stage of the pipeline did with
> it, including the failure modes that are silent rather than loud.

## What exists today

- **A graph IR** (`graph/ir.hpp`, `graph/builder.hpp`): single-assignment,
  host-only, with every operation's shape inference stated exactly once.
- **Reverse-mode autograd as an IR-to-IR transform** (`autograd/backward.hpp`):
  a `constexpr` rule table, tiered fan-out accumulation with no atomics,
  and a fused log-softmax + cross-entropy gradient. It touches no device.
- **Finite-difference gradient checking in FLOAT64** (`autograd/gradcheck.hpp`),
  full-Jacobian and randomised-directional, which is what the integration
  tier verifies every backward rule with.
- **A lowering to SushiBLAS** (`graph/lowering.hpp`, `graph/lowering/`): eight
  per-domain tables (elementwise, linalg, activation, loss, shape, spatial,
  norm, fused) that `lowering/all.hpp` composes into `K_LOWERINGS`, one row per
  operation. `Lowering::record` hands each node to its row with a
  `LoweringContext`, and a row reaches SushiBLAS only through a view built
  from the step's `SushiBLAS::TaskRecorder`. `CompiledStep`
  (`graph/compiled_step.hpp`) owns the `TaskGraph` and that recorder, compiles
  once in `CompiledStep::compile` and replays in `run()`.
- **One failure contract** (`core/status.hpp`, `core/result.hpp`,
  `core/seam.hpp`): anything that allocates, records, compiles or runs returns
  a `Status` or a `Result<T>`, and `core/seam.hpp` is where an exception of
  SushiBLAS or SushiRuntime becomes a status. Graph building and
  differentiation, which touch no device, throw `SushiAI::Error`.
- **A fusion pass** (`graph/fusion.hpp`): a catalog of recipes applied as a
  selector over the IR before lowering, reported per-run by `sa demo mlp` and
  switchable with `--fusion off` / `--no-fuse NAME`. One entry,
  `gemm+bias+gelu`, spills its pre-activation to a second output so an
  activation whose gradient reads its *input* can be folded without the
  backward pass losing what it differentiates at; `sa demo mlp --activation
  gelu` runs it.
- **A memory planner and per-lifetime arenas** (`graph/memory_plan.hpp`,
  `graph/arena.hpp`): IR values packed into arenas by lifetime instead of one
  `Storage` each, pinned by `Regression_StaticMemory`.
- **Mixed-precision dtype plumbing** (`graph/precision.hpp`,
  `graph/cast_insertion.hpp`): a per-operation precision policy and an
  IR-to-IR pass that inserts explicit `CAST` nodes at every dtype boundary, so
  the fp16 a GEMM reads is a *derived* value and the fp32 master weight it came
  from is untouched. `--precision mixed`.
- **Gradient loss scaling** (`optim/loss_scaler.hpp`): the numerics half:
  the backward seed is multiplied by a device-resident power of two, the
  gradient is divided back out inside the optimizer's fused kernel before the
  decay and the moments, `reductions().count_nonfinite()` screens every
  gradient, and an overflowed step is skipped by a real branch in that kernel.
  Halve on overflow, double after N clean steps. `--loss-scale S`. **It does
  nothing for this demo** (see below), and the scale is device-resident
  precisely so that changing it does not recompile the step.
- **Layers and losses** (`nn/`): `Linear`, `ReLU`/`GELU`/`Tanh`/`Sigmoid`,
  `Sequential` with CTAD, `CrossEntropyLoss`, Kaiming/Xavier initialisation.
  No base class and no `register_module`: a layer publishes its parameters
  through a `static constexpr fields()` list, and one generic walk produces
  the whole PyTorch-shaped named-parameter tree.
- **Optimizers** (`optim/`): SGD with PyTorch's momentum formulation, and
  AdamW with decoupled decay applied first and `eps` added after the bias
  correction. Each parameter's whole update is **one kernel launch**, and the
  update is recorded into the same compiled step as the forward and backward
  passes; see [ARCHITECTURE.md §5.1](docs/architecture/LOWERING.md#51-the-optimizer-is-part-of-the-compiled-step).
- **Data and a training loop** (`data/`, `train/`): a `Dataset` seam, a
  hermetic seeded synthetic classification problem generated on device, an
  MNIST IDX loader that activates only when the files are present, and a
  trainer that reports its compile count.
- **A model can be a JSON file** (`config/`, `nn/layer.hpp`,
  `nn/layer_stack.hpp`): `sushiai train train.json` and
  `sushiai inference model.json model.sushi eval.json` contain **no layer,
  loss, dataset or optimizer type name**; each arrives through a constant
  `{name, factory}` table keyed by a string out of the configuration. The
  runtime layer list costs nothing per step, because `forward()` runs once at
  trace time and the graph is compiled once and replayed. An unknown key or
  `type` is an error listing what *is* accepted. See
  [RUNNING_MODELS.md](docs/guides/RUNNING_MODELS.md#sushiai-train-and-sushiai-inference-config-driven-runs).
- **Checkpointing** (`io/`): a versioned, explicitly little-endian `SACP`
  format recording a name, a dtype and a `Shape` per tensor, so `[256, 6]`
  cannot load into `[6, 256]`. A checkpoint written by a hand-written
  `Sequential` loads into a `LayerStack` built from the equivalent
  `model.json`, and the reverse, which is how the compile-time and runtime
  paths are *proved* to describe one model.

```json
{
  "sushiai_model": 1,
  "input": { "features": 6 },
  "layers": [
    { "type": "linear", "units": 256 },
    { "type": "gelu" },
    { "type": "linear", "units": 4 }
  ]
}
```

- **`sa demo mlp`**: a real, flag-driven training program
  (`applications/examples/source/demo_mlp.cpp`), not a wrapper around the example. See
  [the demo section](#the-demo-sa-demo-mlp) for its output.
- `SushiAI::sushiai`: a `STATIC` library linking `SushiRuntime::SushiRuntime`
  and `SushiBLAS::sushiblas` `PUBLIC`.
- A genuine [D3-resolved](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
  two-sibling dependency chain: the same layered CMake resolution
  (TARGET-guard → `find_package` → sibling `add_subdirectory`) used
  stack-wide, run twice (SushiRuntime, then SushiBLAS).
- `find_package(SushiAI)`-ready packaging (SB-6's shape), verified with a
  real out-of-tree consumer in `tests/package/`.
- The `sa` / `sushiai` CLI (Typer + rich), modeled on
  [SushiEngine](https://github.com/SushiSystems/SushiEngine)'s `se`: a
  shared bundled toolchain, no toolchain selection of its own.

## Requirements

- A SYCL 2020 compiler: the bundled **intel-llvm** `clang++ -fsycl`,
  provisioned by SushiStack. SushiAI selects no SYCL toolchain of its own; it
  builds with whatever its sibling checkouts resolve.
- **CMake** 3.20 or newer (3.22 for the test suite's `gtest_discover_tests`).
- **GoogleTest** for the test suite.
- **SushiRuntime** and **SushiBLAS** checkouts, each resolved via
  [D3 layered CMake resolution](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3):
  an in-tree superproject target, an installed `find_package`, or (the
  common case) a sibling `../sushiruntime` / `../sushiblas` checkout.
- **C++17**: the same standard SushiBLAS and SushiRuntime are built to, and
  the same standard a consumer must compile at. Not "or later": the stack has
  to agree on one, because `SushiRuntime::span` picks between `std::span` and
  SushiRuntime's own shim on `__cplusplus` and that choice reaches the compiled
  ABI. See
  [ARCHITECTURE.md §1](docs/architecture/ARCHITECTURE.md#the-whole-stack-is-c17-and-that-is-not-a-style-choice)
  for why, and [INTEGRATION.md §3](docs/guides/INTEGRATION.md#3-what-you-inherit-and-what-you-must-supply)
  for what it means if you are the consumer.

`sa setup` provisions the C++ library dependencies and the SYCL toolchain into
the shared root, `~/.sushisystems` (or `SUSHISYSTEMS_HOME`).
[SushiStack](https://sushisystems.io) (the `hub` CLI) provisions several
checkouts at once.

## Quick start (one script)

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

See the [SushiStack](https://sushisystems.io) docs for linking an existing
checkout into a workspace (`hub link`) instead of a fresh clone.

### Using SushiAI from another project

`find_package(SushiAI REQUIRED)` and link `SushiAI::sushiai`. Read
[INTEGRATION.md](docs/guides/INTEGRATION.md) first. `tests/package/` is a complete
working consumer.

## Building

```bash
hub install-cli sushiai           # always editable, against this checkout
```

This gives you two equivalent commands on your PATH: **`sa`** (short) and
**`sushiai`** (long). In the common case (a SushiStack workspace, or sibling
checkouts of SushiRuntime and SushiBLAS present) no local config is needed:
the CLI finds both siblings and the shared bundled `clang++` on its own.

```bash
sa build                      # release (default)
sa build --type debug
sa build --type relwithdebinfo
sa build --clean              # wipe the build tree first

# Any CMake cache variable, repeatable, passed straight through to configure.
# Values are cached, so dropping the flag later does not unset them.
sa build -D SA_BUILD_TESTS=OFF
sa build -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_61"
```

`-D` is how SushiBLAS's AOT device targeting is reached from here: SushiAI
`add_subdirectory`s SushiBLAS, so `SB_SYCL_TARGETS` is a cache variable of
*this* build tree. Caller-supplied `-D`s are appended after every default the
CLI sets, so an explicit one is never outranked.

### Building with CMake directly

[Building without the CLI](docs/guides/CLI_GUIDE.md#building-without-the-cli) gives the CMake line and the cache variables.

### Other CLI commands

```bash
sa setup          # provision the dependencies and write cli/config.local.toml
sa doctor         # report whether this machine can build SushiAI
sa link           # record this checkout in a workspace's module registry
sa unlink         # remove it from that registry
sa config         # print resolved config with per-field source annotation
sa env            # print the environment the cmake, ctest and run subprocesses see
sa docs           # build the Doxygen API reference into build/docs/api-site/html/
sa demo mlp       # train an MLP end to end (see below)
sa --version      # print the CLI's version
sa --describe     # print the command catalogue as JSON
```

`sa --help` groups the commands under Project, Models and Diagnostics, and each command's own help
ends with examples. `sa train`, `sa inference` and `sa demo mlp` pass `--help` on to the program
they run once it is built, so their flag lists come from that program. `sa doxygen` still runs
`sa docs` until 2027-03-25. On a colour terminal the root screen shows
the logo. On a dark terminal add `[cli]` and `background = "dark"` to `cli/config.local.toml` to
give it a glow; `COLORFGBG` is read when the terminal sets it, and Windows Terminal does not.

## The demo (`sa demo mlp`)

```bash
sa demo mlp                                   # the defaults
sa demo mlp --help                            # the full flag list
```

It trains a small classifier and prints the loss, the device tasks per step and what fusion and mixed precision changed. [`docs/guides/DEMO.md`](docs/guides/DEMO.md) shows the output and what each number means.

## Running the tests

```bash
sa test                          # functional (unit + integration + regression) — the default
sa test --suite unit
sa test --suite integration
sa test --suite regression
sa test --suite all

sa run sushiai_core_graph_tests                        # one module's cases in one process
sa run sushiai_core_version_tests -- --gtest_filter='Unit_Version.*'

ctest --test-dir build -L unit
ctest --test-dir build -L regression
```

Each module builds its own test program, `sushiai_<tier>_<module>_tests`, from
the files under `tests/<tier>/<module>/`; `tests/README.md` describes the tree.
`sa test` runs the cases of every program through CTest, one process per case.
SushiBLAS builds its own suite only when it is the top-level project
(`SB_BUILD_TESTS`), so none of its tests are compiled or counted here.

The `regression` tier pins guarantees whose failure is silent: today
`compile_count() == 1` across repeated `fit()` calls, and the fact that every
replay actually dispatches instead of reporting complete over an unarmed plan
(see `tests/layers/train/test_trainer/test_compile_once.cpp`).

## How it works

[`docs/getting_started/HOW_IT_WORKS.md`](docs/getting_started/HOW_IT_WORKS.md) walks one training program, step by step.

## Project layout

[`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md#project-layout) has the directory tree.

## Documentation

- [Getting Started Tutorial](docs/getting_started/TUTORIAL.md)
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [CLI Guide](docs/guides/CLI_GUIDE.md)
- [Integration Guide](docs/guides/INTEGRATION.md)
- [API Reference](docs/reference/API_REFERENCE.md)
- [Glossary](docs/reference/GLOSSARY.md)
- [Known Issues](docs/reference/KNOWN_ISSUES.md)
- [Changelog](docs/reference/CHANGELOG.md)
- [Design & Roadmap](docs/design/ROADMAP.md)
- [Remaining Work Backlog](docs/design/REMAINING_WORK.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Documentation Style Guide](docs/DOCUMENTATION_STYLE_GUIDE.md)

## License

SushiAI is source-available, free for non-commercial use under the PolyForm Noncommercial
License 1.0.0; see [LICENSE](LICENSE). Commercial use needs a licence from Sushi Systems, see
[COMMERCIAL.md](COMMERCIAL.md). Third-party components and the licences earlier commits carried
are in [NOTICE.md](NOTICE.md).

