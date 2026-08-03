# SushiAI

SushiAI is the AI/ML head of the SushiStack: a C++20 library that will grow a
graph-native autograd core, memory planning, kernel fusion, and mixed
precision on top of [SushiBLAS](https://github.com/SushiSystems/SushiBLAS)'s
tensor and BLAS layer, which itself runs on
[SushiRuntime](https://github.com/SushiSystems/SushiRuntime)'s task graph.

**AI-1 has landed: an MLP trains end to end.** `sa run sushiai_example`
traces a two-layer model into SushiAI's own IR, differentiates that IR,
lowers it to SushiBLAS once, compiles it once, and replays it per batch until
the loss falls from `ln(C)` to convergence. Memory planning, kernel fusion
and mixed precision are still ahead — see
[`docs/slop/make_sushiai_great_again.md`](slop/make_sushiai_great_again.md)
for the roadmap and [ARCHITECTURE.md §7](ARCHITECTURE.md#7-whats-not-here-yet)
for an honest list of what this tree does not do yet. Nothing in this
document promises a feature that plan hasn't reached.

## What exists today

- **A graph IR** (`graph/ir.hpp`, `graph/builder.hpp`) — single-assignment,
  host-only, with every operation's shape inference stated exactly once.
- **Reverse-mode autograd as an IR-to-IR transform** (`autograd/backward.hpp`)
  — a `constexpr` rule table, tiered fan-out accumulation with no atomics,
  and a fused log-softmax + cross-entropy gradient. It touches no device.
- **Finite-difference gradient checking in FLOAT64** (`autograd/gradcheck.hpp`),
  full-Jacobian and randomised-directional, which is what the integration
  tier verifies every backward rule with.
- **A lowering to SushiBLAS** (`graph/lowering.hpp`) — the only file in the
  library that calls `SushiBLAS::Engine` — and `CompiledStep`, which makes
  the compile-once/replay contract a type rather than a comment.
- **Layers and losses** (`nn/`) — `Linear`, `ReLU`/`GELU`/`Tanh`/`Sigmoid`,
  `Sequential` with CTAD, `CrossEntropyLoss`, Kaiming/Xavier initialisation.
  No base class and no `register_module`: a layer publishes its parameters
  through a `static constexpr fields()` list, and one generic walk produces
  the whole PyTorch-shaped named-parameter tree.
- **Optimizers** (`optim/`) — SGD with PyTorch's momentum formulation, and
  AdamW with decoupled decay applied first and `eps` added after the bias
  correction.
- **Data and a training loop** (`data/`, `train/`) — a `Dataset` seam, a
  hermetic seeded synthetic classification problem generated on device, an
  MNIST IDX loader that activates only when the files are present, and a
  trainer that reports its compile count.
- `SushiAI::sushiai` — a `STATIC` library linking `SushiRuntime::SushiRuntime`
  and `SushiBLAS::sushiblas` `PUBLIC`.
- A genuine [D3-resolved](ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
  two-sibling dependency chain — the same layered CMake resolution
  (TARGET-guard → `find_package` → sibling `add_subdirectory`) used
  stack-wide, run twice (SushiRuntime, then SushiBLAS).
- `find_package(SushiAI)`-ready packaging (SB-6's shape), verified with a
  real out-of-tree consumer in `tests/package/`.
- The `sa` / `sushiai` CLI (Typer + rich), modeled on
  [SushiEngine](https://github.com/SushiSystems/SushiEngine)'s `se` — a
  shared bundled toolchain, no toolchain selection of its own.

## Requirements

- A SYCL 2020 compiler — the bundled **intel-llvm** `clang++ -fsycl`,
  provisioned by SushiStack. SushiAI selects no SYCL toolchain of its own; it
  builds with whatever its sibling checkouts resolve.
- **CMake** 3.20 or newer (3.22 for the test suite's `gtest_discover_tests`).
- **GoogleTest** for the test suite.
- **SushiRuntime** and **SushiBLAS** checkouts, each resolved via
  [D3 layered CMake resolution](ARCHITECTURE.md#2-the-sibling-dependency-chain-d3):
  an in-tree superproject target, an installed `find_package`, or (the
  common case) a sibling `../sushiruntime` / `../sushiblas` checkout.
- **C++20.**

C++ library dependencies and the SYCL toolchain are provisioned by
[SushiStack](https://sushisystems.io) (the `ss` CLI), which owns dependency
installation for the whole stack into one shared, workspace-level
`dependencies/` tree.

## Quick start (one script)

```bash
curl -fsSL https://sushisystems.io/install.sh | bash
```

```powershell
irm https://sushisystems.io/install.ps1 | iex
```

See the [SushiStack](https://sushisystems.io) docs for linking an existing
checkout into a workspace (`ss link`) instead of a fresh clone.

### Using SushiAI from another project

`find_package(SushiAI REQUIRED)` and link `SushiAI::sushiai`. Read
[INTEGRATION.md](INTEGRATION.md) first. `tests/package/` is a complete
working consumer.

## Building

```bash
ss install-cli sushiai           # add --no-editable for a non-dev install
```

This gives you two equivalent commands on your PATH: **`sa`** (short) and
**`sushiai`** (long). In the common case (a SushiStack workspace, or sibling
checkouts of SushiRuntime and SushiBLAS present) no local config is needed —
the CLI finds both siblings and the shared bundled `clang++` on its own.

```bash
sa build                      # release (default)
sa build --type debug
sa build --type relwithdebinfo
sa build --clean              # wipe the build tree first
```

### Building with CMake directly

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/clang \
  -DSUSHIRUNTIME_DIR=/path/to/sushiruntime \
  -DSUSHIBLAS_DIR=/path/to/sushiblas
cmake --build build --config Release
```

`CMAKE_C_COMPILER` must be the toolchain's plain `clang`, not `clang++` — a
`clang++`-personality binary hardcodes C++ mode and breaks CMake's C-language
probe (the project declares `LANGUAGES CXX C`, inherited transitively from
SushiBLAS). Both `_DIR` variables are optional; without them,
[D3 resolution](ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) falls back
to sibling `../sushiruntime` / `../sushiblas` checkouts or installed
packages.

### Other CLI commands

```bash
sa config         # print resolved config with per-field source annotation
sa env            # print the environment cmake/ctest/run subprocesses see
sa doxygen        # build the Doxygen API reference
sa demo mlp       # AI-7 showcase — not implemented yet, exits 1 with a pointer to the plan
```

## Running the tests

```bash
sa test                          # functional suite (unit + integration) — the default
sa test --suite unit
sa test --suite integration
sa test --suite all

sa run sushiai_tests -- --gtest_filter='Unit_Version.*'

ctest --test-dir build -L unit
ctest --test-dir build -L integration
```

There is no `regression` tier yet — nothing has shipped that a regression
could guard against.

## How it works (today)

```cpp
#include <SushiAI/SushiAI.h>
#include <SushiBLAS/SushiBLAS.h>

sr::Execution::RuntimeContext ctx;
sb::Engine engine(ctx);

auto x = engine.create_tensor({4});
auto y = engine.create_tensor({4});

engine.blas().axpy(2.0f, x, y);
engine.execute();
ctx.wait_all();

std::cout << "SushiAI " << SushiAI::version_string()
          << " ran an op through SushiBLAS " << SUSHIBLAS_VERSION_STRING << ".\n";
```

SushiAI does not touch SYCL directly and defines no ops of its own yet — all
compute is drawn from `SushiBLAS::sushiblas`, linked `PUBLIC` so the autograd
core inside `sushiai` inherits it automatically. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full picture and
[the master plan](slop/make_sushiai_great_again.md) for what comes next.

## Project layout

```
CMakeLists.txt          Mirrors SushiBLAS's root CMakeLists.txt shape
cmake/
  Vcpkg.cmake            (Windows) locate vcpkg + set the toolchain file
  Toolchain.cmake        pick the default SYCL compiler before project()'s probe
  Runtime.cmake          D3 resolution -> SushiRuntime::SushiRuntime
  BLAS.cmake             D3 resolution -> SushiBLAS::sushiblas
  SushiAISyclTarget.cmake        sushiai_apply_sycl() helper
  SushiAIBuildPolicy.cmake       composes warnings/perf/sanitizers
  SushiAICompilerWarnings.cmake
  SushiAIOptimization.cmake
  SushiAISanitizers.cmake
  SushiAIConfig.cmake.in         find_package(SushiAI) package config
include/SushiAI/
  SushiAI.h              Master header — includes every public header
  version.hpp            SUSHIAI_VERSION_* macros
  core/                  Vocabulary: common, ops, shape, allocation
  graph/                 The IR, its builder, the lowering, CompiledStep
  autograd/              Differentiation and gradient checking
  nn/                    Parameters, the Module concept, layers, losses, init
  optim/                 SGD and AdamW
  data/                  The Dataset seam, synthetic data, the MNIST loader
  train/                 The training loop and its run report
src/                     One .cpp per header that needs one, same tree shape
examples/
  example.cpp            A whole training run, end to end
tests/
  common/                     Process-global RuntimeContext + per-test Engine
  functional/unit/            Device-free: shapes, IR, autograd, the walk
  functional/integration/     Real Engine: forward values, gradcheck, training
  package/                    Out-of-tree find_package(SushiAI) consumer
cli/                     Python `sa` / `sushiai` CLI package (Typer + rich)
docs/
  slop/make_sushiai_great_again.md   The live roadmap (AI-0 through AI-N)
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.
