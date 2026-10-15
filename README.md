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
landed since, and so has **mixed precision** — fp16 products with fp32
master weights (`--precision mixed`) and gradient loss scaling with overflow
backoff (`--loss-scale S`) — see
[`docs/design/ROADMAP.md`](docs/design/ROADMAP.md)
for the roadmap and [`REMAINING_WORK.md`](docs/design/REMAINING_WORK.md)
for the list of what this tree does not do yet, including the limits the
planner and the fusion catalog still have. Nothing in this document promises a
feature that plan hasn't reached.

> **Start here:** [`TUTORIAL.md`](docs/getting_started/TUTORIAL.md) walks from an empty checkout to a
> model that trains, and then explains what each stage of the pipeline did with
> it — including the failure modes that are silent rather than loud.

## What exists today

- **A graph IR** (`graph/ir.hpp`, `graph/builder.hpp`) — single-assignment,
  host-only, with every operation's shape inference stated exactly once.
- **Reverse-mode autograd as an IR-to-IR transform** (`autograd/backward.hpp`)
  — a `constexpr` rule table, tiered fan-out accumulation with no atomics,
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
- **A fusion pass** (`graph/fusion.hpp`) — a catalog of recipes applied as a
  selector over the IR before lowering, reported per-run by `sa demo mlp` and
  switchable with `--fusion off` / `--no-fuse NAME`. One entry,
  `gemm+bias+gelu`, spills its pre-activation to a second output so an
  activation whose gradient reads its *input* can be folded without the
  backward pass losing what it differentiates at; `sa demo mlp --activation
  gelu` runs it.
- **A memory planner and per-lifetime arenas** (`graph/memory_plan.hpp`,
  `graph/arena.hpp`) — IR values packed into arenas by lifetime instead of one
  `Storage` each, pinned by `Regression_StaticMemory`.
- **Mixed-precision dtype plumbing** (`graph/precision.hpp`,
  `graph/cast_insertion.hpp`) — a per-operation precision policy and an
  IR-to-IR pass that inserts explicit `CAST` nodes at every dtype boundary, so
  the fp16 a GEMM reads is a *derived* value and the fp32 master weight it came
  from is untouched. `--precision mixed`.
- **Gradient loss scaling** (`optim/loss_scaler.hpp`) — the numerics half:
  the backward seed is multiplied by a device-resident power of two, the
  gradient is divided back out inside the optimizer's fused kernel before the
  decay and the moments, `reductions().count_nonfinite()` screens every
  gradient, and an overflowed step is skipped by a real branch in that kernel.
  Halve on overflow, double after N clean steps. `--loss-scale S`. **It does
  nothing for this demo** — see below — and the scale is device-resident
  precisely so that changing it does not recompile the step.
- **Layers and losses** (`nn/`) — `Linear`, `ReLU`/`GELU`/`Tanh`/`Sigmoid`,
  `Sequential` with CTAD, `CrossEntropyLoss`, Kaiming/Xavier initialisation.
  No base class and no `register_module`: a layer publishes its parameters
  through a `static constexpr fields()` list, and one generic walk produces
  the whole PyTorch-shaped named-parameter tree.
- **Optimizers** (`optim/`) — SGD with PyTorch's momentum formulation, and
  AdamW with decoupled decay applied first and `eps` added after the bias
  correction. Each parameter's whole update is **one kernel launch**, and the
  update is recorded into the same compiled step as the forward and backward
  passes — see [ARCHITECTURE.md §5.1](docs/architecture/ARCHITECTURE.md#51-the-optimizer-is-part-of-the-compiled-step).
- **Data and a training loop** (`data/`, `train/`) — a `Dataset` seam, a
  hermetic seeded synthetic classification problem generated on device, an
  MNIST IDX loader that activates only when the files are present, and a
  trainer that reports its compile count.
- **A model can be a JSON file** (`config/`, `nn/layer.hpp`,
  `nn/layer_stack.hpp`) — `sushiai train train.json` and
  `sushiai inference model.json model.sushi eval.json` contain **no layer,
  loss, dataset or optimizer type name**; each arrives through a constant
  `{name, factory}` table keyed by a string out of the configuration. The
  runtime layer list costs nothing per step, because `forward()` runs once at
  trace time and the graph is compiled once and replayed. An unknown key or
  `type` is an error listing what *is* accepted. See
  [CLI_GUIDE.md](docs/guides/CLI_GUIDE.md#sushiai-train-and-sushiai-inference--config-driven-runs).
- **Checkpointing** (`io/`) — a versioned, explicitly little-endian `SACP`
  format recording a name, a dtype and a `Shape` per tensor, so `[256, 6]`
  cannot load into `[6, 256]`. A checkpoint written by a hand-written
  `Sequential` loads into a `LayerStack` built from the equivalent
  `model.json`, and the reverse — which is how the compile-time and runtime
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

- **`sa demo mlp`** — a real, flag-driven training program
  (`applications/examples/source/demo_mlp.cpp`), not a wrapper around the example. See
  [the demo section](#the-demo-sa-demo-mlp) for its output.
- `SushiAI::sushiai` — a `STATIC` library linking `SushiRuntime::SushiRuntime`
  and `SushiBLAS::sushiblas` `PUBLIC`.
- A genuine [D3-resolved](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
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
  [D3 layered CMake resolution](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3):
  an in-tree superproject target, an installed `find_package`, or (the
  common case) a sibling `../sushiruntime` / `../sushiblas` checkout.
- **C++17** — the same standard SushiBLAS and SushiRuntime are built to, and
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
checkouts of SushiRuntime and SushiBLAS present) no local config is needed —
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

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/clang++ \
  -DCMAKE_C_COMPILER=/path/to/clang \
  -DSUSHIRUNTIME_DIR=/path/to/sushiruntime \
  -DSUSHIBLAS_DIR=/path/to/sushiblas
cmake --build build --config Release
```

`SA_BUILD_TESTS` (default `ON`) builds the test suite. Turn it off
(`-DSA_BUILD_TESTS=OFF`) when embedding SushiAI with `add_subdirectory` and
you do not want someone else's suite compiled, discovered and registered into
your CTest run.

`CMAKE_C_COMPILER` must be the toolchain's plain `clang`, not `clang++` — a
`clang++`-personality binary hardcodes C++ mode and breaks CMake's C-language
probe (the project declares `LANGUAGES CXX C`, inherited transitively from
SushiBLAS). Both `_DIR` variables are optional; without them,
[D3 resolution](docs/architecture/ARCHITECTURE.md#2-the-sibling-dependency-chain-d3) falls back
to sibling `../sushiruntime` / `../sushiblas` checkouts or installed
packages.

### Other CLI commands

```bash
sa setup          # provision the dependencies and write cli/config.local.toml
sa doctor         # report whether this machine can build SushiAI
sa link           # record this checkout in a workspace's module registry
sa unlink         # remove it from that registry
sa config         # print resolved config with per-field source annotation
sa env            # print the environment cmake/ctest/run subprocesses see
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
sa demo mlp                                   # the defaults below
sa demo mlp --epochs 4 --optimizer sgd        # anything you like
sa demo mlp --help                            # the full flag list
```

It trains on MNIST when the training split's two IDX files are under
`--mnist-dir` (`data/mnist` by default) and on a seeded synthetic
classification problem otherwise — and says which it got, in both the header
and the result block. The IDX files are not shipped and not downloaded, so
out of the box you get the synthetic run below.

Real output from `sa demo mlp` with no flags, on an AMD Ryzen 5 7600X CPU
device. The runtime and SushiBLAS log at INFO to stdout in a Release build
(their level is a compile-time constant, so a consumer cannot turn it down);
those lines are elided at the `...` below, and it is exactly why the demo
holds its result back and prints it as one block at the end:

```text
SushiAI MLP demo - configuration
----------------------------------------------------------------
SushiAI  : 0.1.0 on SushiBLAS 0.1.0
Data     : SYNTHETIC FALLBACK - no MNIST here.
           Looked for images: data/mnist\train-images-idx3-ubyte, labels: data/mnist\train-labels-idx1-ubyte.
Model    : 32 -> 128 -> 6  (4998 learnable scalars)
Optimizer: adamw (lr 0.003)
Batches  : 16 x 32 from 'synthetic-classification'
Graph    : 17 IR nodes over 22 values
Precision: fp32; every operation computes in the graph's own type
Fusion   : 22 node(s) -> 17
           fused    gemm+bias+relu -> value 6
           fused    gemm+bias -> value 10
           fused    exp-sub-scale -> value 15
Buffers  : 24 tensors, 96.7 KiB (one allocation each: 96.7 KiB)
Arenas   : 1 (activation), 5 buffers in 49.5 KiB instead of 49.5 KiB - 0% saved
Schedule : 8 epochs x 16 batches = 128 steps, seed 942917386647760909

...

================================================================
 SushiAI MLP demo - result
 synthetic fallback, adamw, 32 -> 128 -> 6
================================================================
  epoch   1/8   loss 1.791759 -> 0.037844   accuracy 100.00%
  epoch   2/8   loss 0.024981 -> 0.001700   accuracy 100.00%
  epoch   3/8   loss 0.001294 -> 0.000673   accuracy 100.00%
  epoch   4/8   loss 0.000541 -> 0.000515   accuracy 100.00%
  epoch   5/8   loss 0.000421 -> 0.000464   accuracy 100.00%
  epoch   6/8   loss 0.000380 -> 0.000432   accuracy 100.00%
  epoch   7/8   loss 0.000353 -> 0.000405   accuracy 100.00%
  epoch   8/8   loss 0.000331 -> 0.000380   accuracy 100.00%

compiles       : 1
steps          : 128
ops per step   : 25
loss           : 1.791759 -> 0.000380
final accuracy : 100.00%
wall clock     : 2122.9 ms (16.58 ms/step)
per-node timing: none recorded (build the RuntimeContext with RuntimeSettings::profiling to collect it)
----------------------------------------------------------------
  compiles: 1  <- one lowering, one compile, 128 replays.
================================================================
```

`1.791759` is `ln 6` to the last digit — the exact uniform-prior loss a
zero-initialised classifier must start at. **`compiles: 1`** is the claim the
whole design rests on: 128 training steps, one lowering, one compile. The
CTest entry `Integration_Demo.MLPRunsAndReportsOneCompile` and
`cli/tests/test_demo.py` both pin that line, so the demo cannot rot quietly.

**`ops per step: 25`** is 21 graph nodes plus one fused update per parameter —
the optimizer is *inside* that single compiled plan
([ARCHITECTURE.md §5.1](docs/architecture/ARCHITECTURE.md#51-the-optimizer-is-part-of-the-compiled-step)),
and the graph itself has been through the fusion pass
([§5.2](docs/architecture/ARCHITECTURE.md#52-fusion-is-a-selector-over-a-catalog-not-a-code-generator)).
Spelled the old way — twelve separate optimizer calls per parameter flushed in
a graph of their own, and no fusion — the same run issues 81 device tasks per
step where this issues 25.

The **`Fusion`** line is the pass reporting itself, refusals included. Run
`--fusion off` to get the unfused graph back, or `--no-fuse gemm+bias+relu` to
switch one catalog entry off; every spelling produces the same numbers, which
is what `Regression_Fusion` pins byte for byte.

### What fusion is actually worth here, including the part that costs

Measured on this machine (CPU/OpenCL, 12 cores, idle, `--fusion off` and
`--fusion on` interleaved from one binary against one SushiBLAS build):

| | unfused | fused |
|---|---|---|
| kernels per step | 35 | 25 |
| steady-state cost | 0.57 ms/step | **0.40 ms/step** |
| first step | 0.97 s | **1.88 s** |

The steady-state figure is the slope of total wall clock against step count,
taken between 1 024 and 4 096 steps so the fixed startup term cancels; the
median of three replicates each, whose spread is about ±0.06 ms/step. **A
step is about 30% cheaper, which is roughly the 29% of its kernels that
stopped being launched** — these tensors are small enough that a step is mostly
launch overhead, which is exactly the regime where removing launches pays one
for one.

**And it costs about 0.9 s once, on the first step.** Every fused form is a
distinct C++ template instantiation and therefore a distinct SYCL kernel, which
this OpenCL backend compiles at first launch. A single-step run costs 0.97 s
unfused and 1.88 s fused, which is the whole of the difference; it does not
scale with the model, the batch or the step count. So **break-even on this
machine is around 5 000 steps**, and the 128-step run printed above is far
below it — fusion makes the default demo about 1.9x *slower* in wall clock
while issuing 29% fewer kernels. That is not a contradiction and it is not
hidden: it is one-time device-code compilation, it is what AOT targeting
(`sa build -D SB_SYCL_TARGETS=spir64`) exists to move to build time, and it is
irrelevant to any training run long enough to be worth doing. The launch count
is the number this change controls; the wall clock is reported both ways
because reporting only the favourable half would be dishonest.

The **`Arenas`** line now says `0% saved`, and that is fusion's doing rather
than the planner failing: a fused subtree's intermediates stop being values at
all, so the buffers the planner used to overlap no longer exist to overlap.
The whole pool is 96.6 KiB against 130.9 KiB before either pass existed.
`Regression_StaticMemory` pins both halves — packing still shrinks the
*unfused* graph, and the fused graph is never larger than the packed unfused
one.

### Mixed precision: what `--precision mixed` does, and what it is worth here

`--precision mixed` runs the matrix products — and the bias and activation a
fused GEMM folds into them — in **fp16**, keeping fp32 master weights, fp32
gradients, fp32 reductions and an fp32 optimizer update. It is structure in
the IR, not a runtime switch: a per-operation policy
(`graph/precision.hpp`) says what each node computes in, and a pass
(`graph/cast_insertion.hpp`) inserts explicit `CAST` nodes at every boundary
so that every node's operands agree with its computed type by construction.
The fp16 copy of a weight is a *derived* value with an activation lifetime;
the master weight keeps its id, its `DType::Float32` and its
`Lifetime::Parameter`.

Same command, same seed, `--epochs 8 --batches 16` on this machine:

| | `--precision fp32` | `--precision mixed` |
|---|---|---|
| casts inserted | 0 | 10 (6 narrowing, 4 widening) |
| ops per step | 25 | 35 |
| loss, epoch 1 | 1.791759 → 0.037844 | 1.791759 → 0.037880 |
| loss, epoch 8 | 0.000331 → 0.000380 | 0.000332 → 0.000380 |
| final accuracy | 100.00% | 100.00% |
| wall clock | 2105.9 ms (16.45 ms/step) | 9364.9 ms (73.16 ms/step) |

(The wall clock differs by about 1% from the block quoted above; that is a
different run of the same command, not a different configuration.)

Read that table honestly, because two of its rows say uncomfortable things.

**It is 4.5x slower, and that is expected rather than a defect.** There is no
fp16 matrix unit on this device; `sycl::half` arithmetic is emulated, so every
narrowed kernel does strictly more work than its fp32 twin and then pays for
ten conversions on top. **No benchmark taken here would say anything about
mixed precision on hardware that has a low-precision matrix unit**, which is
the hardware this exists for. Speed is not asserted anywhere in the test
suite for the same reason.

**The loss tracks fp32 closely and does not stall.** Both runs finish at
0.000380 with 100% accuracy. At the largest loss where the printed digits are
actually meaningful — epoch 1's last step, `0.037844` against `0.037880` —
the relative deviation is 9.5e-4. The demo prints six decimals, so by epoch 8
the losses are down at `0.000331` against `0.000332` and the apparent 3e-3
there is dominated by the printing rather than by the arithmetic; the
per-step comparison that is not (`Regression_MixedPrecision`) reads the
doubles directly. That is a *measurement of this workload*, not a general
claim: fp16 training without loss scaling is
expected to lose small gradients to underflow, and on this small MLP it
demonstrably does not. Repeating it with `--hidden 512 --batch-size 64` (a
512-long fp16 reduction inside each product) gives the same answer — both
arms reach 0.000003 after 48 steps. So the loss scaling that has since landed
is justified by theory and by larger models, and **not** by anything this
workload does — turning it on here makes the demo no better, and saying so is
worth more than a claim that it does.

What `--loss-scale` *is* good for on this box is watching the machinery work.
`sa demo mlp --epochs 1 --batches 4 --precision fp32 --loss-scale 65536` gives
`loss 1.791759 -> 1.133352`, bit-identical to the unscaled run, which is the
whole point of a dyadic scale: the round trip through 65536 and back is exact.
`--precision mixed --loss-scale 65536` reports one overflowed step and a
backoff to 32768 — PyTorch's own default scale is too high for this model's
fp16 backward pass, and the backoff finds that out in one step. Pushing it to
`--loss-scale 16777216` overflows on all 8 steps and walks the scale from
2^24 down to 2^16 without recompiling: `compiles: 1` throughout.

One honest caveat: the skip is **per parameter**, not per step. A parameter
whose own gradient stayed finite still updates on an overflowed step, so the
report says "overflowed steps" rather than "skipped". No infinity or NaN
reaches a parameter or a moment either way — that is asserted bit for bit in
`Regression_LossScaling` — but it is a partial step where PyTorch would take
none. `optim/loss_scaler.hpp` explains what a whole-step skip would cost.

`gradcheck` and mixed precision are incompatible, and deliberately so:
`autograd/gradcheck.hpp` requires FLOAT64 because a central difference in
fp16 has a noise floor about a thousand times *larger* than the discrepancy a
wrong backward rule produces. A gradient rule is a statement about the
mathematics, so checking it once in fp64 checks it for every precision; what
mixed precision can get wrong is the plumbing, and that is checked by walking
the graph (`Graph::find_dtype_mismatches`) rather than by a finite difference.

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

The `regression` tier pins guarantees whose failure is silent — today
`compile_count() == 1` across repeated `fit()` calls, and the fact that every
replay actually dispatches instead of reporting complete over an unarmed plan
(see `tests/layers/train/test_trainer/test_compile_once.cpp`).

## How it works (today)

```cpp
#include <SushiAI/SushiAI.hpp>
#include <SushiBLAS/SushiBLAS.hpp>

namespace sb = SushiBLAS;

sa::Status train()
{
    sr::Execution::RuntimeContext context;
    sb::Engine engine(context);

    // A model is an aggregate list, not a registration ritual.
    sa::NN::Sequential model{
        sa::NN::Linear{32, 128}, sa::NN::ReLU{}, sa::NN::Linear{128, 6},
    };

    // Trace, add cross-entropy, differentiate, insert casts, fuse: all IR-to-IR,
    // no device work yet. The trailing arguments default to float32, every
    // fusion and one element type everywhere; pass a pointer to a
    // sa::Graph::MixedPrecisionPolicy as `precision` for fp16 products.
    SA_TRY_ASSIGN(sa::Train::TrainingGraph built,
                  sa::Train::build_classifier(model, 32, 32, 6));

    SA_TRY_ASSIGN(sa::NN::ModelState state, sa::NN::ModelState::create(engine, model));
    sa::Graph::TensorPool pool(engine);
    SA_TRY_ASSIGN(std::vector<sa::Optim::ParameterSlot> parameters,
                  sa::Train::bind_parameters(built, state, pool));
    SA_TRY(sa::NN::initialize(engine, state));

    SA_TRY_ASSIGN(std::unique_ptr<sa::Optim::AdamW> optimizer,
                  sa::Optim::AdamW::make(engine, std::move(parameters)));

    // The objective build_classifier traced; it feeds and measures every step.
    const sa::Train::CrossEntropyObjective objective(32, 6, sa::DType::Float32);

    // Trainer::make lowers and compiles; it is the only compile of the run.
    SA_TRY_ASSIGN(std::unique_ptr<sa::Train::Trainer> trainer,
                  sa::Train::Trainer::make(engine, built, pool, objective, *optimizer));

    sa::Data::SyntheticOptions synthetic;
    synthetic.features = 32;
    synthetic.classes = 6;
    SA_TRY_ASSIGN(std::unique_ptr<sa::Data::SyntheticClassification> data,
                  sa::Data::SyntheticClassification::make(engine, synthetic));

    SA_TRY_ASSIGN(const sa::Train::RunReport report, trainer->fit(*data));
    std::cout << report.to_string();   // reports compiles: 1
    return sa::Status();
}
```

Every step that allocates, compiles or runs returns a `Result` or a `Status`;
`SA_TRY` and `SA_TRY_ASSIGN` return the failure to the caller.
`applications/examples/source/example.cpp` is this program with the dataset chosen at run time:
MNIST when its
files are present, synthetic data otherwise.

SushiAI writes no SYCL: no file here names a `sycl::` type or submits to a
queue, and every compute call goes through `SushiBLAS::sushiblas` (linked
`PUBLIC`). The rows under `modules/core/graph/source/lowering/` turn IR into those calls. The
optimizers do hand SushiBLAS a device functor for their fused
update — plain arithmetic that SushiBLAS's own elementwise seam inlines into
its own kernel. See
[ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) for the full picture and
[the master plan](docs/design/ROADMAP.md) for what comes next.

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
  SushiAISyclTarget.cmake        sushiai_apply_sycl() helper
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

