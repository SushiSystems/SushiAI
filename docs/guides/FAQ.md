# Frequently asked questions

Short answers for a developer meeting SushiAI for the first time: what it is, what it needs, how
to build and consume it, how it sits in SushiStack, what it does not do yet, how it is licensed
and where to report a problem. Each answer links to the manual page that holds the detail.

## What is SushiAI?

SushiAI is the AI/ML head of SushiStack: a C++17 library with a graph-native autograd core, a
memory planner, a fusion pass and mixed-precision dtype plumbing. It traces a model once into
its own graph IR, differentiates that IR into more IR, fuses it, plans every buffer's address,
lowers the result to a SushiRuntime `TaskGraph` once, compiles it once and replays it per batch.
A regression test pins `compile_count() == 1` however many batches run.

The tree today holds `Linear`, `ReLU`, `GELU`, `Tanh` and `Sigmoid` layers, `Sequential`,
`CrossEntropyLoss`, the SGD and AdamW optimizers, a seeded synthetic dataset, an MNIST IDX
loader, the `SACP` checkpoint format and the `sa` CLI. The root [README](../../README.md) lists
each part with its header, and the [tutorial](../getting_started/TUTORIAL.md) walks from an
empty checkout to a trained model.

## How does SushiAI relate to SushiRuntime, SushiBLAS and the rest of SushiStack?

SushiAI is built on SushiBLAS's tensor and BLAS layer, and SushiBLAS runs on SushiRuntime's task
graph. SushiAI writes no SYCL: no file names a `sycl::` type or submits to a queue, every
compute call goes through SushiBLAS, and the eight per-domain tables under `modules/core/graph/source/lowering/`
turn IR into those calls, each through a view built from the step's `SushiBLAS::TaskRecorder`.
The library target `SushiAI::sushiai` links `SushiRuntime::core`, `SushiRuntime::api`,
`SushiBLAS::sushiblas` and `SushiLog::SushiLog` `PUBLIC`.

The [glossary](../reference/GLOSSARY.md) defines SushiStack as the set of repositories
SushiRuntime, SushiBLAS, SushiEngine, SushiAI and the hub. `hub` is the workspace CLI that
installs module CLIs and provisions several checkouts. The `sa` CLI is modelled on SushiEngine's
`se` and shares the stack's bundled toolchain. Section 2 of the
[architecture page](../architecture/ARCHITECTURE.md) describes how the two dependencies are
resolved.

## What do I need before I can build SushiAI?

The root [README](../../README.md) lists the requirements:

- A SYCL 2020 compiler: the bundled intel-llvm `clang++ -fsycl`, provisioned by SushiStack.
  SushiAI selects no SYCL toolchain of its own.
- CMake 3.20 or newer, and 3.22 for the test suite's `gtest_discover_tests`.
- GoogleTest for the test suite.
- SushiRuntime and SushiBLAS checkouts, most often as siblings `../sushiruntime` and
  `../sushiblas`.
- C++17.

`sa setup` provisions the C++ library dependencies and the SYCL toolchain into the shared root,
`~/.sushisystems` or the directory `SUSHISYSTEMS_HOME` names. `sa doctor` reports whether the
machine can build SushiAI.

## How do I install and build SushiAI?

Place the `sushiruntime`, `sushiblas` and `sushiai` checkouts side by side in one workspace,
then run:

```bash
hub install-cli sushiai   # puts `sa` and `sushiai` on your PATH
sa setup                  # provisions the toolchain and writes cli/config.local.toml
sa build                  # debug build, the default
sa test                   # unit, integration and regression suites
sa run sushiai_core_graph_tests   # one module's cases in one process
```

`sa` and `sushiai` are the same program. If a library checkout cannot be found, `sa build` stops
and names which one; `sa config` prints what the CLI resolved and where each
value came from. `sa build --type release` and `sa build --type relwithdebinfo` select the other
build types, each in a tree of its own, and `-D VAR=VALUE` passes a CMake cache variable through to the configure step.

The [CLI guide](CLI_GUIDE.md) covers every command. The root [README](../../README.md) also
gives a one-script quick start and the plain CMake invocation for building without the CLI.

## How do I run a first training run?

Run `sa demo mlp` after a build. It trains a two-layer MLP end to end and prints the compile
count, the loss per epoch and the final accuracy. `sa demo mlp --help` prints
the full flag list.

The demo trains on MNIST when the training split's two IDX files are under `data/mnist`, or
under the directory `--mnist-dir` names. Those files are not shipped and not downloaded, so
without them the demo trains on a seeded synthetic classification problem and says so in its
header and its result block. The [tutorial](../getting_started/TUTORIAL.md) explains each line
of the output.

## How do I use SushiAI from my own CMake project?

Install SushiAI to a prefix, then call `find_package` and link the exported target:

```cmake
find_package(SushiAI REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SushiAI::sushiai)
```

You must supply the same SYCL compiler every library in the chain was built with, a
`CMAKE_PREFIX_PATH` that names the install prefixes of SushiAI, SushiBLAS and SushiRuntime
unless all three share one prefix, and SushiRuntime's shared library on the loader path at run
time. The SYCL requirement holds even for a translation unit that only calls
`SushiAI::version_string()`, because linking `SushiAI::sushiai` brings in SushiBLAS's `-fsycl`
option.

`add_subdirectory(path/to/sushiai)` gives the same target; set `SA_BUILD_TESTS=OFF` before it to
keep SushiAI's suite out of your CTest run. `tests/package/` in the repository is a working
consumer. The [integration guide](INTEGRATION.md) holds the detail and a troubleshooting
section.

## Which C++ standard must my project use?

C++17, the same standard SushiAI, SushiBLAS and SushiRuntime are built to. A later standard does
not work for code that calls the library. `SushiRuntime::span` is `std::span` at C++20 and
SushiRuntime's own shim at C++17, and SushiAI exports out-of-line functions whose parameter type
is that span, among them `Autograd::differentiate`, `Core::allocate_tensor` and
`Optim::StepScalars::write`. A caller compiled at C++20 mangles those names differently and gets
an unresolved external at link time. `Train::build_classifier` calls `Autograd::differentiate`
for you, so the error can appear without a span in your own code.

The headers compile under C++20, and a translation unit that only builds or inspects a graph
links under either standard. If the rest of your program must be C++20, section 3 of the
[integration guide](INTEGRATION.md) describes putting the code that calls SushiAI in its own
C++17 target, and states that this is a workaround and not a supported configuration.

## Can I describe a model in a file instead of in C++?

Yes. `sushiai train` and `sushiai inference` read JSON:

```bash
sushiai train      applications/examples/heavy_mlp/train.json
sushiai inference  applications/examples/heavy_mlp/model.json \
                   applications/examples/heavy_mlp/heavy_mlp.sushi \
                   applications/examples/heavy_mlp/eval.json
```

`model.json` describes the architecture, `train.json` the objective, data, optimizer, epochs and
checkpoint, and `eval.json` the data to evaluate on. Each document leads with a version key. An
unknown key or `type` is an error that lists what is accepted, and relative paths resolve
against the configuration file's own directory. The
[CLI guide](CLI_GUIDE.md) shows both schemas.

A checkpoint written by a hand-written `Sequential` loads into a `LayerStack` built from the
equivalent `model.json`, and the reverse.

## How does a call report a failure?

A call that allocates, records, compiles or runs returns a `Status` or a `Result<T>`
(`failure/status.hpp`, `failure/result.hpp`). `AdamW`, `SGD`, `Trainer`, `LossScaler`, `StepScalars`
and the datasets are built by a static `make`, `CompiledStep` by `compile`, and the checkpoint
stream classes by `CheckpointWriter::create` and `CheckpointReader::open`; each returns a
`Result`. `initialize`, `save_checkpoint` and `load_checkpoint` return a `Status`.
`seam/seam.hpp` turns an exception of SushiBLAS or SushiRuntime into a status.

Code that touches no device returns a `Status` or a `Result` as well: the graph builder, the
layers' `forward`, `Autograd::differentiate` and `build_layer_stack`. Reading `value()` from a
failed `Result` is a misuse and stops the process.

## Does SushiAI support mixed precision?

Yes. `--precision mixed` runs the matrix products, and the bias and activation a fused GEMM
folds into them, in fp16. The master weights, gradients, reductions and the optimizer update
stay fp32. `--loss-scale S` adds gradient loss scaling: the backward seed is multiplied by `S`,
which must be a power of two, every gradient is screened for infinities and NaNs, and the scale
halves on overflow.

The documents make no speed claim for it. On the machine the README measured, which has no fp16
matrix unit, half arithmetic is emulated and the mixed run took 9364.9 ms against 2105.9 ms for
fp32. Both runs ended at a loss of 0.000380 and 100% accuracy. A GELU layer is not folded into
its GEMM under mixed precision. Section 5.3 of the
[architecture page](../architecture/ARCHITECTURE.md) gives the reasons.

## What does SushiAI not do yet?

The [architecture page](../architecture/ARCHITECTURE.md) names the open work in its section 7:
packing the lowering's scratch, the log-softmax and `K_NLL_LOSS` fusions, the standalone
`K_LOG_SOFTMAX` rule, the whole-step overflow skip, bfloat16, and the `Dataset` and `Trainer`
seams among them. [Known issues](../reference/KNOWN_ISSUES.md) records what behaves wrongly or
narrowly today:

- In mixed precision an overflowed gradient skips that parameter's update only; the other
  parameters still step.
- `gradcheck` cannot verify a mixed-precision graph. Check the rule in FLOAT64.
- NVIDIA Pascal GPUs (GTX 1060, compute capability 6.1) fail when the OpenCL loader ingests
  SPIR-V. The workaround is to set `ONEAPI_DEVICE_SELECTOR` to `opencl:cpu`.

## How is SushiAI licensed?

SushiAI is source-available and free for non-commercial use under the PolyForm Noncommercial
License 1.0.0. [`LICENSE`](../../LICENSE) is the binding text and lists the permitted purposes:
personal, non-commercial use, and use by the non-commercial organisations it names.

Any commercial purpose needs a separate, paid licence from Sushi Systems. That includes use
inside a company, use in paid client work, and use in a product or service that is sold.
[`COMMERCIAL.md`](../../COMMERCIAL.md) says how to ask for one: write to hello@sushisystems.io
with what you want to build and who will use it. Third-party components are listed in
[`NOTICE.md`](../../NOTICE.md).

## Where do I report a problem?

For anything that is not sensitive, open a GitHub issue or Discussion.
[Known issues](../reference/KNOWN_ISSUES.md) lists the defects already recorded.

Do not open a public issue for a security problem. Report it privately by email to
mustafagarip@sushisystems.io, through the community Discord, or with GitHub's private
vulnerability reporting under the repository's Security tab. The
[security policy](../reference/SECURITY.md) promises an acknowledgement within 72 hours and an
initial assessment within 7 days, and asks that a security bug specific to SushiBLAS or
SushiRuntime go to that repository.
