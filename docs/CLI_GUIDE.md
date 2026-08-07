# The SushiAI CLI

SushiAI ships with a small command-line tool that drives everything you do
day to day: building the C++ library, running its tests, running the demo and
the example, and generating documentation. It is a thin wrapper around CMake
and CTest that reads your machine-specific toolchain paths from a config file
so you don't have to retype long compiler flags. It's modeled directly on
[SushiEngine](https://github.com/SushiSystems/SushiEngine)'s `se` CLI, not
SushiBLAS's `sb` (which still models an older, pre-toolchain-refresh build) —
except for `build -D`, which is `sb`'s and is mirrored here flag for flag.

## Installing the CLI

The CLI is a Python package that lives in the `cli/` folder. Install it once
and it puts two commands on your `PATH`:

- **`sa`** — short name.
- **`sushiai`** — long name.

They are identical; every example below uses `sa`.

```bash
ss install-cli sushiai      # always editable, against this checkout
```

This installs through **pipx** on every platform, into an isolated venv, then
injects the shared `sushicli` package — which is published to no index, so pipx
cannot resolve it as an ordinary dependency. The install is always editable and
there is no flag to make it otherwise: a frozen copy would stop tracking
`git pull`s on this checkout without ever saying so.

```bash
pipx uninstall sushiai-cli   # to remove it later
```

### How the CLI finds your compiler and siblings

SushiAI builds against the shared SushiStack SYCL toolchain — it selects no
toolchain of its own — and against **two sibling checkouts**, SushiRuntime
and SushiBLAS. The CLI reads all of this from:

- **`cli/config.toml`** — committed, shared defaults.
- **`cli/config.local.toml`** — your machine's absolute paths (gitignored).

In the common case — sibling `../sushiruntime` and `../sushiblas` checkouts
with the bundled toolchain already provisioned by `ss install`, or a
SushiStack workspace — no local config is needed at all. Run `sa config` to
see exactly what the CLI resolved.

## Command overview

| Command | What it's for |
|---------|----------------|
| `sa build` / `test` / `run` / `clean` | Build, test, run, and clean the C++ project |
| `sa doxygen` | Generate the Doxygen API reference into `docs/api-site/html/` |
| `sa demo mlp` | Train an MLP end to end and report the compile count |
| `sushiai train` | Train a model described by a JSON configuration |
| `sushiai inference` | Evaluate a trained checkpoint over a dataset |
| `sa config` | Show the resolved configuration |
| `sa env` | Show the environment your builds run under |

Run any command with `--help` to see its options.

## `sa build`

```bash
sa build                  # Release build (the default)
sa build --type debug     # Debug build
sa build --type relwithdebinfo
sa build --clean          # delete the build tree first, then build from scratch
```

The `--type` (`-t`) option accepts `release`, `debug`, or `relwithdebinfo`.

If either sibling can't be found, `sa build` fails fast with a clear message
naming which one and where it looked, rather than letting CMake's own
`add_subdirectory` error surface. `sa config` shows what the CLI resolved for
both `SushiRuntime` and `SushiBLAS`; set `sushiruntime_dir` / `sushiblas_dir`
in `cli/config.local.toml` (or `SUSHIRUNTIME_DIR` / `SUSHIBLAS_DIR` in the
environment) to point either elsewhere.

### `-D` / `--define` — CMake cache variables

```bash
sa build -D SA_BUILD_TESTS=OFF
sa build -D SB_SYCL_TARGETS="spir64;nvidia_gpu_sm_61" -D SA_BUILD_TESTS=OFF
```

Repeatable, `VAR=VALUE`, passed straight through to `cmake` — the same flag
`sb build` takes. A malformed entry is rejected before configure runs, with
the message naming what you typed.

Two things worth knowing:

- **The values are cached.** They persist in the build tree, so dropping the
  flag on a later run does *not* unset them. Pass `VAR=` to clear one, or
  `--clean` to start over.
- **They are appended last**, after every default the CLI sets, so an explicit
  `-D` is never silently outranked by one of them.

This is how AOT device targeting is reached: SushiAI `add_subdirectory`s
SushiBLAS, so SushiBLAS's `SB_SYCL_TARGETS` is a cache variable of *this*
build tree and there is no other sanctioned way to set it.

## `sa test`

```bash
sa test                          # functional (unit+integration+regression) — the default
sa test --suite unit
sa test --suite integration
sa test --suite regression
sa test --suite all              # every test
```

```bash
sa test --filter 'Version.*'  # ctest -R regex over 'Suite.Case' test names
sa test --repeat 5            # re-run each test up to 5x, stop on first failure
```

The three tiers come from the GTest suite-name prefix (`Unit_*`,
`Integration_*`, `Regression_*`) and nothing else — see
`tests/CMakeLists.txt`. `regression` holds the guarantees whose failure is
silent (see [CONTRIBUTING.md §3](CONTRIBUTING.md#3-building-and-testing-your-change)).

For GoogleTest-level options CTest doesn't expose, run the binary directly
with `sa run` and pass flags after `--`.

`sa test` needs a build with `SA_BUILD_TESTS=ON`, which is the default; a tree
configured with `-D SA_BUILD_TESTS=OFF` has no SushiAI tests registered.

## `sa run`

Runs a built executable. With no target it runs the configured default
(`sushiai_example`).

```bash
sa run                                # run the default target
sa run sushiai_example
sa run --sort                         # interactively pick from the list of executables

sa run sushiai_tests -- --gtest_filter='Unit_Version.*'
```

## `sa demo` — the AI-7 showcase

```bash
sa demo mlp                              # the defaults
sa demo mlp --epochs 4 --optimizer sgd   # anything you like
sa demo mlp --fusion off                 # the unfused graph, for an A/B
sa demo mlp --no-fuse gemm+bias+relu     # one catalog entry off; repeatable
sa demo mlp --activation gelu            # the epilogue that spills its input
sa demo mlp --precision mixed            # fp16 products, fp32 master weights
sa demo mlp --loss-scale 65536           # gradient loss scaling, halving on overflow
sa demo mlp --help                       # the full flag list
```

`--fusion` and `--no-fuse` are how the operator fusion pass is measured rather
than assumed: both spellings produce the same numbers and differ only in how
many kernels they launch, so the comparison has to come from one binary against
one SushiBLAS build. `--no-fuse` takes a form's name exactly as the demo's
`Fusion` line prints it.

`--activation` picks the hidden layer's nonlinearity, and the two it offers are
the two *kinds* rather than a sample of the library's four. `relu` — the
default — has a gradient that reads its own output, so folding it into the GEMM
epilogue destroys nothing and needs nothing extra. `gelu` has a gradient that
reads its **input**, so `gemm+bias+gelu` only exists because its epilogue
spills that input to a second buffer on the way past (cuBLASLt calls the shape
`GELU_AUX_BIAS`). Measured here on the default schedule, at `--precision fp32`:

| run | `ops per step` | buffers | final loss |
|---|---|---|---|
| `--activation gelu` | 25 | 112.7 KiB | 0.000343 |
| `--activation gelu --no-fuse gemm+bias+gelu` | 27 | 112.7 KiB | 0.000343 |
| `--activation gelu --fusion off` | 35 | 112.7 KiB | 0.000343 |

Two launches saved against the fallback and ten against the unfused graph, at
**no cost in memory at all** — the spill is not a new buffer, it is the
pre-activation the unfused `BIAS_ADD` would have written anyway, kept instead of
eliminated. The identical final loss across all three is the point of the
exercise: the spill is what makes the gradients agree, and
`Regression_Fusion.AGeluMlpTrainsIdenticallyFusedAndUnfused` pins it as a
byte-for-byte parameter comparison rather than as six printed decimals.

Under `--precision mixed` the GELU layer is **not** folded — the fusion line
shows `gemm+bias` twice — because the tanh approximation computes `x*x*x`,
which overflows fp16 at |x| ≈ 40.3. See
[ARCHITECTURE.md §6.2](ARCHITECTURE.md) for why that is a structural
consequence of the pass order rather than a special case.

`--precision` is the same idea for mixed precision. `fp32` — the default —
means every operation computes in the graph's own element type and the cast
pass inserts nothing, so nothing about an existing run changes. `mixed` runs
the matrix products and the epilogues a fused GEMM folds into them in fp16,
keeping fp32 master weights, fp32 gradients, fp32 reductions and an fp32
optimizer update; the demo's `Precision` line reports how many conversions
that cost and in which direction.

**On this machine `--precision mixed` is a correctness switch, not a speed
one.** There is no fp16 matrix unit here, so half arithmetic is emulated and
the mixed run is several times slower — see
[README.md § Mixed precision](../README.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here)
for the measured figures. What the flag is for is running the two arms from
one binary and comparing the *loss curves*.

`--loss-scale S` turns on gradient loss scaling: the backward seed is
multiplied by `S`, every gradient is screened for infinities and NaNs, an
overflowed parameter's update is skipped, the scale halves, and 2000 clean
steps double it. `S` must be a power of two so that the unscale before the
update is exact — the demo run at `--precision fp32 --loss-scale 65536`
reports a loss identical to the unscaled one, which is that exactness being
visible rather than a coincidence.

**It makes this demo no better, and that is the honest reading.** fp16 here
neither stalls nor underflows, so the flag is for watching the machinery:
`--precision mixed --loss-scale 65536` overflows once and backs off to 32768
(PyTorch's default scale is too high for this model's fp16 backward pass), and
`--loss-scale 16777216` overflows on every step and walks the scale down eight
halvings — with `compiles: 1` throughout, which is what the scale being a
device-resident value rather than a graph attribute buys.

`sa demo mlp` runs `sushiai_demo_mlp` (`examples/demo_mlp.cpp`) from the repo
root, so a relative data path such as the demo's `data/mnist` default means
the same thing wherever you invoked `sa` from. It trains an MLP end to end and
reports the compile count, the per-epoch loss trajectory, the final accuracy
and the wall clock. See
[README.md § The demo](../README.md#the-demo-sa-demo-mlp) for real output.

Two deliberate choices:

- **The demo is its own binary, not `sushiai_example` with flags.** A demo is
  a supported entry point whose flags and printed lines are a contract the
  test suite pins; an example is a teaching file people are meant to edit.
  `examples/demo_mlp.cpp`'s header comment argues it in full.
- **Every flag is forwarded verbatim, including `--help`.** The demo owns its
  flag surface; restating it in Typer would give two places to change and one
  to forget. `sa demo --help` still lists the available demos.

## `sushiai train` and `sushiai inference` — config-driven runs

```bash
sushiai train      examples/heavy_mlp/train.json
sushiai inference  examples/heavy_mlp/model.json  heavy_mlp.ckpt  examples/heavy_mlp/eval.json
```

`sa` and `sushiai` are the same program (`cli/pyproject.toml` binds both), so
these work under either name; they are spelled `sushiai` here because that is
how the design document names them.

### What the two commands are for

A model's architecture lives in a JSON file rather than in the C++ type
system, and these two drivers consume it. The point is not convenience: it is
that `apps/train.cpp` contains **no layer, loss, dataset or optimizer type
name** and therefore never has to be edited as the library grows. Every one of
those arrives through a constant registry table keyed by a string out of the
configuration, so adding a convolution or a detection loss is a table row and
a factory function in the library.

### The three documents

| File | Describes | Named by |
|---|---|---|
| `model.json` | the architecture alone | `train.json`, and `inference`'s first argument |
| `train.json` | objective, data, optimizer, epochs, checkpoint | `sushiai train` |
| `eval.json` | the data to evaluate on | `inference`'s third argument |

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

```json
{
  "sushiai_train": 1,
  "model": "model.json",
  "objective": { "type": "cross_entropy" },
  "data":      { "type": "rings", "batch_size": 64, "batches": 64,
                 "classes": 4, "seed": 3735928559 },
  "optimizer": { "type": "adamw", "learning_rate": 3e-3, "weight_decay": 0.01 },
  "epochs": 50,
  "checkpoint": "heavy_mlp.ckpt"
}
```

Four things about the schema are worth knowing before you write one, and each
closes a way of being silently wrong:

- **Each document leads with a version key.** A format change is then an error
  rather than a misread.
- **`units` is an output width only.** The input width is chained from
  `input.features`, so a dimension mismatch between adjacent layers is not
  expressible.
- **An unknown key is an error, not an ignored one.** A typed `"unitss": 256`
  must not train silently against a default, so the message lists what the
  layer *does* accept. An unknown `type` likewise lists every type the build
  has.
- **Relative paths resolve against the configuration file's own directory**,
  not your working directory — so `sushiai train examples/heavy_mlp/train.json`
  works from the repository root and a configuration directory can be moved
  without editing what is inside it.

### Why inference takes its data as an argument

The architecture file carries no data block, because it describes a model. So
`inference` needs to be told what to run on, and an optional flag that always
errored when omitted would be a lie about optionality. It traces the forward
pass only — no loss, no backward pass, no optimizer — and reports the
predicted-class distribution, which is the one summary that needs neither a
loss nor label semantics.

### Exit codes and output

Both exit `0` on success and non-zero on any configuration or runtime error,
printing the library's own diagnostic — which names the file, the key and what
was found. Both print a `compiles: 1` line; two CTest cases pin it, the same
way `sa demo mlp`'s is pinned.

## `sa clean` and `sa doxygen`

```bash
sa clean       # remove the build/ tree
sa doxygen     # generate the Doxygen API reference
```

It runs Doxygen against the repo-root `Doxyfile`, which is **committed** — for
a while it was not, and `.gitignore` listed it, so the command reported
`Doxyfile not found at repo root.` and exited 1 in every fresh checkout.
sushiruntime commits its own for the same reason.

Output lands in `docs/api-site/html/`, and warnings — every public symbol
missing a doc comment — in `docs/api-site/doxygen-warnings.log`. Both are
gitignored; the log is where the documentation debt is read off, and
`WARN_AS_ERROR` is deliberately `NO` so a missing `@param` cannot fail a docs
build.

Scope is `include/SushiAI` minus `Detail`. Unlike sushiruntime, which curates a
fluent `api/` subtree, SushiAI has no such split: every installed header is
surface a consumer writes against.

Doxygen itself is an external tool, not vendored — install it or point
`doxygen_exe` in `cli/config.local.toml` at an existing binary; the command
prints the right install line if it cannot find one. The stack's provisioned
toolchain ships one, so `ss install` is usually enough.

## `sa config` and `sa env` — diagnostics

```bash
sa config         # print the resolved config and where each value came from
sa env            # print the environment cmake/ctest/run subprocesses run under
sa env --all      # show every variable, not just the build-relevant ones
```

`sa config` prints both resolved sibling directories, the compiler, and the
vcpkg root, plus which source (default / `config.toml` / `config.local.toml`
/ an `SA_*` environment variable) supplied each field.

## Building without the CLI

The CLI is a convenience, not a requirement. See **Building with CMake
directly** in the top-level [README.md](../README.md#building-with-cmake-directly).
