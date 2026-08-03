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
| `sa doxygen` | Generate the API reference — **not usable yet: no `Doxyfile` in this repo** |
| `sa demo mlp` | Train an MLP end to end and report the compile count |
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
sa demo mlp --precision mixed            # fp16 products, fp32 master weights
sa demo mlp --loss-scale 65536           # gradient loss scaling, halving on overflow
sa demo mlp --help                       # the full flag list
```

`--fusion` and `--no-fuse` are how the operator fusion pass is measured rather
than assumed: both spellings produce the same numbers and differ only in how
many kernels they launch, so the comparison has to come from one binary against
one SushiBLAS build. `--no-fuse` takes a form's name exactly as the demo's
`Fusion` line prints it.

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

## `sa clean` and `sa doxygen`

```bash
sa clean       # remove the build/ tree
sa doxygen     # generate the Doxygen API reference
```

**`sa doxygen` does not work yet.** It runs Doxygen against a repo-root
`Doxyfile`, and this repo has none — the command reports
`Doxyfile not found at repo root.` and exits 1. `Doxyfile` is also listed in
`.gitignore`, so authoring one means editing that first. Until a Doxyfile lands
(sushiruntime's is the model: a committed `Doxyfile` scoped to the public
header surface), treat this command as unimplemented.

When one does exist, Doxygen itself is an external tool, not vendored — install
it or point `doxygen_exe` in `cli/config.local.toml` at an existing binary; the
command prints the right install line if it cannot find one.

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
