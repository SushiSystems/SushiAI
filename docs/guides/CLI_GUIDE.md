# The SushiAI CLI

SushiAI ships with a small command-line tool that drives everything you do
day to day: building the C++ library, running its tests, running the demo and
the example, and generating documentation. It is a thin wrapper around CMake
and CTest that reads your machine-specific toolchain paths from a config file
so you don't have to retype long compiler flags.

## Installing the CLI

The CLI is a Python package that lives in the `cli/` folder. Install it once
and it puts two commands on your `PATH`:

- **`sa`** — short name.
- **`sushiai`** — long name.

They are identical; every example below uses `sa`.

```bash
hub install-cli sushiai      # always editable, against this checkout
```

This installs through **pipx** on every platform, into an isolated venv.
`sushicore`, the package the stack's CLIs share, is a PyPI dependency declared
in `cli/pyproject.toml`, so pipx resolves it with the rest. The install is
always editable and there is no flag to make it otherwise: a frozen copy would
stop tracking `git pull`s on this checkout without ever saying so.

```bash
pipx uninstall sushiai-cli   # to remove it later
```

### How the CLI finds your compiler and siblings

SushiAI builds against the shared SushiStack SYCL toolchain — it selects no
toolchain of its own — and against **two sibling checkouts**, SushiRuntime
and SushiBLAS. The CLI reads all of this from:

- **`cli/config.toml`** — committed, shared defaults.
- **`cli/config.local.toml`** — your machine's absolute paths (gitignored).

With sibling `../sushiruntime` and `../sushiblas` checkouts, or inside a
SushiStack workspace, run `sa setup` once: it provisions the toolchain and
writes `cli/config.local.toml` for you. `hub install` does the same for
several checkouts at once. Run `sa config` to see what the CLI resolved.

## Command overview

| Command | What it's for |
|---------|----------------|
| `sa build` / `test` / `run` / `clean` | Build, test, run, and clean the C++ project |
| `sa docs` | Build the Doxygen API reference into `build/docs/api-site/html/` |
| `sa docs bundle` | Pack the published manual pages and the API reference into one archive |
| `sa demo mlp` | Train an MLP end to end and report the compile count |
| `sushiai train` | Train a model described by a JSON configuration |
| `sushiai inference` | Evaluate a trained checkpoint over a dataset |
| `sushiai import` | Write a model of another framework as a model, a weight and a training file |
| `sushiai tokenize` | Learn a byte-pair vocabulary from a corpus, or encode a corpus with one |
| `sa config` | Show the resolved configuration |
| `sa env` | Show the environment your builds run under |
| `sa setup` | Provision the dependencies and write the tool paths |
| `sa doctor` | Report whether this machine can build SushiAI |
| `sa link` / `unlink` | Add this checkout to a workspace's module registry, or remove it |

Run any command with `--help` to see its options.

`sa --version` prints `sushiai-cli` and its version. `sa --describe` prints the
command catalogue as JSON: every command, its options and their defaults.

A failure you can act on, such as a `cli/config.toml` that is not valid TOML,
ends in one line naming the file and exit code 1. Ctrl+C while a build, a test
run or a launched program is running stops it and exits 130.

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
`sb build` takes. A malformed entry exits 2 before the build tree is touched,
`--clean` included, with the message naming what you typed.

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
sa test --suite all              # every CTest test
sa test --suite package          # the out-of-tree package consumer (no CTest)
```

```bash
sa test --filter 'Version.*'  # ctest -R regex over 'Suite.Case' test names
sa test --repeat 5            # re-run each test up to 5x, stop on first failure
```

The three tiers come from the GTest suite-name prefix (`Unit_*`,
`Integration_*`, `Regression_*`) and nothing else — see
`tests/CMakeLists.txt`. `regression` holds the guarantees whose failure is
silent (see [CONTRIBUTING.md §3](../CONTRIBUTING.md#3-building-and-testing-your-change)).

For GoogleTest-level options CTest doesn't expose, run the binary directly
with `sa run` and pass flags after `--`.

`sa test` needs a build with `SA_BUILD_TESTS=ON`, which is the default; a tree
configured with `-D SA_BUILD_TESTS=OFF` has no SushiAI tests registered.

### `--suite package`

Every other suite is a CTest label. This one is not, and runs no CTest at all.
It installs SushiAI to `build/package/prefix`, configures `tests/package/`
into `build/package/consumer` against that prefix, builds it and runs it —
the way a third party consumes the library, through `find_package(SushiAI)`,
with no access to this source tree. Both intermediates live under `build/`,
so `sa clean` reclaims them.

This is the only test in the repo that can see what `find_package(SushiAI)`
actually hands somebody. Every in-tree test includes individual headers from
the source tree, so a public header that the umbrella names but
`install(DIRECTORY)` does not ship, or that only compiles because a sibling
checkout happened to sit next to it, is invisible to all of them.

**What it does not cover**, and this is worth being exact about since the
suite was written in response to a header bug: the suite compiles what
`SushiAI/SushiAI.hpp` names, so it catches *referenced but not installed*. The
inverse — *installed but never referenced by the umbrella*, which is the actual
bug that prompted this work, four headers missing from the umbrella — it cannot
catch by construction. A header the umbrella does not name is simply never
compiled, and nothing fails. That half is checked at configure time instead, in
`src/CMakeLists.txt`: every `include/SushiAI/**/*.hpp` must appear in the
umbrella or the configure fails with the list. The two checks are
complementary and neither subsumes the other.

It also checks what the exported CMake interface *does not* carry — warnings,
`-Werror`, sanitizers, `-march=`, the in-tree policy targets, and the config
layer's private `nlohmann_json`. Those assertions are in
`tests/package/CMakeLists.txt`, because what a consumer inherits is only
observable from the consumer side.

Three things the CLI does on the consumer's behalf, all host-environment
rather than package concerns:

- One install prefix covers all three modules. SushiBLAS and SushiRuntime are
  `add_subdirectory`'d into this build (see `cmake/BLAS.cmake`), so their
  install rules run as part of this one and the `find_dependency` calls in
  `SushiAIConfig.cmake` resolve from the same prefix.
- The consumer configure reuses the same compiler and host-tool flags as the
  main configure, from one shared helper (`_toolchain_args`), so the two
  cannot drift. `CMAKE_PREFIX_PATH` is the deliberate exception — the consumer
  must also see the install prefix. `SUSHIRUNTIME_DIR` / `SUSHIBLAS_DIR` are
  excluded too, for a stronger reason: handing the consumer the sibling source
  trees is the shortcut this suite exists to forbid.
- On Windows the built binary needs `sushiruntime.dll` **and `hwloc-15.dll`**
  beside it or it exits `0xC0000135` before reaching `main`, which looks like a
  package failure and is not one. hwloc is linked PRIVATE by SushiRuntime and
  so never appears on a consumer's link line — but PRIVATE is a link-time
  statement and the loader does not read `INTERFACE_LINK_LIBRARIES`; hwloc is
  in `sushiruntime.dll`'s import table either way. In-tree binaries get it from
  PATH (`sa` puts the vcpkg installed `bin` there, see `cli/sushiai/env.py`);
  the CLI stages both DLLs for the consumer instead, and runs it with that PATH
  entry removed so the staging is actually load-bearing rather than shadowed by
  the developer's environment.

Both `build/package/prefix` and `build/package/consumer` are deleted before
each run: a stale prefix would let a header deleted from the tree keep
satisfying an include, and a stale consumer directory would let a DLL staged by
an earlier run keep satisfying the loader.

It is **not** part of `--suite all`, which covers only the CTest tests. A full
install plus a second configure and build costs minutes and is rarely what you
want mid-loop, so `all` prints a line saying the package suite is not included
rather than leaving the gap to be discovered. `--filter` and `--repeat` are
CTest knobs; passing either with `--suite package` exits 2 instead of being
silently ignored.

The consumer compiles at C++17 — the same standard as the library, and
deliberately not SushiBLAS's C++20 consumer. `tests/package/CMakeLists.txt`
records the measured reason (SushiAI exports out-of-line functions whose
parameter type is `SushiRuntime::span`, so the C++17/C++20 spelling difference
reaches the mangled name); see also [INTEGRATION.md](INTEGRATION.md) §3.

## The memory ceiling of `sa test` and `sa run`

Both commands cap the memory that the programs they start may commit. The ceiling is
`--memory-limit GB`, else the variable `SA_MEMORY_LIMIT_GB`, else half of the machine's
physical memory, and `--memory-limit 0` applies none. The command prints the ceiling in
force. A test that reaches it fails with `out_of_memory` and the size it asked for. The
ceiling is enforced on Windows only.

```
sa run sushiai_vision_importers_tests --memory-limit 6 -- --gtest_filter=Integration_Darknet*
```

## `sa run`

Runs a built executable. With no target it runs the configured default
(`sushiai_example`).

```bash
sa run                                # run the default target
sa run sushiai_example
sa run --sort                         # interactively pick from the list of executables

sa run sushiai_core_version_tests -- --gtest_filter='Unit_Version.*'
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
`Regression_Fusion.AGELUMLPTrainsIdenticallyFusedAndUnfused` pins it as a
byte-for-byte parameter comparison rather than as six printed decimals.

Under `--precision mixed` the GELU layer is **not** folded — the fusion line
shows `gemm+bias` twice — because the tanh approximation computes `x*x*x`,
which overflows fp16 at |x| ≈ 40.3. See
[ARCHITECTURE.md §5.3](../architecture/ARCHITECTURE.md#53-mixed-precision-is-structure-in-the-ir-not-a-runtime-switch) for why that is a structural
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
[README.md § Mixed precision](../../README.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here)
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
[README.md § The demo](../../README.md#the-demo-sa-demo-mlp) for real output.

Two deliberate choices:

- **The demo is its own binary, not `sushiai_example` with flags.** A demo is
  a supported entry point whose flags and printed lines are a contract the
  test suite pins; an example is a teaching file people are meant to edit.
  `examples/demo_mlp.cpp`'s header comment argues it in full.
- **Every flag is forwarded verbatim, including `--help`.** The demo owns its
  flag surface; restating it in Typer would give two places to change and one
  to forget. `sa demo --help` still lists the available demos. Until the demo
  is built there is nothing to forward to, so `sa demo mlp --help` prints `sa`'s
  own page and then says to run `sa build`. `sa train` and `sa inference` do
  the same.

## `sushiai train` and `sushiai inference` — config-driven runs

```bash
sushiai train      examples/heavy_mlp/train.json
sushiai inference  examples/heavy_mlp/model.json \
                   examples/heavy_mlp/heavy_mlp.sushi \
                   examples/heavy_mlp/eval.json
```

`sa` and `sushiai` are the same program (`cli/pyproject.toml` binds both), so
these work under either name; they are spelled `sushiai` here because that is
how the design document names them.

[`examples/mnist_cnn/`](../../applications/examples/mnist_cnn/README.md) holds a second set of the three
files, for a convolutional network on MNIST. Its README says where the IDX files go.

### What the two commands are for

A model's architecture lives in a JSON file rather than in the C++ type
system, and these two drivers consume it. The point is not convenience: it is
that `apps/train.cpp` contains **no layer, loss, dataset or optimizer type
name** and therefore never has to be edited as the library grows. Every one of
those arrives through a constant registry table keyed by a string out of the
configuration, so adding a layer type or a detection loss is a table row and
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
  "checkpoint": "heavy_mlp.sushi"
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
  without editing what is inside it. This holds for `model`, `weights` and
  `checkpoint`.
  The `data` block is passed to its factory as written, so the `directory` of
  an `mnist` dataset resolves against the working directory of the process.

### Starting weights, a schedule and data from files

Five optional parts of `train.json`:

| Key | Holds |
| --- | --- |
| `weights` | A `.sushi` file loaded by name before training, in place of the initialiser. A name, shape or element type that differs is an error |
| `resume` | A `.sushi` checkpoint a run of the same model, optimizer and `seed` wrote. It loads the model state as `weights` does and the run state with it: the optimizer's step count and its momentum buffers or moments, and where the run stood. A file states `weights` or `resume`; both is an error |
| `seed` | An integer of 0 or above, 0 when absent. It seeds the random sites of the training graph, which today are the dropout layers. The `seed` of a `data` block is a different key: it seeds the data |
| `schedule` | `{ "type": "step", "base_rate": ..., "warmup_iterations": ..., "warmup_power": ..., "steps": [...], "scales": [...] }`. A file with a schedule states its rate there: `learning_rate` in `optimizer` is then an error |
| `data` with `reader` | A dataset read from files, in place of `type` |

A `data` block with `reader`:

| Key | Default | Meaning |
| --- | --- | --- |
| `reader` | required | `yolo_txt`, `image_folder`, `idx` or `cifar` |
| `task` | required | `detection` or `classification` |
| `root` | required | The dataset's directory, resolved against the working directory |
| `split` | `"train"` | As the reader takes it |
| `batch_size` | 16 | |
| `input` | required | `[height, width]` of the network input |
| `channels` | 3 | |
| `classes` | The count the samples name | `yolo_txt` requires it |
| `max_objects` | required for `detection` | The object slots of one image; an image with more is an error |
| `cache`, `cache_side` | none, 0 | A directory of raw images written on the first pass and read afterwards; `cache_side` bounds the longest side |
| `shuffle`, `seed` | false, 0 | A sample order drawn anew at each epoch from the seed |
| `pipeline` | `"none"` | `"default"` runs mosaic, the affine warp, flips and colour on each sample |
| `workers`, `prefetch` | 0, the worker count | Threads that read batches ahead, and how many batches they hold |

`applications/examples/yolov7_tiny/` holds a training file that uses all three.

### Resuming a run

`sushiai train` writes run state into every checkpoint, so the file `checkpoint` names is
also the file a later run can state as `resume`:

```json
"resume": "heavy_mlp.sushi",
"checkpoint": "heavy_mlp.sushi"
```

The resumed run goes on at the iteration the saved one stopped at. A `schedule` reads that
count, so warm-up is not repeated and a step that was passed stays passed; AdamW keeps its
moments and SGD its momentum.

It also goes on at the epoch index the saved one stopped at. A dataset with `shuffle` draws
the order of an epoch from its own seed and the epoch index, so the resumed run reads the
orders the saved run would have read next, and a dropout layer draws the masks it would have
drawn next. A run of 20 epochs, saved and resumed for 30 more, trains on the same batches
under the same masks as one run of 50. `epochs` counts the epochs of this run, not of both.

The `seed` has to be the one the saved run stated. A checkpoint written under `"seed": 7`
and resumed under `"seed": 8`, or under no `seed`, is refused with both numbers.

The optimizer has to be the one that wrote the file. A checkpoint of `sgd` resumed under
`adamw` is refused with both names, and so is SGD with momentum resumed from a run without
it, with the name of the record that is missing. A weight file with no run state, such as
the `weights.sushi` of `sushiai import` or a checkpoint written before version 2 of the
format, is refused as `resume` and says to state it as `weights`. So is a checkpoint whose
run state holds no `run.seed` record, which a build from before the seed existed wrote.

`sushiai inference` and `weights` read the same file and leave the run state unread.

### Layer types

`NN::LAYER_REGISTRY` in `include/SushiAI/nn/layer_registry.hpp` holds thirteen
rows. The shapes below are per sample and leave out the batch axis. The chain
starts at `[input.features]`, and each layer receives the shape the layer
before it reports.

| `type` | Keys | Input | Output |
|---|---|---|---|
| `linear` | `units` (required, positive), `use_bias` (default `true`) | `[features]` | `[units]` |
| `gelu`, `relu`, `tanh`, `sigmoid` | none | any shape | the same shape |
| `leaky_relu` | `slope` (default `0.01`, inside the open range 0 to 1) | any shape | the same shape |
| `unflatten` | `height`, `width`, `channels` (all required) | `[height * width * channels]` | `[height, width, channels]` |
| `conv2d` | `filters`, `kernel` (required); `stride` (default `1`), `padding` (default `0`), `dilation` (default `1`), `use_bias` (default `true`) | `[H, W, C]` | `[out_h, out_w, filters]` |
| `maxpool2d`, `avgpool2d` | `kernel` (required); `stride` (default: the kernel), `padding` (default `0`) | `[H, W, C]` | `[out_h, out_w, C]` |
| `global_avgpool` | none | `[H, W, C]` | `[C]` |
| `flatten` | none | any shape | `[element count]` |
| `upsample` | `scale` (default `2`, at least `1`) | `[H, W, C]` | `[H * scale, W * scale, C]` |

`kernel`, `stride`, `padding` and `dilation` each take an integer, which sets
both axes, or a two-entry `[h, w]` array. A padding applies to both sides of
its axis. A convolution reads its input channel count from the shape before
it, so there is no key for it.

Images enter a model flat. A model that convolves starts with `unflatten`,
and a `linear` layer after an image-shaped layer needs a `flatten` before it.
The factories refuse both mistakes and name the missing layer.

### Dataset types

`Data::DATASET_REGISTRY` in `include/SushiAI/data/dataset.hpp` holds two rows.

| `type` | Keys |
|---|---|
| `rings` | `batch_size` (default `64`), `batches` (`64`), `classes` (`4`), `distractors` (`4`), `ring_width` (`1.0`), `position_noise` (`0.12`), `seed`; all optional |
| `mnist` | `directory` (required); `split` (`"train"` or `"test"`, default `"train"`), `batch_size` (default `64`), `max_samples` (default `4096`; `0` loads every sample) |

`mnist` reads the four uncompressed IDX files under their original names and
stops with an error that names them when the split's two files are not in
`directory`.

### Why inference takes its data as an argument

The architecture file carries no data block, because it describes a model. So
`inference` needs to be told what to run on, and an optional flag that always
errored when omitted would be a lie about optionality. It traces the forward
pass only — no loss, no backward pass, no optimizer.

It reports the predicted-class distribution **and** top-1 agreement with the
labels the dataset supplies. The distribution alone would be useless: on
class-balanced data a perfect model and a random one both give a uniform
spread. Reading the labels names no loss and no objective — labels are part of
the `Data::Dataset` seam, and top-1 agreement is a property of the data and
the output rather than of what the model was trained to minimise. It is
reported only when the model's output width matches the dataset's class count,
because otherwise the two are not comparable and a number would be fiction.

### Exit codes and output

Both exit `0` on success and non-zero on any configuration or runtime error,
printing the library's own diagnostic — which names the file, the key and what
was found. Both print a `compiles: 1` line; two CTest cases pin it, the same
way `sa demo mlp`'s is pinned.

## `sushiai import` — a model of another framework

```bash
sushiai import darknet yolov7-tiny.cfg yolov7-tiny.weights --output models/yolov7_tiny
```

| Argument | Meaning |
| --- | --- |
| `<format>` | The importer to read with. `darknet` is the one a build with the vision domain has; a format with no importer is refused with the formats that have one |
| `<files...>` | The files of the other framework, in the order the format states them. `darknet` takes a `.cfg` and optionally a `.weights` |
| `--output`, `-o` | The directory that receives the files; it is created when absent |
| `--force` | Replace a file that is already there. Without it an existing file stops the import before anything is written |

It writes three files into the directory:

| File | Holds |
| --- | --- |
| `model.json` | The architecture, with its task section |
| `weights.sushi` | The weights, when the foreign files held any |
| `train.json` | `model`, `weights`, `"data": null`, `"checkpoint": "trained.sushi"` and the training recipe the foreign files state |

The run prints the format, each input file, the size of the model and the tensor count,
one `wrote <file>` line per file, and then a list of notes. A note names each value of the
foreign files that none of the three files holds; for Darknet these are `batch`,
`subdivisions`, `max_batches` and the augmentation keys.

`train.json` is not ready to run. No foreign file names a dataset, so the import writes
`"data": null`, and `sushiai train` refuses the file until you fill the block. Without a
`.weights` file there is no `weights` key and a run initialises the model.

The command creates no engine and needs no device.

## `sushiai tokenize` — a vocabulary and a token file

```bash
sushiai tokenize train  corpus.txt --vocabulary-size 4096 --output vocab.savb
sushiai tokenize encode corpus.txt --vocabulary vocab.savb --output corpus.satk
```

The command exists in a build with the language domain.

| Argument | Verb | Meaning |
| --- | --- | --- |
| `<corpus>` | both | A text file; its bytes are read as they are. Exactly one |
| `--vocabulary-size <N>` | `train` | The tokens of the learned vocabulary, bytes and end-of-text included |
| `--vocabulary <file>` | `encode` | The vocabulary file `tokenize train` wrote |
| `--output <file>` | both | The file written |
| `--force` | both | Replace an output file that is already there. Without it an existing file stops the run before anything is read |

`train` learns a byte-pair vocabulary and prints the corpus with its size in bytes, the
token count with how many of the tokens are merges, and `wrote <file>`. `encode` prints the
corpus, the vocabulary with its token count, the number of tokens written, and
`wrote <file>`. The token file is what a `tokens` data block of a training file names.

`train` holds the whole corpus in memory. `encode` reads it a mebibyte at a time and holds
the ids, four bytes a token, until it writes them.

A verb other than `train` and `encode`, a missing corpus or output, a second corpus and a
flag with no value are each refused with the usage line.

## `sa clean` and `sa docs`

```bash
sa clean       # remove the build/ tree
sa docs        # build the Doxygen API reference
sa docs bundle --release 1.2.3   # pack the documentation bundle
```

`sa clean` removes the generated API reference too, because it lives under
`build/`. If the tree is still there afterwards, for example because a program
holds a file in it open, the command names the tree and exits 1.

`sa docs` was `sa doxygen`. The old word still runs, prints one notice naming
the new one when it runs on a terminal, and is removed after 2027-03-25; `SA_NO_DEPRECATION_NOTICE=1`
silences the notice.

It runs Doxygen against the repo-root `Doxyfile`, which is **committed** — for
a while it was not, and `.gitignore` listed it, so the command reported
`Doxyfile not found at repo root.` and exited 1 in every fresh checkout.
sushiruntime commits its own for the same reason.

Output lands in `build/docs/api-site/html/`, and warnings — every public symbol
missing a doc comment — in `build/docs/api-site/doxygen-warnings.log`. Both are
gitignored; the log is where the documentation debt is read off, and
`WARN_AS_ERROR` is deliberately `NO` so a missing `@param` cannot fail a docs
build.

Scope is `include/SushiAI` minus `Detail`. Unlike sushiruntime, which curates a
fluent `api/` subtree, SushiAI has no such split: every installed header is
surface a consumer writes against.

Doxygen itself is an external tool, not vendored — install it or point
`doxygen_exe` in `cli/config.local.toml` at an existing binary; the command
prints the right install line if it cannot find one. `sa setup` installs one
with the rest of the shared tools.

`sa docs bundle --release X.Y.Z` writes `docs-bundle-X.Y.Z.tar.gz` and its
`.sha256` file to `build/docs/bundle/`; `--out` names another folder. The
archive is what `docs.sushisystems.io` reads. `docs/publish.toml` lists the
sections of `docs/` that go into it and the pages that stay out. Because that
file sets `api = true`, the command runs the same Doxygen build as `sa docs`
and adds the XML from `build/docs/api-site/xml/`. It stops, naming the page
and the line, when a published page has no level-one heading, has no link from
`docs/README.md`, or links to a file that does not exist. Without `--release`
it exits 2. The release is three integers: `v1.2.3` and `1.2.3-rc1` are
refused.

## `sa config` and `sa env` — diagnostics

```bash
sa config         # print the resolved config and where each value came from
sa env            # print the environment cmake/ctest/run subprocesses run under
sa env --all      # show every variable, not just the build-relevant ones (-a for short)
```

`sa config` prints both resolved sibling directories, the compiler, and the
vcpkg root, plus the source of each field: `default`, `config.toml` or
`env:<VARIABLE>`. A value from `cli/config.local.toml` is reported as
`config.toml`, and a value from the workspace file as `default`.

## `sa setup`

```bash
sa setup --dry-run   # report what is missing and what would be written
sa setup             # provision it
```

Provisions what `cli/sushistack.deps.toml` lists, and what the fragments of
SushiRuntime and SushiBLAS list, into the shared root, `~/.sushisystems` (or
`SUSHISYSTEMS_HOME`). It then writes the tool paths it found to
`cli/config.local.toml` and prints the doctor report. It exits 2 and names the
module when SushiRuntime or SushiBLAS has no checkout.

- `--dry-run` reports what is missing and what would be written, and changes nothing.
- `--yes` skips the confirmation prompt before the intel/llvm download.
- `--toolchain <name>` also installs that toolchain. Repeatable.
- `--no-gpu` skips the GPU toolkit.

`hub install` provisions several checkouts at once; `sa setup` covers this one
and the two it builds on.

## `sa doctor`

Reports whether this machine can build SushiAI, one row per check, and exits
non-zero when a required check fails.

- `--for GROUP` runs only that group's checks and counts its optional checks as required.

## `sa link` and `sa unlink`

`sa link` records this checkout in a SushiStack workspace's module registry;
`sa unlink` removes it.

- `--workspace <path>` names the workspace root.

## Building without the CLI

The CLI is a convenience, not a requirement. See **Building with CMake
directly** in the top-level [README.md](../../README.md#building-with-cmake-directly).
