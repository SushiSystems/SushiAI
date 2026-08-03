# Changelog

All notable changes to SushiAI are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — versions follow [Semantic Versioning](https://semver.org/).

The project has not yet been assigned a semantic version — it is in
prototype stage. Formal per-release entries begin at the v1.0.0 cut; until
then, `[Unreleased]` records the prototype-era milestones informally. See
[CONTRIBUTING.md §5](CONTRIBUTING.md#5-documentation-is-part-of-the-change).

---

## [Unreleased]

### Added
- **AI-7 (first cut): `sa demo mlp` runs a real training program.** The
  command used to print "not implemented yet" and exit 1; it now runs
  `sushiai_demo_mlp` (`examples/demo_mlp.cpp`), a flag-driven MLP classifier
  that trains end to end and reports the compile count, the per-epoch loss
  trajectory, the final accuracy and the wall clock.
  - A **separate binary** from `sushiai_example`, not a wrapper over it: a
    demo is a supported entry point whose flags and printed lines are a
    contract, an example is a teaching file people are meant to edit. Reusing
    the example would make every edit to a teaching file a change to a shipped
    command.
  - Flags for epochs, batches, batch size, hidden width, features, classes,
    learning rate, optimizer (`adamw`/`sgd`), seed and MNIST directory,
    parsed by a hand-written loop rather than a new dependency. The CLI
    forwards every argument verbatim — including `--help` — so the flag
    surface has exactly one definition.
  - The learning-rate default is per-optimizer (3e-3 AdamW, 0.2 SGD), because
    a shared default would make `--optimizer sgd` look broken rather than
    merely different.
  - MNIST when the IDX files are present, the seeded synthetic problem
    otherwise, stated plainly in both the header and the result block. The
    fallback itself is `data/dataset.hpp`'s, reused rather than reimplemented.
  - The result is held back and printed as one block at the end.
    SushiRuntime and SushiBLAS log at INFO to stdout in a Release build and
    their level is a compile-time constant, so a report printed line by line
    comes out shredded through it.
  - Pinned by two tests: the `Integration_Demo.MlpRunsAndReportsOneCompile`
    CTest entry (a two-step run, asserted against the compile-count line) and
    `cli/tests/test_demo.py` (the same run through `sa demo mlp`, asserting
    exit 0). A demo that rots silently is worse than no demo.
- **`sa build -D/--define VAR=VALUE`**, repeatable, mirroring `sb build`'s
  flag exactly. Cache variables are appended after every default the CLI
  sets, so an explicit `-D` is never outranked; a malformed entry is rejected
  before configure runs. This is the only sanctioned way to reach SushiBLAS's
  `SB_SYCL_TARGETS` from here — SushiAI `add_subdirectory`s SushiBLAS, so AOT
  device targeting is a cache variable of *this* build tree, and building
  SushiAI for an NVIDIA GPU was previously impossible through `sa`.
- **`SA_BUILD_TESTS`** (default `ON`), guarding `enable_testing()` and
  `add_subdirectory(tests)` together — the same shape as SushiRuntime's
  `SR_BUILD_TESTS`, but defaulted `ON` because `sa test` expects a test binary
  to exist. A superproject that `add_subdirectory`s SushiAI can now turn the
  suite off instead of compiling, discovering and registering 100 tests it
  never asked for.
- **A `regression` test tier** (`tests/functional/regression/`, CTest label
  `regression`, `sa test --suite regression`), completing the
  `unit`/`integration`/`regression` triple the rest of the stack uses. Its
  first inhabitant, `test_compile_once.cpp`, pins the two claims in this
  library whose failure is completely silent: `compile_count() == 1` across
  repeated `fit()` calls (the shape `sa demo mlp` uses, which a single-`fit()`
  test cannot see), and the fact that every replay actually dispatches rather
  than reporting complete over an unarmed plan.
- **AI-1: a graph-native autograd core, and an MLP that actually trains.**
  The whole pipeline exists end to end — trace a model into IR, differentiate
  the IR, lower it to SushiBLAS once, compile it once, replay it per batch.
  `sa run sushiai_example` trains a two-layer MLP on a seeded synthetic
  classification problem (or MNIST, when the IDX files are present) from
  `ln(C)` to convergence.
  - `graph/builder.hpp` — the single home for every operation's shape
    inference, so a layer, a backward rule and a future fusion pass cannot
    disagree about what `MATMUL` produces.
  - `graph/lowering.hpp` — `TensorPool` and `Lowering`: the only file in the
    library that calls `SushiBLAS::Engine`. One `Storage` per value, because
    almost every SushiBLAS kernel ignores `Tensor::storage_offset` and an
    arena view would be silently mis-read rather than rejected. Constant
    operands (ones vectors, zero buffers) are filled once after the step is
    compiled, not re-filled on every replay.
  - `graph/compiled_step.hpp` — RAII over lower/compile/arm/replay, so
    `rearm()` before `capture_base_state()` (which leaves the completion
    sentinel armed at zero and makes a step silently do nothing) is
    unrepresentable. Exposes `compile_count()`, the AI-1 gate.
  - `autograd/backward.hpp` — reverse-mode differentiation as an IR-to-IR
    transform, with a `constexpr` rule table rather than a self-registering
    map (a static library drops a TU that only runs a registration
    constructor). Tiered fan-out accumulation: symbolic zero, alias, or
    ordered `ACCUMULATE` nodes — never atomics, so replay stays byte-equal
    under SushiRuntime's `sushiruntime_deterministic_fp` contract. The
    log-softmax + NLL pair is differentiated as one fused
    `(softmax - target) / N`.
  - `autograd/gradcheck.hpp` — central-difference verification in FLOAT64,
    full-Jacobian and randomised-directional modes, with kink-avoidance for
    the ReLU family.
  - `nn/` — `Parameter`, the `Module` concept, one generic
    `for_each_parameter` walk producing PyTorch-shaped names (`"0.weight"`),
    `Linear`/`ReLU`/`GELU`/`Tanh`/`Sigmoid`/`Sequential` (with CTAD),
    `CrossEntropyLoss`, and Kaiming/Xavier initialisation with a
    zero-initialised final classifier so the initial loss is exactly `ln(C)`.
    No base class and no `register_module`: forgetting a registration call
    produces a layer the optimizer silently skips, which shows up as
    slightly-worse convergence rather than an error.
  - `optim/` — SGD (PyTorch's momentum formulation, including `b_1 = g_1`
    with no dampening on the first step) and AdamW (decoupled decay applied
    to `theta_{t-1}` first, `eps` added *after* dividing by `sqrt(bc2)`).
  - `data/` — a `Dataset` seam, a hermetic seeded `SyntheticClassification`
    generated on device, and an MNIST IDX loader that activates only when the
    files are present and says so when they are not.
  - `train/trainer.hpp` — the loop, plus a run report carrying the compile
    count, step count and (when the runtime has profiling on) per-node
    timings.
  - `core/ops.hpp` gained `EXP`, `ZERO` and the four activation-gradient
    identities, all covered by the existing collision `static_assert`.
- A test suite that holds the claims to account: 98 SushiAI cases at AI-1
  (100 now, with the `regression` tier above) across
  device-free `Unit_*` (shape semantics, op-id uniqueness, single-assignment
  enforcement, `propagate_requires_grad`, backward-graph topology, the
  parameter walk, the gradient checker's own arithmetic) and `Integration_*`
  (hand-computed forward values, FLOAT64 finite-difference checks for every
  backward rule, a real training run, `compile_count() == 1` across 40 steps,
  an allocation-free steady state, and an initial loss equal to `ln(C)` to
  1e-5). `tests/common/` now holds a process-global `RuntimeContext` in a
  GTest environment and a per-test `Engine` fixture, mirroring SushiBLAS.

### Known limitations
- The optimizer update and the per-step batch feed are not part of the
  compiled plan: AdamW's bias corrections are per-step scalars and SushiBLAS
  takes its alphas by value, and a compiled call captures raw pointers so a
  batch must be copied into fixed buffers rather than bound to them. Both
  fold into the plan once SushiRuntime's late-binding block reaches the BLAS
  layer.
- `LOG_SOFTMAX` is differentiable only as part of the fused
  `LOG_SOFTMAX` + `NLL_LOSS` pair; the standalone rule needs a row-wise
  reduction SushiBLAS does not expose, and throws saying so.
- `Lifetime` is recorded on every value but no memory planner reads it yet.

- **AI-0: a clean scope-reset skeleton**, replacing the retired 2025-06
  CUDA/cuBLAS eager-autograd prototype (`core/tensor.h`, `enable_language(CUDA)`,
  hardcoded toolkit paths, zero stack integration — deleted, not migrated
  from; it predates the SushiStack integration entirely).
- A real two-sibling dependency chain to SushiRuntime and SushiBLAS, resolved
  through the same [D3 layered CMake pattern](ARCHITECTURE.md#2-the-sibling-dependency-chain-d3)
  every other SushiStack module uses (TARGET-guard → `find_package` → sibling
  `add_subdirectory`), run in an order (SushiRuntime, then SushiBLAS) that
  lets SushiBLAS's own internal runtime resolution TARGET-guard cleanly onto
  the one already brought in.
- `SushiAI::sushiai` — a `STATIC` library with one real compiled symbol,
  `SushiAI::version_string()`, linking `SushiRuntime::SushiRuntime` and
  `SushiBLAS::sushiblas` `PUBLIC`.
- `find_package(SushiAI)`-ready packaging from day one, mirroring SushiBLAS's
  SB-6 packaging shape exactly: namespaced alias, `install(EXPORT)`/
  `export(EXPORT)`, a generated `SushiAIConfig.cmake` with guarded
  `find_dependency` calls for both siblings, `SameMinorVersion` package
  version compatibility, and a version-drift check between
  `include/SushiAI/version.hpp` and the CMake project version. Verified with
  a real `tests/package/` out-of-tree consumer.
- A test suite structure (`tests/functional/{unit,integration}/`) using the
  same `Unit_`/`Integration_`-prefixed GTest suite-name convention as
  SushiRuntime/SushiBLAS, with `test_version.cpp` (macro sanity) and
  `test_engine_smoke.cpp` (a real `Engine` + `axpy` op through both siblings,
  end to end) — the real proof the dependency chain compiles, links, and
  runs, not a placeholder.
- The `sa` / `sushiai` CLI (Typer + rich), modeled on SushiEngine's `se`
  (shared bundled toolchain, no toolchain selection of its own), with the
  full `build`/`test`/`run`/`clean`/`doxygen`/`config`/`env` surface plus a
  `demo mlp` command that, at AI-0, correctly reported "not implemented yet —
  see AI-7" rather than faking a showcase that didn't exist. It runs the real
  showcase now — see the AI-7 entry above.
- Full documentation set, mirroring SushiRuntime's documentation structure
  and scaled honestly to AI-0's actual scope: `README.md`, `ARCHITECTURE.md`,
  `CONTRIBUTING.md`, `CLI_GUIDE.md`, `INTEGRATION.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, this changelog, and `CLAUDE.md`.

### Fixed
- The bundled `clang++`-personality toolchain binary hardcodes C++ mode
  regardless of file extension, which broke CMake's C-language probe (the
  project declares `LANGUAGES CXX C`, inherited transitively through
  SushiBLAS). `sa build`'s `_configure_args()` now resolves the toolchain's
  sibling `clang`/`clang.exe` binary for the `CMAKE_C_COMPILER` slot instead
  of reusing the C++ compiler path directly — a bug found by actually
  running `sa build` fresh, and one that exists verbatim (but unexercised)
  in `sb`'s and `se`'s own `_configure_args()` too.
- `cmake/Runtime.cmake` / `cmake/BLAS.cmake`'s `SUSHISTACK_HOME`-based sibling
  default now checks the resolved path actually exists before using it,
  rather than trusting the environment variable unconditionally — the same
  fix applied to SushiRuntime's and SushiBLAS's own `Runtime.cmake` files,
  needed on any machine where `SUSHISTACK_HOME` is set globally but modules
  are checked out flat rather than inside it.

---

<!--
## [x.y.z] — YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing behavior

### Fixed
- Bug fixes

### Removed
- Removed features

[x.y.z]: https://github.com/SushiSystems/SushiAI/compare/vX.Y.Z-1...vX.Y.Z
-->
