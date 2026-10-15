# SushiBLAS & SushiAI: SushiStack Construction and AAA-Quality Plan

**Status:** Shipped — AI-0 to AI-4 landed; superseded on 2026-10-07 by [`PROGRAMME.md`](PROGRAMME.md), which carries what is still open

**Mandate.** The macro architecture of the stack, stated once:

```
                sushistack (hub)  — workspace, provisioning, module registry
                sushicli         — shared CLI presentation/config library (sr/se/sb/sa/hub)
   ─────────────────────────────────────────────────────────────────────────
                         S U S H I R U N T I M E          ← the backbone
                          ▲         ▲         ▲
                          │         │         │
                    SushiBLAS   SushiEngine   (any future head)
                      ▲   ▲         │
                      │   └─────────┘  (engine's PINN/RL forward evaluators, later)
                   SushiAI
```

SushiRuntime is the backbone; SushiEngine, SushiBLAS, and SushiAI are heads. SushiBLAS is both a
head of the runtime and the power source for the other two: SushiAI draws **all** of its compute
from SushiBLAS (`SushiAI → SushiBLAS → SushiRuntime`, never touching SYCL directly), and
SushiEngine will draw its future in-engine inference evaluators (PINN physics, RL agents) from
SushiBLAS: a dependency already recorded in the engine's own
`docs/slop/cross_platform_engineering_plan.md` (2026-08-01 entries). SushiBLAS becomes a real
SushiStack member (not a frozen vendored snapshot), and SushiAI, a solo CUDA/cuBLAS
hobby prototype (`enable_language(CUDA)`, hard-coded toolkit path, links `cudart`), unrelated to
the stack: is rebuilt from zero as an AAA-quality ML framework. Three numbered tracks: **ST-\***
(stack & CLI wiring), **SB-\*** (SushiBLAS repair), **AI-\*** (SushiAI construction): kept apart
from the runtime's `BB-*`/`WP-*` series.

---

## 1. Ground truth (verified 2026-08-01, corrections to v1 marked ✱)

**SushiBLAS**

1. **The runtime dependency is a 5-month-stale fork, not merely a vendored copy.** ✱
   `include/SushiRuntime/*` (18 headers) and `lib/{sushiruntime.dll,.lib,hwloc-15.dll}` are
   git-tracked snapshots last touched 2026-03-02 / 2026-02-22. Upstream has since grown whole
   subtrees the snapshot has never seen (`api/`, `distributed/`, determinism, co-tenancy,
   packaging; `graph/task_graph.hpp` is 128 lines vendored vs 249 upstream). SB-0 is a
   **migration**, not a file swap. `src/CMakeLists.txt:135` links via
   `target_link_directories(... "${CMAKE_SOURCE_DIR}/lib")`.
2. **Only 4 of 94 op `.cpp` files are in the build** (`engine.cpp`, `io.cpp`,
   `blas/level3/gemm.cpp`, `blas/level3/syrk.cpp`); ~90 are written but commented out of
   `src/CMakeLists.txt:12-124`.
3. **Only 2 of 75 test files are in the build.** ✱ (v1 said "per-op gtests exist": true as files,
   but `tests/CMakeLists.txt:5-106` comments out 73 of them; the real tested surface is
   gemm + syrk.)
4. **The two compiled BLAS ops use the bad dispatch shape** (5× hand-duplicated
   `add_task` per dtype in `gemm.cpp`/`syrk.cpp`; `gemv.cpp` 4× with a HALF TODO), while the
   codebase's own `level1_internal.hpp::execute_level1` and
   `ew_internal.hpp::execute_binary/execute_unary_inplace` already demonstrate the correct
   single-`add_task`, dtype-switch-inside-lambda shape. `asum.cpp`/`add.cpp` prove the fix is
   cheap. Note: `ew_internal` silently no-ops on complex dtypes (`default: break;`): silent
   no-op is not acceptable; unsupported dtypes must fail loudly.
5. **The compute backend is oneMKL, and only oneMKL** (`kernels/sycl/*` wraps `oneapi::mkl`;
   `kernels/reference/*` is naive loops; RNG is documented as MKL VSL; MKL flags gated on
   `CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` via `/Qmkl`), which conflicts with the stack's
   default open-source intel-llvm nightly toolchain and fits neither the owner's hardware
   (Ryzen 7600X + GTX 1060).
6. **`Engine::linalg()` and `Engine::signal()` are declared-but-undefined** ✱: 8 + 4 declared ops
   with zero definitions in `src/`; any call is an unresolved-external link error. Resolved by D1.
7. **No `install()`/export rules exist** ✱: `find_package(SushiBLAS)` is impossible today; there
   is no clean way for SushiAI to consume SushiBLAS at all. `scripts/{win,linux}/run.*` reference
   a `sushiblas_app` target that does not exist in any CMakeLists.
8. **DataType is `HALF/FLOAT32/FLOAT64/COMPLEX32/COMPLEX64`: no BF16, no INT8** ✱
   (`core/common.hpp:49-56`); coverage is ragged (gemv skips HALF, `io.cpp` cannot print HALF).
   v1's claim that "the dtype plumbing AI-4 needs already exists" was optimistic.
9. Tensor/Storage API is real and usable: `Tensor` = non-owning view (shape/strides, rank ≤ 6,
   `transpose/reshape/slice` as views) over refcounted `Storage` (USM via
   `SushiRuntime::USMAllocator`, 64-byte aligned). SushiRuntime types (`sushi_ptr`,
   `AllocStrategy`, `RuntimeContext`, `sycl::event`) are part of the public surface: consumers
   need the runtime headers on their include path by design.

**SushiRuntime (the backbone: what SB/AI build on)**

10. **BB-4 packaging landed** (`cb894ff`): `SushiRuntime::SushiRuntime` alias,
    `install(EXPORT)` + `SushiRuntimeConfig.cmake` + `SameMinorVersion` version file +
    `export(EXPORT)` into the build tree, and a `tests/package/` out-of-tree consumer that
    asserts flag hygiene. **BB-4.6 (running `tests/package/` against a real install prefix) has
    never been executed**: SB-0 will be its first real-world exercise.
11. **The determinism contract is PUBLIC and travels to consumers**: kernels are header templates
    instantiated in the consumer's TUs, so the consumer's FP flags are part of the byte-equal
    replay guarantee (`sushiruntime_deterministic_fp` is the only exported policy target). The
    documented mitigation is the **two-lane build**: a SYCL lane that includes runtime headers, an
    ordinary lane that includes none, meeting at a POD seam.
12. **`OpID` string-hash registration (`"blas.gemm"_op`) is the intended extension seam** for
    named operations, and is what the distributed layer ships over the wire. SushiBLAS should
    register through it, never extend a runtime enum.
13. The `sr` CLI is the stack's reference CLI (see §7/ST-1 for the anatomy worth copying).
    Known open items upstream: 3 unresolved ASan failures (exception-propagation crashes) and the
    untagged `[Unreleased]` BB-2/3/4 changelog block. Not blockers for SB, but SB-0 must not be
    debugged in ignorance of them.

**SushiEngine (the head template)**

14. Engine consumes the runtime via `cmake/Runtime.cmake`: `$SUSHISTACK_HOME/sushiruntime` else
    `../sushiruntime` sibling, one `SUSHIRUNTime_DIR` cache var, `add_subdirectory`. It has
    **not** adopted BB-4 `find_package` yet. Engine is header-only (`INTERFACE` target), forwards
    the runtime target, post-build-copies the runtime DLL on Windows (`cmake/SyclTarget.cmake`).
15. Engine's `core/types.hpp` + `core/blas_placeholder.hpp` (772 lines) carry a documented
    "swap to SushiBLAS" contract (`SushiBLAS::Float`, `Vector3`, parametric `Vector3T<T>`,
    `Mat4`, 23 free functions, fold operators, `fold_range` as the reduction reference).
    **Resolved by D2: the swap is cancelled**: see SB-5.
16. Engine's CMake-namespace rule (`docs/CONTRIBUTING.md:135-148`): every target/function/option
    prefixed with the full program name (`sushiblas_`, `sushiai_`, ...) because `add_subdirectory`
    puts all trees in one flat namespace where duplicate functions silently override.
17. The compile-once/replay precedent is real and test-pinned: ECS `Schedule` and PGS/XPBD solvers
    assert `compile_count == 1` after warm-up across many integration tests; counts are late-bound
    (`API::sized(closure)`), structure recompiles only on `structure_version` change.

**SushiStack / sushicli (the wiring layer)**

18. **The registry side is already done**: `hub`'s `MODULES` map contains `sushiblas` and `sushiai`
    with reserved aliases `sb`/`sa` (`sushistack/cli/sushistack/services/modules.py:42-67`). What
    is missing is entirely module-side: neither repo has a `cli/` package, a
    `cli/sushistack.deps.toml`, nor a `hub link` entry in this machine's `modules.local.toml`.
19. The module contract (from `modules.py`, `cli_install.py`, `dependency_source.py`):
    `<repo>/cli/pyproject.toml` (`[project.scripts]` short+long names), `<repo>/cli/<pkg>/`
    (Typer app importing `sushicli`), `<repo>/cli/config.toml` + gitignored `config.local.toml`,
    `<repo>/cli/sushistack.deps.toml` with a `[module] depends_on = [...]` table. `hub install-cli`
    pipx-installs the package editable and injects the unpublished `sushicli` checkout.
20. `sushicli` is a shared presentation/config **library** (Theme/IconSet/Renderer/Console,
    `ToolConfig` base, workspace walk-up helpers), not a framework or launcher; `sr`/`se`/`hub` all
    import it. `sb`/`sa` plug into the same seams.
21. Hygiene items: sushistack sits on unmerged `feature/toolchain-refresh` while `hub update`
    ff-only-pulls the workspace; sushistack's and sushicli's shared CLAUDE.md both point at a
    `docs/CONTRIBUTING.md` that does not exist in either repo.

**SushiAI (the retired prototype)**

22. `main@0bc4b04`; last real code commit 2025-06-23 ("MLP Demo"), README-only since. Eager,
    host-side `std::vector<float>` autograd (`core/tensor.h`: gradient closures +
    `shared_ptr` parents + host topo-sort), a CUDA/cuBLAS build with a hard-coded toolkit path.
    Zero references to the stack. Retired by AI-0; not a migration source.

**Hardware reality (shapes AI-4's expectations)**

23. The dev machine is Ryzen 7600X + **GTX 1060 (Pascal, sm_61)**: fp16 arithmetic runs at 1/64
    rate on Pascal, and bf16/tensor cores do not exist there. Mixed precision is therefore
    architected in from AI-1 (dtype plumbing, master-weight discipline) but **no fp16 throughput
    win is expected on the owner's current GPU**: the payoff arrives with tensor-core-capable
    hardware. This is stated here so nobody benchmarks fp16 on Pascal and concludes AI-4 failed.

---

## 2. Decisions (resolved with the owner, 2026-08-01)

- **D1: FFT/linalg: quarantine.** `signal()` and `linalg()` leave the `Engine` facade; their
  headers are marked experimental and excluded from the umbrella header. The FFT backend decision
  (own SYCL FFT vs optional oneMKL DFT vs other) is deferred to a dedicated decision after the SB
  track lands. The declared-but-undefined link trap is closed immediately. SushiAI v1 needs no FFT.
- **D2: Fixed-size geometry stays engine-side.** SushiBLAS remains a pure n-dimensional tensor
  library. The engine keeps (and now permanently owns) its `Vector3T<T>`/`QuaternionT<T>`/`Mat4`
  math; the documented "swap `core/types.hpp` to SushiBLAS" contract is **cancelled** and the
  engine's docs/comments must be updated to say so (SB-5). SushiBLAS never grows a small-geometry
  surface.
- **D3: Runtime resolution is layered.** SushiBLAS (and SushiAI, for both of its upstreams):
  if the target already exists (superproject case) use it → else try `find_package` (QUIET) → else
  `add_subdirectory` of the `$SUSHISTACK_HOME`/sibling checkout. The link line is always the
  namespaced alias, so all three modes are spelled identically. BB-4 gains its first real
  consumer; today's `hub`/`sr` workflow keeps working unchanged.
- **D4: SushiAI v1 showcase is an MLP training demo** (MNIST-class task): end-to-end training
  with graph-native autograd, validated by `compile_count == 1`, zero steady-state allocations,
  finite-difference gradient checks, and a convergence curve. PINN alignment with the engine's
  forward evaluator is v1.x; a Python frontend is v2.
- **D5: portBLAS port goes ahead; attribution via `NOTICE.md`.** Provenance for every ported
  algorithm is recorded in a repo-root `NOTICE.md` in sushiblas (file/algorithm → portBLAS origin,
  per Apache-2.0), not in per-function comments. The license/health due-diligence check in SB-2
  remains a pre-code step, but the mechanism question is closed.
- **D6: sushistack merges to `main` directly.** `feature/toolchain-refresh` is merged straight
  into `main` (executed as part of landing this plan), which also unblocks the ST-1 `sb` CLI
  work: `hub update`'s ff-only pull of the workspace is coherent again.
- **D7: sushiblas and sushiai relicense from MIT to Apache-2.0** (2026-08-01). Root `LICENSE` and
  every per-file header signature updated in both repos (sushiblas: 150 source files, short
  ASF-standard header matching sushiruntime/sushiengine's existing convention; sushiai: root
  `LICENSE` + README only, since its current tree is the retired prototype AI-0 deletes wholesale).
  Makes D5's portBLAS attribution story simpler (same license family, no cross-license friction)
  and brings sushiblas/sushiai in line with sushiruntime/sushiengine, which were already Apache-2.0.
  Superseded on 2026-10-05: both repositories moved to PolyForm Noncommercial 1.0.0; see `NOTICE.md`.

---

## 3. Principles

**P1: One seam, not parallel paths.** Every place SushiBLAS hand-duplicates dispatch logic a
sibling module already solved generically is technical debt being actively grown. Fix the pattern
before scaling the op count (SB-1 gates SB-3). Corollary from ground truth #4: unsupported dtypes
fail loudly, never silently no-op.

**P2: A vendored snapshot is not a dependency, it's a fork nobody decided to make.** SB-0
replaces the frozen copy with live resolution (D3).

**P3: When you do fork, fork on purpose and own it.** SB-2 ports portBLAS algorithms into
SushiBLAS's own kernel files and conventions. Apache-2.0 attribution follows the code, not the
folder; and there is no free rebase from upstream afterwards: GEMM tuning becomes SushiBLAS's own
responsibility from that point on.

**P4: Compile once, replay.** The engine's ECS Schedule and PGS solver already prove the pattern
(`compile_count == 1`, test-pinned). SushiAI's training loop is the best-case workload for it and
AI-1 uses it as the foundation, not a bolt-on.

**P5: Engineer it once carefully, run it a thousand times cheaply.** The runtime's own
performance principle; AI-2 (static arena) and AI-3 (fusion) are its direct application.

**P6: The API-reality rule (from the runtime's `add_reduce` incident).** Every public upstream
API that SushiBLAS or SushiAI leans on gets at least one dependent-side TU that *instantiates* it,
and every feature ships with a consumer-shaped example compiled in CI. `-fsyntax-only` blessing is
not evidence.

**P7: The CLI is the only door, and `hub` owns provisioning.** Builds/tests/runs go through the
module's own CLI (`sb`, `sa`) exactly as `sr`/`se` mandate in their CLAUDE.md. Module CLIs never
install dependencies or link modules: that is `hub`'s job (`hub install`, `hub install-cli`,
`hub link`). Mirrors the deliberate absence of `sr install`/`sr link`.

**P8: Honesty about scope.** Explicitly unsolved here: the FFT backend (deferred by D1) and
value-dependent control flow in autograd (AI-1 non-goal). Named, not hand-waved.

---

## 4. The program at a glance

| Pkg | Title | Depends on | What it unblocks |
|---|---|---|---|
| ST-0 | Workspace registration & stack hygiene | none | `hub link` for both modules; sushistack branch/docs cleanup |
| ST-1 | `sb`: SushiBLAS CLI package + deps fragment | ST-0 | SushiBLAS becomes a first-class `hub`-managed module |
| ST-2 | `sa`: SushiAI CLI package + deps fragment | ST-0 | Same for SushiAI (repo skeleton lands with AI-0) |
| ST-3 | CLI install, doctor green, CI lanes | ST-1, ST-2 | `hub install-cli sushiblas sushiai`; pyflakes+pytest lanes |
| SB-0 | Live runtime dependency (D3 layered resolution) | ST-0 | Kills the stale fork; first real BB-4 consumer |
| SB-1 | Dispatch pattern cleanup (`execute_blas_task`, `dispatch_with_mode`) | SB-0 | gemm/syrk/gemv stop duplicating; sets the SB-3 shape |
| SB-2 | Backend strategy: portBLAS-derived kernels + native RNG; FFT quarantined (D1) | SB-0 | Removes oneMKL as the only path; builds on default toolchain |
| SB-3 | Enable & fix the ~90 disabled ops | SB-1, SB-2 | Full op surface builds against the current runtime |
| SB-4 | Test suite: enable all 75 test files, green | SB-3 | Correctness gate for everything AI builds on |
| SB-5 | Engine seam resolution per D2 (docs + comment updates, engine-side) | none | Cancels the stale swap promise; engine owns its math |
| SB-6 | SushiBLAS packaging (alias, install/export, `tests/package/`) | SB-0 | `find_package(SushiBLAS)`; SushiAI gets a clean upstream |
| AI-0 | Scope reset: retire the prototype, new repo skeleton + `sa` CLI | ST-2 | Clean construction against `SushiBLAS::Tensor` |
| AI-1 | Graph-native autograd core (+ finite-difference gradient checks) | AI-0, SB-4 | Forward+backward as one TaskGraph, compiled once, replayed |
| AI-2 | Static memory planning (liveness, single arena) | AI-1 | No allocator traffic in the training hot loop |
| AI-3 | Operator fusion over the generic `op_func` dispatch | AI-1 | Fewer launches, fewer intermediates |
| AI-4 | Mixed precision plumbing (fp16 now, BF16 enum staged; fp32 masters) | AI-1 | Dtype half and loss scaling both landed 2026-08-03; the skip is per parameter, not per step. See §20 and ground truth #23 |
| AI-5 | Device-resident data discipline | AI-1 | No host round-trips stalling the async chain |
| AI-6 | Distributed data/model parallelism (deferred) | AI-1..5 | Hangs off the runtime's distributed layer when needed |
| AI-7 | v1 showcase: MLP training demo + metric gates (D4) | AI-1..3, AI-5 | The proof the framework is real |

**Sequencing.** ST-0 first (an afternoon, unblocks everything). Then SB-0 with ST-1 alongside
(the `sb` CLI and the CMake rework land together; the CLAUDE.md "CLI is not optional" clause lands
with them). SB-1 next; SB-2 design can overlap SB-1 but its landing gates SB-3 (no point enabling
90 ops against a backend about to be replaced). SB-3 → SB-4 in batches (an op enables together
with its test, never separately). SB-5 is independent (engine-side, doc-only) and can happen any
time. SB-6 can land right after SB-0 proves resolution works, and must land before AI-1 starts
consuming SushiBLAS as a package. **SB-4 is the gate into the AI track.** Within AI: AI-0 (with
ST-2) → AI-1 validated for correctness before AI-2/3/4/5 layer on → AI-7 assembles the showcase.
AI-6 is explicitly out of scope for v1.

---

## 5. ST-0: Workspace registration & stack hygiene

- `hub link sushiblas D:\Projects\sushiblas` and `hub link sushiai D:\Projects\sushiai` (registry
  entries and `sb`/`sa` aliases already exist; only `modules.local.toml` is missing them). `hub`
  will warn about the missing `cli/sushistack.deps.toml` until ST-1/ST-2 land: expected.
- sushistack's `feature/toolchain-refresh` is merged straight into `main` per D6 (a long-lived
  feature branch as the de-facto main was a trap, since `hub update` ff-only-pulls the workspace).
- Fix the dangling `docs/CONTRIBUTING.md` reference shared by sushistack's and sushicli's
  CLAUDE.md: either write the file (thin, pointing at the runtime's CONTRIBUTING as the style
  authority) or repoint the reference.

## 6. ST-1: `sb`: the SushiBLAS CLI package

Copy the `se` anatomy (the head-CLI template), not `sr` wholesale: like the engine, SushiBLAS
selects no SYCL toolchain of its own (it builds with the runtime's choice), so there is no
`toolchain` command; and per P7 there is no `install`/`link`.

- `sushiblas/cli/pyproject.toml`: dist `sushiblas-cli`, scripts `sb` + `sushiblas`, Typer+Rich,
  `sushicli` deliberately undeclared (injected by `hub install-cli`), same explanatory comment as
  the other three repos.
- `sushiblas/cli/sushiblas_cli/` (package name must not collide with a future Python binding of
  the library itself: note `sr`/`se` use the bare module name; `sb` should follow suit as
  `sushiblas/cli/sushiblas/` only if no Python tensor binding will ever want that name; decide at
  implementation, default to matching `se`): `cli.py` (thin Typer, commands parse args and
  delegate), `config.py` (`Config(ToolConfig)` + `SB_*` `_ENV_OVERRIDES` map + `sushiruntime_dir`
  resolution copied from `se`), `console.py` (lazy `build_console`), `env.py` (vcvars snapshot,
  runtime toolchain on PATH), `services/{project,diag,discovery}.py`, `cli/tests/` with the
  `sushicli` stub conftest.
- Command surface: `sb build [-t type] [--clean] [--force] [--no-test]`, `sb test [-f regex]
  [-r N]`, `sb run [example]`, `sb clean`, `sb doxygen`, `sb config`, `sb env`. Benchmarks arrive
  later as `sb test --suite benchmark` (mirroring `sr`), not a separate command.
- `sushiblas/cli/sushistack.deps.toml`: `[module] depends_on = ["sushiruntime"]`; no oneMKL
  entry (SB-2 removes the need); gtest via the same mechanism the engine uses.
- `sushiblas/cli/config.toml`: `[tool]`/`[tool.windows]`/`[tool.linux]` + the shared `[cli]`
  theme block; gitignore `cli/config.local.toml`.
- Delete `scripts/{win,linux}/` (stale, references a nonexistent target) once `sb` covers them.
- `sushiblas/docs/CLAUDE.md` gains the runtime's mandatory-CLI clause verbatim with `sb` as the
  name; `docs/CLI_GUIDE.md` written per stack convention.

## 7. ST-2: `sa`: the SushiAI CLI package

Identical anatomy with `sa` + `sushiai` entry points, `SA_*` env prefix, and
`[module] depends_on = ["sushiruntime", "sushiblas"]`. Lands together with AI-0's repo skeleton:
the new tree is born CLI-first, there is never a commit of the new SushiAI that must be built by
raw cmake. Extra command beyond the standard set: `sa demo mlp` (runs the AI-7 showcase: its own
`sushiai_demo_mlp` binary, not a wrapper over `sa run` against the example; see §23).

**Status (2026-08-03): done and pytest-verified (12/12), including the
`_resolved_c_compiler()` fix for the bundled toolchain's `clang++`-personality
C-probe bug and a real `sa build --type release` since.** `sa --help` /
`sa config` / `sa demo mlp` all verified against the real dev environment.
`sa build` also gained `-D/--define VAR=VALUE` (repeatable, appended after
every CLI-set default), mirroring `sb build`, without it SushiBLAS's
`SB_SYCL_TARGETS` was unreachable through the sanctioned path, since SushiAI
`add_subdirectory`s SushiBLAS and AOT targeting is a cache variable of this
build tree.

## 8. ST-3: Install, doctor, CI

- `hub install-cli sushiblas sushiai`; `hub doctor` shows both modules ready with their deps.
- Both repos get the `sr`-style CLI CI lane: pyflakes import-safety + `pytest cli/tests -q`
  standalone (the conftest `sushicli` stub makes this hermetic).
- C++ CI mirrors the engine's: checkout module + `SushiSystems/SushiRuntime@main` sibling (+
  sushiblas sibling for sushiai), configure through the CLI, build, run tests. P6's
  consumer-shaped examples compile in these lanes.

---

## 9. SB-0: Live runtime dependency (D3)

Delete `include/SushiRuntime/*` and `lib/{sushiruntime.dll,sushiruntime.lib,hwloc-15.dll}` and the
`.gitignore` whitelist comment that protects them. Add `cmake/Runtime.cmake` implementing the
layered resolution:

```cmake
if(NOT TARGET SushiRuntime::SushiRuntime)
    find_package(SushiRuntime QUIET)
endif()
if(NOT TARGET SushiRuntime::SushiRuntime)
    # $ENV{SUSHISTACK_HOME}/sushiruntime, else sibling; SUSHIRUNTIME_DIR cache override
    add_subdirectory("${SUSHIRUNTIME_DIR}" "${CMAKE_BINARY_DIR}/sushiruntime")
endif()
```

Link everywhere as `SushiRuntime::SushiRuntime` (never bare `sushiruntime`). Vcpkg/toolchain
shaping runs before `project()` exactly as the engine's root CMakeLists documents. All targets,
functions, and options gain the `sushiblas_` prefix (ground truth #16). Windows post-build DLL
copy for examples/tests copied from `cmake/SyclTarget.cmake`.

Expect real API-drift breakage in `engine.cpp`/`gemm.cpp`/`syrk.cpp`/`io.cpp` (the four live
files): that is the point: five months of runtime evolution (result/error types, task metadata,
allocator surface) get absorbed here, once, against the smallest possible file set. This is also
the first real execution of the runtime's BB-4.6 (`tests/package/` against an install prefix):
run it, and report findings upstream to the runtime rather than working around them (P2 boundary
rule: what the runtime cannot do yet is runtime work).

Verify: clean `hub status` → `sb build` resolves the runtime from the live checkout in all three
D3 modes (superproject, package, sibling); nothing runtime-shaped remains checked into sushiblas.

## 10. SB-1: Dispatch pattern cleanup

Add to `include/SushiBLAS/engine/blas/utils.hpp` (currently an empty TODO header):
- `execute_blas_task(Engine&, TaskMetadata, reads, writes, dtype, Func)`: the `execute_level1`
  shape (one `add_task`, one `switch(dtype)` dispatching a generically-templated closure),
  generalized for level2/level3.
- `dispatch_with_mode<T>(ExecutionMode, reference_fn, sycl_fn)`: collapses the
  `ForceReference`/normal pair `gemm.cpp` hand-writes five times.
- Loud failure for unsupported dtype/op combinations (P1 corollary): a thrown error naming op and
  dtype, not `default: break;`.

Migrate `gemm.cpp`, `syrk.cpp`, `gemv.cpp` (closing gemv's HALF TODO in the process). These three
become the reference shape every SB-3 op is written against. Add the missing HALF branch to
`io.cpp`'s printer while in there (ground truth #8).

## 11. SB-2: Backend strategy

**BLAS levels 1–3: port, don't wrap, don't vendor-as-subtree.** Take the GEMM/BLAS
tiling-and-blocking algorithms out of Codeplay's portBLAS and rewrite them into SushiBLAS's own
`kernels/sycl/blas/*` flat template-function convention (not portBLAS's expression-tree executor
framework). Attribution is decided (D5): a repo-root `NOTICE.md` records every ported
file/algorithm and its portBLAS origin: Apache-2.0 attribution follows the ported code, not a
folder boundary, and the NOTICE entry is written in the same commit as the port, never later. One
pre-code due-diligence step remains: confirm portBLAS's current license terms and repo health.
Accept explicitly: no future free rebase; GEMM tuning is SushiBLAS's own responsibility from the
moment of the port (P3).

**Random:** replace oneMKL VSL with a hand-written counter-based portable SYCL PRNG
(Philox-family). Small problem, no dependency.

**FFT/DFT + LAPACK: quarantined per D1.** `signal()`/`linalg()` accessors removed from `Engine`;
`transforms.hpp`/`linalg.hpp` moved out of the umbrella header and marked experimental. The FFT
backend decision is a named follow-up, not silently deferred.

**Net effect:** `Kernels::SYCL::*` stops meaning "call oneMKL" and starts meaning "the fast,
portable, owned path"; the `IntelLLVM`-only `/Qmkl` block and `find_package(IntelSYCL REQUIRED)`
hard-coupling are deleted; SushiBLAS builds on the stack's default toolchain with no `--oneapi`.

## 12. SB-3: Enable the disabled op surface

Uncomment the ~90 op files in batches (level1 → level2 → elementwise → reductions → nonlinear →
random → logic), fixing runtime-API drift as it surfaces. Every op is enabled **together with its
test file** (see SB-4): never separately. Any op still in gemm's old copy-paste shape is
rewritten onto SB-1's helpers as part of enabling it. Ops register their identity via the
runtime's `OpID` mechanism (`"blas.gemm"_op` convention, ground truth #12).

## 13. SB-4: Test suite green

All 75 test files enabled and passing (correction #3: they were never in the build). Suites split
per stack convention so `sb test` maps onto them cleanly. This is SushiBLAS's correctness gate:
nothing in the AI track starts against a SushiBLAS that hasn't cleared it. The two runtime-harness
tests (`test_framework_sim`, `test_dependency_stress`) are the P6 consumer-shaped examples for the
runtime API surface SushiBLAS leans on.

## 14. SB-5: Engine seam resolution (D2, engine-side, doc-only)

The swap contract is cancelled: the engine permanently owns its fixed-size math.
- Update `core/types.hpp`'s file comment (delete the "when SushiBLAS is available, replace…"
  block), `core/blas_placeholder.hpp`'s header comment, `docs/ARCHITECTURE.md` §6, and the
  `animation_system.md`/`physics_system.md` slop references.
- Rename `core/blas_placeholder.hpp` → `core/math.hpp` (it is not a placeholder for
  anything; the name is now actively misleading). One-file rename, still reached only through
  `core/types.hpp`.
- The engine's *future* SushiBLAS dependency (PINN/RL forward evaluators, per its own
  cross-platform plan) is unaffected: that is a tensor-compute dependency, not a value-type one.

## 15. SB-6: SushiBLAS packaging

Mirror BB-4 exactly: `add_library(SushiBLAS::SushiBLAS ALIAS sushiblas)`,
`install(TARGETS … EXPORT SushiBLASTargets)`, `install(DIRECTORY include/)`,
`SushiBLASConfig.cmake.in` (with `find_dependency(SushiRuntime)`),
`write_basic_package_version_file(… COMPATIBILITY SameMinorVersion)`, `export(EXPORT)` into the
build tree, a `version.hpp` with configure-time literal check, and a `tests/package/` out-of-tree
consumer asserting flag hygiene (no leaked warnings/sanitizers; only genuine usage requirements:
the runtime's FP contract: travel). Policy targets split `sushiblas_warnings` /
`sushiblas_sanitizers` PRIVATE, applied via one `sushiblas_apply_build_policy()`.

---

## 16. AI-0: Scope reset

The current `sushiai/` tree (CUDA/cuBLAS executable, `core/tensor.h` autograd) is retired:
explicitly not a migration source. The new skeleton lands CLI-first (ST-2): `modules/`, `applications/`,
`tests/`, `cli/`, `docs/` per stack convention; targets `sushiai` (library) +
`sushiai_tests` + examples, all `sushiai_`-prefixed; `SushiAI::SushiAI` alias and packaging shape
copied from SB-6 from day one. Upstreams resolved via the same D3 layered pattern for **both**
SushiRuntime and SushiBLAS. Old prototype code is deleted on the new branch (git history preserves
it; no `legacy/` folder). The old `docs/Roadmap.md` (CUDA/Unity era) is superseded by this plan;
its PINN/GNN ambitions survive as v1.x/v2 candidates.

**Status: CLOSED (2026-08-03).** Skeleton, tests, packaging shape, CLI,
`.gitignore`, and the full sushiruntime-style `docs/` set are all in place
and committed, and the verification list is done:
- Library + tests + example build, link, and run against both live siblings
  on real hardware; a real out-of-tree `find_package(SushiAI)` consumer
  (`tests/package/`) configured, built, and ran.
- **The Release build passes.** A
  full `sa build --type release` compiles
  `test_constant.cpp`/`test_leaky_relu.cpp`/`test_gelu.cpp` cleanly (they
  failed under `-O3 -DNDEBUG` on 2026-08-02 with the diagnostics unread), and
  `sa test --suite all` is green. It predated the SB-3/SB-6 commits and was
  a SushiBLAS-side issue, as suspected. Verified from both sides; see §25.

## 17. AI-1: Graph-native autograd core

Forward ops call `SushiBLAS::Engine` (already graph-dispatched). The departure from the old
design: each op's backward rule does not execute as an eager host closure: it emits its own
node(s) into the **same** `SushiRuntime::TaskGraph` the forward pass used. Most backward rules are
themselves BLAS/elementwise calls (`d(matmul)/dA = dC @ Bᵀ` is another GEMM), so the op surface is
reused rather than inventing a second computation model.

Host-side bookkeeping (tape/topo-sort) stays host-side: cheap, structural, not the hot path. The
compiled graph is cached and replayed keyed on `(architecture, batch-shape)`: the engine's
`structure_version` invalidation shape: value changes never rebuild, shape changes do.
`compile_count() == 1` across a training run is a **gtest assertion from the first commit**, per
the engine's precedent (ground truth #17).

**Correctness harness (gate for everything after):** finite-difference gradient checks per op
backward rule, and a tiny-model end-to-end gradient check against an analytic reference.

**Explicit non-goal for v1:** value-dependent control flow inside a model (breaks
replay-without-rebuild). Documented limitation; revisit if PINN/GNN work actually needs it.

## 18. AI-2: Static memory planning

Once a graph is compiled for a fixed shape, run a liveness/interval pass: any buffer not
read by a later op becomes reusable. One static USM arena (via SushiBLAS's refcounted `Storage`)
sized by the plan; tensor storages bind to arena offsets. Gate: zero allocator calls in the
steady-state training loop (assert via allocator instrumentation, not by eyeballing).

**Status (2026-08-03): landed, with §25's corrections applied rather than the sketch above.**
`graph/memory_plan.hpp` is a pure host-side function of the IR; `graph/arena.hpp` owns the memory.
Not "one static arena" and not an interval pass: **one arena per non-persistent lifetime class**
(the persistent four are never reused, so packing them would only merge dependency-tracker shards),
and liveness on the **reachability partial order**, which is the difference between reuse that adds
no ordering and reuse that silently serialises two parallel branches. The closure is O(V²/64)
memory, so above `PlanOptions::reuse_node_limit` nodes the planner declines to overlap anything and
says so instead of falling back to the linear approximation. Gate met:
`Medium_StaticMemory.ATrainingLoopAllocatesNothingAfterTheFirstStep` counts live allocations
through the allocator's reference count. `sa demo mlp` reports 112.6 KiB against 130.9 KiB.

Two things the sketch assumed and the implementation does not do: the lowering's own temporaries
(`TensorPool::scratch()`, the shared `ones()`) are not IR values and, but for the buffers a
scratch rule states (batch norm and its gradient since 2026-10-13), are still one `Storage` each
(~2 KiB of 130 KiB on the demo); and two values stay out of the arena because
`SushiBLAS::Dispatch::execute_reduction` still builds its `ResourceRegion`s from raw `data()`
pointers rather than `Tensor::region()`, which for a view at a non-zero offset produces **no
dependency edge at all**; see `Lowering::values_needing_own_allocation()`.

## 19. AI-3: Operator fusion

SushiBLAS's `execute_binary`/`execute_unary_inplace` pattern takes an arbitrary `op_func` functor:
fusion composes chains of elementwise/activation functors (matmul-epilogue → bias-add → relu) into
one functor before a single kernel is emitted, instead of one node + one launch + one intermediate
per op.

**Status (2026-08-03): landed, in the shape §25 forced rather than the one sketched above.** It is
**not** "compose functors into one before a kernel is emitted": that is a code generator, and
`sycl_ext_codeplay_kernel_fusion` is removed from intel-llvm, so a composition exists only if some
template was instantiated with it at *this library's* compile time. What landed is therefore a
**catalog** of five pre-instantiated forms (`core/ops.hpp`), a **static tree pass** matching
subtrees against it (`graph/fusion.hpp`, host lane, pure IR→IR, device-free unit tests), and a
**dynamic selector** choosing between matched entries at bind time from `Device::Profile` and the
engine's `ExecutionMode` (`graph/fusion_selector.hpp`). The functors themselves live in
`graph/lowering.cpp`, because a functor inlined into a SushiBLAS kernel is device code.

- `matmul → bias_add [→ relu | sigmoid | tanh]` through `Dispatch::execute_gemm`'s epilogue, and
  `exp → sub → scale` (the cross-entropy gradient's whole elementwise tail) through
  `Dispatch::execute_zip_inplace`. 35 recorded calls per step become 25 on the demo's model.
- **§25's activation warning is narrower than it reads.** Sigmoid and tanh *are* exactly
  recoverable from their outputs (`y(1-y)`, `1-y²`) and SushiBLAS's `sigmoid_backward` /
  `tanh_backward` already take the output, so they fuse. Only GELU is genuinely blocked. ReLU was
  moved to differentiate at its output too, which is bit-identical rather than approximate, and is
  what makes the pass need no activation-specific knowledge: the pre-activation then has one
  consumer, so an ordinary use count folds ReLU and refuses GELU. (GELU folds now as well, but not
  on a use count; see the `GELU_AUX_BIAS` entry below.)
- **The `GELU_AUX_BIAS` shape is built.** Without an access set, `execute_gemm` declares only A, B and C and hands the epilogue's own accesses back to
  the caller, so an aux buffer the functor wrote would get no edge and its reader would race the
  write. SushiBLAS now takes the epilogue's access set:
  `Dispatch::gemm_epilogue(f).reads(t).writes(t)`, so the aux write is declarable. Everything this
  entry then listed as outstanding landed as `FusedForm::GEMMBiasGELU`, with one item resolving
  more cheaply than expected: the "backward rule reading the spill" turned out to need no new rule
  at all. Fusion runs *after* differentiation, so `gelu_backward` is already in the graph reading
  the pre-activation by id; making that same id the fused node's second output means the existing
  rule reads the spill without knowing anything changed. The `static_assert` did learn the
  distinction, and also checks that the entry spills the *right* step of the chain: spilling the
  `K_MATMUL`'s result instead of the `K_BIAS_ADD`'s would be as silently wrong as spilling nothing.
  The one thing the entry cannot do is run under a narrowing precision policy, and that is
  deliberate: see §25's precision note and `graph/precision.hpp` rule 5.
- **A fused GEMM costs no extra ordering node.** The epilogue's *bias*
  read is a parameter the optimizer rewrites in the same plan, and it was declared with a
  `Lowering::declare_epilogue_operand()` that emitted a do-nothing `TaskGraph::add_native` node,
  counted apart in `RunReport::ordering_nodes_per_step`. `record_fused_gemm` declares it through
  the seam now (`gemm_epilogue(f).reads(bias)`, per catalog arm), so the node, the counter and the
  `Ops::EPILOGUE_OPERANDS` identity are gone: 27 plan nodes per fused step become 25, with `ops per
  step` unchanged at 25.
- **The largest remaining term is `K_LOG_SOFTMAX`'s nine calls**, and it is not a fusion-pass
  problem: it needs the row-wise reduction SushiBLAS does not expose. `K_NLL_LOSS`'s three are
  blocked the same way: the reduction seam takes no transform functor.

## 20. AI-4: Mixed precision (plumbing now, payoff on future hardware)

fp16/bf16 compute with fp32 gradient accumulation and master weights, built into the dtype
plumbing from AI-1 (not retrofitted). Concretely: SushiBLAS's `DataType` gains `BFLOAT16` (and
`INT8` while the enum is open) with loud-failure dispatch until kernels exist; HALF coverage gaps
(gemv, io) are closed in SB-1/SB-3. Per ground truth #23, Pascal makes fp16 a correctness feature,
not a speed feature, on the current dev box: benchmark gates for AI-4 are defined on
tensor-core-capable hardware only.

**Status (2026-08-03): both halves have landed.**

Landed:
- `graph/precision.hpp`: `PrecisionPolicy`, injected the way `FusionSelector` is, with a
  `UniformPrecisionPolicy` default (so nothing that existed changed) and a
  `MixedPrecisionPolicy` that narrows `K_MATMUL`, the fused GEMM forms, `K_BIAS_ADD` and the
  foldable activations, and nothing else. The narrow type is a constructor parameter, so
  bfloat16 is a call-site change on the day `DataType` grows one.
- `graph/cast_insertion.hpp`: the IR-to-IR pass, run between `differentiate()` and `fuse()`.
  `Ops::K_CAST` lowers to SushiBLAS's new `elementwise().convert()`.
- fp32 master weights, fp32 gradients, fp32 optimizer update: a parameter keeps its id, dtype
  and lifetime, and the fp16 a GEMM reads is a derived activation-lifetime value.
- `--precision fp32|mixed` on `sa demo mlp`, `TrainerOptions::precision` as a pre-lowering check,
  and `Graph::find_dtype_mismatches()` as the invariant anyone can re-assert after any later pass.

Landed in the second change (the numerics half):
- `optim/loss_scaler.hpp`: `LossScaleState` (the host state machine, engine-free and
  unit-tested on its own) and `LossScaler` (its device-resident scale, its count buffer, and
  the `count_nonfinite` screens it records into the plan).
- `Ops::K_DYNAMIC_SCALE`: `y = x * s[0]` with `s` a 1-element value. `K_SCALE`'s factor is an
  `Attributes` slot and therefore a compile-time constant; this one is a second *operand*, so
  the scale changes between replays without recompiling. Emitted by `differentiate()`'s
  K_NLL_LOSS seed rule, which is where the seed is built and therefore the one place the node
  lands in topological position without re-routing anything.
- `INVERSE_LOSS_SCALE` in both optimizers' `StepScalars`, applied before the decay and the
  moments; exactly `1.0f` when nothing is scaling, so an unscaled run is byte-identical.
- The skip is a **real branch** in the fused kernel, not a neutralised step size: `0 * inf` is
  NaN, so zeroing the coefficients would write NaN into AdamW's first moment on exactly the
  step the mechanism exists for.
- `--loss-scale S` on `sa demo mlp`.

Not landed:
- **A whole-step skip.** The gate is per parameter. The edge that orders an update after its
  own screen is the write-after-read hazard on that gradient; there is none between parameter
  A's update and parameter B's screen, and `execute_zip_inplace` cannot declare an extra read
  region. `--precision mixed --loss-scale 16777216` shows the difference: 8 overflowed steps
  out of 8, and the loss still moves. Closing it means one hand-written `add_task` declaring
  the count buffer as a read and a gradient as a write.
- **bfloat16.** `SushiBLAS::Core::DataType` still has no entry.
- **Any speed claim, and any claim that loss scaling helps here.** Measured on this box,
  `--precision mixed` is about 4.5x *slower*
  (16.45 → 73.16 ms/step on the demo's default), because half arithmetic is emulated here. What
  was measured and is real: the loss tracks fp32 to 9.5e−4 relative and converges to the same
  place, on the default model and on a `--hidden 512 --batch-size 64` one. **Neither underflow
  stall nor divergence was observed on this workload**, which is a fact worth knowing before
  commissioning the loss-scaling work: it is justified by theory and by larger models, not by
  anything this demo does.

## 21. AI-5: Device-resident data discipline

Training-path tensors default to `AllocStrategy::DEVICE`/`SHARED`, never `HOST`. Stated rule: no
synchronous host read of tensor data anywhere inside forward/backward: it stalls the async
graph-submission chain AI-1..3 depend on. Enforced by review convention plus a debug-mode
stall detector if the runtime exposes one.

## 22. AI-6: Distributed (deferred, not blocking v1)

The runtime's opt-in master-worker distributed layer exists; data/model parallelism hangs off it
when actually needed. Per the engine's cross-platform plan, SushiAI consumes it directly from the
runtime's native API (it is explicitly outside any portable seam). Noted so it isn't rediscovered;
not designed here.

## 23. AI-7: v1 showcase (D4): the MLP that proves it

MNIST-class supervised training, end to end, driven by `sa demo mlp`:
- data loaded once to device (AI-5), MLP forward/backward as one compiled graph (AI-1), static
  arena (AI-2), fused epilogues (AI-3);
- **hard gates:** `compile_count == 1` after warm-up across all epochs; zero steady-state
  allocations; finite-difference checks green; loss converges to a pinned reference curve within
  tolerance; a run report printing the same counters the engine's statistics pin
  (compile/compose counts, per-node timings).
- the demo doubles as P6's consumer-shaped example for SushiBLAS and as SushiAI's integration
  test.

**Status (2026-08-03): first cut landed, gates partly met.** `sa demo mlp` runs
`applications/examples/source/demo_mlp.cpp`: a flag-driven MLP classifier (epochs, batches, batch size, hidden
width, features, classes, lr, optimizer, seed, MNIST directory) that trains end to end and
reports the compile count, the per-epoch loss trajectory, the final accuracy and the wall clock.
It is a separate binary from `sushiai_example` on purpose: a demo is a supported entry point with
an output contract, an example is a teaching file people edit. Pinned by the
`Medium_Demo.MLPRunsAndReportsOneCompile` CTest entry and `cli/tests/test_demo.py`.

Still open against the hard-gate list above:
- **AI-2 and AI-3 are both in it now** (2026-08-03). AI-2: `graph/memory_plan.hpp` +
  `graph/arena.hpp`, per-lifetime arenas, liveness on the reachability partial order, and the demo
  reports the footprint (§18). AI-3: fused GEMM epilogues and a fused cross-entropy gradient tail,
  35 recorded calls per step down to 25, and the demo prints the fusion report including its
  refusals (§19).
- **No pinned reference loss curve.** The demo reports the trajectory; nothing compares it to a
  stored one, so a convergence regression that still converges would pass.
- **No per-node timing table.** The runtime only records timings under a
  `RuntimeSettings::profiling` context, which the demo does not build; the report says so rather
  than printing zeroes.
- **MNIST is opportunistic.** The IDX files are not shipped and not downloaded, so the demo runs
  on the seeded synthetic problem unless someone puts them in `data/mnist`, which means the
  "MNIST-class" part of this gate is untested in CI.

v1.x candidates after the gate: PINN training (aligning with the engine's forward-evaluator
roadmap), GNN, Python frontend (v2).

---

## 24. Open questions (explicitly unresolved)

1. **FFT backend**: deferred by D1; decide after SB lands (own SYCL FFT vs optional oneMKL DFT
   vs drop).
2. **portBLAS license/health check** (SB-2): due diligence before the port lands; the port
   itself and its attribution mechanism are decided (D5).
3. **`cli/` package module name for `sb`** (ST-1): `sushiblas/` (matching `sr`/`se` convention)
   vs a `_cli`-suffixed name reserved against a future Python tensor-binding package. Default if
   undecided at implementation time: match the `se` convention.
4. **Value-dependent control flow in autograd** (AI-1): out of scope for v1; revisit for
   PINN/GNN.

---

## 25. Substrate findings, 2026-08-03 (these change AI-1/AI-2/AI-3 as written above)

A full literature review plus a four-repo API survey turned up substrate facts that invalidate
parts of §17–§19's design as originally sketched. Recorded here because each was expensive to
find and each would be expensive to rediscover.

**AI-0 is closed.** The Release build passes: a full `sa build
--type release` (208/208, reconfigure in place) compiled `test_leaky_relu.cpp`, `test_gelu.cpp`
and `test_constant.cpp` cleanly, and `sa test --suite all` is green. It predated the SB-3/SB-6
commits. §16 above has been corrected; SushiBLAS's own CHANGELOG is that repo's to fix.

**§17's "backward rules emit into the same TaskGraph" must become "emit into SushiAI's own IR."**
Four independent reasons: a `TaskGraph` node carries `ResourceRegion` *byte ranges*, which cannot
be assigned before the memory planner runs, which itself needs the joint forward+backward graph;
fusion partitions a DAG, and there is no DAG left once differentiation has flushed to the runtime;
closure over its own output is the one property that makes higher-order derivatives fall out
(AOTAutograd cannot do double-backward precisely because it lacks it); and `TaskGraph::compile()`
**drains** its builder, so it is not a structure that can be held and rewritten. Pipeline:
forward IR → differentiate → fuse → plan → lower to `TaskGraph` once → compile once → replay.

**Compile-once/replay has an unguarded footgun.** `CompiledGraph::rearm()` before
`capture_base_state()` leaves `base_in_degree` empty, so the node loop no-ops *but the sentinel is
still stored with in-degree 0*: it fires immediately and the run reports complete before any work
runs. No assert catches it, and sushiruntime has zero tests that replay a raw `CompiledGraph` or
that reference `rearm`/`capture_base_state` by name. Wrap both calls in one RAII type.

**§18's "single arena" is the wrong shape.** The dependency tracker shards its access history on
the *base pointer*, so one arena collapses 1024 shards into one chain and makes hazard resolution
quadratic. Worse, memory reuse *injects* WAR/WAW anti-dependencies: reusing bytes across parallel
branches silently serialises them: still correct, but the parallelism loss is invisible. Use one
arena per lifetime class (parameter / gradient / optimizer-state / input / activation / scratch),
and compute liveness over the **reachability partial order**, not linear timestep intervals.

**§19's fusion is reachable but bounded.** `sycl_ext_codeplay_kernel_fusion` has been *removed*
from intel-llvm, so there is no runtime IR-level fusion. Every functor composition is a distinct
template instantiation, so composition is a compile-time choice and a "dynamic fusion pass" can
only *select among pre-instantiated combinations*. The generic helpers that make composition
possible (`execute_binary` and friends) live in `src/` and are not installed, so they must be
promoted to `include/` before SushiAI can reach them at all. Highest value-per-line change
available: an `Epilogue` template parameter on `kernels/sycl/blas/level3/gemm.hpp`'s write-back
line, signature `T operator()(T value, int64_t row, int64_t col)`: the row/col arguments are what
make bias broadcast expressible. Note the trap cuBLASLt's `GELU_AUX_BIAS` exists to solve: an
activation epilogue in a *forward training* pass must write its pre-activation to an aux buffer or
be invertible, or the backward is silently wrong.

**The op registry cannot use static-initialiser self-registration.** `sushiai` is a STATIC
library; linker GC drops a TU nothing references, so self-registering ops vanish in some link
configurations with no diagnostic. A `constexpr` table plus a `static_assert` uniqueness check is
correct, greppable and free. (32-bit FNV-1a `OpID`s do collide eventually: the assert is not
paranoia.)

**Most SushiBLAS ops ignore `Tensor::storage_offset`.** Only the reductions, `logic().all/any`,
`random().constant` and `io()` honour it; the other ~28 files capture `t.storage->data_ptr`
directly. Feeding a sliced view to gemm/axpy/elementwise silently operates from element 0. This is
a live latent bug today and a blocking one for any arena design: fix it in SushiBLAS before AI-2.

> **Fixed in SushiBLAS (2026-08-03), with one remainder.** Every op family honours the offset now,
> `Storage::data_ptr` is private, and `Tensor::region()` reports a precise byte interval. The
> remainder is the mirror image of the original bug and was found by AI-2's first packed run:
> `Dispatch::execute_reduction` computes the right *pointers* but builds its `ResourceRegion`s from
> those pointers rather than from `Tensor::region()`. For a tensor at offset zero the two coincide;
> for an arena view they name a resource nothing else touches, so the tracker emits **no edge** and
> the reduction races its own producer. Observed as `reductions().max` reading un-written logits and
> `blas().scal` beating `reductions().sum` to the loss. Worked around in SushiAI by keeping those
> operands out of the arena (`Lowering::values_needing_own_allocation()`); the real fix is one line
> per region in that dispatch header.

**The GTX 1060 is currently not executing anything.** NVIDIA's OpenCL driver implements no
SPIR-V ingestion (`clCreateProgramWithIL` is absent), so the SYCL OpenCL backend cannot reach the
card at all, which is why every prior session observed "GPU auto-detect falling back to
CPU/OpenCL". Reaching the GPU requires intel-llvm built with `--cuda` and
`-fsycl-targets=nvidia_gpu_sm_61`, and note that intel-llvm documents support only for sm_75+, so
sm_61 is below the tested floor. Until then every benchmark number on this machine is a CPU
number, and ground truth #23's fp16 caveat is moot because no fp16 runs on the GPU either.

**An MLP training step needs no new SushiBLAS kernels.** Stable log-softmax and cross-entropy
compose entirely from already-built ops: global (not per-row) max shift via `reductions().max`:
numerically sufficient, since all it has to guarantee is that every `exp` argument is ≤ 0: row
sums via `gemv` against a ones vector, and broadcast of the log-sum-exp back over rows via `ger`
(which is unconjugated, so `ger(ones, b, C)` is also the bias-add). A fused
`log_softmax_rows` op remains worth adding later for the kernel-count win, not for correctness.

**Numerics that are silently wrong if written the obvious way** (each verified against primary
sources): AdamW's `eps` goes outside the sqrt and is added *after* dividing by
`sqrt(bias_correction2)`: the other order makes the effective epsilon 31.6× too large on step 1;
LayerNorm's `eps` goes *inside* (different units); `t` starts at 1; decoupled decay applies to
`theta_{t-1}` before the adaptive step; gradient clipping must run unscale → clip → step. And
finite-difference gradient checks must run in FLOAT64: fp32 central differences bottom out near
2.4e-5 relative accuracy, which cannot distinguish a correct gradient from a real bug.
