# Remaining work

The single backlog of this repository. Work and decisions that are still open are listed here;
a defect that exists today is in [`KNOWN_ISSUES.md`](../reference/KNOWN_ISSUES.md). The design
behind the library items is in [`ROADMAP.md`](ROADMAP.md) and
[`ARCHITECTURE.md`](../architecture/ARCHITECTURE.md).

## Memory planning

- **Pack the lowering's scratch.** `MemoryPlan::build` packs graph IR values into arenas.
  `TensorPool::scratch()` and `TensorPool::ones()` are not IR values, so each still gets one
  `Storage`; on the demo that is about 2 KiB of 130 KiB. Packing them needs a host-lane
  description of each operation's scratch, shared by the plan and `Lowering`. Today those shapes
  are written inline in `record_log_softmax()` and `record_nll_loss()`
  (`src/graph/memory_plan.cpp`, `src/graph/lowering.cpp`).
- **Feed a batch without two copies.** A step issues two `blas().copy()` calls to move the batch
  into the plan's input buffers, because a recorded SushiBLAS call captures its operand addresses
  by value and every batch is a different source tensor. It needs either an operand-address late
  binding that device tasks can use, or a `Dataset` that writes each batch into the two buffers
  the plan owns. SushiRuntime's `LateBindings` re-points a node's iteration count, base index
  and enabled flag, not the addresses its kernel captured, and
  `TaskGraph::add_offload_dynamic()` carries re-pointable lists only for offloadable tasks with
  a host fallback (`src/train/trainer.cpp`).

## Fusion and the IR

- **Fuse row-wise log-softmax.** `LOG_SOFTMAX` lowers to nine SushiBLAS calls, the largest term
  in a step. `max_along` and `sum_along` exist in SushiBLAS, so what is missing is a fusion
  catalog entry (`include/SushiAI/graph/fusion.hpp`, `src/graph/lowering.cpp`).
- **Fuse `NLL_LOSS`.** It lowers to three calls because the SushiBLAS whole-tensor reduction
  takes no transform functor, so the elementwise multiply cannot ride into the sum
  (`src/graph/lowering.cpp`).
- **Differentiate standalone `LOG_SOFTMAX`.** The rule is `dx = dy - exp(L) * rowsum(dy)` and a
  backward rule writes IR, so it needs an IR node for a row-wise sum: an op identity, a builder
  entry, a shape rule, a lowering and the rule (`include/SushiAI/core/ops.hpp`,
  `src/autograd/rules/loss.cpp`).

## Training and precision

- **Skip the whole step on overflow.** `LossScaler` screens each parameter's gradient and skips
  that parameter's update. Freezing every parameter when any gradient overflows needs a
  dependency edge `execute_zip_inplace` cannot declare: one hand-written task that reads the
  count buffer and writes a gradient (`include/SushiAI/optim/loss_scaler.hpp`).
- **bfloat16.** `MixedPrecisionPolicy` takes its narrow type as a constructor argument, and
  `SushiBLAS::Core::DataType` has no bfloat16 entry (`include/SushiAI/graph/precision.hpp`).

## Data and objectives

- **Widen `Dataset` beyond classification.** `Data::Dataset::num_classes()` is pure virtual and
  `Data::Batch` carries classification targets; a regression task or a detector has no answer
  for either (`include/SushiAI/data/dataset.hpp`).
- **Feed arbitrary objective leaves.** `Train::Trainer` writes `StepSpec::targets` each step. An
  objective that declares another leaf in `ObjectiveLeaves` has nothing writing it
  (`include/SushiAI/train/trainer.hpp`, `include/SushiAI/train/objective.hpp`).

## Diagnostics

- **Pin a reference loss curve.** `sa demo mlp` reports the compile count and the steady-state
  allocations; no test compares a run's loss with a recorded one (`examples/demo_mlp.cpp`).
- **Print per-node timings in the demo.** The table needs a `RuntimeSettings::profiling`
  context, which the demo does not build (`examples/demo_mlp.cpp`, `src/train/trainer.cpp`).

## Open questions from the roadmap

[`ROADMAP.md`](ROADMAP.md) section 24 holds the wording. FFT backend for SushiBLAS: own SYCL
FFT, optional oneMKL DFT, or none. Value-dependent control flow in autograd: out of scope for
v1, to revisit for PINN and GNN work.

## Documentation

- **Make the root README a front door.** It is 584 lines and holds the demo transcript, the
  fusion and mixed-precision measurements and the directory tree. The measurements belong in
  the manual; the owner decides what stays (`README.md`).
- **Module READMEs.** No folder beside the code has one except `tools/` and
  `examples/heavy_mlp/`. The owner decides which folders are modules: the nine areas under
  `include/SushiAI/`, `cli/`, `tests/`, `apps/`.
- **Decide who owns the roadmap.** `ROADMAP.md` plans SushiBLAS and SushiAI together; sections 5
  to 15 are about other repositories.
- **Re-home the archived August plans.** `docs/archive/superpowers/` holds five files named in
  lower-kebab; the archive shape for agent work is `docs/archive/agent/`.
- **Humanizer pass** over `README.md`, `docs/guides/CLI_GUIDE.md`,
  `docs/architecture/ARCHITECTURE.md` and `docs/design/ROADMAP.md`, which run 11 to 16 em dashes
  per thousand words.
- **Compare the manual with the code.** The audit did not check the numbers in `TUTORIAL.md`,
  the prose of `ARCHITECTURE.md` sections 1 to 6, or the flag tables of `CLI_GUIDE.md`.

## Layout, tools and release

- **Declare the tiers.** `K_TIER_ORDER` in `tools/layering/check_layering.py` is empty, and the
  checker looks for `modules/<tier>/` and `source/`, which this tree does not have.
- **Folder names.** `src/`, `apps/`, `examples/`, `tests/functional/` and the root `Doxyfile`
  depart from the `repository-layout` skill, as they do in SushiRuntime and SushiBLAS. One
  decision covers the three repositories.
- **Run the checkers in CI**, add a Windows lane and a release workflow
  (`.github/workflows/ci.yml`).
- **Tag a first release.** Every changelog entry sits under `## Unreleased` and the repository
  has no tag; the declared version is 0.1.0.
- **Tie the CLI version to the library version**, or state that they move apart
  (`CMakeLists.txt`, `cli/pyproject.toml`).
- **Add `.gitattributes`** to pin line endings to the `lf` that `.editorconfig` asks for.

## Estate programmes

The estate audit of 2026-10-05 is in
[`docs/agent/2026_10_05_ESTATE_AUDIT/REPORT.md`](../agent/2026_10_05_ESTATE_AUDIT/REPORT.md).
Its "Code shape" section, 22 findings from the lowering's dispatch chain to the error model, is
the input of the code programme and is not restated here.
