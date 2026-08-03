# Changelog

All notable changes to SushiAI are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — versions follow [Semantic Versioning](https://semver.org/).

The project has not yet been assigned a semantic version — it is in
prototype stage. Formal per-release entries begin at the v1.0.0 cut; until
then, `[Unreleased]` records the prototype-era milestones informally. See
[CONTRIBUTING.md §5](CONTRIBUTING.md#5-documentation-is-part-of-the-change).

---

## [Unreleased]

### Fixed
- **`Trainer::score_accuracy()` refuses logits it cannot read instead of
  reinterpreting them.** It chose between `double` and `float` with `float` as
  the fallback, which was right while every graph was fp32 or fp64 and became a
  silent wrong answer the moment a precision policy could hand it a half
  buffer: `data_as<float>()` over half data reads two elements as one and
  reports an accuracy that is fiction — nothing throws, nothing is `NaN`, and
  the run says 100%. It now names the dtype and says how to keep the logits
  readable. Found while wiring AI-4; the shipped path never reached it, because
  `build_classifier` lists the logits as observed and the cast pass cannot
  narrow an observed value. `Integration_MixedPrecision.AccuracyRefusesLogitsItCannotRead`
  builds the graph the unsafe way on purpose and pins the refusal.
- **Log-softmax shifts by each row's own maximum, not the batch's.** The
  previous shift returned `+inf` for any row about 104 below the batch maximum,
  which made the loss `NaN` and every gradient under it non-finite — so a
  diverged run trained on `NaN` and reported success rather than a large loss.
  The justification for the global shift was that log-softmax is invariant to
  subtracting any single scalar from a row, which is true of the subtraction and
  false of the recipe: what destroys the invariance is the exponential
  underneath it, and a row far enough down has every one of its exponentials
  round to zero, a row sum of exactly zero, and `log(0) == -inf`.

  Measured in binary32 on a [2, 3] input whose true log-probabilities are −0.41
  to −2.41. The quantity is the worst absolute **deviation from the answer the
  same code gives at a gap of 0** — the one figure directly measured on both
  sides, by `TheAnswerDoesNotDependOnTheGapAtAll`'s `EXPECT_EQ`:

  | row-to-batch-max gap | before | after |
  |---------------------:|--------:|-----------:|
  |                   90 | 2.15e−06 | 0 (exactly) |
  |                  100 | 2.70e−02 | 0 (exactly) |
  |                  104 |    `inf` | 0 (exactly) |
  |                  200 |    `inf` | 0 (exactly) |

  The fix does not move the boundary, it removes it: the gap never enters the
  arithmetic, so the answer does not depend on it — which is why that assertion
  is an **exact equality** rather than a tolerance. Measured against the
  hand-computed expectation directly instead, the gap-100 case was 2.7033e−02
  off, 6.6% of the largest true log-probability (−0.40761). The residual at the
  gaps that pass is not quoted on either side because it was not measured; a
  passing `EXPECT_NEAR` only bounds it under 1e−4.

  `tests/functional/regression/test_log_softmax_stability.cpp` also pins the
  loss and the fused cross-entropy gradient at every gap — the two places the
  `+inf` actually ends a run.

  It needed a **row-wise maximum**, which needed an axis reduction SushiBLAS did
  not expose; that reduction (`ReductionOps::max_along` / `sum_along`, over a new
  segmented launch shape) is the sibling half of this change. The lowering is one
  recorded call *cheaper* than before, not dearer: a per-row maximum is already a
  vector, so it broadcasts back over the columns through the `ger` the recipe
  performs anyway and needs no broadcast kernel of its own. `record_log_softmax()`
  goes from ten recorded SushiBLAS calls to nine — and `emitted_ += 9` was
  already claiming nine, so the reported `ops per step` was under-counting by one
  and is now accurate at the same value. Steady-state cost on `sa demo mlp`,
  interleaved A/B on a quiet CPU/OpenCL box, slope between 1 024 and 4 096 steps:
  **0.50 → 0.45 ms/step** (median of three each; the box's run-to-run spread is
  about ±0.05, so this is a small real saving rather than a precise one). The
  demo's loss trajectory is unchanged to the last printed digit.

### Added
- **AI-4 (part 1 of 2): mixed-precision dtype plumbing — a precision policy
  and a cast-insertion pass.** fp16 compute with fp32 master weights, expressed
  as structure in the IR rather than as a runtime reinterpretation.
  **Loss scaling and overflow backoff are explicitly not in this change**, so
  an fp16 gradient small enough to underflow is still simply lost.
  - **`graph/precision.hpp` is the policy** — `IPrecisionPolicy` answers, per
    operation, which element type it computes in, split from the pass the same
    way `fusion_selector.hpp` is split from `fusion.hpp`: the pass is a
    mechanical rewrite, the policy is a set of judgements about numerics that
    has to be argued. `UniformPrecisionPolicy` is the default and answers "the
    graph's own type" to everything, so nothing that existed before this change
    behaves differently.
  - **`MixedPrecisionPolicy` narrows two families and nothing else.** `MATMUL`
    and the fused GEMM forms, because that is where the FLOPs are; and
    `BIAS_ADD` plus the *foldable* activations, because a `GEMM_BIAS_RELU` node
    applies its epilogue inside the GEMM's kernel in the GEMM's element type —
    so the unfused spelling has to agree or fusion would change the numbers,
    which the fusion pass promises it never does. Everything that sums many
    terms stays wide (`SUM_ROWS`, `LOG_SOFTMAX`, `NLL_LOSS`, `ACCUMULATE`), and
    so does the cross-entropy tail (`EXP`, `SUB`, `SCALE`): its gradient is
    `(softmax(x) − y)/N`, a subtraction of nearly-equal numbers for the correct
    class, and cancellation is the one thing loss scaling cannot rescue. `GELU`
    stays wide too — nothing folds it, so nothing forces its precision, and its
    tanh approximation computes `x*x*x`, which overflows fp16 at |x| ≈ 40.
  - **`graph/cast_insertion.hpp` is the pass** — host lane, pure IR to IR,
    the same shape as `fusion.hpp` and `memory_plan.hpp`. It inserts an
    `Ops::CAST` node at every dtype boundary so that every node's operands
    agree with its computed type *by construction*, and it checks that
    invariant on its own output before returning.
  - **It runs between `differentiate()` and `fuse()`, and both neighbours are
    load-bearing.** After differentiation because a backward node needs casting
    as much as a forward one and there are none to cast before the transform
    runs; before fusion because a conversion dropped into a `MATMUL` →
    `BIAS_ADD` → `RELU` chain would break the chain the matcher looks for.
  - **Master weights are never rewritten.** A parameter keeps its id, its
    `FLOAT32` dtype and its `Lifetime::PARAMETER`; so does its gradient, and
    the optimizer is untouched. The fp16 a GEMM reads is a *derived* value with
    an activation lifetime. The backward direction falls out of the same rule:
    a gradient is a persistent lifetime class, so it is pinned, so an fp16
    matmul writes a fresh narrow result and a cast widens it into the buffer
    the optimizer accumulates in.
  - **`Graph::find_dtype_mismatches()` states the invariant as a function**, so
    it can be checked after the fact — including after fusion has rewritten
    nodes, which is exactly where a master weight could be swallowed.
    `TrainerOptions::precision` runs it before lowering and refuses a graph
    that fails, which is the last point at which a misplaced conversion is
    still visible rather than a model that merely learns worse.
  - **`sa demo mlp --precision fp32|mixed`**, defaulting to `fp32`. The demo
    reports the conversions it inserted, in both directions, next to the fusion
    report and for the same reason: every way this pass can be wrong produces a
    run that still converges.
  - **What it is worth on this machine: correctness, and nothing else.** There
    is no fp16 matrix unit here, so half arithmetic is emulated and
    `--precision mixed` is *slower*. Measured on `--epochs 8 --batches 16`,
    same seed, one binary: 25 → 35 recorded operations per step (the ten
    conversions, and nothing else structural — the fused forms all survive),
    2105.9 ms → 9364.9 ms wall clock, and a loss trajectory that tracks fp32
    to 9.5e−4 relative at the largest printed loss and finishes at the same
    0.000380 with the same 100% accuracy. Repeated at `--hidden 512
    --batch-size 64` — a 512-long fp16 reduction inside each product — both
    arms reach 0.000003 after 48 steps. **On this workload, fp16 without loss
    scaling neither stalls nor diverges**, which means the loss-scaling work
    that follows is justified by theory and by larger models rather than by
    anything measured here.
  - **`gradcheck` is deliberately not extended to fp16.** A central difference
    in half precision has a noise floor around 1e−2, three orders of magnitude
    *worse* than the discrepancy a wrong backward rule produces, so such a
    check could not fail for the right reason. A gradient rule is a statement
    about the mathematics, so verifying it once in FLOAT64 verifies it for
    every precision; what mixed precision can get wrong is the plumbing, and
    that is a graph-shape question.
  - Consumes SushiBLAS's new `engine.elementwise().convert()`, the one
    operation in that library whose operands may disagree on dtype.

- **AI-3: automatic operator fusion — a static tree pass and a dynamic
  selector.** A step's forward and backward graph goes from 31 recorded
  SushiBLAS calls to 21 on `sa demo mlp`'s model (35 to 25 counting the
  optimizer).
  - **Fusion is a selector over a pre-compiled catalog, not a code generator.**
    `sycl_ext_codeplay_kernel_fusion` is gone from intel-llvm, so a functor
    composition is a C++ template instantiation and the set of compositions is
    fixed when this library is compiled. `graph/fusion.hpp` matches subtrees
    against that catalog; it cannot synthesise an entry and does not pretend to.
  - **`graph/fusion.hpp` is the static pass** — host lane, pure IR to IR, no
    engine and no SYCL, unit-tested over synthetic graphs with no device. It
    runs after differentiation, on the joint forward-and-backward DAG, and
    matches longest-chain-first so a ReLU layer folds whole and a GELU layer
    still gets its bias folded.
  - **`graph/fusion_selector.hpp` is the dynamic half** — the bind-time choice
    between matched entries, from `SushiBLAS::Device::Profile` and the engine's
    `ExecutionMode`. Device facts live only here, so the IR is identical on
    every machine and only one node's spelling changes.
  - **Five catalog entries:** `gemm_bias`, `gemm_bias_relu`,
    `gemm_bias_sigmoid`, `gemm_bias_tanh` through
    `Dispatch::execute_gemm`'s epilogue, and `exp_sub_scale` — the whole
    elementwise tail of the cross-entropy gradient — through
    `Dispatch::execute_zip_inplace`.
  - **The per-activation decision is a property of the op, not a case in the
    pass.** `Ops::ACTIVATIONS` states where each activation's gradient reads;
    the differentiation pass and a `static_assert` over the catalog both read
    it, so a fused form for an activation differentiated at its *input* fails
    the build rather than training a model towards the wrong place in silence.
    ReLU now differentiates at its output, which is bit-identical
    (`relu(x) > 0` exactly when `x > 0`) and is what leaves its pre-activation
    with a single consumer so an ordinary use count licenses the fold. GELU is
    declined and the report says why.
  - **Byte for byte, not within a tolerance.** Every fused form is compared
    against the unfused graph over a full training run in
    `Regression_Fusion`, and `FusionOptions::enabled = false` keeps that
    reference reachable — the same shape `Optim::UpdateKernel::SEPARATE` has.
  - **A fused epilogue's operands are declared to the dependency tracker.**
    `Dispatch::execute_gemm` describes A, B and C and cannot see into the
    functor, and the bias it dereferences is a parameter the optimizer rewrites
    inside the same plan — an undeclared read is not a wrong edge, it is *no*
    edge. `Lowering::record_fused_gemm` therefore passes the functor and its
    operands together, `gemm_epilogue(BiasReluEpilogue{...}).reads(bias)`, one
    per catalog arm; the edge lands on the GEMM task itself.
    `Regression_Fusion.AFusedGemmIsOrderedAgainstTheTaskThatWritesItsBias`
    asserts it on `in_degree` rather than on a loss.

    This was originally paid for differently: a `declare_epilogue_operand()`
    that emitted a `TaskGraph::add_native` node running nothing, one per fused
    GEMM, counted apart from the kernels by a
    `RunReport::ordering_nodes_per_step`. SushiBLAS's seam takes the epilogue's
    access set now, so the node, the counter and the
    `Ops::EPILOGUE_OPERANDS` identity are all gone and `operations_per_step` is
    the plan's node count again. On `sa demo mlp`'s model that is **27 plan
    nodes per fused step before and 25 after**: `ops per step` is unchanged at
    25 — no kernel was added or removed — and the two host-side nodes that sat
    beside them are gone, one per taken GEMM-epilogue fusion (`gemm+bias+relu`
    and `gemm+bias`; `exp-sub-scale` never needed one).
  - **The pass explains itself.** `FusionReport` names every subtree it
    collapsed and every one it refused, with the value that blocked it and the
    reason — `sa demo mlp` prints it, because every way this pass can be wrong
    produces a run that still converges.
  - **An auxiliary-buffer GELU epilogue is declined, not approximated.** It was
    declined because the seam declared no epilogue-side regions, so an aux
    buffer the functor wrote would get no dependency edge and its reader would
    race the write. That half is fixed —
    `Dispatch::gemm_epilogue(f).writes(aux)` puts the buffer in the GEMM task's
    write set — and the fusion itself is still not implemented, because it also
    needs an op identity, a catalog row, a second output on the fused node, a
    lowering arm, a backward rule that reads the spill, and a `static_assert`
    that can tell "no aux buffer" from "an aux buffer is provided". Named
    rather than worked around, as before; the seam is no longer the reason.
  - **The wall clock has two terms and only one is an improvement.** A step's
    steady-state cost falls from 0.57 to 0.40 ms on `sa demo mlp`'s model, but
    the first step rises from 0.97 to 1.88 s because each fused form is a
    distinct SYCL kernel this backend compiles at first launch. Break-even is
    about 5 000 steps, so the demo's default 128-step run is ~1.9x slower in
    wall clock while issuing 29% fewer kernels. Both halves are in
    `docs/README.md`; AOT targeting is what moves the one-time term to build
    time.
  - **`sa demo mlp --fusion off` and `--no-fuse NAME`** make that A/B
    reproducible from one binary, which is the only way the claim is checkable.
- **One kernel launch per parameter, and the optimizer inside the compiled
  step.** AdamW's update used to be about eleven SushiBLAS calls per parameter
  per step — ~88 launches for an eight-tensor model, against the ~31 the whole
  forward and backward graph costs. Both optimizers now issue one
  `Dispatch::execute_zip_inplace` per parameter instead, and that update is
  recorded into the same `TaskGraph` as the rest of the step.
  - **The temporaries became registers.** `g²`, the denominator, `m/denom` and
    SGD's staged gradient existed only to carry a value between two launches.
    The fused path allocates none of them, including the buffer of ones whose
    only job was to let an `axpy` add epsilon.
  - **A device pointer is what unlocked folding it into the plan.** The
    recorded obstacle was that SushiBLAS takes its alphas by value and AdamW's
    bias corrections change every step, so a compiled `axpy` could never be
    handed a number that changes between replays. One functor capturing a
    pointer to `Optim::StepScalars` sidesteps that entirely: the pointer is
    capturable, the bytes behind it are not frozen, and the recorded work
    becomes launch-invariant. `Graph::IStepRecorder` is the seam;
    `TrainerOptions::optimizer_placement` is the switch.
  - **Byte for byte, not within a tolerance.** The fused kernel and the
    separate operations produce bit-identical parameters, and so do the folded
    and unfolded placements. Both are pinned in
    `Regression_FusedOptimizer`. That was reachable because the stack builds
    with `-ffp-contract=off` through SushiRuntime's determinism contract, so no
    `a*b + c` collapses into an FMA on one side of the comparison and not the
    other, and because every fused expression keeps the operand order of the
    kernel it replaced.
  - **`UpdateKernel::SEPARATE` keeps the reference spelling reachable**, and a
    trainer asked to compile one into the step throws instead of quietly
    stepping it outside — folding it would replay step one's bias corrections
    for ever, and the model would converge anyway.
  - **Calling `step()` on an optimizer the plan already owns is now an error.**
    The update used to be visible in `Trainer::run_step()`, so a caller who
    also stepped it got an obvious double update; inside a replay the same
    mistake would quietly train at twice the learning rate.
  - **Turning SGD's momentum on after construction is now an error.** The
    momentum buffer is allocated at construction or not at all; the separate
    path used to dereference a null tensor and the fused path would have
    trained without momentum and said nothing.
- **AI-2: a static memory planner.** `graph/memory_plan.hpp` assigns every
  non-persistent IR value an (arena, byte offset) before anything is lowered;
  `graph/arena.hpp` allocates each arena once at bind time and hands out
  views. `sa demo mlp` goes from 130.9 KiB to 112.6 KiB of buffers, with the
  same loss trajectory to the last digit.
  - **One arena per lifetime class, and only the non-persistent classes.**
    SushiRuntime's dependency tracker shards its access history on an
    allocation's base pointer, so one global arena collapses every shard into
    a single chain and makes hazard resolution quadratic. Parameters,
    gradients, optimizer state and inputs are never reused, so packing them
    would save nothing while merging the shards that matter most.
  - **Liveness is computed on the reachability partial order.** A value may
    hand its bytes to another only when every use of the first is a strict
    ancestor of the second's first write. The interval version — sort the
    nodes, overlap disjoint `[first_use, last_use]` ranges — passes every
    obvious test and quietly serialises branches the runtime could have run in
    parallel, because memory reuse *injects* WAR and WAW anti-dependencies and
    only the reachability form guarantees the injected edge was already there.
    `Unit_MemoryPlan.TwoUnorderedBranchesNeverShareBytes` is the diamond the
    interval planner gets wrong.
  - The closure is O(V²/64) memory, so above `PlanOptions::reuse_node_limit`
    nodes the planner **declines to overlap anything and says so** through
    `reuse_note()`, rather than silently substituting the linear version.
  - **Mixed dtypes are rounded to their own element width.**
    `Tensor::storage_offset` counts elements, so a byte offset that is not a
    whole number of them has no representation; `view_tensor()` refuses one
    rather than truncating it into a tensor that starts in the wrong place.
  - `ACCUMULATE`'s result *is* its accumulator's storage, so the two are
    planned as one buffer with one live range. Planning them apart would let
    the accumulator's bytes be handed away while the result is still read.
  - Proved rather than asserted:
    `Regression_StaticMemory.ATrainingLoopAllocatesNothingAfterTheFirstStep`
    counts live allocations through the allocator's reference count, and
    `PackedAndUnpackedTrainingAgreeByteForByte` pins that packing changes no
    number at all.
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
  suite off instead of compiling, discovering and registering 153 tests it
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
    compiled, not re-filled on every replay. *(Both superseded by the AI-2
    entry above: SushiBLAS honours `storage_offset` now, the non-persistent
    classes are packed into arenas, and `ZERO` refills every replay.)*
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
- **A resilience and durability pass over the suite** — 53 further cases
  (153 SushiAI cases in total), all of them written to fail if the thing they
  name breaks rather than to raise the count:
  - `tests/common/test_throws.hpp` and `SA_EXPECT_ERROR`, which assert the
    *content* of a diagnostic as well as the throw. `EXPECT_THROW` alone
    survives a refactor that collapses five specific messages into one
    generic one, which is the failure this guards.
  - `Unit_Diagnostics` (20 cases): every host-lane precondition — rank past
    `MAX_RANK`, a non-positive extent, a contraction mismatch, a rank-1
    operand where rank 2 belongs, a dtype an op cannot serve, a second
    producer for one value, a loss no rule can seed — asserted to name the
    offending value.
  - `Integration_Resilience` (21 cases): the device-lane preconditions in the
    same style, plus degenerate extents (one sample, one feature, one class),
    the GEMM register-block edge and one past it against a host reference,
    the numerical extremes the max shift and AdamW's `eps` placement exist
    for, and a construct/destroy loop asserted against the allocator's own
    reference count so a leaked `Storage` is a number rather than a hunch.
  - `Regression_Determinism` (5 cases): two identical runs, bit for bit —
    parameters, losses, initialisation, the synthetic dataset and a bare
    replay of one compiled step. This is the `sushiruntime_deterministic_fp`
    claim the ordered `ACCUMULATE` chain exists to keep, and nothing proved
    it end to end before. It holds.
  - `Regression_LearningCurve` (2 cases): the loss trajectory of a fixed
    seed and configuration, pinned to a recorded reference. The test that
    catches "everything passes and the model stops training".
  - `Regression_CompileOnce` gained the inverse of its own guarantee — that a
    replay actually dispatches, caught at run #1 by poisoning the output
    buffer rather than at replay #2 by watching the loss go flat — and a
    stress case holding `compile_count() == 1` across changed batch contents,
    a stopped and resumed trainer, interleaved optimizer steps and a
    learning-rate schedule.
  - `Unit_Backward` gained a coverage ledger that fails when an operation
    gains a backward rule with no finite-difference test named against it,
    and `Integration_Autograd` gained the two cases the walk of that table
    found missing: `ACCUMULATE` in a forward graph (its rule had never been
    reached, because the fan-out chains are emitted into the *backward* graph,
    which `differentiate()` does not revisit) and `MUL`'s right operand.

### Changed
- **The library is built at C++17 instead of C++20**, matching SushiBLAS and
  SushiRuntime, so that one standard now holds across the whole stack. SushiAI
  was the last repository above the line. This is not tidiness:
  `SushiRuntime::span` (`SushiRuntime/core/span.hpp`) selects between
  `std::span` and SushiRuntime's own shim on `__cplusplus`, and that selection
  reaches the compiled ABI, so SushiAI at C++20 over siblings at C++17 had the
  two halves of one build reading that branch differently.
  **No API changed** — the same functions take the same arguments, spelled the
  same way, and every header still compiles under C++20. What did change is the
  linkage of the entry points that take a span and are *not* inline
  (`Autograd::differentiate`, `Core::allocate_tensor`,
  `Optim::StepScalars::write`, the gradcheck entry points): they export the
  shim's mangling now, so a consumer must be at C++17 too. That is the one
  place SushiAI cannot mirror SushiBLAS, whose span-taking API is entirely
  header-inline and whose `tests/package/` therefore stays at C++20;
  SushiAI's moves to 17, with the two mangled symbols recorded in the file so
  the difference is documented where someone would change it back. See
  `INTEGRATION.md` §3.
  The C++20 facilities the codebase used were replaced rather than removed:
  - `std::span` -> `SushiRuntime::span` at 105 sites across 31 files. It *is*
    `std::span` wherever `std::span` exists, so no signature changed shape.
  - `std::format` -> `SushiBLAS::Core::format`, **shared with SushiBLAS rather
    than copied a third time**. SushiBLAS already carried the stand-in as an
    installed public header and SushiAI links SushiBLAS unconditionally, so the
    alternative was a third copy of one file to delete in a third place when
    the stack returns to C++20. Its entry point was promoted out of
    `SushiBLAS::Core::Detail` to a documented `SushiBLAS::Core::format` for
    this — an addition to SushiBLAS's API, not a break; `Detail::format` is
    untouched and the public entry forwards to it. The shared header was
    extended to cover the specs SushiAI writes and SushiBLAS did not:
    fill/alignment (`{:<18}`, `{:>8.3f}`, `{:^9}`) and the integer
    presentation types (`{:08x}`, `X`, `o`, `d`). Forking it would have been
    the dishonest fix. No message text changed.
    What is genuinely lost is `std::format`'s `consteval` check on a literal
    format string: a placeholder count that disagrees with the argument count
    is now a best-effort message rather than a compile error — the same loss
    SushiBLAS and SushiRuntime already took.
  - `concept Module` / `concept ModuleContainer` -> `constexpr bool` variable
    templates over `std::void_t` detectors, so every use site
    (`Module<Linear>`, `if constexpr (ModuleContainer<M>)`, the eight
    `static_assert`s in `layers.hpp`) is spelled exactly as before and keeps
    its message. `Module` still demands that `forward()` return **exactly**
    `Graph::ValueId` (`std::is_same`, not convertible-to).
    `ModuleContainer`'s probe could not be transliterated: its lambda lived in
    an unevaluated context, which is C++20-only, so it became a named
    `Detail::ChildProbe` function object — still generic in the child type,
    because a probe that was less permissive would quietly answer "not a
    container" and `for_each_parameter` would stop recursing with no error.
    `Unit_Module.NestedContainersCompoundTheirIndices` is the test that pins
    it.
  - `template<Module... Ls> class Sequential` -> `template<typename... Ls>`
    plus a `static_assert` fold at the top of the class, which still states the
    contract in a sentence and still fails against `Sequential`'s own
    instantiation — clang's fold diagnostic still names the offending layer
    (`requirement 'Module<NotALayer>'`). What is lost is that
    `Sequential<NotAModule>` is now a valid specialization that fails when
    instantiated rather than one that never matched — it can no longer be
    SFINAE'd away. Nothing overloads on it.
  - `consteval` -> `constexpr` at all three sites (`Ops::op`,
    `Ops::all_ids_unique`, `every_fused_activation_survives`). Each is called
    only from a `static_assert` or a `constexpr` variable initialiser, both of
    which force constant evaluation, so nothing moved to run time: the OpID
    collision proof was re-checked by introducing a colliding name and
    confirming the `static_assert` still fires, through the C++17 span shim.
  - `std::string_view::starts_with` -> `rfind(prefix, 0) == 0` in
    `Unit_Ops.EveryNameCarriesTheLibraryPrefix`.

### Known limitations
- The optimizer update and the per-step batch feed are not part of the
  compiled plan: AdamW's bias corrections are per-step scalars and SushiBLAS
  takes its alphas by value, and a compiled call captures raw pointers so a
  batch must be copied into fixed buffers rather than bound to them. Both
  fold into the plan once SushiRuntime's late-binding block reaches the BLAS
  layer.
  - *Superseded for the optimizer* by the fused-update entry at the top of
    `[Unreleased]`: a functor capturing a device pointer needs no late binding
    at all, and the update is in the plan now. The batch feed's half stands,
    but the stated fix does not: `LateBindings` re-points a node's iteration
    count, base index and enabled flag, not the operand addresses its kernel
    captured. See [ARCHITECTURE.md §7](ARCHITECTURE.md#7-whats-not-here-yet).
- `LOG_SOFTMAX` is differentiable only as part of the fused
  `LOG_SOFTMAX` + `NLL_LOSS` pair; the standalone rule needs a row-wise
  reduction SushiBLAS does not expose, and throws saying so.
  - *Partly superseded* by the row-wise-shift entry at the top of
    `[Unreleased]`: SushiBLAS exposes `sum_along` now, so the blocker is no
    longer a missing kernel. It is a missing **IR node** — a backward rule
    writes IR, not SushiBLAS calls. The rule still throws; its message says
    which of the two it is waiting on.
- **`LOG_SOFTMAX` shifts by the batch's global maximum, and that has a
  numerical edge.** Shift invariance is exact in real arithmetic but not in
  floating point: a row whose own maximum sits far below the batch maximum has
  every exponential rounded to zero, so its row sum underflows and its
  log-probabilities come back `+inf`, which makes the loss and every gradient
  under it non-finite. Measured in binary32: exact to a gap of about 90, about
  7% low at 100, `+inf` from about 104 on. Rows a training run actually
  produces sit within a few tens of each other, so this is reachable only from
  a model that has already diverged — but it turns divergence into silent NaN
  rather than a large loss. Removing the edge needs a real row-wise maximum,
  which needs either an axis reduction SushiBLAS does not expose or the fused
  `log_softmax_rows` that would replace the whole seven-op recipe. Documented
  in `lowering.hpp`; the supported side of the boundary is pinned by
  `Integration_Resilience.LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange`.
  - ***Fixed***, by the row-wise-shift entry at the top of `[Unreleased]`.
    SushiBLAS grew the axis reduction; `record_log_softmax()` shifts by
    `max_along(x, 1)` and the gap no longer reaches the arithmetic at all.
- `Lifetime` is recorded on every value but no memory planner reads it yet.
  *(Superseded: the AI-2 entry above is that planner.)*

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
- **`ZERO` filled its buffer once instead of on every replay.** Correct while
  every value owned a permanent buffer, and silently wrong the moment the
  memory planner could hand those bytes to a later value: the buffer would
  hold zeros for exactly one step and whatever overwrote it thereafter, which
  trains — badly — for ever. The fill is now recorded into the step, at the
  cost of one small kernel per `ZERO` node.
- **`TensorPool::alias()` threw when the target was already bound to the same
  buffer.** Reachable whenever a parameter received more than one gradient
  contribution: `bind_parameters()` materialises the resolved gradient, which
  for an accumulated one is the `ACCUMULATE` output, and the lowering then
  tried to alias an already-bound value. It is now idempotent for an identical
  binding and still refuses a different one.
- **`differentiate()` returned empty gradients and no error for a loss no rule
  can seed.** The root's own gradient is the implicit scalar one, so
  `resolve()` hands back `NO_VALUE` for it and every rule except
  `nll_loss_backward` reads that as "nothing flows here" and returns. A
  1-element value produced by anything else — a `[1, 1]` `MATMUL`, say —
  therefore passed the scalar check and differentiated to a complete set of
  `NO_VALUE` gradients with nothing reported, contradicting the contract
  `backward.hpp` already documented ("anything else is rejected rather than
  silently differentiated"). The caller's first symptom was
  `bind_parameters()` blaming the *model* for a loss-side mistake. Now
  rejected where the cause is still known, naming the producing operation.
- **A gradient buffer was readable before anything had written it.** There is
  no `zero_grad()` in this library by design, which leaves one window —
  between `bind_parameters()` and the first replay — in which an optimizer
  stepping would read a fresh allocation and feed whatever the pool last had
  in that block straight into the weights. Nothing downstream catches that.
  `bind_parameters()` now zeroes and flushes each gradient as it materialises
  it, so a step taken before the first backward is a defined no-op.
- **`allocate_tensor()` multiplied extents with unchecked signed arithmetic.**
  Overflow there is undefined behaviour, which a compiler may assume away
  along with the `elements <= 0` check written to catch it. The product is now
  overflow-checked before the multiply, the byte count is checked against
  `size_t`, and a failed allocation is reported as a `SushiAI::Error` naming
  the shape, the element type and the byte count — SushiBLAS's own refusal
  knows only the bytes, which does not identify which of a model's tensors
  asked for them.
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
