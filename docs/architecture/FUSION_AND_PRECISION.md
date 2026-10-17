# Fusion and mixed precision

How operations fuse through a catalog and how mixed precision is written into the IR. Part of the architecture chapter; [`LOWERING.md`](LOWERING.md) holds §5 to §5.1.

## 5.2 Fusion is a selector over a catalog, not a code generator

`sycl_ext_codeplay_kernel_fusion` has been **removed** from intel-llvm, so
nothing in this stack fuses two kernels at run time. A fused kernel is a C++
template instantiation of a functor composition, which means *which
compositions exist* is decided when SushiAI is compiled and a pass running at
graph-build time can only choose among them. Everything about the shape of this
work follows from that one fact:

- **`ops/ops.hpp` holds the catalog's identities**: `gemm_bias`,
  `gemm_bias_relu`, `gemm_bias_sigmoid`, `gemm_bias_tanh`, `gemm_bias_gelu`,
  `exp_sub_scale`:
  each a distinct kernel in the binary rather than a flag on a generic one.
- **`graph/fusion.hpp` is the static pass**: host lane, pure IR to IR, no
  engine and no SYCL, unit-tested over synthetic graphs with no device. It
  matches subtrees longest-first and rewrites each accepted match into one
  node. It runs *after* differentiation, on the joint forward-and-backward
  graph, because a forward node's fusability depends on what the backward pass
  still reads.
- **`graph/fusion_selector.hpp` is the dynamic half**: given the device's
  `SushiBLAS::Device::Profile` and the engine's `ExecutionMode` (`Auto`,
  `ForceSYCL` or `ForceReference`), it chooses which matched entry to actually
  spell. Device facts live only here; the IR is the same on every machine and
  only one node's spelling changes.
- **`modules/core/graph/source/lowering/fused.cpp` holds the functors**, because a functor
  inlined into a SushiBLAS kernel is device code.

Adding a form is four mechanical edits: an identity, a `FusedForm`, a `K_CATALOG`
row, and a row in `K_FUSED_LOWERINGS` with its function. Nothing self-registers, for the same reason
`autograd/rules/all.hpp`'s rule table does not (§4.1).

### What is allowed to be fused away, and what is not

Three refusals, each of which produces a graph that still trains if you get it
wrong, and each therefore a checked rule:

1. **A value with a second consumer.** A fused kernel never writes the chain's
   interior values, so fusing one that something else reads means recomputing
   it or losing it. The pass counts uses; it does not assume a chain.
2. **A value the caller reads by id.** The loss and the logits are ordinary
   activations to the pass and are read by the training loop after every
   replay. `FusionOptions::observed` is how `build_classifier` says so;
   parameters, gradients, optimizer state and inputs are protected by their
   lifetime class without being listed.
3. **A value the backward pass differentiates at.** This is rule 1 again,
   deliberately: by the time the pass runs the backward nodes are in the graph,
   so a pre-activation the gradient reads is simply a second consumer.

That third point is where the per-activation decision lives, and it is a
property of the op rather than a case inside the pass. `Ops::K_ACTIVATIONS` states
where each activation's gradient reads, once, and both the differentiation pass
and the catalog's `static_assert` consult it:

| activation | gradient reads | foldable | entry it needs |
|---|---|---|---|
| ReLU | its output | yes | any; the pre-activation was dead already |
| sigmoid | its output | yes | any |
| tanh | its output | yes | any |
| GELU | its **input** | yes, **with a spill** | one that writes the pre-activation to a second output |

`K_ACTIVATIONS` has two more rows, LeakyReLU (reads its output) and SiLU (reads
its input). No catalog entry folds either.

ReLU moved from reading its input to reading its output as part of this work,
and that is a substitution rather than an approximation: SushiBLAS's
`relu_backward` is `dy * (primal > 0 ? 1 : 0)` and `relu(x) > 0` exactly when
`x > 0`, so the two spellings are **bit-identical**. Making it unconditional -
not a rewrite the fusion pass applies - is what leaves the pre-activation with
a single consumer in the plain graph too, so an ordinary use count says yes for
ReLU with no activation-specific code in the matcher at all, and would say no
for GELU. The three entries in `docs/design`'s §25 note about GELU/SiLU/sigmoid/
tanh being non-recoverable is narrower than it looked: sigmoid and tanh **are**
exactly recoverable from their outputs (`y(1-y)`, `1-y²`) and SushiBLAS's rules
already take the output, so GELU was the only one the use count genuinely
refused.

**The auxiliary-buffer GELU epilogue (cuBLASLt's `GELU_AUX_BIAS` shape) is
implemented.** `FusedForm::GEMMBiasGELU`'s epilogue stores the fully formed
pre-activation to a second buffer on its way past and then returns `gelu` of it,
so the tensor `gelu_backward` differentiates at is written after all and the
fold is exact rather than approximate.

Four things make that work, and each is a place it could have gone silently
wrong:

- **The second output is the same value.** `Graph::Node::outputs` was already a
  vector; the fused node's `outputs[1]` is the very `ValueID` the absorbed
  `K_BIAS_ADD` would produce. So `gelu_backward` reads what it always read, no
  backward rule changed, and no pass had to learn about a new tensor.
- **Rule 1 is relaxed for exactly that value and nothing else.** A catalog entry
  names one `spill` step; the interior value at that step may have any number of
  consumers and may be observed, because it still reaches memory. Every other
  interior value is governed by the use count as before.
- **Order, not just liveness.** The fused node is emitted at the chain's root,
  later than the absorbed producer, so a consumer of the spilled value that sat
  between the two would read a value written after it.
  `FusionDecline::SpillReadTooEarly` refuses that case; the training graph
  never hits it, because `gelu_backward` is appended after the forward pass.
- **The write is declared.** `Dispatch::gemm_epilogue(f).reads(bias).writes(aux)`
  puts the spill in the GEMM task's own write set, so whatever reads it is
  ordered behind the kernel that filled it. Without that declaration nothing
  fails and nothing is slower: the reader simply races the write, which is the
  hazard SushiBLAS's seam exists to close.

The `static_assert` in `fusion.cpp` does not refuse every entry folding a
`GradientSource::Input` activation; it refuses one that folds such an
activation **without** a spill, or that spills the wrong step of the chain. It
still fails the build for a hypothetical `GEMM_BIAS_SILU` added without an
auxiliary write.

Under a narrowing precision policy the entry is unreachable, and deliberately
so; see §6.2.

### The cost the epilogue carried

The same gap bit the bias. A fused GEMM's epilogue dereferences a bias vector
the tracker cannot see, and that bias is a parameter the optimizer rewrites
*inside the same compiled plan* - so an undeclared read is a write-after-read
hazard nothing orders. A transitive path through the gradient happens to exist
for a plain feed-forward model, and relying on it is exactly the accident that
survives every test and breaks on the first second writer.

Without that access set the order would be paid for with a host node of the SushiRuntime graph running nothing, present only for its two
access declarations: one node per fused GEMM, with no kernel launch.
`record_fused_gemm` (`modules/core/graph/source/lowering/fused.cpp`) hands SushiBLAS the
epilogue and its operands together,
`gemm_epilogue(BiasReLUEpilogue{...}).reads(bias)`. Each row of the fused table
builds its own epilogue and access set, because the access set belongs to the
spelling: the row that spills a pre-activation declares a write the others do
not. The edge lands on the GEMM task itself, so
`operations_per_step` is the plan's node count again.
`Medium_Fusion.AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias` asserts
that edge on `in_degree` rather than on a loss: with no edge the tasks may still
come out in the right order on any given run, so a loss that falls proves
nothing.

### What it is worth, measured, and what it costs

On `sa demo mlp`'s model a step goes from **35 recorded SushiBLAS calls to 25**,
all of it out of the forward and backward
graph: the first layer's `gemm+bias+relu` replaces five calls with one, the
second layer's `gemm+bias` replaces three with one, and the cross-entropy
gradient's `exp/sub/scale` tail replaces five with one. Every fused form is
compared byte for byte against the unfused spelling in
`tests/layers/train/test_trainer/test_fusion_numerics.cpp`, and
`FusionOptions::enabled = false` keeps that reference reachable, exactly as
`Optim::UpdateKernel::Separate` does for the optimizer. The demo's loss
trajectory is identical to the last printed digit with the pass on and off.

The wall clock has **two terms, and only one of them is an improvement**:

- **Steady state: 0.57 → 0.40 ms/step**, the slope of total time against step
  count taken between 1 024 and 4 096 steps (median of three replicates,
  spread about ±0.06). A 30% saving against a 29% reduction in kernels, which
  is what one expects when the tensors are small enough that a step is mostly
  launch overhead.
- **First step: 0.97 → 1.88 s.** Each fused form is a distinct template
  instantiation and therefore a distinct SYCL kernel, which this OpenCL backend
  compiles at first launch. The cost is one-time, per process, and independent
  of model size, batch and step count.

Break-even is therefore around **5 000 steps** on this machine, and the demo's
default 128-step run sits far below it, with fusion on it is about 1.9x slower
in wall clock while issuing 29% fewer kernels. AOT device targeting
(`SB_SYCL_TARGETS`) is what moves that term to build time. This is recorded
here rather than quietly omitted because a fusion pass is exactly the kind of
change whose headline number invites reporting only the favourable half.

---

## 5.3 Mixed precision is structure in the IR, not a runtime switch

`graph/precision.hpp` and `graph/cast_insertion.hpp` are the dtype half of
AI-4. They are split along the same seam `fusion.hpp` and
`fusion_selector.hpp` are split along, and for the same reason: the **pass**
is a mechanical rewrite anybody can verify, and the **policy** is a set of
judgements about numerics that has to be argued. Putting them in one file
would mean the argument and the machinery move together.

- `PrecisionPolicy` answers, per operation, which element type it computes
  in. `UniformPrecisionPolicy` (the default) answers "the graph's own
  type" to everything, so a graph built without a policy is byte-for-byte what
  it was before this existed. `MixedPrecisionPolicy` narrows two families and
  nothing else: `K_MATMUL` and the fused GEMM forms, and the `K_BIAS_ADD` plus
  foldable activation those forms absorb.
- `insert_casts()` reads the policy and rewrites the graph so every node's
  operands and results carry its computed type, emitting an `Ops::K_CAST` node
  at each boundary. `find_dtype_mismatches()` states the resulting invariant
  as a function anyone can call, including after fusion, which is where it
  earns its keep.

**Why the epilogue's precision is forced rather than chosen.** A
`K_GEMM_BIAS_RELU` node applies its epilogue inside the GEMM's kernel, in the
GEMM's element type. If the IR said `K_BIAS_ADD` computes in fp32, fusing it
into an fp16 GEMM would change the numbers, and §5.2 promises fusion changes
none. So the policy does not get to pick for anything a catalog entry **will**
absorb; it has to agree with the kernel that absorbs it.

**`GELU` is the one place that rule meets a limit, and the resolution is the
pass order.** Since `K_GEMM_BIAS_GELU` exists, a catalog entry *can* absorb GELU,
so "nothing folds it, therefore nothing forces its precision" is not an
argument for anything. The reason it still stays wide is arithmetic and nothing
else: the tanh approximation computes `x*x*x`, fp16's largest finite value is
65504, so the cube overflows at |x| ≈ 40.3 and `tanh(±inf)` then returns ±1:
`0.5*x*(1+tanh)` yields `x` or `0` rather than a NaN, a wrong number with no
symptom. (That last step is IEEE-754 arithmetic, not a figure measured here.)
Loss scaling cannot reach it either, because it is a *forward* activation and
the scale is applied to the backward seed.

Keeping it wide makes `K_GEMM_BIAS_GELU` an **fp32-only catalog entry**, and that
falls out of the pass order rather than needing a special case: the cast pass
runs before fusion, so an activation the policy keeps wide gets a `K_CAST` between
it and the narrowed `K_BIAS_ADD`, and a chain with a conversion in the middle
matches no entry. Under a narrowing policy a GELU layer therefore folds to
`gemm+bias` and the activation runs on its own: exactly as it did before the
entry existed, and `narrows_under_mixed_precision` answers false for both
`K_GELU` and `K_GEMM_BIAS_GELU` so the two halves cannot drift.
`Medium_MixedPrecisionRegressions.AGELULayerIsNotFoldedUnderANarrowingPolicy` pins it.

**Where the pass runs, and why exactly there.** Between `differentiate()` and
`fuse()`. After differentiation, because a backward node needs casting as much
as a forward one and there are none to cast before the transform runs; before
fusion, because a conversion dropped into the middle of a `K_MATMUL` →
`K_BIAS_ADD` → `K_RELU` chain would break the chain the matcher is looking for.

**Master weights.** A parameter keeps its id, its `DType::Float32` and its
`Lifetime::Parameter`; so does its gradient, and the optimizer is untouched.
The fp16 a GEMM reads is a *derived* value with an activation lifetime,
produced by a cast and rewritten every replay because the parameter it reads
changed. The rule the pass applies is one sentence: a value's element type
follows its producer's computed type unless something outside the pass depends
on it, and "outside" means a persistent lifetime class or a value the caller
named as observed. The backward direction falls out of it: a gradient value is
`Lifetime::Gradient`, so it is pinned, so an fp16 matmul writes a fresh narrow
result and a cast widens it into the buffer the optimizer accumulates in.

**What it is worth here: correctness, and nothing else.** This machine has no
fp16 matrix unit, so half arithmetic is emulated and `--precision mixed` is
several times *slower* than fp32. No timing gate exists for it and none should
until the hardware does. What is real and tested is that the graph carries the
right types, that the masters stay fp32, and that the model still learns: see
[README.md § Mixed precision](../guides/DEMO.md#mixed-precision-what---precision-mixed-does-and-what-it-is-worth-here)
for the measured loss curves both ways.

**Loss scaling is here too, and it is `optim/loss_scaler.hpp`.** The backward
seed is multiplied by a device-resident power of two (`Ops::K_DYNAMIC_SCALE`,
emitted by `differentiate()`'s seed rule so it lands in topological position
and before anything narrows); SushiBLAS's `count_nonfinite` reduction screens every
gradient inside the same compiled plan; the fused update divides the scale back
out before the decay and the moments, and returns without touching anything
when its parameter's screen came back non-zero. The scale halves on overflow
and doubles after N clean steps, and it is device-resident so that neither
move recompiles the step.

Two things about it are worth knowing before relying on it. **On this workload
it buys nothing**: fp16 without it neither stalls nor diverges here, so it is
justified by theory and by larger models. And **the skip is per parameter**:
the ordering that makes an update wait for its own screen is the
write-after-read hazard on that gradient, and there is no such hazard between
parameter A's update and parameter B's screen, so a step where only some
gradients overflowed is a partial update rather than an empty one. Nothing
unsafe results (no non-finite value reaches a parameter or a moment) and the
header states what a whole-step skip would cost.
