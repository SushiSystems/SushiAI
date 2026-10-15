# The demo

`sa demo mlp` trains a small classifier and prints what the compiled step did. This page shows its output and what each number means. The flags are in the [CLI guide](CLI_GUIDE.md).


```bash
sa demo mlp                                   # the defaults below
sa demo mlp --epochs 4 --optimizer sgd        # anything you like
sa demo mlp --help                            # the full flag list
```

It trains on MNIST when the training split's two IDX files are under
`--mnist-dir` (`data/mnist` by default) and on a seeded synthetic
classification problem otherwise, and says which it got, in both the header
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

`1.791759` is `ln 6` to the last digit: the exact uniform-prior loss a
zero-initialised classifier must start at. **`compiles: 1`** is the claim the
whole design rests on: 128 training steps, one lowering, one compile. The
CTest entry `Medium_Demo.MLPRunsAndReportsOneCompile` and
`cli/tests/test_demo.py` both pin that line, so the demo cannot rot quietly.

**`ops per step: 25`** is 21 graph nodes plus one fused update per parameter:
the optimizer is *inside* that single compiled plan
([ARCHITECTURE.md §5.1](../architecture/LOWERING.md#51-the-optimizer-is-part-of-the-compiled-step)),
and the graph itself has been through the fusion pass
([§5.2](../architecture/FUSION_AND_PRECISION.md#52-fusion-is-a-selector-over-a-catalog-not-a-code-generator)).
Spelled the old way (twelve separate optimizer calls per parameter flushed in
a graph of their own, and no fusion), the same run issues 81 device tasks per
step where this issues 25.

The **`Fusion`** line is the pass reporting itself, refusals included. Run
`--fusion off` to get the unfused graph back, or `--no-fuse gemm+bias+relu` to
switch one catalog entry off; every spelling produces the same numbers, which
is what `Medium_Fusion` pins byte for byte.

## What fusion is actually worth here, including the part that costs

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
stopped being launched**: these tensors are small enough that a step is mostly
launch overhead, which is exactly the regime where removing launches pays one
for one.

**And it costs about 0.9 s once, on the first step.** Every fused form is a
distinct C++ template instantiation and therefore a distinct SYCL kernel, which
this OpenCL backend compiles at first launch. A single-step run costs 0.97 s
unfused and 1.88 s fused, which is the whole of the difference; it does not
scale with the model, the batch or the step count. So **break-even on this
machine is around 5 000 steps**, and the 128-step run printed above is far
below it: fusion makes the default demo about 1.9x *slower* in wall clock
while issuing 29% fewer kernels. That is not a contradiction and it is not
hidden: it is one-time device-code compilation, it is what AOT targeting
(`sa build -D SB_SYCL_TARGETS=spir64`) exists to move to build time, and it is
irrelevant to any training run long enough to be worth doing. The launch count
is the number this change controls; the wall clock is reported both ways
because reporting only the favourable half would be dishonest.

The **`Arenas`** line now says `0% saved`, and that is fusion's doing rather
than the planner failing: a fused subtree's intermediates stop being values at
all, so the buffers the planner would overlap do not exist.
The whole pool is 96.6 KiB against 130.9 KiB before either pass existed.
`Medium_StaticMemory` pins both halves: packing still shrinks the
*unfused* graph, and the fused graph is never larger than the packed unfused
one.

## Mixed precision: what `--precision mixed` does, and what it is worth here

`--precision mixed` runs the matrix products, and the bias and activation a
fused GEMM folds into them, in **fp16**, keeping fp32 master weights, fp32
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
actually meaningful (epoch 1's last step, `0.037844` against `0.037880`):
the relative deviation is 9.5e-4. The demo prints six decimals, so by epoch 8
the losses are down at `0.000331` against `0.000332` and the apparent 3e-3
there is dominated by the printing rather than by the arithmetic; the
per-step comparison that is not (`Medium_MixedPrecisionRegressions`) reads the
doubles directly. That is a *measurement of this workload*, not a general
claim: fp16 training without loss scaling is
expected to lose small gradients to underflow, and on this small MLP it
demonstrably does not. Repeating it with `--hidden 512 --batch-size 64` (a
512-long fp16 reduction inside each product) gives the same answer: both
arms reach 0.000003 after 48 steps. So the loss scaling that has since landed
is justified by theory and by larger models, and **not** by anything this
workload does, turning it on here makes the demo no better, and saying so is
worth more than a claim that it does.

What `--loss-scale` *is* good for on this box is watching the machinery work.
`sa demo mlp --epochs 1 --batches 4 --precision fp32 --loss-scale 65536` gives
`loss 1.791759 -> 1.133352`, bit-identical to the unscaled run, which is the
whole point of a dyadic scale: the round trip through 65536 and back is exact.
`--precision mixed --loss-scale 65536` reports one overflowed step and a
backoff to 32768: PyTorch's own default scale is too high for this model's
fp16 backward pass, and the backoff finds that out in one step. Pushing it to
`--loss-scale 16777216` overflows on all 8 steps and walks the scale from
2^24 down to 2^16 without recompiling: `compiles: 1` throughout.

One honest caveat: the skip is **per parameter**, not per step. A parameter
whose own gradient stayed finite still updates on an overflowed step, so the
report says "overflowed steps" rather than "skipped". No infinity or NaN
reaches a parameter or a moment either way: that is asserted bit for bit in
`Medium_LossScaling`, but it is a partial step where PyTorch would take
none. `optim/loss_scaler.hpp` explains what a whole-step skip would cost.

`gradcheck` and mixed precision are incompatible, and deliberately so:
`autograd/gradcheck.hpp` requires FLOAT64 because a central difference in
fp16 has a noise floor about a thousand times *larger* than the discrepancy a
wrong backward rule produces. A gradient rule is a statement about the
mathematics, so checking it once in fp64 checks it for every precision; what
mixed precision can get wrong is the plumbing, and that is checked by walking
the graph (`Graph::find_dtype_mismatches`) rather than by a finite difference.
