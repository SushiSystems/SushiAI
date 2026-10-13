# Plan-packed scratch: report

**Status:** Done 2026-10-13. A full-size training step of YOLOv4 and of YOLOv7 runs under the
default memory limit. Five full-size cases still fail, on a comparison with the Darknet dump
and not on memory; their bounds wait for the owner's word.

## Task 1: the share of scratch

Measured 2026-10-13 with one step of YOLOv4-tiny under the default memory limit of `sa`:
`Integration_DarknetYolov4TinyTraining.TheFirstConvolutionReceivesAGradient`, which passed.

| Pool | Total | Arenas | Own values | Ones | Scratch | Share of scratch |
| --- | --- | --- | --- | --- | --- | --- |
| Probe | 128.2 MB | 75.8 MB | 5.6 MB | 0.004 MB | 46.7 MB | 36.5% |
| Training | 581.7 MB | 409.4 MB | 30.4 MB | 1.7 MB | 140.2 MB | 24.1% |

Each pool has one arena. The probe pool's holds 194 buffers that would take 498.4 MB unpacked;
the training pool's holds 319 buffers that would take 1008.6 MB unpacked.

Scratch is above a tenth of the total in both pools, so the stop condition of SPEC section 7.1
does not apply.

## The plans of the full-size graphs

Measured 2026-10-13 on the host: `ThePlansOfTheStepOverlapLiveRanges` traces both graphs of a
model, builds their memory plans and prints them. Nothing is allocated on the device, so the
full-size models run under the default limit. All three cases passed.

| Model | Graph | Nodes | Values | Arenas | Unpacked | Own values |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv4-tiny | Probe | 198 | 260 | 75.8 MB | 500.1 MB | 28.7 MB |
| YOLOv4-tiny | Training | 395 | 495 | 409.4 MB | 1008.6 MB | 55.1 MB |
| YOLOv4 | Probe | 1164 | 1491 | 719.0 MB | 8023.3 MB | 268.7 MB |
| YOLOv4 | Training | 2314 | 2855 | 5436.7 MB | 16723.9 MB | 544.0 MB |
| YOLOv7 | Probe | 850 | 1128 | 640.6 MB | 5825.9 MB | 166.6 MB |
| YOLOv7 | Training | 1644 | 2100 | 4420.2 MB | 11886.3 MB | 340.3 MB |

The arenas of YOLOv4-tiny equal the ones the device run allocated, so the host figures are the
ones a pool would allocate. "Own values" counts every unpacked value of the plan, parameters
included; a pool allocates fewer of them because the model state owns the parameters.

What this settles:

- **The node limit is not the cause.** The largest graph has 2314 nodes against a limit of 8192,
  and every plan overlaps live ranges.
- **The arenas of a YOLOv4 step are 6.2 GB**, 5.4 GB for training and 0.7 GB for the probe.
  A YOLOv7 step asks for 5.1 GB.

## An estimate of the full-size scratch

Scratch is not part of a plan, so the host run does not report it. On YOLOv4-tiny scratch is
0.34 of the training arena and 0.62 of the probe arena. If YOLOv4 keeps those ratios, its scratch
is about 1.9 GB and 0.4 GB, 2.3 GB together. This is an estimate, not a measurement.

With it, a YOLOv4 step is about 6.2 GB of arenas, 2.3 GB of scratch and up to 0.8 GB of own
values, 9.3 GB before the weights and the optimizer's state. The measured commit was 11.1 GB.

Packing the scratch would bring that estimate to about 7 GB plus whatever the scratch still
needs inside the arena. The default limit is 7.6 GB. The work is needed for the 12 failing
cases, and by this estimate it is not certain to be enough alone. The probe pool, about 1.2 GB
with its scratch, is the next largest part; whether a step can do without it is not examined.

## What is live at the peak

Measured 2026-10-13 on the host by the same three cases. A packed buffer is taken as live from
the first node that writes it to the last node that reads or writes it, with the nodes in
insertion order. No arena that follows that order can be smaller than the peak.

| Model | Graph | Arena | Peak in insertion order | Peak at node |
| --- | --- | --- | --- | --- |
| YOLOv4-tiny | Probe | 75.8 MB | 53.0 MB | 21 of 198 |
| YOLOv4-tiny | Training | 409.4 MB | 379.0 MB | 229 of 395 |
| YOLOv4 | Probe | 719.0 MB | 338.5 MB | 59 of 1164 |
| YOLOv4 | Training | 5436.7 MB | 5262.6 MB | 1194 of 2314 |
| YOLOv7 | Probe | 640.6 MB | 500.1 MB | 21 of 850 |
| YOLOv7 | Training | 4420.2 MB | 4268.8 MB | 880 of 1644 |

The training arenas of the two full-size models are within 4% of the peak, so a better packing
has little
left to win there. The probe arena of YOLOv4 is about twice its peak.

The bytes live at the peak of a training graph, by the operation that wrote them:

| Model | `im2col` | `reshape` | `batchnorm2d` | Activation | Rest |
| --- | --- | --- | --- | --- | --- |
| YOLOv4-tiny | 225.8 MB | 80.4 MB | 0.01 MB | 45.4 MB | 27.4 MB |
| YOLOv4 | 1897.4 MB | 1749.2 MB | 723.5 MB | 858.1 MB | 34.4 MB |
| YOLOv7 | 2250.0 MB | 1286.0 MB | 646.2 MB | 0 MB | 86.6 MB |

Two things follow from reading the code beside these figures:

- **Every column matrix is kept from the forward pass to the backward pass.** In YOLOv4 that is
  1.9 GB of a 5.3 GB peak, in YOLOv7 2.25 GB of 4.3 GB. `rematerialise_im2col` recomputes the
  columns before the backward pass instead; the eager front end calls it
  (`modules/core/eager/source/region_program.cpp`) and the compiled training path does not.
- **`reshape` copies.** `GraphBuilder::reshape` declares a new value and `lower_reshape` records
  a copy, so a convolution keeps a second copy of its input for a 1x1 kernel and writes its
  result twice. How much of the 1.7 GB of YOLOv4 an alias would give back is not measured.

## Rematerialised columns and Scratch planned by node order

Done 2026-10-13 on the owner's word. `GraphOptions::rematerialise`, on by default, runs
`rematerialise_im2col` over a step's graph after the fusion pass. The pass alone made the
arenas larger: nothing in the graph orders one recomputed column matrix after another, so the
planner gave each its own bytes. The planner now overlaps Scratch buffers by the order of the
nodes and plans Activation buffers as before (`modules/core/graph/README.md`, Memory plan).

Arenas of the training graph, measured on the host:

| Model | Before | Pass alone | Pass and node order | Peak in insertion order |
| --- | --- | --- | --- | --- |
| YOLOv4-tiny | 409.4 MB | 439.1 MB | 240.1 MB | 180.3 MB |
| YOLOv4 | 5436.7 MB | 5823.6 MB | 4040.5 MB | 3665.4 MB |
| YOLOv7 | 4420.2 MB | 4899.5 MB | 3001.1 MB | 2417.8 MB |

The training pool of YOLOv4-tiny on the device fell from 581.7 MB to 412.3 MB. The nine
training cases of YOLOv4-tiny passed with the bounds they had, and so did the 27 unit tests of
the plan and of the pass.

A temporary measurement, not committed, merged each `reshape` result with the buffer it reads
and took the peak again: 3257.2 MB for YOLOv4, 2180.9 MB for YOLOv7 and 159.2 MB for
YOLOv4-tiny. An alias would take about 0.4 GB off the peak of YOLOv4.

With the arenas at 4.0 GB and 0.7 GB, the estimate of a YOLOv4 step is about 7.9 GB before
the weights, against 9.3 GB before. The full-size training cases were not run.

## Scratch rules and borrowed arenas

Done 2026-10-13 on the owner's word to finish both.

**Scratch rules.** `LoweringEntry` has a third member, a `ScratchRule` that appends the buffers
a row writes on every run before it reads them. `scratch_of` collects the requests of a graph,
`MemoryPlan::build` takes them as a third argument and places each in the Scratch arena with
its node as the one user, and `TensorPool::scratch` returns the arena view.

This leaves the SPEC in three places:

- `LoweringContext::scratch(slot, shape, dtype)` keeps its signature and its 39 call sites.
  The pool stops the process when a row asks for other bytes than its rule stated, so the two
  cannot disagree without notice.
- There is no `Held` kind. A buffer no rule states keeps a `Storage` of its own, which is the
  old behaviour and what a host-filled buffer needs.
- Only `BATCHNORM2D` and `BATCHNORM2D_BACKWARD` have a rule, ten buffers of the 39. On
  YOLOv4-tiny the scratch outside the arenas was 140.2 MB and 46.7 MB, which is three times
  and once the bytes of the batch norm results, and it is 0 MB with the two rules. The other
  rows are listed in `REMAINING_WORK.md`.

**Borrowed arenas.** `TensorPool::adopt` takes a lender and `ArenaSet::make` uses the lender's
arena of the same lifetime where it is large enough. `AnchorTrainingStep::make` lends the
replay's pool to the forward-only pass. The outputs of that pass are pinned and keep their own
bytes, so they are still read after the replay.

Pools after one step, measured on the device under the default limit:

| Model | Probe pool | Training pool | Both | Before any change | Darknet, whole process |
| --- | --- | --- | --- | --- | --- |
| YOLOv4-tiny | 5.6 MB | 272.1 MB | 277.7 MB | 709.9 MB | 378 MB |
| YOLOv4 | 23.2 MB | 4339.0 MB | 4362.2 MB | not run | 5950 MB |
| YOLOv7 | 25.7 MB | 3200.6 MB | 3226.3 MB | not run | 4940 MB |

**The plan changes no value.** `ThePlanOfTheStepChangesNoValue` runs one step with its pools
planned and once more with an empty plan, where every value has a `Storage` of its own, and
compares each head, delta and gradient bit for bit. It passes for YOLOv3-tiny, YOLOv4-tiny and
YOLOv7-tiny. A full-size model does not fit unplanned, so it has no such case.

## The full-size suites under the default limit

Run 2026-10-13: `Integration_DarknetYolov4Training.*` and `Integration_DarknetYolov7Training.*`,
20 cases, memory limit 7.6 GB. No case returned `OutOfMemory`. 15 passed, the two overfit cases
among them (694 s and 466 s for 200 steps). Five failed on a comparison with the dump:

| Case | Model | Worst difference |
| --- | --- | --- |
| Heads under batch statistics | YOLOv4 | -40.2478 against -40.4527; 1450412 of 2945760 values of head 0 outside 1e-3 + 1e-4 relative |
| Deltas of the step | YOLOv4 | 0.0043 on one entry of head 2; 7 entries outside their bounds |
| Updates of the head convolutions | YOLOv4 | 2.00339 against 1.98274 in `layer_160_weight_updates`; 101 values outside |
| Heads under batch statistics | YOLOv7 | 0.69399 against 0.68877; 260 of 4284000 head values outside |
| Deltas of the step | YOLOv7 | 0.0010 on one entry of head 1 |

`test_darknet_yolov4_training.cpp` says of its step bounds that they are starting values no
run has measured, and these runs are the first. The bit-for-bit case above rules the memory
plan out as the cause for the tiny models; for the full-size ones nothing compares a planned
step with an unplanned one. The bounds were not changed.

## What Darknet allocates for the same step

Measured 2026-10-13 outside the repository: the CPU build of the oracle ran one training step
of the same cfg, batch and input, and its peak private bytes were read while it ran.

| Model | Batch and input | Darknet, peak private bytes | SushiAI |
| --- | --- | --- | --- |
| YOLOv4-tiny | 2 x 416 x 416 | 378 MB | 710 MB in the two pools |
| YOLOv4 | 2 x 608 x 608 | 5950 MB | 11.1 GB committed |
| YOLOv7 | 2 x 640 x 640 | 4940 MB | not measured |

The Darknet figure is a whole process on the CPU, weights included. The SushiAI figure of
YOLOv4-tiny leaves out the weights and the process itself, so the two columns are not the same
quantity; they show that a step here takes close to twice what the reference takes.

## Where the memory is counted

Every tensor is allocated with `Residency::Shared` (`modules/core/core/source/allocation.cpp`),
which SushiRuntime turns into `sycl::malloc_shared`. The memory limit of `sa` is a limit on the
committed bytes of the process, and the arenas count against it. Whether `Residency::Device`
would count against it too is not measured.

## Not measured

- What the remaining 1.8 GB of the 11.1 GB commit is.
- What the recomputed column matrices cost in time; the GPU was power limited on the day.
- Whether the probe pool can share the bytes of the training arena.

## Build

The first build of task 1's code failed: `tests/vision_support/darknet_training.cpp` added a
`std::string_view` to a string literal. The two lines now build a `std::string` first. The
build after that passed.
