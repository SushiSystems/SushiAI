# Integration tests

The integration tier runs on a real `SushiBLAS::Engine`. Each file here checks a claim that
needs a device to answer; the host-only half of the same claim lives in
`tests/functional/unit/`. This document holds the reasoning behind the tests: why a case is
shaped the way it is, which wrong implementation it is built to catch, and the numbers worked
out by hand. The source keeps one line per case and cites this file.

## The SaTest alias

`test_autograd_numeric.cpp`, `test_forward.cpp`, `test_resilience.cpp` and `test_training.cpp`
declare `namespace SaTest = SushiAI::Test`. Inside a `TEST_F` body the fixture's base class
injects the name `Test`, so `Test` alone resolves to `::testing::Test`. The alias keeps the
helpers in `SushiAI::Test` reachable from a test body.

## test_arena.cpp

The memory planner rests on three device-side claims, and the file checks each one.

1. Two values packed into one arena report non-overlapping `ResourceRegion`s over the same
   allocation base. Packing is safe only because of this. SushiRuntime's dependency tracker
   orders two accesses if and only if their regions overlap. Every driver once passed a region
   that named the bare allocation base; with that region every tensor in an arena would alias
   every other one, and the tracker would serialise them all.
2. A view addresses its own elements. `storage_offset` counts elements, so an offset that is
   not a whole number of them is refused. Truncating it would give a tensor that starts in the
   wrong place without an error.
3. Mixed dtypes in one arena each start aligned for their own element width. This is the one
   part of the offset arithmetic `Tensor` cannot do for the planner, because the offset is
   already chosen by the time a `Tensor` exists.

The file is laid out in four groups: region precision, element-counted offsets, mixed dtypes in
one arena, and the pool's side of the contract.

### OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther

This is the functional half of the region claim. Real SushiBLAS work runs on two views of one
arena in one step, and each result must land only in its own bytes.

### AnOffsetThatIsNotAWholeNumberOfElementsIsRefused

An offset of four bytes into a float64 view can be written in bytes but not in the element
count `Tensor` stores. Truncating it would put the tensor somewhere the planner never reserved.
The same offset is valid for float32.

### MixedDtypesEachStartAlignedForTheirOwnElementWidth

The plan's alignment is set to 1 so that only the element-width floor holds the two values
apart. With the default alignment the property holds for an uninteresting reason; at 1 it
holds only because the planner rounds.

The round trip through each view proves the addresses are usable as well as aligned. A float64
read through a mis-rounded start would come back as two halves of neighbouring values, not as
the number written.

### TheEmptyPlanPacksNothing

The empty plan is the control an A/B measurement of the whole pass needs. It is also what a
caller uses to get unpacked allocation.

### APlannedPoolAllocatesOneStorageForAWholeArena

The test compares no byte counts. Its tensors are 64 bytes each and the arena aligns every slot
to 128, so packing them costs memory. The saving belongs to realistic tensor sizes and is
measured where that holds, in `Regression_StaticMemory`.

## test_autograd_numeric.cpp

The whole graph (forward, loss and backward) runs on the device in double precision. The
analytic gradient it produces is compared against a central difference of the loss the same
graph computes. No other check separates a correct backward rule from a plausible one. It has
to be fp64: an fp32 difference quotient bottoms out near 2.4e-5 relative accuracy, the same
order as the error a wrong rule produces. See `gradcheck.hpp`.

### Probe

The variable under test is declared as the only gradient-requiring leaf. Everything else the
build callback declares is a constant the test fills. The loss is always the mean negative
log-likelihood against a target matrix, which is a valid scalar functional of anything shaped
[N, C]. When the model ends in a log-softmax it exercises the fused cross-entropy rule as
well.

Fusion is off by default, because almost every case checks a backward rule and fusion must not
change one. It can be switched on because one case checks the opposite direction: a fused form
that rewrites what the backward pass reads must leave the same derivative behind. Value ids
survive the fusion pass, so nothing the probe recorded before it needs remapping.

### MatmulGradientOfTheRightOperand

The same node is differentiated from the other side. The variable is the weight, so the case
covers the dB half of the rule where the earlier matmul cases cover dA.

### ElementwiseMultiplyFromTheRight

The case checks dB = dC * A, the half the left-hand check cannot reach. The rule is written
symmetrically, so a copy-paste slip that reads `inputs[1]` on both sides would leave the
left-hand test green.

### AccumulateInTheForwardGraph

ACCUMULATE has a backward rule that nothing else exercises. The two fan-out tests build an
ACCUMULATE chain in the backward graph, and `differentiate()` walks only the forward nodes, so
those chains are never differentiated themselves. Reaching the rule needs an ACCUMULATE the
caller put in the forward pass.

The lowering aliases the result onto the accumulator's own buffer, so running the step leaves
the variable holding `x + other`. Every probe rewrites the whole variable first, which makes
that harmless.

### ReluAwayFromItsKink

ReLU is not differentiable at zero. A difference quotient that straddles the kink reports the
average of the two one-sided slopes, which no correct rule can match, so the point is sampled
clear of zero.

### LeakyReluHasBothOfItsSlopes

The point is sampled off the kink and over a range wide enough that elements land on each
side. A probe that only saw positive inputs would pass with the negative slope hard-coded to
zero, which is a plain ReLU. `LeakyReluBranchSwapIsCaughtByProbePoints` counts the negative
points the sample contains.

### LeakyReluBranchSwapIsCaughtByProbePoints

The test pins the negative-side split of the exact point `LeakyReluHasBothOfItsSlopes` uses:
same seed, same scale, same `push_off_kinks`. The branch-swap mutation depends on that split.
If a change to the seed or the noise generator collapsed the sample to all-positive, the
mutation would go undetected, and this test fails first.

### AFusedGeluEpilogueHasTheSameDerivativeAsAnUnfusedOne

The central difference is the strongest oracle in the repository. Here it is pointed at the
one fused form whose correctness does not follow from running the same operations in the same
order. GEMM_BIAS_GELU rewrites what the backward pass reads: the pre-activation arrives from a
store inside a GEMM kernel, where the unfused graph had a `ger` write it. A central difference
does not care where the value came from. It asks the graph what its own loss does when the
variable moves and compares that with the gradient the graph reports. A spill holding the
wrong tensor cannot pass that check, though it would pass any number of runs where the model
trained and the loss fell.

The case is FLOAT64 throughout, as `gradcheck.hpp` requires. `execute_gemm`'s REAL domain
instantiates the whole epilogue for double.

The test only counts if the fused arm ran, and both halves of that are asserted. The static
pass must have matched, and the selector must not have handed the lowering the unfused
spelling. On a device without double precision the selector would do that, and the case would
measure the unfused path without saying so.

### FusedCrossEntropyAgainstOneHotTargets

The gradient of the mean NLL with respect to the logits is (softmax - Y) / N. It is emitted as
one fused rule and not by differentiating the seven-op log-softmax recipe, so it needs its own
numeric proof.

### WholeTwoLayerModelUnderRandomDirections

This is the end-to-end check. A full Jacobian over every weight would take 2n device runs.
Random unit directions take 2k runs whatever n is, and the chance that a wrong gradient agrees
along all sixteen of them is negligible.

### ReshapeMovesItsGradientBackUnchanged

A reshape moves no data, so its gradient is its incoming gradient in the original shape. The
numeric check is still worth having: reshaping back to the input shape and reshaping back to
the output shape are both one line, and only one is right.

The fused LOG_SOFTMAX + NLL_LOSS rule's closed form assumes a one-hot target row
(`nll_loss_backward` says it is "valid because the targets are one-hot"). Noise in the targets
would make the analytic gradient disagree with the numeric one for reasons that have nothing
to do with RESHAPE.

### LayoutConvertsGradientIsThePermutedGradient

The case uses a single `layout_convert`, so the loss depends on the permuted element order. A
round trip would not do. A rule that reshaped dy back to the operand's shape without permuting
it has the right magnitude in the wrong place, and it still satisfies every shape check
`Differentiator::contribute` makes. A round trip cannot tell the two apart, because any
involutive, shape-correct pair of backward hops composes to the identity.

CV-4 forces every node that is neither a LEAF nor a LAYOUT_CONVERT to agree with its operand
on layout. A variable therefore cannot reach `log_softmax` by converting one way and stopping:
`graph::finalise()` would refuse the mismatched `log_softmax` output before any gradient ran.
CV-4 exempts a leaf, which has no operands to disagree with, so its layout tag is free. This
is also how the operation is used in practice. LAYOUT_CONVERT exists for boundary interop,
such as loading an NCHW checkpoint, where the source data carries the other tag and no
mid-graph value does. The case tags the variable NCHW before its one LAYOUT_CONVERT node and
reshapes straight into `log_softmax`. That reaches a CV-4-legal NHWC consumer with no round
trip, so the permutation is what the loss depends on.

### LayoutConvertRoundTripComposesBothDirections

The two directions composing to the identity is fair to check. It cannot tell a correct
permutation from a same-shape non-permutation, for the reason given under the single-hop case,
so it is a secondary check and not the proof.

### AWholeConvolutionDifferentiatesThroughIm2colAndMatmul

The case checks CV-1's claim numerically: with no CONV2D identity, a convolution's gradient is
`im2col_backward` composed with the matmul rule that already existed. If this passes,
convolution needs no rule of its own, which is why it has none.

The graph is single-hop by design (A-7): `im2col` feeds straight into a matmul and never into
its own `col2im`. A round trip through `col2im` would make the forward composite the identity,
and then any involutive, shape-correct backward, a plain reshape included, would pass for the
wrong reason.

When this case was written (Task 12) `spatial_mean` had no gradient rule, so [N, C] is
reached straight off the matmul. With a 1x1 window grid folded into rows there would be
nothing to average over, and the matmul's output is already rank 2, [N*H_o*W_o, C_out], so no
reshape is needed.

A probe ending in `log_softmax` must use one-hot targets (A-6). The fused NLL_LOSS rule's
closed form is the true derivative of this loss only when the target row is one-hot.

### Im2colsGradientAccumulatesEveryWindowThatReadAPixel

In a stride-1 padded 3x3 convolution nine windows read every interior pixel, so its gradient
is a sum of nine terms. A `col2im` that overwrote where it should accumulate would give one of
them: a gradient of the right shape and sign, at roughly a ninth of the right size.

### AnAsymmetricallyPaddedConvolutionDifferentiatesOverItsRealWindowGrid

This is the gradient over the geometry a symmetric re-derivation gets wrong without an error.
A [1, 4, 5, 1] input with a 3x3 window, `pad_top` 1 and `pad_left` 1, padded 2 below and 0 on
the right, has a 5x4 window grid. Deriving the grid from the geometry alone gives 4x5, the
transpose, with the same 20 windows, so nothing throws. The forward lowering and
`im2col_backward` both read the real grid out of the node's attribute slots 8 and 9, and the
case checks the whole path against finite differences.

### Col2imsGradientGathersEveryWindowThatWroteAPixel

COL2IM's rule is the mirror image of `im2col_backward`. The case checks it without `im2col`:
the variable is the column matrix, fed straight into `col2im` and reshaped into `log_softmax`.
That keeps it single-hop by construction (A-7). No `im2col` sits on the path for `col2im` to
be the adjoint of, so an involutive round trip cannot satisfy the check.

### MaxPoolRoutesEachGradientToTheElementThatWon

The point is sampled with `push_off_kinks` so that no two elements in a window tie. At a tie
the max is not differentiable, and the numeric check would be comparing against a one-sided
derivative.

### OverlappingMaxPoolWindowsAccumulateOnADominantPixel

At stride 1 a pixel that wins several windows receives all their gradients. This is where
keeping the pooling backward functions separate pays: a shared `pool_backward` would have had
to branch on mode inside this loop.

### AvgPoolSpreadsItsGradientOverTheWholeWindow

The brief's [2, 6, 6, 3] variable has 216 elements, over the 200-input full-Jacobian ceiling
of `expect_gradient_matches`. The case uses one batch row so that the check stays in the mode
it asserts on.

### UpsampleSumsEachSourcePixelsReplicas

Scale 2 replicates every pixel four times, so its gradient is the sum of four. Forgetting the
sum gives a gradient a quarter of the right size. A model with that fault still trains, more
slowly, and only in this one layer.

### SpatialMeanBroadcastsScaledByOneOverHW

An earlier draft of the design lacked this rule. It is a per-(n, c) broadcast. BIAS_ADD's
broadcast is per row and CHANNEL_AFFINE's is per channel, and neither would have produced it.

### ConcatGivesEachOperandItsOwnChannelRange

The gradient of a concat is a slice per operand, at that operand's offset. An off-by-one
offset gives every operand a gradient of the right shape that belongs to its neighbour. A
round trip through slice-then-concat cannot detect that, so the operands are concatenated in a
different order from the one they were sliced out in.

### SliceScattersItsGradientBackIntoZeros

Channels outside the slice receive exactly zero, not stale memory. The reordered concat makes
a wrong offset visible; this case makes a missing zero-fill visible.

### BatchNormsGradientAccountsForTheStatisticsItComputed

The batch mean and variance are functions of x, so dx has three terms. Treating the statistics
as constants gives a gradient that is right on average over the batch and wrong per element.
The model still trains, slightly worse, and nothing points at the cause.

### BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput

The same node is differentiated at gamma. It is a separate case because the `Probe` checks one
variable. dgamma is the term that would go unnoticed if it were summed over the wrong axis,
since the result is a [C] value of the right shape either way.

### ChannelAffinesScaleGradientSumsOverNHAndW

dgamma is a sum over three axes at once. A sum over the wrong ones, per sample where it should
be per channel, is still a [C]-shaped value when N equals C. That is why the case is checked
numerically and not by shape.

## test_checkpoint.cpp

The file covers the device half of checkpointing: binding records to the pool's tensors. The
byte format's malformed-file surface needs no device and lives in
`tests/functional/unit/test_checkpoint_format.cpp`.

The last equivalence test is the point of the file. A `Sequential` and a `LayerStack` built
from the equivalent `model.json` are two spellings of one model. The proof that they are one
model is a checkpoint written by either loading into the other. A separate mechanism would
leave its own correctness as the open question.

### Materialised

Parameters are leaves, so one `materialise_leaves` call allocates every one of them. It also
allocates the input leaf, which costs a [4, 6] buffer and spares the fixture from knowing
which leaves are which.

### ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack

Two spellings of one model prove they describe the same thing by exchanging a checkpoint. The
compile-time and run-time paths then need no separate mechanism to keep them in step, and they
cannot drift without this test failing.

The names are asserted equal first. The checkpoint is matched by name, so a naming difference
would otherwise surface as "does not supply" and hide the clearer statement that the two
models disagree about what they hold.

The return trip, `LayerStack` to `Sequential`, loads into a third model. That makes it a real
load and not a comparison against values that were already there.

### AShapeMismatchThrowsWithBothShapes

The two models have the same parameter names and different widths. An element-count comparison
would catch this pair but would miss a transpose, so the loader compares shapes.

### ADuplicatedNameThrowsAndNamesTheDuplicate

The file is built by hand because `save_checkpoint` cannot produce one: the walk yields each
name once. A counter-based loader would accept this file and leave a parameter reading
uninitialised device memory.

## test_engine_smoke.cpp

### AxpyThroughSushiBLAS

This is the P6 "consumer-shaped example" for AI-0. It proves that the two-sibling D3 dependency
chain, SushiRuntime and SushiBLAS both linked through SushiAI, compiles, links and runs a real
operation end to end. It does not stand in for the autograd operation coverage of AI-1.

## test_forward.cpp

Every lowered operation is checked against values worked out by hand. This tier catches a
transposed gemm, a bias broadcast down the wrong axis, and a log-softmax whose stabilising
shift did not cancel. All three produce plausible numbers and no error.

### LayoutConvertPermutesChannelsToTheFront

In NCHW the channels are outermost of the spatial block. For each c the two W positions follow
in order: (c=0, w=0..1) = 0, 3; (c=1, w=0..1) = 1, 4; (c=2, w=0..1) = 2, 5.

### ActivationsMatchTheirDefinitionsAndLeaveTheInputAlone

Out-of-place semantics matter. The backward rules for relu and gelu read the forward input, so
an in-place activation would hand them the output.

The leaky_relu values are computed by hand at slope 0.1, with inputs on both sides of zero so
that both branches run: -2 gives -0.2, -0.5 gives -0.05, 0.5 gives 0.5, 2 gives 2.

### LogSoftmaxRowsSumToOneAfterExponentiating

The second row is the first offset by a large constant. Log-softmax must be invariant to the
offset, and the max shift is what makes it so.

### ZeroFillsItsBufferExactlyOnce

Replaying must not accumulate. The bias broadcast writes into its result and does not add, and
the zero buffer is not re-filled per step.

### Im2colGathersOneWindowInTapOrderChannelInnermost

A gradcheck says the backward is consistent with whatever the forward computes. It says
nothing about whether the forward computes the right thing. This case is the forward's own
proof, worked out by hand.

N=1, H=2, W=2, C=1 with a 2x2 kernel and no padding covers the whole image in one window. The
column matrix's single row is the input read in tap order (r, s) with the channel innermost,
which for one channel is row-major (h, w).

### Col2imSumsEveryWindowThatOverlapsAPixel

`col2im` claims, and `im2col_backward` relies on it, that a pixel covered by several windows
receives the sum of their contributions and not the last one written. N=1, H=3, W=3, C=1 with a
2x2 kernel and no padding has four windows, with origins (0,0), (0,1), (1,0) and (1,1). The
corner pixels sit under one window, the edge pixels under two, and the centre pixel under all
four. Feeding every column entry a 1 turns the expected image into that overlap count:

```
1 2 1
2 4 2
1 2 1
```

### AnAsymmetricallyPaddedIm2colUsesTheExtentsItWasGivenNotDerivedOnes

A symmetric re-derivation gets this case wrong without an error. A [1, 4, 5, 1] input with a
3x3 window, stride 1, `pad_top` 1 and `pad_left` 1, padded 2 below and 0 on the right, has

```
out_h = (4 + 1 + 2 - 3) + 1 = 5     out_w = (5 + 1 + 0 - 3) + 1 = 4
```

Re-deriving the extents from the geometry alone assumes symmetric padding and yields 4x5, the
transpose of the real window grid. Both grids have 20 windows, so the row-count check agrees
and both sets of window origins stay inside the padded input. Nothing throws; the numbers are
wrong. That is why the extents travel in the node's attributes and are not recomputed.

The input is 0..19 in row-major (h, w) order. A window with origin (oh - 1, ow - 1) reads the
pixels h*5 + w it covers and a zero wherever it hangs off the image. Row oh*4 + ow of the
result is that window in tap order. By that rule the last row (oh = 4, ow = 3, origin (3, 2))
covers only the input's bottom-right three pixels, 17 18 19, and is padding everywhere else.
The transposed grid would put 13 14 0 18 19 0 0 0 0 there.

### MaxPool2dTakesEachWindowsMaximum

This is the forward's own proof, for the same reason as the im2col case. N=1, H=2, W=2, C=1
holds the values 0, 1, 2, 3 in row-major (h, w) order. One 2x2 window covers the whole image,
and the maximum is 3.

### ALayoutConvertRoundTripActuallyPermutesAndComesBack

This case closes the gap that let a dropped layout tag hide. Every other layout test stops at
the tag. The numerical claim behind them is that a tag reaching
`Lowering::record_layout_convert` intact makes a transpose happen, and nothing tested that
half.

A round trip alone would not test it either. Two no-ops compose to the identity, and the final
assertion would pass. So the intermediate is read out and asserted, by value, to be the
permuted buffer. That assertion is the test; the round trip is the control.

The passes run before compilation, so the case also pins tag survival by its consequence and
not by inspecting the field.

The input is N=1, H=2, W=2, C=2 with values 1..8 in NHWC order, at index ((h*2)+w)*2+c. The
values are distinct because a tensor of repeats would make a permutation indistinguishable
from a copy.

[N, H, W, C] seen as [N, C, H, W] has the same extents here only because H, W and C are all 2.
The buffer differs.

The NCHW index is ((c*2)+h)*2+w, so channel 0 collects the odd values and channel 1 the even
ones. The expected buffer is worked out by hand and not read back from the implementation.

### AMaxPoolSurvivesTheFullPassPipelineAndStillRoutesItsGradient

MAXPOOL2D is the first operation whose results are not all one element type. Beside the pooled
tensor it writes an INT64 argmax, and its backward reads that as a table of offsets.
`insert_casts()` derived one precision per node from `outputs[0]` and forced every operand and
result to it. The argmax then read as a value in the wrong type, and was either retyped to
float in place or handed to a cast that refuses anything but real float to real float.

Neither half showed up in a shape check. The case runs the passes in the order
`Trainer::build` runs them and executes the result. A routing table that has been through a
float is no longer a routing table, and the gradient is where that shows.

The graph is assembled by hand and not through `Trainer::build`, because no NN layer emits a
pool yet and the trainer cannot reach this operation.

The input is N=1, H=2, W=2, C=2 with values 0..7 in NHWC order and one 2x2 window. Per channel
the maximum is in the last row: tap 6 (c=0, value 6) and tap 7 (c=1, value 7). Every other tap
must come back with no gradient.

### ChannelAffineScalesAndShiftsPerChannelNotPerRow

The input is N=1, H=1, W=2, C=2. Two pixels sharing one pair of parameters separate a
per-channel broadcast from a per-row one: BIAS_ADD's broadcast would give the second pixel the
second channel's numbers.

### BatchNorm2dNormalisesEachChannelOverTheWholeBatch

The input is N=2, H=2, W=1, C=2, so each channel is folded over four elements. The two channels
have different scales on purpose, because a normalisation that reduced over the wrong axis
would still produce four finite numbers.

Channel 0 sees 1, 2, 3, 4: mean 2.5, biased variance 1.25. Channel 1 sees 10, 20, 30, 40: mean
25, biased variance 125.

The variance is biased, dividing by N*H*W and not by N*H*W - 1. The forward normalises by that
value and the backward reads the saved statistics, so an unbiased divisor in the test would be
a different function.

## test_mixed_precision.cpp

This is the device half of AI-4. `tests/functional/unit/test_precision.cpp` pins the graph
shape with no hardware. This file holds what only a real Engine can answer:

- `sycl::half` executes here, so the fp16 path is a computed answer and not a graph nobody ran.
- The tensors behind the graph carry the types the IR claims. A master weight is a 4-byte
  buffer, and the copy a GEMM reads is a 2-byte one.
- The trainer refuses, before recording anything, a graph whose dtypes disagree with the policy
  it was handed.

No test asserts speed. This machine has no low-precision matrix unit, and a timing gate would
pin an artifact of emulation.

### optimizer_options

The helper returns AdamW at 3e-3, the rate `sa demo mlp` uses, and not the library default.
1e-3 is the reference AdamW default and is right as a default. It does not move a hidden layer
this small far in the couple of hundred steps a test can afford, and a test whose "did it
learn" threshold has to be relaxed until it passes asserts nothing. At the demo's rate the test
and the demo answer the same question at the same setting.

### AModelWithFp16ProductsStillLearns

Correctness is the one claim the mixed-precision change can make on this hardware. It is not
faster here and cannot be.

A zeroed classifier starts at a loss of ln(4). A last loss below half of that shows a model
that learned something and did not merely run.

### TheMastersAreFourByteBuffersAndTheCopiesAreTwo

The IR says FLOAT32 and HALF, and the test checks that the tensors underneath say the same.
The two could disagree. The pool materialises from the value's description, and a planner that
sized a slot from the wrong one would hand out a view that overruns its neighbour.

That the step ran at all is the loud half of the check. SushiBLAS refuses a mixed-dtype gemm
by name, so a graph with a missing conversion could not get this far.

### TheTrainerRefusesAGraphItsPolicyDisowns

A graph built without the cast pass is handed to a trainer that was told it is mixed. A caller
makes this mistake by setting one of the two knobs and forgetting the other. Without the
refusal it would not be an error; it would be an fp32 run reported as a mixed one.

### AccuracyRefusesLogitsItCannotRead

The mixed-precision change would otherwise have introduced this defect. `score_accuracy()`
reads the logits on the host. It once fell back to `data_as<float>()` for anything that was not
FLOAT64, which on a half buffer reads two elements as one and reports an accuracy that is
fiction. Nothing throws, nothing is NaN, and the run says 100%.

`build_classifier` lists the logits as observed, so the shipped path cannot reach this. A
caller running the pass themselves can, and gets an error.

## test_registries.cpp

Two registries need an Engine to exercise: datasets allocate device tensors, and optimizers
allocate their state buffers. The layer and objective registries are host-lane and live in the
unit tier.

Rings is tested on its defining property, that it is not linearly separable, and not only on
its shape. The dataset exists because a single Linear layer cannot solve it, so a test should
fail if it ever becomes solvable.

### RingsIsBalancedAndCarriesItsDistractors

Labels are assigned round-robin, so every batch is exactly balanced and accuracy needs no
class weighting to read correctly.

### RingsPlacesEachClassInItsOwnRadiusBand

The whole problem rests on this property. The class is a function of x^2 + y^2, which is
degree 2 in the inputs and therefore not a half-space. If a change made the radius bands
overlap or the labels track something linear, this test fails, where otherwise the accuracy
would rise in some other test and nobody would see why.

### TheSameSeedGivesTheSameData

The dataset draws from a host RNG and not from the engine's Philox stream, so reproducibility
is a property of the seed alone. The device stream mixes in a monotonic offset counter, which
is why `dataset.hpp` warns about `set_seed`.

### TheOptimizerRegistryBuildsBothRulesFromAConfigValue

The test passes one real parameter and gradient pair, so the optimizers allocate their state
against something and not against an empty list.

### AnUnknownOptimizerKeyIsRejectedByItsOwnFactory

"momentum" is SGD's key, not AdamW's. Asking AdamW for it must fail; it must not be a line
that does nothing.

### AConfigurationCannotReachTheReferenceUpdateKernel

`UpdateKernel` selects the spelling a byte-for-byte regression test compares the fused kernel
against. SEPARATE also cannot join the compiled plan. A config file that could set it would
be able to change what a run is, so "kernel" is not an accepted key.

## test_rematerialise.cpp

The file covers the half of the IM2COL rematerialisation pass that the host-lane tier cannot
reach. `Unit_Rematerialise` proves the rewrite has the right shape; this file proves the
rewritten graph computes the same numbers.

Checking that the pass ran and the graph still executes would prove nothing. The pass deletes
a stored value and recomputes it. Every way that can go wrong, such as a clone reading a
different operand or a clone whose attributes were rebuilt from the geometry and not copied,
produces a column matrix of the right shape holding the wrong pixels. So the assertion is
equality against the same graph run without the pass. The equality is exact and not a
tolerance: the clone recomputes the same operation from the same inputs, and SushiRuntime's
`sushiruntime_deterministic_fp` contract makes "the same" mean bit for bit.

### image_values

The [1, 5, 4, 2] image is not square on purpose. IM2COL's output extents live in attribute
slots 8 and 9. A clone goes wrong by re-deriving them where it should copy them, which can
yield the transpose of the real window grid with the same number of windows. On a square image
that mistake is invisible; on 5x4 it is a different gather.

### TheRewrittenGraphComputesTheSameGradient

The node counts are asserted first, to show the pass did something. A pass that no-ops passes
the comparison that follows trivially.

The comparison is exact, not near. The clone recomputes the same operation from the same
operands in the same order.

### TheColumnMatrixLeavesTheActivationArena

The pass exists to take the column matrix out of the activation arena. The test states that
against the planner and does not infer it from the lifetime field. The planner turns the
shortened live range into an arena decision, and it has to agree.

## test_resilience.cpp

The file checks what the library does when it is handed something it cannot serve, and what it
does at the edges of what it can. The happy path belongs to `test_forward.cpp` and
`test_training.cpp`. This file makes three other claims.

1. Every device-lane precondition fails loudly and names the actual problem.
   `tests/common/test_throws.hpp` explains why the message is asserted and not only the throw.
2. Degenerate and boundary extents produce right answers and not off-by-one ones: one sample,
   one feature, one class, exactly the GEMM register-block edge and exactly one past it. Index
   arithmetic breaks here or nowhere.
3. Numbers that would make a naive implementation produce NaN do not. A NaN is the worst
   failure this library has. It propagates into every weight on the next update, the loss
   stops moving, and nothing reports an error. Finiteness is therefore asserted explicitly and
   not inferred from a loss that looks plausible.

The file is laid out in four groups: loud failures that name the problem, degenerate and
boundary extents, numerical extremes, and resource discipline.

### AnOptimizerRefusesADtypeItCannotServeAndExplainsWhy

Complex is not a training dtype at all. Half is a real float, and it is refused for a reason
the message has to carry: the refusal exists to prevent a model that compiles and trains
worse.

### AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage

The library has no `zero_grad()`, by design. Every gradient buffer in a step is written and
not accumulated into (see `optimizer.hpp`). That leaves one window in which a gradient has
never been written: between `bind_parameters()` and the first replay. Stepping in that window
would feed whatever the allocator last had in that memory straight into the weights. Nothing
downstream catches it. The parameters become garbage or NaN, and the loss never recovers.

### TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep

A step's buffers are addresses baked into a compiled plan. A batch of the wrong size cannot be
accommodated, only reported.

### AllocationRefusesAnEmptyOrUnrepresentableTensor

Signed overflow in the extent product is undefined behaviour, and a compiler may assume it
away along with any check written after it. The overflow is detected before the multiply, so
the diagnostic is about the shape and not about what the overflow left behind.

### AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel

The request is a petabyte, which no allocator serves. What matters is what the caller is told:
the shape and element type it asked for. A bare byte count from four frames down would not
help, and neither would a null pointer for a kernel to dereference.

### ASingleClassIsDegenerateButNotUndefined

With C = 1 the softmax is identically one, so the loss is exactly zero and the gradient
(p - y) / N is exactly zero. Nothing learns, which is correct. A 0/0 anywhere in the recipe
would show up as NaN.

### MatmulIsExactAtTheRegisterBlockEdgeAndOnePastIt

The local-tiled GEMM stages a 64x64 block of float per work-group and pads the ragged
remainder with zeros. An extent of exactly 64 exercises the padding-free path. 65 leaves a
one-row, one-column remainder, where an off-by-one in the boundary test shows up as a wrong
answer and not as a fault. Both are checked against a host reference, with the transpose flag
the Linear layer emits.

The values that reached the device are float. The reference is computed from those same
floats, or the comparison would measure rounding and not indexing.

### LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude

The max shift is the only reason these inputs are representable. Without it exp(1e30) is inf
and exp(-1e30) is zero, and both give NaN. With it every magnitude must land on the same
answer, because a uniform row is uniform whatever its offset.

### LogSoftmaxSurvivesBothExtremesInsideOneRow

The row holds +1e30 against -1e30. The shift drives the winner to exactly zero and every loser
to -2e30, whose exponential underflows to zero without becoming a NaN. The row sum is exactly
one and log(1) is exactly zero, so the winner's log-probability is exactly zero. That is a
probability of one, the right answer for a logit gap of 2e30.

### LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange

The two rows' own maxima are 60 apart. The case once pinned the supported side of a boundary.
The shift was by the batch's global maximum, so a row far enough below it came back +inf, and
60 was inside the range that still worked.

The boundary is gone: the shift is each row's own maximum, so the gap does not reach the
arithmetic. The case is kept unchanged. It is the shape a training run produces, and a
regression that brought back a global shift would still pass every test whose rows sit on top
of each other. The gaps that break a global shift are pinned separately, with their measured
errors, in `tests/functional/regression/test_log_softmax_stability.cpp`.

Both rows have the same internal spread, so both must produce the same log-probabilities. That
is the claim of shift invariance.

### CrossEntropyIsFiniteOnPerfectlyConfidentPredictions

The logits carry a confidence of exp(80) either way. Anything that exponentiates without
shifting first overflows here, and anything that takes the log of a rounded zero produces
-inf.

With the same logits against the opposite targets the loss is large, and large is the correct
answer. It must not be inf, because an infinite loss makes every gradient below it a NaN.

### AdamWsFirstStepWithAZeroGradientStaysFinite

m and v are both exactly zero on step one, so the update is 0 / (sqrt(0 / bc2) + eps). Only
the eps in the denominator keeps that a zero and not a NaN. eps is added after the bias
correction, so it is not scaled away. Folding it inside makes the denominator
sqrt(0 + eps) / sqrt(bc2), which is still non-zero here and wrong everywhere else. Dropping it
makes this 0/0.

A second step still has no gradient to work with, and the bias corrections have changed. A
divide that only happened to be safe on step one would fail there.

### BuildingAndDestroyingTrainersReachesASteadyState

Every `SushiBLAS::Storage` allocates through the runtime context, which records each
allocation, so `RuntimeContext::live_usm_count()` is the number of live allocations. Whether the last
iteration leaked a buffer is then a number the test can read, where otherwise somebody would
have to watch Task Manager. One leaked Storage moves the count by exactly one.

The baseline is taken after the first iteration and not before it. A one-time lazy cache
inside the runtime is tolerated; a per-iteration leak is not.

### BatchNormUpdateStatsIsNotAnIrOperationAndSaysSo

CV-7 keeps a BatchNorm's running statistics out of the IR. They are recorded into the training
step beside the graph, the same way `src/optim/` emits SGD_UPDATE and ADAMW_UPDATE directly.
BATCHNORM_UPDATE_STATS is therefore the one declared identity with no case in
`Lowering::record_node`, and that is deliberate.

From the outside a deliberate gap and an omission look the same, which is why the test exists.
Pinning the throw records the decision. If somebody later adds a lowering, the test fails and
asks them to say why.

## test_training.cpp

This is the end-to-end gate: a real MLP on the synthetic classification dataset that has to
converge, with the compile-once guarantee checked across every step.

### InitialLossIsExactlyLogCWithAZeroedClassifier

Zeroing the last layer makes every logit zero, so every class is equally likely and the
initial cross-entropy is ln(C) to the last digit. The test can assert that number; a loss that
merely looks reasonable would have to be judged by eye.

### TheStepIsCompiledExactlyOnceAcrossManySteps

This is the AI-1 headline gate. A loop that recompiles per step still converges and pays graph
analysis every step. That regression hides for months unless something counts compilations.

### TheStepAllocatesNothingInSteadyState

Compile-once is half the claim. The other half is that replaying reuses the same buffers, so a
long run's memory does not grow.

### AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate

The "eps after the bias correction" ordering exists to preserve this property. On step one,
m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so every coordinate moves by almost exactly the
learning rate. Folding eps inside the correction makes it 31.6x larger and visibly damps this
first step, the only place the bug is easy to see.

The loop skips coordinates whose gradient is below 1e-4, where eps does matter. The claim is
about the ordinary case.

### ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes

The shared test context is built without `RuntimeSettings::profiling`, so the runtime records
no timing. The report has to say that. A table of zeroes would read as if the work took no
time.

### ADisconnectedParameterIsRejectedRatherThanSilentlySkipped

The test fakes a parameter that no gradient reaches. It stands in for a layer wired into the
model list but not into the forward pass.

### TheSyntheticDatasetIsDeterministicAndBalanced

Each dataset gets a fresh Engine on purpose. SushiBLAS's `Engine::set_seed()` sets the seed and
leaves the monotonic RNG offset counter alone, so a second generator built on an engine that
has already drawn gets a different stream from the same seed. Reproducibility is therefore a
property of a fresh engine, which is how the trainer uses it.

Round-robin labels make every batch exactly balanced, so accuracy reads correctly without
class weighting.
