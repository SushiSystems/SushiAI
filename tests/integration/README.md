# Integration tests

The integration tier runs on a real `SushiBLAS::Engine`. Each file here checks a claim that
needs a device to answer; the host-only half of the same claim lives in
`tests/unit/`. This document holds the reasoning behind the tests: why a case is
shaped the way it is, which wrong implementation it is built to catch, and the numbers worked
out by hand. The source keeps one line per case and cites this file.

## The SaTest alias

`test_autograd_numeric.cpp`, `test_forward.cpp`, `test_resilience.cpp` and `test_training.cpp`,
among others, declare `namespace SaTest = SushiAI::Test`. Inside a `TEST_F` body the fixture's base
class injects the name `Test`, so `Test` alone resolves to `::testing::Test`. The alias keeps the
helpers in `SushiAI::Test` reachable from a test body.

## test_anchor_training_step.cpp

Suites: `Integration_AnchorTrainingStep`, in `sushiai_vision_tests`.

The model reads `[8, 8, 3]` images through two normalised stride-2 convolutions of 8 channels, each
followed by a LeakyReLU of 0.1 and a 1x1 head convolution of 14 channels. It publishes `head.0` on a
4 by 4 grid, `head.1` on a 2 by 2 grid and `stem`, the first convolution before its normalisation.
The layout holds four anchors, two for each head, over two classes. The batch is two seeded images
with three truths, two assigned to head 0 and one to head 1. An `AnchorTrainingStep` runs a
forward-only graph and a training graph over one state, with plain SGD at a rate of 0.1.

### TheLeavesFedAreTheDeltasOfTheForwardOnlyHeads

After one run, the delta of each head is computed on the host from the forward-only output with
`compute_head_delta`. The leaf the training graph was fed equals that delta element for element,
with a tolerance of 0, and the assignments are 2 and 1. The stored cost is the sum of the squared
deltas and the total is that sum over the batch of two, both to 1e-9 relative.

### OneStepMovesAHeadBiasAlongItsDelta

Each of the 14 entries of `fine.bias` after one step equals its value before plus rate / batch times
the sum of its channel's deltas over the 32 positions, within 1e-5. At least one bias moves by more
than 1e-7.

### OneStepBlendsARunningMeanOnce

Each entry of `stem_norm.running_mean` after one step equals `(1 - momentum) * before + momentum *
mean`, within 1e-5, where the mean is taken on the host over the published `stem` output. The
largest batch mean must exceed 1e-2, so that a second blend would be visible.

### ATruthOfAnotherBatchIsRefused

A truth of three images against a step of two is refused as `Incompatible`. The message contains "3
images" and "batch of 2".

### ALabelOutsideTheClassesIsRefused

A valid row whose label is 2, with two classes, is refused as `InvalidDescription`. The message
contains "row 2", "label 2" and "head.0".

### ACentreOutsideTheInputIsRefused

Row 0 of image 1 is moved to columns 7 to 9, which puts its centre at exactly 1. The run is refused
as `InvalidDescription`, and the message contains "truth row 0 of image 1" and "centre".

### AnObjectiveNamingAnUnpublishedHeadIsRefused

An objective that names `head.9` is given to `AnchorTrainingStep::make` over a model that publishes
`head.0` and `head.1`. The step is refused as `InvalidDescription`, and the message names "head.9".

## test_arena.cpp

Suites: `Integration_Arena`, `Integration_ArenaDeathTest`.

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

### TwoPackedTensorsShareABaseButNotAByte

The plan packs both values of the chain at different offsets, and the `ArenaSet` holds one arena.
The two `[16]` float32 views report regions with the same base, which is the allocation base of the
storage, and neither region overlaps the other. Each region is 64 bytes long and starts at the
offset its placement states.

### OpsRecordedOnTwoPackedTensorsDoNotTreadOnEachOther

This is the functional half of the region claim. Real SushiBLAS work runs on two views of one
arena in one step, and each result must land only in its own bytes.

### AViewAddressesItsOwnElements

The view of the second value has a `storage_offset` equal to its placement offset divided by the
size of a float. Its data pointer lies exactly that many bytes past the allocation base.

### AnOffsetThatIsNotAWholeNumberOfElementsIsRefused

An offset of four bytes into a float64 view can be written in bytes but not in the element
count `Tensor` stores. Truncating it would put the tensor somewhere the planner never reserved.
The same offset is valid for float32.

### AViewThatWouldRunPastTheArenaIsRefused

`view_tensor` of 64 float32 elements at byte 64 of a 128 byte arena is a fatal stop whose message
contains "run past the end".

### MixedDtypesEachStartAlignedForTheirOwnElementWidth

The plan's alignment is set to 1 so that only the element-width floor holds the two values
apart. With the default alignment the property holds for an uninteresting reason; at 1 it
holds only because the planner rounds.

The round trip through each view proves the addresses are usable as well as aligned. A float64
read through a mis-rounded start would come back as two halves of neighbouring values, not as
the number written.

### AdoptingASecondPlanIsRefused

A `TensorPool` adopts one plan. Adopting a second is a fatal stop whose message contains "already
adopted".

### TheEmptyPlanPacksNothing

The empty plan is the control an A/B measurement of the whole pass needs. It is also what a
caller uses to get unpacked allocation.

### APlannedPoolAllocatesOneStorageForAWholeArena

The test compares no byte counts. Its tensors are 64 bytes each and the arena aligns every slot
to 128, so packing them costs memory. The saving belongs to realistic tensor sizes and is
measured where that holds, in `Regression_StaticMemory`.

## test_attention_gradient.cpp

Suites: `Integration_AttentionGradient`, in `sushiai_language_tests`.

Every graph is FLOAT64. The file's `Probe` builds a graph, takes the cross-entropy of the `[N, C]`
value its build function names against targets whose row r names class r mod C, differentiates it at
one leaf and compiles it. Every other leaf is filled from the seeded host stream.
`Autograd::check_full_jacobian` compares the analytic gradient with central differences. The block
cases read a `[1, 4, 4]` sequence through a `CausalSelfAttention` of two heads.

### AnUnfusedLogSoftmaxMatchesItsClosedForm

A SCALE by one sits between LOG_SOFTMAX and NLL_LOSS over a `[3, 4]` parameter, and the graph holds
one LOG_SOFTMAX_BACKWARD node. The targets are positive weights whose rows do not sum to one. After
two runs the gradient equals `(softmax(x) * rowsum(Y) - Y) / N`, computed on the host, within 1e-12.

### ASoftmaxGradientMatchesCentralDifferences

A `[3, 4]` softmax is read by a matrix product with a `[4, 3]` input. The graph holds one
LOG_SOFTMAX_BACKWARD node, and the gradient at the softmax's operand is checked against central
differences.

### ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes

A softmax over the last axis of a `[2, 3, 4]` parameter is reshaped to `[6, 4]`. The graph holds one
LOG_SOFTMAX_BACKWARD node, and the gradient is checked against central differences.

### ACausalAttentionDifferentiatesAtItsInput

The gradient of a causal attention block is taken at the sequence it reads. The graph holds two
LOG_SOFTMAX_BACKWARD nodes, one for each head, and the gradient is checked against central
differences.

### ACausalAttentionDifferentiatesAtItsProjectionWeight

The same block is differentiated at the weight of its `qkv` projection, with the same node count and
the same check.

### AnUnmaskedAttentionDifferentiatesAtItsInput

A block built with `AttentionMask::None` is differentiated at its input, with the same node count
and the same check.

## test_attention_operations.cpp

Suites: `Integration_AttentionOperations`.

The file runs the rank-3 matrix product, the softmax and the product's gradient on an engine in
FLOAT32. The inputs follow a fixed pattern, `scale * ((i * stride) % 11 - 5)`. Every result is
compared with a host computation in double within 1e-6.

### BatchedProductWithTransposedRightOperandMatchesHost

`matmul` of a `[2, 3, 4]` value with a `[2, 5, 4]` value, the right operand transposed, gives the
`[2, 3, 5]` product a host loop computes.

### SoftmaxOverTheLastAxisMatchesHost

The softmax of a `[2, 3, 3]` value equals the host softmax of each row of three.

### BatchedProductGradientEmitsRankThreeProducts

The loss is the sum of every element of the same product, differentiated at both operands. The
gradients have the shapes `[2, 3, 4]` and `[2, 5, 4]`, and the graph holds three MATMUL nodes, each
with a rank-3 output. The values equal the host sums: for the left operand the sum of the right over
j, for the right operand the sum of the left over i.

## test_autograd_numeric.cpp

Suites: `Integration_Autograd`.

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

### MatmulWithoutTransposes

The variable is a `[4, 5]` left operand multiplied by a `[5, 3]` input with neither side transposed.
Its gradient is checked against central differences.

### MatmulWithATransposedRightOperand

A `[4, 5]` variable against the transpose of a `[3, 5]` input, the form a `Linear` layer emits. The
gradient is checked against central differences.

### MatmulWithATransposedLeftOperand

The transpose of a `[5, 4]` variable against a `[5, 3]` input. The gradient is checked against
central differences.

### MatmulWithBothOperandsTransposed

The transpose of a `[5, 4]` variable against the transpose of a `[3, 5]` input. The gradient is
checked against central differences.

### MatmulGradientOfTheRightOperand

The same node is differentiated from the other side. The variable is the weight, so the case
covers the dB half of the rule where the earlier matmul cases cover dA.

### BiasAddPassesTheMatrixGradientThrough

The variable is the `[4, 3]` matrix of a `bias_add` whose bias of three elements is an input. The
gradient is checked against central differences.

### BiasAddSumsOverTheBroadcastAxis

The same node is differentiated at the bias of three elements, with the `[4, 3]` matrix as an input.

### ElementwiseAddAndSubtract

The case runs twice over a `[4, 3]` variable and a `[4, 3]` input, once through `add` and once
through `sub` with the variable on the left. Each gradient is checked against central differences.

### ElementwiseSubtractFromTheRight

The variable is the right operand of `sub`, so the case covers dB = -dC.

### ElementwiseMultiply

The variable is the left operand of `mul` against a `[4, 3]` input.

### ElementwiseMultiplyFromTheRight

The case checks dB = dC * A, the half the left-hand check cannot reach. The rule is written
symmetrically, so a copy-paste slip that reads `inputs[1]` on both sides would leave the
left-hand test green.

### AccumulateInTheForwardGraph

K_ACCUMULATE has a backward rule that nothing else exercises. The two fan-out tests build an
K_ACCUMULATE chain in the backward graph, and `differentiate()` walks only the forward nodes, so
those chains are never differentiated themselves. Reaching the rule needs an K_ACCUMULATE the
caller put in the forward pass.

The lowering aliases the result onto the accumulator's own buffer, so running the step leaves
the variable holding `x + other`. Every probe rewrites the whole variable first, which makes
that harmless.

### Scale

A `[4, 3]` variable is scaled by -2.5.

### Exp

The gradient of `exp` over a `[4, 3]` variable.

### ReLUAwayFromItsKink

ReLU is not differentiable at zero. A difference quotient that straddles the kink reports the
average of the two one-sided slopes, which no correct rule can match, so the point is sampled
clear of zero.

### LeakyReLUHasBothOfItsSlopes

The point is sampled off the kink and over a range wide enough that elements land on each
side. A probe that only saw positive inputs would pass with the negative slope hard-coded to
zero, which is a plain ReLU. `LeakyReLUBranchSwapIsCaughtByProbePoints` counts the negative
points the sample contains.

### LeakyReLUBranchSwapIsCaughtByProbePoints

The test pins the negative-side split of the exact point `LeakyReLUHasBothOfItsSlopes` uses:
same seed, same scale, same `push_off_kinks`. The branch-swap mutation depends on that split.
If a change to the seed or the noise generator collapsed the sample to all-positive, the
mutation would go undetected, and this test fails first.

### GELU

The gradient of `gelu` over a `[4, 3]` variable sampled with a noise scale of 1.5.

### AFusedGELUEpilogueHasTheSameDerivativeAsAnUnfusedOne

The central difference is the strongest oracle in the repository. Here it is pointed at the
one fused form whose correctness does not follow from running the same operations in the same
order. K_GEMM_BIAS_GELU rewrites what the backward pass reads: the pre-activation arrives from a
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

### SigmoidReadsItsForwardOutput

The gradient of `sigmoid` over a `[4, 3]` variable sampled with a noise scale of 1.5.

### TanhReadsItsForwardOutput

The gradient of `tanh` over a `[4, 3]` variable sampled with a noise scale of 1.5.

### SumRowsBroadcastsItsGradientBack

`sum_rows` reduces a `[4, 3]` variable to three sums, and a `bias_add` onto a `[1, 3]` value of
zeros turns them into the one-row batch the loss reads. The gradient at the variable is checked
against central differences.

### GatherRowsScatterAddsARepeatedIndexIntoItsTable

The variable is a `[5, 3]` table gathered with the index `{3, 1, 3, 3, 0, 1}`. Row 3 is read three
times and row 1 twice, and rows 2 and 4 are read by no index.

### GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape

A `[4, 3]` table is gathered with a `[2, 3]` index into `[2, 3, 3]` and reshaped to `[6, 3]` for the
loss. The gradient at the table is checked against central differences.

### ScatterAddRowsGathersARepeatedIndexBackOntoItsRows

The variable is the `[6, 3]` rows scattered into five table rows with the index `{3, 1, 3, 3, 0,
1}`. Row 3 is written three times and row 1 twice, and rows 2 and 4 are written by no index.

### ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape

`[2, 3, 3]` rows over a `[2, 3]` index sum into a `[4, 3]` table. The gradient at the rows is
checked against central differences.

### FusedCrossEntropyAgainstOneHotTargets

The gradient of the mean NLL with respect to the logits is (softmax - Y) / N. It is emitted as
one fused rule and not by differentiating the seven-op log-softmax recipe, so it needs its own
numeric proof.

### GradientAccumulationAtAFanOut

The variable feeds `tanh` and `sigmoid`, and their sum is the logits. The two branches make the
gradient of the variable an ACCUMULATE chain.

### WholeTwoLayerModelUnderRandomDirections

This is the end-to-end check. A full Jacobian over every weight would take 2n device runs.
Random unit directions take 2k runs whatever n is, and the chance that a wrong gradient agrees
along all sixteen of them is negligible.

### ReshapeMovesItsGradientBackUnchanged

A reshape moves no data, so its gradient is its incoming gradient in the original shape. The
numeric check is still worth having: reshaping back to the input shape and reshaping back to
the output shape are both one line, and only one is right.

The fused K_LOG_SOFTMAX + K_NLL_LOSS rule's closed form assumes a one-hot target row
(`nll_loss_backward` says it is "valid because the targets are one-hot"). Noise in the targets
would make the analytic gradient disagree with the numeric one for reasons that have nothing
to do with K_RESHAPE.

### LayoutConvertsGradientIsThePermutedGradient

The case uses a single `layout_convert`, so the loss depends on the permuted element order. A
round trip would not do. A rule that reshaped dy back to the operand's shape without permuting
it has the right magnitude in the wrong place, and it still satisfies every shape check
`Differentiator::contribute` makes. A round trip cannot tell the two apart, because any
involutive, shape-correct pair of backward hops composes to the identity.

CV-4 forces every node that is neither a K_LEAF nor a K_LAYOUT_CONVERT to agree with its operand
on layout. A variable therefore cannot reach `log_softmax` by converting one way and stopping:
`graph::finalise()` would refuse the mismatched `log_softmax` output before any gradient ran.
CV-4 exempts a leaf, which has no operands to disagree with, so its layout tag is free. This
is also how the operation is used in practice. K_LAYOUT_CONVERT exists for boundary interop,
such as loading an NCHW checkpoint, where the source data carries the other tag and no
mid-graph value does. The case tags the variable NCHW before its one K_LAYOUT_CONVERT node and
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

A probe ending in `log_softmax` must use one-hot targets (A-6). The fused K_NLL_LOSS rule's
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

K_COL2IM's rule is the mirror image of `im2col_backward`. The case checks it without `im2col`:
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

An earlier draft of the design lacked this rule. It is a per-(n, c) broadcast. K_BIAS_ADD's
broadcast is per row and K_CHANNEL_AFFINE's is per channel, and neither would have produced it.

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

### BatchNormsShiftGradientIsThePlainSumOfDy

The same batch norm over a `[4, 3, 3, 5]` input is differentiated at beta, its shift of five
elements. The logits are the spatial mean of the output through a `log_softmax`, against one-hot
targets.

### ChannelAffineDifferentiatesAtItsInput

The variable is the `[2, 4, 4, 6]` input of a `channel_affine` whose scale and shift are parameters
filled with noise. The logits are the spatial mean through a `log_softmax`, against one-hot targets.

### ChannelAffinesScaleGradientSumsOverNHAndW

dgamma is a sum over three axes at once. A sum over the wrong ones, per sample where it should
be per channel, is still a [C]-shaped value when N equals C. That is why the case is checked
numerically and not by shape.

### ChannelAffinesShiftGradientSumsOverNHAndW

The same node is differentiated at beta, its shift of six elements.

### DepthwiseConvolutionDifferentiatesOverItsInputAndItsTaps

One flat variable is sliced into a `[2, 5, 5, 3]` image and `[3, 3, 3]` taps, so one check covers
both operands. The convolution has a 3x3 window, a stride of 2 and a pad of 1 at the top and the
left, and gives `[2, 3, 3, 3]` windows whose spatial mean is the logits.

## test_buffer_update_recorder.cpp

Suites: `Integration_BufferUpdateRecorder`.

The recorder blends each buffer's batch statistic into its running buffer, once per replay. The
cases check how many tasks it records, the value one replay leaves behind, the empty recorder,
and the two refusals that binding makes.

### RecordsTwoTasksPerSlot

Two slots record four tasks in all, two for each slot. The test asserts that `slot_count()` is 2
and that `record_into_step` returns 4.

### OneReplayBlendsTheSourceIntoTheTarget

One slot over four float64 elements. After one replay each element equals the
`NN::updated_statistic` of its running value and its batch value, with momentum 0.1 and a source
scale of 2, within 1e-9.

### NoSlotRecordsNothing

A recorder built with no slots records zero tasks, reports zero slots and runs without error.

### BindRefusesATargetTheStateLacks

The state holds one entry, `bn.running_mean`, under identity 1. The update targets identity 7.
Binding is refused as `Incompatible`, and the message contains "no entry".

### BindRefusesADTypeMismatch

The target entry is float32 and the source tensor is float64. Binding is refused as
`Incompatible`, and the message names both dtypes by their `dtype_name`.

## test_cached_generation.cpp

Suites: `Integration_CachedGeneration`, in `sushiai_language_tests`.

The fixture builds the decoder of `decoder_model.hpp`: a learned position table, two blocks, a
window of 8 and a vocabulary of 32. It traces the `[1, 8]` window graph and the `[1, 1]` incremental
graph, initialises one state, and builds a `WindowGenerator` and a `CachedGenerator` over it. Tokens
come from a greedy sampler. Float32 logits of the two passes are compared with an absolute tolerance
of 1e-4.

### TheGeneratorReadsTheExtentsOfTheStep

The cached generator reports a window of 8 and a vocabulary of 32. Its step carries state and has
three inputs: `position` is one of them and `cache_slot` is refused as `InvalidDescription`. The
incremental graph carries four values, two for each block.

### TheCachesAreNoEntryOfTheModelState

A state created again from the model after both traces has as many entries as the first one, and
every entry is a parameter.

### APromptThatFitsGivesTheTokensOfTheWindowGenerator

A prompt of three ids and four generated tokens stay inside the window. The cached generator gives
the tokens of the window generator, in 6 runs of one compile.

### TheLogitsEqualThoseOfTheWindowAtTheLastPosition

After a prompt of five ids, the 32 logits of the one-position step equal the window step's logits at
position 4 within 1e-4.

### AContextThatSlidesGivesTheTokensOfTheWindowGenerator

A prompt of three ids and twelve generated tokens pass the window. The tokens equal the window
generator's, and the step ran 56 times from one compile: 3 for the prompt, 5 to fill the window, and
8 for each of 6 slides.

### APromptLongerThanTheWindowKeepsItsLastIDs

A prompt of eleven ids gives the same four tokens from both generators.

### ASecondPromptMatchesAFreshGenerator

A generator that has produced twelve tokens for one prompt is given a second prompt. Its four tokens
equal those of a generator compiled fresh over the same state and those of the window generator.
Both steps report one compile.

### AStepThatDoesNotFitIsRefused

`CachedGenerator::make` over the window step is refused as `InvalidDescription` with "keeps nothing
between runs". A mask leaf named `scores` is refused with a message that names 'scores' and
'attention_mask'.

### APromptOrACountThatDoesNotFitIsRefused

An empty prompt, the id 32 and a count of zero tokens are each refused as `InvalidDescription`; the
messages contain "no id", "0 to 31" and "0 tokens". Nothing is generated and the step has run zero
times.

## test_causal_attention.cpp

Suites: `Integration_CausalAttention`, in `sushiai_language_tests`.

The mask cases run CAUSAL_MASK with a fill of -1e9 on an engine and compare with a host loop. The
block cases trace a block over a `[1, 5, 8]` input with two heads, fill every parameter from the
seeded host stream, and run it twice: once on an input, once with positions 3 and 4 replaced. Values
that must be the same number in both runs are compared with a tolerance of 0.

### TheMaskReplacesTheEntriesAboveTheDiagonal

A FLOAT32 `[2, 4, 4]` value of dyadic numbers is masked and the step is run twice. The lower
triangle of each matrix is unchanged and every entry above the diagonal holds the fill.

### TheMaskReplacesTheEntriesAboveTheDiagonalInFloat64

The same check in FLOAT64.

### TheMaskPassesItsGradientThroughTheKeptEntriesAlone

The loss is the cross-entropy of the masked FLOAT64 `[4, 4]` rows against class 0. The gradient at
the scores equals `(softmax - target) / 4` on and below the diagonal within 1e-12, where the softmax
is taken over the kept entries, and is exactly zero above it.

### AnEarlierPositionIgnoresALaterInput

Through a causal `CausalSelfAttention`, no element of positions 0 to 2 differs between the two runs,
and at least one element of positions 3 and 4 does. Both outputs hold 40 elements and the first is
finite.

### WithoutTheMaskAnEarlierPositionReadsALaterInput

With `AttentionMask::None`, at least one element of positions 0 to 2 differs between the two runs.

### ATransformerBlockKeepsEarlierPositions

A causal `TransformerBlock` with a feed-forward width of 32 keeps positions 0 to 2 unchanged and
changes positions 3 and 4.

## test_checkpoint.cpp

Suites: `Integration_Checkpoint`.

The file covers the device half of checkpointing: saving and loading records against the tensors
of a `NN::ModelState`. The byte format's malformed-file surface needs no device and lives in
`modules/layers/io/tests/test_checkpoint_format.cpp`.

The last equivalence test is the point of the file. A `Sequential` and a `LayerStack` built
from the equivalent `model.json` are two spellings of one model. The proof that they are one
model is a checkpoint written by either loading into the other. A separate mechanism would
leave its own correctness as the open question.

### Materialised

`traced_state` traces the model over a [4, 6] input and creates its `NN::ModelState` from the
traced graph. `NN::initialize` then fills that state with a policy that leaves the final layer
random, so every parameter carries real numbers and not the zeros of the default policy. The
class owns the state, because every tensor a test reads has to outlive the comparison.

### SavesAndLoadsEveryParameterBitForBit

A model materialised from seed 11 is saved. A second one from seed 22 differs before the load, and
after `load_checkpoint` its parameters have the saved bit patterns.

### ASequentialCheckpointLoadsIntoAnEquivalentLayerStackAndBack

Two spellings of one model prove they describe the same thing by exchanging a checkpoint. The
compile-time and run-time paths then need no separate mechanism to keep them in step, and they
cannot drift without this test failing.

The names are asserted equal first. The checkpoint is matched by name, so a naming difference
would otherwise surface as "does not supply" and hide the clearer statement that the two
models disagree about what they hold.

The return trip, `LayerStack` to `Sequential`, loads into a third model. That makes it a real
load and not a comparison against values that were already there.

### AParameterTheFileDoesNotSupplyIsRefusedAndNamesIt

The file is saved from a shallower model, so it has no `2.weight` or `2.bias`. Loading it into the
deeper model is refused with a message that names `2.weight`, and the load reports
`StatusCode::Incompatible` as a failure.

### AParameterTheModelDoesNotHaveIsRefusedAndNamesIt

The direction is reversed: the file is saved from the deeper model and loaded into the shallower
one, so the file holds `2.weight`, which the model does not have. The load is refused with a
message that names `2.weight` and says that this model has no such entry.

### AShapeMismatchIsRefusedWithBothShapes

The two models have the same parameter names and different widths. An element-count comparison
would catch this pair but would miss a transpose, so the loader compares shapes.

### ADuplicatedNameIsRefusedAndNamesTheDuplicate

The file is built by hand because `save_checkpoint` cannot produce one: the walk yields each
name once. A counter-based loader would accept this file and leave a parameter reading
uninitialised device memory.

## test_darknet_decode.cpp

Suites: `Integration_DarknetDecode`, in `sushiai_vision_tests`.

The suite is parameterised over six models: `yolov3-tiny`, `yolov7-tiny`, `yolov7`, `yolov3`,
`yolov4-tiny` and `yolov4`. For each it reads the `.cfg` from the directory `SA_REFERENCE_DATA_DIR`
names and, for every [yolo] head, three reference tensors under `references/<run>/`: the raw head
input, the decoded boxes and the decoded scores before any threshold. The weights are not read. The
decode runs on the host through `decode_head`.

### TheHeadsDecodeToTheDumpedRows

The translated model has the head count the file lists, each head at the listed layer with 80
classes and options that `validate_head` accepts. The reference input has the extents `[grid, grid,
channels]`, the boxes `[rows, 4]` and the scores `[rows, 80]`. Every decoded box entry and score is
within 1e-6 absolute of the reference, and the largest difference is printed. The case is skipped,
naming the variable or the first missing file, when the reference data is absent.

## test_darknet_import.cpp

Suites: `Integration_DarknetImport`, in `sushiai_vision_tests`.

The suite runs once for each model of the parity table in `vision_support/darknet_forward.cpp`:
`yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7`. The fixture
`DarknetImportTest` imports the model in memory from its `.cfg` and `.weights` in the directory
`SA_REFERENCE_DATA_DIR` names. A case is skipped when a file of its model is missing.

### TheImportedRowsAndTaskAreThoseOfTheTranslation

The imported model file, without its task, is the text of the translated model. The task reads back
as the anchors, the heads and the suppression criterion.

### TheImportedWeightsAreThoseOfTheDirectStore

A state filled from the imported tensors equals, byte for byte, the state `store_darknet_weights`
fills. The imported tensors supply every entry exactly once.

## test_darknet_store.cpp

Suites: `Integration_DarknetStore`, in `sushiai_vision_tests`.

The file needs an engine and no reference data. The state is built with `ModelState::from_entries`
for a model of two convolutions: layer 0 with 2 filters, 3 channels and a kernel side of 2, and
layer 1 with 2 filters, 1 channel, a kernel side of 1 and batch norm in its own entries. The weights
are written by hand, and `store_darknet_weights` moves them into the state.

### WeightsMoveToKernelLastAndEveryVectorLands

The weight of layer 0 lands in `0.conv.weight` kernel last: the expected values are built by walking
the destination order and reading the source as `[out, in, kh, kw]`. The biases, the scales, the
rolling mean and the rolling variance land in their entries unchanged.

### AFoldedConvolutionLeavesItsNormalisationEntriesAlone

Layer 1 is folded and its target names only a weight and a bias. Those two entries receive the
folded values, and the four batch-norm entries keep the sentinel -7 written before the store.

### AMissingEntryIsRefusedAndNamesTheLayer

A target that names `1.bn.missing` is refused as `Incompatible`. The message contains "layer 1" and
the entry name.

### AShapeMismatchIsRefusedAndNothingIsWritten

`1.bn.running_var` is given the shape `[3]`. The store is refused as `Incompatible`, the message
names the entry and "shape", and `0.conv.weight` still holds the sentinel in all 24 elements.

### AVectorOfTheWrongLengthIsRefused

Layer 1 is folded but its target still names the batch-norm entries. The store is refused as
`Incompatible`, and the message contains "layer 1", "0 values" and '1.bn.weight'.

### AnEntryNamedTwiceIsRefused

Two targets name `0.conv.bias`. The store is refused as `Incompatible`, and the message contains
"layer 1", the entry name and "already fills".

### MismatchedCountsAreRefused

Two convolutions are stored against a list of one shape. The store is refused as `Incompatible` with
"the weights hold 2 convolutions".

## test_darknet_yolov3_forward.cpp

Suites: `Integration_DarknetYolov3Forward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv3 from `yolov3.cfg` and `yolov3.weights` in the
directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 416 pixel input with
the reference tensors under `references/`: `yolov3` at a threshold of 0.25 and `yolov3_thresh_0p01`
at 0.01. The model has three heads on grids of 13, 26 and 52, each with three anchor slots over 80
classes, coded as logits. A head value is held to 1e-3 absolute plus 1e-4 relative, and a box entry
or a score of a detection row to 1e-4. Every case is skipped when a file is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The three published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the logistic of every entry
but the sizes, and compared with the dumped layer output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: two at
0.25 and three at 0.01. Suppression runs at 0.45 on the IoU. Both sides are sorted by class, then by
score, and a row matches when its class is equal and its box entries and score are within 1e-4.
Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov3_tiny_forward.cpp

Suites: `Integration_DarknetYolov3TinyForward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv3-tiny from `yolov3-tiny.cfg` and `yolov3-tiny.weights`
in the directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 416 pixel input
with the reference tensors under `references/`: `yolov3_tiny` at a threshold of 0.25 and
`yolov3_tiny_thresh_0p01` at 0.01. The model has two heads on grids of 13 and 26, each with three
anchor slots over 80 classes, coded as logits. A head value is held to 1e-3 absolute plus 1e-4
relative, and a box entry or a score of a detection row to 1e-4. Every case is skipped when a file
is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The two published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the logistic of every entry
but the sizes, and compared with the dumped layer output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: none
at 0.25 and 16 at 0.01. Suppression runs at 0.45 on the IoU. Both sides are sorted by class, then by
score, and a row matches when its class is equal and its box entries and score are within 1e-4.
Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov3_tiny_layers.cpp

Suites: `Integration_DarknetYolov3TinyLayers`, `Integration_DarknetLayers`, in
`sushiai_vision_tests`.

The file compares the body of a model layer by layer. It covers two models that have a dump of every
layer: `yolov3-tiny` in the run `yolov3_tiny` and `yolov7-tiny` in the run
`yolov7_tiny_layers_plain_pool`. The fixture loads the `.cfg` from the directory
`SA_REFERENCE_DATA_DIR` names and skips, naming the model and the first missing file, when the cfg,
the weights, the input or a layer dump is absent. A [yolo] layer is not compared here; the forward
test of its model covers it.

### TheSectionsAndRowsAreThoseOfTheDumps

`yolov3-tiny.cfg` has the 24 sections the file lists, in that order, and its [yolo] sections are
layers 16 and 23 of the parity table. In the folded translation the heads sit at those layers, layer
17, a one-layer route, shares the row of layer 13, and the layers read are the network input for
layer 0, layer 13 for layer 17, and layers 19 and 8 for layer 20.

### EveryLayerMatchesItsDumpWithinTolerance

Parameterised over the two models. The folded model is run with every layer that is not a [yolo]
section observed, and each output is compared with its dump within 1e-3 absolute plus 1e-4 relative.
Every layer prints its index, type, extents and largest difference. A failure names the first layer
outside the bound, the layers that layer reads, and how many later layers are outside it too.

## test_darknet_yolov3_tiny_training.cpp

Suites: `Integration_DarknetYolov3TinyTraining`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov3-tiny.cfg`, `yolov3-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov3_tiny`, one image of
YOLOv3-tiny with its deltas and assignments, and `yolov3_tiny_train_step`, one training step of two
images. The box loss is `BoxLoss::SquaredError` and the objectness target is
`ObjectnessTarget::One`. Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. Two differences from the reference
cause them, and neither is a difference of float32 rounding.

The first is the divisor of the batch variance. The library normalises a training batch by the
variance divided by the count and keeps the variance divided by the count less one for the rolling
statistic. The reference normalises by the second. On a 13 by 13 map of two images the count is 338,
so each normalised layer differs in scale by about 0.15 percent, and the heads end 0.0466 and 0.0334
from the dump. A host model of this network in double, written outside the repository and checked
against finite differences, sits 0.0018 and 0.0026 from the dump with the reference's divisor and
0.04664 and 0.03336 with the library's, against 0.04665 and 0.03337 measured here. The deltas and
the updates of the head convolutions inherit the difference.

The second is the reference's backward pass. Its convolution clears the delta of the layer before
it and then writes its own, so a delta that a route had already added to that layer is lost. In this
model the second head's delta through layers 18 and 17 never reaches layer 13. The dumped updates of
the first convolution are therefore not the gradient: the host model is 14.18 from the dumped weight
update with every path and 0.36 with that path removed, against a largest entry of 27.9. The library
keeps every path, and is 15.60 from the dump where the host model with the library's divisor is
15.61. The suite compares no update of the first convolution with the dump for that reason.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov3_tiny`. Its assignments equal the
dumped ones exactly, every entry is within 1e-6 of the dumped delta, and the stored cost agrees to
1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov3_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 7e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 6e-3 absolute plus 1e-4
relative. The largest differences measured were 0.0466 and 0.0037.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 2.5e-3 absolute plus
1.5e-4 relative, any other entry within 2.5e-3; the largest differences measured were 0.0017
and 0.0016. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 7e-3 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 4.2e-3 of that magnitude.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; the start of this section says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov3_tiny_train_step` bring the stored cost of the
Separate model below 0.1 of the first step's.

The stored cost is `HeadCost::stored` summed over the heads: the squared delta entries for a
squared-error head, and for a CIoU head the box weight times the mean of 1 - IoU plus the weighted
squares of the other entries. The summed squares of the delta are not used here, because the box
entries of a CIoU delta do not shrink as the boxes fit: on YOLOv4-tiny their squares stay between
1.0 and 2.8 over 300 steps while the stored cost falls from 15.9 to 0.03.

## test_darknet_yolov3_training.cpp

Suites: `Integration_DarknetYolov3Training`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov3.cfg`, `yolov3.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov3`, one image of YOLOv3 with its
deltas and assignments, and `yolov3_train_step`, one training step of two images. The box loss is
`BoxLoss::SquaredError` and the objectness target is `ObjectnessTarget::One`. Every case is skipped
without its files. The source states that no run has measured the bounds of the step cases yet.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov3`. Its assignments equal the
dumped ones exactly, every entry is within 1e-6 of the dumped delta, and the stored cost agrees to
1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov3_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics, and the [yolo] output computed from it, are
within 1e-3 absolute plus 1e-4 relative of the dump.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 1e-3. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1e-3 of the dump's largest magnitude plus 1e-3 of the
entry.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov3_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.

## test_darknet_yolov4_forward.cpp

Suites: `Integration_DarknetYolov4Forward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv4 from `yolov4.cfg` and `yolov4.weights` in the
directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 608 pixel input with
the reference tensors under `references/`: `yolov4_plain_pool` at a threshold of 0.25 and
`yolov4_plain_pool_thresh_0p01` at 0.01. The model has three heads on grids of 76, 38 and 19, each
with three anchor slots over 80 classes, coded as logits. A head value is held to 1e-3 absolute plus
1e-4 relative, and a box entry or a score of a detection row to 1e-4. Every case is skipped when a
file is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The three published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the logistic of every entry
but the sizes, with x and y stretched by the `scale_x_y` of 1.2, 1.1 and 1.05, and compared with the
dumped layer output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: one at
0.25 and six at 0.01. Suppression runs at 0.45 on the IoU less the distance term to the power 0.6.
Both sides are sorted by class, then by score, and a row matches when its class is equal and its box
entries and score are within 1e-4. Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov4_tiny_forward.cpp

Suites: `Integration_DarknetYolov4TinyForward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv4-tiny from `yolov4-tiny.cfg` and `yolov4-tiny.weights`
in the directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 416 pixel input
with the reference tensors under `references/`: `yolov4_tiny` at a threshold of 0.25 and
`yolov4_tiny_thresh_0p01` at 0.01. The model has two heads on grids of 13 and 26, each with three
anchor slots over 80 classes, coded as logits. A head value is held to 1e-3 absolute plus 1e-4
relative, and a box entry or a score of a detection row to 1e-4. Every case is skipped when a file
is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The two published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the logistic of every entry
but the sizes, with x and y stretched by the `scale_x_y` of 1.05, and compared with the dumped layer
output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: none
at 0.25 and 16 at 0.01. Suppression runs at 0.45 on the IoU less the distance term to the power 0.6.
Both sides are sorted by class, then by score, and a row matches when its class is equal and its box
entries and score are within 1e-4. Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov4_tiny_training.cpp

Suites: `Integration_DarknetYolov4TinyTraining`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov4-tiny.cfg`, `yolov4-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov4_tiny`, one image of
YOLOv4-tiny with its deltas and assignments, and `yolov4_tiny_train_step`, one training step of two
images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is `ObjectnessTarget::One`.
Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. `test_darknet_yolov3_tiny_training.cpp`
above gives the two differences from the reference that cause them.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov4_tiny`. Its assignments equal the
dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of the dumped delta and
every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov4_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 7e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 8e-3 absolute plus 1e-4
relative. The largest differences measured were 0.0481 and 0.0051.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 2e-3; the largest differences measured were 0.00043
and 0.0014. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1.3e-2 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 8.6e-3 of that magnitude.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov4_tiny_train_step` bring the stored cost of the
Separate model below 0.1 of the first step's.

## test_darknet_yolov4_training.cpp

Suites: `Integration_DarknetYolov4Training`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov4.cfg`, `yolov4.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov4_plain_pool`, one image of
YOLOv4 with its deltas and assignments, and `yolov4_train_step`, one training step of two images.
The box loss is `BoxLoss::CompleteIoU` and the objectness target is `ObjectnessTarget::One`. Every
case is skipped without its files. The source states that no run has measured the bounds of the step
cases yet.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov4_plain_pool`. Its assignments
equal the dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of the dumped
delta and every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov4_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics, and the [yolo] output computed from it, are
within 1e-3 absolute plus 1e-4 relative of the dump.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 1e-3. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1e-3 of the dump's largest magnitude plus 1e-3 of the
entry.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov4_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.

## test_darknet_yolov7_forward.cpp

Suites: `Integration_DarknetYolov7Forward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv7 from `yolov7.cfg` and `yolov7.weights` in the
directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 640 pixel input with
the reference tensors under `references/`: `yolov7_plain_pool_thresh_0p01` at a threshold of 0.01,
the only run the table holds for this model. The model has three heads on grids of 80, 40 and 20,
each with three anchor slots over 80 classes, coded as activated values. A head value is held to
1e-3 absolute plus 1e-4 relative, and a box entry or a score of a detection row to 1e-4. Every case
is skipped when a file is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The three published heads of the folded model are compared with the raw heads of the dump, in its
one run, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the head with x and y
stretched to `2 * v - 0.5`, and compared with the dumped layer output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: 9 at
0.01. Suppression runs at 0.45 on the IoU less the distance term to the power 0.6. Both sides are
sorted by class, then by score, and a row matches when its class is equal and its box entries and
score are within 1e-4. Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov7_tiny_forward.cpp

Suites: `Integration_DarknetYolov7TinyForward`, in `sushiai_vision_tests`.

The fixture `DarknetParityTest` imports YOLOv7-tiny from `yolov7-tiny.cfg` and `yolov7-tiny.weights`
in the directory `SA_REFERENCE_DATA_DIR` names and compares its forward pass over a 416 pixel input
with the reference tensors under `references/`: `yolov7_tiny_layers_plain_pool` at a threshold of
0.25 and `yolov7_tiny_plain_pool_thresh_0p01` at 0.01. The model has three heads on grids of 52, 26
and 13, each with three anchor slots over 80 classes, coded as activated values. A head value is
held to 1e-3 absolute plus 1e-4 relative, and a box entry or a score of a detection row to 1e-4.
Every case is skipped when a file is missing.

### TheFoldedHeadsMatchTheDumpedHeads

The three published heads of the folded model are compared with the raw heads of the dump, in both
runs, within the head tolerance.

### TheHostYOLOOutputsMatchTheDumpedLayers

The [yolo] output is computed on the host from the library's heads, as the head with x and y
stretched to `2 * v - 0.5`, and compared with the dumped layer output within the head tolerance.

### TheDetectionsMatchTheDumpedRows

Decode, threshold and suppression over the library's heads are compared with the dumped rows: none
at 0.25 and 12 at 0.01. Suppression runs at 0.45 on the IoU less the distance term to the power 0.6.
Both sides are sorted by class, then by score, and a row matches when its class is equal and its box
entries and score are within 1e-4. Every decision within 1e-6 of a threshold is printed.

### TheSeparateFormComputesTheFoldedHeads

The Separate model, which keeps its own batch-norm rows, is run beside the Folded one. Each of its
heads is within 1e-3 absolute plus 1e-4 relative of the folded head, and the largest difference at
each head is printed.

## test_darknet_yolov7_tiny_training.cpp

Suites: `Integration_DarknetYolov7TinyTraining`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov7-tiny.cfg`, `yolov7-tiny.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov7_tiny_plain_pool_thresh_0p01`,
one image of YOLOv7-tiny with its deltas and assignments, and `yolov7_tiny_train_step`, one training
step of two images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is
`ObjectnessTarget::BestIoU`. Every case is skipped without its files.

The run of 2026-10-12 on this model set the bounds of the step cases: each bound that run exceeded
is now about 1.5 times the largest difference it measured. `test_darknet_yolov3_tiny_training.cpp`
above gives the two differences from the reference that cause them.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov7_tiny_plain_pool_thresh_0p01`. Its
assignments equal the dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of
the dumped delta and every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov7_tiny_train_step` is compared with the delta
that run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics is within 2e-2 absolute plus 1e-4
relative of the dump, and the [yolo] output computed from it within 3e-2 absolute plus 1e-4
relative. The largest differences measured were 0.0128 and 0.0197.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 7.5e-3; the largest differences measured were 0.00094
and 0.0049. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 7e-3 of the dump's largest
magnitude plus 1e-3 of the entry; the largest difference measured was 4.4e-3 of that magnitude.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov7_tiny_train_step` bring the stored cost of the
Separate model below 0.1 of the first step's.

## test_darknet_yolov7_training.cpp

Suites: `Integration_DarknetYolov7Training`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov7.cfg`, `yolov7.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov7_plain_pool_thresh_0p01`, one
image of YOLOv7 with its deltas and assignments, and `yolov7_train_step`, one training step of two
images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is
`ObjectnessTarget::BestIoU`. Every case is skipped without its files. The source states that no run
has measured the bounds of the step cases yet.

### TheTrainingOptionsMatchTheDumpedSettings

The options read from the cfg hold the box loss and the objectness target above, and the weights and
thresholds that the descriptions of the dump hold.

### TheDeltaOfADumpedHeadMatchesDarknet

The delta is computed on the host from each dumped head of `yolov7_plain_pool_thresh_0p01`. Its
assignments equal the dumped ones exactly, a box entry is within 1e-6 absolute plus 1e-4 relative of
the dumped delta and every other entry within 1e-6, and the stored cost agrees to 1e-5 relative.

### TheDeltaOfATwoImageHeadMatchesDarknet

The same computation over each dumped head of `yolov7_train_step` is compared with the delta that
run dumped, under the same bounds, and the stored cost to 1e-5 relative.

### TheHeadsUnderBatchStatisticsMatchDarknet

Each head of the Separate model under batch statistics, and the [yolo] output computed from it, are
within 1e-3 absolute plus 1e-4 relative of the dump.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 1.5e-3 absolute plus
1.5e-4 relative, any other entry within 1e-3. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 1e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1e-3 of the dump's largest magnitude plus 1e-3 of the
entry.

### TheFirstConvolutionReceivesAGradient

The weight, the scale and the shift of the first convolution each have a finite gradient that is
not zero after one step. The dumped updates of that layer are not the gradient and are not
compared; `test_darknet_yolov3_tiny_training.cpp` above says why.

### TheRollingStatisticsAfterOneStepMatchDarknet

The rolling mean and variance of the first batch norm equal the dump before the step, and the mean
is within 1e-5 of the dump after one blend. The variance after the blend is checked twice at the
same 1e-5, each time against a variance the test computes on the host from the batch and the
convolution's weight: ours against the sum in double, and the dump's against the sum taken term by
term in float32. The dump's variance is the float32 sum, which sits 1.0e-4 to 2.3e-4 below the sum
in double on YOLOv3-tiny, so the two are not compared directly.

### TheCostOfOneBatchFallsUnderATenth

200 steps on the batch and the truth of `yolov7_train_step` bring the stored cost of the Separate
model below 0.1 of the first step's.

## test_decoder_from_files.cpp

Suites: `Integration_DecoderModelFile`, `Integration_DecoderFromFiles`, in `sushiai_language_tests`.

The cases build the decoder that `decoder_model.hpp` states as a model file: a vocabulary of 32, a
window of 8, a width of 16 and two blocks, with a head tied to the token table. The graphs are
traced for a batch of 2. `Integration_DecoderModelFile` needs no engine.

### TheFileStatesAnInt32Input

The spec has the input shape `[8]`, the input dtype int32 and a task that is not null.

### TheTiedHeadSharesTheTokenTable

After a probe trace the model has the parameters `0.weight`, `1.weight` and `4.norm.weight` and no
`4.linear.weight`. The parameter element count equals the count worked out by hand: `V * D + T * D +
blocks * (12 * D^2 + 13 * D) + 2 * D`. The state refuses `4.linear.weight` as `InvalidDescription`.

### TheTrainingGraphBindsEveryParameterOnce

The training graph of the next-token objective binds as many slots as the model has parameters, and
exactly one of them is named `0.weight`.

## test_decoder_overfit.cpp

Suites: `Integration_DecoderOverfit`, in `sushiai_language_tests`.

The model is a decoder with a token table of 64 rows, a learned position table over 16 positions,
two causal blocks of width 32 with two heads and a feed-forward width of 64, and a head tied to the
token table. The test writes a token file into the system's temporary directory, reads it back as
one batch of four rows through `TokenDataset`, and removes the file afterwards. Training is AdamW
with a weight decay of 0.01, gradients clipped to a norm of 1, and a cosine schedule from 3e-3 to
3e-4 with 20 warm-up iterations. The parameters start from Xavier values.

### ATwoBlockDecoderOverfitsOneBatch

The dataset holds one batch. The state has no `head.linear.weight`; the token table, the position
table, the head's normalisation weight and a `qkv` bias do not decay, and a `qkv` weight does. Over
400 updates every loss is finite. The first loss is within 0.5 of ln(64), the last is below 0.1 of
the first, and its perplexity is below the first's. The step reports one compile and the optimizer
400 steps.

## test_dense_objectives.cpp

Suites: `Integration_DenseObjectives`.

A `Linear` over three features is trained through `Train::Trainer` on a batch of four with plain
SGD. The inputs and the weights follow fixed integer patterns. The value cases use a rate of 0, so
the logits the step computed can be read back and the loss recomputed on the host from them. A
float32 loss of order one is compared within 1e-5 and a ratio of counts within 1e-12.

### TheBinaryLossAndMetricsEqualTheHostComputation

`BinaryCrossEntropyObjective` over one output. The loss equals the mean of `max(x, 0) - x y + log(1
+ exp(-|x|))` on the host. The step reports three metrics, and `accuracy`, `precision` and `recall`
equal the ratios counted from the signs of the logits. Both signs occur among the logits.

### ALargeLogitGivesAFiniteBinaryLoss

The inputs are scaled by 400, so the largest logit exceeds 100 in magnitude. The loss is finite and
equals the host value within 1e-3.

### TheMultiLabelLossEqualsTheHostComputation

`MultiLabelObjective` over three classes. The loss over the twelve logits equals the host value. The
one metric, `f1`, equals the mean over the classes of `2 hits / (2 hits + wrong)`, with zero for a
class without a hit.

### TheSquaredErrorEqualsTheHostComputation

`MeanSquaredErrorObjective` over two outputs. The loss equals the sum of squared differences over 8,
and the one metric, `mean_absolute_error`, the sum of absolute differences over 8, both within 1e-5.

### FitLowersTheSquaredError

`fit` runs 40 epochs over a dataset of one batch at a rate of 0.05. The report holds 40 steps and
one compile, the last loss is below the first, and `mean_absolute_error` is among the last metrics
and in the report's text.

### FitRefusesLabelsOfAnotherType

A binary objective is fitted to a dataset whose `labels` role is int32 class indices. `fit` is
refused as `Incompatible`, the message names 'classes', 'labels', int32 and float32, and the trainer
has taken zero steps.

## test_dropout.cpp

Suites: `Integration_Dropout`.

The probe runs DROPOUT on an engine over a `[64, 64]` FLOAT32 parameter at a rate of 0.25. Its loss
is `NLL_LOSS` over the masked value with every target at -64, which is the sum of the masked
elements, so the gradient at the input is the mask. A mask is checked against what `dropout_keeps`
answers for the seed and the run. The model cases use `Linear, ReLU, Dropout(0.2), Linear` over 8
features, 32 hidden units and 4 classes with a batch of 16.

### TheOutputIsTheInputTimesAMaskOfZeroOrTheScale

Each mask element is 0 or `1 / 0.75`, and each output is 0 or `x / 0.75`, within 1e-5. The kept
fraction of the 4096 elements is within 0.03 of 0.75, and the mask is the one of run 0.

### TwoStepsOfOneCompiledStepDrawDifferentMasks

The step registers one replay prelude. Two runs give different masks, those of runs 0 and 1, with a
compile count of 1 and a run count of 2.

### TwoFreshStepsWithTheSameSeedAgreeStepForStep

Over three runs, two fresh steps with one seed give the same output bytes, and a third step with the
seed plus one gives different bytes.

### TheGradientAtTheInputIsTheMaskOfThatStep

On each of two runs the gradient at the input equals the mask of that run within 1e-5, and the loss
equals the sum of the outputs to 1e-4 relative.

### AStepWithoutDropoutRegistersNoPrelude

A graph that only scales a `[2, 3]` input by 2 compiles to a step with zero replay preludes, and the
step doubles its input.

### AnEvaluationTraceOfAModelEmitsNoDropoutNode

The evaluation graph of the model holds no DROPOUT node and its probe graph holds one. That node
carries the rate 0.2 and the seed `dropout_site_seed` gives for the graph seed 7 and site 0.

### AModelWithDropoutTrainsThroughTheTrainer

The training graph holds at least one DROPOUT node. `Train::Trainer` fits the model with AdamW at
1e-2 on a synthetic dataset of 8 batches for 10 epochs. The report holds 80 steps and one compile,
and the last loss is finite and below the first.

## test_engine_smoke.cpp

Suites: `Integration_EngineSmoke`.

### AXPYThroughSushiBLAS

This is the P6 "consumer-shaped example" for AI-0. It proves that the two-sibling D3 dependency
chain, SushiRuntime and SushiBLAS both linked through SushiAI, compiles, links and runs a real
operation end to end. It does not stand in for the autograd operation coverage of AI-1.

## test_evaluation_step.cpp

Suites: `Integration_EvaluationStep`.

The evaluation step replays a forward-only graph over a `NN::ModelState`. The cases check that it
compiles once, that it reads the weights a training step wrote, that a probe run leaves the
batch norm buffers alone, and the two refusals it makes.

### TheStepCompilesOnce

The step is made from an MLP's evaluation graph. `compile_count()` is 1 after `make()` and still
1 after three runs.


### AnEvaluationStepSeesAWeightTheTrainingStepJustMoved

A training pool and an evaluation pool attach to one state. The evaluation output for fixed
inputs is read before and after one training step on the synthetic dataset. The two byte
sequences differ. The test notes that the default policy starts the classifier at zero.

### AProbeRunLeavesBothRunningVectorsUnchanged

The evaluation graph of a batch norm gives its buffers their leaves, and its probe graph is run
once. The bytes of the running mean and the running variance are the same before and after.

### AnInputOfAnotherShapeIsRefusedWithBothShapes

A step compiled for a [4, 6] input refuses a [2, 6] input. The refusal is `Incompatible`, and its
message names "inputs", "[2, 6]" and "[4, 6]".

### AnUnknownOutputNameListsTheNames

`output("boxes")` is refused as `InvalidDescription`. The message names "boxes" and the single
port name, `NN::K_SINGLE_PORT_NAME`.

## test_forward.cpp

Suites: `Integration_Forward`, `Integration_ForwardDeathTest`.

Every lowered operation is checked against values worked out by hand. This tier catches a
transposed gemm, a bias broadcast down the wrong axis, and a log-softmax whose stabilising
shift did not cancel. All three produce plausible numbers and no error.

### LinearLayerMatchesHandComputedValues

`X * W^T + b` for a `[2, 2]` input and a `[3, 2]` weight gives `{-1, 1.5, 4, 1, 3.5, 8}`, and the
ReLU after it turns the -1 into 0, both within 1e-5.

### SumRowsProducesColumnSums

`sum_rows` of the `[3, 2]` value `{1, 2, 3, 4, 5, 6}` gives 9 and 12.

### ReshapeReinterpretsRowMajorLinearOrder

A `[3, 4]` value holding 0 to 11 reshaped to `[2, 6]` still reads 0 to 11 in order.

### LayoutConvertPermutesChannelsToTheFront

In NCHW the channels are outermost of the spatial block. For each c the two W positions follow
in order: (c=0, w=0..1) = 0, 3; (c=1, w=0..1) = 1, 4; (c=2, w=0..1) = 2, 5.

### ElementwiseAndScaleMatchTheirDefinitions

`add`, `sub`, `mul`, a `scale` by -2.5 and `exp` over two `[2, 2]` values equal the numbers worked
out by hand, within 1e-5.

### ActivationsMatchTheirDefinitionsAndLeaveTheInputAlone

Out-of-place semantics matter. The backward rules for relu and gelu read the forward input, so
an in-place activation would hand them the output.

The leaky_relu values are computed by hand at slope 0.1, with inputs on both sides of zero so
that both branches run: -2 gives -0.2, -0.5 gives -0.05, 0.5 gives 0.5, 2 gives 2.

### LogSoftmaxRowsSumToOneAfterExponentiating

The second row is the first offset by a large constant. Log-softmax must be invariant to the
offset, and the max shift is what makes it so.

### CrossEntropyOnUniformLogitsIsExactlyLogC

With `[4, 5]` logits of zero and one-hot targets, the loss is ln(5) within 1e-5.

### CrossEntropyMatchesAHandComputedSingleRow

For the logits `{1, 2, 3}` and the target class 2, the loss is `log(1 + exp(-1) + exp(-2))` within
1e-5.

### AccumulateAddsInPlaceAndAliasesItsResult

`accumulate` of `{1, 2, 3, 4}` and `{10, 20, 30, 40}` gives `{11, 22, 33, 44}`, and the accumulator
read back holds the same values as the result.

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

This case closes the gap that let a dropped layout tag hide. Every other layout test stops at the
tag. The numerical claim behind them is that a tag reaching `lower_layout_convert`
(`modules/core/graph/source/lowering/shape.cpp`) intact makes a transpose happen, and nothing tested
that half.

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

K_MAXPOOL2D is the first operation whose results are not all one element type. Beside the pooled
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

### AvgPool2dDividesByTheFullWindowArea

A 2x2 average pool over the `[1, 2, 2, 1]` values 0, 1, 2 and 3 gives one value, 1.5.

### Upsample2dReplicatesEachPixelOverItsBlock

One pixel of value 5 upsampled by 2 in each direction becomes four values of 5.

### SpatialMeanAveragesEveryHWPlane

The spatial mean of the `[1, 2, 2, 1]` values 0, 1, 2 and 3 is one value, 1.5.

### ConcatChannelsPlacesEachOperandAtItsOwnOffset

Two one-channel values holding 10 and 20 concatenate to `{10, 20}`.

### SliceChannelsExtractsTheMiddleRangeUnchanged

Two channels from channel 1 of `{1, 2, 3, 4}` are `{2, 3}`.

### ChannelAffineScalesAndShiftsPerChannelNotPerRow

The input is N=1, H=1, W=2, C=2. Two pixels sharing one pair of parameters separate a
per-channel broadcast from a per-row one: K_BIAS_ADD's broadcast would give the second pixel the
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

### AnUnboundLeafIsRefusedRatherThanFilledWithGarbage

Compiling a graph whose input leaf has no tensor in the pool is a fatal stop. The message contains
"Leaf value", 'x' and "has no tensor bound".

### AnUnimplementedOperationNamesItself

A node with an operation identity that has no lowering makes the compile a fatal stop whose message
contains "No lowering for operation".

## test_gather_rows.cpp

Suites: `Integration_GatherRows`.

The file runs GATHER_ROWS, SCATTER_ADD_ROWS and the `Embedding` layer on an engine. The tables hold
the dyadic values 0.25, 0.5, 0.75 and so on, whose copies and sums are exact, so results are
compared with a host loop at a tolerance of 0. Indices are written into INT32 leaves.

### GatherCopiesTheRowsItsIndexNames

A `[2, 3]` index over a `[5, 3]` table, with row 2 named three times, gives the rows a host loop
copies.

### AReplayReadsTheIndexOfItsOwnRun

One compiled step over a FLOAT64 `[6, 2]` table is run with two different indices of four. Each run
returns the rows of the index written before it.

### ScatterAddSumsTheRowsOfARepeatedIndex

Six FLOAT64 rows are scattered into five table rows with the index `{3, 1, 3, 3, 0, 1}`, and the
step is run twice. The result equals the host sums, and rows 2 and 4, which no index names, are
zero.

### TheTableGradientIsBitIdenticalAcrossReplays

A FLOAT32 `[13, 8]` table is gathered with 256 indices, so each row is named about twenty times, and
the loss is differentiated at the table. The bytes of the gradient are the same after each of four
runs.

### AnEmbeddingYieldsTheRowsOfItsTable

An `Embedding` of 7 rows and width 4 over a `[2, 3]` batch of ids has the output shape `[2, 3, 4]`,
and its output holds the table rows the ids name.

## test_generation.cpp

Suites: `Integration_Generation`, in `sushiai_language_tests`.

The fixture builds the decoder of `decoder_model.hpp`, with a window of 8 and a vocabulary of 32,
initialises its state, compiles the `[1, 8]` forward pass and builds a `WindowGenerator` over it.

### TheGeneratorReadsTheExtentsOfTheStep

The generator reports a window of 8 and a vocabulary of 32.

### ARunEqualsOneTokenAtATimeOverTheGrownPrompt

Twelve greedy tokens after a prompt of three ids equal twelve runs of one token each over the prompt
grown by the tokens so far. Every token lies in 0 to 31, and the step reports one compile.

### ASeedFixesADrawnContinuation

A `TemperatureSampler` at 1.0 with two random generators of one seed gives the same twelve tokens
twice.

### APromptOrACountThatDoesNotFitIsRefused

An empty prompt, the id 32 and a count of zero tokens are each refused as `InvalidDescription`; the
messages contain "no id", "0 to 31" and "0 tokens". Nothing is generated.

## test_head_delta_loss.cpp

Suites: `Integration_HeadDeltaLoss`, in `sushiai_vision_tests`.

The graph holds a `[2, 3, 3, 7]` float32 parameter, the head, and an input leaf of the same shape,
the delta. `emit_head_delta_loss` joins them with a scale of 0.5, and the loss is differentiated at
the head. Both leaves are filled from seeded host noise, with every third delta set to zero. The
expectations start from the values rounded to float32.

### TheGradientAtTheHeadIsMinusScaleTimesDelta

Each element of the gradient equals `-0.5 * delta` within 1e-6.

### TheValueIsMinusScaleTimesTheSumOfProducts

The loss is one value and equals `-0.5 * sum(delta * head)`, summed in double on the host, within
1e-4.

## test_import_weights.cpp

Suites: `Unit_RenameRule`, `Integration_ImportWeights`.

The four `Unit_RenameRule` cases run the prefix rule with no device. They share this file with the
integration cases, which load a checkpoint written under foreign names into a `NN::ModelState`.

### RewritesAMatchingPrefix

The rule `model.` with the empty prefix turns `model.0.weight` into `0.weight`.

### LeavesANameNoRuleMatches

A name that no rule's prefix starts is returned unchanged.

### TheFirstMatchingRuleWins

With `model.` to `first.` listed before `model.0` to `second`, `model.0.weight` becomes
`first.0.weight`. The order decides, and the longer prefix does not win.

### AnEmptyToPrefixStripsThePrefix

`model.1.bias` becomes `1.bias`.

### ARuleStripsAForeignPrefixAndEveryRecordLands

A checkpoint holds `model.0.weight` and `model.0.bias`, written from a source state. A target state
holds other values. Loading with the rule `model.` to the empty prefix leaves both target tensors
with the bytes of the source.

### WithoutTheRuleTheFirstRecordIsRefused

The same file, loaded with no rules, is refused as `Incompatible`. The message contains
`'model.0.weight'`.

### ARecordTheStateLacksIsRefusedUnderBothNames

A record named `model.9.weight` has no entry once renamed. The refusal is `Incompatible`, and its
message contains "holds 'model.9.weight'" and "renames to '9.weight'".

### AnEntryTheFileLacksIsRefusedAndNamed

The file holds only `model.0.weight`. The refusal is `Incompatible`, and its message contains
"does not supply" and "'0.bias'".

### AShapeMismatchIsRefusedWithBothShapes

A record shaped [3, 2] carries the six elements of a [2, 3] entry. The refusal is `Incompatible`,
and its message names both shapes. The shapes are compared, not the element counts.

### AMissingFileIsFileNotFound

A path that was never written is refused as `FileNotFound`, and the message contains
"could not open".

## test_init_state.cpp

Suites: `Integration_InitState`.

The cases check the fill that `NN::initialize()` writes over a `NN::ModelState`: the constants of a
batch norm, the zeroed final layer, and the determinism of the default seed.

### AScaleIsOneAShiftIsZeroAndABufferHoldsItsInitialValue

A training trace of a batch norm named `bn` is initialised. Afterwards `bn.weight` holds only ones,
`bn.bias` and `bn.running_mean` hold only zeros, and `bn.running_var` holds only ones.

### ZeroFinalLayerZeroesOnlyTheLastFanScaledEntry

A convolution named `conv`, followed by a batch norm named `bn`, is initialised with the default
policy. Every element of `conv.weight` is zero, and `bn.weight` is still one.

### WithoutZeroFinalLayerTheLastWeightIsNotZero

An MLP initialised with `zero_final_layer` set to false has at least one nonzero element in
`2.weight`.

### TheSameSeedGivesTheSameBits

Two MLPs of the same architecture are each created as a state and initialised with the default
policy. Every entry has the same bytes in both states.

### ANormalFillHasTheDeviationItStates

The table of an `Embedding` of 64 rows and width 32 is initialised with the default policy. Over its
2048 values the mean is within 0.0025 of zero and the deviation within 0.002 of
`NN::K_EMBEDDING_DEVIATION`.

### ANormalFillOfNoDeviationIsRefused

An `Embedding` whose weight asks for `ParameterInit::normal(0.0)` is refused by `NN::initialize` as
`InvalidDescription`, with "positive, finite deviation" in the message.

## test_layer_norm.cpp

Suites: `Integration_LayerNorm`.

The file runs LAYER_NORM and the `LayerNorm` layer on an engine with an epsilon of 1e-5 and compares
with a host loop in double: FLOAT32 results within 2e-5 and FLOAT64 results within 1e-12. The
gradient cases normalise a FLOAT64 `[2, 3, 4]` value, reshape it to `[6, 4]`, and compare the
analytic gradient of one leaf with central differences while the other leaves hold seeded noise.

### EachRowIsNormalisedByItsOwnStatistics

A FLOAT32 `[2, 3, 5]` value with a gain and a bias of five elements is normalised and the step is
run twice. The output equals the host's rows.

### TheSavedStatisticsAreTheRowMeanAndInverseDeviation

For the FLOAT64 rows `{1, 2, 3, 4}` and `{-4, -2, 0, 2}` the saved means are 2.5 and -1, and the
saved inverse deviations are `1 / sqrt(1.25 + eps)` and `1 / sqrt(5 + eps)`.

### TheInputGradientMatchesCentralDifferences

The gradient at the `[2, 3, 4]` input.

### TheGainGradientMatchesCentralDifferences

The gradient at the gain of four elements.

### TheBiasGradientMatchesCentralDifferences

The gradient at the bias of four elements.

### ALayerNormalisesWithItsGainAndItsBias

A `LayerNorm` of width 4 over a FLOAT32 `[2, 3, 4]` value keeps the shape, and with its weight and
bias written from noise its output equals the host's rows.

## test_lowered_operations.cpp

Suites: `Integration_LoweredOperations`.

`tests/CMakeLists.txt` does not list this file yet, so `sushiai_tests` does not hold its cases and
no run has executed them.

This file runs fourteen lowerings through a compiled step and compares the result with values
computed on the host: twelve forward cases on FLOAT32 and two gradient cases on FLOAT64. Each case
builds a graph with one operation, compiles it with `CompiledStep::compile`, runs it, and reads the
result back. The gradient cases reach the gradient through `Autograd::differentiate`, on a
negative log-likelihood root, and compare it with the closed-form derivative.

The forward cases allow an absolute error of 1e-6 and a relative error of 1e-5. The gradient cases
allow 1e-12 and 1e-9. These bounds are estimates, and nothing has checked them against the device
kernels.

Before this file, the integration and regression tiers built none of these operations. The unit
tier builds them as nodes and never compiles them. The fixture is a local copy of `OneShot` from
`test_forward.cpp`.

### DivDividesElementwise

DIV writes the element-wise quotient of two FLOAT32 rows.

### MinimumTakesTheSmallerOperand

MINIMUM writes the smaller of each pair of elements.

### MaximumTakesTheLargerOperand

MAXIMUM writes the larger of each pair of elements.

### AbsReturnsTheMagnitude

ABS writes the magnitude of each element.

### AtanReturnsTheArctangent

ATAN writes the arctangent of each element.

### LessEqualWritesOneForLessAndForEqual

LESS_EQUAL writes 1 where the left operand is less than or equal to the right, so the pairs that
are equal count, and 0 elsewhere.

### SelectTakesEachBranchWhereTheMaskSaysSo

SELECT takes the first branch where the mask is 1 and the second branch where it is 0.

### ConstantFillsEveryElementWithItsValue

CONSTANT fills a tensor with its value, and CONSTANT_LIKE fills one with the shape of another
operand. The step runs twice, and the outputs are checked after each run.

### SiLUMatchesItsDefinition

SILU writes x * sigmoid(x) for each element.

### SoftplusMatchesItsDefinition

SOFTPLUS writes log(1 + exp(x)) for each element.

### SiLUBackwardMatchesTheClosedFormDerivative

SILU_BACKWARD, reached through `Autograd::differentiate` on a FLOAT64 graph, matches the closed-form
derivative of SiLU.

### SoftplusBackwardMatchesTheClosedFormDerivative

SOFTPLUS_BACKWARD, reached through `Autograd::differentiate` on a FLOAT64 graph, matches the
closed-form derivative of softplus.

### StopGradientCopiesItsOperand

STOP_GRADIENT copies its operand forward unchanged. The case checks only that forward copy. It does
not check that the gradient is blocked on the device.

### BatchNormFoldWritesTheScaleAndTheShift

BATCHNORM_FOLD writes `scale = gamma / sqrt(variance + epsilon)` and
`shift = beta - mean * scale`. One of the variances is zero, so epsilon alone sets that element's
denominator.

### DepthwiseConv2dMatchesTheWindowSums

A 3x3 depthwise convolution with a pad of 1 runs over a `[1, 4, 4, 2]` image of integer values with
integer taps. Each output equals the sum of the taps whose input pixel lies inside the image,
computed by a host loop, within 1e-6.

## test_max_pool_trailing.cpp

Suites: `Integration_MaxPoolTrailing`.

A `MaxPool2d` with a 2x2 window, a stride of one and one trailing row and column of padding pools a
`[1, 3, 3, 1]` input into nine values. Each is compared with the maximum of its truncated window
within 1e-6.

### TrailingWindowsTakeTheMaximaOfTheirTruncatedTaps

The input `[[5, -1, 2], [-3, 8, 0], [4, -7, 6]]` pools to `[[8, 8, 2], [8, 8, 6], [4, 6, 6]]`.

### AllNegativeTrailingWindowsKeepTheirTrueMaximum

The input runs from -18 to -10. The result is `[[-14, -13, -13], [-11, -10, -10], [-11, -10, -10]]`;
a pad that took part in the maximum as zero would show as 0.

## test_mish_values.cpp

Suites: `Integration_MishValues`.

The `Mish` layer runs over one FLOAT32 row on the engine. Each result is compared with `x *
tanh(log1p(exp(x)))`, computed in double, within 1e-6 of the exact value's magnitude. A zero must be
matched exactly.

### MatchesTheExactFormAtTheAcceptancePoints

The inputs are -20, -1, 0, 1 and 20.

### SweepMatchesTheExactFormAndStatesTheDarknetDifference

The inputs run from -30 to 30 in steps of 0.5, 121 values. The engine's results match the exact
form. The test also evaluates, on the host in double, the form with a softplus cut-off at a
magnitude of 20, records its largest difference from the exact form as the property
`darknet_cut_off_largest_difference`, and expects that difference to be at most 1e-12.

## test_mixed_precision.cpp

Suites: `Integration_MixedPrecision`.

This is the device half of AI-4. `modules/core/autograd/tests/test_precision.cpp` pins the graph
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

### AModelWithFP16ProductsStillLearns

Correctness is the one claim the mixed-precision change can make on this hardware. It is not
faster here and cannot be.

A zeroed classifier starts at a loss of ln(4). A last loss below half of that shows a model
that learned something and did not merely run.

### TheMastersAreFourByteBuffersAndTheCopiesAreTwo

The IR says FLOAT32 and HALF, and the test checks that the tensors underneath say the same.
The two could disagree. The test reads each parameter and gradient slot and requires FLOAT32, and
it reads each cast destination the pool holds and requires HALF. A planner that sized a slot from
the wrong dtype would hand out a view that overruns its neighbour.

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

## test_model_state.cpp

Suites: `Integration_ModelState`, `Integration_ModelStateDeathTest`.

`NN::ModelState` lists a model's parameters and buffers in walk order, finds them by name or by
identity, refuses two entries with one name, stops on an entry with no identity, and attaches its
storage to a graph and a pool. The cases pin each of those.

### CreateListsParametersInWalkOrder

A traced `Sequential` of Linear, GELU and Linear lists four parameters in order: `0.weight` [8, 6],
`0.bias` [8], `2.weight` [4, 8] and `2.bias` [4].

### CreateListsBuffersAfterATrainingTrace

A `ModelGraph` holding a batch norm named `bn`, traced in training mode, lists `bn.weight` and
`bn.bias` as parameters. It then lists `bn.running_mean`, with constant initial value 0, and
`bn.running_var`, with constant initial value 1, as buffers.

### AnUntracedModelGivesAnEmptyState

A model that was never traced gives a valid state with no entries.

### APrefixIsJoinedToEveryName

A prefix of "model" passed to `create()` makes the first entry `model.0.weight`.

### TwoEntriesWithOneNameAreRefused

`from_entries` with two entries named "shared" is refused as `InvalidDescription`, and the message
names "shared".

### AnEntryWithNoIdentityIsFatal

`from_entries` with an entry whose identity is 0 stops the process, as the death test expects. The
message contains "'orphan'", the entry's name, and "no identity".

### TensorByNameAndByStateAreTheSameStorage

`tensor()` called with the name and with the identity of one entry returns the same tensor object.

### AnUnknownNameIsRefusedAndNamed

`tensor("missing.weight")` is refused as `InvalidDescription`. The message contains the name and
"4 entries".

### AttachBindsEveryStateLeafToTheStatesStorage

`attach()` records four state leaves on the graph. For each leaf, the pool tensor's data pointer is
the same as the state tensor's.

### AttachRefusesAGraphOfAnotherModel

A graph traced from a second instance of the same architecture is refused by `attach()` as
`Incompatible`.

### TwoPoolsAttachedToOneStateShareStorage

Two pools attached to one state read the same numbers. A value written through a leaf of the first
pool is read back through the second.

## test_moe.cpp

Suites: `Integration_Moe`, in `sushiai_language_tests`.

The block cases run a FLOAT64 mixture-of-experts block of three experts with a hidden width of 3
over a `[1, 3, 4]` sequence, with every parameter filled from seeded noise. The mixture is compared
with a host computation within 1e-10. The gradient cases take the cross-entropy of the block's rows
plus 0.5 times the balance term and check it with the full-Jacobian central difference. The training
cases build a decoder of two blocks of four experts from a model file in the test, with a window of
8 and a vocabulary of 32, and fit one batch of four rows for 200 updates with AdamW at 3e-3.

### ADenseMixtureEqualsTheHostComputation

With every expert kept, the output equals the gate-weighted sum of the experts the host computes.

### AMixtureOfTheTopExpertsEqualsTheHostComputation

With one expert kept and with two, the output equals the host computation over the renormalised
gate.

### ADenseMixtureDifferentiatesAtItsRouter

The gradient of the fit and the balance at the weight of the router.

### ADenseMixtureDifferentiatesAtItsInput

The gradient of the fit and the balance at the sequence the block reads.

### AMixtureOfTheTopExpertDifferentiatesAtItsRouter

A block that keeps one expert is differentiated at the router weight. The graph holds no
MAXPOOL2D_BACKWARD node, and the gradient matches central differences.

### ADecoderOfExpertsLowersItsLossWithoutTheBalance

Trained with the `next_token` objective alone, the report holds 200 steps and one compile, and the
last loss is below half the first. The last metrics hold no `expert_usage_min`.

### TheBalanceTermKeepsEveryExpertInUse

Trained with a `sum` objective that adds the balance term, the report again holds 200 steps, one
compile and a last loss below half the first. `expert_usage_min` is above 0.10 and at most 0.25 plus
1e-5, `expert_usage_max` is at least 0.25 minus 1e-5, and the report's text names
`expert_usage_min`.

## test_next_token_objective.cpp

Suites: `Integration_NextTokenObjective`, in `sushiai_language_tests`.

The model is an `Embedding` of five rows and width five, so the logits of a position are the table
row of its token. The table holds `0.3 r - 0.2 c + 0.1 r c`. The batch is two rows of three tokens,
in which token 3 occurs twice with two different labels. The step runs with SGD at a rate of 0, so
the table can be read as it was written.

### TheLossAndItsGradientEqualTheHostComputation

The graph binds one parameter slot. After one step the loss equals the mean of `-log p(label)` over
the six positions, and each element of the table's gradient equals `(softmax - one_hot) / 6` summed
by table row, both within 1e-5.

## test_next_token_trainer.cpp

Suites: `Integration_NextTokenTrainer`, in `sushiai_language_tests`.

The fixture fits a bigram table, an `Embedding` of five rows and width five, to one batch of two
rows of three tokens through `Train::Trainer` and the next-token objective's own feed, for 30
epochs.

### FitLowersTheLossAndReportsPerplexity

The report holds 30 steps and one compile, and the last loss is below the first. The last metrics
hold one entry, `perplexity`, which is not a fraction and equals `exp` of the last loss within 1e-9.
The report's text names it.

### ALabelOutsideTheVocabularyStopsTheFirstStep

A label of 9 with five tokens is refused as `Incompatible`. The message contains "position 3", "is
9" and "holds 5 tokens".

## test_pixel_objectives.cpp

Suites: `Integration_PixelObjectives`, in `sushiai_vision_tests`.

The logits are one 2 by 3 map of four classes. The model, `LogitMap`, adds a learnable map to an
input of zeros, so the gradient of the map is the gradient at the logits. The mask ignores two of
the six pixels with the value 255 and holds no pixel of class 3. One step runs with SGD at a rate of
0, and the loss and the gradient are compared with a host computation in double within 1e-5.

### TheCrossEntropyAndItsGradientEqualTheHostComputation

The loss equals the mean of `-log p(label)` over the four kept pixels. The gradient has 24 elements:
`(softmax - one_hot) / 4` on a kept pixel and zero on an ignored one.

### TheDiceLossAndItsGradientEqualTheHostComputation

The loss equals the mean over the present classes of `1 - 2 I / (P + Y + eps)`. The gradient is
compared with a central difference of that host loss with a step of 1e-4.

### AWeightedSumOfBothLossesEqualsTheHostComputation

A `SumObjective` of the cross entropy at weight 1 and the Dice loss at weight 0.5. The loss equals
the weighted sum on the host, and the gradient its central difference.

### AMaskWithNoKeptPixelGivesZero

With all six pixels ignored, the cross-entropy loss and every gradient element are zero within 1e-5.

### ALabelOutsideTheClassesStopsTheFeed

A mask that holds 4 at pixel 2 is refused as `Incompatible`. The message contains "holds 4 at pixel
2" and "below 4".

## test_program_runs.cpp

Suites: `Integration_ProgramRuns`, in `sushiai_program_tests`.

The cases exercise the tables of `applications/programs`: the run table and its fallback, the report
table, and the start of a run. The pieces of a run are an MLP of 6 features, 12 hidden units and 4
classes, its classifier graph for a batch of 8, SGD at a rate of 0.1 and a synthetic dataset of 8
batches. A classification run takes 2 epochs.

### TheRunTablePicksARowOrTheFallback

A second row for 'anchor' and a second fallback are each refused as `InvalidDescription` with
"already registered". Without a fallback, the type 'cross_entropy' is refused with a message that
names it and 'anchor'. With both registered, 'anchor' runs the row's stub, which reports a compile
count of 7, and 'cross_entropy' runs the fallback's, which reports 9.

### AClassificationRunTrainsThroughTheFallback

`register_runs` sets a fallback. A 'cross_entropy' run through it reports one compile, and the
optimizer has taken 16 steps. The run prints two losses, and the last is below the first.

### TheReportTableRefusesAnUnknownTask

A second row for 'classification' is refused with "already registered". A report for
'classification' runs. A report for 'pose' is refused as `InvalidDescription`, and the message names
"model.json", 'pose' and 'classification'.

### ARunStartsFromWhatItsFileStates

A run is trained and saved with its run state to a file in the temporary directory. For a spec
without files `describe_start` returns "initialised" and the optimizer stays at step 0. With
`weights` set it returns the path, and the step count stays 0. With `resume` set it returns "resumed
from " and the path, and the step count equals the trained run's 16.

## test_registries.cpp

Suites: `Integration_Registries`.

Two registries need an Engine to exercise: datasets allocate device tensors, and optimizers
allocate their state buffers. The layer and objective registries are host-lane and live in the
unit tier.

Rings is tested on its defining property, that it is not linearly separable, and not only on
its shape. The dataset exists because a single Linear layer cannot solve it, so a test should
fail if it ever becomes solvable.

The fixture fills a `Data::DatasetRegistry` through `Data::register_datasets`, which adds the
"rings" row the dataset cases name. The optimizer cases call `Optim::make_optimizer` with the types
"adamw" and "sgd".

### RingsIsBalancedAndCarriesItsDistractors

Labels are assigned round-robin, so every batch is exactly balanced and accuracy needs no
class weighting to read correctly.

### RingsPublishesItsLabelsAsOneIndexPerSample

With three classes and a batch of eight, the dataset describes one target: the labels role, one
int32 per sample, over 3 classes. The batch holds one target tensor of rank 1 and length 8 whose
labels are `{0, 1, 2, 0, 1, 2, 0, 1}`.

### RingsReportItsSampleShape

The default options add four distractors, so a sample has shape [6]. A batch of eight samples is
rank 2 with shape [8, 6].

### RingsPlacesEachClassInItsOwnRadiusBand

The whole problem rests on this property. The class is a function of x^2 + y^2, which is
degree 2 in the inputs and therefore not a half-space. If a change made the radius bands
overlap or the labels track something linear, this test fails, where otherwise the accuracy
would rise in some other test and nobody would see why.

### TheDatasetRegistryBuildsRingsFromAConfigValue

`make_dataset` for "rings" with a batch size of 16, 2 batches, 3 classes and one distractor returns
a dataset with those extents and the sample shape `[3]`.

### TheSameSeedGivesTheSameData

The dataset draws from a host RNG and not from the engine's Philox stream, so reproducibility
is a property of the seed alone. The device stream mixes in a monotonic offset counter, which
is why `dataset.hpp` warns about `set_seed`.

### AnUnknownDatasetTypeListsEveryTypeThisBuildHas

The type "coco" is refused as `InvalidDescription`. The message contains "coco", "dataset type" and
'rings'.

### AnUnknownDatasetKeyIsRejectedByItsOwnFactory

The key "batch_sizes" raises an error whose message contains the context "data", 'batch_sizes' and
'batch_size'.

### ANonPositiveRingsExtentIsRefused

A batch size of 0 is refused as `InvalidDescription` with "positive" in the message.

### TheOptimizerRegistryBuildsBothRulesFromAConfigValue

The test passes one real parameter and gradient pair, so the optimizers allocate their state
against something and not against an empty list.

### AnUnknownOptimizerTypeListsEveryTypeThisBuildHas

The type "lion" raises an error whose message contains "lion", "optimizer type", 'adamw' and 'sgd'.

### AnUnknownOptimizerKeyIsRejectedByItsOwnFactory

"momentum" is SGD's key, not AdamW's. Asking AdamW for it must fail; it must not be a line
that does nothing.

### AConfigurationCannotReachTheReferenceUpdateKernel

`UpdateKernel` selects the spelling a byte-for-byte regression test compares the fused kernel
against. SEPARATE also cannot join the compiled plan. A config file that could set it would
be able to change what a run is, so "kernel" is not an accepted key.

## test_rematerialise.cpp

Suites: `Integration_Rematerialise`.

The file covers the half of the K_IM2COL rematerialisation pass that the host-lane tier cannot
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

The [1, 5, 4, 2] image is not square on purpose. K_IM2COL's output extents live in attribute
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

Suites: `Integration_Resilience`, `Integration_ResilienceDeathTest`.

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

### AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch

A `[4, 3]` parameter with a `[4, 2]` gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.weight", "12 elements" and "gradient has 8".

### AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType

A float32 parameter with a float64 gradient is refused by `SGD::make` as `InvalidDescription`. The
message contains "0.bias", "float32" and "float64".

### AnOptimizerRefusesADTypeItCannotServeAndExplainsWhy

Complex is not a training dtype at all. Half is a real float, and it is refused for a reason
the message has to carry: the refusal exists to prevent a model that compiles and trains
worse.

### AdamWNamesTheHyperparameterThatIsOutOfRange

`AdamW::make` refuses a `beta1` of 1.0 and a `beta2` of -0.5, each with its name and "[0, 1)" in the
message, and an `eps` of 0 with "eps must be positive". All three are `InvalidDescription`.

### ADisconnectedParameterNamesItselfAndSaysWhatToDo

A binding named "ghost" with no gradient value is appended to a classifier's parameters.
`bind_parameters` refuses it as `InvalidDescription`, and the message contains 'ghost', "receives no
gradient" and "no optimizer could ever move it".

### AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage

The library has no `zero_grad()`, by design. Every gradient buffer in a step is written and
not accumulated into (see `optimizer.hpp`). That leaves one window in which a gradient has
never been written: between `bind_parameters()` and the first replay. Stepping in that window
would feed whatever the allocator last had in that memory straight into the weights. Nothing
downstream catches it. The parameters become garbage or NaN, and the loss never recovers.

### TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep

A step's buffers are addresses baked into a compiled plan. A batch of the wrong size cannot be
accommodated, only reported. The refusal names both shapes, [4, 6] and [8, 6].

### FitNamesTheDatasetDimensionThatDisagrees

A trainer compiled for a batch of 8 and 4 classes refuses three datasets. One without batches is
`InvalidDescription` with "The dataset holds no batches". A batch size of 4 is `Incompatible` with
"batch size is 4" and "compiled for 8". Five classes are `Incompatible` with "5 classes" and
"compiled for 4".

### TheSyntheticDatasetNamesEveryExtentWhenOneIsNotPositive

`SyntheticClassification::make` refuses zero batches, a batch size of zero and zero classes as
`InvalidDescription`. Each message contains "positive extents" and the extent that is zero, as
"batches=0", "batch=0" or "classes=0".

### AllocationRefusesAnEmptyOrUnrepresentableTensor

Signed overflow in the extent product is undefined behaviour, and a compiler may assume it
away along with any check written after it. The overflow is detected before the multiply, so
the diagnostic is about the shape and not about what the overflow left behind.

### AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel

The request is a petabyte, which no allocator serves. What matters is what the caller is told:
the shape and element type it asked for. A bare byte count from four frames down would not
help, and neither would a null pointer for a kernel to dereference.

### ASingleSampleWithASingleFeatureTrains

One `Linear` over a batch of one sample with one feature and three classes. The first loss is ln(3)
within 1e-5, the loss stays finite over eight more steps at a rate of 0.5, and the weight and its
gradient are finite afterwards.

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

### TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt

The case runs for n = 64 and n = 65, with n as the batch size, the feature count and the hidden
width. Over four steps of SGD at a rate of 0.1 every loss is finite, and so is every parameter and
every gradient afterwards.

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
errors, in `tests/regression/test_log_softmax_stability.cpp`.

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

Every `SushiBLAS::Storage` allocates through the runtime, which records each allocation, so
`runtime.advanced().live_allocation_count()` is the number of live allocations. Whether the last
iteration leaked a buffer is then a number the test can read, where otherwise somebody would have
to watch Task Manager. One leaked Storage moves the count by exactly one.

The baseline is taken after the first iteration and not before it. A one-time lazy cache
inside the runtime is tolerated; a per-iteration leak is not.

### BatchNormUpdateStatsIsNotAnIROperationAndSaysSo

CV-7 keeps a BatchNorm's running statistics out of the IR. They are recorded into the training step
beside the graph, the same way `modules/core/optim/source/` emits K_SGD_UPDATE and K_ADAMW_UPDATE
directly. K_BATCHNORM_UPDATE_STATS is therefore one of the three identities listed in `K_UNLOWERED`
(`modules/core/graph/include/SushiAI/graph/lowering/unlowered.hpp`), and it has no row in
`K_LOWERINGS`. That is deliberate.

From the outside a deliberate gap and an omission look the same, which is why the test exists.
Pinning the fatal stop records the decision. If somebody later adds a lowering, the test fails and
asks them to say why.

## test_rotary_decoder_overfit.cpp

Suites: `Integration_RotaryDecoderOverfit`, in `sushiai_language_tests`.

The setup is that of `test_decoder_overfit.cpp` without the position table: a token table of 64
rows, two rotary blocks of width 32 with two heads and a feed-forward width of 64, one batch of four
rows of 16 tokens read from a token file, and AdamW with clipping and a cosine schedule.

### ATwoBlockRotaryDecoderOverfitsOneBatch

The dataset holds one batch. The training graph holds four host tables, the cosines and the sines of
each block. Every state entry is a parameter, and one slot is bound for each entry. Over 400 updates
every loss is finite; the first is within 0.5 of ln(64) and the last is below 0.1 of the first. The
step reports one compile.

## test_rotary_embedding.cpp

Suites: `Integration_RotaryEmbedding`, in `sushiai_language_tests`.

A `RotaryEmbedding` is run through a compiled step over rows of float32 values in groups of four
channels, two pairs each. Results are compared with the host's double within 1e-5 absolute.

### ATurnMatchesTheHostFormula

Over three rows of one group, each pair `(a, b)` becomes `(a cos - b sin, a sin + b cos)` for the
angle `rotary_angle` gives its row and pair. Pair 0 of row 1, `(1, 3)` turned by one radian, is
-1.984110649 and 2.462377903.

### PositionZeroIsTheIdentity

Row 0 leaves the block with exactly the values it entered with. Row 1 of the same values differs by
more than 1 in its first channel.

### EveryGroupTurnsByTheSameAngles

Two groups of one row that hold the same values turn to exactly the same values, on each of three
rows.

### AScoreDependsOnTheDistanceOfThePositionsAlone

A query and a key are turned at six positions. The score of positions 2 and 0 equals the scores of 3
and 1, 4 and 2, and 5 and 3, and equals the number the host computes from a distance of 2. The score
of positions 3 and 0 differs from it by more than 1.

## test_rotary_generation.cpp

Suites: `Integration_RotaryGeneration`, in `sushiai_language_tests`.

The fixture builds rotary decoders from the model files of `decoder_model.hpp`: two blocks over a
window of 8, one block over a window of 8, and two blocks over a window of 64. For each it can
compile the window step and the one-position step and build a `WindowGenerator` and a
`CachedGenerator`. Float32 logits of two passes are compared with an absolute tolerance of 1e-4. A
ring of two or more blocks does not equal `WindowGenerator` after a slide; see
`modules/language/generation/README.md`.

### TheStepIsFedTheRingAndNoPosition

The cached generator of the two-block decoder reports a window of 8 and a vocabulary of 32. Its step
has five inputs, among them `cache_slot`, `rotary_cosine` and `rotary_sine`, and it refuses
`position` as `InvalidDescription`. The incremental graph carries four values and holds no host
table; the window graph holds four.

### TheStateHoldsParametersOnly

The state of the two-block decoder has 27 entries. Each is a parameter, and no name contains
"rotary".

### TheLogitsEqualThoseOfTheWindowAtTheLastPosition

After a prompt of five ids, the 32 logits of the one-position step equal the window step's at
position 4 within 1e-4.

### APromptThatFitsGivesTheTokensOfTheWindowGenerator

With two blocks, a prompt of three ids and four tokens give the same tokens from both generators.

### AOneBlockRingGivesTheTokensOfTheWindowGeneratorAcrossSlides

With one block, 26 greedy tokens after a prompt of three ids equal the window generator's. The
one-position step ran 28 times from one compile.

### TheLogitsOfAOneBlockRingEqualTheWindowsAtEveryPosition

A text of 30 ids is fed to the one-block ring one position at a time. At each position the window
step is run over the last 8 ids, and the 32 logits of the ring equal the window's at its last held
position within 1e-4.

### ARingEqualsAWideCacheReadThroughItsLastPositions

The state of the two-block decoder is copied into the decoder with a window of 64, whose cache never
wraps in this run and is masked to its last 8 positions. Greedy tokens sampled from that step equal
the 26 tokens the ring's generator produced.

### AContextThatSlidesTakesOneRunPerToken

Twelve tokens after a prompt of three ids take 14 runs of one compile.

### ASlotWrittenTwiceHoldsTheSecondRow

The one-block ring is fed 8 positions and then position 8, which wraps to row 0. A second step is
fed position 8 alone, into zeros. In both caches of the block, the keys and the values, row 0 of the
first step equals row 0 of the second exactly, and row 1 keeps the values it held before the wrap.

### ASecondPromptMatchesAFreshGenerator

A generator that has produced 26 tokens for one prompt gives, for a second prompt, the 26 tokens of
a generator compiled fresh over the same state.

### AStepOrARingThatDoesNotFitIsRefused

A ring without its angle function is refused as `InvalidDescription` with "no function" and
'rotary_cosine'. A decoder with a learned position table before a rotary block, whose step is fed
both leaves, is refused with a message that contains "both", 'position' and 'cache_slot'.

## test_run_checkpoint.cpp

Suites: `Integration_RunCheckpoint`.

A run is a classifier over 6 features and 4 classes with a batch of 8, its state, an optimizer and a
schedule, built from a seed. The cases save and load checkpoints with run state through
`IO::save_run_checkpoint` and `IO::load_run_checkpoint`, in scratch files that are removed
afterwards. Parameters are compared by their bit patterns.

### AResumedSGDRunMatchesTheUninterruptedOneBitForBit

SGD with a momentum of 0.9. One run takes 12 updates without interruption. A second takes 6 and is
saved. A third, built from another seed, differs before the load, equals the second after it with a
step count of 6, and after 6 more updates equals the uninterrupted run bit for bit with a step count
of 12. The same file loaded as weights alone leaves the step count at 0, and 6 more updates from
there do not give the uninterrupted run's parameters.

### AResumedAdamWRunMatchesTheUninterruptedOneBitForBit

The same sequence with AdamW.

### AFileWithoutRunStateIsWeightsAndNoResume

A file written by `save_checkpoint` is refused by `load_run_checkpoint` as `Incompatible`; the
message contains the file name, "version 2", "no run state" and 'weights'. `load_checkpoint` reads
it, and the parameters equal the saved ones.

### AVersionOneFileStillLoadsAsWeights

The run-state count is cut out of the header and the version byte set to 1. `load_checkpoint` reads
the file and the parameters equal the saved ones. `load_run_checkpoint` refuses it as `Incompatible`
with "version 1" and "no run state".

### AFutureVersionIsRefusedByBothLoaders

With the version byte set to 3, both loaders refuse the file as `InvalidFile`. Each message contains
"version 3" and "versions 1 to 2".

### RunStateOfAnotherOptimizerTypeIsRefused

A file saved from an SGD run is refused for an AdamW run as `Incompatible`. The message contains the
file name, 'sgd' and 'adamw'.

### AMissingOrSurplusRunStateRecordIsRefusedByName

A file saved by SGD without momentum is refused by SGD with momentum; the message contains "does not
supply" and a `.momentum` record. The file of the run with momentum is refused by the run without;
the message contains "does not keep". Both refusals are `Incompatible` and name 'sgd'.

The dropout cases train a classifier with one dropout of rate 0.25, traced with the seed
20261011, under SGD with a momentum of 0.9, over a dataset that reads the four batches in an
order `Data::EpochPermutation` draws for each epoch. Their loop keeps an `IO::RunProgress`.

### ARunCheckpointHoldsTheSeedAndThePositionOfItsRun

A progress of seed 9, 6 step runs, epoch 2 and batch 3 is saved and read back field for field.
`IO::check_run_seed` accepts 9 and refuses 10 with both. A save that states no progress reads
back as zeros.

### AResumedRunWithDropoutAndAShuffledOrderMatchesBitForBit

One run takes 12 updates. A second takes 6, which ends at epoch 1, batch 2, and is saved. A
third, built from another seed, loads the file, restores the run count of its trainer and takes
6 more updates from the loaded position: its parameters equal the first run's bit for bit, and
its progress ends at 12 step runs, epoch 3, batch 0.

### AResumeThatDropsTheRunCountOrThePositionDiverges

The same file. A run that does not restore the run count, and a run that restarts at epoch 0,
batch 0, each end with other parameters than the uninterrupted run.

### RunStateWithoutTheProgressRecordsIsRefusedOnResume

The last 151 bytes of a saved file are cut and its run-state count lowered by 4. The resume is
`Incompatible`, names `run.seed` and `weights`, and the file still loads as weights.

### AProgressRecordOfAnotherTypeOrRangeIsRefused

`run.batch` retyped to float64 is `Incompatible`; a `run.epoch` with its top bit set is
`InvalidFile`.

## test_segment_overfit.cpp

Suites: `Integration_SegmentOverfit`, in `sushiai_vision_tests`.

The model, `TinySegmenter`, is a 3x3 convolution of stride 2, a second 3x3 convolution and a
`SegmentHead` that upsamples by 2, all 8 channels wide, over two classes. The dataset is one batch
of two 16 by 16 images with three channels: sample 0 is split left and right, sample 1 above and
below, and two corners of each mask hold the ignore value 255. `Train::Trainer` fits it with
`PixelCrossEntropyObjective` and SGD at a rate of 0.1 for 80 epochs. The source marks the epoch
count, the rate and both thresholds as not yet confirmed by a run.

### TwoConvolutionsAndAHeadFitOneBatch

The report holds 80 steps and one compile. The last loss is below half the first, `mean_iou` of the
last step is above 0.8, and `pixel_accuracy` is at least `mean_iou`.

## test_state_checkpoint.cpp

Suites: `Integration_StateCheckpoint`.

The cases save and load checkpoints over a `NN::ModelState`. The model is a batch norm named `bn`,
so the state holds two parameters and two buffers. The cases pin the round trip, the order of the
records in the file, the refusal of a file without buffers, and the refusal of an unwritable path.

### SavesAndLoadsParametersAndBuffersBitForBit

A state whose every entry is filled, buffers included, is saved. A second state of the same model,
filled with other values, differs before the load. After the load, the bit patterns of every entry
equal the saved ones.

### TheFileListsEveryEntryInWalkOrder

The file holds one record for each state entry, and the record names come in the state's entry
order. The test reads each record and its payload back, and each payload is read at the length its
header declares.

### ACheckpointWithoutBuffersIsRefusedAndNamesTheRunningMean

A file that holds only the parameters is refused by `load_checkpoint` as `Incompatible`, and the
message contains "running_mean".

### AnUnwritablePathIsInvalidFile

`save_checkpoint` to a path inside a directory that does not exist is refused as `InvalidFile`.

## test_token_feed.cpp

Suites: `Integration_TokenFeed`.

A materialised dataset named "tokens" holds int32 ids, two rows of five per batch, all below 7. The
cases feed it to an int32 input leaf and to a float32 one.

### AnInt32BatchOfADatasetFeedsAnEmbedding

An `Embedding` of 7 rows and width 4 is traced for a `[2, 5]` int32 input, and its state has one
entry. The input leaf describes itself as int32 and `match_input` accepts the dataset. After one
evaluation step over the batch, each output row equals the table row of its id, with a tolerance of
0.

### FitRefusesAnInt32DatasetForAFloat32Leaf

A classifier MLP whose input leaf is float32 `[5]` is fitted to the same dataset. `fit` is refused
as `Incompatible`; the message names 'tokens', "as int32 [5] per sample" and "compiled for float32
[5] per sample". The trainer has taken zero steps.

## test_training.cpp

Suites: `Integration_Training`.

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

### SGDDrivesTheLossDownOnASeparableProblem

An MLP of 8 features, 24 hidden units and 4 classes is fitted to a synthetic dataset of 6 batches of
16 for 30 epochs, with SGD at a rate of 0.2 and a momentum of 0.9. The first loss is ln(4) within
1e-4, the last is below a quarter of the first, the accuracy is above 0.9 and the compile count is
1.

### AdamWDrivesTheLossDownOnASeparableProblem

The same model and data with AdamW at a rate of 0.02. The last loss is below a quarter of the first,
the accuracy is above 0.9 and the compile count is 1.

### AdamWsFirstStepMovesEveryParameterByAboutTheLearningRate

The "eps after the bias correction" ordering exists to preserve this property. On step one,
m / (sqrt(v / bc2) + eps) is g / (|g| + eps), so every coordinate moves by almost exactly the
learning rate. Folding eps inside the correction makes it 31.6x larger and visibly damps this
first step, the only place the bug is easy to see.

The loop skips coordinates whose gradient is below 1e-4, where eps does matter. The claim is
about the ordinary case.

### SGDWithoutMomentumIsExactlyMinusLearningRateTimesTheGradient

One step of SGD at a rate of 0.1 over a `Linear` of 4 features and 3 classes. Each element of the
first parameter after the step equals its value before minus 0.1 times its gradient, within 1e-6.

### ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes

The shared test runtime is created without `RuntimeConfig::profiling`, so the runtime records
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

### RunningMeanMovesAcrossTwoStepsAsUpdatedStatisticSays

The model is Unflatten, a batch norm over three channels of a 2 by 2 grid, Flatten and a Linear
layer. It is trained for two steps on two synthetic batches. After each step, the running mean of
every channel equals the `NN::updated_statistic` of that step's channel mean, with the default
momentum and a source scale of 1, within 1e-5. After the second step, every channel's running mean
differs from its value after the first step by more than 1e-7.

### APerStepOptimizerStillStepsWhenTheModelHoldsABatchNorm

The same model is trained with the per-step optimizer placement. After one step, the optimizer is
not in the plan, the step count is 1, and at least one element of `3.weight` has changed by more
than 1e-7. The test notes that the classifier starts at zero, so the first gradient reaches its
weight alone.

### ABatchOfAnotherShapeIsRefusedWithBothShapes

A trainer compiled for a batch of 8 samples with 6 features refuses a batch of 4 samples. The
refusal is `Incompatible`, and its message contains "[4, 6]" and "[8, 6]".

### BoundSlotsDecayByTheRoleOfTheirParameter

The classifier graph of a two-layer MLP binds four slots. The first is named `0.weight` and decays,
the second is `0.bias` and does not, and the third and fourth follow the same pattern.

## test_yolo26_forward.cpp

Suites: `Integration_Yolo26Forward`, in `sushiai_vision_tests`.

The model is built from `applications/examples/yolo26/model.json`, with rows 9, 10, 16, 19, 21 and
22 published beside the head. The weights and the references come from the `yolo26` folder of the
fixture directory: `weights.sushi`, `input.sacp` and `forward.sacp`. A case is skipped, naming the
file, when one of them is missing. The input is `[1, 640, 640, 3]`, and the whole-model cases share
one forward pass. A feature map or a raw head output is held to 1e-4 absolute, a decoded box to
1.3e-4 pixels and a class probability to 1e-6.

### TheFeatureRowsMatchTheReferenceLevels

Rows 16, 19 and 22, the three feature maps the head reads, equal the records `features.level0`,
`features.level1` and `features.level2`.

### TheInnerRowsMatchTheReferenceRows

Rows 9, 10 and 21 of the whole-model pass equal the records `features.row9`, `features.row10` and
`features.row21`.

### C2PSAAloneMapsTheReferenceRow9ToRow10

Row 10 is built alone, with the records of `weights.sushi` that rename to its entries and the
reference's row 9 as its input. Its output equals the reference's row 10.

### TheTwelvePortsMatchTheReferenceRawOutputs

The `boxes` and `scores` ports of the `one2many` and `one2one` branches at each of the three levels
equal the records of the same names.

### TheHostDecodeMatchesTheReferenceDecodes

The raw outputs of each branch are joined over the levels of 80, 40 and 20 cells a side and decoded
on the host, `one2many` as centre and size and `one2one` as corners. The boxes and the sigmoid of
the scores equal the `decoded.boxes` and `decoded.scores` records of the branch.

### TheTopScoresMatchTheReferenceDetections

`select_top_detections` over the `one2one` decode gives rows whose score column equals that of the
`one2one.detections` record within 1e-6.
