# tests/core/autograd/rules/test_all.cpp


Suites: `Medium_RuleTable`, `Medium_Autograd`.

Tests that every domain's backward rules reach the combined table exactly once.

## EveryDomainsRulesReachTheCombinedTable

`modules/core/autograd/include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

## NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

## TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

## NoGradientBarrierHasARule

For every row of `Ops::K_ALL` with `blocks_gradient` set, `Autograd::has_rule` and
`Autograd::has_backward_rule` are both false. The loop counts the rows it checked and expects
4, so it cannot pass by checking none.

## TheComparisonIsABarrierWithoutARule

`Ops::K_LESS_EQUAL` has `differentiable` false and `blocks_gradient` true, and
`Autograd::has_rule` is false for it.

## EveryActivationRowHasARuleForItsForwardIdentityOnly

For every row of `Ops::K_ACTIVATIONS`, `Autograd::has_rule` is true for the `forward` identity
and false for the `backward` one. `Autograd::K_ACTIVATION_RULES` has as many entries as
`Ops::K_ACTIVATIONS` has rows.

## BatchNormFoldHasNoRuleAndBatchNormKeepsIts

`Autograd::has_rule` is true for K_BATCHNORM2D and false for K_BATCHNORM_FOLD, `has_backward_rule`
is false for K_BATCHNORM_FOLD, and `Autograd::K_NORM_RULES` has two entries.

## LayerNormHasItsRuleAndItsGradientHasNone

`has_rule` is true for LAYER_NORM. LAYER_NORM_BACKWARD is not differentiable and has no rule.

## LogSoftmaxHasItsRuleAndItsGradientHasNone

`K_LOSS_RULES` has two rows. LOG_SOFTMAX is differentiable and has a rule.
LOG_SOFTMAX_BACKWARD is not differentiable and has none.

## TheCausalMaskHasItsRule

`K_LINALG_RULES` has six rows and `has_rule` is true for CAUSAL_MASK.

## DropoutHasTheRuleOfTheRandomDomain

`K_RANDOM_RULES` has one row. DROPOUT is differentiable and has a rule.

## TheGatherAndTheScatterAddDifferentiateIntoEachOther

GATHER_ROWS and SCATTER_ADD_ROWS are both differentiable and both have a rule.

## TheDepthwiseForwardHasItsRuleAndItsGradientsHaveNone

`K_SPATIAL_RULES` has nine rows and `has_rule` is true for DEPTHWISE_CONV2D. Its two backward
identities are not differentiable and have no rule.


The whole graph (forward, loss and backward) runs on the device in double precision. The
analytic gradient it produces is compared against a central difference of the loss the same
graph computes. No other check separates a correct backward rule from a plausible one. It has
to be fp64: an fp32 difference quotient bottoms out near 2.4e-5 relative accuracy, the same
order as the error a wrong rule produces. See `gradcheck.hpp`.

## Probe

The variable under test is declared as the only gradient-requiring leaf. Everything else the
build callback declares is a constant the test fills. The loss is always the mean negative
log-likelihood against a target matrix, which is a valid scalar functional of anything shaped
[N, C]. When the model ends in a log-softmax it exercises the fused cross-entropy rule as
well.

Fusion is off by default, because almost every case checks a backward rule and fusion must not
change one. It can be switched on because one case checks the opposite direction: a fused form
that rewrites what the backward pass reads must leave the same derivative behind. Value ids
survive the fusion pass, so nothing the probe recorded before it needs remapping.

## MatmulWithoutTransposes

The variable is a `[4, 5]` left operand multiplied by a `[5, 3]` input with neither side transposed.
Its gradient is checked against central differences.

## MatmulWithATransposedRightOperand

A `[4, 5]` variable against the transpose of a `[3, 5]` input, the form a `Linear` layer emits. The
gradient is checked against central differences.

## MatmulWithATransposedLeftOperand

The transpose of a `[5, 4]` variable against a `[5, 3]` input. The gradient is checked against
central differences.

## MatmulWithBothOperandsTransposed

The transpose of a `[5, 4]` variable against the transpose of a `[3, 5]` input. The gradient is
checked against central differences.

## MatmulGradientOfTheRightOperand

The same node is differentiated from the other side. The variable is the weight, so the case
covers the dB half of the rule where the earlier matmul cases cover dA.

## BiasAddPassesTheMatrixGradientThrough

The variable is the `[4, 3]` matrix of a `bias_add` whose bias of three elements is an input. The
gradient is checked against central differences.

## BiasAddSumsOverTheBroadcastAxis

The same node is differentiated at the bias of three elements, with the `[4, 3]` matrix as an input.

## ElementwiseAddAndSubtract

The case runs twice over a `[4, 3]` variable and a `[4, 3]` input, once through `add` and once
through `sub` with the variable on the left. Each gradient is checked against central differences.

## ElementwiseSubtractFromTheRight

The variable is the right operand of `sub`, so the case covers dB = -dC.

## ElementwiseMultiply

The variable is the left operand of `mul` against a `[4, 3]` input.

## ElementwiseMultiplyFromTheRight

The case checks dB = dC * A, the half the left-hand check cannot reach. The rule is written
symmetrically, so a copy-paste slip that reads `inputs[1]` on both sides would leave the
left-hand test green.

## AccumulateInTheForwardGraph

K_ACCUMULATE has a backward rule that nothing else exercises. The two fan-out tests build an
K_ACCUMULATE chain in the backward graph, and `differentiate()` walks only the forward nodes, so
those chains are never differentiated themselves. Reaching the rule needs an K_ACCUMULATE the
caller put in the forward pass.

The lowering aliases the result onto the accumulator's own buffer, so running the step leaves
the variable holding `x + other`. Every probe rewrites the whole variable first, which makes
that harmless.

## Scale

A `[4, 3]` variable is scaled by -2.5.

## Exp

The gradient of `exp` over a `[4, 3]` variable.

## ReLUAwayFromItsKink

ReLU is not differentiable at zero. A difference quotient that straddles the kink reports the
average of the two one-sided slopes, which no correct rule can match, so the point is sampled
clear of zero.

## LeakyReLUHasBothOfItsSlopes

The point is sampled off the kink and over a range wide enough that elements land on each
side. A probe that only saw positive inputs would pass with the negative slope hard-coded to
zero, which is a plain ReLU. `LeakyReLUBranchSwapIsCaughtByProbePoints` counts the negative
points the sample contains.

## LeakyReLUBranchSwapIsCaughtByProbePoints

The test pins the negative-side split of the exact point `LeakyReLUHasBothOfItsSlopes` uses:
same seed, same scale, same `push_off_kinks`. The branch-swap mutation depends on that split.
If a change to the seed or the noise generator collapsed the sample to all-positive, the
mutation would go undetected, and this test fails first.

## GELU

The gradient of `gelu` over a `[4, 3]` variable sampled with a noise scale of 1.5.

## AFusedGELUEpilogueHasTheSameDerivativeAsAnUnfusedOne

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

## SigmoidReadsItsForwardOutput

The gradient of `sigmoid` over a `[4, 3]` variable sampled with a noise scale of 1.5.

## TanhReadsItsForwardOutput

The gradient of `tanh` over a `[4, 3]` variable sampled with a noise scale of 1.5.

## SumRowsBroadcastsItsGradientBack

`sum_rows` reduces a `[4, 3]` variable to three sums, and a `bias_add` onto a `[1, 3]` value of
zeros turns them into the one-row batch the loss reads. The gradient at the variable is checked
against central differences.

## GatherRowsScatterAddsARepeatedIndexIntoItsTable

The variable is a `[5, 3]` table gathered with the index `{3, 1, 3, 3, 0, 1}`. Row 3 is read three
times and row 1 twice, and rows 2 and 4 are read by no index.

## GatherRowsOverARankTwoIndexDifferentiatesThroughItsReshape

A `[4, 3]` table is gathered with a `[2, 3]` index into `[2, 3, 3]` and reshaped to `[6, 3]` for the
loss. The gradient at the table is checked against central differences.

## ScatterAddRowsGathersARepeatedIndexBackOntoItsRows

The variable is the `[6, 3]` rows scattered into five table rows with the index `{3, 1, 3, 3, 0,
1}`. Row 3 is written three times and row 1 twice, and rows 2 and 4 are written by no index.

## ScatterAddRowsOverARankTwoIndexGathersInTheRowsOwnShape

`[2, 3, 3]` rows over a `[2, 3]` index sum into a `[4, 3]` table. The gradient at the rows is
checked against central differences.

## FusedCrossEntropyAgainstOneHotTargets

The gradient of the mean NLL with respect to the logits is (softmax - Y) / N. It is emitted as
one fused rule and not by differentiating the seven-op log-softmax recipe, so it needs its own
numeric proof.

## GradientAccumulationAtAFanOut

The variable feeds `tanh` and `sigmoid`, and their sum is the logits. The two branches make the
gradient of the variable an ACCUMULATE chain.

## WholeTwoLayerModelUnderRandomDirections

This is the end-to-end check. A full Jacobian over every weight would take 2n device runs.
Random unit directions take 2k runs whatever n is, and the chance that a wrong gradient agrees
along all sixteen of them is negligible.

## ReshapeMovesItsGradientBackUnchanged

A reshape moves no data, so its gradient is its incoming gradient in the original shape. The
numeric check is still worth having: reshaping back to the input shape and reshaping back to
the output shape are both one line, and only one is right.

The fused K_LOG_SOFTMAX + K_NLL_LOSS rule's closed form assumes a one-hot target row
(`nll_loss_backward` says it is "valid because the targets are one-hot"). Noise in the targets
would make the analytic gradient disagree with the numeric one for reasons that have nothing
to do with K_RESHAPE.

## LayoutConvertsGradientIsThePermutedGradient

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

## LayoutConvertRoundTripComposesBothDirections

The two directions composing to the identity is fair to check. It cannot tell a correct
permutation from a same-shape non-permutation, for the reason given under the single-hop case,
so it is a secondary check and not the proof.

## AWholeConvolutionDifferentiatesThroughIm2colAndMatmul

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

## Im2colsGradientAccumulatesEveryWindowThatReadAPixel

In a stride-1 padded 3x3 convolution nine windows read every interior pixel, so its gradient
is a sum of nine terms. A `col2im` that overwrote where it should accumulate would give one of
them: a gradient of the right shape and sign, at roughly a ninth of the right size.

## AnAsymmetricallyPaddedConvolutionDifferentiatesOverItsRealWindowGrid

This is the gradient over the geometry a symmetric re-derivation gets wrong without an error.
A [1, 4, 5, 1] input with a 3x3 window, `pad_top` 1 and `pad_left` 1, padded 2 below and 0 on
the right, has a 5x4 window grid. Deriving the grid from the geometry alone gives 4x5, the
transpose, with the same 20 windows, so nothing throws. The forward lowering and
`im2col_backward` both read the real grid out of the node's attribute slots 8 and 9, and the
case checks the whole path against finite differences.

## Col2imsGradientGathersEveryWindowThatWroteAPixel

K_COL2IM's rule is the mirror image of `im2col_backward`. The case checks it without `im2col`:
the variable is the column matrix, fed straight into `col2im` and reshaped into `log_softmax`.
That keeps it single-hop by construction (A-7). No `im2col` sits on the path for `col2im` to
be the adjoint of, so an involutive round trip cannot satisfy the check.

## MaxPoolRoutesEachGradientToTheElementThatWon

The point is sampled with `push_off_kinks` so that no two elements in a window tie. At a tie
the max is not differentiable, and the numeric check would be comparing against a one-sided
derivative.

## OverlappingMaxPoolWindowsAccumulateOnADominantPixel

At stride 1 a pixel that wins several windows receives all their gradients. This is where
keeping the pooling backward functions separate pays: a shared `pool_backward` would have had
to branch on mode inside this loop.

## AvgPoolSpreadsItsGradientOverTheWholeWindow

The brief's [2, 6, 6, 3] variable has 216 elements, over the 200-input full-Jacobian ceiling
of `expect_gradient_matches`. The case uses one batch row so that the check stays in the mode
it asserts on.

## UpsampleSumsEachSourcePixelsReplicas

Scale 2 replicates every pixel four times, so its gradient is the sum of four. Forgetting the
sum gives a gradient a quarter of the right size. A model with that fault still trains, more
slowly, and only in this one layer.

## SpatialMeanBroadcastsScaledByOneOverHW

An earlier draft of the design lacked this rule. It is a per-(n, c) broadcast. K_BIAS_ADD's
broadcast is per row and K_CHANNEL_AFFINE's is per channel, and neither would have produced it.

## ConcatGivesEachOperandItsOwnChannelRange

The gradient of a concat is a slice per operand, at that operand's offset. An off-by-one
offset gives every operand a gradient of the right shape that belongs to its neighbour. A
round trip through slice-then-concat cannot detect that, so the operands are concatenated in a
different order from the one they were sliced out in.

## SliceScattersItsGradientBackIntoZeros

Channels outside the slice receive exactly zero, not stale memory. The reordered concat makes
a wrong offset visible; this case makes a missing zero-fill visible.

## BatchNormsGradientAccountsForTheStatisticsItComputed

The batch mean and variance are functions of x, so dx has three terms. Treating the statistics
as constants gives a gradient that is right on average over the batch and wrong per element.
The model still trains, slightly worse, and nothing points at the cause.

## BatchNormsScaleGradientIsTheSumOfDyTimesTheNormalisedInput

The same node is differentiated at gamma. It is a separate case because the `Probe` checks one
variable. dgamma is the term that would go unnoticed if it were summed over the wrong axis,
since the result is a [C] value of the right shape either way.

## BatchNormsShiftGradientIsThePlainSumOfDy

The same batch norm over a `[4, 3, 3, 5]` input is differentiated at beta, its shift of five
elements. The logits are the spatial mean of the output through a `log_softmax`, against one-hot
targets.

## ChannelAffineDifferentiatesAtItsInput

The variable is the `[2, 4, 4, 6]` input of a `channel_affine` whose scale and shift are parameters
filled with noise. The logits are the spatial mean through a `log_softmax`, against one-hot targets.

## ChannelAffinesScaleGradientSumsOverNHAndW

dgamma is a sum over three axes at once. A sum over the wrong ones, per sample where it should
be per channel, is still a [C]-shaped value when N equals C. That is why the case is checked
numerically and not by shape.

## ChannelAffinesShiftGradientSumsOverNHAndW

The same node is differentiated at beta, its shift of six elements.

## DepthwiseConvolutionDifferentiatesOverItsInputAndItsTaps

One flat variable is sliced into a `[2, 5, 5, 3]` image and `[3, 3, 3]` taps, so one check covers
both operands. The convolution has a 3x3 window, a stride of 2 and a pad of 1 at the top and the
left, and gives `[2, 3, 3, 3]` windows whose spatial mean is the logits.
