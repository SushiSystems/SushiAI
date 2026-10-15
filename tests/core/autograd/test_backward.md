# tests/core/autograd/test_backward.cpp


Suites: `Unit_Layout`.

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

## EveryValueIsNHWCUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

## FinaliseAcceptsAUniformGraph

`finalise` does not throw on a graph of one MATMUL over a [2, 4] input and a [4, 3] parameter.

## FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data.

## LayoutConvertIsAllowedToChangeIt

K_LAYOUT_CONVERT is the one exemption, and the reason the check runs per node and not as a single
sweep over every value.

## DeclareLikeInheritsTheSourceValuesLayout

Fourteen tasks still to come each emit a value and carry its layout forward by hand.
`declare_like` is the one place that does it, so a forgotten carry cannot happen.

## Crossed

Both rebuilding passes copy the value table entry by entry, so that copy is what the `Crossed`
graph tests, and not the pass's own subject. The chain is deliberately plain for the same
reason: a fusion candidate or a precision decision in it would only obscure which field went
missing.

## FusionCarriesTheLayoutTagAcrossItsRebuild

`fuse()` rewrites the node list and copies the value table. A field it forgets to copy is reset
to `add_value`'s default, which is NHWC. A dropped tag is therefore a wrong layout, not a
missing one, and every shape check downstream still agrees with it.

The cost of a dropped tag shows in `lower_layout_convert` in `modules/core/graph/source/lowering/shape.cpp`, which
decides from these two tags whether each side is read or written through a transpose. A genuine
NCHW -> NHWC convert that reads as NHWC -> NHWC applies no transpose and copies transposed data
straight through.

## CastInsertionCarriesTheLayoutTagAcrossItsRebuild

Cast insertion is the other rebuilding pass and could make the same omission. Trainer runs both
passes unconditionally, so either one is enough to lose the tag on the default path.

## ARebuiltGraphStillProvesUniform

The consequence is worth stating separately. With every tag reset to one value, `finalise()`
passes for the wrong reason. A graph that is uniform because it was flattened cannot be told
from one that was uniform to begin with, so the proof has to be re-run against tags that
survived.

## TheThreeRulesThatEmitZerosCannotBeReachedWithAnNCHWOperand

The test checks why the zeros exposure is latent and not live, and does not leave that to a
comment. It is the premise the `zeros_like()` change rests on, and it is a premise about other
emitters' guards, which is the kind that stops being true without anyone editing this file.

The two `EXPECT_THROW` lines on `slice_channels` and `channel_affine` cover the rank-4 sites,
`slice_channels_backward` and `channel_affine_backward`. CV-3 makes their forward emitters
refuse an NCHW operand outright, so their dy can never carry that tag.

The `reshape` to `rows` covers the rank-2 site, `sum_rows_backward`. Reshape carries an NCHW
tag into rank-2 land with no guard at all, so unlike the two rank-4 sites the emitters let this
chain be built.

The last `EXPECT_THROW(graph.finalise(), Error)` shows what blocks that chain: the far end.
`differentiate()` only seeds a loss produced by K_NLL_LOSS, whose one-hot operand is a leaf and so
always NHWC, and nothing but K_LAYOUT_CONVERT may cross back, which is rank-4 only. `finalise()`
rejects that node before differentiation ever runs, so the rule is never reached with an NCHW
gradient. The guard is the pre-existing proof in `finalise()`, not a check in any emitter, and
that is why it is worth a test and not a sentence.

## ZerosLikeCarriesTheTagAndPlainZerosCannot

The three backward rules now depend on this contract. K_ZERO reads nothing, so plain `zeros()` has
no operand to inherit from and its NHWC tag is a default, not an answer. That is harmless for
the K_ZERO node itself, which the per-node proof skips for having no inputs. The consumer that
reads those zeros beside a real operand is the node that would be seen crossing layouts.

## FinaliseStillPropagatesRequiresGrad

The rename must not quietly drop the behaviour the old name described. The assertion is the
same one `test_ir.cpp` made.

Suites: `Unit_Precision`, `Unit_CastInsertion`.

Tests the mixed-precision policy and where the cast insertion pass places its casts.

This file needs no device. The precision policy is a table of judgements and the cast pass is a
pure function of the IR, as the fusion pass and the memory planner are.

The cases pin placement, not survival. A mixed-precision graph that runs proves almost nothing:
a missing cast throws in SushiBLAS, so "it ran" only rules out the loud half of the failure
space. The quiet half, a cast in the wrong place, leaves a graph that trains and learns worse.
So every case is one of these:

- the whole-graph invariant (every node's operands agree with the type it computes in), asserted
  by walking the graph and not by spot checks,
- a control (the same graph under the fp32 policy must get zero casts, computed and not
  short-circuited), or
- one of the two places the master-weight arrangement can silently break: a parameter retyped in
  place, or a gradient that reaches its fp32 buffer without being widened.

The cases fall into five groups, in file order:

- the policy itself
- the control: the fp32 policy changes nothing
- the invariant
- master weights, and the direction that goes silently wrong
- what the pass promises about ids, reuse and repetition

## Traced

The `Traced` graph is built by hand and not through NN::Sequential, so that the test can name
every value it wants to assert about. It is also the smallest graph that still contains all four
interesting shapes: a parameter read by both passes, a reduction, a loss, and a gradient buffer.

## TheUniformPolicyAnswersTheGraphsOwnTypeToEverything

For every entry of `Ops::K_ALL` and each of `Half`, `Float32` and `Float64`,
`UniformPrecisionPolicy::choose` returns the type it was asked about. The policy's name is
"fp32".

## OnlyTheProductAndWhatAGEMMFoldsIntoItNarrow

The `narrows` list states the rule set as a table, so that adding an operation to the narrowing
set is a visible edit in this test and not a silent behaviour change. Everything absent from it
stays in the graph's own precision, which is the safe answer for an operation nobody has thought
about yet.

The K_LEAKY_RELU entries are in `narrows` because K_LEAKY_RELU joined `Ops::K_ACTIVATIONS` with
`GradientSource::Output`. It narrows by the same substitution argument as K_RELU.

The `stays` list opens with the reductions and anything that sums many terms. fp16 carries about
11 bits of significand and SushiBLAS's own reduction header documents a 2048 ceiling for HALF,
so these are the operations that go quietly wrong.

K_DYNAMIC_SCALE is in `stays` because it carries the gradient loss scale, the one number in the
graph that exists to keep values out of fp16's subnormals. Narrowing the node that applies it
would overflow the very quantity it protects; see `modules/core/optim/include/SushiAI/optim/loss_scaler.hpp`.

K_GELU, K_GELU_BACKWARD and K_GEMM_BIAS_GELU, the one fused form that folds K_GELU, are in `stays`. The
form is a GEMM and rule 1 would otherwise take it. Narrowing it puts the tanh approximation's
`x*x*x` into fp16, which overflows at |x| ~ 40.3. Rule 5 in
`modules/core/graph/include/SushiAI/graph/precision.hpp` says so, and also explains why that leaves rules 2 and 5
consistent and not merely quiet.

The twenty spatial and CV identities are in `stays` because none of them is a GEMM epilogue or
an activation this policy folds. Each gets the same safe default as everything else nobody has
special-cased: it stays at the graph's own precision.

K_SILU, K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD are in `stays`. K_SILU and K_SOFTPLUS read
their input, so neither they nor their gradients narrow. K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN,
K_LESS_EQUAL, K_SELECT, K_STOP_GRADIENT and K_CONSTANT are in `stays` under the same default.

K_BATCHNORM_FOLD is in `stays`. It takes the square root of a variance plus epsilon.

The closing `EXPECT_EQ` on the two list sizes holds because every identity is accounted for by
`narrows` or by `stays`. A new operation cannot join the enum without someone deciding which
list it belongs to.

## TheDepthwiseIdentitiesKeepTheirDeclaredTypeUnderMixedPrecision

For DEPTHWISE_CONV2D and its two backward identities, `narrows_under_mixed_precision` is
false and `MixedPrecisionPolicy::choose` returns `Float32` for `Float32` and `Float64` for
`Float64`.

## GELUStaysWideEvenThoughACatalogEntryNowFoldsIt

Rule 5 rests on arithmetic. `FusedForm::GEMMBiasGELU`
folds GELU by spilling the pre-activation, so "nothing folds it, therefore nothing forces its
precision" is not a reason for anything. The argument is the arithmetic: the
tanh approximation computes `x*x*x`, which overflows fp16 at |x| about 40.3. A unit test cannot
assert that without a device.

The first three assertions check what a unit test can: that the conclusion did not quietly
follow the dead premise. The catalog can fold GELU and the policy still keeps it wide, which is
the pair of facts rule 5 in `modules/core/graph/include/SushiAI/graph/precision.hpp` now asserts.

`narrows_under_mixed_precision(K_GEMM_BIAS_GELU)` is false because the fused form has to go with
GELU. Otherwise the cast pass would narrow a node whose epilogue is the very expression rule 5
refuses to put in fp16.

## ChoosingIsIdempotentForEveryOperation

`modules/core/graph/include/SushiAI/graph/precision.hpp` states this contract, and it is what makes the invariant
checkable after the pass has run and not only while it runs. Without it,
`find_dtype_mismatches` would report a graph the pass had just finished making correct.

## AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType

The checks on the fp64 and fp16 graphs say that an fp64 graph keeps fp64 masters and still gets
fp16 products, and that an fp16 graph is already at the floor and is left alone, not "narrowed"
to something wider.

The `to_single` policy narrows to fp32. That is a legal thing to ask for, and it is what an fp64
model would use.

## ANonFloatLowTypeIsRefusedAtConstruction

`MixedPrecisionPolicy{DType::Int32}` and `MixedPrecisionPolicy{DType::Complex32}` throw
`Error`, and `MixedPrecisionPolicy{DType::Half}` does not.

## EveryChoiceCarriesAReasonSomebodyCanRead

For every entry of `Ops::K_ALL`, the choice of `MixedPrecisionPolicy` over `Float32` has a
non-empty `reason`.

## TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Unit_CastInsertion cases measures nothing.

## NoPolicyMeansTheUniformPolicy

`insert_casts` without options reports 0 casts and leaves the node count unchanged.

## EveryNodesOperandsAgreeWithTheTypeItComputesIn

This assertion is worth more than a dozen spot checks. It walks the whole graph and states the
rule, so it keeps holding as operations are added.

## TheInvariantSurvivesTheFusionPassThatRunsAfterIt

Fusion rewrites nodes and re-pairs operands, so it is where a correctly cast graph could stop
being one, and where a master weight could get swallowed.
`modules/core/graph/include/SushiAI/graph/cast_insertion.hpp` argues the ordering, and this test is the assertion
behind the argument.

## ADeliberatelyBrokenGraphIsReportedRatherThanAccepted

`find_dtype_mismatches` is the test file's own instrument, so it has to be shown to fail on a
graph that is wrong. Otherwise every "no mismatches" in the earlier cases could come from a
checker that never looks.

`mismatches.size()` is 3 because the graph is an uncast fp32 matmul under a mixed policy: two
operands and one result, all three float32 where the policy says half.

## AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy

The fp16 value the GEMM reads is a different value from the parameter. A cast produces it, and
it lives with the step's scratch and not for the whole run.

## AGradientIsWidenedBeforeItReachesItsBuffer

The test covers the backward direction. A matmul computing in fp16 produces an fp16 gradient,
and if that reached the fp32 gradient buffer uncast, SushiBLAS would throw. If the buffer had
been retyped to fp16 instead, nothing would throw and the optimizer would accumulate in fp16,
which is the failure master weights exist to prevent.

The loop over `gradients[0]` and `gradients[2]` covers the weights only. A backward K_MATMUL,
which computes low, produces the gradients of w1 and w2. K_SUM_ROWS, which does not, produces
those of b1 and b2, so only the weights need widening.

The loop over `gradients[1]` and `gradients[3]` covers the biases. Their gradients come out of
a reduction that stayed wide, so nothing was widened into them and nothing needed to be.

## AccumulatingGradientContributionsHappensWide

A parameter read twice gets two gradient contributions and an K_ACCUMULATE to sum them. Two things
have to hold at once. The sum runs in the wide type, because it is a sum. Its result and its
first operand agree, because the lowering makes the result alias the accumulator's buffer, so a
disagreement there is a wrong-width write and not merely a wrong number.

## AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot

This case covers the half of the pinning rule that a lifetime class does not. The logits are an
ordinary activation. The training loop reads them back by id for accuracy, and a caller who
forgets to say so gets an fp16 buffer where their host-side argmax expects floats.

## AValueReadTwiceIsConvertedOnce

w2 is read by the forward matmul and again by the backward one. Two conversions would be two
kernels and two buffers for one number, and the reuse is safe because the IR is
single-assignment.

## ValueIDsSurviveThePass

After `insert_casts` with a `MixedPrecisionPolicy`, the graph has more values than before,
and every value that existed before keeps its shape and its lifetime under the same id.

## RunningThePassTwiceAddsNothingTheSecondTime

The first `insert_casts` reports at least one cast. A second run with the same options
reports 0 and leaves the node count and the value count unchanged.

## TheReportNamesEveryConversionAndItsDirection

The report's policy is "mixed". Its count equals the number of CAST nodes, the difference
between `nodes_after` and `nodes_before`, and the sum of `count_into(Half)` and
`count_into(Float32)`, both of which are above 0. `retyped_values` is above 0, and
`to_string` holds "mixed" and "casts inserted".

## AGraphThatIsNotInTopologicalOrderIsRefused

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first.

## Pooled

K_MAXPOOL2D is the first operation in this IR whose results are not all one element type: it
writes the pooled tensor and, beside it, the INT64 argmax its backward reads. The cast pass
derives one precision per node from `outputs[0]`, which makes the argmax look like a value in
the wrong type and not like a value with no precision to choose.

`x` is a parameter and not an input for the reason
`tests/core/graph/test_memory_plan.cpp` gives: K_MAXPOOL2D_BACKWARD is only emitted
when something upstream of the pool needs a gradient.

## AnIndexBufferHasNoPrecisionToChoose

The pass forced every operand and every result of a node to one dtype, derived from the node's
first result. For K_MAXPOOL2D that is FLOAT32 from the pooled tensor, so the INT64 argmax read as
a mismatch and the pass tried to convert it. `GraphBuilder::cast` refuses anything but
real-float to real-float by design, so the whole build threw.

The defect does not depend on the policy. `wanted` comes from `outputs[0]`, not from the
policy's `choose()`, so the uniform policy reaches it too.

## AMixedPolicyLeavesTheIndexBufferAloneToo

`insert_casts` with a `MixedPrecisionPolicy` over the pooled trace does not throw. The pool's
index value stays `Int64` and no CAST node reads or writes it.

## APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer

This is the loud half of the same defect. An unpinned argmax is retyped in place and the
mismatch disappears silently. A pinned one keeps its id and its INT64 buffer, so the pass
reaches for the widening cast, and K_CAST is real-float to real-float by design.

## ABufferKeepsItsDeclaredTypeAndItsStateLeafAcrossThePass

The graph is a K_MATMUL and a K_BIAS_ADD whose bias is a buffer. The weight and the buffer are
bound to the states 31 and 32 and the graph is marked refused before `insert_casts` runs under
the mixed policy. The report counts at least one cast and the K_BIAS_ADD's result is HALF, so
the pass did rewrite the graph. The buffer is still FLOAT32 with the lifetime `BUFFER` and no
gradient. Both `state_leaf` lookups, the whole `state_leaves()` table and `has_refusal()` are
what they were, and `find_dtype_mismatches` returns nothing.

## BatchNormFoldStaysInTheMasterTypeUnderTheMixedPolicy

The graph is one `batchnorm_fold` over two parameters and two buffers. `insert_casts` under
the mixed policy reports zero casts and the graph holds no K_CAST node. The four operands and
the two results are all FLOAT32.

Suites: `Unit_Rematerialise`.

Tests the structure of the graph that the K_IM2COL rematerialisation pass rewrites.

This file needs no device. The K_IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Integration_Rematerialise in `tests/core/graph/test_rematerialise.cpp`
does that by running the same graph twice.

The brief's three cases are all in this file, with two deviations that the library forced:

1. `Autograd::differentiate()` refuses any root that is not a scalar produced by K_NLL_LOSS
   (`modules/core/autograd/source/differentiator.cpp` explains why every other root would silently return empty
   gradients). The brief's graphs stopped at a K_MATMUL, so each is carried on to a loss. The
   shapes, geometry and expectations are the brief's.
2. A "backward consumer" cannot be identified by node index, because the pass runs when the
   backward nodes are already in the graph and no marker separates the phases. The rule the pass
   applies is "every consumer after the first", which is the same set for every graph
   differentiation produces and is well-defined for one it did not produce.

## TheColumnBufferStopsBeingAnActivation

This case checks the number CV-10 is about. ResNet-18's first block at batch 128 holds
`128*32*32*64*9*4` bytes, about 302 MB, in this one buffer, and there are twenty convolutions.
Cloning the gather next to its backward consumer drops the lifetime to SCRATCH, so one buffer is
live at a time and not twenty.

## TheCloneComputesTheSameThingAsTheOriginal

The operand and attribute checks guard against a clone that read a different operand or carried
different attributes. Such a clone would produce a plausible column matrix and a wrong weight
gradient. Both are checked because the geometry lives in the attributes and is easy to drop when
copying a node.

The check on `copy` expects the same description and the same layout. The layout is a
graph-wide invariant, and a clone that dropped it would fail `finalise()` on a graph that is
numerically correct.

## TheCloneIsInsertedAheadOfEveryConsumerItServes

The failure this case catches is not a wrong number. It is a graph whose insertion order stopped
being a topological order, with a recomputation appended after the GEMM that reads it.
`MemoryPlan::build()` rests on that order and says so by name, and a lowering that got past it
would record a GEMM against a buffer nothing had written yet.

## AnIm2colWithNoBackwardConsumerIsLeftAlone

This is the inference case. Nothing was differentiated, so there is no second consumer and
cloning would be a free slowdown.

## AForwardOnlyConvolutionIsLeftAlone

This is the other half of the inference case, and the one that would break if the pass cloned
for every consumer and not for every consumer after the first. The column matrix has exactly one
reader, the GEMM.

## RunningItTwiceAddsNothingTheSecondTime

Once the rewiring is done the clone has one consumer of its own, so the pass has no reason to
rematerialise its own output. If it did, the pass would not terminate on a graph with three
consumers.

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

## image_values

The [1, 5, 4, 2] image is not square on purpose. K_IM2COL's output extents live in attribute
slots 8 and 9. A clone goes wrong by re-deriving them where it should copy them, which can
yield the transpose of the real window grid with the same number of windows. On a square image
that mistake is invisible; on 5x4 it is a different gather.

## TheRewrittenGraphComputesTheSameGradient

The node counts are asserted first, to show the pass did something. A pass that no-ops passes
the comparison that follows trivially.

The comparison is exact, not near. The clone recomputes the same operation from the same
operands in the same order.

## TheColumnMatrixLeavesTheActivationArena

The pass exists to take the column matrix out of the activation arena. The test states that
against the planner and does not infer it from the lifetime field. The planner turns the
shortened live range into an arena decision, and it has to agree.

Suites: `Integration_Autograd`.

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
