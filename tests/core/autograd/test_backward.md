# tests/core/autograd/test_backward.cpp


Suites: `Small_Layout`.

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

## EveryValueIsNHWCUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

## FinaliseAcceptsAUniformGraph

`finalise` returns a success on a graph of one MATMUL over a [2, 4] input and a [4, 3] parameter.

## FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data. The test sets the result's layout by hand, which no emitter
does, so `finalise()` stops the process and the report names the MATMUL.

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

## CastInsertionReturnsMixedOperandsAndLeavesTheGraphAlone

`make_mixed_operands` adds an NHWC input to its own NCHW conversion, which no emitter refuses.
`insert_casts()` proves its input before it rewrites anything, so it returns
`InvalidDescription` naming the ADD and the node count is unchanged.

## FusionStopsOnAGraphWhoseLayoutsAreNotProven

`fuse()` returns a report and no status, and takes a proven graph as a precondition. On the
same mixed graph it stops the process, and the report names the precondition and the proof's
message.

## TheThreeRulesThatEmitZerosCannotBeReachedWithAnNCHWOperand

The test checks why the zeros exposure is latent and not live, and does not leave that to a
comment. It is the premise the `zeros_like()` change rests on, and it is a premise about other
emitters' guards, which is the kind that stops being true without anyone editing this file.

The two `SA_EXPECT_FAILURE` lines on `slice_channels` and `channel_affine` cover the rank-4 sites,
`slice_channels_backward` and `channel_affine_backward`. CV-3 makes their forward emitters
refuse an NCHW operand outright, so their dy can never carry that tag.

The `reshape` to `rows` covers the rank-2 site, `sum_rows_backward`. Reshape carries an NCHW
tag into rank-2 land with no guard at all, so unlike the two rank-4 sites the emitters let this
chain be built.

The last `SA_EXPECT_FAILURE` on `graph.finalise()` shows what blocks that chain: the far end.
`differentiate()` only seeds a loss produced by K_NLL_LOSS, whose one-hot operand is a leaf and so
always NHWC, and nothing but K_LAYOUT_CONVERT may cross back, which is rank-4 only. `finalise()`
returns `InvalidDescription` for that node before differentiation ever runs, so the rule is never reached with an NCHW
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

Suites: `Small_Precision`, `Small_CastInsertion`.

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

## ANonFloatLowTypeIsRefusedByMake

`MixedPrecisionPolicy::make(DType::Int32)` and `make(DType::Complex32)` return
`InvalidDescription` naming a real floating type, and `make(DType::Half)` succeeds.

## EveryChoiceCarriesAReasonSomebodyCanRead

For every entry of `Ops::K_ALL`, the choice of `MixedPrecisionPolicy` over `Float32` has a
non-empty `reason`.

## TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Small_CastInsertion cases measures nothing.

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

## AGraphThatIsNotInTopologicalOrderIsFatal

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first. The pass stops the process with a report naming "Node 0 reads value 1" and "which
node 1 writes later".

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

Suites: `Small_Rematerialise`.

Tests the structure of the graph that the K_IM2COL rematerialisation pass rewrites.

This file needs no device. The K_IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Medium_Rematerialise in `tests/core/graph/test_rematerialise.cpp`
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

## AGraphThatHoldsARefusalIsRefusedBeforeAnyNodeIsAppended

`slice_channels` refuses an NCHW operand, and the refusal sets the graph's flag. The loss is then
built from the NHWC parameter and is differentiable on its own, so only the flag stops
`differentiate()`. It returns `InvalidDescription` and appends no node.
