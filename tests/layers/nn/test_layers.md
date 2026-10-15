# tests/layers/nn: test_layers.cpp, tests/core/autograd/test_backward.cpp


Suites: `Small_Backward`.

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`modules/core/graph/include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

The cases from DivReadsItsDivisorAndItsOwnQuotient on each put one operation under an K_NLL_LOSS
root and compare the appended nodes with a literal list of operation names. The first appended
node is always the K_SCALE that K_NLL_LOSS's rule emits, and its result is the gradient the tested
operation receives. The cases then check which operands each appended node reads, since a rule
that emits the right operations over the wrong operands gives a graph of the same shape.

## K_ATTRIBUTE_TOLERANCE

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule writes as an exact
constant.

## K_UNPROVEN_RULES

`K_UNPROVEN_RULES` is 8, the number of rules no finite-difference case names.
EveryRuleInTheTableIsNamedByANumericTest compares the size of its `unproven` list with it, so
adding a rule to that list takes an edit in two places.

## ForwardCase

`ForwardCase` holds a failure label and a function that emits one forward operation over two
parameters.

## Appended

`Appended` holds what one differentiation appended: the index of the first appended node, the
gradient the tested operation receives, and one gradient per requested value.

## differentiate_through

`differentiate_through` declares a `targets` input in the shape of a rank-2 result, puts
`nll_loss(result, targets)` over it and differentiates. It returns an `Appended` whose `seed`
is the first result of the first appended node.

## appended_node

`appended_node` returns the node a given number of places after the first appended one.

## EveryImplementedOperationHasARule

`has_backward_rule` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU.

`has_backward_rule` is false for K_LESS_EQUAL, K_CONSTANT and K_STOP_GRADIENT. Each is a gradient
barrier: the reverse walk steps over it, so no rule is ever asked of it.

`has_backward_rule` is false for K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD. An activation gradient
has no rule, so a second differentiation over one throws.

`has_backward_rule(K_BATCHNORM_UPDATE_STATS)` is false, and that is decision CV-7, not an
omission. A running mean is a buffer the training step writes beside the graph. No gradient
flows through it.

`has_backward_rule(K_BATCHNORM_FOLD)` is false too. The fold is the node an evaluation trace of
`NN::BatchNorm2d` emits, and DifferentiatingThroughBatchNormFoldFails holds what a gradient
request through it does.

`has_backward_rule` is false for K_LEAF and K_ZERO because both produce values from nothing. There
is nothing upstream to hand a gradient to, so there is no rule to write.

`has_backward_rule(K_CAST)` is false because of the order of the pipeline. Casts are inserted
after differentiation (`modules/core/graph/include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

## EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `modules/core/autograd/source/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::K_ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/core/autograd/rules/test_all.cpp` and all run in
FLOAT64, for the reason `modules/core/autograd/include/SushiAI/autograd/gradcheck.hpp` derives.

The K_LOG_SOFTMAX entries name the cases of `tests/language/blocks/test_causal_self_attention.cpp`:
the unfused rule against its closed form, a softmax, and a causal attention block.

The `unproven` vector lists the eight rules no finite-difference case names: K_DIV, K_MINIMUM,
K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU. Its size must equal `K_UNPROVEN_RULES`, every
entry must have a rule, and no operation may appear in both the ledger and this list. The loop
over `Ops::K_ALL` accepts a rule that either list names. An entry leaves `unproven` when a
finite-difference case names its rule in the ledger.

## TwoLayerMLPProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, K_ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

## TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one K_MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another K_MATMUL. It is skipped, not
computed and discarded.

## FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered K_ACCUMULATE form.

## ThreeContributorsChainTwoAccumulates

A parameter feeds a ReLU, a tanh and a sigmoid whose sum is the loss operand. After
`differentiate` the graph holds two ACCUMULATE nodes.

## ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no K_ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

## ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

## AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

## ANonScalarRootIsRejected

`differentiate` from a [4, 3] ReLU output throws `Error`.

## LogSoftmaxOutsideTheFusedPairAppendsItsGradientIdentity

A RELU between LOG_SOFTMAX and NLL_LOSS keeps the fusion away. The walk appends SCALE,
RELU_BACKWARD and one LOG_SOFTMAX_BACKWARD over the incoming gradient and the forward result.

## TheFusedPairAppendsNoLogSoftmaxGradient

NLL_LOSS directly over LOG_SOFTMAX appends EXP, SUB and SCALE, and no LOG_SOFTMAX_BACKWARD.

## ALogSoftmaxWhoseOperandNeedsNoGradientAddsNoNode

A log-softmax over an input, multiplied by a parameter, appends no gradient node of its own.

## ASoftmaxDifferentiatesThroughItsExponentialAndItsLogSoftmax

The LOG_SOFTMAX_BACKWARD of a softmax reads the MUL that the rule of EXP emitted and the
forward log-probabilities.

## DivReadsItsDivisorAndItsOwnQuotient

For `c = a / b` with both operands parameters, the appended nodes are K_SCALE, K_DIV, K_MUL, K_SCALE.
The K_DIV reads the seed and `b` and is the gradient of `a`. The K_MUL reads that quotient and the
forward result `c`, and the last K_SCALE negates it with the factor -1 and is the gradient of
`b`. The rule reuses `c` and does not divide by `b` a second time.

## DivEmitsOnlyTheQuotientWhenTheDivisorNeedsNoGradient

With `b` an input, the appended nodes are K_SCALE and K_DIV.

## DivEmitsTheDivisorGradientAloneWhenTheDividendNeedsNone

With `a` an input, the appended nodes are still K_SCALE, K_DIV, K_MUL, K_SCALE, because the divisor's
gradient is computed from the quotient. The one returned gradient is the result of the last
K_SCALE.

## MinimumRoutesByALessEqualBSoATieGoesToTheFirstOperand

For `minimum(a, b)` the appended nodes are K_SCALE, K_LESS_EQUAL, K_ZERO, K_SELECT, K_SELECT. The mask
reads `a` then `b`. The first K_SELECT reads the mask, the seed and the zero and is the gradient
of `a`; the second reads the mask, the zero and the seed and is the gradient of `b`. Where the
operands are equal the mask holds, so the first operand takes the gradient.

## MaximumRoutesByBLessEqualASoATieGoesToTheFirstOperand

For `maximum(a, b)` the node list is the same and the mask reads `b` then `a`. The two K_SELECT
nodes read the same operands as in the K_MINIMUM case, so a tie again goes to the first operand.

## MinimumEmitsOneSelectWhenOnlyTheSecondOperandNeedsAGradient

With `a` an input, one K_SELECT is appended, and it reads the mask, the zero and the seed.

## MaximumEmitsOneSelectWhenOnlyTheFirstOperandNeedsAGradient

With `b` an input, one K_SELECT is appended. The mask reads `b` then `a`, and the K_SELECT reads
the mask, the seed and the zero.

## AbsMultipliesByASignThatIsZeroAtZero

For `abs(x)` the appended nodes are K_SCALE, K_ZERO, K_LESS_EQUAL, K_LESS_EQUAL, K_SUB, K_MUL. The K_ZERO has
the shape of `x`. The first mask reads the zero then `x`, the second reads `x` then the zero,
the K_SUB reads the first mask then the second, and the K_MUL reads the seed and the difference.
The sign is `[0 <= x] - [x <= 0]`, which is 0 at x = 0.

## AtanDividesByOnePlusTheSquaredInput

For `atan(x)` the appended nodes are K_SCALE, K_MUL, K_CONSTANT, K_ADD, K_DIV. The K_MUL reads `x` twice.
The K_CONSTANT has no input, holds 1 and has the shape and element type of `x`. The K_ADD reads the
square and the one, and the K_DIV reads the seed and that sum.

## SoftplusBackwardReadsTheForwardInputNotTheOutput

For `softplus(x)` one K_SOFTPLUS_BACKWARD is appended after the K_SCALE, and it reads the seed and
`x`. The case asserts that `x` and the forward result are different values first, so the
operand check distinguishes the input from the output.

## SiLUBackwardReadsTheForwardInputNotTheOutput

For `silu(x)` one K_SILU_BACKWARD is appended after the K_SCALE, and it reads the seed and `x`.

## SelectRoutesItsGradientByItsOwnCondition

For `select(condition, a, b)` the appended nodes are K_SCALE, K_ZERO, K_SELECT, K_SELECT. The first
K_SELECT reads the condition, the seed and the zero and is the gradient of `a`. The second reads
the condition, the zero and the seed and is the gradient of `b`.

## SelectEmitsOneSelectWhenOnlyTheSecondBranchNeedsAGradient

With `a` an input, the appended nodes are K_SCALE, K_ZERO, K_SELECT, and the K_SELECT reads the
condition, the zero and the seed.

## SelectGivesItsConditionNoGradient

With the condition a parameter and both branches inputs, only the K_SCALE is appended and the
condition's gradient is `K_NO_VALUE`.

## AComparisonUnderASelectLeavesItsOperandsWithoutAGradient

The condition is `less_equal(p, q)` with `p` a parameter. The gradient of `p` is `K_NO_VALUE`,
the gradient of the branch `a` exists, and the appended nodes are K_SCALE, K_ZERO, K_SELECT.

## AComparisonMultipliedIntoAProductEmitsNothingForTheMask

For `mul(less_equal(p, q), a)` differentiation does not throw. The mask does not require a
gradient, the gradient of `p` is `K_NO_VALUE`, and the appended nodes are K_SCALE and one K_MUL that
reads the seed and the mask and is the gradient of `a`. K_MUL's rule emits no node for its mask
operand.

## AConstantAddedToAParameterPathEmitsNothingForTheConstant

For `(x + epsilon) * epsilon`, with `epsilon` a `constant_like` of `x`, differentiation does
not throw. The constant does not require a gradient, and the appended nodes are K_SCALE and one
K_MUL that reads the seed and the constant and is the gradient of `x`.

## MaximumOfXAndAZeroConstantRoutesATieToX

For `maximum(x, floor)` with `floor` a zero K_CONSTANT, the mask reads `floor` then `x`, and the
K_SELECT reads the mask, the seed and the zero. The mask `[floor <= x]` holds at x = 0, so `x`
takes the gradient there.

## MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie

For `maximum(floor, x)` the mask reads `x` then `floor`, and the K_SELECT reads the mask, the
zero and the seed. The mask `[x <= floor]` holds at x = 0, so `x` takes the zero branch there.
The operand order of a clamp therefore decides its gradient at the bound. The case also checks
that no node of the graph, forward or appended, narrows under the mixed-precision policy, so
the clamp stays in the loss's precision.

## NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows

For each of `div`, `minimum`, `maximum`, `abs`, `atan`, `softplus`, `silu` and a `select` on a
`less_equal` mask, the case differentiates the operation over two parameters. The rule appends
a node beyond the K_SCALE, and `Graph::narrows_under_mixed_precision` is false for every appended
node.

## AGradientOnTheBatchVarianceOfBatchNormThrows

`GraphBuilder::batchnorm2d` returns four values, and `batch_variance` is output 3 of the node.
A loss built on a K_SCALE of that value is differentiated against gamma. `differentiate` throws
with a message containing "K_BATCHNORM2D" and "saved statistics received a gradient".

## BatchNormWithFourOutputsStillDifferentiatesThroughItsFirst

A loss built on `sum_channels` of output 0 is differentiated against gamma and beta. Both
gradients are values other than `K_NO_VALUE` and the graph holds one K_BATCHNORM2D_BACKWARD node.
Inputs 3 and 4 of that node are the saved mean and the saved inverse deviation. The batch
variance is not among its operands.

## DifferentiatingThroughBatchNormFoldFails

A loss built on a K_CHANNEL_AFFINE over the two results of `batchnorm_fold` is differentiated
against gamma, a parameter. `differentiate` throws with a message containing "No backward
rule" and "sushiai.batchnorm_fold".

## ABufferInWrtIsRefusedAndAppendsNoNode

A request that lists a weight and a buffer named "running_shift" throws with a message
containing "differentiate()", "running_shift" and "buffer", and the graph's node count is
unchanged. The same graph then differentiates against the weight alone: the one gradient has
the lifetime `GRADIENT`, and the buffer keeps the lifetime `BUFFER`.

## test_layers.cpp, tests/core/autograd/test_backward.cpp


Suites: `Small_LossRoots`.

Tests which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT
barrier.

This file needs no device. Every case in `Small_LossRoots` builds a host-side graph, calls
`Autograd::differentiate` and reads the appended nodes or the error. A root produced by
K_NLL_LOSS is seeded by that operation's own rule. Any other one-element root of a real floating
type whose producer has a rule gets a K_CONSTANT of 1 as its gradient. Most refusal cases also
check that the node count is unchanged, so a refused call leaves the graph as it was.

## Constants

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule copies from the
forward node. `K_TERM_WEIGHT` is 2.0, the factor of the K_SCALE term in `WeightedSum`.
`K_ROOT_WEIGHT` is 0.5, the factor of the K_SCALE that ends a loss in the cases that need one.

## op_names

`op_names` returns the operation names of the nodes from a given index to the end. A case
passes the forward node count, so the list covers the appended nodes.

## count_op

`count_op` counts the nodes of one operation in the whole graph.

## differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. The seed factor defaults to `K_NO_VALUE` and the provenance to null.

## WeightedSum

`WeightedSum` holds the loss `K_TERM_WEIGHT * a + b * c` over three one-element parameters:
a K_SCALE, a K_MUL and the K_ADD of the two. It keeps the ids of the parameters and of the three
results.

## ASumOfTwoTermsEndingInAddReachesEveryParameter

The root is the K_ADD. Four nodes are appended: K_CONSTANT, K_MUL, K_MUL, K_SCALE. The K_CONSTANT is the
seed. The two K_MUL nodes read the seed with `c` and with `b`, and produce the gradients of `b`
and `c`. The K_SCALE reads the seed with the factor `K_TERM_WEIGHT` and produces the gradient of
`a`. All three gradients have the lifetime GRADIENT.

## ASumOfTwoTermsEndingInScaleReachesEveryParameter

The root is a K_SCALE of the sum by `K_ROOT_WEIGHT`. The appended nodes are K_CONSTANT, K_SCALE, K_MUL,
K_MUL, K_SCALE. The first K_SCALE reads the seed with the factor `K_ROOT_WEIGHT`, and the three nodes
after it read that K_SCALE's result, so both terms under the root receive the scaled seed. No
gradient is `K_NO_VALUE`.

## TheSeedIsAConstantOneShapedAndTypedLikeTheRoot

The root is a FLOAT64 [1, 1, 1, 1] value tagged NCHW. The first appended node is a K_CONSTANT
with no input and the value 1 in attribute slot 0, and its result is [1, 1, 1, 1], FLOAT64 and
NCHW. The graph holds one K_CONSTANT, no K_ZERO and no K_EXP.

## TheSeedFactorMultipliesTheSeedOnce

With a seed factor, the appended nodes are K_CONSTANT, K_DYNAMIC_SCALE, K_MUL, K_MUL, K_SCALE. The
K_DYNAMIC_SCALE reads the K_CONSTANT and the factor, the graph holds one K_DYNAMIC_SCALE, and the
three rule nodes read its result.

## ANonScalarRootThrowsNamingItsShape

A [4, 3] root throws with a message containing "needs a scalar to differentiate" and "[4, 3]".

## ALeafRootThrowsNamingTheLeaf

A parameter passed as the root throws with a message containing "it is a leaf" and
"sushiai.leaf".

## ARootThatNeedsNoGradientThrowsNamingTheCause

The root is an K_ADD of two inputs and the requested parameter is not read by it. The message
contains "requires no gradient" and "sushiai.add".

## ARootWhoseProducerHasNoRuleThrowsNamingTheOperation

The root is a K_DYNAMIC_SCALE of a parameter. The message contains "no backward rule" and
"sushiai.dynamic_scale".

## ARootThatIsNotTheFirstResultThrowsNamingTheProducer

The root is the index result of a K_MAXPOOL2D, its second output. The message contains "not the
first result" and "sushiai.maxpool2d".

## ARootThatIsNotRealFloatingThrowsNamingItsType

The root is a K_SCALE of an INT32 parameter. The message contains "element type is int32" and
"sushiai.scale".

## AFusedCrossEntropyRootEmitsNoSeedNode

A two-layer `NN::Sequential` under `NN::CrossEntropyLoss` appends K_EXP, K_SUB, K_SCALE, K_SUM_ROWS,
K_MATMUL, K_MATMUL, K_RELU_BACKWARD, K_SUM_ROWS, K_MATMUL. The graph holds no K_ZERO and no K_CONSTANT: the
fused rule starts from the root's implicit gradient and needs no seed node.

## AnUnfusedNLLLossRootEmitsNoSeedNode

K_NLL_LOSS over a K_RELU result appends K_SCALE then K_RELU_BACKWARD. The K_SCALE reads the targets
alone, so no seed node precedes it.

## AnNLLLossRootAppliesTheSeedFactorInItsOwnRule

With a seed factor, a K_LOG_SOFTMAX and K_NLL_LOSS pair appends K_EXP, K_SUB, K_SCALE, K_DYNAMIC_SCALE, and
the result of the K_DYNAMIC_SCALE is the gradient of `x`.

## StopGradientEmitsAnIdentityShapedLikeItsOperand

`stop_gradient` of a FLOAT64 NCHW value writes one K_STOP_GRADIENT node that reads the operand.
The result has the operand's shape and is FLOAT64 and NCHW, and `finalise()` accepts the graph.

## StopGradientIsDeclaredAGradientBarrierAndNotDifferentiable

`Ops::K_STOP_GRADIENT` has `blocks_gradient` true and `differentiable` false.
`Ops::is_gradient_barrier` is true for it and false for K_RESHAPE, and its name is
"sushiai.stop_gradient".

## RequiresGradDoesNotCrossStopGradient

After `finalise()`, a parameter requires a gradient and so does a K_SCALE that reads it. The
K_STOP_GRADIENT result and a K_SCALE that reads that result do not.

## StopGradientBlocksAParameterReachableOnlyThroughIt

The loss is `stop_gradient(scale(blocked)) * open`. Differentiation does not throw and appends
K_CONSTANT and K_MUL. The gradient of `blocked` is `K_NO_VALUE`, and the gradient of `open` is the
K_MUL of the seed and the stopped value.

## AValueReadDirectlyAndThroughStopGradientKeepsTheDirectGradient

The loss is `stop_gradient(x) * x`. The appended nodes are K_CONSTANT and K_MUL, the K_MUL reads the
seed and the stopped value, and its result is the gradient of `x`. The graph holds no
K_ACCUMULATE, so the stopped path contributes nothing to the sum.

## ALossReachableOnlyThroughStopGradientThrowsNamingTheBarrier

The root is a K_SCALE of `stop_gradient(x)`. The message contains "requires no gradient" and
"sushiai.scale".

## AStopGradientRootThrowsForRequiringNoGradient

The root is the K_STOP_GRADIENT result. The message contains "requires no gradient" and
"sushiai.stop_gradient".

## AContributionToAStopGradientResultIsDroppedWithoutAskingForARule

The graph is that of StopGradientBlocksAParameterReachableOnlyThroughIt, with `requires_grad`
of the stopped value set by hand after the graph is built. `finalise()` only sets the flag, so
the hand-set value survives it and K_MUL's rule contributes a gradient to the stopped value.
Differentiation does not throw. It appends K_CONSTANT, K_MUL, K_MUL, the gradient of `blocked` is
`K_NO_VALUE`, and no appended node has the K_STOP_GRADIENT node as its origin in the `Provenance`.
The reverse walk steps over the barrier and asks no rule of it.

## TheSeedNodesHaveTheRootProducerAsOrigin

`WeightedSum` is differentiated with a seed factor and a `Provenance`. Five nodes are appended.
The origin of the K_CONSTANT and of the K_DYNAMIC_SCALE is the K_ADD that produces the root. The
origin of the two K_MUL nodes is the forward K_MUL, and the origin of the K_SCALE is the forward
K_SCALE.

## AConstantRootThrowsForRequiringNoGradient

The root is a `constant_like` of a parameter. The message contains "requires no gradient" and
"sushiai.constant".

## AComparisonRootThrowsForRequiringNoGradient

The root is `less_equal(x, bound)` over a parameter. The message contains "requires no
gradient" and "sushiai.less_equal".

## RequiresGradCrossesNoGradientBarrier

After `finalise()`, a K_LESS_EQUAL mask over a parameter, a K_CONSTANT, a K_ZERO and the K_MUL of the
mask with the K_CONSTANT require no gradient. The K_MUL of the mask with the parameter does,
through its second operand.
