# tests/core/autograd/test_tape.cpp


Suites: `Small_GradientTape`.

Tests `GradientTape`, the view a backward rule reads. A tape is made only by
`Autograd::differentiate` over a run whose bookkeeping class, `Differentiator`, is private to the
module, so a test cannot build one. The cases observe the tape through the nodes a rule appends.
`test_backward.cpp` in this folder covers the seed factor, accumulation at a fan-out, the shape
check on a contribution and the skipping of a fused log-softmax, all of which go through the
tape; this file pins what those cases leave out.

## appended_of

`appended_of` returns the nodes that run one operation, from a given node index on. A case
passes the node count taken before differentiation, so only appended nodes are searched.

## ATapeIsNeverDefaultConstructed

`GradientTape` is not default-constructible: a tape always belongs to a run.

## AnOperandThatNeedsNoGradientReceivesNoContribution

`product` is a K_MUL of the parameter `weight` and the input `input`, both of shape [1], and is
differentiated with respect to `weight`. One gradient comes back. The appended nodes hold one
K_MUL, the seed times `input`, and its result is that gradient. The rule asked the tape whether
each operand requires a gradient and emitted nothing for `input`. The graph carries no refusal
afterwards.

## ARefusedContributionAddsNothingAndMarksTheGraph

`product` is a K_MUL of the [1] parameter `weight` and the [1, 1] input `input`, added with
`Graph::add_node` so that no emitter checks it. The rule's `builder.mul(dy, input)` is refused,
because dy is [1]. The gradient of `weight` comes back as `K_NO_VALUE`, no K_MUL is appended,
and `graph.has_refusal()` is true, so `CompiledStep::compile` refuses the graph.
