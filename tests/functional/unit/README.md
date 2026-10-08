# Unit tests

The unit tier runs without a device. Every file in this folder builds a graph, a config
document, a byte stream, an image or a plain array on the host and inspects the result, so no
case needs an Engine, a TensorPool or SYCL. A few cases touch the disk. The path-resolution
cases of `test_config.cpp` use config files on disk. `test_imaging_raw_cache.cpp` writes small
files under the system's temporary directory and removes them. The `test_detect_*.cpp` cases
that compare with the reference, and the two reference cases of `test_batch_norm2d.cpp`,
read tensors from `tests/fixtures/yolo26/` and report themselves skipped when a file is
absent. The YOLO26n cases of `test_node_registry.cpp` read `examples/yolo26/model.json` and
`tests/fixtures/yolo26/MANIFEST.json`.

This page holds the reasoning behind the cases. A `//` line in a test states the invariant the
case relies on and cites this page. The section under the test's name says why the invariant
matters, which number it was measured at, or which failure it once let through. Where a test has
several such lines, each paragraph opens with the assertion or the statement it explains.
Sections follow the files in path order, and inside a file the headings follow the order of the
tests.

## Files

| File | What it tests |
| --- | --- |
| [`test_avg_pool2d.cpp`](test_avg_pool2d.cpp) | The shapes the AvgPool2d layer traces and reports, and the inputs it refuses. |
| [`test_backward.cpp`](test_backward.cpp) | The graph that differentiation emits and the coverage of the backward rule table. |
| [`test_batch_norm2d.cpp`](test_batch_norm2d.cpp) | BatchNorm2d's two trace forms and the running-statistic update. |
| [`test_block_bottleneck.cpp`](test_block_bottleneck.cpp) | Bottleneck's trace, its state names against YOLO26n row 2, and its factory. |
| [`test_block_c3k.cpp`](test_block_c3k.cpp) | C3k's trace, its state names against YOLO26n row 6, and its factory. |
| [`test_block_c3k2.cpp`](test_block_c3k2.cpp) | C3k2's trace, its state names against YOLO26n rows 2 and 6, and its factories. |
| [`test_block_conv_norm.cpp`](test_block_conv_norm.cpp) | ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node. |
| [`test_block_sppf.cpp`](test_block_sppf.cpp) | SPPF's trace, its state names against YOLO26n row 9, and its factory. |
| [`test_buffer.cpp`](test_buffer.cpp) | Buffer: its description, its one leaf per graph and its identity. |
| [`test_checkpoint_format.cpp`](test_checkpoint_format.cpp) | The checkpoint format's round trip and its refusal of each malformed byte pattern. |
| [`test_concat.cpp`](test_concat.cpp) | The Concat node's shape check, its port and the node it traces. |
| [`test_config.cpp`](test_config.cpp) | How the config layer parses its JSON documents and which ones it refuses. |
| [`test_conv2d.cpp`](test_conv2d.cpp) | The Conv2d layer's trace, parameters, shape inference and rejections. |
| [`test_detect_anchors.cpp`](test_detect_anchors.cpp) | The anchor grid, the per-image anchor leaves and the copies between a level and the joined anchor axis. |
| [`test_detect_assigner.cpp`](test_detect_assigner.cpp) | The task-aligned assigner, its presets, the padded ground truth and the target leaves. |
| [`test_detect_average_precision.cpp`](test_detect_average_precision.cpp) | Detection matching, the 101-point average precision and the validation table. |
| [`test_detect_boxes.cpp`](test_detect_boxes.cpp) | Box decoding and encoding, the two box formats and the IoU forms. |
| [`test_detect_postprocess.cpp`](test_detect_postprocess.cpp) | The top-k selection, both postprocess paths and the overlap suppression. |
| [`test_diagnostics.cpp`](test_diagnostics.cpp) | That each host-lane precondition raises a message carrying the values that disagree. |
| [`test_elementwise_ops.cpp`](test_elementwise_ops.cpp) | The nodes the builder emits for the elementwise, comparison, selection and constant operations. |
| [`test_flatten.cpp`](test_flatten.cpp) | The Flatten layer's trace, shape rule, rejections and parameter walk. |
| [`test_fusion.cpp`](test_fusion.cpp) | The matches and the refusals of the static fusion pass on synthetic graphs. |
| [`test_global_avg_pool.cpp`](test_global_avg_pool.cpp) | The shapes GlobalAvgPool traces and reports, and the inputs it refuses. |
| [`test_gradcheck.cpp`](test_gradcheck.cpp) | The gradient checker against closed-form functions and a wrong gradient. |
| [`test_identity.cpp`](test_identity.cpp) | That the Identity layer returns its input, emits nothing and holds no state. |
| [`test_imaging_affine.cpp`](test_imaging_affine.cpp) | The plan and the application of the scale and translation warp to an image and its labels. |
| [`test_imaging_boxes.cpp`](test_imaging_boxes.cpp) | The two box forms, their conversions and the candidate filter. |
| [`test_imaging_flip.cpp`](test_imaging_flip.cpp) | The plan and the application of the horizontal flip to an image and its labels. |
| [`test_imaging_gray.cpp`](test_imaging_gray.cpp) | The expansion of a one-channel image to three equal channels. |
| [`test_imaging_hsv.cpp`](test_imaging_hsv.cpp) | The hue, saturation and value jitter: its plan, its tables and both colour conversions. |
| [`test_imaging_image.cpp`](test_imaging_image.cpp) | The owned image, the view over foreign pixels and the checks both share. |
| [`test_imaging_label_set.cpp`](test_imaging_label_set.cpp) | The label set that holds classes, boxes and polygons in step, and its conversions. |
| [`test_imaging_labels.cpp`](test_imaging_labels.cpp) | The parser of a label file's text. |
| [`test_imaging_letterbox.cpp`](test_imaging_letterbox.cpp) | The plan and the application of the letterbox to an image and its labels. |
| [`test_imaging_mosaic.cpp`](test_imaging_mosaic.cpp) | The draw, the plan and the application of the four-image mosaic. |
| [`test_imaging_partners.cpp`](test_imaging_partners.cpp) | The two sources of a mosaic's three partner images. |
| [`test_imaging_pipeline.cpp`](test_imaging_pipeline.cpp) | The training and the validation pipeline that compose the imaging transforms into a sample. |
| [`test_imaging_polygon.cpp`](test_imaging_polygon.cpp) | Polygon resampling, mapping, clamping, the point test and the visible box. |
| [`test_imaging_random.cpp`](test_imaging_random.cpp) | The generator every augmentation draws from. |
| [`test_imaging_raw_cache.cpp`](test_imaging_raw_cache.cpp) | The raw image file: its bytes, its header checks and its file reader and writer. |
| [`test_imaging_rectangle.cpp`](test_imaging_rectangle.cpp) | The plan that orders a dataset by aspect ratio and gives each batch one shape. |
| [`test_imaging_resize.cpp`](test_imaging_resize.cpp) | The bilinear resize and the fit of an image's long side. |
| [`test_ir.cpp`](test_ir.cpp) | The IR's single-assignment and ordering rules and the builder's shape inference. |
| [`test_layer_registry.cpp`](test_layer_registry.cpp) | Each layer registry row's factory, the keys it reads and the model it builds. |
| [`test_layer_stack.cpp`](test_layer_stack.cpp) | The runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread. |
| [`test_layout.cpp`](test_layout.cpp) | That every value carries a tensor layout and that the rebuilding passes keep it. |
| [`test_leaky_relu.cpp`](test_leaky_relu.cpp) | The LeakyReLU layer's trace, shape inference and slope range. |
| [`test_loss_roots.cpp`](test_loss_roots.cpp) | Which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT barrier. |
| [`test_loss_scaling.cpp`](test_loss_scaling.cpp) | The dynamic loss-scaling rule on LossScaleState, with no device involved. |
| [`test_max_pool2d.cpp`](test_max_pool2d.cpp) | The MaxPool2d layer's trace, shape inference and rejections. |
| [`test_memory_plan.cpp`](test_memory_plan.cpp) | That the memory planner never shares a byte between two values live together. |
| [`test_model_graph.cpp`](test_model_graph.cpp) | ModelGraph: what add and output refuse, what a trace emits, and the state names. |
| [`test_model_outputs.cpp`](test_model_outputs.cpp) | ModelOutputs: publishing named values and reading them back. |
| [`test_module.cpp`](test_module.cpp) | Parameter across graphs and the state walk over layers, containers and fields. |
| [`test_module_list.cpp`](test_module_list.cpp) | ModuleList: its storage, its index names and its place in the state walk. |
| [`test_node.cpp`](test_node.cpp) | ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time. |
| [`test_node_registry.cpp`](test_node_registry.cpp) | The node registry, make_node, build_model and the YOLO26n model file. |
| [`test_objective.cpp`](test_objective.cpp) | The objective seam: the leaves it declares, the loss it emits and the resulting IR. |
| [`test_ops.cpp`](test_ops.cpp) | That the operation table is complete and that the reverse lookup resolves names. |
| [`test_precision.cpp`](test_precision.cpp) | The mixed-precision policy and where the cast insertion pass places its casts. |
| [`test_provenance.cpp`](test_provenance.cpp) | The forward origin that differentiation records for every node it appends. |
| [`test_region_signature.cpp`](test_region_signature.cpp) | The signature that `signature_of` computes for one region of a partition. |
| [`test_rematerialise.cpp`](test_rematerialise.cpp) | The structure of the graph that the K_IM2COL rematerialisation pass rewrites. |
| [`test_result.cpp`](test_result.cpp) | Result, its reference form and SA_TRY_ASSIGN. |
| [`test_rule_table.cpp`](test_rule_table.cpp) | That every domain's backward rules reach the combined table exactly once. |
| [`test_scope_partitioner.cpp`](test_scope_partitioner.cpp) | The region keys ScopePartitioner gives the nodes of a differentiated graph. |
| [`test_scope_trace.cpp`](test_scope_trace.cpp) | How ScopeTrace cuts a trace into segments and which calls it rejects. |
| [`test_shape.cpp`](test_shape.cpp) | Shape's extents, indexing, validation, equality and hash. |
| [`test_silu.cpp`](test_silu.cpp) | The SiLU layer's trace, shape inference and empty state. |
| [`test_spatial_shapes.cpp`](test_spatial_shapes.cpp) | The shapes the builder infers for spatial operations and the layouts it refuses. |
| [`test_status.cpp`](test_status.cpp) | Status, its macros, the fatal family and the assertions of test_status.hpp. |
| [`test_step_recorder.cpp`](test_step_recorder.cpp) | That StepRecorderList drives its members as one recorder, in the order added. |
| [`test_throwing_bridge.cpp`](test_throwing_bridge.cpp) | bridge_throwing: a value passes through and a thrown Error becomes a Status. |
| [`test_trace.cpp`](test_trace.cpp) | Trace, the buffer-update sinks, the three layer forms and trace_layer. |
| [`test_traced.cpp`](test_traced.cpp) | Traced and the refusal it carries through a chain of emitters. |
| [`test_unflatten.cpp`](test_unflatten.cpp) | The trace, the shape rule and the rejected inputs of the Unflatten layer. |
| [`test_upsample2d.cpp`](test_upsample2d.cpp) | The shape, layout and rejections of the Upsample2d layer. |
| [`test_version.cpp`](test_version.cpp) | That the runtime version agrees with the compile-time macros. |

## test_avg_pool2d.cpp

Tests the shapes the AvgPool2d layer traces and reports, and the inputs it refuses.

This file needs no device. `NN::AvgPool2d` in `include/SushiAI/nn/avg_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter, so a case either traces the layer into a host-side
graph or calls `output_shape()` on a per-sample shape.

### window

`window` returns a geometry with the given kernel and stride on each axis, dilation 1 and no
padding. A case that needs padding or dilation sets the field afterwards.

### traced_shape

`traced_shape` traces a layer over a batch of two images of the given per-sample shape and
returns the shape of the result, batch axis included.

### per_sample

`per_sample` drops the batch axis of a rank-4 shape.

### SatisfiesTheModuleTrait

Four `static_assert` lines hold the layer's place in the layer library: `AvgPool2d` is a
`Module`, `const AvgPool2d` is one too, which requires a const `forward()`, it is not a
`ModuleContainer`, and it has no parameter walk of its own, so the walk reaches it through
`fields()`.

### ForwardTracesThePooledShapeInNhwc

A 3x3 window with stride 3 over a [2, 6, 6, 3] input traces a [2, 2, 2, 3] value tagged NHWC.

### OutputShapeAgreesWithTheTracedShapeForASquareWindow

`output_shape([6, 6, 3])` is [2, 2, 3], and it equals the traced shape with the batch axis
dropped. The literal is asserted beside the comparison so the two cannot agree on a wrong
shape. The next two cases have the same form.

### OutputShapeAgreesWithTheTracedShapeForAPaddedUnevenWindow

A 3x2 window with stride 2 by 1 and one pixel of padding on each axis maps [7, 5, 4] to
[4, 6, 4].

### OutputShapeAgreesWithTheTracedShapeForADilatedWindow

A 3x3 unit-stride window dilated by 2 on the height axis maps [9, 8, 1] to [5, 6, 1]. The two
axes have different dilations, so an implementation that applied one axis's dilation to both
gives another shape.

### OutputShapeRejectsAFeatureVector

`output_shape([8])` throws, because a per-sample shape must be rank 3.

### OutputShapeRejectsABatchedShape

`output_shape([2, 4, 4, 3])` throws. A shape that still carries the batch axis is rank 4.

### OutputShapeRejectsAWindowLargerThanTheImage

A 5x5 window over a [4, 4, 3] sample throws.

### OutputShapeRejectsAZeroStride

A geometry with stride 0 on both axes throws.

### OutputShapeNamesAZeroStride

The message for the same geometry contains "AvgPool2d" and "stride 0,0". The case holds the
message to the stride, where a check ordered differently would report a window that does not
fit.

### OutputShapeRejectsANegativePadding

A top padding of -1 throws with a message containing "AvgPool2d" and "padding -1,0".

### ForwardRejectsARankTwoInput

`forward()` throws on a [2, 8] input.

### ForwardRejectsAnNchwInput

A [1, 6, 6, 3] input is converted with `layout_convert` to NCHW, and `forward()` throws on the
converted value.

### ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with stride 0 on both axes throws with a message containing "AvgPool2d" and
"stride 0,0".

### ForwardRejectsANegativePadding

`forward()` with a top padding of -1 throws with a message containing "AvgPool2d" and
"padding -1,0".

### ForwardRejectsAWindowLargerThanTheImage

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ATracedLayerHasNoParameter

After a trace the parameter walk visits nothing and `parameter_count` is 0. The graph holds
two values, the input and the pooled result.

## test_backward.cpp

Tests the graph that differentiation emits and the coverage of the backward rule table.

This file needs no device. Differentiation is an IR-to-IR transform, so what it produces is a
graph that can be inspected node by node with no hardware involved. That is why
`include/SushiAI/graph/ir.hpp` insists that a backward rule emits IR and never device work.

The cases from DivReadsItsDivisorAndItsOwnQuotient on each put one operation under an K_NLL_LOSS
root and compare the appended nodes with a literal list of operation names. The first appended
node is always the K_SCALE that K_NLL_LOSS's rule emits, and its result is the gradient the tested
operation receives. The cases then check which operands each appended node reads, since a rule
that emits the right operations over the wrong operands gives a graph of the same shape.

### K_ATTRIBUTE_TOLERANCE

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule writes as an exact
constant.

### K_UNPROVEN_RULES

`K_UNPROVEN_RULES` is 8, the number of rules no finite-difference case names.
EveryRuleInTheTableIsNamedByANumericTest compares the size of its `unproven` list with it, so
adding a rule to that list takes an edit in two places.

### ForwardCase

`ForwardCase` holds a failure label and a function that emits one forward operation over two
parameters.

### Appended

`Appended` holds what one differentiation appended: the index of the first appended node, the
gradient the tested operation receives, and one gradient per requested value.

### differentiate_through

`differentiate_through` declares a `targets` input in the shape of a rank-2 result, puts
`nll_loss(result, targets)` over it and differentiates. It returns an `Appended` whose `seed`
is the first result of the first appended node.

### appended_node

`appended_node` returns the node a given number of places after the first appended one.

### EveryImplementedOperationHasARule

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
after differentiation (`include/SushiAI/graph/cast_insertion.hpp` says why), so no graph this
transform sees can contain one. A graph that somehow did would fail loudly and name the
operation. It would not differentiate a conversion whose precision the pass had not established
yet.

### EveryRuleInTheTableIsNamedByANumericTest

The `ledger` vector records coverage; the test is not a tautology. "Every operation has a rule"
is the easy half. The half that rots is "every rule has been checked against a finite
difference". `differentiate()` picks up a rule added to a `src/autograd/rules/<domain>.cpp`
table automatically, so a wrong rule produces a model that trains slightly worse and nothing
anywhere fails. Listing the proof beside the rule makes the omission visible: the test's second
loop, the one over `Ops::K_ALL`, fails when an operation gains a rule that nobody added a ledger
line for.

The named tests live in `tests/functional/integration/test_autograd_numeric.cpp` and all run in
FLOAT64, for the reason `include/SushiAI/autograd/gradcheck.hpp` derives.

The K_LOG_SOFTMAX entry names a message test. K_LOG_SOFTMAX's rule exists only to produce a good
error, so there is no gradient to verify and the message test is its proof.

The `unproven` vector lists the eight rules no finite-difference case names: K_DIV, K_MINIMUM,
K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU. Its size must equal `K_UNPROVEN_RULES`, every
entry must have a rule, and no operation may appear in both the ledger and this list. The loop
over `Ops::K_ALL` accepts a rule that either list names. An entry leaves `unproven` when a
finite-difference case names its rule in the ledger.

### TwoLayerMlpProducesTheExpectedBackwardTopology

The `op_names` comparison expects the cross-entropy pair to be differentiated as one fused
rule: exp of the log-probabilities, minus the one-hot targets, scaled by 1/N. The chain then
walks back through the second linear layer, the activation and the first linear layer.

`count_op(graph, K_ACCUMULATE)` is 0 because every value in a plain feed-forward stack has
exactly one contributor, so the tiered accumulation emits nothing.

### TheInputIsNotDifferentiatedWhenItDoesNotRequireOne

The backward pass holds one K_MATMUL, the weight gradient, which with the forward one makes the
two the test counts. The input gradient would have been another K_MATMUL. It is skipped, not
computed and discarded.

### FanOutEmitsAnAccumulateChain

`x` feeds two branches, so its gradient has two contributors and the tiered accumulation drops
to its ordered K_ACCUMULATE form.

### ADisconnectedParameterGetsASymbolicZeroNotABuffer

A zero gradient must cost nothing: no K_ZERO node, no allocation, no wasted kernel. Only the live
path emitted anything, and what it emitted is the loss gradient and the activation gradient.

### ADeadLossPathEmitsNothingAtAll

Nothing in this graph requires a gradient, so the reverse walk visits every node and writes
none of them. A frozen model costs zero nodes. It does not cost a set of nodes computing
gradients nobody reads.

### AnOperationWithNoRuleThrowsAndNamesItself

The `mystery` value is produced by a synthetic identity that no rule covers. It stands in for a
future operation added to the lowering but not to the rule table.

### LogSoftmaxOutsideTheFusedPairSaysSoClearly

A K_RELU consumes the log-softmax result, so the cross-entropy fusion does not apply and the
standalone rule has to refuse.

### DivReadsItsDivisorAndItsOwnQuotient

For `c = a / b` with both operands parameters, the appended nodes are K_SCALE, K_DIV, K_MUL, K_SCALE.
The K_DIV reads the seed and `b` and is the gradient of `a`. The K_MUL reads that quotient and the
forward result `c`, and the last K_SCALE negates it with the factor -1 and is the gradient of
`b`. The rule reuses `c` and does not divide by `b` a second time.

### DivEmitsOnlyTheQuotientWhenTheDivisorNeedsNoGradient

With `b` an input, the appended nodes are K_SCALE and K_DIV.

### DivEmitsTheDivisorGradientAloneWhenTheDividendNeedsNone

With `a` an input, the appended nodes are still K_SCALE, K_DIV, K_MUL, K_SCALE, because the divisor's
gradient is computed from the quotient. The one returned gradient is the result of the last
K_SCALE.

### MinimumRoutesByALessEqualBSoATieGoesToTheFirstOperand

For `minimum(a, b)` the appended nodes are K_SCALE, K_LESS_EQUAL, K_ZERO, K_SELECT, K_SELECT. The mask
reads `a` then `b`. The first K_SELECT reads the mask, the seed and the zero and is the gradient
of `a`; the second reads the mask, the zero and the seed and is the gradient of `b`. Where the
operands are equal the mask holds, so the first operand takes the gradient.

### MaximumRoutesByBLessEqualASoATieGoesToTheFirstOperand

For `maximum(a, b)` the node list is the same and the mask reads `b` then `a`. The two K_SELECT
nodes read the same operands as in the K_MINIMUM case, so a tie again goes to the first operand.

### MinimumEmitsOneSelectWhenOnlyTheSecondOperandNeedsAGradient

With `a` an input, one K_SELECT is appended, and it reads the mask, the zero and the seed.

### MaximumEmitsOneSelectWhenOnlyTheFirstOperandNeedsAGradient

With `b` an input, one K_SELECT is appended. The mask reads `b` then `a`, and the K_SELECT reads
the mask, the seed and the zero.

### AbsMultipliesByASignThatIsZeroAtZero

For `abs(x)` the appended nodes are K_SCALE, K_ZERO, K_LESS_EQUAL, K_LESS_EQUAL, K_SUB, K_MUL. The K_ZERO has
the shape of `x`. The first mask reads the zero then `x`, the second reads `x` then the zero,
the K_SUB reads the first mask then the second, and the K_MUL reads the seed and the difference.
The sign is `[0 <= x] - [x <= 0]`, which is 0 at x = 0.

### AtanDividesByOnePlusTheSquaredInput

For `atan(x)` the appended nodes are K_SCALE, K_MUL, K_CONSTANT, K_ADD, K_DIV. The K_MUL reads `x` twice.
The K_CONSTANT has no input, holds 1 and has the shape and element type of `x`. The K_ADD reads the
square and the one, and the K_DIV reads the seed and that sum.

### SoftplusBackwardReadsTheForwardInputNotTheOutput

For `softplus(x)` one K_SOFTPLUS_BACKWARD is appended after the K_SCALE, and it reads the seed and
`x`. The case asserts that `x` and the forward result are different values first, so the
operand check distinguishes the input from the output.

### SiluBackwardReadsTheForwardInputNotTheOutput

For `silu(x)` one K_SILU_BACKWARD is appended after the K_SCALE, and it reads the seed and `x`.

### SelectRoutesItsGradientByItsOwnCondition

For `select(condition, a, b)` the appended nodes are K_SCALE, K_ZERO, K_SELECT, K_SELECT. The first
K_SELECT reads the condition, the seed and the zero and is the gradient of `a`. The second reads
the condition, the zero and the seed and is the gradient of `b`.

### SelectEmitsOneSelectWhenOnlyTheSecondBranchNeedsAGradient

With `a` an input, the appended nodes are K_SCALE, K_ZERO, K_SELECT, and the K_SELECT reads the
condition, the zero and the seed.

### SelectGivesItsConditionNoGradient

With the condition a parameter and both branches inputs, only the K_SCALE is appended and the
condition's gradient is `K_NO_VALUE`.

### AComparisonUnderASelectLeavesItsOperandsWithoutAGradient

The condition is `less_equal(p, q)` with `p` a parameter. The gradient of `p` is `K_NO_VALUE`,
the gradient of the branch `a` exists, and the appended nodes are K_SCALE, K_ZERO, K_SELECT.

### AComparisonMultipliedIntoAProductEmitsNothingForTheMask

For `mul(less_equal(p, q), a)` differentiation does not throw. The mask does not require a
gradient, the gradient of `p` is `K_NO_VALUE`, and the appended nodes are K_SCALE and one K_MUL that
reads the seed and the mask and is the gradient of `a`. K_MUL's rule emits no node for its mask
operand.

### AConstantAddedToAParameterPathEmitsNothingForTheConstant

For `(x + epsilon) * epsilon`, with `epsilon` a `constant_like` of `x`, differentiation does
not throw. The constant does not require a gradient, and the appended nodes are K_SCALE and one
K_MUL that reads the seed and the constant and is the gradient of `x`.

### MaximumOfXAndAZeroConstantRoutesATieToX

For `maximum(x, floor)` with `floor` a zero K_CONSTANT, the mask reads `floor` then `x`, and the
K_SELECT reads the mask, the seed and the zero. The mask `[floor <= x]` holds at x = 0, so `x`
takes the gradient there.

### MaximumOfAZeroConstantAndXGivesTheClampedOperandNothingAtATie

For `maximum(floor, x)` the mask reads `x` then `floor`, and the K_SELECT reads the mask, the
zero and the seed. The mask `[x <= floor]` holds at x = 0, so `x` takes the zero branch there.
The operand order of a clamp therefore decides its gradient at the bound. The case also checks
that no node of the graph, forward or appended, narrows under the mixed-precision policy, so
the clamp stays in the loss's precision.

### NoRuleOfTheElementwiseOperationsAppendsANodeThatNarrows

For each of `div`, `minimum`, `maximum`, `abs`, `atan`, `softplus`, `silu` and a `select` on a
`less_equal` mask, the case differentiates the operation over two parameters. The rule appends
a node beyond the K_SCALE, and `Graph::narrows_under_mixed_precision` is false for every appended
node.

### AGradientOnTheBatchVarianceOfBatchNormThrows

`GraphBuilder::batchnorm2d` returns four values, and `batch_variance` is output 3 of the node.
A loss built on a K_SCALE of that value is differentiated against gamma. `differentiate` throws
with a message containing "K_BATCHNORM2D" and "saved statistics received a gradient".

### BatchNormWithFourOutputsStillDifferentiatesThroughItsFirst

A loss built on `sum_channels` of output 0 is differentiated against gamma and beta. Both
gradients are values other than `K_NO_VALUE` and the graph holds one K_BATCHNORM2D_BACKWARD node.
Inputs 3 and 4 of that node are the saved mean and the saved inverse deviation. The batch
variance is not among its operands.

### DifferentiatingThroughBatchNormFoldFails

A loss built on a K_CHANNEL_AFFINE over the two results of `batchnorm_fold` is differentiated
against gamma, a parameter. `differentiate` throws with a message containing "No backward
rule" and "sushiai.batchnorm_fold".

### ABufferInWrtIsRefusedAndAppendsNoNode

A request that lists a weight and a buffer named "running_shift" throws with a message
containing "differentiate()", "running_shift" and "buffer", and the graph's node count is
unchanged. The same graph then differentiates against the weight alone: the one gradient has
the lifetime `GRADIENT`, and the buffer keeps the lifetime `BUFFER`.

## test_batch_norm2d.cpp

Tests BatchNorm2d's two trace forms and the running-statistic update.

This file needs no device. `NN::BatchNorm2d` in `include/SushiAI/nn/batch_norm2d.hpp` is a
modal layer. `trace_training` emits one K_BATCHNORM2D node and requests two buffer updates
through the trace. `trace_evaluation` reads the two running statistics as buffer leaves and
emits a K_BATCHNORM_FOLD and a K_CHANNEL_AFFINE. Both forms record a refusal on the trace and
return their input.

The last two cases read `tests/fixtures/yolo26/batchnorm.sacp` through
`tests/common/reference_fixture.hpp` and report themselves skipped when the file is absent.
They compute the updated statistics on the host from the momentum and the source scale the
trace requested. No device runs the update.

Two `static_assert` lines after the helpers hold that `BatchNorm2d` is a `ModalModule` and
neither a `Module` nor a `TraceModule`, and that `Sequential<Conv2d, BatchNorm2d>` is a
`TraceModule` and not a `Module`.

### Constants

`K_REFERENCE_EPSILON` is 1e-3 and `K_REFERENCE_MOMENTUM` is 0.03, the two values the reference
layers ran with. `K_UNIT_CASE_TOLERANCE` is 5.3e-7 and `K_MODEL_CASE_TOLERANCE` is 7.0e-7. Each
is four float32 epsilons of the case's largest running variance, 1.1086 and 1.4578.
`K_EXACT_TOLERANCE` is 1e-15, for a double computed the same way on both sides.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### only_node

`only_node` returns the first node of a graph that carries one operation. Each caller has
asserted a count of one before it.

### count_buffer_leaves

`count_buffer_leaves` counts the values of a graph whose lifetime is `BUFFER`.

### ChannelStatistics

`ChannelStatistics` holds the per-channel mean and biased variance of one NHWC activation.

### batch_statistics

`batch_statistics` computes a `ChannelStatistics` over batch, height and width from a flat
NHWC array, in double.

### updated_vector

`updated_vector` applies `NN::updated_statistic` per channel and rounds each result to
float32.

### view

`view` returns a `SushiRuntime::span<const float>` over a vector, the argument
`Test::near_reference` takes.

### expect_reference_case

`expect_reference_case` reads one case of `batchnorm.sacp`: an input and the running mean and
variance before and after one torch training step. It traces a layer of the input's channel
count with the reference epsilon and momentum, and applies the two requested updates to the
"before" statistics on the host. The results must be within the tolerance of the "after"
statistics. The variance recomputed with a source scale of 1.0 must not be, so the fixture
tells the unbiased update from the biased one.

### refused_training

`refused_training` traces the training form, expects the input id back and returns the trace's
`status()`.

### refused_evaluation

`refused_evaluation` does the same for the evaluation form.

### train_under_an_evaluation_trace

`train_under_an_evaluation_trace` calls `trace_training` with an evaluation trace. It is the
statement of a death test.

### evaluate_over_a_value_the_graph_lacks

`evaluate_over_a_value_the_graph_lacks` calls `trace_evaluation` over value 7 in a graph of
one value. It is the statement of a death test.

### ATrainingTraceEmitsOneBatchNorm

A training trace over [2, 2, 2, 3] emits one K_BATCHNORM2D node and no K_BATCHNORM_FOLD or
K_CHANNEL_AFFINE. The graph gains three nodes: the two parameter leaves and the K_BATCHNORM2D. The
node reads the input, the weight's leaf and the bias's leaf, has four outputs of which the
first is the returned value, and carries the epsilon 1e-3 in attribute 0.

### ATrainingTraceReadsNoBuffer

After a training trace the graph holds no value of lifetime `BUFFER`, and neither running
statistic has a leaf in it.

### ATrainingTraceRequestsTheTwoRunningUpdates

The list holds two updates. The first reads output 1 of the node and targets the running
mean's state, and the second reads output 3 and targets the running variance's state. The two
states are distinct and neither is `K_NO_STATE`. Both momenta are 0.03 and the mean's source
scale is 1.0.

### TheVarianceUpdateCarriesTheUnbiasedFactor

The variance update's source scale is n / (n - 1), where n is the number of values per channel.
For [2, 2, 2, 3] it is 8/7, and for [1, 20, 20, 128] it is 400/399.

### AnEvaluationTraceFoldsTheRunningStatistics

An evaluation trace over [2, 4, 4, 3] emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE and no
K_BATCHNORM2D, and the graph holds two buffer leaves. It gains six nodes: two parameter leaves,
two buffer leaves, the fold and the affine. The fold reads the weight, the bias, the running
mean and the running variance in that order and carries the epsilon 1e-3. The affine reads the
input and the fold's two outputs, and its output is the returned value of shape [2, 4, 4, 3].

### BothFormsNameTheSameState

One layer is traced in training form into one graph and in evaluation form into another. In
the evaluation graph, `state_leaf` of each update's target is the leaf of the matching running
statistic. The weight's and the bias's states are unchanged by the second trace, and each has
a leaf in the evaluation graph.

### TheWalkYieldsTwoParametersThenTwoBuffers

`NN::walk_state(layer, "bn", visitor)` over a traced five-channel layer yields "bn.weight",
"bn.bias", "bn.running_mean" and "bn.running_var" in that order, each of shape [5]. Both
parameters have a constant fill. The starting values are 1.0 for the weight, 0.0 for the bias,
0.0 for the running mean and 1.0 for the running variance.

### TrainingOverOneValuePerChannelFails

A training trace over [1, 1, 1, 3] records `InvalidDescription` with a message containing
"BatchNorm2d(3)", "more than one value per channel" and "got 1". The graph gains no value and
no node, the list holds no update, the weight is not declared, and the graph is marked refused.

### AnIntegerInputFails

Over an INT32 input, both forms record `InvalidDescription` with a message containing
"BatchNorm2d(3)", "half, float32 or float64 input" and "int32". Nothing is added to the graph
or the list, and neither the weight nor the running mean is declared.

### EvaluationOverOneValuePerChannelTraces

An evaluation trace over [1, 1, 1, 3] returns a value of that shape. The limit that
TrainingOverOneValuePerChannelFails holds belongs to the training form.

### AnInputThatDoesNotFitFails

Over a [2, 2, 2, 4] input both forms fail with a message containing "BatchNorm2d(3)" and
"[N, H, W, 3]". Over a [2, 3] input both fail with "rank-4". The graph gains no value and the
list no update.

### AnNchwInputFails

Over a value converted to NCHW both forms fail with a message containing "BatchNorm2d(3)" and
"NHWC", and the graph gains no value.

### ALayerDescribedWrongFails

A default-constructed layer fails with "positive channel count" and "got 0". A layer with
epsilon 0.0 fails with "positive epsilon". A layer with momentum 1.5 fails with "momentum in
[0, 1]" and "1.5". The graph gains no value.

### ASecondTraceUnderAnotherElementTypeFails

A layer traced over a FLOAT32 input and then, in a second graph, over a FLOAT64 input fails
the second time with a message containing "'weight'" and both type names. The running mean's
type stays FLOAT32.

### TheScaleStartsAtOneAndTheShiftAtZero

After a trace the weight has the role `ParameterRole::Scale` and a constant fill of 1.0, and
the bias has the role `BIAS` and a constant fill of 0.0.

### TheFactoryReadsTheChannelsAndTheTwoConstants

`NN::make_batch_norm2d_node` over the shape [8, 8, 6] with `bn_epsilon` 0.001 and `bn_momentum`
0.03 returns a `ModuleNode<BatchNorm2d>` whose layer has 6 channels and those two options. The
node has one port of the input's shape and an input count of 1.

### TheFactoryRefusesARowThatDoesNotFit

The key `channels` fails with a message containing the context, "'channels'" and "bn_epsilon".
The input shape [48] fails with "[48]" and "[H, W, C]". A `bn_epsilon` of -1.0 fails with
"'bn_epsilon'" and "positive".

### TheTrainingFormUnderAnEvaluationTraceIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_training` given an evaluation
trace stops the process with a report containing "trace_training" and "was given an evaluation
trace".

### ATraceOverAValueTheGraphLacksIsFatal

The case is in the suite `Unit_BatchNorm2dDeathTest`. `trace_evaluation` over value 7 stops the
process with a report containing "BatchNorm2d(3)", "value 7" and "holds 1 values".

### TraceLayerPicksTheFormOfTheMode

`NN::trace_layer` under a training trace leaves one K_BATCHNORM2D and no K_BATCHNORM_FOLD in its
graph. Under an evaluation trace it leaves one K_BATCHNORM_FOLD and no K_BATCHNORM2D.

### OutputShapeIsTheInputShape

`output_shape([7, 5, 3])` is [7, 5, 3]. A three-channel layer also returns [7, 5, 4] for
[7, 5, 4]: `output_shape` does not check the channel count.

### UpdatedStatisticBlendsTheScaledBatchValue

`NN::updated_statistic(running, batch, momentum, scale)` is
(1 - momentum) * running + momentum * scale * batch. (1.0, 3.0, 0.25, 2.0) gives 2.25 and
(4.0, 8.0, 0.5, 1.0) gives 6.0. A momentum of 0 returns the running value 4.0 and a momentum
of 1 returns the batch value 8.0. Every operand and result is exact in a double.

### TheUpdatesReproduceTheUnitReferenceCase

`expect_reference_case("unit", K_UNIT_CASE_TOLERANCE)` checks the [2, 2, 2, 3] case. The
test's own comment gives the miss of the unscaled variance on this case as 8.3e-3.

### TheUpdatesReproduceTheModelReferenceCase

`expect_reference_case("model.9.cv1.bn", K_MODEL_CASE_TOLERANCE)` checks the case recorded at
that layer of the reference model. The test's own comment gives the miss of the unscaled
variance on this case as 1.4e-5.

## test_block_bottleneck.cpp

Tests Bottleneck's trace, its state names against YOLO26n row 2, and its factory.

This file needs no device. `NN::Bottleneck` in `include/SushiAI/nn/blocks/bottleneck.hpp` is
two `Conv` blocks, `cv1` and `cv2`, and an optional K_ADD of the block's input to the second
one's result. `NN::make_bottleneck_node` builds the node of a `bottleneck` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 4 'bottleneck'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`Bottleneck(16, 24, 8, true)` maps [5, 7, 16] to [5, 7, 24].

### AShortcutOverEqualWidthsAddsTheInput

A training trace of `Bottleneck(16, 16, 8, true)` reports `adds_input` true and emits two
K_MATMUL, two K_BATCHNORM2D, two K_SILU and one K_ADD. The K_ADD's first operand is the block's input and
its second is produced by a K_SILU. Its output is the returned value, of shape [2, 4, 4, 16].

### NoShortcutTracesNoAdd

With the shortcut argument false, `adds_input` is false and the trace holds two K_MATMUL and no
K_ADD.

### AShortcutOverUnequalWidthsTracesNoAdd

`Bottleneck(16, 24, 8, true)` reports `adds_input` false and traces no K_ADD, although a shortcut
was asked for. The result has the shape [2, 4, 4, 24].

### StateMatchesManifestRow2Inner

The walk of a traced `Bottleneck(16, 16, 8)` under "model.2.m.0" yields ten records. The five
of `cv1` come first: "cv1.conv.weight" [8, 3, 3, 16], then "cv1.bn.weight", "cv1.bn.bias",
"cv1.bn.running_mean" and "cv1.bn.running_var", each [8]. The five of `cv2` follow in the same
order with the weight [16, 3, 3, 8] and the rest [16].

### EachKernelSetsItsOwnConvolution

`Bottleneck(16, 16, 8, false, K_YOLO_NORM, 1, 3)` traces a `cv1` weight of [8, 1, 1, 16] and a
`cv2` weight of [16, 3, 3, 8].

### TheFactoryBuildsTheBlockAndDefaultsTheShortcutToTrue

`make_bottleneck_node` over [4, 4, 16] with 16 filters and a hidden width of 8 has one port of
shape [4, 4, 16]. Its training trace holds one K_ADD, so a row without `shortcut` adds its input.
The walk under "model.2.m.0" yields ten records; record 0 is the `cv1` weight [8, 3, 3, 16]
and record 5 the `cv2` weight [16, 3, 3, 8].

### TheFactoryReadsTheShortcut

With `"shortcut": false` the trace holds no K_ADD.

### TheFactoryRefusesARowThatDoesNotFit

The key `kernel` fails with a message containing "layer 4 'bottleneck'" and "'kernel'". A row
without `hidden` fails with "'hidden'" and "absent". A `shortcut` given as the string "yes"
fails with "'shortcut'" and "true or false".

### TheFactoryRefusesAnInputThatIsNotAFeatureMap

The input shape [256] fails with "[256]" and "rank-3".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

## test_block_c3k.cpp

Tests C3k's trace, its state names against YOLO26n row 6, and its factory.

This file needs no device. `NN::C3k` in `include/SushiAI/nn/blocks/c3k.hpp` holds three `Conv`
blocks, `cv1`, `cv2` and `cv3`, and a `ModuleList` of `Bottleneck`s named `m`. The cases show
its trace: a chain of bottlenecks on one branch, `cv2` on the other, one K_CONCAT_CHANNELS of the
two, and `cv3` after it. `NN::make_c3k_node` builds the node of a `c3k` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 5 'c3k'", the row name handed
to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`C3k(64, 96, 2, true)` maps [5, 7, 64] to [5, 7, 96].

### TheChainHoldsOneBottleneckPerRepeat

`C3k(64, 64, 3, true)` has `m.size()` 3.

### ATrainingTraceEmitsSevenConvBlocksTwoAddsAndOneConcat

A training trace of `C3k(64, 64, 2, true)` over [2, 4, 4, 64] emits seven each of K_MATMUL,
K_BATCHNORM2D and K_SILU, two K_ADD, one K_CONCAT_CHANNELS and no K_SLICE_CHANNELS. The list holds 14
updates and the result has the shape [2, 4, 4, 64]. Seven is three `Conv` blocks of the C3k
plus two in each of the two bottlenecks.

### TheJoinReadsTheChainThenTheSecondBranch

The K_CONCAT_CHANNELS node has two inputs. The first is produced by an K_ADD, the end of the last
bottleneck, and the second by a K_SILU, the end of `cv2`. Each is [2, 4, 4, 32] and the joined
value is [2, 4, 4, 64].

### NoShortcutTracesNoAdd

With the shortcut argument false, an evaluation trace holds seven K_MATMUL and no K_ADD.

### TheHiddenWidthIsHalfTheOutputWidthRoundedDown

`C3k(8, 7, 1, true)` has a hidden width of 3. The traced weights are [3, 1, 1, 8] for `cv1`
and `cv2`, [7, 1, 1, 6] for `cv3` and [3, 3, 3, 3] for the bottleneck's `cv1`. The result is
[2, 4, 4, 7], and `output_shape([4, 4, 8])` is [4, 4, 7].

### StateMatchesManifestRow6Inner

The walk of a traced `C3k(64, 64, 2)` under "model.6.m.0" yields 35 records: five each for
`cv1`, `cv2`, `cv3`, `m.0.cv1`, `m.0.cv2`, `m.1.cv1` and `m.1.cv2`, in that order. Each group
is the convolution weight, the norm's weight and bias as parameters, then its running mean and
variance as buffers. The weights are [32, 1, 1, 64] for `cv1` and `cv2`, [64, 1, 1, 64] for
`cv3` and [32, 3, 3, 32] for the four bottleneck convolutions.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

`make_c3k_node` over [4, 4, 64] with only 32 filters has one port of shape [4, 4, 32]. Its
evaluation trace appends one [2, 4, 4, 32] value and holds five K_MATMUL and one K_ADD: one
bottleneck that adds its input.

### TheFactoryReadsRepeatsAndShortcut

With 64 filters, `repeats` 2 and `shortcut` false, the trace holds seven K_MATMUL and no K_ADD. The
walk under "model.6.m.0" yields 35 records, and record 30 is "model.6.m.0.m.1.cv2.conv.weight"
[32, 3, 3, 32].

### TheFactoryRefusesAWidthThatLeavesNoHiddenChannel

`filters` 1 fails with `InvalidDescription` and a message containing "layer 5 'c3k'",
"'filters'" and "at least 2".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `repeats` 0 fails with "'repeats'" and "positive".

### TheFactoryRefusesTwoInputs

Two input shapes fail with a message containing "one input" and "2".

### TheFactoryHandsTheNormOptionsToEveryNorm

A row with `repeats` 2, `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces seven K_BATCHNORM2D
nodes and 14 updates, and `expect_norm_options` finds those two values on all of them.

## test_block_c3k2.cpp

Tests C3k2's trace, its state names against YOLO26n rows 2 and 6, and its factories.

This file needs no device. `NN::C3k2<Inner>` in `include/SushiAI/nn/blocks/c3k2.hpp` holds a
stem `cv1`, a `ModuleList<Inner>` named `m` and a closing `cv2`. The cases show its trace: the
stem's result is cut into two halves by two K_SLICE_CHANNELS, the inner blocks are chained off
the second half, and one K_CONCAT_CHANNELS joins both halves and every inner block's result
before `cv2`. `Inner` is `Bottleneck` or `C3k`. `NN::make_c3k2_node` reads the row's `inner`
key and selects a factory from `NN::K_C3K2_INNER`.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 2 'c3k2'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### row2_records

`row2_records` lists the 20 records `MANIFEST.json` holds under "model.2": five for `cv1`
(weight [32, 1, 1, 32]), five for `cv2` (weight [64, 1, 1, 48]), then five each for `m.0.cv1`
(weight [8, 3, 3, 16]) and `m.0.cv2` (weight [16, 3, 3, 8]).

### row6_records

`row6_records` lists the 45 records `MANIFEST.json` holds under "model.6": five for `cv1`
(weight [128, 1, 1, 128]), five for `cv2` (weight [128, 1, 1, 192]), then the 35 records of
the inner C3k under "model.6.m.0" that `test_block_c3k.cpp` lists.

### make_row2

`make_row2` builds `C3k2<Bottleneck>(32, 64, 16)` over one `Bottleneck(16, 16, 8, true)`.

### make_row6

`make_row6` builds `C3k2<C3k>(128, 128, 64)` over one `C3k(64, 64, 2, true)`.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

Row 2 maps [160, 160, 32] to [160, 160, 64].

### Row2EmitsFourConvBlocksTwoSlicesAndOneConcat

A training trace of row 2 over [2, 4, 4, 32] emits four each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, one K_ADD and one K_CONCAT_CHANNELS. The list holds eight updates, the result
is [2, 4, 4, 64], and every norm carries `K_YOLO_NORM`.

### TheJoinReadsBothHalvesThenTheChainOffTheSecondHalf

The K_CONCAT_CHANNELS of row 2 has three inputs. The first two are produced by K_SLICE_CHANNELS
nodes that read the same value, with the offsets 0 and 16 in attribute 0, and each is
[2, 4, 4, 16]. The third is produced by the bottleneck's K_ADD, whose first operand is the second
half. The joined value is [2, 4, 4, 48].

### TwoInnerBlocksAreChainedAndBothJoined

With two bottlenecks the join has four inputs. The producer of input 2 reads input 1, and the
producer of input 3 reads input 2. The `cv2` weight is [64, 1, 1, 64], four parts of 16
channels, and the second bottleneck's `cv1` weight is [8, 3, 3, 16].

### Row6EmitsNineConvBlocksAndTwoConcats

A training trace of row 6 over [2, 4, 4, 128] emits nine each of K_MATMUL, K_BATCHNORM2D and K_SILU,
two K_SLICE_CHANNELS, two K_ADD and two K_CONCAT_CHANNELS. The list holds 18 updates, the result is
[2, 4, 4, 128], every norm carries `K_YOLO_NORM`, and `output_shape([40, 40, 128])` is
[40, 40, 128].

### StateMatchesManifestRow2

The walk of traced row 2 under "model.2" equals `row2_records()`.

### StateMatchesManifestRow6

The walk of traced row 6 under "model.6" equals `row6_records()`.

### TheInnerTableListsBottleneckAndC3k

`NN::K_C3K2_INNER` has two entries: "bottleneck" with `&NN::make_c3k2_of_bottleneck` and "c3k"
with `&NN::make_c3k2_of_c3k`.

### TheFactoryBuildsRow2

`make_c3k2_node` over [160, 160, 32] with 64 filters, hidden 16, one repeat, inner
"bottleneck", shortcut true and the two norm keys has one port of shape [160, 160, 64]. Its
training trace holds one K_ADD, four K_BATCHNORM2D and eight updates carrying `K_YOLO_NORM`, and
its walk under "model.2" equals `row2_records()`.

### TheFactoryBuildsRow6

`make_c3k2_node` over [40, 40, 128] with 128 filters, hidden 64, one repeat, inner "c3k",
shortcut true and the two norm keys has a port of shape [40, 40, 128]. Its training trace
holds two K_CONCAT_CHANNELS, two K_ADD, nine K_BATCHNORM2D and 18 updates carrying `K_YOLO_NORM`,
and its walk under "model.6" equals `row6_records()`. The inner C3k therefore has two
bottlenecks that add their input.

### TheFactoryReadsTheShortcutOfAC3kInnerBlock

With inner "c3k" and shortcut false the evaluation trace holds nine K_MATMUL and no K_ADD.

### TheFactoryDefaultsToOneBottleneckWithAShortcut

A row stating only 64 filters and hidden 16 traces four K_MATMUL, one K_ADD and one
K_CONCAT_CHANNELS.

### TheFactoryReadsRepeatsAndShortcut

With `repeats` 3 and `shortcut` false the trace holds eight K_MATMUL and no K_ADD. The walk yields
40 records: record 5 is "cv2.conv.weight" [64, 1, 1, 80] and record 30 is "m.2.cv1.conv.weight"
[8, 3, 3, 16].

### TheFactoryRefusesAnInnerBlockItDoesNotList

`"inner": "bottleneck_psa"` fails with `InvalidDescription` and a message containing "layer 2
'c3k2'", "'bottleneck_psa'" and "bottleneck, c3k". `"inner": 2` fails with "'inner'" and
"string".

### TheFactoryRefusesAHiddenWidthOfOne

`hidden` 1 fails with "'hidden'" and "at least 2" for the default inner block and for inner
"c3k".

### TheFactoryRefusesARowThatDoesNotFit

The key `heads` fails with the context and "'heads'". A row without `hidden` fails with
"'hidden'" and "absent". The input shape [512] fails with "[512]" and "rank-3".

### AnInnerFactoryRefusesARowThatNamesAnotherInnerBlock

`make_c3k2_of_bottleneck` given `"inner": "c3k"` fails with a message containing the context,
"'inner' is 'c3k'" and "builds 'bottleneck'". `make_c3k2_of_c3k` given `"inner": "bottleneck"`
fails with "'inner' is 'bottleneck'" and "builds 'c3k'". `make_c3k2_of_bottleneck` given
`"inner": 2` fails with "'inner'" and "string". Each factory accepts a row that names its own
block.

## test_block_conv_norm.cpp

Tests ConvNorm's trace, its state names against YOLO26n row 0, and make_conv_node.

This file needs no device. `NN::ConvNorm` in `include/SushiAI/nn/blocks/conv_norm.hpp` is a
bias-free same-padded `Conv2d`, a `BatchNorm2d` and an activation, traced in that order.
`NN::Conv` is `ConvNorm<Conv2d, SiLU>` and `NN::ConvLinear` is `ConvNorm<Conv2d, Identity>`.
`NN::make_conv_node` builds the node of a `conv` row of a model file.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`, the epsilon and momentum `MANIFEST.json`
records for YOLO26n. `K_CONTEXT` is "layer 0 'conv'", the row name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### OutputShapeHalvesAnEvenExtentAtStrideTwo

`Conv(3, 16, 3, 2)` maps [640, 640, 3] to [320, 320, 16].

### OutputShapeOfAPointwiseBlockKeepsTheExtents

`Conv(32, 64, 1, 1)` maps [7, 9, 32] to [7, 9, 64].

### ATrainingTraceEmitsConvolutionNormThenSilu

A training trace of `Conv(3, 16, 3, 2)` over [2, 8, 8, 3] emits one K_MATMUL, one K_BATCHNORM2D
and one K_SILU, in that order, and no K_BIAS_ADD. The K_SILU's output is the returned value, of
shape [2, 4, 4, 16].

### ATrainingTraceCarriesTheNormOptions

The K_BATCHNORM2D node carries the epsilon 1e-3 in attribute 0, and both requested updates carry
the momentum 0.03.

### AnEvaluationTraceFoldsTheNorm

An evaluation trace emits one K_BATCHNORM_FOLD, one K_CHANNEL_AFFINE, one K_SILU and no K_BATCHNORM2D,
and returns a [2, 4, 4, 16] value.

### ConvLinearEmitsNoActivation

A training trace of `ConvLinear(3, 16, 1, 1)` emits one K_BATCHNORM2D and no K_SILU. The returned
value is output 0 of the K_BATCHNORM2D node.

### StateMatchesManifestRow0

The walk of a traced `Conv(3, 16, 3, 2)` under "model.0" yields five records in this order:
"model.0.conv.weight" [16, 3, 3, 3], "model.0.bn.weight" [16] and "model.0.bn.bias" [16] as
parameters, then "model.0.bn.running_mean" [16] and "model.0.bn.running_var" [16] as buffers.

### AnUntracedBlockYieldsNoState

The walk of a block that was never traced is empty.

### TheFactoryBuildsRow0

`make_conv_node` over [640, 640, 3] with 16 filters, kernel 3, stride 2, `bn_epsilon` 0.001 and
`bn_momentum` 0.03 returns a node with one port named `NN::K_SINGLE_PORT_NAME` of shape
[320, 320, 16]. A training trace over [1, 640, 640, 3] appends one value of shape
[1, 320, 320, 16], the K_BATCHNORM2D carries 1e-3, and the first update carries 0.03. The walk
under "model.0" yields five records, the first "model.0.conv.weight" [16, 3, 3, 3] and the
last "model.0.bn.running_var" [16].

### TheFactoryDefaultsKernelAndStrideToOne

A row stating only 8 filters over [6, 10, 3] has a port of shape [6, 10, 8]. After a trace its
first state record is "conv.weight" [8, 1, 1, 3].

### TheFactoryRefusesAnUnknownKey

The key `padding` fails with `InvalidDescription` and a message containing "layer 0 'conv'",
"'padding'" and "filters".

### TheFactoryRefusesARowWithoutFilters

A row without `filters` fails with a message containing the context, "'filters'" and "absent".

### TheFactoryRefusesAWindowThatIsNotAPositiveWholeNumber

A `stride` of 0 fails with "'stride'", "positive" and "0". A `kernel` given as the string "3"
fails with "'kernel'" and "whole number".

### TheFactoryRefusesNormOptionsOutOfRange

A `bn_epsilon` of 0.0 fails with "'bn_epsilon'" and "positive", and one given as a string
fails with "'bn_epsilon'" and "number". A `bn_momentum` of 1.5 fails with "'bn_momentum'" and
"[0, 1]", and one of -0.1 fails with those and "-0.1".

### TheFactoryRefusesInputsThatAreNotOneFeatureMap

Two input shapes fail with a message containing the context, "one input" and "2". The one
shape [192] fails with "[192]" and "rank-3".

## test_block_sppf.cpp

Tests SPPF's trace, its state names against YOLO26n row 9, and its factory.

This file needs no device. `NN::SPPF` in `include/SushiAI/nn/blocks/sppf.hpp` holds a
`ConvLinear` named `cv1`, a `MaxPool2d` named `pool` and a `Conv` named `cv2`. The cases show
its trace: `cv1` narrows the input, the pool is applied a number of times in a chain, one
K_CONCAT_CHANNELS joins `cv1`'s result and every pooled value, `cv2` follows, and an optional K_ADD
closes the block. `NN::make_sppf_node` builds the node of an `sppf` row.

The expected state names are typed into the test from `tests/fixtures/yolo26/MANIFEST.json`.
No case opens a fixture file.

### Constants

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged. `K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_CONTEXT` is "layer 9 'sppf'", the row
name handed to the factory.

### StateKind

`StateKind` has the enumerators `Parameter` and `Buffer`.

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### walk

`walk` returns one `record` per parameter and buffer that `NN::walk_state` yields under a
prefix, in walk order.

### count_nodes

`count_nodes` counts the nodes of a graph that carry one operation.

### first_node

`first_node` returns the index of the first node of a graph that carries one operation.

### one_input

`one_input` views one shape as the `NN::ShapeSpan` of a one-input row.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### expect_norm_options

`expect_norm_options` expects the given epsilon in attribute 0 of every K_BATCHNORM2D node of a
graph and the given momentum on every update of a list.

### OutputShapeKeepsTheExtentsAndSetsTheWidth

`SPPF(256, 256, 5, 3, true)` maps [20, 20, 256] to [20, 20, 256], and `SPPF(64, 96, 5, 3,
true)` maps [5, 7, 64] to [5, 7, 96].

### ThePoolKeepsTheExtents

The pool of `SPPF(256, 256, 5, 3, true)` has a 5x5 kernel, stride 1 and a pad of 2 on top and
left, and its `output_shape([20, 20, 128])` is [20, 20, 128].

### Row9EmitsThreePoolsOneConcatAndOneAdd

A training trace over [2, 4, 4, 256] emits two K_MATMUL, two K_BATCHNORM2D, one K_SILU, three
K_MAXPOOL2D, one K_CONCAT_CHANNELS and one K_ADD. The list holds four updates and the result is
[2, 4, 4, 256]. The one K_SILU is `cv2`'s.

### TheJoinReadsTheNarrowedValueThenEachPoolInAChain

The K_CONCAT_CHANNELS has four inputs. The first is produced by a K_BATCHNORM2D, because `cv1` is
a `ConvLinear` and emits no activation. Each of the other three is produced by a K_MAXPOOL2D
that reads the input before it, and each is [2, 4, 4, 128]. The joined value is
[2, 4, 4, 512].

### TheClosingAddReadsTheResultThenTheInput

`adds_input` is true. The K_ADD's first operand is produced by a K_SILU and its second is the
block's input, and its output is the returned value. The operand order is the reverse of
`Bottleneck`'s.

### NoAddWithoutAShortcutOrOverUnequalWidths

`SPPF(64, 64, 5, 3, false)` and `SPPF(64, 96, 5, 3, true)` both report `adds_input` false and
trace no K_ADD.

### ThePoolCountSetsThePoolsAndTheJoinedWidth

`SPPF(64, 64, 3, 1, false)` traces one K_MAXPOOL2D. Its `cv1` weight is [32, 1, 1, 64] and its
`cv2` weight is [64, 1, 1, 64], two parts of 32 channels.

### StateMatchesManifestRow9

The walk of a traced `SPPF(256, 256, 5, 3)` under "model.9" yields ten records. The five of
`cv1` come first, with the weight [128, 1, 1, 256] and the rest [128]. The five of `cv2`
follow, with the weight [256, 1, 1, 512] and the rest [256]. The pool has no state.

### TheFactoryBuildsRow9

`make_sppf_node` over [20, 20, 256] with 256 filters, kernel 5, 3 pools and shortcut true has
one port of shape [20, 20, 256]. Its training trace holds three K_MAXPOOL2D and one K_ADD. The walk
under "model.9" yields ten records; record 0 is the `cv1` weight and record 5 the `cv2`
weight.

### TheFactoryDefaultsToThreePoolsOfFiveAndNoShortcut

A row stating only 64 filters traces three K_MAXPOOL2D and no K_ADD, and its `cv2` weight is
[64, 1, 1, 128].

### TheFactoryRefusesAnEvenKernel

`kernel` 4 fails with `InvalidDescription` and a message containing "layer 9 'sppf'",
"'kernel'", "odd" and "4".

### TheFactoryRefusesAnInputOfOneChannel

The input shape [4, 4, 1] fails with "[4, 4, 1]" and "at least 2 channels".

### TheFactoryRefusesARowThatDoesNotFit

The key `hidden` fails with the context and "'hidden'". A row without `filters` fails with
"'filters'" and "absent". `pools` 0 fails with "'pools'" and "positive".

### TheFactoryHandsTheNormOptionsToBothNorms

A row with `bn_epsilon` 0.001 and `bn_momentum` 0.03 traces two K_BATCHNORM2D nodes and four
updates, and `expect_norm_options` finds those two values on all of them.

### TheFactoryReadsKernelAndPools

With `kernel` 3 and `pools` 1 the trace holds one K_MAXPOOL2D whose attributes 0, 1 and 6, which
are kernel_h, kernel_w and pad_top, are 3, 3 and 1. The `cv2` weight is [64, 1, 1, 64].

## test_buffer.cpp

Tests Buffer: its description, its one leaf per graph and its identity.

This file needs no device. `NN::Buffer` in `include/SushiAI/nn/buffer.hpp` is state a model
owns that takes no gradient. `define()` records its shape, starting value and element type and
returns a `Status`. `read()` declares a leaf of lifetime `Graph::Lifetime::Buffer` in the graph
of the builder it is given.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### read_undefined

`read_undefined` calls `read()` on a buffer under the name "running_mean" with a builder over
a fresh graph. It is the statement of the death test.

### ANewBufferIsUndefined

A default-constructed buffer reports `declared()` false and the state `Graph::K_NO_STATE`.

### DefineRecordsTheDescription

After `define([8], 1.0, FLOAT64)` the buffer is declared and returns that shape, that element
type and the starting value 1.0.

### DefineDefaultsToFloat32

`define([8], 0.0)` with no element type records `DType::Float32`.

### ADefinedBufferAddsNoLeafUntilItIsRead

After `define()` alone a graph holds no value and `id_in(graph)` is `Graph::K_NO_VALUE`.

### DefineDrawsTheIdentity

Two buffers that were only defined have states other than `K_NO_STATE`, and the two states
differ. A second identical `define()` and a `read()` leave the first buffer's state unchanged.

### ReadDeclaresABufferLeaf

`read()` adds one value to the graph. It has the lifetime `BUFFER`, `requires_grad` false and
the shape [8] and type FLOAT64 that were defined. `id_in(graph)` and
`graph.state_leaf(buffer.state())` both return it.

### ASecondReadInOneGraphReturnsTheFirstLeaf

A second `read()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ABufferReadInTwoGraphsHasALeafInEach

The first graph holds a padding input, so the buffer's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, and the buffer's state is
the same after both reads.

### ACopiedBufferHasANewStateId

A copy of a defined and read buffer has the original's shape and starting value and a state
that is neither `K_NO_STATE` nor the original's. `copy.id_in(graph)` is `K_NO_VALUE` until the
copy is read, and its leaf then differs from the original's, which keeps its own.

### AMovedBufferKeepsItsStateId

A buffer move-constructed from a read one has the same state and the same leaf in the graph.
Reading it again returns that leaf and adds no value.

### AMovedFromBufferIsUndefined

After a move construction and after a move assignment the source reports `declared()` false
and the state `K_NO_STATE`. The assigned target is declared and has a state.

### ACopyAssignedBufferHasANewStateId

A defined [2] buffer copy-assigned from a [8] FLOAT64 one takes that shape and type. Its state
differs from the state it had before and from the source's.

### ACopyOfAnUndefinedBufferHasNoIdentity

A copy of a buffer that was never defined is not declared and has the state `K_NO_STATE`.

### DefiningTwiceWithOneDescriptionIsAccepted

A second `define()` with the same shape, value and type returns ok and leaves the description
as it was.

### DefiningWithAnotherShapeFails

`define([4], 0.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as", "[8]" and "[4]". The shape stays [8].

### DefiningWithAnotherElementTypeFails

`define([8], 0.0, FLOAT64)` after a FLOAT32 definition returns `InvalidDescription` with a
message naming both types. The type stays FLOAT32.

### DefiningWithAnotherStartingValueFails

`define([8], 1.0)` after `define([8], 0.0)` returns `InvalidDescription` with a message
containing "Buffer was defined as" and "is now defined as". The starting value stays 0.0.

### ReadingBeforeDefineIsFatal

The case is in the suite `Unit_BufferDeathTest`. `read()` on an undefined buffer stops the
process with a report containing "'running_mean'" and "before define()".

## test_checkpoint_format.cpp

Tests the checkpoint format's round trip and its refusal of each malformed byte pattern.

This file holds the checkpoint format's whole malformed-file surface in the device-free tier.
That is the point of splitting `include/SushiAI/io/checkpoint_format.hpp` out of
`include/SushiAI/io/checkpoint.hpp`. Every case is a byte pattern, so none of them needs an
Engine, a TensorPool or a device to run.

Each corruption is written as a patch at a known offset into a good file, not as a hand-written
byte array. A case therefore says what is wrong with the file and does not restate the whole
format.

### Field offsets

The `OFFSET_` constants are byte offsets into `sample_stream()`, whose first record is
"0.weight", FLOAT32, [2, 3]. They are stated once so that a case names the field it breaks.

### DistinguishesTwoShapesWithTheSameElementCount

This case covers the defect the format exists to close: an element count alone lets a [2, 3]
weight load into a [3, 2] one. The shape is recorded, so the two records are not
interchangeable.

## test_concat.cpp

Tests the Concat node's shape check, its port and the node it traces.

This file needs no device. `NN::Concat` in `include/SushiAI/nn/concat.hpp` is an `NN::Node`
that joins several feature maps along the channel axis. `Concat::create` takes one per-sample
shape per input and returns a `Result`, and `forward` returns a `Status`.

### create_over

`create_over` calls `Concat::create` over a vector of shapes viewed as an `NN::ShapeSpan`.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### CreateSumsTheChannelsOfItsInputs

Over [8, 6, 3], [8, 6, 5] and [8, 6, 1] the node has one port named "output" of shape
[8, 6, 9].

### CreateAcceptsOneInput

Over the one shape [8, 6, 3] the port's shape is [8, 6, 3].

### CreateRejectsUnequalHeight

[8, 6, 3] with [4, 6, 3] fails with `InvalidDescription` and a message containing "concat
input 1" and "H and W".

### CreateRejectsUnequalWidth

A third input of [8, 12, 3] fails with a message containing "concat input 2" and "H and W".

### CreateRejectsAnInputThatIsNotAFeatureMap

[8, 6, 3] with [144] fails with a message containing "concat input 1" and "rank-3".

### CreateRejectsAFirstInputThatIsNotAFeatureMap

[144] with [8, 6, 3] fails with a message containing "concat input 0" and "rank-3".

### CreateRejectsNoInput

No shape at all fails with a message containing "at least one input".

### ForwardEmitsOneConcatChannelsReadingEveryInputInOrder

`forward` over the values b and a adds one `Ops::K_CONCAT_CHANNELS` node whose inputs are b then
a. It appends one id to `outputs`, the node's one output.

### ForwardResultIsThePortShapeWithABatchAxis

Over [2, 8, 6, 3] and [2, 8, 6, 5] the traced value has the shape [2, 8, 6, 8], and the port
reports [8, 6, 8].

### ForwardRejectsAnInputCountItWasNotBuiltOver

A node built over two shapes and given one value fails with a message containing "input count
is 2" and "given is 1". `outputs` stays empty and the graph gains no node.

### TheInputCountIsTheNumberOfShapesItWasBuiltOver

`input_count()` is 1 for a node built over one shape and 3 for a node built over three.

### ForwardReturnsTheFailureOfValuesThatDoNotJoin

A node built over two [8, 6, 3] shapes is given a [2, 8, 6, 3] value and a [2, 4, 6, 3] value.
`forward` returns an `InvalidDescription` failure, appends nothing to `outputs` and adds no
node.

### TheFactoryBuildsANodeFromARowWithNoKey

`NN::make_concat_node` over [8, 6, 3] and [8, 6, 5] with the attributes `{}` returns a
`Concat` with an input count of 2 and one port of shape [8, 6, 8].

### TheFactoryRefusesAKeyAndShapesThatDoNotJoin

The attributes `{"axis": 3}` fail with a message containing the context "layer 2 'concat'",
"'axis'" and "no key". The shapes [8, 6, 3] and [4, 6, 5] with no attribute fail with the
context and "H and W".

### TheStateWalkVisitsNothing

After a trace, `walk_state("merge", visitor)` calls neither the parameter callback nor the
buffer callback.

## test_config.cpp

Tests how the config layer parses its JSON documents and which ones it refuses.

This file needs no device. The config layer is host-lane by construction: it turns text into a
typed tree and never touches an Engine.

Most of what the cases assert is a diagnostic, not a value. That is deliberate. A config
layer's job is to refuse a wrong file loudly, and a test that only checked "it threw" would keep
passing after a refactor collapsed five specific messages into one generic complaint.
`tests/common/test_throws.hpp` gives the argument in full.

The cases fall into two groups, in file order:

- the model document in both of its versions, and the layer registry it selects through
- the train and eval documents

### KeepsTheKeyOrderTheFileSpelled

Key order is what lets a "this object carries ..." diagnostic read the way the file does, so
the test treats it as a property and not as an accident.

### AnIntegerReadsAsARealButARealIsNotAWholeNumber

JSON's grammar distinguishes 3 from 3.0 and the config layer keeps that, so `"units": 2.5` is
refused, not truncated. The widening goes one way only.

### AnAllowedKeySetIsAcceptedAndAnEmptyOneRefusesEverything

A layer that takes no attributes, such as gelu, passes an empty list. Any key at all is then an
error and is not quietly dropped.

### WithoutRemovesTheDiscriminatorAndLeavesTheRest

Removing a key that is absent is not an error, because a layer spelled with no attributes still
goes through the same call.

### AtOrFallsBackWithoutThrowing

A non-object falls back and does not throw. A layer spelled `{"type": "gelu"}`, whose attributes
are an empty object, therefore reads its optional keys through the same call as any other layer.

### ParsesAModelAndSeparatesEachLayersTypeFromItsAttributes

"type" is gone from the attributes, so no factory has to allow the key that selected it.

### ChainsLayerWidthsFromTheInputFeatures

The activation preserves the width it was handed, which is why the file never states a width
for it.

### AnUnknownLayerTypeListsEveryTypeThisBuildHas

The model names a layer type "hologram", which no row of `NN::K_LAYER_REGISTRY` carries. The
message contains "hologram", "layer type", "'linear'" and "'gelu'". The unknown type has to
be a name the registry does not hold, and "conv2d" is a row of the table.

### AnUnknownLayerKeyIsRejectedByTheLayersOwnFactory

The `stray` document writes a width onto an activation. An activation takes no attributes, so
that width is an error and not a line that does nothing.

### The version 2 model document

The cases from AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite to
AMalformedOutputIsAnError cover the two versions of the model document that
`Config::parse_model` reads. `Config::K_MODEL_VERSION` is 2. A version 1 document carries the key
`sushiai_model` and an `input` of `features`. A version 2 document carries `format_version`,
an `input` of `shape`, an optional `from` on each row, and the optional top-level keys
`defaults` and `outputs`. Both parse to one `Config::ModelSpec`, in which every row holds its
sources as absolute row indices and the model input is `Config::K_MODEL_INPUT`, which is -1.

### K_FIRST_VERSION_MODEL

`K_FIRST_VERSION_MODEL` is a version 1 document of five rows over 784 features: `unflatten` to
28 by 28 by 1, `conv2d` of 8 filters and kernel 3 with a bias, `relu`, `flatten` and `linear`
of 10 units.

### K_REWRITTEN_MODEL

`K_REWRITTEN_MODEL` is the same model written as version 2, with the input shape [784] and
`"from": -1` on every row.

### scalar_text

`scalar_text` renders one scalar attribute as "int n", "bool true", "bool false" or "string
s", and any other kind as its kind name.

### spec_text

`spec_text` renders a `ModelSpec` as text: one line for the input shape and feature count, one
line per row with its type, its sources as `node:'port'` and its attributes, and one line per
output. Two specs compare equal as text when every one of those fields agrees.

### current_model_from

`current_model_from` wraps a `layers` array, and optionally more top-level keys, in a version
2 document whose input shape is [6].

### AVersionOneDocumentParsesToTheSameSpecAsItsVersionTwoRewrite

`spec_text` of the two documents is equal. The expected text is asserted as a literal too: the
input is [784] with 784 features, row 0 reads -1, each later row reads the row above it, and
the one output is row 4 with no port and no name.

### AVersionOneDocumentFillsTheInputShapeAndTheFeatureCount

`K_GOOD_MODEL`, a version 1 document of 6 features, parses to `input_shape` [6] and
`input_features` 6. The case also asserts that `Config::K_MODEL_VERSION` is 2.

### AVersionTwoInputShapeKeepsItsRankAndCountsItsElements

The input shape [28, 28, 1] is kept at rank 3 and `input_features` is 784.

### AnEmptyOrNonPositiveInputShapeIsAnError

An empty `shape` throws with a message containing "'input' 'shape'" and "empty". The shape
[28, 0, 1] throws with "'shape' entry 1" and "positive". A version 2 input that states
`features` throws with "'features'" and "'shape'".

### AnInputShapeAboveTheRankOrTheElementLimitIsAnError

A shape of `K_MAX_RANK` + 1 extents throws with "'shape'" and "at most". The shape
[4294967296, 4294967296], whose product is 2^64, throws with "'shape'", "entry 1" and "more
elements".

### AModelRootThatIsNotAnObjectIsAnError

The document `[1]` throws with a message containing "model.json", "'format_version'" and "an
array".

### AVersionAboveTheNewestIsRefusedWithBothNumbers

`"format_version": 3` throws with a message containing "model.json", "version 3" and "version
2".

### FormatVersionOneIsRefusedAndNamesTheVersionOneKey

`"format_version": 1` throws with a message containing "'format_version'" and
"'sushiai_model'". Version 1 files carry the second key and never the first.

### AVersionOneDocumentRefusesTheVersionTwoKeys

A version 1 document with a `from` on row 1 throws with "layer 1", "'from'" and "version 1".
One with a top-level `defaults` throws with "'defaults'" and "'layers'".

### EachFromFormResolvesToAnAbsoluteRow

Six rows use every form of `from`. An absent `from` on row 0 and the string "input" resolve to
-1. The number -1 resolves to the row above. An array `[-1, 0]` on row 3 resolves to rows 2
and 0. An object `{"node": 3, "port": "boxes"}` resolves to row 3 with that port. An array may
mix objects, "input" and a relative object. The whole spec is compared as text. Row 0's source
is `K_MODEL_INPUT`, and "from" is not among row 3's attributes.

### ANegativeFromOtherThanMinusOneNamesTheRow

`"from": -2` on row 2 throws with a message containing "layer 2 of 'model.json'", "'from'" and
"row -2". `{"node": -3}` inside an array throws with the layer and "row -3".

### AForwardReferenceNamesTheRow

`"from": 2` on row 1 throws with "layer 1 of 'model.json'", "row 2" and "above row 1".
`"from": [0, 1]` on row 1, which names the row itself, throws with the layer and "row 1".

### AnUnknownKeyOrFormInAFromEntryNamesTheRow

Each of five `from` entries on row 1 throws with "layer 1 of 'model.json'" and one more
needle: the key `prot` with "'prot'" and "'port'"; the string "previous" with "'previous'" and
"'input'"; an empty array with "empty"; the number 0.5 with "it is a number"; and an empty
`port` string with "'port'" and "empty string".

### AnUnknownTopLevelKeyOfAVersionTwoDocumentIsAnError

A top-level key `output` throws with a message containing "'output'", "'outputs'" and
"'defaults'".

### DefaultsFillAnOmittedKeyAndNeverOverrideAStatedOne

The `defaults` block gives `conv` rows `bn_momentum` 3 and `kernel` 5. Row 0 states only
`filters` and ends with the keys `filters`, `bn_momentum`, `kernel` and the values 3 and 5.
Row 1 states `bn_momentum` 7 and keeps it. Row 2 is a `relu` and receives no key.

### AMalformedDefaultsBlockIsAnError

A `defaults` that is an array throws with "'defaults'" and "an array". A type whose default is
the number 3 throws with "'defaults' for 'conv'" and "object". A default that sets `from`
throws with that context and "'from'", and one that sets `type` with "'type'".

### ADefaultsEntryForATypeNoLayerStatesIsAnError

A default for "covn" in a model of `conv` and `relu` rows throws with a message containing
"'defaults'", "'covn'" and "'conv', 'relu'".

### OutputsDefaultToTheLastRowAndKeepWhatTheFileStates

A document with no `outputs` gets one output: the last row, with an empty port and an empty
name. A document that states three outputs keeps each as written: row 0 alone, row 1 with the
port "boxes", and row 1 with the port "boxes" and the name "predictions".

### AMalformedOutputIsAnError

Seven `outputs` values each throw. Row 1 in a one-row model: "'outputs' entry 0", "row 1" and
"0 to 0". The key `nme`: "'nme'" and "'name'". A `name` without a `port`: "'name'" and
"'port'". An empty array: "'outputs'" and "empty". The number 3 as an entry: "a whole number".
An empty `port` string: "'outputs' entry 0 'port'" and "empty string". An empty `name` string:
"'outputs' entry 0 'name'" and "empty string".

### Path resolution in the train and eval documents

Path resolution is the one thing in this file that needs real files on disk. The property is
that a relative path resolves against the config file's own directory and not against the
process working directory, and that cannot be asserted without a config file that exists
somewhere.

### LeavesAnAbsolutePathAlone

`generic_string()` spells separators as '/', which JSON does not need to escape. The test
therefore builds the document by concatenation and needs no second, test-only escaper that
nobody would maintain.

### AnEvalDocumentCarriesNoModelBlock

The architecture is a separate file that inference is handed separately. Naming one in an eval
document is an error; it is not silently preferred over the one on the command line.

## test_conv2d.cpp

Tests the 2-D convolution layer's trace, parameters, shape inference and rejections.

This file needs no device. `NN::Conv2d` in `include/SushiAI/nn/conv2d.hpp` declares a weight
and a bias on its first trace and hands the node sequence to `GraphBuilder::conv2d`, so every
case reads a host-side graph, the layer's two parameters or the result of `output_shape()`.
`test_spatial_shapes.cpp` covers the nodes the builder emits; this file covers the layer.

Two `static_assert` lines at file scope require `Conv2d` to be a `Module` and not a
`ModuleContainer`.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### make_conv

`make_conv` builds a layer with a square window and one stride and one padding for both axes.
The stride defaults to 1 and the padding to 0. A case that needs an anisotropic window or a
dilation sets the geometry field afterwards.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### traced_shape

`traced_shape` traces a copy of a layer over a batch of the given per-sample shape and returns
the result's shape. It takes the layer by value, so a const layer can be traced although
`forward()` declares parameters.

### walk

`walk` returns the parameter walk's output as "name:extents" strings.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### ForwardOfAPaddedThreeByThreeKeepsTheSpatialExtents

A 3x3 window with padding 1, one input channel and eight filters traces a [4, 28, 28, 1]
input to a [4, 28, 28, 8] value tagged NHWC.

### TheParametersAndTheResultTakeTheInputElementType

With a FLOAT64 input the weight, the bias and the result are FLOAT64.

### TheParameterWalkNamesTheWeightAndTheBias

The walk is empty before the trace. After it the walk yields "weight:[8, 3, 3, 1]" then
"bias:[8]", the names `Linear` uses, and `parameter_elements` is 8 * 3 * 3 * 1 + 8.

### TheWeightOfAnAnisotropicWindowIsOutHeightWidthIn

A 2x4 window with three input channels and five filters gives a weight of shape [5, 2, 4, 3].
The four extents differ, so a weight with two axes swapped fails the comparison.

### ALayerWithABiasTracesOneBiasAdd

The traced graph holds exactly one K_BIAS_ADD node.

### ABiasFreeLayerHasNoBiasEntryAndNoBiasAdd

With `use_bias` false the walk yields the weight alone, the bias is not declared, the graph
holds no K_BIAS_ADD node, and the result shape is still [4, 28, 28, 8].

### TheWeightAndTheBiasRecordTheirFans

For a 3x5 window with four input channels and six filters, both parameters record a fan-in of
3 * 5 * 4 and a fan-out of 3 * 5 * 6. The bias carries the same pair as the weight.

### TheParametersRecordTheirRoles

The weight has role WEIGHT and the bias has role BIAS.

### TracingTwiceDoesNotDuplicateParameters

After a second trace into the same graph the weight and the bias keep their value ids and
`parameter_count` is 2.

### OutputShapeAgreesWithForwardForAStridedWindow

A 3x3 stride-2 unpadded window with sixteen filters maps [9, 12, 3] to [4, 5, 16], and the
traced shape of a batch of two equals that shape with the batch extent in front. The four
cases from here assert the literal and the agreement together.

### OutputShapeAgreesWithForwardForAPaddedWindow

A 5x5 window with padding 2 maps [7, 10, 2] to [7, 10, 4].

### OutputShapeAgreesWithForwardForADilatedStridedWindow

A 3x3 window with stride 2, padding 1 and dilations 2 by 3 maps [11, 13, 2] to [5, 5, 3].

### OutputShapeAgreesWithForwardForAPointwiseWindow

A 1x1 window with six filters maps [5, 7, 4] to [5, 7, 6]. This is the geometry for which
`GraphBuilder::conv2d` emits no K_IM2COL.

### SamePaddedPadsByHalfTheKernelAndDropsTheBias

`Conv2d::same_padded(3, 16, 5, 2)` has 3 input and 16 output channels, a 5x5 kernel, stride 2
on both axes, a pad of 2 on top and left, dilation 1 and `use_bias` false.

### SamePaddedAtStrideOneKeepsTheSpatialExtents

`same_padded(4, 4, 3, 1)` maps [7, 10, 4] to [7, 10, 4] in `output_shape()` and traces
[2, 7, 10, 4] over a batch of two.

### SamePaddedAtStrideOneKeepsTheExtentsForAPointwiseAndAFiveByFiveKernel

Over [9, 6, 2], `same_padded(2, 8, 1, 1)` and `same_padded(2, 8, 5, 1)` both report [9, 6, 8].

### SamePaddedAtStrideTwoHalvesAnEvenExtent

`same_padded(3, 16, 3, 2)` maps [640, 320, 3] to [320, 160, 16] in `output_shape()` and in the
trace.

### SamePaddedTracesNoBias

A traced `same_padded(2, 8, 3, 1)` layer walks as "weight:[8, 3, 3, 2]" alone. Its bias is not
declared and the graph holds no K_BIAS_ADD node.

### ANonZeroBiasInitReachesTheBiasParameter

A layer with `bias_init` set to -4.5 traces a bias whose `init()` is a constant of -4.5 and
whose role is `ParameterRole::Bias`. -4.5 is exact in binary.

### TheDefaultBiasInitIsTheConstantZero

A layer left at the default traces a bias whose `init()` is a constant of 0.0.

### BiasInitLeavesTheWeightFanScaled

With `bias_init` set to 1.25 the weight's `init()` is still `ParameterInit::Kind::FanScaled`
and its role is `WEIGHT`.

### OutputShapeRejectsANonPositiveInputChannelCount

A layer with `in_channels` 0 throws with a message containing "Conv2d", "in=0" and "out=8".

### OutputShapeRejectsANonPositiveOutputChannelCount

A layer with `out_channels` -1 throws with a message containing "Conv2d", "in=3" and "out=-1".

### OutputShapeRejectsAWrongChannelCount

A three-channel layer given [8, 8, 4] throws. The message contains "Conv2d(3, 8)", the shape
"[8, 8, 4]" and the expected form "[H, W, 3]".

### OutputShapeRejectsAFeatureVector

`output_shape([3])` throws with a message containing "Conv2d(3, 8)", "[3]" and "[H, W, 3]".

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 3, 3])` throws with a message containing the shape and "[H, W, 3]". The
third extent equals `in_channels`, so only the rank check can refuse this shape.

### OutputShapeRejectsAWindowThatDoesNotFit

A 5x5 window over [4, 4, 3] throws with a message containing "Conv2d(3, 8)", "[4, 4, 3]" and
"5x5".

### OutputShapeRejectsAZeroStride

A stride of 0 throws with a message containing "Conv2d(3, 8)" and "stride 0,0".

### OutputShapeRejectsANegativePadding

A padding of -1 throws with a message containing "Conv2d(3, 8)" and "padding -1,-1".

### ForwardRejectsAWrongChannelCount

`forward()` on a [2, 8, 8, 4] input throws with a message containing "Conv2d(3, 8)",
"[8, 8, 4]" and "[H, W, 3]", the three strings OutputShapeRejectsAWrongChannelCount expects.
The weight is not declared afterwards.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` on a [2, 3] input throws with a message containing "Conv2d(3, 8)", "rank-4" and
"[2, 3]", and the weight is not declared afterwards.

### ForwardRejectsNonPositiveChannelCounts

`forward()` on a layer with `out_channels` 0 throws with a message containing "Conv2d",
"in=3" and "out=0".

### ForwardRejectsAWindowThatDoesNotFit

`forward()` with a 5x5 window over a [2, 4, 4, 3] input throws with a message containing
"Conv2d(3, 8)", "[4, 4, 3]" and "5x5".

### ForwardRejectsAnNchwInput

The input is [2, 3, 3, 3] converted to NCHW. Every extent is 3, so the permuted shape passes
every check but the one on the layout tag. The message contains "Conv2d" and "NHWC". The node
count and the value count are unchanged and neither parameter is declared, so the layer
refuses before it writes anything to the graph.

### DifferentiatingALossYieldsAGradientForTheWeightAndTheBias

The layer's result is reduced with `spatial_mean` and passed to `NN::CrossEntropyLoss`.
`Autograd::differentiate` returns two gradients, neither is `K_NO_VALUE`, and each has the shape
of its parameter. The case checks that a gradient exists and has the right shape; it computes
no number.

## test_detect_anchors.cpp

Tests the anchor grid, the per-image anchor leaves and the copies between a level and the joined
anchor axis.

This file needs no device. Its three suites, `Unit_DetectAnchors`, `Unit_DetectAnchorLeaves` and
`Unit_DetectLevelRows`, cover `include/SushiAI/detect/anchors.hpp`,
`include/SushiAI/detect/anchor_leaves.hpp` and `include/SushiAI/detect/level_rows.hpp`.
`include/SushiAI/detect/README.md` quotes the reference lines each unit ports. One case reads
`loss.sacp` through `tests/common/reference_fixture.hpp` and reports itself skipped when the
file is absent.

### K_EXACT

`K_EXACT` is the tolerance 0. Every expected value in the file is a multiple of 0.5, which
float32 holds exactly.

### K_REFERENCE_LEVELS

`K_REFERENCE_LEVELS` holds the three levels `tests/fixtures/yolo26/MANIFEST.json` records for a
640 px input: 80 by 80 at stride 8, 40 by 40 at stride 16 and 20 by 20 at stride 32.

### K_SMALL_LEVELS

`K_SMALL_LEVELS` is a grid of 2 by 3 cells at stride 8 followed by 1 by 2 cells at stride 16,
eight anchors in all, so a case can write every expected value out.

### CountsOneAnchorPerCell

`count_anchors` returns 8 for `K_SMALL_LEVELS` and 8400 for `K_REFERENCE_LEVELS`.

### OrdersPointsByLevelThenRowThenColumn

`make_anchors` over `K_SMALL_LEVELS` writes the sixteen point values and the eight strides the
case lists: the first level row by row, x before y, then the second level.

### ScalesEachPointByItsStride

`scale_points` multiplies both coordinates of a point by its anchor's stride. The centres 0.5,
1.5 and 2.5 become 4, 12 and 20 on the first level, and 0.5 and 1.5 become 8 and 24 on the
second.

### ScaleRejectsStridesThatDoNotMatchThePoints

Seven strides for eight points throw with a message containing "[A] strides" and "got 16, 7 and
16 values".

### RejectsALevelWithoutCellsOrStride

`count_anchors` throws on a level of height 0 with a message containing "height 0" and
"width 3", and on a level of stride 0 with a message containing "stride 0".

### RejectsOutputsOfTheWrongSize

`make_anchors` given a point array of 15 values throws with a message containing "16 point
values" and "got 15".

### MatchesTheReferenceAnchorPoints

The points of `K_REFERENCE_LEVELS`, scaled to pixels, equal the tensor `anchors.points` of
`loss.sacp` at tolerance 0, and that tensor has shape [8400, 2]. The case is skipped when
`loss.sacp` is absent.

### GivesEachLevelItsStrideOverItsAnchorRange

The strides of `K_REFERENCE_LEVELS` are 8 over anchors 0 to 6399, 16 over 6400 to 7999 and 32
over 8000 to 8399. The offsets 0, 6400 and 8000 are the `anchor_offset` entries of the manifest.

### RepeatsThePointsAndTheStridesForEveryImage

`build_anchor_leaves` for a batch of two writes the eight points of `K_SMALL_LEVELS` twice. It
writes the strides with four values per anchor: 24 values of 8 and 8 values of 16, also twice.

### SplitsIntoALevelsPointsAndStrides

`split_level` cuts the second level, anchors 6 and 7, out of both leaves. The points come out as
[2, 1, 2, 2] and the strides as sixteen values of 16.

### RejectsABatchWithoutImagesAndAModelWithoutLevels

A batch of 0 throws with a message containing "positive batch" and "batch 0 and 2 levels". An
empty level list throws with a message containing "at least one level" and "batch 1 and 0
levels".

### JoinPlacesALevelAtItsAnchorOffset

`join_level` with two images, five anchors of two values and a level at anchors 3 and 4 writes
the level's rows into values 6 to 9 of each image and leaves the other values at 0.

### SplitReturnsTheRowsJoinPlaced

The joined array holds its own index in every value. `split_level` for anchors 1 to 3 returns
values 2 to 7 of image 0 and 12 to 17 of image 1.

### RejectsALevelPastTheLastAnchor

A level of two anchors at offset 4 of five throws with a message containing "2 anchors",
"offset 4" and "5 anchors".

### SplitRejectsALevelArrayOfTheWrongSize

`split_level` into an array of 10 values, where the placement needs 12, throws with a message
containing "12 level values", "20 joined values" and "got 10 and 20".

## test_detect_assigner.cpp

Tests the task-aligned assigner, its presets, the padded ground truth and the target leaves.

This file needs no device. Its three suites are `Unit_DetectAssigner`, `Unit_DetectGroundTruth`
and `Unit_DetectTargetLeaves`, over `include/SushiAI/detect/assigner.hpp`,
`include/SushiAI/detect/assigner_config.hpp`, `include/SushiAI/detect/ground_truth.hpp` and
`include/SushiAI/detect/target_leaves.hpp`. Most cases build one image on a 4 by 4 grid by
hand, with `alpha` and `beta` both 1, so the metric is the class score times the overlap and
each expected score is a fraction a comment derives. Three cases read `loss.sacp` and report
themselves skipped when the file is absent. `include/SushiAI/detect/README.md`, under
"Reference tensors the assigner tests read", lists the tensors.

### Tolerances

`K_EXACT` is 0, for copied values and multiples of 0.5. `K_SCORE_TOLERANCE` is 1e-6, eight
float32 steps at 1, for a target score computed by hand. `K_REFERENCE_SCORE_TOLERANCE` is 1e-5
for a reference target score, where `std::pow` and `std::atan` may round apart from torch.
`K_REFERENCE_BOX_TOLERANCE` is 1.3e-4, two float32 steps at 640, for a reference box in pixels.

### The hand-built grid

`K_GRID_SIDE` is 4 cells, `K_GRID_ANCHORS` is 16, `K_GRID_STRIDE` is 8 px and `K_CLASSES` is 2.
`K_FAR_BOX` is the box (100, 100, 101, 101), which no hand-built ground truth touches.

### Scene

`Scene` holds one image on that grid: anchor `a` sits in column `a % 4` and row `a / 4`, with
pixel centres 4, 12, 20 and 28. Its constructor gives every anchor one score for both classes
and the prediction `K_FAR_BOX`. `add_truth` appends a valid ground-truth box, `predict` sets one
anchor's predicted box, and `inputs` returns the scene as an `AssignerInputs`.

### make_hand_config

`make_hand_config` returns an `AssignerConfig` with the given `topk` and `topk2`, two classes,
`alpha` 1, `beta` 1 and `stride_val` 8.

### foreground_anchors

`foreground_anchors` returns the rows an `Assignment` marks as foreground, in ascending order.

### make_crowded_scene

`make_crowded_scene` returns a scene with one box of class 1 over the whole 32 px image. Every
anchor predicts that box and scores 0.1, except anchors 5, 10 and 2, which score 0.9, 0.6 and
0.3 in class 1.

### make_contested_scene

`make_contested_scene` returns a scene with two boxes, of classes 0 and 1, that both contain
anchor 5. Anchors 0, 1 and 4 predict the first box, anchors 6, 9 and 10 the second, and anchor 5
predicts the box the caller passes.

### box_row

`box_row` returns the four values of one row of a `[rows, 4]` array.

### make_idle_assignment

`make_idle_assignment` returns an `Assignment` of the given batch and anchor count with no
foreground and every array at zero.

### set_foreground

`set_foreground` marks one row of an `Assignment` as foreground and gives it a pixel box and two
class scores.

### Reference constants

`K_REFERENCE_LEVELS` holds the manifest's three levels for a 640 px input,
`K_REFERENCE_IMAGE_SIDE` is 640, and `K_SMALL_BOX_ROW` is 3, the ground-truth row of the
fixture that is smaller than a stride.

### pad_reference_truth

`pad_reference_truth` reads `batch.batch_idx`, `batch.cls` and `batch.bboxes` from `loss.sacp`
and pads them with `pad_ground_truth` for one 640 px image. It returns an empty string, or the
reason a tensor could not be read.

### count_holders

`count_holders` counts the foreground anchors of a mask whose target index is a given
ground-truth row. It is a template over the mask's element type, so it takes the assigner's
mask and the reference's.

### expect_reference_assignment

`expect_reference_assignment` runs one branch's assigner on the reference's scores, boxes and
anchor points and compares all five outputs: the mask, the indices and the labels for equality,
the boxes within `K_REFERENCE_BOX_TOLERANCE` and the scores within
`K_REFERENCE_SCORE_TOLERANCE`. It then compares the foreground count and the count of anchors
that hold box 3 with the two numbers the caller passes, and the second with the same count
taken from the reference. Equality of the mask and the indices relies on the fixture having no
tied metrics. The helper skips when `loss.sacp` is absent.

### OneToManyPresetMatchesTheReferenceSettings

`one_to_many_assigner_config(80, {8, 16, 32})` has `topk` 10, `topk2` 10, 80 classes, `alpha`
0.5, `beta` 6, `stride_val` 16 and `eps` 1e-9. These are the `assigner.one2many` values of
`loss.sacp` in `tests/fixtures/yolo26/MANIFEST.json`.

### OneToOnePresetMatchesTheReferenceSettings

`one_to_one_assigner_config(80, {8, 16, 32})` has `topk` 7, `topk2` 1, `alpha` 0.5, `beta` 6 and
`stride_val` 16, the `assigner.one2one` values of the manifest.

### PresetTakesStrideValFromTheSecondLevel

`stride_val` is 12 for the strides 4 and 12, 4 for the one stride 4, and 16 for an empty stride
list.

### ConfigDefaultsMatchTheReferenceConstructor

A default `AssignerConfig` has `topk` 13, `topk2` 13, 80 classes, `alpha` 1, `beta` 6,
`stride_val` 16 and `eps` 1e-9, the arguments of the reference's `TaskAlignedAssigner.__init__`.

### RejectsAConfigWithoutCandidatesOrWithANegativeMargin

Constructing a `TaskAlignedAssigner` throws for `topk` 0, with a message containing "positive
topk" and "got 0, 3 and 2", for `stride_val` -1, with "stride_val", "at least 0" and "-1", and
for `eps` -1, with "eps of at least 0" and "-1".

### TheMetricRaisesTheScoreToAlphaAndTheOverlapToBeta

The case uses the one-to-many preset, so `alpha` is 0.5 and `beta` is 6. Anchor 5 predicts the
box exactly with score 0.25, so its metric is sqrt(0.25) times 1, which is 0.5. Anchor 10
predicts a 24 px box with score 0.81; its overlap is 576 / 1024 - 32 / 2048, which is 35 / 64,
and its metric is 0.9 times (35 / 64)^6, about 0.024. With `topk` 1 the foreground is anchor 5
alone, and with `topk` 2 it is anchors 5 and 10. The target score of anchor 5 is 1 and that of
anchor 10 is its metric over 0.5. An assigner that used the scores or the overlaps unraised
gives another second score.

### NormalisesByTheLargestOverlapOfTheBox

Anchors 0 and 1 both predict a box offset by 8 px from a 16 px truth, with scores 0.8 and 0.4.
The overlap is 64 / 448 - 128 / 1152, which is 2 / 63. The target scores are that overlap and
half of it: each metric over the largest metric, times the largest overlap of the box.

### APredictionApartFromItsBoxHasNoOverlap

Anchors 0 and 1 score 0.9 and predict `K_FAR_BOX`, whose complete IoU with the box is negative
and is clamped to 0. With `topk` 2 the foreground is anchors 4 and 5, which predict the box.
With `topk` 3 the third place goes to anchor 0, the first candidate of metric zero, and its
target score is 0.

### AContestedAnchorOfEqualOverlapsGoesToTheFirstBox

Two ground-truth rows hold the same box with classes 0 and 1, and anchor 5 predicts it. Anchor 5
is the one foreground anchor, its target index and label are 0, its score in class 0 is 1 and
its score in class 1 is 0.

### AContestedAnchorCanGoToABoxThatDidNotKeepIt

With `topk` 1, boxes 0 and 1 both keep anchor 5 and box 2 keeps anchor 2. Anchor 5 predicts box
2 exactly, so its largest overlap is with box 2 and it goes there, with target index 2 and
label 1. Its score in class 1 is 0.1 / 0.9 and its score in class 0 is 0. The case pins that
the maximum overlap is taken over every box, including one that did not keep the anchor.

### AnImageWithoutGroundTruthGetsNoForeground

A scene with no ground truth has no foreground anchor. Every target label is 2, the class
count, and every index, box and score is 0.

### APaddingRowAssignsNothing

The crowded scene's one box is marked invalid. No anchor is foreground and every target score
is 0, although the box contains all sixteen anchors.

### KeepsTheTopkCandidatesWhenABoxHoldsMore

In the crowded scene every overlap is 1, so the metric is the class score. With `topk` 3 the
foreground is anchors 2, 5 and 10, with target scores 0.3 / 0.9, 1 and 0.6 / 0.9 in class 1 and
0 in class 0. Every target label is 1, the label of ground-truth row 0, on and off the
foreground.

### TheSecondTopkKeepsTheBestAnchorOfABox

With `topk` 3 and `topk2` 1 the crowded scene keeps anchor 5 alone, with target score 1.

### AContestedAnchorGoesToTheHigherOverlap

Anchor 5 predicts the second box of the contested scene. The foreground is anchors 0, 1, 4, 5,
6, 9 and 10, and the target index and label of anchor 5 are 1. Its score is 1 in class 1 and 0
in class 0, and its target box starts at x = 8.

### AContestedAnchorCanGoToTheFirstBox

The same scene with anchor 5 predicting the first box gives anchor 5 the target index 0, score
1 in class 0 and 0 in class 1.

### ASmallBoxIsEnlargedToStrideVal

A 2 px box centred at 14 contains no anchor centre. With `stride_val` 16 it is tested as the
span 6 to 22, which holds the centres 12 and 20 on both axes, so anchors 5, 6, 9 and 10 are
foreground. With `stride_val` 1 there is no foreground. The target box of anchor 5 still starts
at 13: the enlargement applies to the candidate test and not to the target.

### AssignsEachImageFromItsOwnRows

A batch of two holds an image with one invalid row and the crowded scene. The foreground rows
are 18, 21 and 26, which are anchors 2, 5 and 10 of image 1. The target label of row 0 is 0 and
that of row 16 is 1.

### RejectsTooFewAnchorsAndAnUnknownClass

`assign` with `topk` 17 on sixteen anchors throws with a message containing "16 anchors" and
"topk 17". A valid box of class 2 with two classes throws with "class 2 of 2".

### PadsLabelRowsPerImage

Three label rows, two for image 0 and one for image 2, are padded to a batch of three 64 by 32
images with `max_boxes` 2. The boxes are converted from normalised centre and size to pixel
corners, the labels are 3, 1, 0, 0, 7, 0 and the valid flags are 1, 1, 0, 0, 1, 0.

### MarksARowWithoutExtentInvalid

A row whose four values are 0 is padded in with its label 4 and the valid flag 0. The row after
it is valid.

### PadsNoRowsToABatchWithoutBoxes

With no label rows and a batch of 2, `max_boxes` is 0 and the labels, boxes and flags are empty.

### RejectsARowOutsideTheBatch

A row naming image 3 in a batch of 3 throws with a message containing "row 0 names image 3" and
"batch of 3".

### BuildsTheLossLeavesOfAnAssignment

The crowded scene is assigned with `topk` 3 and its leaves are built for an image 64 px wide and
32 px high. The three weights are 1, 2/3 and 1/3, so `target_scores_sum` is 2 and the
normalised weights of anchors 5 and 10 are 1/2 and 1/3. Row 5 has a box mask of four ones, the
target box (0, 0, 4, 4) in grid units, the target distances (1.5, 1.5, 2.5, 2.5) and an
`l1_scale` of 8 / 64 * 0.25 * 0.5 across and 8 / 32 * 0.25 * 0.5 down. The unequal width and
height separate the two `l1_scale` values.

### ReadsEachRowsAnchorAcrossImagesAndLevels

Two levels give five anchors, and a hand-built assignment for two images marks rows 1 and 9.
Row 1 is anchor 1 of image 0, at point (1.5, 0.5) on stride 8. Row 9 is anchor 4 of image 1, at
point (0.5, 0.5) on stride 16. Each row's target box, distances and `l1_scale` are computed
from its own point and stride, and `target_scores_sum` is 1.5. Row 4, the same anchor as row 9
in image 0, has a box mask of zeros.

### AnUnassignedRowCarriesTheSafeBoxAndNoWeight

Row 0 of the crowded scene is off the foreground. Its target box is (0, 0, 1, 1), and its box
mask, `l1_scale`, target distances, weight and normalised weight are 0. `safe_boxes` holds
(0, 0, 1, 1) in row 5 as well, which is a foreground row.

### TheScoreSumIsAtLeastOne

With no ground truth, `target_scores_sum` is 1.

### PadsTheReferenceLabelsToItsTargetBoxes

The reference labels pad to four valid rows. Gathering the padded boxes and labels by the
reference's `one2many.assignment.target_indices` gives its `target_boxes` within
`K_REFERENCE_BOX_TOLERANCE` and its `target_labels` exactly. The case is skipped when
`loss.sacp` is absent.

### ReproducesTheReferenceOneToManyAssignment

`expect_reference_assignment` runs the one-to-many preset and expects 31 foreground anchors,
the manifest's `foreground_anchor_count.one2many`, of which 4 hold box 3. The case is skipped
when `loss.sacp` is absent.

### ReproducesTheReferenceOneToOneAssignment

The same helper runs the one-to-one preset and expects 4 foreground anchors, the manifest's
`foreground_anchor_count.one2one`, of which 1 holds box 3. The case is skipped when `loss.sacp`
is absent.

## test_detect_average_precision.cpp

Tests detection matching, the 101-point average precision and the validation table.

This file needs no device and reads no fixture. Every case in `Unit_AveragePrecision` builds
detection rows and ground-truth boxes by hand and calls `match_detections`,
`integrate_precision` or the `AveragePrecision` accumulator of
`include/SushiAI/detect/average_precision.hpp`. A comment in each case derives the expected
number from the recall and precision the detections produce.

### K_TOLERANCE

`K_TOLERANCE` is 1e-12, the absolute tolerance of a metric, about 4500 float64 steps at 1.

### K_FULL_AP

`K_FULL_AP` is 0.995, the area under an envelope that is 1 up to recall 1. The integration
samples 101 recall points and the envelope is 0 at the last one, so the area is
0.99 + 0.01 / 2 and not 1.

### The boxes

`K_BOX_A` is the 8 by 8 box at the origin. `K_BOX_B` and `K_BOX_FAR` are 8 by 8 boxes that
touch neither it nor each other. `K_BOX_LOOSE` is the 8 by 5 box whose IoU with `K_BOX_A` is
40 / 64, which is 0.625, and `K_BOX_NEAR` is the 8 by 7 box whose IoU with it is 56 / 64, which
is 0.875. `K_NO_VALUES` and `K_NO_CLASSES` are empty arrays.

### make_row

`make_row` returns one detection row: a box, then a score and a class.

### join

`join` lays several arrays end to end.

### match

`match` calls `match_detections` for one image and returns the flags. It sizes the array from
`K_DETECTION_VALUES` and fills it with 7 first, so a flag the function does not write is
neither 0 nor 1.

### add_image

The file-local `add_image` passes three vectors to `AveragePrecision::add_image`.

### summarize_image

`summarize_image` returns the table of an accumulator that holds one image.

### HoldsTheTenReferenceThresholds

`K_IOU_THRESHOLDS` has ten entries, and each equals a hexadecimal float literal that carries
the float32 bits of `torch.linspace(0.5, 0.95, 10)`. The comparison is exact, so a threshold
that rounds to a neighbouring float32 fails it.

### IntegratesAnEmptyCurveToOneHalf

`integrate_precision` of two empty curves is 0.5. With no points the envelope is the line from
1 at recall 0 to 0 at recall 1.

### IntegratesUnderThePrecisionEnvelope

Recall 0.5, 0.5, 1 with precision 1, 0.5, 2/3 integrates to
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3. The envelope lifts the dip to 0.5 up to the 2/3 that
follows it at higher recall.

### RejectsCurvesOfUnequalLength

Two recall values with one precision value throw with a message containing "1 and 2".

### PerfectDetectorScoresTheInterpolationCeiling

Two images each hold one box, of classes 0 and 1, and one detection that equals it. Both rows
have precision 1, recall 1 and `K_FULL_AP` in both AP columns, and the four means are the same
numbers.

### NoDetectionsScoreZero

One box of class 0 and no detection give one row, with label 0 and one instance. Precision,
recall and both AP columns are 0 in the row and in the means.

### FalsePositiveAboveTheTruePositiveHalvesPrecision

A miss at score 0.9 ranks above a hit at 0.6. Recall is 0 then 1 and precision 0 then 0.5, so
the AP is 0.99 * 0.5 + 0.01 * 0.25, which is 0.4975, at every threshold. The mean F1 is flat
below confidence 0.6, so the first confidence point is read: precision 0.5 and recall 1.

### ReadsPrecisionAndRecallAtTheBestF1Confidence

A hit at 0.9 ranks above a miss at 0.3. F1 rises from 2/3 at confidence 0.3 to 1 at 0.9 and is 0
above it, so the smoothed curve is largest at point 849 of 1000. Precision there is
1 - 0.5 * (0.9 - 849 / 999) / (0.9 - 0.3), and the case also compares it with the literal
0.9582082255854392. Recall is 1. The row and the means hold the same values. Reading the first
confidence point gives precision 0.5.

### TwoDetectionsOnOneBoxCountOneMatch

Two detections equal `K_BOX_A` and a third equals `K_BOX_B`, all of class 0. The flags are 1
for the first, 0 for the second and 1 for the third at all ten thresholds. The AP at IoU 0.5 is
0.49 + 0.05 / 6 + 0.98 / 3 + 0.01 / 3, precision is 2/3 and recall is 1. Three matches give
0.995.

### ClassWithoutGroundTruthIsLeftOutOfTheMean

A detection of class 3 has no ground truth of its class. The table has one row, for class 0,
and the means are `K_FULL_AP`, 1 and 1. A row for class 3 halves every mean.

### ClassWithoutDetectionsCountsAsZeroInTheMean

Classes 0 and 1 both have a box and only class 0 has a detection. The row of class 1 has AP 0,
and the means are 0.4975 for both AP columns and 0.5 for precision and recall.

### EqualScoresKeepTheOrderTheyWereAddedIn

A miss and a hit share the score 0.5. Added miss first, the AP at IoU 0.5 is 0.4975; added hit
first, it is `K_FULL_AP`. The ranking is therefore stable.

### LooseDetectionCountsAtTheLowThresholdsOnly

A detection at IoU 0.625 has the flags 1, 1, 1 and then seven zeros. Its AP at IoU 0.5 is
`K_FULL_AP` and its mean over the ten thresholds is 3 * 0.995 / 10, which is 0.2985.

### ImageWithoutGroundTruthAddsFalsePositives

The second image has no ground truth and one detection at score 0.9, which ranks above the
first image's hit at 0.6. The summary counts 2 images and 1 instance, the class row counts 1
image, the AP at IoU 0.5 is 0.4975, precision is 0.5 and recall is 1.

### ImageWithoutDetectionsLowersRecall

The second image has a box and no detection. `image_count()` is 2, the class row counts 2
images and 2 instances, recall is 0.5, precision is 1, and both AP columns are 0.49 + 0.005,
which is 0.495.

### DetectionWhoseBestBoxIsTakenStaysUnmatched

The first detection takes `K_BOX_A`. The second overlaps `K_BOX_A` by 0.875 and `K_BOX_LOOSE`
by 40 / 56, about 0.714. Its flags are ten zeros: a detection is matched to its box of largest
IoU or to none.

### DetectionTakesTheBoxOfLargestIou

The boxes are listed `K_BOX_LOOSE` then `K_BOX_A`, and one detection equals `K_BOX_NEAR`. Its
flags are eight ones and two zeros, which is the IoU 0.875 it has with the box listed second. A
matcher that took the first box passing 0.5 gives five ones.

### DetectionWithNanBoxDoesNotMatch

A detection whose x1 is NaN has a NaN IoU with `K_BOX_A`, and its flags are ten zeros. The row
is built by hand because `make_row` takes a box constant.

### BoxGoesToItsFirstDetectionAtEachThreshold

A detection at IoU 0.625 ranks above one at IoU 1. The first holds the box at 0.5, 0.55 and
0.6; the second gets it from 0.65 up. The flags are 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 and
0, 0, 0, 1, 1, 1, 1, 1, 1, 1.

### DetectionOfAnotherClassDoesNotMatch

A detection of class 1 that equals a box of class 0 has ten zero flags.

### NoGroundTruthGivesNoRows

An empty accumulator has no rows. After an image with one detection and no ground truth it
counts 1 image, has no rows, and its four means are 0.

### ClearDropsEveryImage

An image with a miss is added and `clear()` is called. `image_count()` is 0. An image with a
hit added afterwards gives 1 image, 1 instance and AP `K_FULL_AP`, so nothing of the first
image remains.

### RejectsMalformedImages

`add_image` throws for a detection array of 5 values ("[N, 6]", "got 5"), for one box with two
classes ("4 values", "2 classes"), for a ground-truth class of -1 ("box 0", "negative class
-1"), for a detection class of 0.5 ("detection 0", "not a class index") and for a NaN score
("detection 0", "score"). The accumulator still holds no image and no rows afterwards.
`match_detections` throws for a box without a class ("0 classes") and for a flag array of 9
values ("need 10 values", "got 9").

## test_detect_boxes.cpp

Tests box decoding and encoding, the two box formats and the IoU forms.

This file needs no device. `Unit_DetectBoxes` covers `include/SushiAI/detect/box_coding.hpp`,
`include/SushiAI/detect/box_format.hpp` and `include/SushiAI/detect/iou.hpp`. The hand cases
write the expected value as the arithmetic that produces it. Three cases read `forward.sacp`
and report themselves skipped when the file is absent.

### Tolerances

`K_EXACT` is 0, for values float32 holds exactly. `K_DECODE_TOLERANCE` is 1.3e-4, two float32
steps at 640, for a decoded box in pixels. `K_ROUND_TRIP_TOLERANCE` is 1.6e-5, two steps at 80,
for a decode followed by an encode in grid units. `K_IOU_TOLERANCE` is 1e-6, eight float32
steps at 1.

### K_REFERENCE_ANCHORS and K_REFERENCE_LEVELS

`K_REFERENCE_ANCHORS` is 8400 and `K_REFERENCE_LEVELS` holds the three levels of a 640 px
input, both from "levels" in `tests/fixtures/yolo26/MANIFEST.json`.

### ReferenceGrid

`ReferenceGrid` holds the `[8400, 2]` points and the `[8400]` strides of the reference levels
in grid units, and `make_reference_grid` fills one with `make_anchors`.

### join_raw_boxes

`join_raw_boxes` reads a branch's three per-level tensors, `<branch>.level<i>.boxes`, and
copies each into its place in an `[8400, 4]` array with `join_level`. It returns an empty
string, or the reason a level could not be read.

### DecodesDistancesToCorners

`decode_xyxy` of the distances (0.5, 1, 1.5, 2) at the point (1.5, 2.5) gives
(1, 1.5, 3, 4.5): left and top are subtracted from the point, right and bottom added.

### DecodesDistancesToCentreAndSize

`decode_xywh` of the same input gives the midpoint (2, 3) and the size (2, 3) of those corners.

### RepeatsTheAnchorPointsForEachImage

Two anchor points and four rows of distances decode as two images: rows 2 and 3 use points 0
and 1 again.

### EncodeInvertsDecode

`encode_distances` of the boxes `decode_xyxy` writes returns the three rows of distances at
tolerance 0.

### ScalesEachBoxByItsAnchorStride

`scale_boxes` multiplies all four values of a row by its anchor's stride: the row (1, 2, 3, 4)
becomes (8, 16, 24, 32) at stride 8 and (16, 32, 48, 64) at stride 16.

### ConvertsBetweenCornersAndCentreSize

`xyxy_to_xywh` maps (1, 2, 5, 8) to (3, 5, 4, 6), and `xywh_to_xyxy` maps that back to the
corners.

### RejectsRowsThatDoNotCoverTheAnchors

Three rows of distances over two anchor points throw with a message containing "2 anchors" and
"12".

### ReproducesTheReferenceOneToManyDecode

The raw one-to-many boxes, decoded with `decode_xywh` and scaled by the strides, equal
`one2many.decoded.boxes` within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp`
is absent.

### ReproducesTheReferenceOneToOneDecode

The raw one-to-one boxes, decoded with `decode_xyxy` and scaled, equal `one2one.decoded.boxes`
within `K_DECODE_TOLERANCE`. The case is skipped when `forward.sacp` is absent.

### EncodeInvertsTheDecodeOfTheReferenceBoxes

Encoding the decode of the raw one-to-one boxes returns them within `K_ROUND_TRIP_TOLERANCE`.
The case is skipped when `forward.sacp` is absent.

### IdenticalBoxesScoreOne

A 2 by 2 box has IoU 1 and complete IoU 1 with itself.

### DisjointBoxesScoreZeroAndPayForDistance

Two unit squares two units apart have IoU 0. Their complete IoU is minus the squared centre
distance, 4, over the squared hull diagonal, 10.

### OverlappingSquaresLoseTheCentreTerm

Two 2 by 2 squares offset by 1 on both axes have IoU 1 / 7, and their complete IoU is that
minus 2 / 18. The aspect ratios are equal, so the aspect term is 0.

### DifferentAspectRatiosLoseTheAspectTerm

A 4 by 2 box and a 2 by 2 box at one corner have IoU 0.5. The expected complete IoU is computed
in double from the centre term, the angle gap `atan(1) - atan(2)` and the trade-off factor, and
compared within `K_IOU_TOLERANCE`.

### ADegenerateBoxStaysFinite

A box of zero size inside a 2 by 2 box has IoU 0. Its width over its height is 0 over eps, so
the angle gap is pi / 4, the aspect term is 0.25, and the complete IoU is -0.25 * 0.25 / 1.25,
which is -0.05. The complete IoU of the zero-size box with itself is finite and 0.

### ElementwiseFormsPairRows

`elementwise_iou` and `elementwise_complete_iou` over two rows give 1 / 7 and 0, and
1 / 7 - 2 / 18 and -4 / 10: row i of the first array is paired with row i of the second.

### PairwiseFormFillsEveryPair

`pairwise_iou` of two boxes against three writes six values, row n and column m holding the
IoU of box n with box m: 1, 1 / 7, 0, then 1 / 4, 0, 0.

### RejectsElementwiseInputsOfDifferentLength

Arrays of 8 and 4 values with an output of 2 throw with a message containing "got 8, 4 and 2".

## test_detect_postprocess.cpp

Tests the top-k selection, both postprocess paths and the overlap suppression.

This file needs no device. `Unit_DetectTopK` covers `include/SushiAI/detect/top_k.hpp`, and
`Unit_DetectPostprocess` covers `include/SushiAI/detect/detections_topk.hpp` and
`include/SushiAI/detect/detections_nms.hpp`. The suppression path takes boxes as centre and
size, so the hand cases write them that way and expect corners. Four cases read `forward.sacp`
and report themselves skipped when the file is absent.

Two of the reference comparisons cannot compare row by row. The reference orders equal scores
in an order torch does not specify, and the fixture's scores are nearly constant.
`include/SushiAI/detect/README.md`, under `top_k.hpp` and "Reference rows the suppression test
reads", gives the counts.

### K_EXACT

`K_EXACT` is the tolerance 0. A postprocess copies its output values from its inputs.

### K_SUPPRESSION_CONFIDENCE

`K_SUPPRESSION_CONFIDENCE` is 1.8e-4, the `suppression.arguments.conf_thres` of `forward.sacp`
in `tests/fixtures/yolo26/MANIFEST.json`.

### Row

`Row` is an array of six floats, one detection.

### sorted_rows_above

`sorted_rows_above` returns the rows whose score exceeds a floor, sorted by their six values.
Two detection arrays that hold the same rows in different orders compare equal through it.

### is_anchor_class_pair

`is_anchor_class_pair` reports whether a row's box is some anchor's box and its score is that
anchor's score in the row's class.

### make_nms_config

`make_nms_config` returns an `NmsConfig` with the given confidence threshold, IoU threshold,
label mode and class mode, and the default limits.

### suppress_one_image

`suppress_one_image` runs `select_nms_detections` on one image of `[A, 4]` boxes and returns
that image's rows.

### OrdersByValueThenIndex

`select_top_k` of the values 0.5, 0.9, 0.5, 0.1, 0.9 with k = 4 returns the indices 1, 4, 0, 2:
larger values first, equal values by ascending index. Asking for 6 of 5 values throws with a
message containing "6" and "5 values".

### SelectsTheHighestScoringAnchorClassPairs

Three anchors with two classes and a limit of 2 give the row of anchor 1 with score 0.9 and
class 0, then the row of anchor 0 with score 0.7 and class 1.

### OneAnchorCanYieldTwoDetections

Anchor 1 scores 0.8 and 0.7 in its two classes, both above anchor 0's 0.6. Both output rows
carry anchor 1's box, with classes 0 and 1.

### LimitsRowsToTheAnchorCount

`count_top_detections` returns 3 for three anchors and a limit of 300, and 300 for 8400 anchors.

### SelectsEachImageFromItsOwnRows

A batch of two images with two anchors each and a limit of 1 gives one row per image: anchor 1
of image 0 at 0.6 in class 0, and anchor 1 of image 1 at 0.9 in class 1.

### RejectsADetectionArrayOfTheWrongSize

An output of 6 values where the shape needs `[1, 2, 6]` throws with a message containing
"[1, 2, 6] detections" and "6 values".

### ReproducesTheReferenceDetections

`select_top_detections` on `one2one.decoded.boxes` and `one2one.decoded.scores` with
`K_MAX_DETECTIONS` is compared with `one2one.detections` in three ways. The score column and
the class column are equal row by row. The rows above the last score are equal as a set,
through `sorted_rows_above`. Every output row passes `is_anchor_class_pair`. The case is
skipped when `forward.sacp` is absent.

### TheConfidenceFilterDropsAScoreAtTheThreshold

`keep_confident_detections` at 0.25 keeps the rows at 0.5 and 0.26, in their order, and drops
the rows at 0.25 and 0.1.

### TheConfidenceFilterKeepsEachImagesRows

Four rows split between two images at `K_VALIDATION_CONFIDENCE`: image 0 keeps its row at 0.9
and image 1 keeps none.

### TheConfidenceFilterRejectsRowsTheBatchDoesNotDivide

Three rows for a batch of 2 throw with a message containing "[2, k, 6] detections" and "18
values". A batch of 0 throws with "positive batch" and "got 0".

### TheConfidenceFilterKeepsTheRowsTheReferenceValidatorKeeps

At `K_VALIDATION_CONFIDENCE` the filter keeps none of the 300 rows of `one2one.detections`,
which is `validation.row_count.one2one` of the manifest. At threshold 0 it keeps all of them
unchanged, since every reference score is a sigmoid above zero. The case is skipped when
`forward.sacp` is absent.

### SuppressionReproducesTheReferenceRows

`select_nms_detections` at `K_SUPPRESSION_CONFIDENCE` on the one-to-many decode returns as many
values as `one2many.suppressed.detections`. The score column is equal row by row and the rows
above the last score are equal as a set. The case is skipped when `forward.sacp` is absent.

### NoReferenceScorePassesTheValidationConfidence

With a default `NmsConfig` the one-to-many decode yields no row, which is
`validation.row_count.one2many` of the manifest. The case is skipped when `forward.sacp` is
absent.

### SuppressionDefaultsMatchTheValidator

A default `NmsConfig` has confidence 0.001, IoU 0.7, `LabelMode::MultiLabel`,
`ClassMode::ClassAware`, 300 detections, 30000 candidates and a class offset of 7680.

### SuppressesTheLowerOfTwoOverlappingBoxes

Boxes 0 and 1 share 81 of 119 px, an IoU of 0.68, and box 2 is apart. `suppress_overlaps` at 0.5
keeps the indices 0 and 2.

### ListsSurvivorsByDescendingScore

Four boxes that do not overlap, with scores 0.1, 0.9, 0.5 and 0.5, are kept in the order
1, 2, 3, 0.

### KeepsABoxWhoseIouEqualsTheThreshold

Two boxes share 2 of 4 px, an IoU of exactly 0.5 in float32. At threshold 0.5 both are kept; at
0.25 only the first is.

### OverlappingBoxesOfOneClassCollapse

Two anchors hold the same box and score 0.9 and 0.8 in class 0. One row remains, with corners
(0, 0, 10, 10), score 0.9 and class 0.

### OverlappingBoxesOfDifferentClassesBothSurvive

The same two boxes score in classes 0 and 1. `ClassMode::ClassAware` keeps both rows, and
`ClassMode::ClassAgnostic` keeps the one at 0.9.

### DropsAScoreEqualToTheConfidenceThreshold

An anchor at 0.25 with the threshold at 0.25 yields no row. The anchor at 0.5 yields the one
row.

### AnAnchorContributesEveryPassingClassOrItsBest

One anchor scores 0.6, 0.1 and 0.7 in three classes. `LabelMode::MultiLabel` yields two rows,
class 2 at 0.7 then class 0 at 0.6. `LabelMode::BestClass` yields the row of class 2 alone.

### LimitsDetectionsToTheHighestScores

Three boxes that are apart score 0.5, 0.9 and 0.7. With `max_detections` 2 the rows at 0.9 and
0.7 remain.

### LimitsCandidatesBeforeSuppression

With `max_candidates` 1 the same input yields the row at 0.9 alone.

### RejectsAThresholdOutsideTheUnitInterval

A confidence threshold of 1.5 throws with a message containing "[0, 1]" and "1.5".

### AnImageBelowTheThresholdYieldsNoRows

Scores of 0.25, 0.1, 0.2 and 0 at the threshold 0.25 yield an empty row vector.

### SuppressesEachImageFromItsOwnRows

A batch of two images with two anchors each. Image 0 holds one box twice, so the row at 0.9
remains. Image 1 holds two boxes that are apart, and its rows are 0.6 then 0.3.

### RejectsScoresThatDoNotFillTheShape

Six scores where the shape `[1, 2, 2]` needs four throw with a message containing "[1, 2, 2]
scores" and "8 and 6 values".

## test_diagnostics.cpp

Tests that each host-lane precondition raises a message carrying the values that disagree.

This file needs no device. It holds every host-lane precondition to the message it promises.

Other tests already prove that these preconditions throw. They do not prove that the messages
still say anything. A refactor that funnels five specific diagnostics into one generic "invalid
argument" leaves every EXPECT_THROW green and destroys the only thing that makes a shape
mismatch debuggable. So each case names the values the message must still carry (the two
extents that disagree, the operation that has no rule, the dtype that cannot be served) and
fails when one of them goes missing.

`tests/common/test_throws.hpp` defines SA_EXPECT_ERROR.

The cases fall into five groups, in file order:

- Shape
- GraphBuilder, the shape-inference boundary
- Graph: single assignment and operand existence
- layers
- differentiation

### ShapeRankBeyondTheMaximumNamesTheRankAndTheLimit

K_MAX_RANK is SushiBLAS's, so the number in the message is the one a reader has to reconcile
against the tensor they were trying to build.

### ShapeNonPositiveExtentNamesTheOffendingExtent

A zero-element tensor is refused at the type that describes extents, which is why nothing
further down ever has to handle one.

### MatmulRejectsARank1OperandAndPrintsWhatItGot

Rank 1 where rank 2 belongs is the commonest way a bias or a flattened sample reaches a gemm.
The message has to show both shapes or the reader cannot tell which side was wrong.

### ASecondProducerNamesBothOperationsAndTheRule

The second write is what would make a differentiated graph silently wrong, so the message has
to say which two operations collided.

### AOneElementRootOfADifferentiableOperationIsSeeded

The root is a [1, 1] K_MATMUL of a parameter `x` and an input `w`. Differentiation does not
throw and appends two nodes. The first is a K_CONSTANT in the root's shape, the seed. The second
is a K_MATMUL that reads the seed and `w`, and its result is the gradient of `x`, in the shape of
`x`. `test_loss_roots.cpp` covers the seed in full.

### ARootThatRequiresNoGradientNamesItsProducerAndTheCause

Two roots are refused. A K_MATMUL of two inputs is reached by no parameter, and the message
contains "value 3", "sushiai.matmul" and "requires no gradient". A K_MATMUL that reads the
parameter through `stop_gradient` is refused with a message containing "sushiai.matmul",
"requires no gradient" and "K_STOP_GRADIENT", so the message names the barrier as a cause the
caller can look for.

### ARootWhoseProducerHasNoRuleNamesTheProducer

The root is a K_DYNAMIC_SCALE of a parameter, an operation with no backward rule. The message
contains "sushiai.dynamic_scale", "has no backward rule" and "autograd/rules/", the folder a
rule is added in.

### ALossNoOperationProducesIsRejected

The `orphan` value is declared but never written. There is no operation to blame, so the
message has to say that and cannot name one.

### AnOperationWithNoBackwardRuleNamesItselfAndItsNode

"sushiai.unknown" is what `name_of()` reports for a foreign identity, and the node index is
what turns that into something findable.

### LogSoftmaxOutsideTheFusedPairExplainsTheRestriction

Naming the way out matters more than naming the failure. The caller has to be told to route
through K_NLL_LOSS, and not merely that log-softmax said no.

### AGradientContributedToTheWrongShapeNamesBothShapes

This error is reachable only from inside a backward rule, so the test provokes it the way a
buggy rule would. It builds a graph by hand whose K_SUB claims a result of a different shape from
its operands, so the rule contributes a [4, 3] gradient to a [2, 6] value.

## test_elementwise_ops.cpp

Tests the nodes the builder emits for the elementwise, comparison, selection and constant
operations.

This file needs no device. Every case in `Unit_ElementwiseOps` calls an emitter of
`include/SushiAI/graph/builder.hpp` on a host-side graph and reads the node and the value it
wrote. The emitters are `div`, `minimum`, `maximum`, `less_equal`, `abs`, `atan`, `softplus`,
`silu`, `select`, `activation_backward`, `constant` and `constant_like`. The cases that concern
several emitters loop over a table and label each iteration with `SCOPED_TRACE`, so a failure
names the emitter. `test_backward.cpp` covers the backward rules of the same operations.

### BinaryEmitter and UnaryEmitter

`BinaryEmitter` holds a label, the identity an emitter must write and a pointer to a
two-operand member of `GraphBuilder`. `UnaryEmitter` is the same for one operand.

### K_BINARY_EMITTERS

`K_BINARY_EMITTERS` lists `div`, `minimum`, `maximum` and `less_equal` with K_DIV, K_MINIMUM,
K_MAXIMUM and K_LESS_EQUAL.

### K_UNARY_EMITTERS

`K_UNARY_EMITTERS` lists `abs`, `atan`, `softplus` and `silu` with K_ABS, K_ATAN, K_SOFTPLUS and K_SILU.

### K_INPUT_READING_GRADIENTS

`K_INPUT_READING_GRADIENTS` lists K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, the two activation
gradients that read the forward input.

### nchw_input

`nchw_input` declares a [2, 2, 2, 2] input and converts it to NCHW. Every extent is 2, so the
permuted shape equals the original and two such values still agree in shape.

### expect_produced_by

`expect_produced_by` expects a value to be the single result of a node of a given operation
that reads a given operand list in order.

### expect_described_like

`expect_described_like` expects a value to have the shape, element type and layout of another
value and the lifetime ACTIVATION.

### BinaryEmittersWriteTheirIdentityOverBothOperandsInOrder

Each two-operand emitter writes one node of its own identity whose inputs are `a` then `b`.

### BinaryEmittersKeepShapeDtypeAndLayout

With two FLOAT64 NCHW operands, the result of each two-operand emitter has the first operand's
shape and is FLOAT64 and NCHW.

### BinaryEmittersRejectUnequalShapesWithoutBroadcasting

Each two-operand emitter throws for a [4, 3] operand with a [4, 1] one, in both orders, and
with a [3] one. These are the pairs a broadcasting library accepts. The node count is unchanged
after the three calls.

### BinaryEmittersRejectUnequalDtypes

Each two-operand emitter throws for a FLOAT32 operand with a FLOAT64 one, in both orders.

### LessEqualYieldsAMaskInTheOperandsOwnDtype

The result of `less_equal` is FLOAT32 for FLOAT32 operands and FLOAT64 for FLOAT64 ones. The
mask has no element type of its own, which is what lets `mul` and `select` read it beside a
value of the same type.

### UnaryEmittersWriteTheirIdentityOverTheirOperand

Each single-operand emitter returns a value other than its operand, written by one node of its
own identity that reads the operand.

### UnaryEmittersKeepShapeDtypeAndLayout

With a FLOAT64 NCHW operand, the result of each single-operand emitter has the operand's shape
and is FLOAT64 and NCHW.

### SelectReadsItsConditionThenBothBranches

`select(condition, a, b)` writes one K_SELECT node whose inputs are the condition, `a` and `b`,
in that order.

### SelectKeepsShapeDtypeAndLayout

With three FLOAT64 NCHW operands the result has the shape, element type and layout of `a`.

### SelectRejectsAConditionOfAnotherShape

A [4, 1] condition over [4, 3] branches throws, and the node count is unchanged.

### SelectRejectsAConditionOfAnotherDtype

An INT64 condition over FLOAT32 branches throws.

### SelectRejectsBranchesOfUnequalShape

A [4, 3] branch with a [4, 1] branch throws in both orders.

### SelectRejectsBranchesOfUnequalDtype

A FLOAT32 branch with a FLOAT64 branch throws in both orders.

### ActivationBackwardEmitsTheInputReadingGradients

For K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD, `activation_backward(gradient, dy, x)` writes one node
of that identity whose inputs are `dy` then `x`, and the result is described like `dy`.

### InputReadingGradientsRejectAGradientOfAnotherShape

A [4, 1] `dy` with a [4, 3] `x` throws for both gradients.

### InputReadingGradientsRejectAGradientOfAnotherDtype

A FLOAT32 value with a FLOAT64 one throws for both gradients, in both orders.

### ActivationBackwardRejectsAnIdentityThatIsNotAnActivationGradient

`activation_backward` throws when the identity is K_SILU, K_SOFTPLUS or K_DIV. The first two are
forward identities of `Ops::K_ACTIVATIONS` and the third is not in the table.

### FinaliseAcceptsAGraphUsingEveryEmitter

Two NCHW inputs are chained through all four two-operand emitters and all four single-operand
ones, and the chain ends in a `select` on a `less_equal` mask. `Graph::finalise()` does not
throw.

### FinaliseRejectsOperandsInTwoLayouts

For each two-operand emitter, a node that reads one NCHW operand and one NHWC operand is
emitted, and `finalise()` throws. The emitter itself accepts the pair because the shapes are
equal.

### FinaliseRejectsASelectConditionInAnotherLayout

A `select` with an NHWC condition over NCHW branches is emitted, and `finalise()` throws.

### ResultsOfAParameterRequireAGradient

After `finalise()`, `silu(div(weight, constant))` requires a gradient and
`atan(maximum(constant, constant))` over an input does not.

### AComparisonOfAParameterRequiresNoGradient

After `finalise()`, the mask `less_equal(weight, bound)` requires no gradient, and neither does
a `select` that reads it as its condition over two inputs.

### ConstantWritesAnOperandFreeNodeHoldingItsValue

`constant([4, 3], 1e-7, FLOAT64)` writes one K_CONSTANT node with no input. Attribute slot 0
holds 1e-7 within 1e-20, and the value is [4, 3], FLOAT64, NHWC and ACTIVATION. `K_VALUE` and
`K_VALUE_TOLERANCE` are constants local to the case.

### ConstantDefaultsToFloat32

`constant([2], 1.0)` with no element type is FLOAT32.

### ConstantRejectsAnElementTypeThatIsNotRealFloating

`constant` throws for INT64 and for COMPLEX32, and the node count and the value count are
unchanged.

### ConstantLikeCopiesShapeDtypeAndLayoutWithoutReadingItsOperand

`constant_like(x, 0.5)` over a FLOAT64 NCHW value writes a K_CONSTANT node with no input whose
result is described like `x`. The value it names is not an operand of the node.

### ConstantLikeRejectsAValueThatIsNotRealFloating

`constant_like` over an INT64 value throws, and the node count and the value count are
unchanged.

### AConstantLikeAValueIsAnOperandThatFinaliseAccepts

`add(x, constant_like(x, 1e-7))` over an NCHW input passes `finalise()`, and the sum requires
no gradient.

### ZerosStillWritesTheZeroIdentity

`zeros` and `zeros_like` each write a K_ZERO node with no input. K_ZERO and K_CONSTANT are two
identities.

## test_flatten.cpp

Tests the Flatten layer's trace, shape rule, rejections and parameter walk.

This file needs no device. `NN::Flatten` in `include/SushiAI/nn/flatten.hpp` has no field and
no parameter. It folds every axis after the batch axis into one.

### traced_shape

`traced_shape` traces a `Flatten` over a new input of the given batched shape and returns the
result's shape.

### SatisfiesTheModuleTrait

Two `static_assert` lines require `Flatten` to be a `Module` and not a `ModuleContainer`.

### ARank4InputTracesToOneReshape

A [2, 4, 5, 3] input traces to a [2, 60] value. The trace adds exactly one node, a K_RESHAPE
with the input as its one operand and the result as its one output.

### ARank3InputFoldsItsTwoInnerAxes

A [2, 3, 4] input traces to [2, 12].

### ARank2InputPassesThroughWithNoNode

For a [2, 7] input `forward()` returns the id it was given. The node count and the value count
are unchanged, so a flatten placed after a layer that already yields feature vectors adds
nothing to the step.

### AnNhwcInputYieldsAnNhwcResult

The input is asserted to be NHWC first, and the flattened result is NHWC too.

### ForwardRejectsAnNchwInput

The layout tag of a [2, 3, 4, 5] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeAgreesWithTheTracedShape

`output_shape()` maps [4, 5, 3] to [60], [3, 4] to [12] and [7] to [7]. For each, the traced
shape of a batch equals the batch extent followed by that one extent.

### ForwardRejectsARank1Input

`forward()` throws on a [6] input, which has a batch axis and no per-sample axis.

### OutputShapeRejectsARank0Shape

`output_shape()` throws on an empty shape.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## test_fusion.cpp

Tests the matches and the refusals of the static fusion pass on synthetic graphs.

This file needs no device. The static fusion pass is a pure function of the IR, as the memory
planner is, so every case is a synthetic graph and a table of expectations with no engine
anywhere.

The cases pin the set of refusals more than the matches. A pass that fuses too little is slow.
A pass that fuses too much is silently wrong: the graph still runs, the loss still falls, and
the only symptom is a model that converges somewhere slightly different. So most cases are
graphs the pass must decline, one per rule:

- an intermediate with a second consumer (fusing it means losing it),
- an intermediate the caller reads by id (the same loss, seen from outside),
- a spilled pre-activation read by a node that runs before the fused one.

The activation whose gradient differentiates at its input used to be a fourth entry on that
list. It no longer is. K_GEMM_BIAS_GELU folds it by spilling the pre-activation to a second
output, so the refusal became a condition: an INPUT-gradient activation may be folded only by an
entry that writes the tensor its gradient reads, and only if it writes the right one. That
condition has a compile-time half in `src/graph/fusion.cpp`, where an entry breaking either half
fails a static_assert. A test cannot cover that half, because a test that fails to compile is
not a test. This file checks the pair of properties the assertion reads and, above all, that the
fold leaves the pre-activation written and readable by the backward node.

The cases fall into four groups, in file order:

- the catalog is well formed
- the matches
- the refusals, which are the half that matters
- the pass keeps the graph usable

### EveryFormRoundTripsThroughItsOperationIdentity

The lowering dispatches on `fused_form_op` and `fused_form_of`. A form that does not round-trip
would lower as "no recorder for this operation", which is loud, but only at run time and only on
a graph that happens to contain it.

### AnActivationFoldsForFreeOrWithASpillAndTheTwoAreDifferentFacts

`activation_survives_fusion(K_RELU)` is the property the static_assert in `src/graph/fusion.cpp`
reads. If it ever flips for an activation that has a catalog entry, the static_assert catches
it, but only while the property stays true of the table the static_assert consults. This test
pins that.

`activation_survives_fusion(GELU)` is false because SushiBLAS's `gelu_backward` takes the
pre-activation. An entry that wrote only its result would differentiate at a tensor that never
reached memory. That has not changed and must not, because it is the premise the static_assert
reasons from.

`catalog_folds_activation(GELU)` is true because the conclusion drawn from that premise
changed. The one entry that spills folds GELU. "Survives fusion" and "the catalog folds it" are
now two questions with two answers, and a caller who reads the first as the second gets the
wrong answer for exactly one activation.

### AMatmulAndABiasAddBecomeOneNode

`report.fused().size()` is 1, and the fused node reads A, B and the bias and still writes the
value the caller named. That is the whole reason ids are preserved.

The `flag(0)` and `flag(1)` checks hold the transposition flags. They have to survive, because
the fused recorder hands them straight to `execute_gemm`, and losing one gives a wrong answer of
exactly the right shape.

### TheLongestMatchWinsOverTheShorterOneInsideIt

Both gemm+bias and gemm+bias+relu match this graph. Rooted at the bias_add, the shorter one
would win and the relu would be left outside for good, which is why the matcher walks roots in
reverse.

### ASubtractionWithItsOperandsTheOtherWayRoundDoesNotMatch

(exp(a) - b) and (b - exp(a)) are different functions with the same shapes, so nothing
downstream could tell them apart. The link position in the catalog entry is what refuses, and
this test shows that the pass consults it.

### AnIntermediateWithASecondConsumerIsNotFusedAway

The `also` value is a second reader of the K_MATMUL's result. Fusing would mean recomputing the
result for that reader, or losing it.

### AnIntermediateTheCallerReadsByIdIsNotFusedAway

The `observed` list names `biased`, which is the logits of a real classifier. The training loop
reads it after every replay to score accuracy, and a fused form gives it no buffer to read.

`report.count(K_GEMM_BIAS)` is still 1: the pass falls back to the shorter form rooted at the
bias_add, whose own intermediate is free.

### AParameterIsProtectedWithoutBeingListed

A persistent value's buffer outlives the step whatever the graph says, so the caller must not
have to remember to list it. In this graph the "intermediate" is a parameter. That is an odd
graph, but it is the shape a caller would get wrong.

The assignment `lifetime = OPTIMIZER_STATE` forces the exp's result into a persistent class, as
a checkpointed activation would be.

### AnActivationDifferentiatedAtItsInputFoldsOnlyBecauseTheEntrySpills

The graph is the trap the fusion design is arranged around. The differentiation pass has
already emitted `gelu_backward` reading the pre-activation, so that value has two consumers.
Every other entry would stop at that point under rule 1. This entry does not, for the same
reason it is allowed to exist: the fused node keeps writing that value, as a second output,
under the same id.

The check on the fused node's second output is the assertion that carries the test, and a plain
K_GEMM_BIAS_GELU with no spill would fail it. The fused node still writes the pre-activation as
its second output, and it is the same value, so `gelu_backward`'s operand needed no rewriting
and the backward rule needed no change.

`fused()[0].eliminated` holds only the K_MATMUL's result. A report that listed the spilled value
as eliminated would read like the report of the silently wrong version of this pass.

### ASpilledPreActivationReadBeforeTheFusedNodeIsRefused

The `early` reader shows the one new way the K_GEMM_BIAS_GELU entry could break the graph: a
wrong order, not a wrong number. The fused node is emitted at the K_GELU's position, so a reader
of the pre-activation sitting between the bias_add and the K_GELU would end up reading a value
written after it. Every pass downstream of fusion assumes that cannot happen.

`report.count(K_GEMM_BIAS)` is 1: the pass still falls back to the shorter form, as it does after
any other refusal of the three-node entry.

The closing loop over the nodes checks that the rewritten graph is still topological, which is
the property the refusal exists to protect.

### ASpilledPreActivationMayBeReadFromOutsideTheGraph

This case mirrors VALUE_IS_OBSERVED. That rule exists because a fused form gives an interior
value no buffer to read. The K_GEMM_BIAS_GELU entry gives it one, so naming the pre-activation as
observed must not refuse the fold.

### TheGeluFoldStillObeysEveryOtherRefusal

A spill licenses one interior value and no others. The K_MATMUL's result is not spilled, so a
second reader of it refuses the whole chain as it does for every other entry. The fallback is
nothing, because the same value blocks gemm+bias.

### DisablingTheGeluFormFallsBackToTheShorterOne

The last two assertions check that the GELU left outside is not also reported as unfoldable.
"Its gradient differentiates at its input" and "no catalog entry composes it" are both false for
this graph, and either would send a reader to change the wrong thing. The reason asserted just
before them, `FusionDecline::Disabled`, is the only one.

### ARecoverableActivationNextToAFusedGemmIsFoldedNotReported

This case mirrors DisablingTheGeluFormFallsBackToTheShorterOne, and it is the reason ReLU's
backward rule reads the output. With the gradient reading the activation's own result, the
pre-activation has one consumer and folds with no special case anywhere.

### ValueIdsAndTheirDescriptionsSurviveTheRewrite

Everything a caller holds across the pass (a StepSpec, a Parameter's own id, a gradient)
depends on ids surviving. A pass that compacted ids would need a remap reaching into the
model's layer objects, and forgetting one would bind the optimizer to the wrong tensor.

### TheRewrittenGraphIsStillInTopologicalOrder

The memory planner's reachability closure and the lowering's single forward walk both rest on
topological order. The fused node is emitted at the root's position while it reads operands
from the chain's first node, and that is where a rewrite gets the order wrong.

### StateLeavesAndTheRefusalMarkSurviveTheRewrite

The graph is a K_MATMUL, a K_BIAS_ADD whose bias is a buffer, and a K_RELU. The weight and the buffer
are bound to the states 21 and 22 and the graph is marked refused before `Graph::fuse` runs.
The report lists one fused form and the graph holds one K_GEMM_BIAS_RELU node, so the pass did
rebuild the graph. Both `state_leaf` lookups and the whole `state_leaves()` table are what they
were. The buffer keeps the lifetime `BUFFER` and no gradient, the weight still requires one,
and `has_refusal()` is still true.

## test_global_avg_pool.cpp

Tests the shapes GlobalAvgPool traces and reports, and the inputs it refuses.

This file needs no device. `NN::GlobalAvgPool` in `include/SushiAI/nn/global_avg_pool.hpp` has
no field and no parameter. It reduces the two spatial axes of an image to their mean.

### traced_shape

`traced_shape` traces the layer over an [N, H, W, C] input built from its four arguments and
returns the result's shape.

### SatisfiesTheModuleTrait

Three `static_assert` lines require `GlobalAvgPool` to be a `Module`, not to be a
`ModuleContainer`, and to have a `forward()` that is callable on a const object and returns
`Graph::ValueId`.

### ForwardReducesTheSpatialAxes

A [4, 7, 5, 16] input traces to [4, 16].

### ForwardEmitsOneSpatialMeanNode

The trace adds exactly one node, a K_SPATIAL_MEAN with the input as its one operand and the
result as its one output.

### ForwardKeepsTheNhwcLayoutTag

The result of an NHWC input is tagged NHWC, and `finalise()` accepts the graph.

### OutputShapeKeepsOnlyTheChannels

`output_shape([7, 5, 16])` is [16].

### OutputShapeAgreesWithForward

For a square [7, 7, 512] sample and an oblong [3, 9, 2] one, `output_shape()` is rank 1 and
the traced shape is the batch extent followed by its one extent.

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsARankTwoShape

`output_shape([7, 16])` throws.

### OutputShapeRejectsAShapeWithTheBatchAxis

`output_shape([4, 7, 5, 16])` throws.

### ForwardRejectsAFeatureMatrix

`forward()` throws on a [4, 16] input.

### ForwardRejectsAnInputWithoutTheBatchAxis

`forward()` throws on a [7, 5, 16] input.

### ForwardRejectsAnNchwInput

A [1, 4, 4, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace, and `parameter_elements` is 0 after it.

## test_gradcheck.cpp

Tests the gradient checker against closed-form functions and a wrong gradient.

This file needs no device. The gradient checker is the instrument the integration tier measures
with, so this file calibrates it against closed-form functions whose right answer is known. One
of them is a deliberately wrong gradient, which proves the check can fail.

### RejectsAGradientThatIsWrongByATenthOfAPercent

This case is the reason for insisting on FLOAT64. A 1e-3 relative error is far above the noise
floor of about 1e-9 in FLOAT64. It would sit inside the floor of about 2.4e-5 that an fp32
difference quotient bottoms out at, and would pass unnoticed.

### AReluFamilyFunctionNeedsToBeSampledOffTheKink

`options.min_step` is large enough that the difference survives the cancellation of summing it
against the O(1) terms. A step of 1e-4 keeps the pushed coordinates at 1e-3, where a central
difference of x^2 is still exact to 1e-9.

The `unshifted` point straddles the kink. The quotient taken across it reports the average of
the two one-sided derivatives, which no correct rule can match.

## test_identity.cpp

Tests that the Identity layer returns its input, emits nothing and holds no state.

This file needs no device. `NN::Identity` is in `include/SushiAI/nn/identity.hpp`.

Three `static_assert` lines at the top of the file hold that `Identity` is a `Module`, is not
a `ModuleContainer`, and that `Identity::fields()` is an empty tuple.

### ForwardReturnsItsInput

`forward` over a [2, 8, 6, 3] input returns the id it was given.

### ForwardEmitsNothing

The graph's node count and value count are the same before and after `forward`.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## test_imaging_affine.cpp

Tests the plan and the application of the scale and translation warp to an image and its
labels.

This file needs no device. `Unit_ImagingAffine` covers `include/SushiAI/imaging/affine.hpp`.
`plan_affine` draws a scale and a translation per axis and stores the 2 by 3 matrix;
`apply_affine` warps an image with bilinear weights, or warps, clips and filters a label set.
The matrix has no rotation and no shear, and `apply_affine` refuses a plan that has one.

### K_PIXEL_TOLERANCE

`K_PIXEL_TOLERANCE` is 1e-3 px, the tolerance of a box edge or a matrix term.

### Matrix3 and multiply

`Matrix3` is a row-major 3 by 3 float matrix, the form the reference builds its factors in.
`multiply` returns the product of two of them with each term summed left to right.

### make_pattern_image

`make_pattern_image` returns an image whose byte at row y, column x, channel c is
(31 y + 7 x + 53 c + 1) modulo 256.

### make_gray_image

`make_gray_image` returns a one-channel image holding the given bytes.

### make_plan

`make_plan` returns an `AffinePlan` with a given scale and matrix translation, built without a
draw. A case uses it to state the matrix it tests.

### make_labels

`make_labels` returns a label set with one object per box and the classes 0, 1, 2 and so on.

### expect_box_near

`expect_box_near` compares the four edges of two boxes within `K_PIXEL_TOLERANCE`.

### bytes_of

`bytes_of` returns the bytes of an image.

### ZeroRangesPutTheSourceCentreOnTheOutputCentre

With the scale and translate ranges at 0, a 1200 by 1000 source and a 480 by 640 output give
scale 1, translation (320, 240) and the matrix [1 0 -180; 0 1 -360]: the source centre
(500, 600) maps to the output centre.

### MatrixEqualsTranslationTimesScaleTimesCentre

The case replays the three draws on a copy of the generator and builds the centring, scaling
and translation matrices. The plan's matrix equals translation times scaling times centring.
The seed is required to give a scale more than 0.05 from 1, because at scale 1 the product is
the same in any order.

### PlanConsumesExactlyThreeDraws

After `plan_affine` the generator equals a copy advanced by three draws.

### PlanCarriesSizesBorderValueAndFilter

The plan holds the source size, the output size, the border value 37 and the filter fields of
the settings.

### DrawnParametersStayInRangeAndCentreOnTheirMeans

Over 20000 plans from 200 seeds, the scale lies in [0.5, 1.5) and each translation in [0.4,
0.6] of its output side. The means are 1 within 0.012 and 0.5 within 0.0025, the extremes come
within 0.01 and 0.002 of the range ends, and the x and y fractions are within 1e-6 of each
other in fewer than 1 percent of the plans, so each axis has its own draw.

### PlanRejectsSettingsAndSizesOutsideTheirRanges

A scale half-range of 1, -0.1 or NaN and a translate half-range of 0.6 or -0.1 throw with
"validate_affine" or the field name. A source of 0 by 16 and an output of 16 by 0 throw with
"plan_affine" and the size. No rejected call makes a draw, and the default settings pass.

### IdentityPlanLeavesTheImageUnchanged

With both ranges at 0 and equal source and output sizes, the warped image has the source's
bytes.

### IdentityPlanLeavesBoxesUnchanged

Under the same plan three boxes keep their classes and their edges.

### TranslationMovesPixelsAndFillsTheUncoveredBorder

A 3 by 4 source shifted by (2, 1) into a 5 by 6 output puts source pixel (y - 1, x - 2) at
(y, x) and fills every other pixel with 114.

### ScaleByTwoInterpolatesAndFadesIntoTheBorder

A 2 by 2 image scaled by 2 with border value 100 gives the sixteen bytes the case lists. Odd
output coordinates fall halfway between two source pixels, and the last row and column blend
with the border.

### HalfPixelBlendRoundsToNearest

A row (10, 13) shifted by half a pixel gives 62, 12 and 64: the means 62, 11.5 and 63.5 with
the border 114, rounded half up.

### FractionRoundsToTheNearestOf2048Weights

With an offset of 6141 / 8192 the fraction times 2048 is 512.75, so the weights are 1535 and
513. The three output bytes are 90, 19 and 42, computed from those weights; the weights 1536
and 512 give other bytes.

### OutputFarFromTheSourceIsAllBorder

A translation of (1000, -1000) leaves every output byte at the border value 9.

### ImageWarpRejectsAPlanItCannotApply

`apply_affine` on an image throws for a plan made for another source size, a matrix with an
off-diagonal term, a diagonal term of 0, a matrix scale that is not the plan's scale, an
infinite translation, a source without pixels, a source of five channels, the destination as
its own source and a view of two rows of the destination. The aliased image keeps its bytes.

### TranslationMovesBoxesByTheOffset

A translation of (30, -10) maps (10, 20, 50, 60) to (40, 10, 80, 50).

### ScaleMultipliesBoxCornersBeforeTheOffset

Scale 2 with translation (10, -20) maps (5, 20, 25, 40) to (20, 20, 60, 60).

### WarpedBoxIsTheHullOfItsFourWarpedCorners

The case maps the four corners of a box through the plan's matrix and takes their hull. The
warped box equals the hull and the literal (30, 40, 80, 90).

### BoxesAreClippedThenFilteredAndSurvivorsKeepTheirOrder

Five 40 by 40 boxes are shifted right by 80 in a 100 px wide output, which leaves widths of 10,
5, 3, 2 and 0 inside it. The first two remain, in order, clipped to x2 = 100. Width 3 fails the
area ratio at 0.075, width 2 fails the minimum side and width 0 fails both.

### FilterComparesAgainstTheSourceBoxTimesTheScale

At scale 0.25 a 40 by 40 box becomes 10 by 10 and is kept: its ratio is 1 against the scaled
box and 0.0625 against the unscaled one. At scale 2 a 20 by 20 box that keeps 3 by 40 inside
the output is removed: its ratio is 0.075 against the scaled box and 0.3 against the unscaled
one.

### BoxWarpUsesThePlansFilter

A 4 px wide box is kept with the default filter and removed when the plan's `min_side` is 4.

### BoxWarpRejectsARotatedMatrixAndAScaleThatIsNotThePlans

`apply_affine` on labels throws for an off-diagonal matrix term and for a matrix scale that is
not the plan's scale.

### FiltersDefaultToTheBoxAndThePolygonAreaRatios

The settings and a plan made from them have `filter.min_area_ratio` 0.10 and
`polygon_filter.min_area_ratio` 0.01, and a default `AffinePlan` has the second value too.

### PolygonVerticesAreScaledThenTranslated

Scale 2 with translation (10, -5) maps the vertices (5, 10), (25, 12) and (15, 30) to (20, 15),
(60, 19) and (40, 55), and the box to (20, 15, 60, 55).

### PolygonObjectTakesTheBoxOfItsVisiblePart

A triangle that reaches outside a 100 by 80 output gets the box (0, 0, 36.666664, 33), the
reference's `segment2box` result. The same bounds without a polygon are clipped to
(0, 0, 70, 60).

### WarpedPolygonIsClampedToItsNewBox

The same triangle's vertices become (0, 0), (36.666664, 0) and (0, 33): each is clamped to the
box rebuilt from the polygon.

### WarpedPolygonStaysInsideTheOutputWhenACrossingRoundsOutside

In float the edge from x = -0.03 to x = 0.25 crosses x = 0 at -1.86e-9. The box's x1 is exactly
0, and every vertex lies inside the output and between the box's x1 and x2.

### PolygonObjectIsFilteredAtTheLowerAreaRatio

A box that keeps an area ratio of 0.075 is kept when its object has a polygon and removed when
it has none.

### PolygonWarpUsesThePlansPolygonFilter

The same object is removed when `polygon_filter.min_area_ratio` is 0.08, with
`filter.min_area_ratio` set to 0. The polygon filter decides for an object with a polygon.

### PolygonOutsideTheOutputDropsItsObject

With a translation of 500, the object whose polygon leaves the output is removed and the one
the translation moves into it is kept, with class 1 and the box (40, 10, 80, 50).

## test_imaging_boxes.cpp

Tests the two box forms, their conversions and the candidate filter.

This file needs no device. `Unit_ImagingBoxes` covers `include/SushiAI/imaging/boxes.hpp`: a
`NormalizedBox` is a centre and an extent in fractions of the image, a `PixelBox` is two corners
in pixels, and `is_box_candidate` decides whether a box survives an augmentation.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px and `K_FRACTION_TOLERANCE` is 1e-6 of the image.
`K_EPSILON_TOLERANCE` is 1e-22, for the filter's default epsilon of 1e-16. `K_SIZE` is 480 rows
by 640 columns, the size the hand cases use.

### expect_pixel_box

`expect_pixel_box` compares the four corners of a `PixelBox` within `K_PIXEL_TOLERANCE`.

### expect_normalized_box

`expect_normalized_box` compares the four values of a `NormalizedBox` within
`K_FRACTION_TOLERANCE`.

### box_of_sides

`box_of_sides` returns a box with its left top corner at the origin and the given width and
height.

### ToPixelBoxPlacesCornersFromCentreAndExtent

The normalized box (0.5, 0.25, 0.5, 0.25) in `K_SIZE` is the pixel box (160, 60, 480, 180).

### ToPixelBoxUsesWidthForXAndHeightForY

A box covering a 100 by 200 image is (0, 0, 200, 100). The two sides differ, so a conversion
that swapped them gives other corners.

### ToPixelBoxDoesNotClip

A box centred on the left edge converts to (-50, 75, 50, 125). The conversion leaves the
negative corner as it is.

### ToNormalizedBoxDividesCentreAndExtentByTheImageSize

The pixel box (160, 60, 480, 180) in `K_SIZE` is the normalized box (0.5, 0.25, 0.5, 0.25).

### NormalizedToPixelAndBackReturnsTheBox

Four normalized boxes, one of them all zeros, come back from the two conversions within
`K_FRACTION_TOLERANCE`.

### PixelToNormalizedAndBackReturnsTheBox

The pixel box (12.5, 7.25, 630, 471.75) comes back within `K_PIXEL_TOLERANCE`.

### ScaleBoxMultipliesEachAxisByItsOwnFactor

`scale_box` by 2 across and 0.5 down maps (10, 20, 30, 40) to (20, 10, 60, 20).

### TranslateBoxAddsEachAxisItsOwnOffset

`translate_box` by 5 across and -25 down maps (10, 20, 30, 40) to (15, -5, 35, 15).

### ClipBoxClampsEachEdgeToTheImage

`clip_box` maps (-10, 20, 700, 500) in `K_SIZE` to (0, 20, 640, 480).

### ClipBoxLeavesABoxInsideTheImageAlone

The box (0, 0, 640, 480) is unchanged.

### ClipBoxCollapsesABoxOutsideTheImageOntoTheEdge

A box to the right of the image clips to (640, 10, 640, 20) and one above it to
(10, 0, 20, 0). Both have area 0.

### BoxAreaIsWidthTimesHeight

`box_area` of (1, 2, 4, 6) is 12.

### PolygonFilterLowersOnlyTheAreaRatioToOneHundredth

`make_polygon_filter` returns a `CandidateFilter` whose `min_area_ratio` is 0.01, which is
`K_POLYGON_MIN_AREA_RATIO`. Its `min_side`, `max_aspect_ratio` and `epsilon` equal those of a
default filter.

### PolygonFilterKeepsABoxThePlainFilterRejects

A box that goes from 100 by 100 to 50 by 10 keeps 5 percent of its area. The default filter
rejects it and the polygon filter keeps it.

### CandidateFilterDefaultsAreTheReferenceThresholds

A default `CandidateFilter` has `min_side` 2, `max_aspect_ratio` 100, `min_area_ratio` 0.10 and
`epsilon` 1e-16.

### CandidateNeedsAWidthAboveTheMinimumSide

A width of 2 is rejected and a width of 2.25 is kept. The comparison is strict.

### CandidateNeedsAHeightAboveTheMinimumSide

A height of 2 is rejected and a height of 2.25 is kept.

### CandidateNeedsMoreThanTheMinimumAreaRatio

Against a 100 by 100 box, a 10 by 100 box has a ratio of 0.10 in float and is rejected. A width
of 10.5 is kept and a width of 9.5 is rejected.

### CandidateAreaRatioFollowsTheFilterField

With `min_area_ratio` set to 0.01, the 10 by 100 box is kept and a 10 by 10 box, whose ratio is
0.01, is rejected.

### CandidateNeedsAnAspectRatioBelowTheMaximum

A 300 by 3 box and a 3 by 300 box have an aspect ratio of 100 in float and are rejected. Boxes
of 297 by 3 and 3 by 297 are kept. The filter takes the larger of the two ratios.

### CandidateMeasuresSidesNotPositions

A 50 by 40 box moved to another position is kept. The filter reads widths and heights and no
coordinate.

### CandidateEpsilonIsAddedToTheAreaBefore

With `epsilon` 100 and `min_area_ratio` 0.6, a 10 by 10 box against itself has the ratio
100 / (100 + 100), which is 0.5, and is rejected. Without the epsilon in the denominator the
ratio is 1.

### CandidateEpsilonIsAddedToTheHeightOfAWideBox

With `epsilon` 1, a 300 by 3 box has the ratio 300 / (3 + 1), which is 75, and is kept.

### CandidateEpsilonIsAddedToTheWidthOfATallBox

With `epsilon` 1, a 3 by 300 box has the ratio 75 and is kept.

### CandidateWithAnEmptyBeforeBoxIsKeptAndAnEmptyAfterBoxIsNot

An empty box before and a 10 by 10 box after is kept. A 10 by 10 box before and an empty box
after is rejected.

## test_imaging_flip.cpp

Tests the plan and the application of the horizontal flip to an image and its labels.

This file needs no device. `Unit_ImagingFlip` covers `include/SushiAI/imaging/flip.hpp`.
`plan_flip` makes the one random draw and records the decision in a `FlipPlan`; `apply_flip`
then mirrors an image or a label set from the plan and draws nothing.

### Constants

`K_BOX_TOLERANCE` is 1e-4 px. `K_SEED_COUNT` is 10000, the seeds 0 to 9999 a statistical case
draws from. `K_FREQUENCY_TOLERANCE` is 0.03, six standard deviations of a frequency of 0.5 over
10000 draws.

### make_counted_image

`make_counted_image` returns an image whose byte at offset i is i modulo 256.

### expect_same_image

`expect_same_image` expects two images to have the same size, channel count and bytes.

### expect_box

`expect_box` compares the four coordinates of two boxes within `K_BOX_TOLERANCE`.

### PlanRecordsTheSizeAndMakesExactlyOneDraw

The plan holds the source size 480 by 640. Its `is_flipped` equals one `bernoulli` draw of a
second generator with the same seed, and the two generators are equal afterwards.

### ProbabilityZeroNeverFlipsAndOneAlwaysFlips

Over 10000 seeds, probability 0 never flips and probability 1 always does, and both leave the
generator in the same state.

### DefaultProbabilityFlipsAboutHalfOfTheSeeds

With the default settings the flip frequency over 10000 seeds is 0.5 within
`K_FREQUENCY_TOLERANCE`.

### PlanRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5 and -0.1 throw with a message containing "validate_flip" and
"probability", and the generator is unchanged.

### PlanRejectsAProbabilityThatIsNotANumberBeforeAnyDraw

A NaN probability throws and the generator is unchanged.

### PlanRejectsASideThatIsNotPositiveBeforeAnyDraw

Sizes of 0 by 8 and 8 by -2 throw with a message containing "plan_flip", "source size" and the
size, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_flip` on a default `ImageView` throws with "apply_flip" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_flip` with the image as both source and destination throws with "destination holds the
source's pixels", and the image is unchanged.

### ViewInsideTheDestinationIsRejected

A view of three rows of the destination is refused with the same message, and the image is
unchanged.

### FlippedImageHoldsEachRowInReverseColumnOrder

For a 2 by 3 image of three channels, pixel x of each destination row equals pixel 2 - x of the
source row, channel by channel. The first pixel of row 0 is bytes 6, 7, 8: the pixels are
reversed and the channels inside a pixel are not.

### UnflippedImageIsCopiedUnchanged

A plan with `is_flipped` false copies the source.

### FlippingAnImageTwiceRestoresIt

A 4 by 7 image of three channels and a 3 by 4 image of one channel both come back after two
flips.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 3 source with a plan for 2 by 4 or for 5 by 3 throws with a message naming the plan's
size and "2x3".

### FlippedBoxesMirrorAboutTheWidthAndKeepTheirOrder

In a 640 px wide image the box (100, 50, 300, 200) becomes (340, 50, 540, 200), and a box
spanning the whole width is unchanged. Classes and order are kept, and x1 stays at or below x2.

### UnflippedBoxesAreReturnedUnchanged

An unflipped plan returns the labels as they are.

### FlippingBoxesTwiceRestoresThem

Two boxes come back after two flips within `K_BOX_TOLERANCE`.

### FlippedPolygonMirrorsEveryVertexAboutTheWidth

In a 640 px wide image the vertices at x = 100, 300 and 150.5 move to 540, 340 and 489.5, and
every y is kept. An object without a polygon still has none.

### UnflippedPolygonIsReturnedUnchanged

An unflipped plan returns the polygon's vertices as they are.

### ValidateAcceptsZeroAndOne

`validate_flip` does not throw for probability 0, 1 or the default.

## test_imaging_gray.cpp

Tests the expansion of a one-channel image to three equal channels.

This file needs no device. `Unit_ImagingGray` covers `expand_gray` of
`include/SushiAI/imaging/gray.hpp`.

### make_counted

`make_counted` returns an image whose byte at offset i is 17 * i + 3 modulo 256.

### ExpandCopiesEachLevelIntoThreeChannels

A 3 by 5 grey image expands to a 3 by 5 image of `K_RGB_CHANNELS` channels, and all three
channels of every pixel hold the pixel's grey level.

### ExpandRewritesAReusedDestination

A destination that was 9 by 9 with four channels and filled with 255 becomes 2 by 2 with three
channels, and every byte holds its pixel's grey level.

### ExpandRejectsASourceThatIsNotGrey

Sources of three and of four channels throw with "expand_gray" and the channel count.
`K_GRAY_CHANNELS` is 1.

### ExpandRejectsASourceWithoutPixelsAndOneInTheDestination

An empty view throws with "no pixels". The destination's own view, and a view of two of its
rows, throw with "destination holds the source's pixels". The image keeps its size, channel
count and bytes.

## test_imaging_hsv.cpp

Tests the hue, saturation and value jitter: its plan, its tables and both colour conversions.

This file needs no device. `Unit_ImagingHsv` covers `include/SushiAI/imaging/hsv.hpp`.
`plan_hsv` draws three gains, `make_hsv_tables` turns a plan into three 256-entry tables, and
`apply_hsv` converts each pixel with `to_hsv`, looks it up and converts back with `to_rgb`. Hue
runs over `K_HUE_STEPS` (180) steps and every level is an integer, so the expected colours are
exact and a comment in the case derives them.

### Constants

`K_SEED_COUNT` is 10000, the seeds a statistical case draws from. `K_ROUND_TRIP_LEVELS` is 4,
the largest level error of `to_hsv` followed by `to_rgb`. `K_SWEEP_STEP` is 5, the step between
the levels of the colour sweep.

### make_row and pixel_at

`make_row` returns a one-row RGB image of the given pixels, and `pixel_at` returns the pixel at
a column of such an image.

### jitter

`jitter` applies a plan to a single pixel through `apply_hsv` and returns the result.

### expect_rgb and expect_hsv

`expect_rgb` and `expect_hsv` compare the three levels of a pixel for equality.

### level_distance

`level_distance` returns the absolute difference of two levels.

### PlanDrawsHueThenSaturationThenValueAndNothingMore

The plan's three gains equal three `uniform(-1, 1)` draws of a second generator, each times its
setting, in the order hue, saturation, value. The comparison is exact, and the two generators
are equal afterwards.

### DrawnGainsStayInsideTheConfiguredRangesOverManySeeds

Over 10000 seeds every drawn gain lies in [-gain, gain). The smallest of each is below -0.9
times its gain and the largest above 0.9 times it, so the draws cover the range.

### ZeroGainsDrawThreeTimesAndGiveAZeroPlan

With three gains of 0 the plan is three zeros and the generator has advanced by three draws.
The draw count does not depend on the settings.

### PlanRejectsANegativeGain

A saturation gain of -0.5 throws with "validate_hsv" and "gain", and the generator is
unchanged.

### PlanRejectsAGainThatIsNotFiniteBeforeAnyDraw

An infinite hue gain and a NaN value gain throw, and the generator is unchanged.

### SourceWithoutPixelsIsRejected

`apply_hsv` on a default view throws with "apply_hsv" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_hsv` with the image as source and destination throws with "destination holds the
source's pixels", and both pixels are unchanged.

### ViewInsideTheDestinationIsRejected

A view of the destination's second pixel is refused with the same message, and the image keeps
its size and pixels.

### ZeroPlanGivesIdentityTables

For a zero plan the saturation and value tables map every level to itself. The hue table maps
x to x below 180 and to x - 180 from 180 up.

### HueTableShiftsByAFractionOfATurnAndWrapsBothWays

A hue gain of 0.25 shifts by 45 steps: 0 maps to 45, 134 to 179 and 135 to 0. A gain of -0.25
maps 0 to 135 and 45 to 0. Every entry of both tables is below 180.

### SaturationAndValueTablesScaleTruncateAndClamp

A saturation gain of 0.5 multiplies by 1.5: 3 maps to 4, which is 4.5 truncated, and 170 and
above map to 255. A value gain of 0.25 multiplies by 1.25: 203 maps to 253 and 204 to 255. A
saturation gain of -0.5 maps 255 to 127, and a value gain of -2 maps every level to 0.

### PrimaryColoursHaveHuesSixtyStepsApart

`to_hsv` gives red, green and blue the hues 0, 60 and 120 at saturation 255 and value 255.

### GreyHasHueZeroAndBlackHasSaturationZero

Black, a mid grey and white have hue 0 and saturation 0, and their value is their level.

### MixedColourConvertsToTheHandComputedLevels

(100, 50, 50) converts to hue 0, saturation 128, value 100. (200, 100, 50) converts to hue 10,
saturation 191, value 200. The comments give the integer arithmetic.

### EachSectorConvertsBackToTheHandComputedColour

`to_rgb` is checked at eight HSV values, one or more in each of the six hue sectors, among them
hue 178 in the last sector and a colour that is not fully saturated.

### PureHueShiftOnRedGivesTheHandComputedColours

Red with a hue gain of 0.25 becomes (128, 255, 0), with -0.25 it becomes (128, 0, 255) and with
0.5 it becomes (0, 255, 255).

### ValueGainClampsAtFullAndAtZero

(100, 50, 50) with a value gain of 2 becomes (255, 127, 127), and with -2 it becomes black.

### SaturationGainClampsAtFullAndAtZero

(100, 50, 50) with a saturation gain of 2 becomes (100, 0, 0), and with -2 it becomes the grey
(100, 100, 100).

### WhiteStaysWhiteUnderASaturationGain

White with a hue gain of 0.3 and a saturation gain of 0.7 stays white. Its saturation is 0, and
0 times any factor is 0.

### ZeroPlanIsTheIdentityOnGreysPrimariesAndSecondaries

A zero plan returns nine colours unchanged: three greys, the three primaries and the three
secondaries.

### ZeroPlanCopiesColoursTheRoundTripWouldChange

`to_rgb(to_hsv(.))` changes the blue level of (12, 200, 99) to 100 and of (250, 3, 180) to 176,
which the case asserts first. A zero plan returns both colours unchanged, so `apply_hsv` copies
the image for an identity plan and does not send it through the conversion.

### PlanOfZeroGainsCopiesTheImage

Over 64 seeds a plan drawn from three zero gains satisfies `is_hsv_identity`, and the same two
colours come back unchanged.

### PlanWithOneGainAwayFromZeroIsNotAnIdentity

`is_hsv_identity` is true for a zero plan and for one with negative zeros, and false when any
one gain is 1e-6 or -1e-6. A value gain of 1e-6 changes no table entry, and (12, 200, 99) still
comes back as (12, 200, 100): the image went through the conversion.

### ValidateAcceptsZeroGainsAndTheDefaults

`validate_hsv` does not throw for three zero gains or for the defaults.

### RoundTripMovesNoLevelByMoreThanItsBound

Over a sweep of every fifth level on each channel, `to_hsv` gives a hue below 180 and `to_rgb`
of it differs from the original by at most `K_ROUND_TRIP_LEVELS` on any channel.

### ImageWithoutThreeChannelsIsCopiedUnchanged

A one-channel image with a plan that is not the identity comes back with the same size, one
channel and the same bytes.

### SamePlanGivesTheSameImageTwice

Two applications of one plan to one image give the same bytes.

## test_imaging_image.cpp

Tests the owned image, the view over foreign pixels and the checks both share.

This file needs no device. `Unit_ImagingImage` covers `include/SushiAI/imaging/image.hpp`:
`ImageSize`, `ImageView`, `Image`, `overlaps` and `check_source`. Pixels are bytes in
height, width, channel order.

Eleven `static_assert` lines at file scope hold the parts that are constant expressions:
equality of `ImageSize`, `is_valid_side`, `is_valid_size` and `is_valid_channel_count` at both
ends of their ranges and one past each, and `row_bytes()` and `byte_count()` of a view.

### write_offsets

`write_offsets` writes each byte's own offset, modulo 251, into an image.

### ImageSizeComparesHeightAndWidth

Two sizes are equal when both sides are, and 3 by 4 differs from 4 by 3.

### ADefaultImageHasNoPixels

A default `Image` has height, width, channel count, row bytes, byte count and allocated bytes
of 0.

### AConstructedImageReportsItsExtents

An image of 4 by 6 with three channels reports those extents, 18 row bytes and 72 bytes, and
its data pointer is not null.

### AConstructedImageIsZero

Every byte of a constructed image is 0.

### TheLargestSideAndChannelCountAreAccepted

An image of `K_MAX_IMAGE_SIDE` by 1 and one of 1 by `K_MAX_IMAGE_SIDE` with `K_MAX_CHANNELS`
channels are constructed and report their byte counts.

### ConstructionRejectsAHeightOutsideItsBounds

Heights of 0, -2 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAWidthOutsideItsBounds

Widths of 0, -7 and 16385 throw with a message containing "Image:" and the size.

### ConstructionRejectsAChannelCountOutsideItsBounds

Channel counts of 0, -1 and 5 throw with a message containing "Image:" and the count; the
first also contains "[1, 4]".

### RowsStartOneRowOfBytesApart

`row(y)` of the image, of a const reference to it and of its view all equal the data pointer
plus 15 * y for a 5 px wide image of three channels.

### AByteWrittenThroughARowIsReadAtItsHwcOffset

A byte written at row 2, column 1, channel 2 of a 3 by 4 image is byte 29 of the data, and
every other byte is 0.

### FillSetsEveryByte

`fill(114)` sets every byte to 114.

### FillOnAnEmptyImageDoesNothing

`fill` on a default image leaves the byte count at 0.

### ResetGivesTheImageNewExtents

`reset` to 5 by 3 with four channels gives 12 row bytes and 60 bytes, and every byte of the new
extent can be written and read back.

### ResetOnADefaultImageAllocates

`reset` on a default image gives a data pointer that is not null and 18 bytes.

### ResetKeepsTheAllocationWhenItIsLargeEnough

An 8 by 8 image of three channels keeps its data pointer through a reset to 6 bytes and through
one to 192 bytes, the count it was constructed with.

### ResetRejectsTheBoundsTheConstructorRejects

`reset` throws for a height of 0, a width of 16385 and five channels, with a message
containing "Image::reset".

### ARejectedResetLeavesTheImageAsItWas

After a `reset` that throws, the image has its size, channel count, data pointer and bytes.

### AViewOfAnImageDescribesTheSamePixels

`view()` has the image's data pointer, size, channel count, row bytes and byte count.

### ADefaultViewIsEmpty

A default `ImageView` has a null pointer, size 0 by 0 and no bytes.

### AViewOverForeignBytesAddressesThemInHwcOrder

A view over a twelve-byte array as 2 by 3 with two channels has 6 row bytes, and row 1,
column 2, channel 1 is byte 11.

### ACopyOwnsItsOwnPixels

A copy has another data pointer, and filling the copy leaves the original's bytes as they
were.

### AllocatedBytesKeepsTheLargestSizeTheImageHad

After a reset from 48 bytes to 4, `byte_count()` is 4 and `allocated_bytes()` is 48.

### OverlapsFindsTheWholeImageAndAViewInsideIt

`overlaps` is true for the image's own view, for a view of two of its rows and for a view of
its last byte.

### OverlapsIsFalseForBytesOfAnotherContainer

A view over an array the image does not own does not overlap it.

### OverlapsCountsTheAllocationBeyondTheCurrentSize

A view of the last row of a 4 by 4 image still overlaps it after the image is reset to one
byte. The test is against the allocation and not the current size.

### OverlapsIsFalseForSeparateOrEmptyOperands

`overlaps` is false for two separate images, for a default view, for an unallocated image, and
for views with a height of 0 or no channels.

### CheckSourceAcceptsAViewOfAnotherImage

`check_source` does not throw for a view of another image, with an allocated or an unallocated
destination.

### CheckSourceRejectsEachFaultUnderTheCallersName

`check_source` throws with the caller's name as prefix for a view without pixels, a height of
0, a width of 16385, five channels, the destination's own view and a view of two of its rows.

## test_imaging_label_set.cpp

Tests the label set that holds classes, boxes and polygons in step, and its conversions.

This file needs no device. `Unit_ImagingLabelSet` covers
`include/SushiAI/imaging/label_set.hpp`. `Labels` holds pixel boxes and `FileLabels` holds
normalized boxes; both keep one class, one box and one polygon per object, and a polygon may be
empty.

### Tolerances and K_SIZE

`K_PIXEL_TOLERANCE` is 1e-3 px, `K_FRACTION_TOLERANCE` is 1e-6 of the image, and `K_SIZE` is
480 rows by 640 columns.

### expect_pixel_box, expect_normalized_box and expect_point

The three helpers compare a `PixelBox`, a `NormalizedBox` and a `Point` with expected values.
`expect_point` takes its tolerance as an argument.

### make_diagonal

`make_diagonal` returns a polygon of a given number of vertices spread along the diagonal from
(0, 0) towards (1, 1).

### DefaultSetHoldsNoObject

A default `Labels` is empty, has size 0 and has no class, box or polygon.

### ConstructorsRejectListsOfDifferentLengths

Two classes with one box, one class with two boxes, two objects with one polygon, and no class
with one box all throw with a message containing "LabelSet" and the two counts.

### SetWithoutPolygonsHoldsOneEmptyPolygonPerObject

A set built from two classes and two boxes has two polygons, both empty.

### AddAppendsOneEntryToEachList

`add` with and without a polygon grows the classes, the boxes and the polygons together. The
object added without one has an empty polygon.

### KeepIfRemovesAnObjectFromAllThreeLists

`keep_if` with a predicate that keeps even classes leaves classes 2 and 4 with their own boxes
and polygons.

### ToPixelLabelsKeepsClassesAndOrder

`to_pixel_labels` converts three normalized boxes in `K_SIZE` and keeps the classes 7, 2, 7 in
order.

### ToPixelLabelsScalesVerticesByWidthAndHeight

A vertex at (0.25, 0.25) becomes (160, 120): x is multiplied by the width and y by the height.
An empty polygon stays empty.

### ToNormalizedLabelsKeepsClassesAndOrder

`to_normalized_labels` converts two pixel boxes and the vertices of one polygon, and keeps
classes and order.

### LabelConversionsOfABackgroundImageAreEmpty

Both conversions of an empty set are empty.

### ClipLabelsClipsBoxesAndVerticesAndRemovesNone

`clip_labels` clips each box to the image and clamps each vertex onto it. A box wholly outside
collapses to zero width and stays in the set.

### RemoveEmptyBoxesDropsZeroAreaAndKeepsOrderClassesAndPolygons

Of five objects, the two with zero width or zero height are removed. The classes 10, 12 and 14
remain in order with their boxes and polygons.

### RemoveEmptyBoxesDropsABoxOfNegativeArea

A box whose x1 is above its x2 is removed.

### ResamplePolygonsGivesEachPolygonAThousandPoints

`resample_polygons` gives a triangle and a polygon of 40 vertices `K_POLYGON_POINTS` points
each. An empty polygon stays empty, and classes and boxes are unchanged.

### ResamplePolygonsUsesTheLongestPolygonPlusOneAboveAThousand

With one polygon of 1200 vertices in the set, both polygons are resampled to 1201 points.

### ResamplePolygonsLeavesASetWithoutPolygonsAlone

A set whose one polygon is empty comes back with the polygon empty and the box unchanged.

## test_imaging_labels.cpp

Tests the parser of a label file's text.

This file needs no device and reads no file. `Unit_ImagingLabels` passes strings to
`parse_label_text` of `include/SushiAI/imaging/labels.hpp`. A row is a class and four
normalized box values, or a class and the x, y pairs of a polygon. The refusal cases check that
the message names the line and quotes the field.

### K_CLASS_COUNT and K_FRACTION_TOLERANCE

`K_CLASS_COUNT` is 80, the class count the cases parse with. `K_FRACTION_TOLERANCE` is 1e-7,
the tolerance of a parsed value.

### expect_box

`expect_box` compares the four values of a `NormalizedBox` within `K_FRACTION_TOLERANCE`.

### ParsesBoxRowsInFileOrder

Two rows give two objects, with classes 0 and 3 and the four values of each row.

### EmptyTextIsABackgroundImage

An empty string gives an empty set.

### TextOfBlankLinesIsABackgroundImage

Text of newlines, spaces, a tab and carriage returns gives an empty set.

### AcceptsTabsRepeatedSpacesAndBlankLinesBetweenRows

Rows with leading spaces, tabs, repeated spaces, a blank line between them and no final newline
parse to two objects.

### AcceptsCarriageReturnLineEndings

Rows ended by "\r\n" and by a lone "\r" parse to three objects.

### AcceptsVerticalTabAndFormFeedLineEndings

A vertical tab and a form feed each end a row.

### AcceptsTheSeparatorBytesAsLineEndings

The bytes 0x1c, 0x1d and 0x1e each end a row: four rows separated by them parse to four
objects.

### AcceptsTheUnitSeparatorByteBetweenFields

The byte 0x1f separates fields, alone or repeated.

### AcceptsExponentsBareDecimalPointsAndAPlusSign

"+7.0 5e-1 .5 2.5E-1 +0.125" parses to class 7 and the box (0.5, 0.5, 0.25, 0.125).

### RejectsARowOfFourColumns

A row of four columns throws with "parse_label_text: line 1", "5 columns" and "this row has 4".

### RejectsARowOfSixColumns

A row of six columns throws with "5 columns" and "this row has 6".

### RejectsARowOfOneColumn

A row of one column throws with "this row has 1".

### NamesTheLineOfTheRejectedRow

A short row after one row and one blank line is reported as "line 3". Blank lines count.

### RejectsAFieldThatIsNotANumber

The field "abc" throws with "'abc'" and "not a number".

### RejectsAFieldWithCharactersAfterTheNumber

The field "0.5x" throws with "'0.5x'".

### RejectsADecimalComma

The field "0,5" throws with "'0,5'".

### RejectsADoubledSign

The field "+-0.5" throws with "'+-0.5'".

### RejectsNotANumberAndInfinity

The fields "nan", "inf" and "1e999" each throw with the field quoted.

### RejectsAFiniteValueAboveTheFloatRange

The field "1e39", finite as a double and not as a float, throws with "not a finite number".

### RejectsANonzeroValueBelowTheSmallestNormalDouble

The fields "1e-310" and "1e-999" throw with the field quoted.

### AcceptsZeroAndTheSmallestNormalDouble

"0.0" and "2.2250738585072014e-308" are accepted and both store 0 within the tolerance.

### RejectsACoordinateBelowTheTolerance

Values of -0.02 and -0.5 throw with "negative".

### RejectsACoordinateAboveTheTolerance

Values of 1.02 and 640 throw with "above". The second is a pixel value in a file that must hold
fractions.

### StoresACoordinateInsideTheToleranceUnclipped

Values of 1.005 and -0.005 are stored as they are.

### AcceptsACoordinateOnEitherToleranceBound

Values of 1.01 and -0.01 are accepted and stored as they are.

### RejectsANegativeClass

Class -1 throws with "class" and "negative".

### RejectsAClassWithAFraction

Class 1.5 throws with "class" and "whole number".

### RejectsAClassAtOrAboveTheClassCount

Class 80 of 80 throws with "out of range" and "80 classes", and class 1 of 1 with "1 classes".

### AcceptsTheLastClass

Class 79 of 80 is accepted.

### RemovesRepeatedRowsKeepingTheFirstAndTheFileOrder

Five rows of which two repeat earlier ones give three objects, with classes 5, 2, 9 in file
order.

### RemovesARowThatRepeatsAnotherInADifferentSpelling

"5.0 5e-1 0.50 .25 0.250" repeats "5 0.5 0.5 0.25 0.25" and is removed. Rows are compared by
value and not by text.

### KeepsRowsThatDifferInOneValue

Three rows that differ in the class or in one box value are all kept.

### ReducesAPolygonRowToTheBoxAroundItsPoints

A row of three points gives the box (0.5, 0.5, 0.5, 0.5) around them.

### KeepsThePointsOfAPolygonRowInFileOrder

A row of five points gives a polygon of those five points in file order.

### GivesABoxRowNoPolygon

Each object of two box rows has an empty polygon.

### ReducesAPolygonOfManyPointsToItsBounds

The five points of the earlier case give the box (0.5, 0.375, 0.75, 0.5).

### RejectsABoxRowInAPolygonFile

A box row after a polygon row throws with "line 2" and "polygon".

### RejectsAPolygonRowWithAHalfPoint

A polygon row of eight columns throws with "polygon" and "this row has 8".

### RejectsAPolygonRowOfFewerThanThreePoints

A row of three columns in a polygon file throws with "line 2", "three points" and "this row
has 3".

### RejectsAPolygonWhoseBoxLeavesTheImage

A polygon with a vertex at x = 1.5 throws with "above".

### RemovesARepeatedPolygonAndKeepsOneWithTheSameBox

Of three polygon rows the second repeats the first and is removed. The third has other points
and the same box, and is kept: both remaining boxes are (0.5, 0.5, 0.5, 0.5) and their second
vertices differ.

## test_imaging_letterbox.cpp

Tests the plan and the application of the letterbox to an image and its labels.

This file needs no device. `Unit_ImagingLetterbox` covers
`include/SushiAI/imaging/letterbox.hpp`. `plan_letterbox` computes the ratio, the resized size
and the four paddings, and draws no random number. `apply_letterbox` resizes with
`resize_bilinear` of `include/SushiAI/imaging/resize.hpp`, pads, and moves boxes and polygons
by the same plan.

### K_BOX_TOLERANCE

`K_BOX_TOLERANCE` is 1e-4 px.

### make_counted_image and make_flat_image

`make_counted_image` returns an image whose byte at offset i is a first value plus i, modulo
256. `make_flat_image` returns an image with every byte at one level.

### is_content

`is_content` reports whether a pixel of the target lies inside the plan's resized image.

### expect_two_levels

`expect_two_levels` expects every content byte of an image to be one level and every padding
byte another.

### WideImageIsPaddedAboveAndBelowWithTheOddRowBelow

A 50 by 200 source in a 100 by 100 target has ratio 0.5 and resizes to 25 by 100. The 75 rows
of padding split into 37 above and 38 below, and the plan records the source and target sizes.

### TallImageIsPaddedLeftAndRightWithTheOddColumnRight

A 200 by 50 source resizes to 100 by 25 with 37 columns left and 38 right.

### ExactFitHasRatioOneAndNoPadding

A 64 by 64 source in a 64 by 64 target has ratio 1 and four paddings of 0.

### SmallImageIsOnlyPaddedUnlessScaleUpIsSet

A 20 by 40 source in an 80 by 80 target keeps its size by default, with paddings 30, 30, 20,
20. With `scale_up` set the ratio is 2, the resized size is 40 by 80 and the paddings are 20
above and below.

### ResizedSideRoundsAHalfToTheEvenInteger

At ratio 0.5 a height of 5 resizes to 2 and a height of 7 to 4: 2.5 and 3.5 round to the even
integer.

### RectangularStrideTargetPadsOnlyTheShortAxis

A 480 by 640 source in a 512 by 640 target has ratio 1 and 16 rows above and below.

### PlanCopiesThePaddingValueOfTheSettings

The plan's `padding_value` is that of the settings, at the default and at 7.

### PlanRejectsASideThatIsNotPositive

A source of 0 by 8 and a target of 8 by -1 throw with "plan_letterbox" and the size.

### PlanRejectsASideAboveTheImageBound

A source height and a target width of 16385 throw with the size and "16384".

### ValidateAcceptsEverySetting

`validate_letterbox` does not throw with `scale_up` set and a padding value of 255, or for the
defaults.

### PlanRejectsAResizedSideOfZeroPixels

A 1 by 1000 source in an 8 by 8 target resizes to a height of 0, and the plan throws with "zero
pixels".

### ImageWithoutResizeKeepsItsPixelsInsideThePadding

A 2 by 4 source in a 4 by 6 target is placed at row 1, column 1. Every content byte equals the
source byte and every other byte is the padding value.

### ExactFitCopiesEveryPixel

A plan without resize or padding copies the source.

### WideImageIsResizedThenPaddedWithTheFillValue

A flat 50 by 200 image at level 200 gives a 100 by 100 result that is 200 inside the content
rectangle and 114, the default padding value, outside it.

### TallImageIsResizedThenPaddedWithACustomFillValue

A flat one-channel 200 by 50 image at level 33 with a padding value of 7 gives 33 inside and 7
outside.

### ResizedContentEqualsTheBilinearResizeOfTheSource

For a counted 50 by 200 image, every content byte equals the byte `resize_bilinear` gives for
the plan's resized size, and every padding byte is 114. The case also expects the content not
to be flat, since a flat rectangle cannot tell a resize from a fill.

### SourceWithoutPixelsIsRejected

`apply_letterbox` on a default view throws with "apply_letterbox" and "no pixels".

### DestinationThatIsTheSourceIsRejected

`apply_letterbox` with the image as source and destination throws with "destination holds the
source's pixels", and the image keeps its size.

### ViewInsideTheDestinationIsRejected

A view of two rows of the destination is refused with the same message, and the image keeps its
size and its bytes.

### ResizedContentBesideSidePaddingEqualsTheBilinearResize

A counted 200 by 50 image in a 100 by 101 target has 38 columns of padding on each side. The
destination starts as a 120 by 120 image filled with 255. After the call no byte differs from
the bilinear resize placed between two bands of 114.

### DestinationIsFullyRewrittenWhenItIsReused

A destination that was 9 by 9 and filled with 255 becomes 4 by 6 with content 50 and padding
114.

### ImageOfAnotherSizeThanThePlanIsRejected

A 2 by 4 source with a plan made for 4 by 4 throws with "plan was made for".

### PlanWhosePaddingsDoNotTileTheTargetIsRejected

A plan whose right padding is raised by 1 throws with "tile".

### BoxesMoveByTheRatioThenTheLeftAndTopPadding

In the tall plan a box maps by x * 0.5 + 37 and y * 0.5, and in the wide plan by x * 0.5 and
y * 0.5 + 37. Classes and order are kept.

### BoxesOutsideTheImageAreNeitherClippedNorRemoved

A box that extends past the source and a box of zero size are both moved by the plan and kept.

### ExactFitLeavesBoxesWhereTheyWere

With ratio 1 and no padding a box keeps its four coordinates.

### PolygonMovesByTheRatioThenTheLeftAndTopPadding

In the tall plan the vertices (10, 20), (30, 40) and (-4, 210) move to (42, 10), (52, 20) and
(35, 105). An object without a polygon still has none.

## test_imaging_mosaic.cpp

Tests the draw, the plan and the application of the four-image mosaic.

This file needs no device. `Unit_ImagingMosaic` covers `include/SushiAI/imaging/mosaic.hpp`.
`draw_mosaic` asks a `PartnerSource` for three partners, draws the gate and then the centre.
`plan_mosaic` turns a centre and four patch sizes into four placements on a canvas of twice the
image size. `apply_mosaic` copies the patches onto the canvas, or shifts, joins, clips and
prunes their labels. The two placement tables are lines 525 to 538 of the reference's
`data/augment.py`, worked by hand.

### K_SIDE and K_BOX_TOLERANCE

`K_SIDE` is 8, the image size of the hand cases, so the canvas is 16 by 16. `K_BOX_TOLERANCE`
is 1e-4 px.

### PatchSizes and Placements

`PatchSizes` holds four `ImageSize` values and `Placements` holds four `MosaicPlacement`
values, patch 0 first.

### The low centre

`K_LOW_CENTER_X` and `K_LOW_CENTER_Y` are 5 and 11, a centre that cuts patch 0 on the left and
patches 2 and 3 at the bottom. `K_LOW_SIZES` are the patch sizes used with it; patch 1 is
shorter and patch 2 narrower than the others. `K_LOW_PLACEMENTS` are the four expected
placements.

### The high centre

`K_HIGH_CENTER_X` and `K_HIGH_CENTER_Y` are 10 and 6, a centre that cuts patches 0 and 1 at the
top and patches 1 and 3 on the right. `K_HIGH_SIZES` are four 8 by 8 patches and
`K_HIGH_PLACEMENTS` are the four expected placements.

### make_draw

`make_draw` returns an enabled `MosaicDraw` with a given centre and the partners 0, 0, 0.

### make_constant and make_textured

`make_constant` returns a three-channel image of one colour. `make_textured` returns one whose
every byte depends on its row, column, channel and a salt, so a copy from the wrong position
shows.

### view_all

`view_all` returns the views of four images in patch order.

### count_wrong_bytes

`count_wrong_bytes` counts the canvas bytes that differ from what four placements name: the
patch byte at the placement's offset inside a placement, and the fill value elsewhere.

### expect_placements

`expect_placements` compares all ten fields of a plan's four placements with the expected ones.

### expect_box and expect_point

`expect_box` and `expect_point` compare a box and a vertex within `K_BOX_TOLERANCE`.

### RecordingPartners

`RecordingPartners` is a `PartnerSource` that returns the partners 4, 5 and 6, makes no draw
and records every index it is asked for.

### pixel_at

`pixel_at` returns the three bytes of one canvas pixel.

### SettingsDefaultToTheReferenceProbabilityAndFill

Default `MosaicSettings` have probability 1 and fill value 114, and `K_MOSAIC_PATCHES` is 4.

### DrawMakesPartnersThenGateThenRowThenColumn

With a `DatasetPartners` of 1000 and image size 640, the draw equals a replay on a copied
generator: three `index(1000)` draws, one `bernoulli(1)`, then `uniform(320, 960)` truncated
for the centre row and again for the centre column. The generators are equal afterwards, so
the call makes six draws.

### DisabledDrawMakesFourDrawsAndLeavesTheFieldsZero

With probability 0 the draw is disabled, its partners and centre are 0, and the generator has
advanced by four draws: the three partners and the gate.

### CentreStaysInTheReferenceRangeOverManySeeds

Over 5000 seeds at image size 640 both centre coordinates lie in 320 to 959. Their mean is
639.5 within 10, the smallest is at most 330 and the largest at least 949.

### CentreOfAnOddSizeUsesTheFlooredBorder

At image size 5 the centre lies in 3 to 6 and all four values occur over 400 seeds. The
reference computes the border as -5 // 2, which is -3, so the range is `int(uniform(3, 7))`.

### PartnersCoverThePoolAndNeverLeaveIt

Over 300 seeds and a pool of 7, every partner is below 7 and every index occurs.

### GateOpensWithItsProbability

At probability 0.25 the draw is enabled for 0.25 of 4000 seeds, within 0.04.

### PlanCutsPatchesAtTheLeftAndBottomEdges

`plan_mosaic` for the low centre gives `K_LOW_PLACEMENTS` and records the image size.

### PlanCutsPatchesAtTheTopAndRightEdges

`plan_mosaic` for the high centre gives `K_HIGH_PLACEMENTS`.

### PlanCarriesTheFillValueOfTheSettings

The plan's `fill_value` is 9 when the settings say 9 and 114 by default.

### CanvasIsTwiceTheSizeAndFilledWhereNoPatchLands

The canvas is 16 by 16 with three channels. Its two outer corners, a pixel in the row above the
shorter patch 1 and a pixel in the column left of the narrower patch 2 hold the fill value.

### EachColourLandsInItsQuadrantAroundTheCentre

Four patches of one colour each are placed around the low centre. The four pixels that meet at
the centre are red, green, blue and yellow, in patch order, and `count_wrong_bytes` is 0.

### CutPatchesKeepThePixelsNearestTheCentre

Four textured patches around the high centre with fill value 3 give a canvas with no wrong
byte. The texture makes the test fail for a patch copied from its far corner.

### ApplyOverwritesWhatTheDestinationHeld

A destination that was 40 by 40 and filled with 200 becomes the 16 by 16 canvas with no wrong
byte.

### LabelsAreShiftedJoinedInPatchOrderClippedAndPruned

Five boxes in patches 0, 1 and 3 are shifted by their placements' offsets and joined in patch
order. The second box of patch 0 lies wholly left of the canvas, has no area after the clip and
is removed. The four that remain have the classes 3, 2, 7, 5 and the boxes the case lists.

### LabelsOfFourBackgroundPatchesAreEmpty

Four empty label sets join to an empty one.

### DrawRejectsANonPositiveSizeBeforeAnyDraw

Image sizes of 0 and -4 throw with "validate_mosaic_size" and the value, and the generator is
unchanged.

### PlanRejectsADisabledDrawAndAnEmptyPatch

`plan_mosaic` throws for a disabled draw ("not enabled"), for a patch of height 0 ("patch 2")
and for a centre column of 17 on a 16 px canvas ("centre").

### ApplyRejectsPatchesThePlanWasNotMadeFor

`apply_mosaic` throws for a one-channel patch among three-channel ones ("patch 1", "channels")
and for a patch of another size than the plan's ("patch 3").

### DrawnCentreIsWhereTheFourColoursMeet

Over 32 seeds the case draws, plans and applies a mosaic of four 8 by 8 patches of one colour
each. The centre lies in 4 to 11, and the four pixels around it are red, green, blue and
yellow. The draw and the plan therefore agree on which coordinate is the row.

### DrawRejectsAProbabilityOutsideZeroToOne

Probabilities of 1.5, -0.25 and NaN throw with "validate_mosaic" and "probability", the
default settings pass `validate_mosaic`, and the generator is unchanged.

### DrawRejectsASizeWhoseCanvasIsTooLarge

An image size of 8193, whose canvas would exceed `K_MAX_IMAGE_SIDE`, throws with "image_size of
8193" and "[1, 8192]". An image size of 8192 is accepted.

### DrawAsksThePartnerSourceForEverySample

With a `RecordingPartners`, one draw with the gate open and one with it closed record the
indices 17 and 23. The open draw holds the partners 4, 5, 6 and the closed one 0, 0, 0.

### PolygonsAreShiftedClampedAndPrunedWithTheirObjects

Polygons move with their patch, by (-3, 3) for patch 0 and (5, 11) for patch 3, and are clamped
to the 16 by 16 canvas. The object whose box leaves the canvas is removed with its polygon, and
the object without a polygon still has none.

### PlanRejectsASizeOutOfRangeAndAPatchSideAboveTheLimit

`plan_mosaic` throws for an image size of 0 and of 8193 ("image_size") and for a patch height
of 16385 ("patch 1").

### ApplyRejectsAPlanWithoutAValidSizeForPixelsAndForLabels

Both forms of `apply_mosaic` throw with "image_size" for a default plan and for one whose image
size is 8193.

### ApplyRejectsAPatchWithoutPixels

A default view as patch 2 throws with "apply_mosaic: patch 2" and "no pixels".

### ApplyRejectsAPatchThatLiesAnywhereInTheDestination

A patch that views the start of the destination, and one that views its row 20, throw with
"destination holds the source's pixels" under the patch's number. The destination keeps its
40 by 40 size and its bytes.

### ApplyRejectsAPlacementOutsideTheCanvasAndOneThatResamples

A placement whose `canvas_x2` is 17 throws with "patch 1" and "outside a canvas". A placement
whose patch span is one pixel narrower than its canvas span throws with "patch 3" and
"resamples nothing".

## test_imaging_partners.cpp

Tests the two sources of a mosaic's three partner images.

This file needs no device. `Unit_ImagingPartners` covers `DatasetPartners` of
`include/SushiAI/imaging/dataset_partners.hpp`, which draws partners from the whole dataset,
and `RecentPartners` of `include/SushiAI/imaging/recent_partners.hpp`, which draws them from a
buffer of recently seen indices.

### buffer_after

`buffer_after` passes a list of indices to `RecentPartners::choose`, one by one with a
generator seeded 1, and returns the buffer, oldest first.

### DatasetSourceReturnsThreeIndexDrawsInOrder

`choose` on a dataset of 1000 returns three values equal to three `index(1000)` draws of a copy
of the generator, in draw order, and the two generators are equal afterwards.

### DatasetSourceCoversTheDatasetAndNeverLeavesIt

Over 300 seeds and a dataset of 7, every partner is below 7 and every index is a partner at
least once.

### DatasetSourceKeepsNoStateBetweenSamples

A source that already answered two calls and a new source return the same partners for the
same index and seed.

### DatasetSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and one above the largest `uint32` throw, and the largest `uint32` is
accepted. `choose(4)` on a dataset of 4 throws with "index 4" and makes no draw.

### RecentCapacityIsTheSmallestOfItsThreeBounds

`recent_capacity` is 5 for a dataset of 5, 128 for a batch of 16 and 1000 for a batch of 200:
the smallest of the dataset size, `K_RECENT_BATCHES` (8) batches and `K_RECENT_LIMIT` (1000).

### RecentSourceGivesTheFirstSampleItselfThreeTimes

The first call, for index 42, returns 42 three times, and the buffer holds 42.

### RecentSourceEvictsTheOldestIndexAtCapacity

With capacity 3 the buffer goes {0, 1}, {1, 2}, {2, 0}. With capacity 2 it holds one index at a
time. The reference removes the oldest entry when the buffer has more than one entry and has
reached its capacity, so a full buffer holds capacity minus one indices.

### RecentSourceDoesNotAddAnIndexItAlreadyHolds

After the indices 7, 9, 7, 7, 9 the buffer is {7, 9}.

### RecentSourceOfOneImageNeverEmptiesItsBuffer

With a dataset of one image, two calls both return 0 three times and the buffer holds 0.

### RecentSourceHoldsTheLastIndicesOfALongRun

With capacity 16, after the indices 0 to 19 the buffer holds 15 indices, from 5 to 19.

### RecentSourceDrawsThreeEntriesOfItsBuffer

After 40 earlier indices and the index 500, the buffer has 41 entries. Each partner is the
buffer entry that one `index(41)` draw of a copied generator names, in draw order, and the
source makes three draws.

### RecentSourceCoversItsBuffer

With the indices 10 to 14 buffered, over 200 seeds every partner is one of them and each is
drawn at least once.

### RecentSourceRejectsSizesAndIndicesOutOfRange

A dataset size of 0 and batch sizes of 0 and -3 throw. `choose(4)` on a dataset of 4 throws
with "index 4", makes no draw and leaves the buffer empty.

## test_imaging_pipeline.cpp

Tests the training and the validation pipeline that compose the imaging transforms into a
sample.

This file needs no device and reads no file. `Unit_ImagingPipeline` covers
`include/SushiAI/imaging/pipeline.hpp`. A `TrainingPipeline` loads an image from a
`SampleSource`, draws and applies mosaic, affine, colour jitter and flip, and returns a
`TrainingTrace` of the plans it used. A `ValidationPipeline` resizes the long side and
letterboxes. The cases replay a sample's draws on a copy of its generator and rebuild the
sample from the trace with the transform functions, so the pipeline is compared with its parts
and not with stored images.

### Constants

`K_SIDE` is 32, the training size of the cases, so a mosaic canvas is 64 by 64.
`K_REPLAY_TOLERANCE` is 0: a replayed draw repeats the same arithmetic. `K_BOX_TOLERANCE` is
1e-4 px. `K_MIN_SCALE_GAP` is 2^-24, one step of the 24-bit uniform over the default scale
range of width 1.

### MemorySource

`MemorySource` is a `SampleSource` that holds images and labels in memory and records the index
of every `load`. `loads` returns the recorded indices and `clear_loads` forgets them.

### FixedPartners

`FixedPartners` is a `PartnerSource` that returns the three partners it was constructed with
and makes no draw.

### make_textured, make_constant and make_flat

`make_textured` returns a three-channel image whose every byte depends on its row, column,
channel and a salt. `make_constant` returns a three-channel image of one colour. `make_flat`
returns an image of a given channel count with every byte at one level.

### one_object

`one_object` returns the `FileLabels` of one object of a given class and box.

### fill_source

`fill_source` adds five textured images to a source: 32 by 64, 64 by 64, 64 by 32, 16 by 32 and
32 by 32. The first four carry one object each and the last carries none.

### make_settings

`make_settings` returns default `TrainingSettings` with the image size `K_SIDE`.

### make_still_settings

`make_still_settings` returns settings under which nothing moves: one epoch with
`close_mosaic` 1, so epoch 0 has no mosaic, affine ranges of 0, HSV gains of 0 and a flip
probability of 0.

### whole_dataset

`whole_dataset` returns a `DatasetPartners` over every image of a source.

### images_match

`images_match` returns a GoogleTest assertion result that says whether two images have the same
size, channel count and bytes, and how they differ when they do not.

### expect_same_labels

`expect_same_labels` expects two label sets to hold the same classes, box edges and polygon
vertices at `K_REPLAY_TOLERANCE`.

### expect_same_affine and expect_same_hsv

`expect_same_affine` compares the sizes and the three drawn values of two `AffinePlan` objects.
`expect_same_hsv` compares the three gains of two `HsvPlan` objects.

### load_patch

`load_patch` loads one image the way a pipeline does: `resize_long_side` to `K_SIDE`,
`expand_gray` when the image has one channel, and `to_pixel_labels` of the resampled polygons.

### compose_from_trace

`compose_from_trace` builds a sample from a `TrainingTrace` by calling the transforms in the
reference's order, mosaic, affine, colour jitter, flip (`data/augment.py`, lines 2870 to 2912),
each with the plan the trace holds.

### SettingsDefaultToTheReferenceRun

Default `TrainingSettings` have image size 640, 100 epochs, `close_mosaic` 10, batch size 16
and seed 0. Default `ValidationSettings` have image size 640 and a letterbox that does not
scale up.

### MosaicClosesForTheLastTenEpochsOfAHundred

`is_mosaic_closed` is false for epochs 0 and 89 of 100 with `close_mosaic` 10 and true for 90
and 99. With `close_mosaic` 0 it is false for epoch 99, and with 5 epochs and `close_mosaic`
10 it is true for epoch 0.

### SampleRandomDerivesFromTheSeedTheEpochAndTheIndex

`make_sample_random(2026, 3, 17)` equals `Random(2026).derive(3).derive(17)`. Another seed,
epoch or index gives another generator, and so do the epoch and the index swapped.

### TrainingDrawsMosaicThenAffineThenHsvThenFlip

The trace of sample 2 in epoch 5 equals a replay on `make_sample_random(41, 5, 2)`:
`draw_mosaic`, then `plan_affine` from the 64 by 64 canvas to 32 by 32, then `plan_hsv`, then
`plan_flip`.

### TrainingLoadsTheSampleThenItsPartnersInDrawOrder

The source records four loads: index 3, then the trace's three partners in order.

### TrainingAppliesItsPlansInTheReferenceOrder

For every index of the source in epochs 0 and 1, the sample's image and labels equal what
`compose_from_trace` builds from the trace.

### TrainingAppliesAForcedFlipAndSkipsAForbiddenOne

Two pipelines with one seed and flip probabilities of 1 and 0 give images that are mirrors of
each other, pixel for pixel. The flipped sample also equals `compose_from_trace` of its trace.

### TrainingProducesTheTrainingSizeAndTheSizesOfItsOwnImage

The sample is 32 by 32 with `K_RGB_CHANNELS` channels. `original_size` is 32 by 64, the stored
size of image 0, and `loaded_size` is 16 by 32, its size after the long-side resize. The affine
plan maps 64 by 64 to 32 by 32.

### MosaicSampleKeepsItsLargestQuadrantBoxInsideTheTrainingSize

Four 32 by 32 images each carry a box over the whole image, and the affine ranges are 0, so
the warp shows canvas rows and columns 16 to 48. Over eight seeds every box of the sample lies
inside the 32 by 32 output with positive width and height, and exactly one box equals the
quadrant that is at least 16 by 16, computed from the trace's centre.

### ClosedEpochSkipsTheMosaicAndItsDraws

With 10 epochs and `close_mosaic` 3, epoch 7 is closed. The trace says so, the mosaic is not
enabled, the source records one load, and the affine, HSV and flip plans equal a replay that
starts with `plan_affine` on the sample's generator. No mosaic draw precedes them. The sample
equals `compose_from_trace`.

### SampleWithoutMosaicKeepsTheBoxOfItsOwnImage

In a closed epoch with affine ranges of 0, the box (8, 4, 24, 12) of the 16 by 32 loaded image
becomes (8, 12, 24, 20) in the 32 by 32 output: the image is centred, which adds 8 to y. The
box is symmetric about the centre column, so a flip leaves it where it is.

### EpochBeforeTheClosedOnesStillMakesAMosaic

Epoch 6 of the same settings is open: the mosaic is enabled and the source records four loads.

### ClosedGateInAnOpenEpochWarpsTheLoadedImage

With a mosaic probability of 0 in an open epoch, the trace is not closed and the mosaic is not
enabled. The affine plan's source is the 32 by 16 loaded image, the source records one load,
and the sample equals `compose_from_trace`.

### SampleDependsOnTheSeedTheEpochAndTheIndexAlone

The same seed, epoch and index give the same image, labels, affine plan and partners a second
time. Another seed and another epoch each give another image and an affine scale that differs
by more than `K_MIN_SCALE_GAP`.

### EarlierSamplesDoNotShiftTheDrawsOfALaterOne

A pipeline that already made a mosaic sample and a closed-epoch sample, which differ by six
draws, gives the same sample for index 0 of epoch 0 as a new pipeline does.

### DefaultPartnersComeFromTheImagesThePipelineLoaded

A pipeline built without a partner source gives the first sample, index 3, the partners
3, 3, 3. The second sample's partners are each 3 or 1, the two indices loaded so far.

### GivenPartnerSourceChoosesThePartners

With `FixedPartners` of 4, 4, 0 the trace holds those partners and the source records the loads
2, 4, 4, 0.

### TrainingRejectsEverySettingOutOfRangeAtConstruction

Constructing a `TrainingPipeline` throws for an image size of 0 or 8193, 0 epochs, a
`close_mosaic` of -1, a batch size of 0, a mosaic probability of 1.5, an affine scale range of
1, a negative HSV gain and a NaN flip probability, each under the name of the validator that
found it. No rejected construction loads an image, and `validate_training` accepts the test
settings.

### TrainingRejectsAnEmptySourceAndANullPartnerSource

An empty source throws with "TrainingPipeline" and "no image", with or without a partner
source. A null partner source throws with "partner source is null".

### TrainingRejectsIndicesAndEpochsOutsideTheirRange

`make_sample` throws for index 5 of five images and for epochs -1 and 100 of 100.

### TrainingExpandsAGreyImageToRgb

A one-channel image at level 77 under the still settings gives a 32 by 32 sample of three
channels whose centre pixel is (77, 77, 77).

### TrainingComposesAMosaicOfGreyAndColourImages

A source of two one-channel images and one colour image, with fixed partners, gives a
three-channel 32 by 32 sample that equals `compose_from_trace`.

### TrainingRejectsAnImageOfFourChannels

A four-channel image throws from `make_sample` with "4 channels" and "image 0".

### TrainingCarriesAPolygonAndTakesTheBoxFromIt

Under the still settings a triangle in a 32 by 32 image keeps its place. The sample's one
object has class 5, the box (8, 8, 24, 24) and a polygon of `K_POLYGON_POINTS` points that
starts at (8, 8).

### ValidationResizesTheLongSideThenOnlyPads

An 8 by 16 image of one colour is loaded as 16 by 32 and letterboxed into 32 by 32 with ratio 1
and 8 rows above and below. The sample's rows 8 to 23 hold the colour and the other rows hold
114. The source records one load.

### ValidationMovesBoxesByThePaddingWithoutClipping

The box (8, 4, 24, 12) of the loaded image becomes (8, 12, 24, 20): the top padding adds 8 to
y.

### ValidationIntoARectangularTargetAddsNoPadding

With a 16 by 32 target for a 16 by 32 image the four paddings are 0 and the sample's image
equals the source image.

### ValidationRepeatsItsSampleAfterAnotherIndex

Sample 0 made again after sample 2 has the same image and labels as the first time, and sample
2 has another image.

### ValidationExpandsAGreyImageToRgb

A one-channel 8 by 16 image at level 61 gives a three-channel 32 by 32 sample with 61 in the
content rows and 114 in the padding.

### ValidationCarriesAPolygonThroughTheLetterbox

A triangle's polygon is resampled to `K_POLYGON_POINTS` points, and its first vertex, (8, 4) on
the loaded image, becomes (8, 12).

### ValidationRejectsSettingsAndIndicesOutsideTheirRange

Constructing a `ValidationPipeline` throws for an image size of 0 ("validate_validation") and
for an empty source ("no image"). `make_sample` throws for index 9 of five images.

## test_imaging_polygon.cpp

Tests polygon resampling, mapping, clamping, the point test and the visible box.

This file needs no device. `Unit_ImagingPolygon` covers
`include/SushiAI/imaging/polygon.hpp`. The expected vertices and boxes are the outputs of the
reference's `resample_segments` and `segment2box` on the same inputs, written as literals; a
comment in each case names the call.

### Tolerances and K_IMAGE

`K_POINT_TOLERANCE` is 1e-6, for a resampled coordinate of magnitude up to 4.
`K_BOX_TOLERANCE` is 1e-4 px, for a box edge up to 100. `K_IMAGE` is 80 rows by 100 columns.

### expect_polygon

`expect_polygon` compares a polygon with a flat list of expected x, y values.

### expect_box

`expect_box` compares the four edges of a `PixelBox` within `K_BOX_TOLERANCE`.

### make_square, make_triangle and make_hexagon

`make_square` returns the square from (0, 0) to (4, 4). `make_triangle` returns a triangle in
fractions of an image. `make_hexagon` returns six vertices with integer coordinates.

### PointCountIsAThousandOrTheLongestPlusOne

`K_POLYGON_POINTS` is 1000. `resampled_point_count` is 1000 for a longest polygon of 0, 3 or
1000 vertices and 1002 for 1001.

### ResampleMergesTheVerticesWithEvenPositions

The square resampled to 8 and to 11 points keeps its four vertices and gains the evenly spaced
positions between them.

### ResampleToOneMoreThanTheClosedLengthRepeatsTheFirstVertex

The square resampled to 6 points gains one point, at the first vertex.

### ResampleInterpolatesAlongTheEdges

The triangle resampled to 9 points holds interpolated points such as (0.55, 0.275) on its first
edge.

### ResampleOfALongPolygonTakesEvenPositionsOnly

The hexagon, closed, has seven points. Resampled to 4 it gives every second vertex and the
first again; resampled to 7 it gives the six vertices and the first again.

### ResampleLeavesAPolygonOfTheCountAndAnEmptyOneAlone

The triangle resampled to 3 points is unchanged, and an empty polygon stays empty.

### ResampleToAThousandPointsKeepsTheOutline

The triangle resampled to 1000 points has 1000 points, starts and ends at x = 0.1, and has the
bounds (0.1, 0.2, 0.7, 0.9).

### ResampleRejectsACountOfZero

A count of 0 throws with "resample_polygon" and "point count of 0".

### MapScalesThenTranslatesEachAxis

`map_polygon` with scales 2 and 0.5 and offsets 10 and -1 maps the square to the corners
(10, -1), (18, -1), (18, 1), (10, 1).

### ClampMovesVerticesOntoTheBounds

`clamp_polygon` to the box (0, 0, 4, 5) moves each coordinate outside onto the nearest bound
and leaves the vertex inside alone.

### PointInsideOrOnTheOutlineIsInside

`is_point_inside` is true for the centre of the square, a point on an edge and two vertices.

### PointOutsideIsOutside

`is_point_inside` is false for four points outside the square, for a point in the notch of a
concave polygon and for any point against an empty polygon. Two other points of the concave
polygon are inside.

### VisibleBoxOfAPolygonInsideTheImageIsItsBounds

A triangle inside `K_IMAGE` has the visible box (10, 20, 50, 60).

### VisibleBoxOfAPolygonOutsideTheImageIsZero

A triangle above and left of the image, one to its right and an empty polygon each give four
zeros.

### VisibleBoxUsesTheCrossingsOfTheImageLines

A triangle crossing the top and left edges gives (0, 0, 60, 50), and one crossing the right
and bottom edges gives (90, 70, 100, 80).

### VisibleBoxIncludesTheImageCornersThePolygonContains

A triangle that contains the image's origin gives (0, 0, 36.666664, 33), and a square around
the whole image gives the image, (0, 0, 100, 80).

## test_imaging_random.cpp

Tests the generator every augmentation draws from.

This file needs no device. `Unit_ImagingRandom` covers `Random` of
`include/SushiAI/imaging/random.hpp`: a 64-bit counter advanced by `K_INCREMENT` and passed
through `mix`. The first cases follow the arithmetic by hand, with each intermediate value as a
literal, so the sequence is fixed by this file and not by a second copy of the code. The
statistical cases use one seed and state their bound as a multiple of the standard error.

### Constants

`K_SEED` is 2026, the seed of every statistical case. `K_SAMPLES` is 65536. `K_UNIFORM_VALUES`
is 2^24, the number of values `uniform()` can return.

### MixFollowsTheFinaliserStepByStep

For the value 0x9E3779B97F4A7C15 the case asserts each of the five steps of `mix`: the shift by
30 and xor, the product with `K_FIRST_MULTIPLIER`, the shift by 27 and xor, the product with
`K_SECOND_MULTIPLIER` and the shift by 31 and xor. `mix` returns the last literal.

### MixOfTheSecondCounterFollowsTheFinaliserStepByStep

The same five steps for 0x3C6EF372FE94F82A, the second counter of seed 0.

### SeedZeroReproducesTheHandComputedSequence

The first four `next_u64` values of seed 0 are the literals the case lists, and the counter
after draw n is n times `K_INCREMENT` modulo 2^64.

### TheCounterStartsAtTheSeed

`state()` of a new generator is its seed. The first draw of seed 42 is
`mix(42 + K_INCREMENT)`, and the counter is then `42 + K_INCREMENT`.

### NextU32IsTheHighHalfOfOneDraw

The first two `next_u32` values of seed 0 are the high 32 bits of its first two draws, and two
calls advance the counter twice.

### UniformIsTheTop24BitsScaledByTwoToTheMinus24

The first three `uniform()` values of seed 0 are 14819496, 7239838 and 443485 over 2^24, the
top 24 bits of each draw.

### UniformStaysInTheHalfOpenUnitInterval

65536 draws are all at or above 0 and below 1.

### UniformHasTheMeanAndVarianceOfAUniformVariable

Over 65536 draws the mean is 0.5 within 0.006 and the variance 1/12 within 0.0015, five
standard errors each.

### RangedUniformMapsTheUnitDrawLinearly

`uniform(-1, 1)` and `uniform(0.5, 1.5)` of seed 0 return `low + (high - low) * u` for the
first two unit draws, as exact floats, and make one draw each.

### RangedUniformStaysInItsHalfOpenInterval

65536 draws of `uniform(0.4, 0.6)` are all at or above 0.4 and below 0.6.

### RangedUniformNeverReturnsItsUpperBound

With `high` the float after `low`, 1024 draws all return `low`. A sum that rounds up to `high`
is brought back.

### RangedUniformOverAnEmptyRangeStillDraws

`uniform(0.25, 0.25)` returns 0.25 and advances the generator by one draw.

### IndexIsTheHighWordOfTheProduct

`index(10)` of seed 0 returns 8, 4 and 0: the high 32 bits of each 32-bit draw times 10.

### IndexOfOneIsAlwaysZeroAndStillDraws

64 calls of `index(1)` return 0 and advance the generator 64 times.

### IndexStaysBelowTheCountAndFillsEveryBucketEvenly

70000 draws of `index(7)` are all below 7, and each bucket holds 10000 within 500, above five
standard deviations of 93. `K_BUCKETS` and `K_DRAWS` are local to the case.

### IndexReachesTheLargestCount

`index(0xFFFFFFFF)` of seed 0 returns 0xE220A838.

### BernoulliComparesOneUniformDrawWithTheProbability

`bernoulli(0.5)` of seed 0 returns false, true, true for the unit draws 0.8833, 0.4315 and
0.0264, one draw per call.

### BernoulliNeverSucceedsAtZeroAndAlwaysSucceedsAtOne

Over 65536 draws, probability 0 never succeeds and probability 1 always does, and both make
one draw per call.

### AZeroDrawIsTheLowerBoundAndFailsAZeroProbability

`mix(0)` is 0, so a seed of minus `K_INCREMENT` makes the first draw 64 zero bits. `uniform()`
returns 0, `bernoulli(0)` returns false and `uniform(0.4, 0.6)` returns 0.4. A draw of exactly
0 does not pass a probability of 0.

### BernoulliSucceedsAtTheRateOfItsProbability

Over 65536 draws `bernoulli(0.25)` succeeds at a rate of 0.25 within 0.01.

### DeriveStartsAtTheMixOfTheStateAndTheKeyedMix

`Random(0).derive(0).state()` is 0x48218226FF3CD4BF. The case follows the five steps of `mix`
from the keyed value 0xE220A8397B1DCDAF.

### DeriveDependsOnTheKeyAndOnTheParentState

`Random(0).derive(1)` and `Random(7).derive(5)` have the two states the case lists. Two keys
of one parent give different generators, and so does one key of two parents.

### DeriveLeavesTheParentUntouched

After `derive(3)` the parent's state is its seed, the child differs from the parent, and a
second `derive(3)` equals the first.

### ACopyReplaysTheSequenceFromThePointOfTheCopy

A copy taken after one draw equals the original, and their next 16 draws are equal.

### EqualityComparesTheCounter

Two generators of one seed are equal, and are unequal after one of them draws.

## test_imaging_raw_cache.cpp

Tests the raw image file: its bytes, its header checks and its file reader and writer.

This file needs no device. `Unit_ImagingRawCache` covers
`include/SushiAI/imaging/raw_cache.hpp`. A file is a 24-byte header of six little-endian
32-bit words (magic, version, height, width, channel count, channel order) followed by the
pixels. Most cases read and write string streams. The cases from ImageRoundTripsThroughAFile on
write files in a directory of their own under the system's temporary directory, which
`ScratchDirectory` creates empty and removes when the case ends.

### K_RGB

`K_RGB` is 1, the channel order word of an RGB file.

### append_u32

`append_u32` appends a value to a byte string as four little-endian bytes.

### header_bytes and layout_bytes

`header_bytes` returns the 24 header bytes for six given fields. `layout_bytes` returns them
for `K_RAW_IMAGE_MAGIC`, `K_RAW_IMAGE_VERSION` and a given height, width, channel count and
order.

### make_image

`make_image` returns an image whose byte at offset i is 7 * i + 3 modulo 256.

### bytes_of

`bytes_of` returns a copy of an image's bytes.

### written_bytes

`written_bytes` returns the bytes `write_raw_image` produces for an image.

### expect_untouched

`expect_untouched` expects an image to be 1 by 1 with one channel still, the destination a
failed read must not replace.

### ScratchDirectory

`ScratchDirectory` owns one directory under `std::filesystem::temp_directory_path()`. It
removes what the path held and creates the directory in its constructor, and removes it in its
destructor. `write` puts bytes in a file of the directory and `file` returns a path inside it.
The class cannot be copied or moved.

### WriterEmitsTheDocumentedBytes

A 1 by 2 RGB image is written as "SAIM", the words 1, 1, 2, 3, 1 and the six pixel bytes. The
case lists all thirty bytes.

### HeaderSizeAndMagicMatchTheDocumentedLayout

`K_RAW_IMAGE_HEADER_BYTES` is 24 and the first four header bytes are "SAIM".

### RgbImageRoundTripsThroughAStream

A 5 by 7 RGB image written and read back has the header fields 5, 7, 3 and
`ChannelOrder::Rgb`, `pixel_bytes()` of 105 and the same bytes.

### GrayImageRoundTripsThroughAStream

A 4 by 3 one-channel image round-trips with `ChannelOrder::Gray`.

### BgrFileIsWrittenAsGivenAndReadAsRgb

Written with `ChannelOrder::Bgr`, the pixel bytes 1 to 6 are stored as given. Read back they
are 3, 2, 1, 6, 5, 4: the reader swaps the first and third channel so the image is RGB.

### ReadReusesADestinationOfAnotherSize

A destination that was 9 by 9 with one channel becomes 2 by 2 with three channels and holds the
file's bytes.

### HeaderRejectsAWrongMagic

A header that starts with the PNG signature bytes throws with "magic" and "not a raw image
file".

### HeaderRejectsANewerVersionAndNamesBoth

Version 2 throws with "format version 2" and "reads version 1".

### HeaderRejectsVersionZero

Version 0 throws with "format version 0" and "reads version 1".

### HeaderRejectsATruncatedHeader

A stream of 23 bytes throws with "header ended" and "23 of 24".

### HeaderRejectsAnEmptyStream

An empty stream throws with "header ended" and "0 of 24".

### HeaderRejectsAZeroHeight

A height of 0 throws with "height 0" and "16384".

### HeaderRejectsAZeroWidth

A width of 0 throws with "width 0" and "16384".

### HeaderRejectsAHeightAboveTheLimit

A height of 16385 throws with "height 16385".

### HeaderRejectsAWidthAboveTheLimit

A width of 16385 throws with "width 16385".

### HeaderAcceptsBothSidesAtTheLimit

A 16384 by 16384 RGB header is accepted, and `pixel_bytes()` is 805306368.

### HeaderRejectsExtentsWhoseProductWouldOverflow

Extents of 0xFFFFFFFF throw with "height 4294967295". The extent is checked before any product
is formed.

### HeaderRejectsAnExtentWithTheSignBitSet

A width of 0x80000001 throws with "width 2147483649". The word is read as unsigned and is not
taken for a negative number.

### HeaderRejectsAnUnknownChannelOrder

A channel order of 3 throws with "channel order 3" and "unknown".

### HeaderRejectsThreeChannelsInAGrayFile

Three channels with order 0 throw with "channel count 3" and "channel order 0".

### HeaderRejectsOneChannelInAnRgbFile

One channel with order 1 throws with "channel count 1" and "channel order 1".

### HeaderRejectsFourChannels

Four channels throw with "channel count 4".

### HeaderRejectsZeroChannels

Zero channels with order 2 throw with "channel count 0" and "channel order 2".

### PixelsRejectAStreamThatEndsEarly

A stream one byte short of its pixels throws with "pixels ended" and "11 of 12".

### PixelsRejectAHeaderTheReaderDidNotValidate

`read_raw_image_pixels` given a hand-built header of height 0 throws with "height 0" and leaves
the destination as it was.

### WriterRejectsAnEmptyView

A default view throws with "no pixels" and nothing is written.

### WriterRejectsAChannelCountThatDoesNotFitTheOrder

One channel with `Rgb` or `Bgr` and three channels with `Gray` throw with the channel count,
and nothing is written.

### WriterRejectsASideAboveTheLimit

A view 16385 px wide throws with "width 16385" and nothing is written.

### WriterReportsAFailedStream

Writing to a stream whose bad bit is set throws with "stream failed".

### ImageRoundTripsThroughAFile

A 6 by 4 RGB image written with `write_raw_image_file` gives a file of 24 + 72 bytes, and
`read_raw_image_file` returns the same extents and bytes.

### FileWriterReplacesALongerFile

Writing a 1 by 1 image over a file that held an 8 by 8 one leaves a file of 24 + 3 bytes that
reads back as the 1 by 1 image.

### FileWriterLeavesTheFileAloneWhenItRejectsTheImage

A write that throws for its channel count leaves an existing four-byte file at four bytes. The
image is checked before the file is opened.

### FileWriterRejectsAPathItCannotOpen

A path inside a directory that does not exist throws with "cannot open".

### FileReaderRejectsAMissingFile

A missing file throws with "cannot open" and the file name, and the destination is untouched.

### FileReaderRejectsATruncatedHeader

A file of 10 bytes throws with "header ended" and "10 of 24", and the destination is untouched.

### FileReaderRejectsTruncatedPixelsBeforeResettingTheDestination

A file one byte short throws with "holds 35 bytes" and "declares 36", and the destination is
untouched.

### FileReaderRejectsTrailingBytesBeforeResettingTheDestination

A file with one byte after its pixels throws with "holds 37 bytes" and "declares 36", and the
destination is untouched.

### FileReaderAllocatesNothingForALargeImageTheFileDoesNotHold

A file that is only a header declaring 16384 by 16384 pixels throws with "holds 24 bytes" and
"declares 805306392". The destination is untouched, so the reader compares the file size with
the header before it allocates.

### FileReaderRejectsAMalformedHeaderBeforeResettingTheDestination

A file whose header has extents of 0xFFFFFFFF throws with "height 4294967295", and the
destination is untouched.

## test_imaging_rectangle.cpp

Tests the plan that orders a dataset by aspect ratio and gives each batch one shape.

This file needs no device. `Unit_ImagingRectangle` covers
`include/SushiAI/imaging/rectangle.hpp`. The expected shapes are what lines 405 to 427 of the
reference's `data/base.py` give for the same sizes.

### make_sizes

`make_sizes` returns seven image sizes with height over width below, at and above 1. Images 0
and 5 share the ratio 0.75.

### make_settings

`make_settings` returns `RectangleSettings` with a batch size, an image size and a pad, at the
default stride of 32.

### SettingsDefaultToTheReferenceValidationRun

Default settings are image size 640, batch size 32, stride 32 and pad 0.5.

### OrderRisesWithTheAspectRatioAndKeepsTies

`plan_rectangular_batches` orders the seven images as 6, 3, 0, 5, 2, 1, 4. Images 0 and 5 keep
their dataset order.

### ShapesMatchTheReferenceArithmetic

With batch 3 and pad 0.5 the shapes are 512 by 672, 672 by 672 and 672 by 448. With batch 2 and
pad 0 they are 448 by 640, 480 by 640, 640 by 640 and 640 by 448. With one batch of 7 at image
size 320 the shape is 352 by 352.

### ShapeAtNamesTheBatchOfAPosition

`shape_at` returns the first batch's shape for positions 0 and 2, the second's for position 3
and the third's for position 6.

### EverySideIsAMultipleOfTheStride

With stride 64 and image size 600, both sides of all four shapes are multiples of 64 and at
least 64.

### OneWideImageGivesOneWideShape

One 480 by 640 image gives the order {0} and the shape 512 by 672:
ceil(0.75 * 640 / 32 + 0.5) * 32 and ceil(20.5) * 32.

### RejectsSettingsAndSizesOutsideTheirRanges

`validate_rectangle` throws for a batch size, a stride or an image size of 0 and for a pad that
is negative or NaN. `plan_rectangular_batches` throws for an empty size list ("no image") and
for a size of 0 by 500 ("image 2", "0x500"). Default settings pass.

### RejectsAShapeAboveTheImageBound

An image size of `K_MAX_IMAGE_SIDE` with pad 0.5 gives a side above the bound, and the call
throws with "batch side" and "16384".

## test_imaging_resize.cpp

Tests the bilinear resize and the fit of an image's long side.

This file needs no device. `Unit_ImagingResize` covers `include/SushiAI/imaging/resize.hpp`:
`resize_bilinear`, `fit_long_side` and `resize_long_side`. The kernel samples at pixel centres
with weights rounded to `K_BILINEAR_WEIGHT_BITS` bits and rounds its result half up. The first
cases compare whole images with values a comment derives.

### Bytes

`Bytes` is a vector of bytes, an image's contents in height, width, channel order.

### make_image and bytes_of

`make_image` returns an image holding the given bytes, and `bytes_of` returns an image's bytes.

### resized_bytes

`resized_bytes` resizes a one-channel source to a size and returns the bytes. It also expects
the destination to have that size and one channel.

### WeightBitsGiveAWeightSumOf2048

`1 << K_BILINEAR_WEIGHT_BITS` is 2048.

### TwoByTwoToFourByFourMatchesTheHandComputedImage

Along one axis the four outputs of a pair (a, b) are a, (3a + b) / 4, (a + 3b) / 4 and b. The
case lists all sixteen output bytes, among them 58.75 rounded to 59 and 76.25 rounded to 76.

### FourByFourToTwoByTwoAveragesEachTwoByTwoBlock

Each output reads position 0.5 or 2.5 on both axes, so it is the mean of a 2 by 2 block. The
sums 142, 220, 460 and 541 give 36, 55, 115 and 135: 35.5 rounds up and 135.25 rounds down.

### ResizeToTheSameSizeCopiesEveryPixel

A 3 by 3 image resized to 3 by 3 is unchanged.

### AnEnlargedRowClampsAtBothEdges

A row (10, 250) enlarged to four columns starts at 10 and ends at 250. Its outer columns read
positions outside the two pixel centres.

### AReducedRowUsesWeightsRoundedToElevenBits

Eight columns reduced to three read positions 0.8333, 3.5 and 6.1667, with upper weights 1707,
1024 and 341 of 2048. The case computes the three bytes from those weights, checks they are 3,
41 and 209, and compares the resize with them.

### EachChannelIsInterpolatedOnItsOwn

A two-pixel row of three channels enlarged to four pixels gives each channel the values of its
own pair. Channel 2 falls from 255 to 3 while the others rise.

### AConstantImageStaysConstantWhenEnlarged

A 5 by 7 image of the colour (17, 114, 255) enlarged to 13 by 23 holds that colour in every
pixel.

### AConstantImageStaysConstantWhenReduced

A 31 by 17 image at level 201 reduced to 4 by 5 holds 201 in every byte.

### AHorizontalGradientStaysMonotoneAtEveryWidth

A rising row of eight levels resized to every width from 1 to 40 never falls from one column to
the next.

### AnEnlargedGradientKeepsItsEndValues

The same row enlarged to 20 columns starts at 0 and ends at 255.

### AVerticalGradientStaysMonotoneAndRowsStayUniform

A 4 by 3 image whose rows are each one level is resized to every height from 1 to 17 at width
5. Every output row is one level, and the levels never fall from one row to the next.

### TheDestinationIsResetToTheRequestedSizeAndChannels

A destination that was 9 by 9 with one channel becomes 2 by 2 with the source's three channels.

### ResizeRejectsASourceWithoutPixels

`resize_bilinear` on a default view throws with "resize_bilinear", "source" and "no pixels".

### ResizeRejectsASizeOutsideTheImageBounds

Destination sizes of 0 by 2, 2 by -1 and 2 by 16385 throw with "destination size" and the size.

### ResizeRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2, of 2 by 16385 and of five channels throws with "source size" or
"source channel count".

### ResizeRejectsADestinationThatHoldsTheSource

`resize_bilinear` with the image as source and destination throws with "destination holds the
source's pixels". The image is still 2 by 2 with its bytes, so the refusal comes before the
reset.

### FitLongSideLeavesAnImageWhoseLongSideAlreadyFits

`fit_long_side` at 640 returns 480 by 640, 640 by 427 and 640 by 640 unchanged.

### FitLongSideReducesALargeImage

1080 by 1920 becomes 360 by 640, and 1920 by 1080 becomes 640 by 360.

### FitLongSideEnlargesASmallImage

100 by 50 becomes 640 by 320, and 500 by 375 becomes 640 by 480.

### FitLongSideRoundsTheShortSideUp

333 by 500 becomes 427 by 640, 7 by 3 becomes 640 by 275 and 3 by 1000 becomes 2 by 640: the
short side is the ceiling of 426.24, 274.29 and 1.92.

### FitLongSideCapsBothSidesAtTheImageSize

1281 by 1280 becomes 640 by 640. The short side would be the ceiling of 639.5.

### FitLongSideAlwaysPutsTheLongSideAtTheImageSize

Over heights and widths from 1 to 1499 in steps of 7 and 11, the long side of the result is 640
and the short side lies in [1, 640].

### ResizeLongSideCopiesAnImageThatAlreadyFits

A 2 by 4 image with image size 4 comes out as a 2 by 4 image with the same bytes under another
data pointer. The destination, which was 3 by 3 with three channels, takes the source's one
channel.

### ResizeLongSideEnlargesToTheFittedSize

A 1 by 2 image with image size 4 becomes 2 by 4, and both rows are 0, 25, 75, 100.

### ResizeLongSideReducesToTheFittedSize

The 4 by 4 image of the block-average case with image size 2 becomes the 2 by 2 image 36, 55,
115, 135.

### ResizeLongSideRejectsAnImageSizeOutsideTheImageBounds

Image sizes of 0 and 16385 throw with "resize_long_side" and the value.

### ResizeLongSideRejectsASourceWithoutPixels

A default view throws with "resize_long_side" and "no pixels".

### ResizeLongSideRejectsADestinationThatHoldsTheSource

The image as source and destination throws with "destination holds the source's pixels".

### ResizesRejectAViewInsideTheDestination

`resize_bilinear` and `resize_long_side` both refuse a view of two rows of the destination, and
the image keeps its size and its bytes.

### ResizeLongSideRejectsASourceWithExtentsOutsideTheImageBounds

A source view of 0 by 2 and one of five channels throw with "source size" and "source channel
count".

## test_ir.cpp

Tests the IR's single-assignment and ordering rules and the builder's shape inference.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### ASecondProducerForOneValueThrows

Single assignment is the property that lets differentiation skip alias analysis entirely, so a
double write is rejected and not tolerated.

### NodesAreEmittedInTopologicalOrder

The lowering relies on insertion order as a topological order, and differentiation relies on
its reverse, so the test asserts it.

### InsertNodeRejectsAResultThatIsAlreadyReadEarlier

The graph exercises the half of `insert_node`'s ordering guard that single assignment does not
already cover. A result with no producer passes the single-assignment check whatever its
position. Without this half the function would accept a producer placed after its own reader,
which is the arrangement it exists to prevent, and `MemoryPlan::build()` would then reject a
graph `insert_node` had approved.

`node_count()` is still 1 after the throw. The insert was rejected, not half-applied: nothing
was inserted and no producer was claimed.

The final `EXPECT_NO_THROW` inserts the same node ahead of that reader and succeeds. That makes
the rejection a statement about position and not about the node.

### AttributesHoldAConvolutionsGeometryAndItsWindowGrid

`Attributes::K_CAPACITY` is ten, not eight. The eight of CV-5 are the geometry: kernel_h/w,
stride_h/w, dilation_h/w, pad_top, pad_left. Slots 8 and 9 are the output extents out_h and
out_w, and they have to be carried and cannot be re-derived. Deriving them from the geometry
assumes symmetric padding, and an asymmetric window grid can derive to its own transpose with
the same window count, which every downstream shape check accepts.

Nothing downstream caps the ten. Nothing copies an attribute into a fixed-width metadata array:
`src/graph/lowering.cpp` reads slots with `get()` and passes the values straight to SushiBLAS.

### ABufferLeafIsPersistentAndNeedsNoGradient

The nine cases from here are in the suite `Unit_Ir`; the cases above are in `Unit_IR`.

`GraphBuilder::buffer` declares a value of the shape, type and name it was given, with the
lifetime `Lifetime::Buffer` and `requires_grad` false. Its producer is a K_LEAF node. A K_BIAS_ADD
that reads an input and a buffer does not require a gradient either. `is_persistent(BUFFER)` is
true and `lifetime_name(BUFFER)` is "buffer".

### ActivationAndScratchAreTheOnlyLifetimesThatDoNotPersist

`Graph::is_persistent` is true for PARAMETER, GRADIENT, OPTIMIZER_STATE, INPUT and BUFFER, and
false for ACTIVATION and SCRATCH.

### StateLeafReturnsTheBoundLeafAndNoValueForAnUnboundState

`Graph::state_leaf` returns `K_NO_VALUE` before any binding. After `bind_state_leaf` of state 7
to a buffer and state 5 to a parameter, it returns each leaf for its state and `K_NO_VALUE` for
state 6 and for `K_NO_STATE`. `state_leaves()` holds the two pairs in the order they were
bound, not in the order of the state numbers.

### BindStateLeafRefusesASecondBindingOfAStateOrOfALeaf

With state 7 bound to a leaf, four bindings throw: state 7 to another leaf ("State 7",
"already has leaf"), state 8 to the bound leaf ("already stands for state 7"), `K_NO_STATE` to
a leaf ("no identity"), and state 9 to value 99 ("value 99", "does not exist"). The table
still holds the first pair alone.

### WithoutNodesKeepsEveryValueAndStateLeafAndHoldsNoNode

`Graph::without_nodes()` returns a graph with no node and the same number of values. Every
value keeps its shape, element type, layout, lifetime, `requires_grad` and debug name, and its
producer is `K_NO_NODE`. The fixture holds an NCHW value, a parameter that requires a gradient
and a HALF buffer, and each is checked by name. The copy's `state_leaves()` equals the
source's. The source graph still has its nodes and producers.

### ARefusalIsReportedAfterItIsMarkedAndSurvivesWithoutNodes

`has_refusal()` is false for a new graph and for its `without_nodes()` copy. After
`mark_refused()` it is true for both.

### StateIdentityDrawsOneIdentityPerHolder

`Graph::StateIdentity` in `include/SushiAI/graph/state_id.hpp` holds the identity of one piece
of model state. `peek()` returns `K_NO_STATE` before the first `get()`. `get()` returns the
same identity on every call, `peek()` then returns it too, and a second holder's `get()`
returns a different one.

### ACopiedStateIdentityDoesNotShareTheSourcesIdentity

A holder copy-constructed from one that has drawn peeks `K_NO_STATE` and then draws a
different identity. A holder that had drawn and is then copy-assigned from the source also
peeks `K_NO_STATE`, and its next identity differs from the source's and from the one it had.
The source keeps its identity.

### AMovedStateIdentityCarriesItsIdentityAway

After a move construction the target peeks the source's identity and the source peeks
`K_NO_STATE`. A move assignment over a holder that had drawn does the same.

## test_layer_registry.cpp

Tests the layer registry: each row's factory, the keys it reads and the model it builds.

This file needs no device. `NN::K_LAYER_REGISTRY` in `include/SushiAI/nn/layer_registry.hpp` is
a constant table of fourteen rows, and a factory turns a per-sample input shape and a parsed
JSON object into an `NN::Layer`. A case therefore needs a `Shape`, a string of JSON and, for
the model cases, a host-side graph.

The cases fall into three groups, in file order:

- every row of the table: that it has a factory, builds its own layer type and refuses an
  unknown key
- the keys of each factory, their defaults and the values they refuse
- the convolutional classifier of `K_CONV_MODEL`, built, traced, named and differentiated

### K_CONV_MODEL

`K_CONV_MODEL` is a model file as a string: 784 input features, then `unflatten` to
28 x 28 x 1, `conv2d` with 8 filters, `relu`, `maxpool2d`, `conv2d` with 16 filters, `relu`,
`maxpool2d`, `flatten` and `linear` with 10 units. Both convolutions have kernel 3 and
padding 1, and both pools have kernel 2. The layers of
`examples/mnist_cnn/model.json` are the same nine, with the optional keys spelled out.

### K_CONV_MODEL_PARAMETERS

`K_CONV_MODEL_PARAMETERS` lists the six parameters that model declares, as "name:extents":
a weight and a bias under the indices 1, 4 and 8. The two tests that compare names read the
expected list from here.

### MinimalCase

`MinimalCase` holds one registry key, a layer object without its "type", the per-sample shape
that enters the layer and the per-sample shape the layer must report.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` holds one `MinimalCase` per registry row, in registry order. Each object
carries the required keys of its row and no other.

### attributes_from

`attributes_from` parses a string as the attributes of one layer.

### make_layer

`make_layer` looks a type up in `NN::K_LAYER_REGISTRY` with `Config::find_entry` and calls the
row's factory with the context "layer 0 '<type>'". The cases that assert a message expect that
context in it.

### wrapped

`wrapped<L>` returns the layer of type `L` inside an `NN::LayerAdapter<L>`, or null when the
`Layer` wraps another type. The `dynamic_cast` is how a case tells which layer type a factory
built and reads the fields the factory set.

### square_window

`square_window` returns a geometry with a square kernel and one stride and one padding for
both axes.

### make_conv

`make_conv` returns a `Conv2d` with the given channel counts and geometry, its parameters
undeclared.

### expect_pool_window_spellings_agree

`expect_pool_window_spellings_agree<Pool>` builds a pool from `"kernel": 3` and from
`"kernel": [3, 3]`. It expects both to wrap a `Pool`, the first to hold a 3x3 kernel, and the
two to hold the same kernel and the same stride.

### make_conv_stack

`make_conv_stack` parses `K_CONV_MODEL` and returns the `LayerStack` that `build_layer_stack`
makes of it.

### traced_names

`traced_names` traces a model over a [4, 784] input and returns its parameters as
"name:extents" strings. It is a template, so it takes a `LayerStack` and a `Sequential`.

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `NN::K_LAYER_REGISTRY`, and entry `i` names the type of
row `i`. A row added to the registry without a case here fails this test, so the two tests
that loop over the cases cover every factory.

### TheTypeStringsAreTheOnesModelFilesSpell

The `type` fields of `NN::K_LAYER_REGISTRY` are, in order, "linear", "gelu", "relu", "tanh",
"sigmoid", "leaky_relu", "unflatten", "conv2d", "maxpool2d", "avgpool2d", "global_avgpool",
"flatten", "upsample" and "silu". A model file names a layer by one of these strings, so a
renamed row fails here.

### EachFactoryBuildsFromAMinimalObject

For every case the factory returns a layer whose `output_shape()` is the shape the case lists.

### EachKeyBuildsItsOwnLayerType

Each of the fourteen keys builds an adapter over the layer type it names: `wrapped<L>` is not
null for the expected `L`. Two rows that pointed at one factory would fail for one of them.

### EachFactoryRejectsAnUnknownKey

For every case, an object holding only `"no_such_key"` throws with a message containing the
layer's context and "'no_such_key'".

### Conv2dTakesItsInputChannelsFromTheInputShape

A `conv2d` built over a [8, 8, 7] input has `in_channels` 7 and `out_channels` 5. An object
that sets `"in_channels"` is refused by name, so a model file cannot state a channel count
that disagrees with the layer before it.

### Conv2dRefusesAnInputThatIsNotAnImage

A `conv2d` given the per-sample shape [784] throws. The message contains the context, "[784]",
"[H, W, C]" and "'unflatten'", the layer that gives a feature vector its image shape.

### Conv2dDefaultsItsOptionalKeys

With only "filters" and "kernel" set, the layer has stride 1, padding 0 and dilation 1 on both
axes, and `use_bias` is true.

### Conv2dReadsEveryKeyItAccepts

The object sets all six keys, four of them as [h, w] pairs with unequal entries. The layer
holds kernel 3 by 5, stride 2 by 1, padding 1 by 2, dilation 1 by 2 and `use_bias` false, and
over a [16, 16, 2] input it reports [8, 12, 5].

### AKernelAsAnIntegerAndAsAPairAgree

For `conv2d`, `maxpool2d` and `avgpool2d`, `"kernel": 3` and `"kernel": [3, 3]` give the same
output shape. The case then reads the stored window: the two convolutions both hold a 3x3
kernel, and `expect_pool_window_spellings_agree` checks the two pools. Equal output shapes
alone would also hold for two layers that both stored a wrong window.

### AWindowSettingOfTheWrongFormIsRefused

A one-entry array, a three-entry array, a string and the real number 1.5 are each refused.
Every message names the layer and the key, as in "layer 0 'maxpool2d' 'kernel'", and contains
"two-entry [h, w]". The two array messages also give the entry count, "1 entries" and
"3 entries".

### APoolStrideDefaultsToItsKernel

A `maxpool2d` with `"kernel": [2, 3]` and no stride holds stride 2 by 3 and padding 0, and
maps [12, 12, 2] to [6, 4, 2]. An `avgpool2d` with kernel 3, stride 1 and padding 1 holds
those values and keeps [12, 12, 2].

### Conv2dRejectsANonPositiveExtent

Eight objects are refused: a filter count of 0 and of -4, a kernel of 0, of [3, 0] and of -3,
a stride of 0 and of [1, -1], and a dilation of 0. Each message contains the context, "Conv2d"
and "positive".

### ThePoolsRejectANonPositiveExtent

For both pools, a kernel of 0, of [2, 0] and of -2 and a stride of 0 and of [-1, 2] are
refused with a message containing the context and "positive".

### AWindowedLayerRejectsANegativePadding

For `conv2d` and both pools, a padding of -1, of [0, -1] and of [-1, 0] is refused with a
message containing the context and "padding". The two pairs put the negative entry on each
axis in turn.

### AFactoryRejectsAMissingRequiredKey

A `conv2d` without "filters", a `conv2d` without "kernel", each pool without "kernel" and a
`linear` without "units" are refused. Each message contains the context and "has no" followed
by the quoted key.

### LeakyReluRejectsASlopeOutsideItsRange

Slopes of 0, 1 and -0.1 are refused when the layer is built, with a message containing the
context and "slope".

### UpsampleRejectsAScaleBelowOne

A scale of 0 is refused when the layer is built, with a message containing the context,
"Upsample2d" and "scale".

### LeakyReluAndUpsampleReadTheirOptionalKey

A `leaky_relu` built from an empty object has slope 0.01, and one built with `"slope": 0.25`
has 0.25. An `upsample` built from an empty object maps [4, 4, 2] to [8, 8, 2], and one built
with `"scale": 3` maps it to [12, 12, 2].

### UnflattenRefusesAnInputOfAnotherSize

A 28 x 28 x 1 `unflatten` given [783] throws with a message containing the context, "[783]"
and "[784]". An object without "channels" throws with a message containing the context and
"channels".

### TheConvModelBuildsAndReportsItsOutputShape

The stack has nine layers and its output shape is [10]. The shapes after layers 1, 3, 6 and 7
are [28, 28, 8], [14, 14, 8], [7, 7, 16] and [784], so each factory received the shape its
predecessor reported.

### TheConvModelTracesToTheShapeItReports

The stack traces a [4, 784] input to a [4, 10] value.

### TheConvModelNamesItsParametersByLayerPosition

The traced stack's parameters are `K_CONV_MODEL_PARAMETERS`: indices 1, 4 and 8, with gaps at
the layers that own no parameter.

### ASequentialOfTheSameLayersYieldsTheSameNames

An `NN::Sequential` of the same nine layer types is traced over the same input. Its names
equal `K_CONV_MODEL_PARAMETERS`, and the stack's names equal the `Sequential`'s. A checkpoint
written from one model therefore names its parameters as the other reads them.

### CrossEntropyDifferentiatesToEveryParameterOfTheConvModel

The stack's output goes into `NN::CrossEntropyLoss`, and the loss is differentiated against
the six parameters the walk yields. Every gradient is a value other than `K_NO_VALUE` with the
shape of its parameter. The case checks that the graph differentiates; it computes no number.

### ASiluRowTracesOneSiluNode

The layer of a `silu` row over [8, 8, 2], traced over [4, 8, 8, 2], adds one node. Its
operation is `Ops::K_SILU` and the result keeps the shape [4, 8, 8, 2].

### AVersionTwoChainStartsFromTheInputShape

A version 2 document over the input shape [8, 8, 2] holds a `conv2d` of 5 filters and kernel
3, a `silu` and a `flatten`. `NN::build_layer_stack` builds three layers. Layer 0 reports
[6, 6, 5], which requires the factory to have received the rank-3 shape and not the count 128,
and the stack reports [180].

### AChainRefusesARowThatReadsTwoRows

Row 2 with `"from": [-1, 0]` makes `build_layer_stack` throw with a message containing "layer
2 'relu'", "the row before it" and "build_model".

### AChainRefusesARowThatSkipsBack

Row 2 with `"from": 0` throws with "layer 2 'relu'" and "the row before it".

### AChainRefusesALaterRowThatReadsTheInput

Row 1 with `"from": "input"` throws with "layer 1 'tanh'" and "the row before it".

### AChainRefusesOutputsThatPublishAnEarlierRow

A two-row document whose `outputs` is `[{"node": 0}]` throws with a message containing
"'outputs'", "last row" and "build_model".

### AChainAcceptsOutputsThatPublishTheLastRow

The same document with `[{"node": 1}]` builds a stack of two layers.

### AChainRefusesARowThatReadsANamedPort

Row 1 with `"from": {"node": 0, "port": "output"}` throws with "layer 1 'tanh'" and "the row
before it". The row it names is the row above; the port is what the chain refuses.

## test_layer_stack.cpp

Tests the runtime layer seam: LayerStack, LayerAdapter and the shape the factories thread.

This file needs no device. A LayerStack is host-lane, and the first property this file pins,
that it names its parameters exactly as Sequential does, is a property of the walk and not of
any memory. The second is the per-sample `Shape` that `NN::Layer::output_shape()` reports
and `NN::build_layer_stack` hands from each layer to the next factory.

The comparison is written as "trace both, walk both, compare the two lists" and not as "check
LayerStack produces these literal names", so it fails if either path changes its naming. A test
against literals would let both drift together and still pass.

### chain_row

`chain_row` returns one `Config::LayerSpec` of a given type and attributes that reads one given
source row.

### hand_built_chain

`hand_built_chain` fills a `Config::ModelSpec` field by field, with no parser involved: the
input shape [6], 6 input features, and the rows `linear` 8, `gelu` and `linear` 4. Row 0 reads
`Config::K_MODEL_INPUT` and each later row reads the row above it.

### NamesItsParametersExactlyAsSequentialDoes

The second `EXPECT_EQ` states the names once as literals too, so the `expected` list that both
walks are compared through cannot become trivially equal by both paths going empty.

### KeepsTheIndexGapAParameterlessActivationLeaves

The gap is the whole reason the naming matches: indices track layer positions, not parameter
positions. A stack that renumbered would produce "1.weight" for the second Linear and load
nothing.

### TracingTwiceDoesNotDuplicateParameters

`Parameter::declare` is idempotent, and the run-time path must inherit that and not reintroduce
the duplicate-on-retrace bug at this layer. Both traces go into one graph, because a
parameter's id names a value in the graph that declared it. Retracing into a second graph would
reuse an index that means something else there.

### AnAdapterBuiltFromAShapeReportsIt

A `LayerAdapter<GELU>` constructed with the shape [4, 5, 3] returns that shape from
`output_shape()`.

### AnAdapterBuiltFromAWidthReportsARankOneShape

A `LayerAdapter<Linear>` constructed with the width 8 reports the shape [8] and
`output_features()` 8. `make_stack` builds its layers through this constructor.

### OutputFeaturesEqualsTheElementCountOfTheOutputShape

Through an `Layer` reference, `output_features()` of an adapter of shape [4, 5, 3] is 60 and
equals `output_shape().num_elements()`. `output_features()` is a non-virtual function of
`Layer` in `include/SushiAI/nn/layer.hpp`, derived from the one virtual.

### ReportsTheLastLayersOutputShape

The stack of `make_stack` reports [4]. With a `GELU` adapter of shape [2, 2, 1] pushed after
it, the stack reports [2, 2, 1] and `output_features()` 4.

### RefusesAnOutputShapeWhenEmpty

`output_shape()` of an empty stack throws with a message containing "no layers" and "output
shape".

### BuildLayerStackThreadsEachOutputShapeIntoTheNextLayer

The spec of `hand_built_chain` builds three layers that report [8], [8] and [4]. The last
layer is cast to `LayerAdapter<Linear>` and its `in_features` is 8, the width the activation
before it reported. The spec never states that number, so it can only come from the shape
`build_layer_stack` hands on.

### BuildLayerStackSeedsTheChainFromTheInputShape

The spec of `hand_built_chain` is given `input_shape` [5] and `input_features` 6, two fields
that disagree. The first layer's `in_features` is 5, so `build_layer_stack` reads the shape.

### BuildLayerStackRefusesAHandBuiltRowWithoutASource

With the `from` list of row 1 cleared, `build_layer_stack` throws with a message containing
"layer 1 'gelu'" and "the row before it".

### OutputShapeOfAnInputIsTheLastLayersOutputShape

`LayerStack::output_shape([6])` of the built chain is [4] and equals the argument-free
`output_shape()`.

### RefusesAnOutputShapeOfAnInputWhenEmpty

`output_shape([6])` of an empty stack throws with a message containing "no layers" and "output
shape".

### AnActivationFactoryKeepsARankThreeInputShape

`NN::make_gelu` given the per-sample shape [4, 4, 3] returns a layer that reports [4, 4, 3]
and 48 features. A factory that reduced its input to a width would report [48].

### ALinearFactoryRefusesARankThreeInputAndNamesFlatten

`NN::make_linear` given [4, 4, 3] throws with a message containing the context, "[4, 4, 3]"
and "flatten". Given [48], the same object builds.

## test_layout.cpp

Tests that every value carries a tensor layout and that the rebuilding passes keep it.

This file needs no device. The IR and its builder are host-lane, which is the point of keeping
shape inference out of the lowering.

### EveryValueIsNhwcUnlessSaidOtherwise

CV-3 fixes one native layout. A value that never mentions layout is NHWC, so a graph built by
code that predates this field is uniform by construction and not by accident.

### FinaliseRejectsAMixedGraphAndNamesTheValue

The failure this check exists to catch is a layer emitting an NCHW value into an NHWC network.
Nothing downstream would throw: the shapes still line up, every kernel still runs, and the
model learns from transposed data.

### LayoutConvertIsAllowedToChangeIt

K_LAYOUT_CONVERT is the one exemption, and the reason the check runs per node and not as a single
sweep over every value.

### DeclareLikeInheritsTheSourceValuesLayout

Fourteen tasks still to come each emit a value and carry its layout forward by hand.
`declare_like` is the one place that does it, so a forgotten carry cannot happen.

### Crossed

Both rebuilding passes copy the value table entry by entry, so that copy is what the `Crossed`
graph tests, and not the pass's own subject. The chain is deliberately plain for the same
reason: a fusion candidate or a precision decision in it would only obscure which field went
missing.

### FusionCarriesTheLayoutTagAcrossItsRebuild

`fuse()` rewrites the node list and copies the value table. A field it forgets to copy is reset
to `add_value`'s default, which is NHWC. A dropped tag is therefore a wrong layout, not a
missing one, and every shape check downstream still agrees with it.

The cost of a dropped tag shows in `Lowering::record_layout_convert`, which decides whether to
relabel a side into [N,H,W,C] by reading these two tags. A genuine NCHW -> NHWC convert that
reads as NHWC -> NHWC applies no transpose and copies transposed data straight through.

### CastInsertionCarriesTheLayoutTagAcrossItsRebuild

Cast insertion is the other rebuilding pass and could make the same omission. Trainer runs both
passes unconditionally, so either one is enough to lose the tag on the default path.

### ARebuiltGraphStillProvesUniform

The consequence is worth stating separately. With every tag reset to one value, `finalise()`
passes for the wrong reason. A graph that is uniform because it was flattened cannot be told
from one that was uniform to begin with, so the proof has to be re-run against tags that
survived.

### TheThreeRulesThatEmitZerosCannotBeReachedWithAnNchwOperand

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

### ZerosLikeCarriesTheTagAndPlainZerosCannot

The three backward rules now depend on this contract. K_ZERO reads nothing, so plain `zeros()` has
no operand to inherit from and its NHWC tag is a default, not an answer. That is harmless for
the K_ZERO node itself, which the per-node proof skips for having no inputs. The consumer that
reads those zeros beside a real operand is the node that would be seen crossing layouts.

### FinaliseStillPropagatesRequiresGrad

The rename must not quietly drop the behaviour the old name described. The assertion is the
same one `test_ir.cpp` made.

## test_leaky_relu.cpp

Tests the LeakyReLU layer's trace, shape inference and slope range.

This file needs no device. `NN::LeakyReLU` in `include/SushiAI/nn/leaky_relu.hpp` holds a
slope and no parameter, and the slope must lie strictly inside (0, 1). The section on
`test_ops.cpp`, under LeakyReluDifferentiatesAtItsOutput, says why a slope of 0 or below is
excluded: the gradient reads the layer's output, which is a substitution only for a positive
slope.

Three `static_assert` lines at file scope require `LeakyReLU` to be a `Module`, not to be a
`ModuleContainer`, and to list no parameter field in `fields()`.

### is_batch_of

`is_batch_of` reports whether one shape is another with a single batch axis in front. It
compares every per-sample axis, so it serves a feature vector and an image alike.

### is_slope_rejected

`is_slope_rejected` traces a layer of the given slope over a [2, 4] input and reports whether
the trace throws `Error`. It uses `Test::capture_error` from `tests/common/test_throws.hpp`,
so an exception of another type does not count as a rejection.

### TheDefaultSlopeIsOneHundredth

A default-constructed layer has slope 0.01.

### ForwardEmitsOneLeakyReluNodeCarryingTheSlope

The trace of a layer with slope 0.2 adds exactly one node, a K_LEAKY_RELU with the input as its
one operand and the result as its one output. Attribute slot 0 of the node holds 0.2.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

A [2, 4] input traces to a [2, 4] value of the input's element type.

### ForwardKeepsTheShapeAndNhwcTagOfAnImageBatch

A [2, 8, 6, 3] input traces to a value of the same shape tagged NHWC.

### ForwardCarriesTheLayoutOfItsInput

The input is converted to NCHW first. The result is tagged NCHW and has the converted value's
shape. The activation is elementwise and reads no axis, so it accepts either layout and
passes the tag on, where the spatial layers refuse NCHW.

### OutputShapeAgreesWithForwardForAFeatureVector

`output_shape([7])` is [7], and the traced shape of a [5, 7] input is a batch of it.

### OutputShapeAgreesWithForwardForAnImage

`output_shape([8, 6, 4])` is [8, 6, 4], and the traced shape of a [3, 8, 6, 4] input is a
batch of it.

### ForwardRejectsASlopeOfZero

A slope of exactly 0 is rejected. The range is open at this end.

### ForwardRejectsASlopeOfOne

A slope of exactly 1 is rejected. The range is open at this end too.

### ForwardRejectsANegativeSlope

A slope of -0.01 is rejected.

### ForwardRejectsASlopeAboveOne

A slope of 1.5 is rejected.

### ForwardRejectsANanSlope

A NaN slope is rejected. NaN compares false against both bounds, so a check written as "throw
when the slope is at or below 0 or at or above 1" lets it through.

### ForwardAcceptsSlopesJustInsideTheOpenRange

The smallest positive normal float32 and 1 minus the float32 epsilon are both accepted. Each
lies inside the range and still differs from its bound after narrowing to float32.

### OutputShapeRejectsASlopeOutsideTheOpenRange

For the five slopes the forward cases reject, `output_shape([4])` throws with a message
containing "LeakyReLU" and "slope". A factory calls `output_shape()` and not `forward()`, so
this is the check that refuses a slope when a model file is read.

### ForwardNamesTheLayerAndTheSlopeItRefuses

`forward()` with slope 1.5 throws with a message containing "LeakyReLU" and "slope".

### ARejectedSlopeEmitsNothing

After `forward()` throws for a slope of 1, the node count and the value count are unchanged.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before a trace. After one the walk visits nothing and
`parameter_elements` is 0.

## test_loss_roots.cpp

Tests which values differentiation accepts as a loss, the seed it emits and the K_STOP_GRADIENT
barrier.

This file needs no device. Every case in `Unit_LossRoots` builds a host-side graph, calls
`Autograd::differentiate` and reads the appended nodes or the error. A root produced by
K_NLL_LOSS is seeded by that operation's own rule. Any other one-element root of a real floating
type whose producer has a rule gets a K_CONSTANT of 1 as its gradient. Most refusal cases also
check that the node count is unchanged, so a refused call leaves the graph as it was.

### Constants

`K_ATTRIBUTE_TOLERANCE` is 1e-12, the error allowed on an attribute a rule copies from the
forward node. `K_TERM_WEIGHT` is 2.0, the factor of the K_SCALE term in `WeightedSum`.
`K_ROOT_WEIGHT` is 0.5, the factor of the K_SCALE that ends a loss in the cases that need one.

### op_names

`op_names` returns the operation names of the nodes from a given index to the end. A case
passes the forward node count, so the list covers the appended nodes.

### count_op

`count_op` counts the nodes of one operation in the whole graph.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. The seed factor defaults to `K_NO_VALUE` and the provenance to null.

### WeightedSum

`WeightedSum` holds the loss `K_TERM_WEIGHT * a + b * c` over three one-element parameters:
a K_SCALE, a K_MUL and the K_ADD of the two. It keeps the ids of the parameters and of the three
results.

### ASumOfTwoTermsEndingInAddReachesEveryParameter

The root is the K_ADD. Four nodes are appended: K_CONSTANT, K_MUL, K_MUL, K_SCALE. The K_CONSTANT is the
seed. The two K_MUL nodes read the seed with `c` and with `b`, and produce the gradients of `b`
and `c`. The K_SCALE reads the seed with the factor `K_TERM_WEIGHT` and produces the gradient of
`a`. All three gradients have the lifetime GRADIENT.

### ASumOfTwoTermsEndingInScaleReachesEveryParameter

The root is a K_SCALE of the sum by `K_ROOT_WEIGHT`. The appended nodes are K_CONSTANT, K_SCALE, K_MUL,
K_MUL, K_SCALE. The first K_SCALE reads the seed with the factor `K_ROOT_WEIGHT`, and the three nodes
after it read that K_SCALE's result, so both terms under the root receive the scaled seed. No
gradient is `K_NO_VALUE`.

### TheSeedIsAConstantOneShapedAndTypedLikeTheRoot

The root is a FLOAT64 [1, 1, 1, 1] value tagged NCHW. The first appended node is a K_CONSTANT
with no input and the value 1 in attribute slot 0, and its result is [1, 1, 1, 1], FLOAT64 and
NCHW. The graph holds one K_CONSTANT, no K_ZERO and no K_EXP.

### TheSeedFactorMultipliesTheSeedOnce

With a seed factor, the appended nodes are K_CONSTANT, K_DYNAMIC_SCALE, K_MUL, K_MUL, K_SCALE. The
K_DYNAMIC_SCALE reads the K_CONSTANT and the factor, the graph holds one K_DYNAMIC_SCALE, and the
three rule nodes read its result.

### ANonScalarRootThrowsNamingItsShape

A [4, 3] root throws with a message containing "needs a scalar to differentiate" and "[4, 3]".

### ALeafRootThrowsNamingTheLeaf

A parameter passed as the root throws with a message containing "it is a leaf" and
"sushiai.leaf".

### ARootThatNeedsNoGradientThrowsNamingTheCause

The root is an K_ADD of two inputs and the requested parameter is not read by it. The message
contains "requires no gradient" and "sushiai.add".

### ARootWhoseProducerHasNoRuleThrowsNamingTheOperation

The root is a K_DYNAMIC_SCALE of a parameter. The message contains "no backward rule" and
"sushiai.dynamic_scale".

### ARootThatIsNotTheFirstResultThrowsNamingTheProducer

The root is the index result of a K_MAXPOOL2D, its second output. The message contains "not the
first result" and "sushiai.maxpool2d".

### ARootThatIsNotRealFloatingThrowsNamingItsType

The root is a K_SCALE of an INT32 parameter. The message contains "element type is int32" and
"sushiai.scale".

### AFusedCrossEntropyRootEmitsNoSeedNode

A two-layer `NN::Sequential` under `NN::CrossEntropyLoss` appends K_EXP, K_SUB, K_SCALE, K_SUM_ROWS,
K_MATMUL, K_MATMUL, K_RELU_BACKWARD, K_SUM_ROWS, K_MATMUL. The graph holds no K_ZERO and no K_CONSTANT: the
fused rule starts from the root's implicit gradient and needs no seed node.

### AnUnfusedNllLossRootEmitsNoSeedNode

K_NLL_LOSS over a K_RELU result appends K_SCALE then K_RELU_BACKWARD. The K_SCALE reads the targets
alone, so no seed node precedes it.

### AnNllLossRootAppliesTheSeedFactorInItsOwnRule

With a seed factor, a K_LOG_SOFTMAX and K_NLL_LOSS pair appends K_EXP, K_SUB, K_SCALE, K_DYNAMIC_SCALE, and
the result of the K_DYNAMIC_SCALE is the gradient of `x`.

### StopGradientEmitsAnIdentityShapedLikeItsOperand

`stop_gradient` of a FLOAT64 NCHW value writes one K_STOP_GRADIENT node that reads the operand.
The result has the operand's shape and is FLOAT64 and NCHW, and `finalise()` accepts the graph.

### StopGradientIsDeclaredAGradientBarrierAndNotDifferentiable

`Ops::K_STOP_GRADIENT` has `blocks_gradient` true and `differentiable` false.
`Ops::is_gradient_barrier` is true for it and false for K_RESHAPE, and its name is
"sushiai.stop_gradient".

### RequiresGradDoesNotCrossStopGradient

After `finalise()`, a parameter requires a gradient and so does a K_SCALE that reads it. The
K_STOP_GRADIENT result and a K_SCALE that reads that result do not.

### StopGradientBlocksAParameterReachableOnlyThroughIt

The loss is `stop_gradient(scale(blocked)) * open`. Differentiation does not throw and appends
K_CONSTANT and K_MUL. The gradient of `blocked` is `K_NO_VALUE`, and the gradient of `open` is the
K_MUL of the seed and the stopped value.

### AValueReadDirectlyAndThroughStopGradientKeepsTheDirectGradient

The loss is `stop_gradient(x) * x`. The appended nodes are K_CONSTANT and K_MUL, the K_MUL reads the
seed and the stopped value, and its result is the gradient of `x`. The graph holds no
K_ACCUMULATE, so the stopped path contributes nothing to the sum.

### ALossReachableOnlyThroughStopGradientThrowsNamingTheBarrier

The root is a K_SCALE of `stop_gradient(x)`. The message contains "requires no gradient" and
"sushiai.scale".

### AStopGradientRootThrowsForRequiringNoGradient

The root is the K_STOP_GRADIENT result. The message contains "requires no gradient" and
"sushiai.stop_gradient".

### AContributionToAStopGradientResultIsDroppedWithoutAskingForARule

The graph is that of StopGradientBlocksAParameterReachableOnlyThroughIt, with `requires_grad`
of the stopped value set by hand after the graph is built. `finalise()` only sets the flag, so
the hand-set value survives it and K_MUL's rule contributes a gradient to the stopped value.
Differentiation does not throw. It appends K_CONSTANT, K_MUL, K_MUL, the gradient of `blocked` is
`K_NO_VALUE`, and no appended node has the K_STOP_GRADIENT node as its origin in the `Provenance`.
The reverse walk steps over the barrier and asks no rule of it.

### TheSeedNodesHaveTheRootProducerAsOrigin

`WeightedSum` is differentiated with a seed factor and a `Provenance`. Five nodes are appended.
The origin of the K_CONSTANT and of the K_DYNAMIC_SCALE is the K_ADD that produces the root. The
origin of the two K_MUL nodes is the forward K_MUL, and the origin of the K_SCALE is the forward
K_SCALE.

### AConstantRootThrowsForRequiringNoGradient

The root is a `constant_like` of a parameter. The message contains "requires no gradient" and
"sushiai.constant".

### AComparisonRootThrowsForRequiringNoGradient

The root is `less_equal(x, bound)` over a parameter. The message contains "requires no
gradient" and "sushiai.less_equal".

### RequiresGradCrossesNoGradientBarrier

After `finalise()`, a K_LESS_EQUAL mask over a parameter, a K_CONSTANT, a K_ZERO and the K_MUL of the
mask with the K_CONSTANT require no gradient. The K_MUL of the mask with the parameter does,
through its second operand.

## test_loss_scaling.cpp

Tests the dynamic loss-scaling rule on LossScaleState, with no device involved.

This file tests the dynamic-scaling rule with no device in the measurement.

LossScaleState is split out of LossScaler so that this file can exist. "The scale doubles after
exactly N clean steps" is a property of the rule, not of a kernel. Asserting it through a
training run would make every one of these cases cost a compile and a replay while proving less.

The control matters as much as the case. A growth test that only ran N clean steps and saw a
doubling would pass against an implementation that doubled after every step, so each growth
assertion in this file is paired with the N-1 run that must not double.

### OneStepShortOfTheIntervalTheScaleDoesNotMove

This case is the control for TheScaleDoublesAfterExactlyTheGrowthInterval. Without it, "the
scale doubled after four clean steps" is equally true of an implementation that doubles after
one.

### AnOverflowRestartsTheRunOfCleanSteps

Three clean steps since the overflow is one short of the interval, so nothing has grown back.
The scale is still the backed-off one.

### BackoffStopsAtTheFloorButStillSkipsTheStep

At the floor the scale cannot go lower. The step is still garbage and must still be thrown
away, or an overflowed gradient reaches the weights the moment the backoff bottoms out.

### ANonDyadicConstantIsRefusedByName

Every constant is a power of two so that scaling and unscaling change only the exponent field.
A scale of 3 would work and would quietly make every gradient in the model carry a rounding
error, so it is refused.

### EveryReachableScaleHasAnExactReciprocal

The loss-scaling design is arranged around this property, and the test asserts it on the rule
itself. Whatever sequence of backoffs and growths the machine takes, the scale is a power of
two, so the reciprocal is exact and the round trip is lossless.

## test_max_pool2d.cpp

Tests the 2-D max pooling layer's trace, shape inference and rejections.

This file needs no device. `NN::MaxPool2d` in `include/SushiAI/nn/max_pool2d.hpp` holds a
`Graph::ConvGeometry` and no parameter. `GraphBuilder::maxpool2d` returns the pooled value and
an INT64 argmax together; the layer returns the first.

Two `static_assert` lines at file scope require `MaxPool2d` to be a `Module` and not a
`ModuleContainer`.

### make_pool

`make_pool` builds a layer from a kernel and a stride per axis and an optional top and left
padding, which default to 0.

### traced_shape

`traced_shape` traces a layer over a batch of the given per-sample shape and returns the
result's shape.

### with_batch

`with_batch` prepends a batch extent to a per-sample [H, W, C] shape.

### ForwardHalvesTheSpatialExtentsOfAnEvenInput

A 2x2 stride-2 pool traces a [2, 8, 6, 3] input to [2, 4, 3, 3].

### ForwardTagsItsResultNhwc

The traced result is rank 4 and tagged NHWC.

### ForwardReturnsThePooledValueNotTheArgmax

The result has the input's element type, and that type is not INT64. A layer that returned
the second result of the node would hand the next layer the index buffer.

### OutputShapeOfAnUnpaddedSquareWindow

`output_shape([8, 6, 3])` of a 2x2 stride-2 pool is [4, 3, 3].

### OutputShapeAgreesWithForwardForASquareWindow

For the same pool and sample, the traced shape of a batch of two equals `output_shape()` with
the batch extent in front.

### OutputShapeAgreesWithForwardForAPaddedAnisotropicWindow

A 3x2 window with stride 2 by 1 and a top padding of 1 maps [5, 7, 4] to [3, 6, 4], and the
traced shape agrees.

### OutputShapeAgreesWithForwardWhenTheStrideLeavesARemainder

A 3x3 stride-2 window maps [8, 10, 1] to [3, 4, 1], and the traced shape agrees. Neither
extent divides evenly, so the case covers the rounding down.

### OutputShapeAgreesWithForwardForADilatedAnisotropicWindow

A 2x2 unit-stride window with dilations 2 by 3 maps [7, 9, 2] to [5, 6, 2], and the traced
shape agrees.

### ForwardRejectsAnInputThatIsNotRankFour

`forward()` throws on a [2, 16] input.

### ForwardRejectsAWindowThatDoesNotFit

`forward()` throws for a 5x5 window over a [2, 4, 4, 3] input.

### ForwardRejectsAnNchwInput

A [2, 8, 8, 3] input is converted to NCHW, and `forward()` throws on the converted value.

### ForwardNamesTheLayoutOfAnNchwInputNarrowerThanTheWindow

A [1, 8, 8, 2] input converted to NCHW has 2 on the axis an NHWC reader takes for the height,
which a 3x3 window does not fit. The message contains "NHWC", so the layout check runs before
the window is fitted.

### ForwardRejectsAZeroStride

`forward()` with a vertical stride of 0 throws with a message containing "MaxPool2d" and
"stride 0,2".

### ForwardRejectsANegativePadding

`forward()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAZeroStride

`output_shape()` throws when the vertical stride is 0.

### OutputShapeNamesAZeroStride

The message for the same layer contains "MaxPool2d" and "stride 0,2", so it reports the
stride and not a window that does not fit.

### OutputShapeRejectsANegativePadding

`output_shape()` with a left padding of -1 throws with a message containing "MaxPool2d" and
"padding 0,-1".

### OutputShapeRejectsAFeatureVector

`output_shape([16])` throws.

### OutputShapeRejectsABatchedShape

`output_shape([2, 8, 8, 3])` throws.

### OutputShapeRejectsAWindowTallerThanTheInput

A 5x2 window over [4, 8, 3] throws. The window fits the width, so the height check refuses.

### OutputShapeRejectsAWindowWiderThanTheInput

A 2x5 window over [8, 4, 3] throws. The window fits the height, so the width check refuses.

### TheParameterWalkVisitsNothing

`parameter_count` is 0 before and after a trace.

## test_memory_plan.cpp

Tests that the memory planner never shares a byte between two values live together.

This file needs no device: no engine, no allocator, no SYCL. The memory planner is a pure
function of the IR, which is the reason the IR is a host-lane structure, and every case is a
synthetic graph and a table of offsets.

The contract is one sentence: two values whose live ranges intersect never share a byte, and
two whose live ranges do not are free to. The first half is checked exhaustively over every
pair. The liveness it is checked against is computed in the test file by a breadth-first search,
`descendants()`, and not by asking the planner what it thinks. A planner checked against its own
reachability code would agree with itself whatever it did.

The individual cases are for the second half, and the diamond in
TwoUnorderedBranchesNeverShareBytes is the one that matters. It is the shape where a planner
that overlaps disjoint [first_use, last_use] intervals on a linearised schedule gets a wrong
answer that no test written from the linear model would catch.

The cases fall into six groups, in file order:

- degenerate graphs
- the two shapes that decide whether the pass is right
- lifetimes
- aliasing
- mixed dtypes
- knobs, refusals and malformed input

### AChainPingPongsBetweenTwoBuffers

The graph is x -> a -> b -> c -> d. Nothing branches, so `a` is finished before `c` starts and
`b` before `d`. Four activations need exactly two buffers, which is the most any planner can do
with a chain.

### TwoUnorderedBranchesNeverShareBytes

This is the case the planner is judged by. Linearised, the two branches have disjoint
[first, last] intervals: a1 dies at node 2 and b1 is born at node 3. An interval planner
overlaps them and is wrong. Nothing in the graph orders node 2 before node 3 and the runtime may
run the branches concurrently, so the sharing would corrupt the result and also serialise the
parallelism it was meant to leave free.

### OnlyTheNonPersistentClassesArePacked

The graph holds all six lifetimes. The four persistent ones live for the whole run, so packing
them would save nothing and would merge the dependency-tracker shards that matter most. Only
ACTIVATION and SCRATCH reach an arena, and they reach different ones.

### AValueNothingWritesKeepsItsOwnAllocation

An ACTIVATION with no producer is written from outside the step and never rewritten, so its
bytes are live for the whole run whatever its class says. Handing them away would survive
exactly one replay.

### AnAccumulateChainIsOneBufferWithOneLiveRange

K_ACCUMULATE's result is its accumulator's storage. Planning the two separately would be silently
wrong and not merely wasteful: the planner would think the accumulator died at the K_ACCUMULATE
node and hand its bytes to `sink`, which is produced by the node still reading them.

### EveryStartIsAWholeNumberOfItsOwnElements

Alignment is turned down to 1 so that the element-width floor is the only thing left doing the
work. A 20-byte float32 buffer at offset 0 leaves the next free byte at 20, which is not a whole
number of float64 elements. The float64 buffer therefore has to start at 24, or its
storage_offset would not be representable at all.

### TooManyNodesDeclinesOverlapRatherThanApproximatingIt

The refusal is the point. A planner that quietly fell back to overlapping linear intervals
would produce a plan that is wrong in the way TwoUnorderedBranchesNeverShareBytes describes, and
would report a saving for it.

### AGraphThatIsNotInTopologicalOrderThrows

GraphBuilder cannot produce this graph, because it always emits a producer before anything can
read it. The whole pass rests on the property all the same, so the test checks it. A closure
built over a back edge would silently miss ordering and the plan would overlap things it must
not.

### ThePlanIsDeterministicForTheSameGraph

The offsets are byte-identical across two builds. A plan that shuffled would move every tensor
and take the library's bit-for-bit replay guarantee with it.

### TheTwoResultsOfOneNodeAreNeverGivenTheSameBytes

`FusedForm::GemmBiasGelu` writes two values from one node: its result and the pre-activation
its epilogue spills. They have the same shape, they are born at the same instant, and the
planner packs by liveness. "One died before the other was written" is the mistake available to
it, and taking it would have the GELU computed from its own output with nothing to show for it.

Nothing special prevents it, which is why it is worth pinning. `precedes()` asks whether every
use of the earlier buffer strictly reaches the later one's first write, and a node does not
reach itself. The planner therefore refuses both directions and the two values conflict. That is
an inference from an existing rule and not a rule of its own, so a future rewrite of
`precedes()` could take it away silently.

## test_model_graph.cpp

Tests ModelGraph: what add and output refuse, what a trace emits, and the state names.

This file needs no device. `NN::ModelGraph` in `include/SushiAI/nn/model_graph.hpp` holds
named nodes in the order they were added. `add` takes a name, a layer or an `NN::Node` and the
`NodeRef`s the node reads, and returns the `NodeRef` of the new node. `output` publishes a ref
under a name. `trace` returns the published values as an `NN::ModelOutputs`, and `forward`
returns the one published value. `add`, `output`, `port`, `output_ports` and `trace` return a
`Status` or a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelGraph` is a `TraceModule`,
is not a `Module`, and is detected by `Detail::HasStateWalk`.

### OpList

`OpList` is a vector of the operation of every node of a graph, in order.

### TwoPortNode

`TwoPortNode` is an `NN::Node` with one input and two ports. `forward` appends the ReLU and
then the GELU of its input. The one-argument constructor names the ports "low" and "high", and
the three-argument one takes the two names.

### BranchingModel

`BranchingModel` holds a `ModelGraph` and the refs `stem`, `up` and `mix` of three of its
nodes.

### build_branching_model

`build_branching_model` builds five nodes over an [8, 8, 3] sample and publishes nothing:
"stem", a same-padded `Conv2d(3, 4)` of kernel 3 and stride 2 on the input; "down", a
`Conv2d(4, 8)` of the same window on the stem; "up", an `Upsample2d{2}` on that; "merge", a
`Concat` of up and stem; and "mix", a pointwise `Conv2d(12, 4)` on the merge.

### ops_of

`ops_of` returns the `OpList` of a graph.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields under a prefix as
"name:extents" strings.

### add_relu_declared_as

`add_relu_declared_as` adds a `LayerNode` over a `LayerAdapter<ReLU>` constructed with a given
shape. The node declares that shape on its port whatever its input is.

### RefsNameTheInputAndTheNodesInTheOrderAdded

`model.input()` reports `is_input()` and port 0. The refs of stem, up and mix report the nodes
0, 2 and 4, mix reports port 0, stem and up compare unequal, and `size()` is 5.

### ShapeOfReturnsTheDeclaredPerSampleShapes

`shape_of` is [8, 8, 3] for the input, [4, 4, 4] for stem, [4, 4, 8] for up and [4, 4, 4] for
mix. Node 3, the merge, declares [4, 4, 12] on its port.

### BranchAndMergeTracesTheNodeSequenceOfTheHandWrittenTrace

The model's `forward` under an evaluation trace is compared with the same five layers traced
by hand through a `GraphBuilder` in a second graph. The two graphs hold the same operations in
the same order, and both results are [2, 4, 4, 4]. The sequence holds one K_UPSAMPLE2D and one
K_CONCAT_CHANNELS.

### MergeEmitsOneConcatChannelsReadingBothBranches

With stem, up and mix published, the traced graph holds one K_CONCAT_CHANNELS node. Its inputs
are the published value of up and then that of stem, and its output is [2, 4, 4, 12].

### TraceReturnsEveryPublishedOutputByName

With "features" and "skip" published in that order, `trace` returns two entries in that order.
`at` returns a different id for each name, and both values are [2, 4, 4, 4].

### ForwardReturnsTheSinglePublishedOutput

A model of one ReLU node whose result is published traces one K_RELU node. That node reads the
input, and its output is the id `forward` returned.

### ForwardFailsWhenTwoOutputsArePublished

With two outputs published, `forward` returns its input id and records `InvalidDescription` on
the trace with a message containing "exactly one model output", "features" and "skip". The
graph is marked refused.

### OutputShapeIsTheShapeOfTheSinglePublishedOutput

`output_shape([8, 8, 3])` is an empty `Shape` while nothing is published, [4, 4, 8] once up is
the one published output, and an empty `Shape` again once a second output is published.

### AGraphIsANodeOfAnotherGraph

A `ModelGraph` holding a `Linear{6, 8}` named "dense" is added to an outer graph as the node
"inner", with a ReLU after it. `shape_of` of the nested node is [8], the one output is
[4, 8], and the outer walk yields "inner.dense.weight:[8, 6]" and "inner.dense.bias:[8]".

### ANestedGraphThatPublishesTwoOutputsFailsTheOuterTrace

An inner graph that publishes "first" and "second" is added as a node. The outer `trace` fails
with `InvalidDescription` and a message containing "node 'inner'", "exactly one model output",
"first" and "second".

### ASequentialAndALayerStackAreNodesOfAGraph

A `Sequential{Linear{6, 8}, ReLU}` is added as "head" and a `LayerStack` of a `Linear{8, 4}`
and a GELU as "tail". `shape_of` is [8] and [4], the output is [4, 4], and the walk yields
"head.0.weight:[8, 6]", "head.0.bias:[8]", "tail.0.weight:[4, 8]" and "tail.0.bias:[4]".

### TheInputCanBePublishedAsAnOutput

A model with no node that publishes its input returns the input id from `forward` and emits no
node.

### WalkNamesStateByNodeNameInTheOrderAdded

After a trace of the branching model, the walk under "model" yields "model.stem.weight:[4, 3,
3, 3]", "model.down.weight:[8, 3, 3, 4]" and "model.mix.weight:[4, 1, 1, 12]", and
`NN::buffer_count` is 0. The upsample and the concat contribute no name.

### ATrainingTraceReachesAModalNodeAndTheWalkNamesItsBuffers

A model of one `BatchNorm2d(3)` node named "bn", traced under a training trace, emits one
K_BATCHNORM2D node and leaves two updates in the list. The parameter walk under "model" yields
"model.bn.weight:[3]" and "model.bn.bias:[3]", and `for_each_buffer` yields
"model.bn.running_mean" and "model.bn.running_var".

### AChainNamedByIndexYieldsTheNamesOfALayerStack

A model whose nodes are named "0", "1" and "2" holds a `Linear{6, 8}`, a GELU and a
`Linear{8, 4}`. A `LayerStack` holds the same three layers. After each is traced into its own
graph, the two parameter walks are equal, the two graphs hold the same operation sequence, and
the names are "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]" and "2.bias:[4]".

### NestsInASequentialUnderItsIndex

A `Sequential<ModelGraph, ReLU>` traces to a [4, 8] value whose last node is a K_RELU. Its walk
yields "0.dense.weight:[8, 6]" and "0.dense.bias:[8]".

### AddRejectsADuplicateName

A second node named "act" fails with `InvalidDescription` and a message containing "already
has a node named 'act'" and "index 0". `size()` stays 1.

### AddRejectsAnEmptyName

The name "" fails with "needs a name" and `size()` stays 0.

### AddRejectsANameWithADot

The name "a.conv" fails with "'a.conv'" and "contains '.'" through the layer overload, and
"b.act" fails the same way through the `NN::Node` overload. `size()` stays 0.

### AddReportsARefusedNameBeforeTheLayerSeesItsSource

A `Conv2d` added under a taken name, or under an empty one, on a [6] source fails with the
message about the name. The convolution would refuse a [6] source, and that failure is not the
one returned. `size()` stays 1.

### AddReturnsTheFailureOfALayerThatRefusesItsSource

`add("stem", Conv2d::same_padded(3, 4, 3, 2), input)` on a [6] sample fails with a message
containing "node 'stem'", "Conv2d(3, 4)" and "[6]", and `size()` stays 0.

### AddRejectsAnUnknownSource

A ref to node 1 of another graph, used as the source in a graph of one node, fails with a
message containing "node 'late'", "source node 1" and "holds 1 nodes". The `NN::Node` overload
fails with "node 'later'" and "source node 1". `size()` stays 1.

### AddRejectsASourcePortTheNodeDoesNotHave

A ref to port 1 of a two-port node of another graph, used on a graph whose node 0 has one
port, fails with a message containing "node 'late'", "'only' has 1 ports" and "port 1".

### PortSelectsANamedPortThatALaterNodeReads

`port(pair, "high")` returns a ref to node 0, port 1, and `port(high, "low")` returns a ref
equal to `pair`. A ReLU added on `high` and published traces three nodes. Node 1 is the GELU,
and node 2 reads node 1's output.

### PortRejectsAnUnknownNameAndTheInput

`port(pair, "middle")` fails with a message containing "node 'pair'", "'middle'" and "low,
high". `port(input, "low")` fails with "the model input has no port named 'low'".

### OutputPortsPublishesEveryPortUnderItsOwnName

`output_ports(pair)` publishes two outputs. `at("low")` is the output of graph node 0 and
`at("high")` that of graph node 1.

### OutputPortsPublishesNothingWhenAPortNameIsTaken

With "high" already published, `output_ports(pair)` fails with "port 'high' of node 'pair'",
and `output_ports(input)` fails with "the model input has no ports to publish". A trace then
returns one output, so the failed call did not publish "low" either.

### OutputRejectsATakenNameAnEmptyNameAndAnUnknownSource

A second `output("echo", ...)` fails with "already publishes an output named 'echo'". The name
"" fails with "needs a name". A ref to a node of another graph fails with "output 'far'",
"source node 0" and "holds 0 nodes".

### TraceRejectsADeclaredShapeThatDisagreesWithTheTracedOne

A node that declares [5] and traces a ReLU of a [4, 6] value fails the trace with
`InvalidDescription` and a message containing "node 'narrow'", "port 'output'", "[5]" and
"[4, 6]".

### TraceRejectsAnInputOfAnotherSampleShape

A model over a [6] sample traced with a [4, 7] value fails with "the model input", "[6]" and
"[4, 7]". Traced with a [6] value, which has no batch axis, it fails with "the model input"
and "[6]". The graph gains no node.

### TraceRejectsAModelThatPublishesNoOutput

A model with one node and no published output fails the trace with "publishes no output", and
the graph gains no node.

### AddRejectsASourceCountTheNodeDoesNotRead

A `Concat` built over two shapes and added with one source fails with a message containing
"node 'merge'", "reads 2 values" and "given is 1". A `TwoPortNode` added with two sources fails
with "node 'split'", "reads 1 values" and "given is 2". `size()` stays 0.

### AMovedGraphKeepsItsRefsAndLeavesAnEmptyUsableGraph

A graph move-constructed from the branching model has `size()` 5, resolves the ref `up` to
[4, 4, 8] and accepts `output("features", mix)`. The graph moved from has `size()` 0. A ReLU
added to it and published traces one node whose output is the published value.

### AccessorsAreFatalOutOfRange

The case is in the suite `Unit_ModelGraphDeathTest`. On an empty graph, `shape_of` of a ref to
another graph's node stops the process with "shape_of" and "source node 0", and `node(0)`
stops it with "ModelGraph::node" and "node 0".

### AddIsFatalOnANullNode

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a null `unique_ptr<NN::Node>`
stops the process with a report containing "null node" and "'ghost'".

### AddIsFatalOnANodeWithTwoPortsOfOneName

The case is in the suite `Unit_ModelGraphDeathTest`. `add` given a `TwoPortNode` whose ports
are both named "low" stops the process with a report containing "'pair'" and "two ports named
'low'".

## test_model_outputs.cpp

Tests ModelOutputs: publishing named values and reading them back.

This file needs no device. `NN::ModelOutputs` in `include/SushiAI/nn/model_outputs.hpp` is
the set of named `ValueId`s one trace of a model publishes. `add()` returns a `Status`, and
`at()` and `only()` return a `Result`.

Three `static_assert` lines at the top of the file hold that `ModelOutputs` is final,
copy-constructible and nothrow move-constructible.

### boxes_and_scores

`boxes_and_scores` returns a set holding "boxes" as value 4 and "scores" as value 9, added in
that order.

### add_without_value

`add_without_value` adds "boxes" with `Graph::K_NO_VALUE`. It is the statement of the death
test.

### StartsEmpty

A new set's `all()` is empty.

### AllKeepsTheOrderOfAdd

After adding "scores", "boxes" and "features", `all()` holds those three entries in that order
with their ids 9, 4 and 0. The order is the order of the calls, not of the names or the ids.

### AddRejectsADuplicateName

A second `add("boxes", ...)` returns `InvalidDescription` with a message containing "'boxes'"
and "boxes, scores". The set still holds two entries and `at("boxes")` is still 4.

### AddRejectsAnEmptyName

`add("", ...)` returns `InvalidDescription` with "needs a name" and adds nothing.

### AddStopsOnNoValue

The case is in the suite `Unit_ModelOutputsDeathTest`. `add("boxes", K_NO_VALUE)` stops the
process with a report containing "'boxes'" and "no value".

### AtReturnsTheNamedValue

`at("boxes")` is 4 and `at("scores")` is 9.

### AtDoesNotMatchAPrefix

`at("box")` and `at("boxes.")` both fail with `InvalidDescription` and the name that was asked
for.

### AtOnAMissingNameListsTheNames

`at("masks")` fails with a message containing "'masks'" and "boxes, scores".

### AtOnAnEmptySetSaysNone

`at("boxes")` on an empty set fails with a message containing "'boxes'" and "none".

### OnlyReturnsTheSingleValue

`only()` on a set of one entry returns that entry's id, 12.

### OnlyFailsOnZero

`only()` on an empty set fails with a message containing "exactly one", "has 0" and "none".

### OnlyFailsOnTwo

`only()` on a set of two fails with a message containing "exactly one", "has 2" and "boxes,
scores".

## test_module.cpp

Tests Parameter across graphs and the state walk over layers, containers and fields.

This file needs no device. The layer library is host-lane, so the parameter walk, which decides
whether the optimizer sees a layer at all, is testable without hardware.

The file holds six groups: the walk over the library's own layers and containers, the
tutorial's custom layer, the run-time arm of the walk, the per-sample output shape of `Linear`
and the four activations of `include/SushiAI/nn/layers.hpp`, a parameter traced into more than
one graph, and the state walk over parameters, buffers and children.

### AnUntracedLayerYieldsNoParameters

Parameters gain a shape and an id when `forward()` first runs, so the walk over an untraced
model is legitimately empty and not full of blanks.

### SequentialNamesParametersByLayerIndex

The naming follows PyTorch's convention: the index is the layer's position, so a parameterless
activation leaves a gap and "2.weight" still names the third layer. A checkpoint diff against a
reference stays a diff.

### TracingTwiceDoesNotDuplicateParameters

The second trace emits nodes but declares no new parameters, so the value count grows by
exactly the two intermediates.

### The tutorial's custom layer

`MyOffset` is the custom layer of `docs/getting_started/TUTORIAL.md` §6, compiled.

A tutorial whose code sample does not compile is worse than no tutorial, and prose cannot be
made to fail a build. So the sample lives in this file as well, and the two must be kept
identical: if you edit one, edit the other.

TheTutorialsCustomLayerTracesAndIsWalkable and TheTutorialIsRightThatMulWouldNotBroadcast also
pin the two claims §6 makes about the sample: that bias_add is required and is not a stylistic
choice, and that the two ways of getting `fields()` wrong fail differently.

### TheTutorialsCustomLayerTracesAndIsWalkable

The test checks the role the tutorial gives the parameter, because the role decides the
initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.

### TheTutorialIsRightThatMulWouldNotBroadcast

§6 says bias_add is required and not merely preferred, because `mul` goes through
`require_same` and this builder has no implicit broadcast anywhere. That is the whole reason the
sample reads the way it does, so the test asserts it and does not leave it as a remark.

### The run-time arm of the walk

`WalkableStub` exercises the third arm of the walk: a layer whose parameters are enumerated at
run time.

The stub is deliberately not a LayerStack. This file tests the walk, and the walk's contract is
`walk_parameters(prefix, ParameterVisitor)`. Anything that satisfies it is enumerated through
it, whether or not it is one of the library's own types. Testing the walk against LayerStack
would tie its test to a type that arrived two commits later.

### AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther

The two arms compose: the container supplies "0.", the walkable child supplies the leaf names,
and nothing about the join changed.

### TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt

`include/SushiAI/nn/layers.hpp` asserts this for Linear and Sequential at build time. The test
restates it so that the property is visible in the test that covers the walk, and so that it
names the detector and not the arm's consequences.

### Per-sample output shapes

Every layer type has `Shape output_shape(const Shape&) const`, which maps the per-sample shape
that enters the layer to the one that leaves it. A layer factory asks the layer for that shape
and does not restate the rule. The six cases below hold `Linear`, `ReLU`, `GELU`, `Tanh` and
`Sigmoid` to it, and compare it with what `forward()` traces.

### traced_shape

The file-local `traced_shape` traces a layer over a [2, features] input and returns the
result's shape.

### LinearReportsItsOutFeaturesAsThePerSampleOutputShape

`Linear{4, 3}` maps [4] to [3].

### LinearsOutputShapeAgreesWithItsTrace

The traced shape of `Linear{4, 3}` over a batch of two is [2, n], where n is the one extent
of `output_shape([4])`.

### LinearRefusesAPerSampleShapeOfTheWrongRank

`output_shape([4, 2, 1])` throws with a message containing "Linear(4, 3)", "[4, 2, 1]" and
"[4]". The first extent equals `in_features`, so only the rank check can refuse this shape.

### LinearRefusesAPerSampleShapeOfTheWrongWidth

`output_shape([7])` throws with a message containing "Linear(4, 3)", "[7]" and "[4]".

### AnActivationReportsItsInputShapeAsItsOutputShape

For the shapes [5] and [4, 4, 3], each of the four activations returns the shape it was given.
The rank-3 shape is the one a convolution hands on.

### AnActivationsOutputShapeAgreesWithItsTrace

The traced shape of each activation over a [2, 5] input is [2, n], where n is the one extent
of its `output_shape([5])`.

### A parameter across graphs

The cases from AParameterTracedIntoTwoGraphsHasALeafInEach to ARoleDeclareFillsByItsRole cover
`NN::Parameter` in `include/SushiAI/nn/module.hpp`. A parameter holds a `Graph::StateId` and
no graph. `declare()` adds a leaf to the graph of the builder it is given and binds the leaf to
the state there, so one parameter has one leaf in each graph it was traced into. `id_in(graph)`
returns that graph's leaf, and `id()` returns the leaf of the most recent `declare()`.

The file-local `K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is
stored and read back unchanged.

### declare_weight

`declare_weight` declares a [2, 3] parameter of role `WEIGHT` named "weight" through a given
builder and returns its leaf.

### AParameterTracedIntoTwoGraphsHasALeafInEach

The first graph holds a padding input, so the parameter's leaf there is value 1, and in the
second graph it is value 0. `id_in` returns each graph's own leaf, `state_leaf` of the
parameter's state returns the same in each graph, and the second leaf has the lifetime
`PARAMETER`.

### AParameterHasNoLeafInAGraphItWasNotTracedInto

Before any `declare()`, `id_in` is `K_NO_VALUE` and `state()` is `K_NO_STATE`. After one
`declare()` the state is set, and `id_in` of a graph that was never traced into is still
`K_NO_VALUE`.

### ASecondDeclareInOneGraphReturnsTheFirstLeaf

A second `declare()` through the same builder returns the first leaf. The graph's value count
does not grow and `state_leaves()` holds one pair.

### ADeclareWithAnotherShapeIsRefused

After a [2, 3] declaration, a `declare()` with [4, 3] throws with a message containing
"'weight'", "[2, 3]" and "[4, 3]", through the same builder and through a builder over a second
graph. The second graph holds no value and the parameter's shape stays [2, 3].

### ADeclareWithAnotherElementTypeIsRefused

After a FLOAT32 declaration, a `declare()` with FLOAT64 throws with a message containing
"'weight'" and both type names, in the same graph and in a second one. The second graph holds
no value and the parameter's type stays FLOAT32.

### ALaterDeclareKeepsTheFirstDescription

A parameter declared as a `SCALE` with a constant fill of 1.5 and fans of 4 is declared into a
second graph as a `WEIGHT` with a fan-scaled fill and fans of 9. Its fill, role and both fans
are still the first ones, and it has a leaf in the second graph.

### ACopiedLayerHasANewStateId

A copy of a traced `Linear` has a weight whose state is `K_NO_STATE` and whose `id_in(graph)`
is `K_NO_VALUE`. `declared()` is true and `id()` is the original's leaf, because both are copied
with the description. Once the copy is traced, its `id()` equals its own `id_in(graph)`, its
state differs from the original's, and its weight and bias leaves differ from the original's.

### AMovedLayerKeepsItsStateId

A `Linear` move-constructed from a traced one has the same weight state and the same weight
leaf in the graph. Tracing it again adds two values, the two activations, and no parameter
leaf.

### IdReturnsTheLeafOfTheMostRecentDeclare

`id()` is `K_NO_VALUE` before any `declare()`, the first graph's leaf after the first, and the
second graph's leaf after the second. The second graph holds a padding input, so the two leaves
have different ids.

### IdFollowsADeclareBackIntoAnEarlierGraph

After declarations into a first and a second graph, `id()` is the second leaf. A `declare()`
through the first builder again returns the first leaf, and `id()` is then the first leaf.

### ARoleDeclareRecordsTheFillOfItsRole

A `declare()` that states a role and no fill records a fan-scaled fill for a `WEIGHT` and a
constant fill of 0.0 for a `BIAS`.

### AnInitDeclareRecordsTheStatedFillAndTheStatedRole

A `declare()` that states both keeps both: a `BIAS` with a constant fill of 1.5 and a fan-in
of 4, and a `WEIGHT` with a fan-scaled fill. The bias's leaf requires a gradient.

### ARoleDeclareFillsByItsRole

The three roles give three fills: fan-scaled for `WEIGHT`, a constant 0.0 for `BIAS` and a
constant 1.0 for `SCALE`.

### The state walk

The cases from WalkStateYieldsTheThreeFieldKindsInDeclarationOrder on cover `NN::walk_state`
and the two walks derived from it. A `fields()` list holds three kinds of entry: `NN::field`
for a `Parameter`, `NN::buffer` for a `Buffer` and `NN::child` for a member that is itself
walked. `walk_state` hands each parameter and each defined buffer to an `NN::StateVisitor`.
`for_each_parameter` and `for_each_buffer` each yield one kind.

### StatePair

`StatePair` lists a parameter "scale" and a buffer "mean". `describe()` declares the parameter
as a [4] `SCALE` and defines the buffer as [4] zeros.

### MixedState

`MixedState` lists, in this order, a buffer "count", a `StatePair` child "pair", a parameter
"offset" and a buffer "unset". `describe()` defines "count" as a [1] INT64 buffer, describes
the pair and declares "offset" as a [4] `BIAS`. It leaves "unset" undefined.

### walk_all

`walk_all` collects what `walk_state` yields under a prefix as "parameter name:extents" and
"buffer name:extents" strings.

### buffer_names

`buffer_names` collects the names `NN::for_each_buffer` yields.

### WalkStateYieldsTheThreeFieldKindsInDeclarationOrder

The walk of a described `MixedState` yields "buffer count:[1]", "parameter pair.scale:[4]",
"buffer pair.mean:[4]" and "parameter offset:[4]". Parameters and buffers interleave in the
order `fields()` lists them.

### WalkStateJoinsItsPrefixToEveryName

Under the prefix "model.3" the same four entries are named "model.3.count",
"model.3.pair.scale", "model.3.pair.mean" and "model.3.offset".

### ForEachParameterNeverYieldsABuffer

`for_each_parameter` yields "pair.scale:[4]" and "offset:[4]". `parameter_count` is 2 and
`parameter_elements` is 8.

### ForEachBufferNeverYieldsAParameter

`for_each_buffer` yields "count" and "pair.mean", and `buffer_count` is 2.

### AnUndefinedBufferIsSkipped

Before `describe()` no buffer is yielded. After it two are. Once `define()` is called on
"unset", the walk yields "count", "pair.mean" and "unset".

### ADefinedBufferIsWalkedWithoutALeaf

The buffer "count" has no leaf in the graph, `id_in` is `K_NO_VALUE`, and it is one of the two
buffers counted.

### SelfWalkingStub

`SelfWalkingStub` has a member `walk_state(prefix, visitor)` and no `fields()`. It yields its
buffer "level" before its parameter "gain".

### StubPair

`StubPair` lists a `SelfWalkingStub` as the child "own" and a `WalkableStub` as the child
"legacy".

### ATypesOwnWalkStateIsTheWalk

`Detail::HasStateWalk` is true for `SelfWalkingStub` and false for `Linear`. The stub's walk
under "head" yields "buffer head.level:[3]" and then "parameter head.gain:[3]", the order its
member function states. `for_each_parameter` yields "gain:[3]" and `for_each_buffer` yields
"level".

### AChildFieldNamesEveryKindOfWalkableChild

The walk of a `StubPair` yields "buffer own.level:[3]", "parameter own.gain:[3]", "parameter
legacy.weight:[2, 3]" and "parameter legacy.bias:[2]". A `child` entry prefixes a type with its
own `walk_state` and a type with `walk_parameters` alike.

### AWalkParametersTypeYieldsNoBuffer

The walk of a `WalkableStub` yields its weight and its bias as parameters, and `buffer_count`
is 0.

### ALayerAdapterYieldsItsLayersParametersUnderThePrefix

A traced `LayerAdapter<Linear>` walked under the prefix "2" yields "parameter 2.weight:[3, 4]"
and "parameter 2.bias:[3]", and `buffer_count` is 0.

### TwoFieldsWithOneNameAreDetected

`Detail::has_unique_names` is checked in four `static_assert` lines. It is true for
`MixedState::fields()` and for an empty tuple. It is false for two `field` entries both named
"weight", and for a `buffer` and a `child` both named "pair". At run time it is true for
`Linear::fields()`. `walk_state` asserts the same predicate.

### ASequentialOfModulesKeepsItsBuilderForward

Three `static_assert` lines hold that `Sequential<Linear, ReLU>` is a `Module`, that it stays
one when nested in another `Sequential`, and that its `forward` over a `GraphBuilder` returns
a `Graph::ValueId`. A nested model of `Linear{4, 8}`, `ReLU` and `Linear{8, 3}` traces to
[2, 3] through the builder and walks as "0.0.weight:[8, 4]", "0.0.bias:[8]", "1.weight:[3, 8]"
and "1.bias:[3]".

## test_module_list.cpp

Tests ModuleList: its storage, its index names and its place in the state walk.

This file needs no device. `NN::ModuleList<M>` in `include/SushiAI/nn/module_list.hpp` holds
children of one type whose count is known at run time.

Three `static_assert` lines at the top of the file hold that `ModuleList<Linear>` is a
`ModuleContainer` and that neither `Detail::HasStateWalk` nor `Detail::HasParameterWalk`
detects it, so the walk reaches it through `visit_children`.

### Stage

`Stage` lists a `ModuleList<Linear>` under the child name "blocks" and a `Linear{4, 2}` under
"head".

### Counter

`Counter` holds one `NN::Buffer` listed as "count", so a list of it carries state that is not
a parameter.

### parameter_names

`parameter_names` collects what `NN::for_each_parameter` yields as "name:extents" strings.

### read_past_the_end

`read_past_the_end` builds a list of two layers and reads `list[2]`. It is the statement of
the death test.

### ANewListIsEmpty

A new list has size 0, `visit_children` calls its visitor zero times, and `begin()` equals
`end()`.

### PushBackAppendsInOrder

After two `push_back` calls the size is 2, element 0 has 8 output features and element 1
has 3. Indexing a const reference to the list reads element 1's 8 input features.

### IterationFollowsIndexOrder

A range-for over three layers reads the output widths 8, 3 and 1.

### VisitChildrenPairsEachChildWithItsIndex

`visit_children` over two layers passes the indices 0 and 1 with the addresses of `list[0]` and
`list[1]`.

### TheWalkNamesChildrenByIndex

Two traced layers, the second without a bias, walk as "0.weight:[8, 4]", "0.bias:[8]" and
"1.weight:[3, 8]", and `parameter_count` is 3.

### AListUnderAChildFieldIsNamedByFieldThenIndex

A traced `Stage` with two blocks walks as "blocks.0.weight", "blocks.0.bias",
"blocks.1.weight", "blocks.1.bias", "head.weight" and "head.bias", each with its extents.

### TheWalkNamesBuffersByIndex

Of three `Counter` children, 0 and 2 have their buffer defined. `for_each_buffer` yields
"0.count" and "2.count": child 1 yields nothing and still keeps its index. `parameter_count`
is 0.

### GrowingTheListKeepsATracedChildsStateIdentity

The first child is traced, then 64 more layers are appended, which moves a one-element
vector's storage at least once. The first child's weight has the same state and the same leaf
in the graph afterwards, and its bias's `id_in(graph)` equals its `id()`.

### AnIndexPastTheEndIsFatal

The case is in the suite `Unit_ModuleListDeathTest`. `list[2]` on a list of two stops the
process with a report containing "index 2" and "holds 2 children".

## test_node.cpp

Tests ModuleNode, LayerNode and the buffer check LayerAdapter makes at compile time.

This file needs no device. `NN::Node` in `include/SushiAI/nn/node.hpp` is the seam a
`ModelGraph` holds its rows through: a node declares its ports, reads a span of values and
appends its results. `NN::ModuleNode<L>` in `include/SushiAI/nn/module_node.hpp` wraps a
layer of any of the three forms, and `NN::LayerNode` in `include/SushiAI/nn/layer_node.hpp`
wraps an `NN::Layer`. Both return a `Status` from `forward`, and an `Error` a layer throws
comes back as a failure.

### BufferedStub

`BufferedStub` lists one `Buffer` as "running" and defines it as [4] zeros in its builder
`forward`.

### BufferedParentStub

`BufferedParentStub` lists a `BufferedStub` as the child "inner".

### ModalStub

`ModalStub` has the modal pair, counts the calls of each form and returns its input.

### TraceStub

`TraceStub` has a trace `forward` that records the mode of each trace it is given.

### ShapelessStub

`ShapelessStub` has `fields()` and a builder `forward` and no `output_shape`.

### RefusingStub

`RefusingStub` records an `InvalidDescription` failure on the trace and returns its input.

### RefusingThenThrowingStub

`RefusingThenThrowingStub` records an `OutOfMemory` failure on the trace and then throws an
`Error` with a different message.

### node_of

`node_of` calls `ModuleNode<L>::create` over a layer and an input shape, expects success and
returns the node.

### span_of

`span_of` views a vector of ids as the `NN::ValueSpan` one `forward` call takes.

### state_names

`state_names` collects the name of every parameter and buffer a node's `walk_state` yields
under a prefix.

### make_test_conv2d

`make_test_conv2d` calls `NN::make_conv2d` with 5 filters and a 3x3 kernel, the layer a
`conv2d` row of a model file builds.

### The two predicates at compile time

`LayerAdapter` asserts that `Detail::HoldsBufferField` is false for its layer. The predicate is
true for `BufferedStub`, for `BufferedParentStub`, for `Sequential<Linear, BufferedStub>` and
for `ModuleList<BufferedStub>`, and false for `Linear` and `Sequential<Linear, ReLU>`.

`ModuleNode` asserts that `Detail::HasOutputShapeOfInput` is true for its layer. The predicate
is true for `Linear` and `TraceStub` and false for `ShapelessStub`.

### ModuleNodeDeclaresOnePortWithTheLayersOutputShape

The node of `Linear{6, 8}` over [6] has one port named "output" of shape [8].

### ALayerNodeOfEitherKindReadsOneValue

`input_count()` is 1 for a `ModuleNode<Linear>` and for a `LayerNode`.

### ModuleNodeCreateReturnsTheFailureOfAShapeTheLayerRefuses

`ModuleNode<Linear>::create(Linear{6, 8}, [5])` returns `InvalidDescription` with a message
containing "Linear(6, 8)", "[5]" and "[6]". `Linear::output_shape` throws for that shape, and
`create` returns the throw as a failure.

### ModuleNodeForwardReturnsTheFailureOfALayerThatThrows

`forward` over a [2, 5] value returns `InvalidDescription` with "Linear(6, 8)" and "[2, 5]",
and `outputs` stays empty.

### ModuleNodeForwardReturnsTheFailureALayerRecorded

The node of `RefusingStub` returns the failure the stub recorded on the trace, and `outputs`
stays empty although the stub returned an id.

### ModuleNodeForwardReturnsTheRecordedFailureBeforeALaterThrow

The node of `RefusingThenThrowingStub` returns the `OutOfMemory` failure with "the stub's
recorded cause". The `Error` thrown after it is not what the caller sees.

### ModuleNodeForwardAppendsTheLayersResult

`outputs` holds one id before the call and two after it. The first is untouched and the second
is a value of shape [2, 8].

### ModuleNodeTracesAModalLayerInTheModeOfTheTrace

Under a training trace the stub counts one training call and no evaluation call. After an
evaluation trace it counts one of each.

### ModuleNodeHandsTheTraceToATraceModule

A training trace and then an evaluation trace leave `TraceStub::modes` as Training,
Evaluation, and `outputs` as the input id twice.

### ModuleNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("stem", ...)` of the node of `Linear{6, 8}` yields "stem.weight" and
"stem.bias".

### ModuleNodeWalksABufferOfTheLayer

After a trace, `walk_state("bn", ...)` of the node of `BufferedStub` yields "bn.running".

### ModuleNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeReportsTheShapeOfTheLayerItWraps

A `LayerNode` over the conv2d layer built for [8, 8, 2] has one port named "output" whose
shape is the layer's `output_shape()`, [6, 6, 5].

### LayerNodeForwardAppendsAValueShapedLikeItsPort

`forward` over a [2, 8, 8, 2] value appends one id of shape [2, 6, 6, 5].

### LayerNodeForwardReturnsTheFailureOfALayerThatThrows

A `LayerNode` over `LayerAdapter<Linear>` given a [2, 5] value returns `InvalidDescription`
with "Linear(6, 8)" and "[2, 5]", and `outputs` stays empty.

### LayerNodeWalksTheLayersParametersUnderThePrefix

After a trace, `walk_state("0", ...)` yields "0.weight" and "0.bias".

### LayerNodeRejectsTwoInputs

`forward` given two values returns `InvalidDescription` with "input count is 1" and "given is
2". `outputs` stays empty and the graph gains no node.

### LayerNodeRejectsNoInput

`forward` given an empty span returns `InvalidDescription` with "input count is 1" and "given
is 0".

### LayerNodeOverANullLayerIsFatal

The case is in the suite `Unit_NodeDeathTest`. Constructing a `LayerNode` from a null pointer
stops the process with a report containing "LayerNode" and "null layer".

## test_node_registry.cpp

Tests the node registry, make_node, build_model and the YOLO26n model file.

This file needs no device. `NN::K_NODE_REGISTRY` in `include/SushiAI/nn/node_registry.hpp` is
the table of node types a model file can name. `NN::make_node` builds the node of one row: it
looks the type up in `K_NODE_REGISTRY` and then in `NN::K_LAYER_REGISTRY`, and wraps a layer of
the second table in an `NN::LayerNode`. `NN::build_model` turns a `Config::ModelSpec` into an
`NN::ModelGraph`. All three return a `Result`.

The last six cases read `examples/yolo26/model.json`, and three of them also read
`tests/fixtures/yolo26/MANIFEST.json`. They locate the repository through
`Test::fixture_directory()` and report themselves skipped when that directory is not defined
for the binary.

### OpList

`OpList` is a vector of the operation of every node of a graph, in order.

### Constants

`K_YOLO_NORM` is `NormOptions{1e-3, 0.03}`. `K_BATCH` is 2, the batch size of every trace in
the file. `K_YOLO_ROWS` is 24, the row count of the YOLO26n model file. `K_ATTENTION_ROW` is
10, the row that names a type no table holds, and `K_ATTENTION_INNER_ROW` is 22, the row whose
inner block no table holds. `K_TOLERANCE` is 1e-12, the absolute bound on a real number read
from a file.

### MinimalCase

`MinimalCase` holds a node registry key, a row's attributes as JSON, the per-sample shapes the
row reads and the per-sample shape of the node's one port.

### K_MINIMAL_CASES

`K_MINIMAL_CASES` has one `MinimalCase` per row of `K_NODE_REGISTRY`, in the table's order.

### LayerCase

`LayerCase` holds a layer registry key, a row's attributes as JSON and the per-sample shape
the row reads.

### K_LAYER_CASES

`K_LAYER_CASES` has one `LayerCase` per row of `NN::K_LAYER_REGISTRY`, in the table's order.

### K_INPUT_JOINING_MODEL

`K_INPUT_JOINING_MODEL` is a version 2 model over [8, 8, 3] with three rows: a `conv` of 4
filters, a `silu`, and a `concat` whose `from` is `["input", -1]`.

### K_BRANCHING_MODEL

`K_BRANCHING_MODEL` is a version 2 model over [16, 16, 3] with a `defaults` block that gives
every `conv` row `bn_epsilon` 0.001 and `bn_momentum` 0.03. Its five rows are a `conv` of 8
filters, kernel 3 and stride 2; a `conv` of 16 filters with the same window; an `upsample` of
scale 2; a `concat` whose `from` is `[-1, 0]`; and a `conv` of 8 filters and kernel 1.

### TraceSummary

`TraceSummary` holds the operation list of one traced graph, each published output as "name
[extents]" and each piece of state as `record` writes it.

### attributes_of

`attributes_of` parses a string of JSON as the attributes of one row.

### model_of

`model_of` parses a string of JSON as a model file through `Config::parse_model`.

### span_of

`span_of` views a vector of shapes as the `NN::ShapeSpan` of one row.

### make_minimal

`make_minimal` calls `NN::make_node` for a `MinimalCase` with given attributes and the context
"layer 0 '<type>'".

### record

`record` writes one piece of state as "name [extents] parameter" or "name [extents] buffer".

### state_records

`state_records` returns one `record` per parameter and buffer that `ModelGraph::walk_state`
yields under a prefix.

### trace_summary

`trace_summary` traces a model under a training trace over a batch of `K_BATCH` samples and
returns its `TraceSummary`.

### build_branching_model_by_hand

`build_branching_model_by_hand` builds the graph `K_BRANCHING_MODEL` describes through
`ModelGraph::add`, with the nodes named "0" to "4", and publishes the last node through
`output_ports`.

### repository_root

`repository_root` returns the directory two levels above `Test::fixture_directory()`, or an
empty path when that directory is unknown.

### yolo_model_path

`yolo_model_path` returns `examples/yolo26/model.json` under a root.

### manifest_path

`manifest_path` returns the `MANIFEST.json` of the YOLO26 fixture folder under a root.

### manifest_file

`manifest_file` returns the entry of the manifest's `files` array with a given name, and throws
when there is none.

### record_row

`record_row` returns the row index of a manifest record named "model.<row>.<rest>".

### manifest_records

`manifest_records` lists the `weights.sacp` records of a range of rows in manifest order, as
`record` writes them. A name that contains ".running_" is a buffer.

### stand_in

`stand_in` replaces the type and the attributes of one row of a `ModelSpec` and keeps its
sources.

### keep_first_rows

`keep_first_rows` truncates a `ModelSpec` to its first rows and sets its outputs to the last
kept row.

### named_output

`named_output` returns an `OutputSpec` that publishes the port `NN::K_SINGLE_PORT_NAME` of a
row under a name.

### K_YOLO_ROW_SHAPES

`K_YOLO_ROW_SHAPES` lists the per-sample shape of rows 0 to 21 of YOLO26n over a
[640, 640, 3] input, from [320, 320, 16] for row 0 to [20, 20, 384] for row 21.

### SA_FIND_REPOSITORY_OR_SKIP

`SA_FIND_REPOSITORY_OR_SKIP(root)` declares `root` as `repository_root()` and skips the test
when it is empty.

### TheTypeStringsAreTheOnesModelFilesSpell

The `type` fields of `K_NODE_REGISTRY` are "batchnorm2d", "concat", "conv", "bottleneck",
"c3k", "c3k2" and "sppf", in that order.

### NoTypeIsInBothTables

No `type` of `K_NODE_REGISTRY` equals a `type` of `K_LAYER_REGISTRY`.

### EveryRowHasAMinimalCase

`K_MINIMAL_CASES` has as many entries as `K_NODE_REGISTRY`, and entry `i` names the type of
row `i`.

### EachRowBuildsFromAMinimalObject

For every case, `make_node` returns a node with one port named `NN::K_SINGLE_PORT_NAME` whose
shape is the one the case lists.

### EachRowRejectsAnUnknownKey

For every case, the attributes `{"no_such_key": 1}` fail with `InvalidDescription` and a
message containing the row's context and "'no_such_key'".

### AConcatRowNamesItselfWhenItsInputsDisagree

A `concat` row over [4, 4, 2] and [8, 4, 2] fails with a message containing "layer 7
'concat'", "[8, 4, 2]" and "H and W".

### ANodeTypeBuildsItsOwnNode

`make_node("conv", ...)` returns a `ModuleNode<Conv>` and not a `LayerNode`.

### ALayerTypeIsWrappedInALayerNode

`make_node("upsample", ...)` with scale 2 over [4, 4, 2] returns a `LayerNode` with one port
named `K_SINGLE_PORT_NAME` of shape [8, 8, 2]. The case then walks `K_LAYER_CASES`, which has
one entry per row of `K_LAYER_REGISTRY` in order. Every type builds a `LayerNode`, and every
type given `{"no_such_key": 1}` fails with "'no_such_key'", which shows the layer's own
factory ran.

### ALayerTypeRefusesTwoInputs

A `relu` row over two shapes fails with a message containing "layer 3 'relu'", "reads 2
values" and "reads one".

### ALayerFactorysRefusalComesBackAsAStatus

A `linear` row with no attribute fails with a message containing "layer 4 'linear'" and
"'units'". A `linear` row over [8, 8, 2] fails with the context, "[8, 8, 2]" and "'flatten'".
The layer factories throw, and `make_node` returns each throw as an `InvalidDescription`
failure.

### AnUnknownTypeListsBothTables

`make_node("conv3d", ...)` fails with a message containing "layer 5 'conv3d'", "the type
'conv3d'", "does not have", "node types" and "layer types". The message also contains every
type of `K_NODE_REGISTRY` and of `K_LAYER_REGISTRY` in quotes.

### AFileAndAHandBuiltGraphAgree

`build_model` of `K_BRANCHING_MODEL` and `build_branching_model_by_hand` both hold five nodes.
Their training traces have equal operation lists, equal outputs and equal state records. The
one output is "output [2, 8, 8, 8]". The state holds 15 records, five for each of the three
`conv` rows, from "model.0.conv.weight [8, 3, 3, 3] parameter" to "model.4.bn.running_var [8]
buffer".

### ARowReadsItsSourcesInFileOrder

In the model built from `K_BRANCHING_MODEL`, node 3 declares [8, 8, 24]: row 2's 16 channels
and then row 0's 8. Node 2 declares [8, 8, 16] and node 1 [4, 4, 16].

### OutputsArePublishedUnderTheirStatedNames

With the outputs set to row 4 as "mixed" and row 0 as "stem", the trace publishes "mixed
[2, 8, 8, 8]" and "stem [2, 8, 8, 8]" in that order.

### AnOutputWithoutANameKeepsThePortsName

An output of row 1 with an empty name is published as "output [2, 4, 4, 16]".

### ASourceMayNameAPort

A source that names the port `K_SINGLE_PORT_NAME` of the row above builds. A source that names
"no_such_port" fails with "layer 1 'conv'" and "'no_such_port'".

### ALaterRowMayReadTheModelInput

Row 2 of `K_INPUT_JOINING_MODEL` parses with `Config::K_MODEL_INPUT` as its first source. The
built node 2 declares [8, 8, 7]: the input's 3 channels and then row 1's 4.

### ASourceCannotNameAPortOfTheModelInput

A source that is the model input and names a port fails with a message containing "layer 2
'concat'", "the model input has no port" and "'output'".

### ARefusedRowNamesItsIndexAndType

Row 4 with `filters` 0 fails with a message containing "layer 4 'conv'", "'filters'" and
"positive".

### ARowThatReadsARowNotAboveItIsRefused

The specs here are changed after parsing, so they hold what the parser would have refused. A
source of row 3 on row 1 fails with "layer 1 'conv'", "row 3" and "not above". A source of -2
on row 2 fails with "layer 2 'upsample'" and "row -2". A row with no source fails with "layer 2
'upsample'" and "no source".

### AnEmptyOutputListPublishesTheLastRowWhole

A spec whose `outputs` were cleared builds a model whose `output_shape` is [8, 8, 8], the same
as that of the model built from the parsed spec.

### AWrongOutputListIsRefused

A spec with no layer fails with "no layers". An output of row 5 fails with "output 0", "row 5"
and "rows 0 to 4". A named output with an empty port fails with "output 0", "'mixed'" and
"without a port". An output of the port "no_such_port" fails with "output 0" and
"'no_such_port'". Two outputs named "mixed" fail with "output 1" and "'mixed'".

### TheYolo26FileHoldsItsTwentyFourRows

`Config::load_model` of `examples/yolo26/model.json` returns 24 rows over [640, 640, 3]. The
types in order are conv, conv, c3k2, conv, c3k2, conv, c3k2, conv, c3k2, sppf, c2psa,
upsample, concat, c3k2, upsample, concat, c3k2, conv, concat, c3k2, conv, concat, c3k2 and
detect. Row 0 reads the model input. The four joins read rows 11 and 6, 14 and 4, 17 and 13,
and 20 and 10, and row 23 reads rows 16, 19 and 22. Row 0 carries `bn_epsilon` 0.001 and row 9
`bn_momentum` 0.03 from the defaults, and row 11, an `upsample`, has no `bn_epsilon`. The one
output is row 23 with no port.

### TheYolo26FileNamesEachMissingBlock

`build_model` of the file fails at row 10 with a message containing "layer 10 'c2psa'", "the
type 'c2psa'", "does not have", "'c3k2'" and "'upsample'". With a `silu` row standing in for
row 10 it fails at row 22 with "layer 22 'c3k2'", "'inner'", "'bottleneck_psa'" and
"bottleneck, c3k". With a `conv` row of 256 filters standing in for row 22 it fails at row 23
with "layer 23 'detect'", "the type 'detect'" and "does not have".

### Yolo26RowsTwoAndFourAddTheirBottlenecksInput

Rows 2 and 4 of the file state `shortcut` true. The model built from rows 0 to 4 traces two K_ADD
nodes, one for the bottleneck of each of the two `c3k2` rows.

### Yolo26RowsZeroToNineMatchTheManifest

The model built from rows 0 to 9 is traced, and its state records under "model" equal the 165
records `manifest_records` reads from `MANIFEST.json` for rows 0 to 9: names, shapes, kinds
and order. The one output is "output [2, 20, 20, 256]".

### Yolo26RowsBuildToTheReferenceShapes

With a `silu` row standing in for row 10, the model built from rows 0 to 21 has 22 nodes, and
node `i` declares `K_YOLO_ROW_SHAPES[i]` on its one port.

### Yolo26NeckTracesToTheReferenceLevels

With the same stand-in, rows 16, 19 and 21 are published as "level0", "level1" and "joined".
For levels 0 and 1 the traced shape is [2, height, width, channels] with the three numbers
read from the `levels` array of the manifest's `forward.sacp` entry, and the level's `stride`
times its height is 640. The third output is "joined [2, 20, 20, 384]". The state records
equal the manifest's records for rows 0 to 9 followed by its 145 records for rows 11 to 21.

## test_objective.cpp

Tests the objective seam: the leaves it declares, the loss it emits and the resulting IR.

This file needs no device. Declaring leaves and emitting a loss are IR-to-IR, so the whole
objective seam is testable without an Engine. Only `score()` needs a tensor, and the integration
tier covers that through the trainer.

The test that carries this file is BuildGraphAndBuildClassifierProduceTheSameIr. The objective
seam's whole promise is that it changed no numbers. The way to check that is not "the loss still
falls". It is that the two entry points emit the same graph, node for node, in the same order. A
graph comparison fails on a reordering that a loss comparison would absorb.

### TheTargetLeafIsDeclaredWhereItAlwaysWas

The target leaf is declared between the input leaf and the model's own trace. Value ids are
positions, so declaring it anywhere else renumbers the whole graph. Every regression test that
pins a node count or a byte-for-byte result would then compare two different graphs while
appearing to compare one.

### TheObjectiveOwnsTheOutputShapeCheck

What shape an output must be is the objective's question, since a detector's is not a
classifier's. The message therefore comes from the objective and not from the graph builder.

### CrossEntropyTakesNoKeysAtAll

Cross-entropy has no hyperparameters, so a key written onto it is an error and not a line whose
author believed it did something.

## test_ops.cpp

Tests that the operation table is complete and that the reverse lookup resolves names.

This file needs no device. The uniqueness proof already runs as a static_assert inside
`include/SushiAI/core/ops.hpp`. These tests cover what that assert cannot: that the table is
complete, and that the reverse lookup behaves.

### EveryNameCarriesTheLibraryPrefix

The prefix keeps a SushiAI identity from colliding with SushiBLAS's own (blas.lvl3.gemm and
friends) in the profiler's report.

The test calls `rfind(prefix, 0)` and not `starts_with()`, which is C++20. `rfind` searches
backwards from position 0, so it can only match at position 0, which is the same question asked
with the vocabulary C++17 has.

### TheTableContainsEveryDeclaredIdentity

A new operation declared but left out of K_ALL would escape the uniqueness static_assert
entirely, so the test checks membership by name.

### EverySpatialIdentityIsListedInAll

K_ALL is what `all_ids_unique()` proves over and what `name_of()` searches. An identity declared
but left out of K_ALL is not covered by the collision proof and prints as "sushiai.unknown" in
every profile. Both failures are silent.

### ThereIsNoConv2dIdentity

CV-1 says convolution is K_IM2COL + K_MATMUL in the IR. A CONV2D identity would be a second, opaque
spelling of the same thing, and the first person to add one will do it by analogy and not by
reading the spec.

### LeakyReluDifferentiatesAtItsOutput

For alpha > 0, leaky_relu(x) > 0 exactly when x > 0. Reading the output in place of the input is
therefore a bit-identical substitution, not an approximation, and it is the same argument the
table already records for K_RELU. It is what makes the fold free: an ordinary use count licenses
it with no activation-specific knowledge in the fusion pass.

### EveryElementwiseAndSelectionIdentityIsListedInAll

`Ops::name_of` resolves K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_LESS_EQUAL, K_SELECT, K_SILU,
K_SILU_BACKWARD, K_SOFTPLUS and K_SOFTPLUS_BACKWARD to a name other than "sushiai.unknown". The
reason is the one EverySpatialIdentityIsListedInAll gives.

### TheComparisonAndTheActivationGradientsAreNotFlaggedDifferentiable

`differentiable` is true for K_DIV, K_MINIMUM, K_MAXIMUM, K_ABS, K_ATAN, K_SOFTPLUS, K_SELECT and K_SILU, and
false for K_LESS_EQUAL, K_SILU_BACKWARD and K_SOFTPLUS_BACKWARD.

### TheGradientBarriersAreTheFourIdentitiesThatPassNoGradient

`Ops::is_gradient_barrier` is true for K_ZERO, K_CONSTANT, K_LESS_EQUAL and K_STOP_GRADIENT. Over every
row of `Ops::K_ALL` it equals the row's `blocks_gradient` flag, and exactly four rows carry the
flag. A fifth barrier therefore fails the count until the list in this test names it.

### NoGradientBarrierIsFlaggedDifferentiable

No row of `Ops::K_ALL` has both `differentiable` and `blocks_gradient` set.

### LeafCastAndTheGradientOperationsAreNotGradientBarriers

`is_gradient_barrier` is false for K_LEAF, K_CAST, K_DYNAMIC_SCALE, K_LOG_SOFTMAX, K_RELU_BACKWARD,
K_SILU_BACKWARD, K_SOFTPLUS_BACKWARD, K_MAXPOOL2D_BACKWARD, K_BATCHNORM2D_BACKWARD,
K_BATCHNORM_UPDATE_STATS, K_BATCHNORM_FOLD, K_GEMM_BIAS, K_EXP_SUB_SCALE and K_ADAMW_UPDATE, and for the
foreign identity 0xDEADBEEF. These either pass a gradient or throw when one reaches them.
Marking one as a barrier would turn that throw into a gradient that is dropped without a
message.

### SiluDifferentiatesAtItsInput

The `Ops::K_ACTIVATIONS` row of K_SILU names K_SILU_BACKWARD as its gradient and
`GradientSource::Input` as its source, and `activation_survives_fusion(K_SILU)` is false.

### SoftplusDifferentiatesAtItsInput

The row of K_SOFTPLUS names K_SOFTPLUS_BACKWARD and `GradientSource::Input`, and
`activation_survives_fusion(K_SOFTPLUS)` is false.

### IsActivationBackwardAcceptsOnlyTheTablesGradientColumn

For every row of `Ops::K_ACTIVATIONS`, `is_activation_backward` is true for the row's `backward`
identity and false for its `forward` one. Over `Ops::K_ALL` it accepts as many identities as the
table has rows.

### BatchNormFoldIsListedAndClaimsNoRule

`Ops::name_of(K_BATCHNORM_FOLD)` is "sushiai.batchnorm_fold", and the row has `differentiable`
false and `blocks_gradient` false. `test_backward.cpp` holds what differentiation does when a
gradient reaches the node.

## test_precision.cpp

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

### Traced

The `Traced` graph is built by hand and not through NN::Sequential, so that the test can name
every value it wants to assert about. It is also the smallest graph that still contains all four
interesting shapes: a parameter read by both passes, a reduction, a loss, and a gradient buffer.

### OnlyTheProductAndWhatAGemmFoldsIntoItNarrow

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
would overflow the very quantity it protects; see `include/SushiAI/optim/loss_scaler.hpp`.

K_GELU, K_GELU_BACKWARD and K_GEMM_BIAS_GELU, the one fused form that folds K_GELU, are in `stays`. The
form is a GEMM and rule 1 would otherwise take it. Narrowing it puts the tanh approximation's
`x*x*x` into fp16, which overflows at |x| ~ 40.3. Rule 5 in
`include/SushiAI/graph/precision.hpp` says so, and also explains why that leaves rules 2 and 5
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

### GeluStaysWideEvenThoughACatalogEntryNowFoldsIt

Rule 5 used to rest on two arguments, and one of them is gone. `FusedForm::GemmBiasGelu`
folds GELU by spilling the pre-activation, so "nothing folds it, therefore nothing forces its
precision" is no longer a reason for anything. The surviving argument is the arithmetic: the
tanh approximation computes `x*x*x`, which overflows fp16 at |x| about 40.3. A unit test cannot
assert that without a device.

The first three assertions check what a unit test can: that the conclusion did not quietly
follow the dead premise. The catalog can fold GELU and the policy still keeps it wide, which is
the pair of facts rule 5 in `include/SushiAI/graph/precision.hpp` now asserts.

`narrows_under_mixed_precision(K_GEMM_BIAS_GELU)` is false because the fused form has to go with
GELU. Otherwise the cast pass would narrow a node whose epilogue is the very expression rule 5
refuses to put in fp16.

### ChoosingIsIdempotentForEveryOperation

`include/SushiAI/graph/precision.hpp` states this contract, and it is what makes the invariant
checkable after the pass has run and not only while it runs. Without it,
`find_dtype_mismatches` would report a graph the pass had just finished making correct.

### AMixedPolicyNeverWidensAndNeverNarrowsPastItsLowType

The checks on the fp64 and fp16 graphs say that an fp64 graph keeps fp64 masters and still gets
fp16 products, and that an fp16 graph is already at the floor and is left alone, not "narrowed"
to something wider.

The `to_single` policy narrows to fp32. That is a legal thing to ask for, and it is what an fp64
model would use.

### TheUniformPolicyInsertsNoCasts

This is the control the mixed cases are read against. If it ever shows casts, the pass is
inserting them unconditionally and every "the mixed policy inserted N casts" assertion in the
later Unit_CastInsertion cases measures nothing.

### EveryNodesOperandsAgreeWithTheTypeItComputesIn

This assertion is worth more than a dozen spot checks. It walks the whole graph and states the
rule, so it keeps holding as operations are added.

### TheInvariantSurvivesTheFusionPassThatRunsAfterIt

Fusion rewrites nodes and re-pairs operands, so it is where a correctly cast graph could stop
being one, and where a master weight could get swallowed.
`include/SushiAI/graph/cast_insertion.hpp` argues the ordering, and this test is the assertion
behind the argument.

### ADeliberatelyBrokenGraphIsReportedRatherThanAccepted

`find_dtype_mismatches` is the test file's own instrument, so it has to be shown to fail on a
graph that is wrong. Otherwise every "no mismatches" in the earlier cases could come from a
checker that never looks.

`mismatches.size()` is 3 because the graph is an uncast fp32 matmul under a mixed policy: two
operands and one result, all three float32 where the policy says half.

### AParameterKeepsItsTypeAndGetsADerivedLowPrecisionCopy

The fp16 value the GEMM reads is a different value from the parameter. A cast produces it, and
it lives with the step's scratch and not for the whole run.

### AGradientIsWidenedBeforeItReachesItsBuffer

The test covers the backward direction. A matmul computing in fp16 produces an fp16 gradient,
and if that reached the fp32 gradient buffer uncast, SushiBLAS would throw. If the buffer had
been retyped to fp16 instead, nothing would throw and the optimizer would accumulate in fp16,
which is the failure master weights exist to prevent.

The loop over `gradients[0]` and `gradients[2]` covers the weights only. A backward K_MATMUL,
which computes low, produces the gradients of w1 and w2. K_SUM_ROWS, which does not, produces
those of b1 and b2, so only the weights need widening.

The loop over `gradients[1]` and `gradients[3]` covers the biases. Their gradients come out of
a reduction that stayed wide, so nothing was widened into them and nothing needed to be.

### AccumulatingGradientContributionsHappensWide

A parameter read twice gets two gradient contributions and an K_ACCUMULATE to sum them. Two things
have to hold at once. The sum runs in the wide type, because it is a sum. Its result and its
first operand agree, because the lowering makes the result alias the accumulator's buffer, so a
disagreement there is a wrong-width write and not merely a wrong number.

### AnObservedValueKeepsItsTypeAndAnUnobservedOneDoesNot

This case covers the half of the pinning rule that a lifetime class does not. The logits are an
ordinary activation. The training loop reads them back by id for accuracy, and a caller who
forgets to say so gets an fp16 buffer where their host-side argmax expects floats.

### AValueReadTwiceIsConvertedOnce

w2 is read by the forward matmul and again by the backward one. Two conversions would be two
kernels and two buffers for one number, and the reuse is safe because the IR is
single-assignment.

### AGraphThatIsNotInTopologicalOrderIsRefused

`fuse()` and `MemoryPlan::build()` carry the same guard for the same reason. The pass emits a
conversion immediately before its consumer, which is only a valid position if producers already
come first.

### Pooled

K_MAXPOOL2D is the first operation in this IR whose results are not all one element type: it
writes the pooled tensor and, beside it, the INT64 argmax its backward reads. The cast pass
derives one precision per node from `outputs[0]`, which makes the argmax look like a value in
the wrong type and not like a value with no precision to choose.

`x` is a parameter and not an input for the reason
`tests/functional/regression/test_argmax_lifetime.cpp` gives: K_MAXPOOL2D_BACKWARD is only emitted
when something upstream of the pool needs a gradient.

### AnIndexBufferHasNoPrecisionToChoose

The pass forced every operand and every result of a node to one dtype, derived from the node's
first result. For K_MAXPOOL2D that is FLOAT32 from the pooled tensor, so the INT64 argmax read as
a mismatch and the pass tried to convert it. `GraphBuilder::cast` refuses anything but
real-float to real-float by design, so the whole build threw.

The defect does not depend on the policy. `wanted` comes from `outputs[0]`, not from the
policy's `choose()`, so the uniform policy reaches it too.

### APinnedIndexBufferIsNotWidenedIntoItsOwnBuffer

This is the loud half of the same defect. An unpinned argmax is retyped in place and the
mismatch disappears silently. A pinned one keeps its id and its INT64 buffer, so the pass
reaches for the widening cast, and K_CAST is real-float to real-float by design.

### ABufferKeepsItsDeclaredTypeAndItsStateLeafAcrossThePass

The graph is a K_MATMUL and a K_BIAS_ADD whose bias is a buffer. The weight and the buffer are
bound to the states 31 and 32 and the graph is marked refused before `insert_casts` runs under
the mixed policy. The report counts at least one cast and the K_BIAS_ADD's result is HALF, so
the pass did rewrite the graph. The buffer is still FLOAT32 with the lifetime `BUFFER` and no
gradient. Both `state_leaf` lookups, the whole `state_leaves()` table and `has_refusal()` are
what they were, and `find_dtype_mismatches` returns nothing.

### BatchNormFoldStaysInTheMasterTypeUnderTheMixedPolicy

The graph is one `batchnorm_fold` over two parameters and two buffers. `insert_casts` under
the mixed policy reports zero casts and the graph holds no K_CAST node. The four operands and
the two results are all FLOAT32.

## test_provenance.cpp

Tests the forward origin that differentiation records for every node it appends.

This file needs no device. `Autograd::differentiate()` fills a `Provenance` while it appends
backward nodes, so every case differentiates a host-side graph and reads the record back.
`include/SushiAI/eager/README.md`, under `autograd/provenance.hpp`, states the origin rule and
what the partitioner needs it for.

### TwoLinearGraph

`TwoLinearGraph` is a two-layer perceptron with a cross-entropy loss: an [8, 6] input, a 6-to-4
linear layer, a K_RELU, a 4-to-3 linear layer, then K_LOG_SOFTMAX and K_NLL_LOSS. It keeps the ids the
cases name: the four parameters, the result of the first K_MATMUL and the loss.

### differentiate

The file-local `differentiate` wraps `Autograd::differentiate` so that a case passes its `wrt`
list as a vector. It passes no seed factor and hands the `Provenance` pointer through unchanged,
which lets one case pass null.

### nodes_of

`nodes_of` returns the ids of the nodes that run one operation, starting at a given node index.
A case passes `forward_node_count` as the start, so the search covers the appended nodes only.

### EveryBackwardNodeHasAForwardOrigin

`forward_node_count` equals the node count taken before differentiation, `origin` holds one
entry per appended node, and every entry is below `forward_node_count`. An origin at or above
that count would name a backward node, which no segment of the forward trace holds.

### ARuleEmitsUnderTheNodeItDifferentiates

The node that produces the first weight's gradient is an appended node, and its origin is the
K_MATMUL that produces `first_product`. The case asserts the operation and the node id, so an
origin that named another K_MATMUL of the graph fails it.

### AFanOutAccumulateBelongsToTheProducer

`shared` is a K_RELU result that a K_TANH and a K_SIGMOID both read, so its gradient has two
contributors. The backward nodes hold exactly one K_ACCUMULATE, and its origin is the K_RELU that
produces `shared`. `x` is read once, so no closing sum is emitted and the count stays at one.

### AParameterReadTwiceAccumulatesUnderItsLeaf

The parameter `x` is read by a K_RELU and a K_TANH. The node that produces its gradient is an
K_ACCUMULATE, and the origin of that node is the K_LEAF of `x`.

`targets` is declared before `x` so that the K_LEAF of `x` is not node 0. The reverse walk ends at
node 0, so with `x` there an origin taken from the walk's last position would equal the K_LEAF's
id and the `EXPECT_NE(origin, 0u)` line could not tell the two apart.

### AReusedProvenanceHoldsOnlyTheLatestRun

One `Provenance` is filled by a first differentiation and then passed to a second one over a
`TwoLinearGraph`. Its `forward_node_count` and `origin` equal those of a fresh `Provenance`
filled from an identical graph. A `differentiate()` that appended to `origin` and did not reset
it would leave the first run's entries in front.

### ARequestedValueWithoutAProducerIsRefused

`x` is added with `add_value` and so has no K_LEAF node, and it is read twice, so the closing sum
of its gradient needs an origin that does not exist. The call throws, and the message contains
"provenance" and "producer node".

### PassingNoProvenanceChangesNothing

Two identical graphs are differentiated, one with a `Provenance` and one with null. The returned
gradient ids are equal, the node counts are equal, and every node has the same operation,
inputs and outputs. Recording is therefore an observation of the transform and not a branch in
it.

## test_region_signature.cpp

Tests the signature that `signature_of` computes for one region of a partition.

This file needs no device. `Eager::signature_of` reads a graph, a partition and one identity per
value, all of which a case builds on the host. `include/SushiAI/eager/README.md`, under
`region_signature.hpp`, lists what the hash covers and gives the reason for each operand rule.

Most cases build two graphs that differ in one property and compare the two signatures. Where
the operation list is the same in both, the case asserts that first, so the difference it then
asserts is a difference in the hash alone.

### Region keys and identities

`K_BODY` is key 1, the one region of a hand-built partition, and `K_READER` is key 2, the region
that reads from it in a two-region partition. `K_PARAMETER_IDENTITY` and `K_EXTERNAL_IDENTITY`
are the default identities of the parameter and of the value `K_READER` reads from outside.
They are stated once so that a case names the one it changes.

### Traced

`Traced` holds a graph, its partition and one identity per value id, which are the three things
`Eager::signature_of` reads.

### signature_of

The file-local `signature_of` has three overloads. The first takes a `Traced` and a region key
and calls `Eager::signature_of` with the identity vector as a span. The other two take a
`Recipe` or an `ExternalRecipe`, build its graph and return the signature of `K_BODY` or of
`K_READER`.

### Recipe

`Recipe` describes `scale(combine(relu(x), tanh(x)), alpha)` over a parameter `x`, and each of
its fields is one property a case varies: the shape, the element type, the K_SCALE factor, K_SUB in
place of K_ADD, the operand order of the combine, and the identity of `x`. `build(Recipe)` puts
the four operations under `K_BODY` and the K_LEAF under no region, and gives every value but `x`
identity zero.

### node_of

`node_of` returns the first node of a graph that runs a given operation. The attribute and
layout cases use it to reach the node they edit after `build` returns.

### assign_regions

`assign_regions` gives a `Traced` its partition from one key per non-K_LEAF node, in node order,
and a region list. A K_LEAF gets `K_NO_REGION`. It also resets every identity to zero, so a
builder that calls it sets afterwards only the identities its case is about.

### ExternalRecipe

`ExternalRecipe` describes a graph whose `K_READER` region holds one K_SCALE that reads a value
the region does not produce. Its fields choose whether that value is the parameter `x` or the
K_RELU result of `K_BODY`, whether `K_BODY` appears in the region list, and which identity the
value has.

### ProbeOperands

`ProbeOperands` names four operand orders for a hand-built three-operand K_ADD in `K_READER`.
`build(ProbeOperands)` leaves every identity at zero, and the one in-region operand is result 0
of the region's node 0. Both orders of a pair therefore feed the hash the same words, and the
two signatures can differ only if the kind of each operand enters the hash.

### build_reader_of_result

`build_reader_of_result` adds a K_RELU node by hand with two results and a K_TANH that reads the
result at the given index, both under `K_BODY`. The K_RELU is added through `Graph::add_node`
because `GraphBuilder::relu` returns one result.

### with_identity

`with_identity` records an identity for one value and returns the value id, so that
`ScopedPerceptron` can assign an identity in the expression that declares a leaf.

### info_of

`info_of` returns the `RegionInfo` of a key, or null when the partition does not list the key.

### role_identity

`role_identity` hashes a region's name, its direction, how many nodes of the region follow the
producing node, and the index of the value among that node's results. Counting from the
region's end keeps the identity of a region's last result the same when nodes are added before
it.

### identify_produced_values

`identify_produced_values` gives every value that a region's node writes its `role_identity`.
In the tests it stands in for the caller of `signature_of`, which owns the storage a boundary
value is bound to; `include/SushiAI/eager/README.md` says why only the caller can supply that
identity.

### ScopedPerceptron

`ScopedPerceptron` traces three scopes, "stem", "body" and "head", then differentiates and
partitions the graph with `ScopePartitioner`. Its argument is the number of linear layers in
"stem". Leaves get fixed identities and produced values get role identities. `key_of` returns
the key of a region by name and direction, and `stem_output` is the value "body" reads from
"stem".

### TheSameTraceGivesTheSameSignature

Two graphs built from the default `Recipe` have equal hashes and compare equal. The operation
list is K_RELU, K_TANH, K_ADD, K_SCALE, which is the region's nodes in order with the K_LEAF left out.

### AChangedAttributeChangesTheHash

A K_SCALE factor of 3.0 in place of 2.0 leaves the operation list equal and makes the hash
differ.

### AChangedLastAttributeSlotChangesTheHash

The case writes 1.0 into slot `Attributes::K_CAPACITY - 1` of the K_SCALE node and expects another
hash. It covers the last slot because a hash that read a fixed number of leading slots would
agree with the other attribute case and miss this one.

### AChangedLayoutChangesTheHash

The K_RELU result is retagged NCHW in one of two otherwise equal graphs, and the hashes differ.
Shape and element type are the same in both, so the layout is the only input that moved.

### AChangedShapeChangesTheHash

A shape of [4, 5] in place of [4, 3] leaves the operation list equal and makes the hash differ.

### AChangedDtypeChangesTheHash

FLOAT64 in place of FLOAT32 leaves the operation list equal and makes the hash differ.

### AChangedOpChangesTheHash

K_SUB in place of K_ADD makes both the operation list and the hash differ. It is the one
`Recipe` case in which the two lists are unequal.

### ADifferentParameterChangesTheHash

The same structure over a parameter with identity `K_PARAMETER_IDENTITY + 1` has the same
operation list and another hash. The parameter is external to the region, so its identity is
the only thing the hash holds about it.

### AnEarlierRegionGrowingDoesNotChangeALaterRegion

`ScopedPerceptron(1)` and `ScopedPerceptron(2)` differ by one linear layer in "stem". The three
`ASSERT` lines at the top establish that the case tests what it claims: the value "body" reads
has a different id in the two graphs, the same identity in both, and the forward and backward
regions of "body" have different keys.

For each direction, the signature of "body#0" has a non-empty operation list and is equal in
the two graphs, by hash and by `operator==`. The forward signature of "stem#0" differs between
them, which shows that the two graphs are not equal throughout.

### ARewiredOperandChangesTheHash

`add(relu(x), tanh(x))` and `add(tanh(x), relu(x))` have the same operation list and different
hashes. Both operands are produced inside the region, so the case covers the in-region operand
position.

### AReadOfAnotherResultChangesTheHash

A K_TANH that reads result 1 of a two-result node hashes apart from one that reads result 0. The
producer's index in the region is the same in both, so the result index is what separates them.

### ABoundaryValueOfAnotherIdentityChangesTheHash

The K_SCALE in `K_READER` reads the K_RELU result of `K_BODY`. With identity
`K_EXTERNAL_IDENTITY + 1` for that value, the operation list is equal and the hash differs.

### ALeafAndABoundaryValueOfOneIdentityHashAlike

The reader's signature is equal, by hash and by `operator==`, whether its operand is the
parameter `x` or the result of another region, as long as both carry `K_EXTERNAL_IDENTITY`.
Where an external operand comes from in the graph does not enter the hash.

### AnOperandFromAnUnlistedRegionHashesAsExternal

The same graph is signed with `K_BODY` present in the region list and with it absent, and the
two signatures of `K_READER` are equal. An operand whose producer has a key other than the
region's is external whether or not the partition lists that key.

### AnOperandInsideTheRegionHashesApartFromOneOutside

The probe node reads one result of `K_READER` and the result of `K_BODY` twice, in the two
orders `InsideThenOutside` and `OutsideThenInside`. The operation lists are equal and the
hashes differ.

### ALeafOperandHashesApartFromOneInsideTheRegion

The same pair of orders with the leaf `x` in place of the `K_BODY` result,
`InsideThenLeaves` and `LeavesThenInside`, also gives equal operation lists and different
hashes.

### EqualHashesWithDifferentOpsCompareUnequal

Three signatures are written by hand with hash 42. The one listing K_ADD and the one listing K_SUB
compare unequal under `==` and `!=`, and two that both list K_ADD compare equal. A real collision
cannot be produced on demand, so the case constructs one.

### ARegionThePartitionDoesNotListThrows

`signature_of` throws for key 6 and for `K_NO_REGION`. Each message names the key, "Region 6"
and "Region 0", and contains "not listed".

### APartitionOfAnotherNodeCountThrows

One entry is removed from `region_of_node`, which leaves four entries for a five-node graph.
The message contains "assigns 4 nodes" and "holds 5".

### AnExternalValueWithoutAnIdentityThrows

The identity vector is cleared, so the id of `x` lies beyond its end. The message contains
"External value 0" and "no identity". The call throws and does not read past the span.

## test_rematerialise.cpp

Tests the structure of the graph that the K_IM2COL rematerialisation pass rewrites.

This file needs no device. The K_IM2COL rematerialisation pass is an IR-to-IR rewrite, so what it
produces is a graph that can be inspected node by node.

The unit tier proves the structure: that a clone appears, that it reads the same operands and
carries the same attributes as the node it copies, that it is inserted ahead of its consumers
and not after them, and that an inference graph is left alone. It cannot prove the numbers are
unchanged. Integration_Rematerialise in `tests/functional/integration/test_rematerialise.cpp`
does that by running the same graph twice.

The brief's three cases are all in this file, with two deviations that the library forced:

1. `Autograd::differentiate()` refuses any root that is not a scalar produced by K_NLL_LOSS
   (`src/autograd/differentiator.cpp` explains why every other root would silently return empty
   gradients). The brief's graphs stopped at a K_MATMUL, so each is carried on to a loss. The
   shapes, geometry and expectations are the brief's.
2. A "backward consumer" cannot be identified by node index, because the pass runs when the
   backward nodes are already in the graph and no marker separates the phases. The rule the pass
   applies is "every consumer after the first", which is the same set for every graph
   differentiation produces and is well-defined for one it did not produce.

### TheColumnBufferStopsBeingAnActivation

This case checks the number CV-10 is about. ResNet-18's first block at batch 128 holds
`128*32*32*64*9*4` bytes, about 302 MB, in this one buffer, and there are twenty convolutions.
Cloning the gather next to its backward consumer drops the lifetime to SCRATCH, so one buffer is
live at a time and not twenty.

### TheCloneComputesTheSameThingAsTheOriginal

The operand and attribute checks guard against a clone that read a different operand or carried
different attributes. Such a clone would produce a plausible column matrix and a wrong weight
gradient. Both are checked because the geometry lives in the attributes and is easy to drop when
copying a node.

The check on `copy` expects the same description and the same layout. The layout is a
graph-wide invariant, and a clone that dropped it would fail `finalise()` on a graph that is
numerically correct.

### TheCloneIsInsertedAheadOfEveryConsumerItServes

The failure this case catches is not a wrong number. It is a graph whose insertion order stopped
being a topological order, with a recomputation appended after the GEMM that reads it.
`MemoryPlan::build()` rests on that order and says so by name, and a lowering that got past it
would record a GEMM against a buffer nothing had written yet.

### AnIm2colWithNoBackwardConsumerIsLeftAlone

This is the inference case. Nothing was differentiated, so there is no second consumer and
cloning would be a free slowdown.

### AForwardOnlyConvolutionIsLeftAlone

This is the other half of the inference case, and the one that would break if the pass cloned
for every consumer and not for every consumer after the first. The column matrix has exactly one
reader, the GEMM.

### RunningItTwiceAddsNothingTheSecondTime

Once the rewiring is done the clone has one consumer of its own, so the pass has no reason to
rematerialise its own output. If it did, the pass would not terminate on a graph with three
consumers.

## test_result.cpp

Tests Result, its reference form and SA_TRY_ASSIGN.

This file needs no device. `Result<T>` and `SA_TRY_ASSIGN` are in
`include/SushiAI/core/result.hpp`. A `Result<T>` holds a value or a failed `Status`, and
`Result<T&>` holds a reference to the caller's object or a failed `Status`.

Nine `static_assert` lines at the top of the file hold the type's shape. Neither form is
default-constructible. Construction of a `Result<int>` from an `int` and from a `Status`, and
its move, are `noexcept`. A `Result<int>` is at most two pointers and a `Result<int&>` is
exactly two. Neither `Result<int&>` nor `Result<const int&>` is constructible from an rvalue
`int`, so a reference form cannot refer to a temporary.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### checked_extent

`checked_extent` returns its extent as a `Result<int>`, or an `InvalidDescription` failure from
`SA_FAIL_IF` when the extent is not positive.

### doubled_extent

`doubled_extent` binds `checked_extent`'s value through `SA_TRY_ASSIGN` and returns twice it.

### boxed_extent

`boxed_extent` returns a `std::unique_ptr<int>` holding the checked extent, a move-only value.

### unboxed_extent

`unboxed_extent` binds `boxed_extent`'s pointer through `SA_TRY_ASSIGN` and returns the
integer it owns.

### forward_temporary_status

`forward_temporary_status` runs `SA_TRY(checked_extent(extent).status())`, which reads a
`Status` reference into a temporary `Result`, and sets a flag after it.

### slot_at

`slot_at` returns a `Result<int&>` to one of two slots, or an `InvalidDescription` failure for
an index outside 0 and 1.

### bump_slot

`bump_slot` binds `int& slot` from `slot_at` through `SA_TRY_ASSIGN` and increments it.

### HoldsAValueWithASuccessStatus

`Result<int>(7)` has a value, an ok status and the value 7. Writing 9 through `value()` is
read back through a const reference.

### HoldsAFailureAndNoValue

A result built from the failure "rank must be 2" has no value. `SA_EXPECT_FAILURE` accepts
both the result and its `status()`.

### ReturnStatementsConvertFromAValueAndFromAStatus

`checked_extent(5)` yields 5 through `SA_ASSERT_OK_AND_ASSIGN`, and `checked_extent(-1)` is an
`InvalidDescription` failure containing "extent must be positive" and "got -1". The function
returns an `int` on one path and the `Status` of `SA_FAIL_IF` on the other.

### SuccessPathCreatesNoFailureBlock

A made, a copied and a moved `Result<int>` and one `SA_TRY_ASSIGN` through `doubled_extent`
sum to 13 and leave `live_failure_count()` where it was.

### RvalueValueMovesAMoveOnlyValueOut

`std::move(result).value()` on a `Result<std::unique_ptr<int>>` returns a pointer that is not
null and owns 11.

### MovedFromValueResultStillHoldsAMovedFromValue

After a `Result<std::unique_ptr<int>>` is moved from, the target owns 2. The source still
reports a value and an ok status, and its pointer is null.

### MovedFromFailureResultKeepsItsFailure

After a failed `Result<int>` is moved from, the source and the target both report
`InvalidDescription` and "layout must be NHWC", and the source has no value.

### CopyAndAssignmentCarryTheHeldState

Assigning a failed result to one that holds 1 leaves no value and the failure "no such key".
Assigning `Result<int>(8)` to it afterwards leaves a value, an ok status and 8.

### TryAssignBindsTheValueOrReturnsTheFailure

`doubled_extent(21)` is 42. `doubled_extent(0)` is the failure of `checked_extent`, with
"got 0" in its message.

### TryAssignMovesFromATemporaryResult

`unboxed_extent(6)` is 6, which requires `SA_TRY_ASSIGN` to move the `unique_ptr` out of the
temporary result. `unboxed_extent(-6)` is the failure with "got -6".

### ReferenceFormRefersToTheCallersObject

`slot_at(slots, 1)` has a value whose address is `&slots[1]`, and writing 25 through it
changes the array. A `Result<const int&>` built from a const `int` returns that object's
address.

### TryReadsAStatusReferenceIntoATemporaryResult

`forward_temporary_status(-4)` returns the failure with "extent must be positive" and
"got -4" and leaves its flag false. `forward_temporary_status(4)` succeeds and sets it.
`live_failure_count()` is at its start after each call.

### MovedFromReferenceResultKeepsItsState

A `Result<int&>` to slot 0 is copied and moved from. The source, the copy and the target all
refer to `&slots[0]`. A failed reference result moved from keeps the failure "slot 2 is not
one of 2" in both source and target. Assigning the failed one over a result to slot 1 gives
the failure, and assigning the successful one over that gives `&slots[0]` and an ok status.

### ReferenceFormHoldsAFailure

`slot_at(slots, 2)` has no value and the `InvalidDescription` failure "slot 2 is not one of 2".

### TryAssignBindsAReferenceFromTheReferenceForm

`bump_slot(slots, 0)` succeeds and slot 0 goes from 10 to 11, so the reference `SA_TRY_ASSIGN`
bound is the array's element. `bump_slot(slots, 5)` fails with "slot 5" and slot 1 stays 20.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Result<int>` and on a failed `Result<int&>` stops the process. The
report contains "Result::value() was called on a failure" and the failure's own message.

### BuildingFromASuccessStatusIsFatal

`Result<int>(Status())` and `Result<int&>(Status())` stop the process with "a Result was built
from a success Status".

## test_rule_table.cpp

Tests that every domain's backward rules reach the combined table exactly once.

### EveryDomainsRulesReachTheCombinedTable

`include/SushiAI/autograd/rules/all.hpp` concatenates the domains. A domain left out of the
concatenation compiles and never differentiates. The count is the cheapest thing that notices.

### NoOperationHasTwoRules

Two entries for one identity is not a compile error, and the first one silently wins. Which
gradient runs would then depend on the order in which two domains happened to be concatenated.

### TheRulesThatExistedBeforeTheSplitStillDo

The split is a move, not a rewrite. These fourteen rules were the table before it, and every
one must still resolve.

### NoGradientBarrierHasARule

For every row of `Ops::K_ALL` with `blocks_gradient` set, `Autograd::has_rule` and
`Autograd::has_backward_rule` are both false. The loop counts the rows it checked and expects
4, so it cannot pass by checking none.

### TheComparisonIsABarrierWithoutARule

`Ops::K_LESS_EQUAL` has `differentiable` false and `blocks_gradient` true, and
`Autograd::has_rule` is false for it.

### EveryActivationRowHasARuleForItsForwardIdentityOnly

For every row of `Ops::K_ACTIVATIONS`, `Autograd::has_rule` is true for the `forward` identity
and false for the `backward` one. `Autograd::K_ACTIVATION_RULES` has as many entries as
`Ops::K_ACTIVATIONS` has rows.

### BatchNormFoldHasNoRuleAndBatchNormKeepsIts

`Autograd::has_rule` is true for K_BATCHNORM2D and false for K_BATCHNORM_FOLD, `has_backward_rule`
is false for K_BATCHNORM_FOLD, and `Autograd::K_NORM_RULES` has two entries.

## test_scope_partitioner.cpp

Tests the region keys ScopePartitioner gives the nodes of a differentiated graph.

This file needs no device. `ScopePartitioner::assign` reads a graph, the segments of its forward
trace and the provenance of its backward nodes, and returns a `Partition`.
`include/SushiAI/eager/README.md`, under `scope_partitioner.hpp` and `partition.hpp`, states the
key rule the cases check: with `m` segments, key `k` forward, key `2m - k` backward, and `2m` for
the optimizer.

The cases fall into three groups, in file order:

- keys and region lists of traced and differentiated steps
- graphs built by hand to reach a K_LEAF origin or a graph with no backward node
- inputs that `assign` refuses

### Step

`Step` holds one traced, differentiated and partitioned training step: the graph, the segments,
the provenance, the parameters, one gradient per parameter and the partition.

### as_span

`as_span` returns a segment vector as the span `assign` takes.

### finish

`finish` seals the trace, differentiates the loss against `step.parameters` with a `Provenance`
and partitions the result. Every step builder ends with it, so the four steps differ only in
what they trace.

### linear

`linear` emits a K_MATMUL and a K_BIAS_ADD over a new weight and bias and appends both to the
parameter list. The parameter order is therefore weight, bias per layer, which is how a case
indexes `gradients`.

### unscoped_step

`unscoped_step` builds two linear layers and a loss with no scope open, which gives one segment.

### three_segment_step

`three_segment_step` builds scopes "first" and "second" with one linear layer and an activation
each, and a scope "loss". It has three segments, six leaves and seven other forward nodes.

### shared_weight_step

`shared_weight_step` declares one weight in a scope "weights", multiplies by it in "first" and
again in "second", and closes with "loss". It has four segments, and the first holds only a
K_LEAF.

### repeated_block_step

`repeated_block_step` opens a scope "block" twice in a loop, with one linear layer each time,
then "loss". The input is declared before the first scope, so the trace opens with an unnamed
segment that holds only a K_LEAF.

### TwoReluGraph

`TwoReluGraph` holds a graph built without differentiation: a parameter, a K_RELU in a segment
"first" and a K_RELU of that in a segment "second". `build_two_relu_graph` fills it, seals the
trace and sets `forward_node_count`, and leaves `origin` empty. A case then appends backward
nodes and writes their origins by hand, which reaches arrangements `differentiate()` does not
emit.

### is_leaf

`is_leaf` reports whether a node of a graph is a K_LEAF.

### segment_number

`segment_number` returns the number, counted from 1, of the segment that holds a forward node,
or 0 when none does. It is computed from the segment ranges and not from the partition, so a
forward key is compared against something the partitioner did not write.

### backward_keys_of

`backward_keys_of` returns the keys of the backward nodes whose origin is a non-K_LEAF node of
one segment. It skips K_LEAF origins because those take their key from their operands.

### AnUnscopedStepIsOneRegion

With one segment, every non-K_LEAF node of the forward and backward graph has key 1 and the
optimizer key is 2. The region list has one entry, named "#0" and marked forward: the empty
scope name followed by its occurrence.

### ForwardKeysFollowTraceOrder

Every non-K_LEAF forward node of `three_segment_step` has the number of its segment as its key.
The closing count of seven checked nodes fails if the loop skips nodes it should compare.

### BackwardKeysMirrorForwardKeys

With three segments the optimizer key is 6, and the backward nodes whose origins lie in segments
1, 2 and 3 have keys 5, 4 and 3. Each segment must have at least one such node, so an empty
result cannot satisfy the loop.

The region list has five entries with keys 1 to 5, named "first#0", "second#0", "loss#0",
"second#0", "first#0". The first three are forward and the last two backward.

### TheLastSegmentSharesItsRegionWithItsBackward

The K_NLL_LOSS node has key 3, and so does every backward node whose origin lies in the third
segment. The region list holds exactly one entry with key 3, and its direction is forward. A
list with a second, backward entry under the same key would describe one region twice.

### EveryOperandComesFromALowerOrEqualKey

For each of the four steps, every node's key is at least the key of the node that produces each
of its operands. Operands with no producer, or whose producer has no region, are skipped. The
check runs over every node of the graph and not over chosen ones, and each step must contribute
at least one comparison. `include/SushiAI/eager/README.md` gives this as the condition for a
valid partition under `RegionKey`.

### AWeightGradientLandsInItsOwnLayersBackwardRegion

The node that produces the second layer's weight gradient has key 4, the backward key of the
second segment. The case then asserts that this node reads a value produced under key 1. If a
change to the differentiation rules removed that read, the first assertion would hold for a
graph that no longer contains the case, and the message on the last assertion says so.

### ASharedParametersAccumulateSitsAboveBothContributors

The gradient of the shared weight is produced by an K_ACCUMULATE whose origin is a K_LEAF. Its two
operands are produced under keys 5 and 6, the backward keys of "second" and "first" among four
segments, and the K_ACCUMULATE has key 6. Key 7, the backward key of the segment that holds the
weight's K_LEAF, is not used for it.

### ALeafOriginNodeTakesItsHighestOperandKeyInEitherOrder

Two K_ADD nodes are appended to a `TwoReluGraph` with the K_LEAF as their origin. One reads the
key 2 value first and the key 1 value second, the other the reverse, and both get key 2. A rule
that took the key of the first or of the last operand would give one of them key 1.

### RegionsListOnlyKeysThatHoldANode

`shared_weight_step` has four segments and optimizer key 8. Its region list has five entries
with keys 2 to 6. Key 1 belongs to the K_LEAF-only segment "weights" and key 7 to its backward
mirror, no node carries either, and neither is listed.

### ARepeatedScopeIsNamedByItsOccurrence

`repeated_block_step` has four segments and optimizer key 8. The region list has keys 2 to 6,
named "block#0", "block#1", "loss#0", "block#1", "block#0", so the backward regions carry the
names of the forward regions they mirror.

### AGraphWithNoBackwardNodeHasForwardRegionsOnly

A `TwoReluGraph` with an empty `origin` gives its two K_RELU nodes keys 1 and 2 and optimizer
key 4. The region list holds "first#0" and "second#0", both forward. The optimizer key is `2m`
whether or not any backward node exists.

### LeavesBelongToNoRegion

In `three_segment_step` every K_LEAF has `K_NO_REGION` and every other node has a key. The count
of six leaves, two inputs and four parameters, shows that the loop reached the K_LEAF branch.

### SegmentsThatDoNotCoverTheForwardGraphThrow

Four malformed segment lists are refused, each with its own message: a list missing its last
segment ("The segments cover nodes 0 to", "forward graph has"), a list with a gap before segment
1 ("Segment 1 ('second')", "contiguous"), a list with an empty segment inserted ("Segment 1
('empty')", "holds no node"), and an empty list ("at least one segment").

### AProvenanceOfAnotherGraphThrows

A provenance with one origin too few is refused with a message containing "The provenance
describes" and "backward nodes". One whose last origin equals `forward_node_count`, the first
id that is not a forward node, is refused with "has origin" and "forward nodes".

### ABackwardNodeThatReadsAHigherKeyThrows

Two K_RELU nodes are appended to a `TwoReluGraph`, the second reading the first, with origins in
segments 1 and 2. The formula gives them keys 3 and 2, so the reader has the lower key. `assign`
throws, and the message contains "has region key 2" and "whose key is 3". The partitioner
therefore checks the ordering it produces and does not assume the provenance is consistent with
the graph.

### ALeafOriginNodeThatReadsOnlyLeavesThrows

An K_ADD of two parameters is appended with a K_LEAF as its origin. Neither operand's producer has
a region, so there is no key to take. The message contains "as its origin" and "reads only
leaves".

## test_scope_trace.cpp

Tests how ScopeTrace cuts a trace into segments and which calls it rejects.

This file needs no device. A `ScopeTrace` reads the node count of a graph, so every case
appends nodes to a host-side graph, opens and closes scopes around them and inspects what
`seal()` returns. `include/SushiAI/eager/README.md`, under `segment.hpp` and `scope_trace.hpp`,
defines a segment and the names a scope may have.

The cases fall into two groups, in file order: the segments a trace yields, and the calls
`open()`, `close()` and `seal()` reject.

### trace_nodes

`trace_nodes` appends a given number of K_ADD nodes to a graph, each adding one input value to
itself. The input is added as a value with no K_LEAF node, so the node count rises by exactly the
number asked for and a case can state every range as a literal.

### segment_is

`segment_is` compares one segment with an expected name, occurrence, begin and end. On a
mismatch it prints all four fields of both, so a failure shows which field differs.

### NodesOutsideAnyScopeFormOneUnnamedSegment

Three nodes traced with no scope open give one segment: empty name, occurrence 0, nodes 0 to 3.

### AScopeSplitsTheTraceIntoThreeSegments

One node, then two in a scope "body", then three give three segments. The unnamed run after the
scope has occurrence 1, because the unnamed run before it was occurrence 0.

### ANestedScopeIsNamedByItsPath

Inside "outer" and "inner" the depth is 2 and the segment is named "outer/inner". The nodes of
"outer" before and after it form two segments named "outer", with occurrences 0 and 1.

### AScopeOpenedTwiceGetsRisingOccurrences

A scope "cell" opened three times in a loop, with two nodes each time, gives three segments
named "cell" with occurrences 0, 1 and 2 and consecutive ranges.

### AnEmptyScopeLeavesNoSegment

A scope "cell" that closes with no node traced leaves no segment. The next "cell", which holds
one node, has occurrence 0, so the dropped scope did not use up an occurrence.

### SealWithAScopeStillOpenThrows

`seal()` throws while "outer" and "inner" are open, and the message contains "seal()", "still
open" and the path "outer/inner". The depth is still 2 afterwards.

The case then traces one more node, closes both scopes and seals. The result is one segment
"outer/inner" over nodes 0 to 2, which shows the failed `seal()` did not cut the running
segment.

### CloseWithNoScopeOpenThrows

`close()` throws on a new trace, and again after one matched `open()` and `close()`. Both
messages contain "close()" and "no scope open".

### SegmentsCoverEveryNodeExactlyOnce

The trace mixes unscoped nodes, a scope opened twice, a nested scope and an empty one. Each
segment begins where the previous one ended and holds at least one node, the first begins at 0,
and the last ends at the graph's node count.

### OpenRejectsAnEmptyName

`open("")` throws with a message containing "open()" and "empty scope name", and the depth
stays 0.

### OpenRejectsANameWithASlash

`open("outer/inner")` throws. The message contains the name and "may not contain", and the
depth stays 0.

### OpenRejectsANameWithAHash

`open("cell#0")` throws. The message contains the name and "may not contain", and the depth
stays 0.

### ARejectedNameDoesNotSplitTheRunningSegment

One node is traced, `open("a/b")` throws, and one more node is traced. `seal()` returns one
unnamed segment over nodes 0 to 2. A trace that ended the running segment before it validated
the name would return two.

### ARejectedCloseDoesNotSplitTheRunningSegment

The same arrangement with a `close()` that throws for having no scope open also returns one
unnamed segment over nodes 0 to 2.

### ASecondSealThrows

A second `seal()` throws with a message containing "seal()" and "sealed trace".

### OpenAndCloseAfterSealThrow

After `seal()`, `open("late")` and `close()` both throw with "sealed trace" in the message, and
the depth stays 0.

## test_shape.cpp

Tests Shape's extents, indexing, validation, equality and hash.

This file needs no device. Shape is host-lane vocabulary and must stay testable without a
RuntimeContext, which is what keeps the unit tier fast.

### HashSeparatesShapesThatDifferOnlyByRank

[4, 1] and [4] hold the same element count and the same leading extent. A hash that folded only
the extents would collide them, and the tensor pool keys its constant operands on this hash.

## test_silu.cpp

Tests the SiLU layer's trace, shape inference and empty state.

This file needs no device. `NN::SiLU` in `include/SushiAI/nn/silu.hpp` has no parameter and no
attribute.

Three `static_assert` lines at the top of the file hold that `SiLU` is a `Module`, is not a
`ModuleContainer`, and that `SiLU::fields()` is an empty tuple.

### ForwardEmitsOneSiluNodeReadingTheInput

`forward` adds one node. Its operation is `Ops::K_SILU`, its one input is the value passed in
and its one output is the value returned.

### ForwardKeepsTheShapeAndTypeOfAFeatureBatch

Over a [2, 4] input the result has the shape [2, 4] and the input's element type.

### ForwardKeepsTheShapeAndNhwcTagOfAnImageBatch

Over a [2, 8, 6, 3] input the result has that shape and the layout `TensorLayout::Nhwc`.

### ForwardCarriesTheLayoutOfItsInput

Over a value converted to NCHW with `layout_convert`, the result is tagged NCHW and has the
converted value's shape.

### OutputShapeReturnsItsInput

`output_shape([7])` is [7] and `output_shape([8, 6, 4])` is [8, 6, 4].

### TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times.

## test_spatial_shapes.cpp

Tests the shapes the builder infers for spatial operations and the layouts it refuses.

### Im2ColProducesTheColumnMatrixAGemmExpects

[N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5 windows per image. Each row is
one window's kernel_h * kernel_w * C taps with channels innermost, so a sub-group reads a
contiguous run (CV-3).

### Im2ColHonoursStrideAndDilation

A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows, and stride 2 over the same
input yields 2. Both numbers are wrong under the naive (H - K)/S + 1 that forgets dilation.

### Im2ColRejectsAWindowLargerThanThePaddedInput

The oversized window produces a zero extent, which Shape refuses anyway. The message a caller
gets should still name the geometry, not the Shape invariant it happened to trip.

### Col2ImReturnsToTheImageItWasAskedFor

col2im cannot derive its output extents. ConvGeometry carries pad_top and pad_left but no
pad_bottom or pad_right, so an asymmetric padding would be silently mis-shaped. The caller
states the image shape instead. It always has it, because it held the value im2col consumed.

### Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward

A 3x3 stride-1 window with padding only on the bottom and right of a 13x13 input still yields
13x13 windows, which is not what the symmetric formula (13-3)/1+1 = 11 would say. The forward
pass (`im2col_with_extents`) states 13x13 outright. The backward rule must be able to state the
same numbers back, because it is the one holding them, and must not re-derive them from
ConvGeometry alone.

The geometry leaves pad_top and pad_left at 0. Padding was applied on the bottom and right
only, which ConvGeometry cannot express and does not need to.

### Col2ImDerivingStillRejectsThatSameAsymmetricPair

This case is the point of having `col2im_with_extents` as a twin. The deriving col2im assumes
symmetric padding and is not a substitute for it. Feeding it the exact cols/image pair that
`col2im_with_extents` accepted in Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward must
still throw, which proves the twin is needed.

### Im2ColRejectsAnNchwInput

im2col reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it would read channels
as height and produce a plausible but silently wrong shape, where it should throw.

### MaxPoolProducesAValueAndItsArgmaxTogether

The indices are a second result of the node, not a second output tensor the caller allocates.
The backward rule needs exactly the indices this forward produced, and a pair returned together
cannot be mismatched the way two separate calls could.

### MaxPoolHandlesYoloTinysAsymmetricThirteenToThirteen

YOLOv3-tiny's last pool is 2x2 stride 1 over 13x13, padded only on the bottom and right, staying
13x13. ConvGeometry cannot express that padding, so the caller states the extents. This case
proves the explicit overload is needed.

### AvgPoolProducesOnlyAValue

avgpool2d records no indices: its backward divides by a constant window area and needs nothing
the forward recorded (CV-6).

### MaxPoolRejectsAnNchwInput

maxpool2d reads extents off axes 1, 2, 3 assuming NHWC. Handed an NCHW value it must throw and
must not pool the channel axis.

### SpatialMeanReducesToNAndC

The result is the [N,C] that the numeric Probe's loss needs, which is why every convolutional
gradient test in this plan ends with this call.

### BatchNorm2dKeepsItsShapeAndDeclaresThreePerChannelStatistics

`GraphBuilder::batchnorm2d` over a [2, 8, 8, 16] FLOAT64 input returns four values: the output,
of the input's shape, and `saved_mean`, `saved_inv_std` and `batch_variance`, each [16]. They
are the four outputs of one K_BATCHNORM2D node in that order, and attribute 0 of the node is the
epsilon 1e-5. Each of the four is FLOAT64 with the lifetime `ACTIVATION`. A gamma of [8]
throws.

A non-positive epsilon is a division by the standard deviation itself, which is exactly zero
for a constant channel.

### BatchNorm2dRejectsAnNchwInput

The layout check runs before any extent is read, so the operand that is wrong is the one the
message names.

### BatchNormFoldYieldsAPerChannelScaleAndShift

`GraphBuilder::batchnorm_fold` over a [16] gamma, beta, running mean and running variance adds
one node. Its operation is `Ops::K_BATCHNORM_FOLD`, its inputs are those four values in that
order, its outputs are `fold.scale` and `fold.shift`, and attribute 0 is the epsilon 1e-3.
Both results are [16] FLOAT64 values of lifetime `ACTIVATION`, the operands' type. A
`channel_affine` of a [2, 8, 8, 16] input with the two results keeps that shape.

### BatchNormFoldRejectsAnOperandThatDoesNotMatchGamma

Five calls throw with "batchnorm_fold" and the operand at fault: a [8] mean ("mean", "[16]",
"[8]"), a [8] beta ("beta"), a FLOAT64 variance among FLOAT32 operands ("variance", "float32",
"float64"), a [4, 4] gamma ("rank-1") and four INT64 operands ("int64"). The graph's node count
and value count are unchanged after all five.

### BatchNormFoldRejectsANonPositiveEpsilon

An epsilon of 0.0 and one of -1e-3 each throw with a message containing "batchnorm_fold",
"epsilon" and "positive". An epsilon of 1e-3 does not throw.

### ReshapeRefusesToChangeTheElementCount

A reshape has one failure mode. It moves no data, so a mismatched count is not a resize; it is
a read past the end.

### Convolution, window extents and upsampled extents

The cases from here cover three additions to `include/SushiAI/graph/builder.hpp`.
`GraphBuilder::conv2d` emits a convolution as a node sequence: a K_RESHAPE of the rank-4 weight
to [C_out, taps], an K_IM2COL of the input, a K_MATMUL of the two with its second operand
transposed, an optional K_BIAS_ADD, and a K_RESHAPE of the rows to [N, out_h, out_w, C_out].
`Graph::window_extents` computes the output extents of a window and validates its geometry,
and `conv2d`, `im2col`, `maxpool2d` and `avgpool2d` all derive their extents through it.
`Graph::upsampled_extents` does the same for K_UPSAMPLE2D.

### op_names_from

`op_names_from` returns the operation names of a graph's nodes from a given index on. A case
records the node count after it has declared its operands and passes that count, so the list
holds only what the emitter under test added.

### square_window

`square_window` returns a unit-stride geometry with a square kernel and the same padding on
both axes.

### window

`window` returns a unit-stride unpadded geometry with the given kernel height and width.

### Conv2dReturnsAnNhwcValueWithOneChannelPerFilter

A [2, 5, 5, 3] input, an [8, 3, 3, 3] weight and an [8] bias under a padded 3x3 window give a
[2, 5, 5, 8] value tagged NHWC.

### Conv2dDividesTheOutputExtentsByItsStride

A padded 3x3 window with stride 2 over a [2, 8, 6, 3] input gives [2, 4, 3, 4]. The call
passes `K_NO_VALUE` as the bias.

### Conv2dWithABiasEmitsIm2colMatmulAndBiasAddBetweenTwoReshapes

The nodes the call adds are, in order, "sushiai.reshape", "sushiai.im2col", "sushiai.matmul",
"sushiai.bias_add" and "sushiai.reshape". The sequence holds no operation of its own, which is
decision CV-1: `test_ops.cpp` asserts under ThereIsNoConv2dIdentity that no CONV2D identity
exists.

### Conv2dWithoutABiasEmitsNoBiasAdd

With `K_NO_VALUE` as the bias the same call adds four nodes: reshape, im2col, matmul, reshape.

### Conv2dContractsTheColumnMatrixAgainstTheTransposedFilterMatrix

The third node added is a K_MATMUL. Its first operand has shape [2 * 5 * 5, 3 * 3 * 3], one row
per window, and its second has shape [8, 3 * 3 * 3], one row per filter. Flag 0 is false and
flag 1 is true, so the filter matrix is the transposed operand and the product has one column
per filter.

### Conv2dPointwiseEmitsNoIm2col

A 1x1 unit-stride unpadded window adds reshape, reshape, matmul, reshape: the input is
reshaped to one row per pixel and no K_IM2COL is emitted. The result is [2, 5, 5, 8].

### Conv2dStridedOneByOneKernelStillEmitsIm2col

A 1x1 window with stride 2 over a [2, 6, 6, 3] input adds reshape, im2col, matmul, reshape and
gives [2, 3, 3, 8]. A stride above one selects pixels, which a reshape cannot do, so the
shortcut of the previous case must not apply.

### Conv2dTakesANonSquareKernelInHeightThenWidthOrder

An [8, 1, 3, 3] weight under a 1x3 window over a [2, 5, 5, 3] input gives [2, 5, 3, 8]. Axis
1 of the weight is the kernel height and axis 2 the kernel width.

### Conv2dRejectsAWeightWhoseKernelExtentsDifferFromTheWindow

An [8, 3, 1, 3] weight under the same 1x3 window throws. It holds as many elements as
[8, 1, 3, 3], so the reshape that follows accepts it and only the check in `conv2d` refuses
the transposed kernel.

### Conv2dRejectsAWeightWhoseInputChannelsDifferFromTheImage

An [8, 3, 3, 4] weight over a three-channel input throws with a message containing "conv2d".
The reshape rejects the same weight by element count in a message that does not name the
convolution, so the case requires the name.

### Conv2dRejectsAFlattenedWeight

An [8, 27] weight throws. The weight enters `conv2d` as rank 4 and is flattened inside it.

### Conv2dRejectsARankThreeInput

A [5, 5, 3] input, which lacks the batch axis, throws.

### Conv2dPointwiseRejectsAnNchwInput

A [2, 5, 5, 3] input is converted to NCHW and paired with an [8, 1, 1, 5] weight, whose
input-channel extent matches the last axis of the converted shape. The pointwise path skips
K_IM2COL, which is the emitter that checks the layout, so `conv2d` has to check it itself. The
call throws.

### Conv2dRejectsABiasWhoseExtentDiffersFromTheFilterCount

A [4] bias with eight filters throws.

### Conv2dRejectsAWindowLargerThanThePaddedInput

A 5x5 unpadded window over a [1, 2, 2, 3] input throws.

### Conv2dEmitsNothingWhenItRejectsItsOperands

Three calls throw: an oversized window, a weight with four input channels over a
three-channel input, and a [4] bias with eight filters. The node count afterwards equals the
count taken before the calls. `conv2d` validates every operand before it emits its first
reshape, so a refused call leaves no orphan node in the graph.

### Conv2dBuildsAGraphThatFinaliseAccepts

A biased 3x3 convolution feeds a pointwise one, and `Graph::finalise()` does not throw. Both
paths therefore leave every value with a layout tag the per-node proof accepts.

### Conv2dDifferentiatesToAWeightAndABiasGradient

The convolution's result goes through `spatial_mean`, `log_softmax` and `nll_loss`.
`Autograd::differentiate` returns two gradients other than `K_NO_VALUE`, of shapes [8, 3, 3, 3]
and [8]. The weight gradient has the rank-4 shape of the parameter and not the [8, 27] shape
of the reshaped operand the K_MATMUL read.

### WindowExtentsPadBothSidesOfAnAxisByTheLeadingPad

A 3x5 window with a top pad of 1 and a left pad of 2 over a 7 by 9 input yields 7 by 9. The
extents are unchanged only if each pad is applied to both sides of its axis.

### WindowExtentsHonourStrideAndDilationPerAxis

A 3x3 window with stride 2 by 3 and a height dilation of 2 over an 11 by 16 input yields 4 by
5. The two axes differ in stride and in dilation, so a value taken from the wrong axis changes
one of the two numbers.

### WindowExtentsOfAWindowThatExactlyFitsAreOneByOne

A 5x5 window with padding 1 over a 3 by 3 input yields 1 by 1. The padded input is exactly as
large as the window.

### WindowExtentsRejectANonPositiveKernel

A kernel of 0 by 3 and one of 3 by -2 throw. Each message contains "positive kernel" and the
kernel as given, "kernel 0x3" and "kernel 3x-2".

### WindowExtentsRejectANonPositiveStrideAndNameIt

A vertical stride of 0 and a horizontal stride of -1 throw with "positive stride" and the
pair, "stride 0,1" and "stride 1,-1". A zero stride is a division by zero in the extent
formula, and the message names the stride and not a window that does not fit.

### WindowExtentsRejectANonPositiveDilation

A horizontal dilation of 0 throws with "positive dilation" and "dilation 1,0".

### WindowExtentsRejectANegativePadding

A padding of -1 on both axes throws with "padding -1,-1", and a left pad of -1 alone throws
with "padding 0,-1".

### WindowExtentsRejectAWindowThatDoesNotFit

A 5x5 window over a 4 by 6 input throws with a message containing "5x5 window" and "4x6
input". The window fits the width and not the height.

### UpsampledExtentsScaleEachAxisByItsOwnFactor

A 3 by 5 input with scales 2 and 4 yields 6 by 20.

### UpsampledExtentsRejectAScaleBelowOne

Scales of 0 by 2 and of 2 by -1 throw with "at least one" and the pair, "0x2" and "2x-1".

### UpsampledExtentsRejectAProductBeyondSixtyFourBits

With a scale of 2^62, an extent of 5 on the height axis throws with "fit in 64 bits", and so
does an extent of 5 on the width axis. Each call overflows one axis and leaves the other at
1, so both products are checked.

### UpsampledExtentsAcceptTheLargestProductThatFits

Half of the largest int64, rounded down, scaled by 2 yields the largest int64 minus 1. The
overflow check therefore refuses nothing that 64 bits hold.

### UpsampleRejectsAScaledExtentBeyondSixtyFourBits

`GraphBuilder::upsample2d` with a horizontal scale of 2^62 over a width of 5 throws with "fit
in 64 bits", and the node count is unchanged.

### TheDerivingEmittersRejectANegativePadding

`im2col`, `conv2d`, `maxpool2d` and `avgpool2d` are each called with a padding of -1, and each
message contains "padding". These are the four emitters that derive their extents from the
geometry.

### Col2ImRejectsANegativePadding

`col2im` with a padding of -1 throws with a message containing "padding".

### Col2ImRejectsAnImageShapeThatIsNotRankFour

`col2im` asked for the image shape [5, 5, 3] throws with a message containing "col2im" and
"rank-4".

### TheDerivingEmittersNameTheLayoutBeforeTheWindow

A [1, 8, 8, 2] input is converted to NCHW, which leaves 2 on the axis an NHWC reader takes
for the height, and a 3x3 window does not fit that. `im2col`, `maxpool2d` and `avgpool2d` each
throw a message containing their own name and "NHWC". A caller who passed the wrong layout is
told so, and is not told that the window is too large.

## test_status.cpp

Tests Status, its macros, the fatal family and the assertions of test_status.hpp.

This file needs no device. `Status` and the `SA_FAIL_IF`, `SA_TRY` and `SA_ANNOTATE` macros are
in `include/SushiAI/core/status.hpp`, and the fatal checks are in
`include/SushiAI/core/check.hpp`. The assertions `SA_EXPECT_OK`, `SA_EXPECT_FAILURE` and
`SA_EXPECT_FATAL` come from `tests/common/test_status.hpp`. `SA_EXPECT_FATAL` is a death test:
the statement runs in a child process and the needles are matched against what it wrote to
stderr.

Six `static_assert` lines at the top of the file hold that a `Status` is the size of one
pointer and that its default construction, copy, move and both assignments are `noexcept`.

Several cases read `Detail::live_failure_count()`. A failed `Status` owns one failure block and
a successful one owns none, so a count that did not move means the code under test allocated
nothing for a failure.

### counted

`counted` returns its value and increments a counter. A case passes it as a format argument to
find out whether a macro evaluated its arguments.

### failure_of

`failure_of` builds a failure of a given code and message through `Status::from_message`.

### check_extent

`check_extent` fails with `StatusCode::InvalidDescription` through `SA_FAIL_IF` when its extent
is not positive. The message's argument goes through `counted`.

### forward_extent

`forward_extent` calls `check_extent` under `SA_TRY` and sets a flag after it, so a case can
tell whether control went past the macro.

### FatalRecord

`FatalRecord` holds the call count, file, line and message of the last report the recording
handler received.

### fatal_record

`fatal_record` returns the one static `FatalRecord` of the file.

### record_fatal

`record_fatal` is a `FatalHandler` that stores its report in `fatal_record()`.

### announce_fatal

`announce_fatal` is a `FatalHandler` that writes "recorded: " and the message to stderr, where
a death test reads it.

### stop_when

`stop_when` runs `SA_FATAL_IF` on its condition with the message "extent {} is refused" and
returns the line number of that check. The extent goes through `counted`.

### stop_with_throwing_composer

`stop_with_throwing_composer` calls `Detail::fail_fast_composed` with a composer that throws
`std::runtime_error`.

### stop_under_announcer

`stop_under_announcer` installs `announce_fatal` and then runs `SA_FATAL("stop {}", 3)`.

### assert_positive

`assert_positive` runs `SA_ASSERT(value > 0)`. It is compiled only when `NDEBUG` is not
defined.

### DefaultIsSuccessWithNoMessage

A default-constructed `Status` reports `is_ok()`, the code `StatusCode::Ok` and an empty
message, and `SA_EXPECT_OK` accepts it.

### EveryCodeHasALowerCaseName

`status_code_name` returns "ok", "invalid_description", "file_not_found", "invalid_file",
"incompatible", "unsupported", "out_of_memory" and "device_failed" for the eight codes, and
"unknown" for the value 200, which is no enumerator.

### FromMessageCarriesTheCodeAndTheText

A status built from `StatusCode::InvalidFile` and "the header is 3 bytes short" is not ok and
returns that code and that text.

### SuccessPathCreatesNoFailureBlock

The case constructs, copies, moves, copy-assigns and move-assigns successful statuses and
calls `annotated` on one. The annotation's composer never runs, both results are ok, and
`live_failure_count()` is what it was before.

### FailIfFormatsOnlyWhenTheConditionHolds

`check_extent(4)` succeeds with zero calls to `counted` and no failure block. `check_extent(-2)`
fails with `InvalidDescription` and a message containing "extent must be positive" and
"got -2", after one call to `counted`. The count of live blocks is back at its start once the
temporary is gone.

### CopiesShareOneBlockAndTheLastOneFreesIt

A failure, a copy of it and a status assigned from the copy leave `live_failure_count()` one
above its start, and the third status's `message().data()` is the first one's pointer. After
the scope closes the count is back at its start.

### MovedFromStatusKeepsItsFailure

A failed status is moved from twice, once by construction and once by assignment. The source
and both targets still report `Unsupported` and "no kernel for int64".

### AssignmentReplacesTheHeldFailure

Assigning a second failure to a status that holds one leaves one live block and the second
failure's code and message. A self-assignment through a const reference keeps the failure.
Assigning `Status()` makes the status ok and frees the block.

### AnnotatedPrefixesTheMessageAndKeepsTheCode

`SA_ANNOTATE(inner, "Conv2d({}, {})", 3, 16)` over the failure "stride must be positive"
returns a status with the same code and the message "Conv2d(3, 16): stride must be positive".
The inner status's message is unchanged.

### ThrowingComposerYieldsTheStaticOutOfMemoryFailure

`Status::from_message` with a composer that throws returns a failure of code `OutOfMemory`
whose message contains "message could not be built". `live_failure_count()` does not move, so
that failure is not a heap block.

### ThrowingAnnotationKeepsTheOriginalFailure

`annotated` with a composer that throws returns the code `InvalidFile` and the message "wrong
magic" of the status it was called on.

### TryForwardsAFailureAndPassesASuccess

`forward_extent(0)` returns `check_extent`'s failure, with "got 0" in the message, and leaves
its flag false. `forward_extent(5)` succeeds and sets the flag.

### FromMessageWithTheOkCodeIsFatal

`Status::from_message(StatusCode::Ok, ...)` stops the process with a report containing
"Status::from_message needs a failure code" and "got ok".

### StatusNamesRejectsSuccessAWrongCodeAndAMissingNeedle

`Test::status_names` is called directly. Over the failure `InvalidFile` "wrong magic 0x1F" it
is true for the right code with both needles and with no needle. It is false for a successful
status, for the code `FileNotFound` and for the needle "wrong version". `Test::status_is_ok` is
false for the failure.

### DeathPatternEscapesAndJoinsItsNeedles

`Test::death_pattern` is called directly. One plain needle is returned as it is. Two needles
are joined with `.*`, and each of the characters `. [ ] ( ) * { } | ^ $ + ? \` is escaped with
a backslash.

### FatalHandlerIsReplacedByARecorderAndRestored

`set_fatal_handler(&record_fatal)` returns a handler that is not null. One call of
`Detail::report_fatal("layer.cpp", 42, "a sealed trace was reused")` reaches the recorder
once, with that file, line and message. Restoring the original returns `&record_fatal`, and
setting the original again returns the original.

### NullFatalHandlerInstallsTheDefaultOne

`set_fatal_handler(nullptr)` returns the handler that was installed, `&record_fatal`. The
handler in place after it is not null and is not the recorder.

### FatalIfPassesAndFormatsNothingWhenTheConditionIsFalse

`stop_when(false, ...)` returns and never calls `counted`.

### FatalIfPrintsTheMessageThenTheFileAndTheLineAndStops

`stop_when(true, ...)` stops the process with a report containing "sushiai: fatal: extent 7 is
refused" followed by "test_status.cpp:", the line `stop_when` returned and ")". The case calls
`stop_when(false, ...)` first to learn that line.

### FatalCheckWhoseComposerThrowsReportsTheFixedMessage

`stop_with_throwing_composer` stops the process with a report containing "sushiai: fatal:",
"its message could not be composed" and "test_status.cpp:".

### ReplacedFatalHandlerRunsBeforeTheStop

`stop_under_announcer` stops the process and its stderr contains "recorded: stop 3", the line
`announce_fatal` writes.

### AssertChecksInDebugBuildsOnly

With `NDEBUG` defined, `SA_ASSERT(counted(evaluations, 0) == 1)` does not evaluate its
condition: the counter stays at zero. Without it, `assert_positive(1)` returns and
`assert_positive(0)` stops the process with "assertion failed: value > 0".

## test_step_recorder.cpp

Tests that StepRecorderList drives its members as one recorder, in the order added.

This file needs no device. `Graph::StepRecorder` and `Graph::StepRecorderList` are in
`include/SushiAI/graph/step_recorder.hpp`. The list is itself an `StepRecorder` and holds
references to its members; it does not own them.

### JournalRecorder

`JournalRecorder` appends "<name>.record", "<name>.before" or "<name>.after" to a journal the
test owns, and `record_into_step()` returns the task count given at construction.

### RecordOnlyRecorder

`RecordOnlyRecorder` overrides `record_into_step()` alone and returns 3. Its `before_replay()`
and `after_replay()` are the interface's own.

### in_order_added

`in_order_added` returns the three journal entries that recorders named a, b and c write for
one call.

### EmptyListRecordsNothing

An empty list reports `empty()`, returns 0 from `record_into_step()` and does not throw from
either replay hook.

### AddMakesTheListNonEmpty

After one `add` the list is not empty and the journal is still empty, so `add` calls nothing
on the member.

### RecordSumsTheMemberCounts

Members that report 2, 0 and 5 tasks give `record_into_step()` 7.

### RecordCallsMembersInTheOrderAdded

`record_into_step()` over a, b and c journals "a.record", "b.record", "c.record".

### OrderFollowsAddNotConstruction

With b added before a, the journal is "b.record", "a.record", although a was constructed
first.

### BeforeReplayCallsMembersInTheOrderAdded

`before_replay()` over a, b and c journals "a.before", "b.before", "c.before".

### AfterReplayCallsMembersInTheOrderAdded

`after_replay()` over a, b and c journals "a.after", "b.after", "c.after". The members are not
called in reverse.

### MemberWithoutReplayHooksIsDriven

A list of a `RecordOnlyRecorder` and a journal recorder of 4 tasks records 7. After both replay
hooks the journal holds "a.record", "a.before" and "a.after" only.

### ListIsDrivenThroughTheInterface

Through an `StepRecorder*`, a list of a and b records 5 tasks and journals the record calls,
then the before calls, then the after calls, each as a then b.

### NestedListKeepsItsPlaceInTheOrder

An outer list of a, an inner list holding b, and c records 7 tasks and journals "a.record",
"b.record", "c.record".

### AddRejectsARecorderAlreadyHeld

A second `add` of the same recorder throws with a message containing "StepRecorderList::add"
and "already". The list still records that member once: 2 tasks and one journal entry.

### AddRejectsTheListItself

`list.add(list)` throws with a message containing "StepRecorderList::add" and "itself", and
the list stays empty.

## test_throwing_bridge.cpp

Tests bridge_throwing: a value passes through and a thrown Error becomes a Status.

This file needs no device. `NN::Detail::bridge_throwing` in
`include/SushiAI/nn/throwing_bridge.hpp` runs a callable that may throw and returns its value
as a `Result`.

### refuse_a_shape

`refuse_a_shape` throws an `Error` through `SA_THROW_IF` with the message "Linear(6, 8) was
given [5]".

### refuse_an_allocation

`refuse_an_allocation` throws `std::bad_alloc`.

### throw_a_logic_error

`throw_a_logic_error` throws `std::logic_error("an index was out of range")`, which is neither
an `Error` nor `std::bad_alloc`.

### bridge_a_logic_error

`bridge_a_logic_error` passes `throw_a_logic_error` to `bridge_throwing`, as the statement of
a death test.

### AValuePassesThrough

A callable that returns 7 gives an ok `Result<int>` holding 7.

### AMoveOnlyValuePassesThrough

A callable that returns `std::make_unique<int>(7)` gives an ok result whose pointer is not
null and owns 7.

### AnErrorBecomesInvalidDescriptionWithItsMessage

Bridging `refuse_a_shape` gives an `InvalidDescription` failure whose message contains
"Linear(6, 8)" and "[5]".

### ARefusedAllocationBecomesOutOfMemory

Bridging `refuse_an_allocation` gives an `OutOfMemory` failure whose message contains
"allocation was refused".

### AnotherExceptionIsFatal

The case is in the suite `Unit_ThrowingBridgeDeathTest`. `bridge_a_logic_error` stops the
process with a report containing "other than Error" and "an index was out of range".

## test_trace.cpp

Tests Trace, the buffer-update sinks, the three layer forms and trace_layer.

This file needs no device. `NN::Trace` in `include/SushiAI/nn/trace.hpp` carries a
`Graph::GraphBuilder`, a `TraceMode` and the sink that receives buffer updates. A layer has one
of three forms: `Module`, whose `forward` takes a builder; `TraceModule`, whose `forward` takes
a `Trace`; and `ModalModule`, which states `trace_training` and `trace_evaluation` and no
`forward`. `NN::trace_layer` calls whichever form a layer has.

### K_EXACT_TOLERANCE

`K_EXACT_TOLERANCE` is 1e-15, the absolute tolerance for a double that is stored and read back
unchanged.

### ModalStub

`ModalStub` has the modal pair and an empty `fields()`. `trace_training` emits one ReLU and
`trace_evaluation` returns its input, and each counts its calls.

### CompositeStub

`CompositeStub` lists a `ModalStub` as the child "inner". Its `forward` takes a `Trace` and
passes it to the child through `NN::trace_layer`.

### ByValueTraceStub

`ByValueTraceStub` has a `forward` that takes its `Trace` by value.

### TrainingOnlyStub

`TrainingOnlyStub` states `trace_training` and no `trace_evaluation`.

### ModalWithForwardStub

`ModalWithForwardStub` states the modal pair and a builder `forward` together.

### BothForwardsStub

`BothForwardsStub` has a builder `forward` and a trace `forward` and counts the calls of each.

### FieldlessStub

`FieldlessStub` has the modal pair and no `fields()`.

### RefusingStub

`RefusingStub` has a trace `forward` that records an `InvalidDescription` failure on the trace
through `Trace::fail` and returns its input.

### failure_saying

`failure_saying` returns an `InvalidDescription` failure with a given message.

### defined_buffer

`defined_buffer` returns a `Buffer` defined as [3] with the starting value 0.0 that no graph
has read.

### update_undefined_buffer

`update_undefined_buffer` requests an update of a buffer on which `define()` was never called.
It is the statement of a death test.

### update_from

`update_from` requests an update whose source is a given id, in a graph that holds one value.
It is the statement of two death tests.

### parameter_names

`parameter_names` collects the names `NN::for_each_parameter` yields, in walk order.

### The layer forms at compile time

The `static_assert` lines after the helpers hold which form each type has. `Linear` is a
`Module` only, `ModalStub` a `ModalModule` only and `CompositeStub` a `TraceModule` only, and
all three are `Traceable`. A `Sequential` of `Module`s is both a `Module` and a `TraceModule`.
A `Sequential` that holds a modal layer or a trace-only composite, at any depth, is a
`TraceModule` and not a `Module`, and is still a `ModuleContainer`. `BothForwardsStub` is a
`Module` and a `TraceModule`.

`Detail::HasHalfModalPair` detects `TrainingOnlyStub` and `Detail::HasModalMemberAndForward`
detects `ModalWithForwardStub`; neither detects `ModalStub` or `Linear`. `TrainingOnlyStub`,
`FieldlessStub`, `ByValueTraceStub` and `int` are not `Traceable`.

`Trace` is final. It is not copy-constructible, move-constructible or default-constructible,
and a `GraphBuilder` does not convert to it. `NN::BufferUpdateSink` has a virtual destructor.

### AFactoryStatesItsModeAndKeepsItsBuilder

`Trace::training(builder, list)` reports `TraceMode::Training` and `Trace::evaluation(builder)`
reports `TraceMode::Evaluation`. `builder()` of each returns the address of the builder it was
made from.

### UpdateUnderAnEvaluationTraceFails

`update()` on an evaluation trace returns `InvalidDescription` with a message containing
"evaluation trace".

### ANewListIsEmpty

A new `BufferUpdateList` has no update and no source.

### EachUpdateAddsOneUpdateAndOneSource

Two `update()` calls on a training trace leave two entries in `updates()` and two in
`sources()`. Entry 0 holds the first buffer's state, the source x, the momentum 0.1 and the
source scale 1.0. Entry 1 holds the second buffer's state, the source y, 0.25 and 1.5.
`sources()` is x then y.

### AnUpdateAddsNothingToTheGraph

An `update()` leaves the graph's value count unchanged and the buffer without a leaf in it.

### ARecordedTargetSurvivesAMoveOfItsBuffer

After the buffer of a recorded update is move-constructed into another, the update's `target`
equals the moved buffer's state and is not `K_NO_STATE`.

### TheDiscardingSinkAcceptsAnUpdateAndIsOneObject

A trace made with `NN::discard_updates()` is in Training mode, its `update()` returns ok and
the graph gains no value. Two calls of `discard_updates()` return the same address.

### AnUpdateOfAnUndefinedBufferIsFatal

The case is in the suite `Unit_TraceDeathTest`. `update_undefined_buffer` stops the process
with "define() has not given a shape".

### AnUpdateOfABufferThatWasNeverReadIsRecorded

An update of a defined buffer that no graph has read is recorded with the buffer's state as
its target. The graph gains no value and the buffer has no leaf in it.

### ANewTraceHoldsNoFailure

A new trace's `status()` is ok and the graph has no refusal.

### FailKeepsTheFirstFailureAndMarksTheGraphRefused

After `fail` with "the first reason" and then "the second reason", `status()` carries the
first message and does not contain "second". `graph.has_refusal()` is true.

### ValueOrFailReturnsAHeldValue

`value_or_fail(Result<ValueId>(y), x)` returns y and leaves `status()` ok.

### ValueOrFailRecordsAFailureAndReturnsTheFallback

`value_or_fail` over a failed result returns the fallback x, and `status()` then carries that
failure.

### FailGivenASuccessIsFatal

The case is in the suite `Unit_TraceDeathTest`. `trace.fail(Status())` stops the process with
"Trace::fail was given a success".

### SequentialThrowsTheFailureAChildRecorded

`Sequential{RefusingStub, ModalStub}::forward` over a trace throws with a message containing
"RefusingStub refuses". The second child's `trace_evaluation` is never called.

### SequentialFoldsAnInputShapeThroughItsChildren

`output_shape([4])` of `Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` is [3].
`output_shape([5])` throws with a message containing "Linear(4, 8)" and "[5]".

### AnUpdateWithoutASourceIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is `K_NO_VALUE` stops the
process with "a buffer update reads value".

### AnUpdateFromAValueOutsideTheGraphIsFatal

The case is in the suite `Unit_TraceDeathTest`. An update whose source is value 7, in a graph
of one value, stops the process with a report containing "reads value 7" and "holds 1 values".

### TraceLayerCallsTheTrainingFormInTrainingMode

Under a training trace, `trace_layer` calls `ModalStub::trace_training` once and
`trace_evaluation` never. The result differs from the input, because the training form emits a
ReLU.

### TraceLayerCallsTheEvaluationFormInEvaluationMode

Under an evaluation trace, `trace_layer` calls `trace_evaluation` once and `trace_training`
never, and returns its input.

### TraceLayerTracesAModuleThroughTheBuilder

`trace_layer` on a `Linear{4, 3}` and `Linear::forward` on a second one, each in its own graph,
return the same id and leave the same value count. The traced result has the shape [2, 3].

### TraceLayerPassesTheTraceToATraceModule

`CompositeStub` is traced once under an evaluation trace and twice under a training trace. Its
child counts one evaluation call and two training calls.

### TraceLayerPrefersTheTraceForwardOverTheBuilderForward

On `BothForwardsStub`, `trace_layer` calls the trace `forward` once and the builder `forward`
never.

### ANestedSequentialTracesItsModalChildInTheTracesMode

A `ModalStub` inside a `Sequential` inside a `Sequential` counts one evaluation call after an
evaluation trace of the outer container, and one training call after a training trace of it.

### SequentialTracesAModalChildInTheTracesMode

`Sequential{Linear{4, 3}, ModalStub}` traced under each mode calls the stub's matching form
once and yields a [2, 3] value both times. Its parameter walk yields "0.weight" and "0.bias".

### SequentialOfModulesTracesTheSameThroughBothForwards

`Sequential{Linear{4, 8}, ReLU, Linear{8, 3}}` traced through the trace `forward` in one graph
and through the builder `forward` in another returns the same id, leaves the same value count
and walks the same parameter names.

## test_traced.cpp

Tests Traced and the refusal it carries through a chain of emitters.

This file needs no device. `Graph::Traced` in `include/SushiAI/graph/traced.hpp` holds a
`ValueId` or a failed `Status`. An emitter that returns one takes its operands as `Traced`
too, so a call can be nested inside another and the first refusal travels to the outermost
result.

Six `static_assert` lines at the top of the file hold that a `Traced` is default-constructible
and constructible from a `ValueId` and from a `Status` without throwing, that both conversions
are implicit, and that its size is two pointers.

The cases up to BuildingFromASuccessStatusIsFatal use `StandInBuilder` and involve no graph.
The last four use a `Graph::GraphBuilder` and read `Graph::has_refusal()`.

### TraceLog

`TraceLog` counts the values the stand-in builder emitted and the refusals it recorded.

### StandInBuilder

`StandInBuilder` has two members. `emit()` returns the next id, and `refuse()` counts one
refusal and returns the failure it was given. `SA_REFUSE_IF` is run over it in `emit_below`.

### emit_below

`emit_below` binds its operand through `SA_TRY_ASSIGN`, refuses through `SA_REFUSE_IF` when
the id is above a limit, and otherwise emits one value.

### emit_over

`emit_over` binds its operand through `SA_TRY_ASSIGN` and emits one value. It has no condition
of its own, so it accepts the absent operand.

### forward_traced

`forward_traced` runs `SA_TRY` on a `Traced` and sets a flag after it.

### declare_below

`declare_below` runs `SA_REFUSE_IF` over a `Graph::GraphBuilder` when an extent is above a
limit, and otherwise declares one input of that extent.

### description_failure

`description_failure` builds an `InvalidDescription` failure with a given message.

### DefaultHoldsTheAbsentValueAndNoFailure

A default-constructed `Traced` has a value, an ok status and the id `Graph::K_NO_VALUE`.

### HoldsAnIdWithASuccessStatus

`Traced(ValueId{12})` has a value, is ok and returns 12.

### HoldsAFailureAndTheAbsentId

A `Traced` built from a failure has no value and reports the failure's code and message.

### SuccessPathCreatesNoFailureBlock

Two chained emitters over an absent operand, a copy and a move yield the id 1, record no
refusal and leave `live_failure_count()` where it was.

### MovedFromTracedKeepsItsState

A failed `Traced` that was moved from still reports its failure, as does the target. A
`Traced` holding 4 that was move-assigned from still returns 4, as does the target.

### AssignmentReplacesAFailureWithAnId

Assigning `ValueId{3}` to a failed `Traced` leaves a value and the id 3. Assigning a failure to
it afterwards leaves that failure.

### AnAbsentOperandPassesThroughAnEmitter

`emit_over(builder, Traced())` succeeds with the id 0, and the log shows one emitted value.

### RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing

`emit_below` given the id 9 and the limit 4 fails with "operand 9 is above the limit 4",
records one refusal and emits nothing. That result is then passed through `emit_below` with
the limit 100, nested inside `emit_over`. The outer result carries the same message, and its
`message().data()` is the first result's pointer, so the failure was forwarded and not rebuilt.
The log still shows one refusal and no emitted value.

### NestedCallsEmitWhenNothingIsRefused

`emit_over(builder, emit_below(builder, ValueId{2}, 4))` yields the id 1 with two values
emitted and no refusal.

### TryForwardsATracedFailureAsAStatus

`forward_traced` over a failed `Traced` returns the failure and leaves its flag false. Over
`ValueId{1}` it succeeds and sets the flag.

### ValueOnAFailureIsFatalAndPrintsTheFailure

`value()` on a failed `Traced` stops the process with a report containing "Traced::value() was
called on a failure" and the failure's message.

### BuildingFromASuccessStatusIsFatal

`Traced(Status())` stops the process with "a Traced was built from a success Status".

### RefuseMarksTheGraphAndReturnsTheFailure

`GraphBuilder::refuse` returns the failure it was given, with the same message pointer, and
`graph.has_refusal()` goes from false to true.

### RefuseIfOverAGraphBuilderMarksTheGraphOnlyOnARefusal

`declare_below(builder, 4, 8)` declares value 0 and leaves the graph without a refusal.
`declare_below(builder, 9, 8)` fails with "extent 9 is above the limit 8", adds no value and
marks the graph.

### ObserveMarksTheGraphForAFailedStatusOnly

`GraphBuilder::observe(Status())` is ok and leaves the graph unmarked. `observe` of a failed
status returns that failure and marks the graph.

### ObserveMarksTheGraphForAFailedResultOnly

`observe(Result<int>(7))` returns a result holding 7 and leaves the graph unmarked. `observe`
of a failed `Result<int>` returns the failure and marks the graph.

## test_unflatten.cpp

Tests the trace, the shape rule and the rejected inputs of the Unflatten layer.

This file needs no device. `NN::Unflatten` in `include/SushiAI/nn/unflatten.hpp` holds a
height, a width and a channel count and no parameter. It reshapes an
[N, height * width * channels] value to [N, height, width, channels]. Images enter a model
flat, as [N, features], so this layer is what gives a convolution its rank-4 input.

### per_sample

`per_sample` returns the shape of a graph value with the batch axis removed.

### SatisfiesTheModuleContract

Three `static_assert` lines require `Unflatten` to be a `Module`, not to be a
`ModuleContainer`, and to have no parameter walk of its own, so the walk reaches it through
`fields()`.

### TracesTheImageShape

`Unflatten{4, 5, 3}` traces a FLOAT64 [2, 60] input to a FLOAT64 [2, 4, 5, 3] value.

### TagsTheResultNhwc

The result is rank 4 and tagged NHWC, and `finalise()` accepts the graph.

### EmitsOneReshapeNode

The trace adds exactly one node. The producer of the result is a K_RESHAPE whose one operand is
the input.

### OutputShapeAgreesWithTheTraceForAnRgbImage

`output_shape()` of the input's per-sample shape equals the per-sample shape of the traced
result, and `output_shape([60])` is [4, 5, 3].

### OutputShapeAgreesWithTheTraceForAGreyImage

The same two assertions hold for `Unflatten{28, 28, 1}` over a [7, 784] input, the MNIST
geometry: `output_shape([784])` is [28, 28, 1].

### ForwardRejectsAZeroExtent

`Unflatten{4, 0, 3}` throws with a message containing "Unflatten needs positive extents". The
element-count check would also throw for this layer, in a message that does not name the
extents, so the case requires the layer's own message.

### ForwardRejectsANegativeExtent

`Unflatten{-4, -5, 3}` throws the same message. The two negative extents multiply to a
positive count of 60, which matches the input, so no later check refuses the layer by count;
`Shape`'s constructor would throw, in a message that does not name the layer.

### ForwardRejectsAnInputThatIsNotRankTwo

A [2, 60, 1, 1] input throws. Its second extent is 60 and it holds 120 elements, so only the
rank check can refuse it.

### ForwardRejectsAMismatchedElementCount

A [3, 40] input to `Unflatten{4, 5, 3}` throws with a message containing "Unflatten(4, 5, 3)"
and "inner extent is 60". `reshape` would also reject the input, in a message that does not
name the layer, so the case requires the layer's own.

### ForwardRejectsAnNchwInput

The layout tag of a [2, 60] input is set to NCHW by hand, and `forward()` throws.

### OutputShapeRejectsAZeroExtent

`Unflatten{4, 5, 0}.output_shape([60])` throws with a message containing "Unflatten needs
positive extents".

### OutputShapeRejectsANegativeExtent

`Unflatten{-4, -5, 3}.output_shape([60])` throws the same message.

### OutputShapeRejectsAShapeThatIsNotRankOne

`output_shape([60, 1])` throws.

### OutputShapeRejectsAMismatchedElementCount

`output_shape([59])` throws.

### HasNoParameters

`parameter_count` is 0 before and after a trace and `parameter_elements` is 0. The trace
raises the value count by one, the reshaped value, so it declares no parameter.

## test_upsample2d.cpp

Tests the shape, layout and rejections of the Upsample2d layer.

This file needs no device. `NN::Upsample2d` in `include/SushiAI/nn/upsample2d.hpp` holds one
integer scale, 2 by default, and no parameter. It repeats every pixel of an NHWC image over a
scale-by-scale block.

Two `static_assert` lines at file scope require `Upsample2d` to be a `Module` and not a
`ModuleContainer`.

### traced_per_sample_shape

`traced_per_sample_shape` traces a layer over a new input of the given batched shape and
returns the result's shape with the batch axis dropped. The forward rejection cases call it
too, since it throws whatever `forward()` throws.

### K_OVERFLOWING_SCALE

`K_OVERFLOWING_SCALE` is 2^62. Its product with an extent of 5 does not fit in 64 bits.

### TheDefaultScaleDoublesHeightAndWidth

A default layer traces a [2, 3, 5, 7] input to [2, 6, 10, 7].

### TheScaleFieldSetsTheFactorOnBothSpatialAxes

`Upsample2d{3}` traces a [1, 4, 2, 5] input to [1, 12, 6, 5].

### TheResultIsTaggedNhwc

The traced result is rank 4 and tagged NHWC.

### ForwardEmitsOneUpsampleNodeCarryingTheScale

The trace of `Upsample2d{3}` adds exactly one node. The producer of the result is an
K_UPSAMPLE2D whose one operand is the input, and attribute slots 0 and 1 both hold 3.

### OutputShapeScalesHeightAndWidthOnly

A default layer maps [3, 5, 7] to [6, 10, 7], and `Upsample2d{4}` maps [2, 1, 9] to [8, 4, 9].
The channel extent is unchanged in both.

### OutputShapeAgreesWithTheTracedShape

For scales 2, 3 and 1, `output_shape()` equals the traced shape without its batch axis. A
scale of 1 is the smallest the layer accepts and leaves the shape as it is.

### OutputShapeRejectsAZeroScale

`Upsample2d{0}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=0".

### OutputShapeRejectsANegativeScale

`Upsample2d{-2}.output_shape([3, 5, 7])` throws with a message containing "Upsample2d" and
"[3, 5, 7] with scale=-2".

### OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws with a message containing "Upsample2d", the expected form
"[H, W, C]" and "[16]".

### OutputShapeRejectsAScaledHeightThatOverflows

With `K_OVERFLOWING_SCALE`, `output_shape([5, 1, 7])` throws with a message containing
"Upsample2d" and "fit in 64 bits". The width is 1, so the height is the product that
overflows.

### OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws. The shape still carries its batch axis.

### ForwardRejectsAZeroScale

`forward()` of `Upsample2d{0}` over a [1, 2, 2, 3] input throws with a message containing
"Upsample2d" and "[2, 2, 3] with scale=0".

### ForwardRejectsARankTwoInput

`forward()` over a [2, 12] input throws with a message containing "Upsample2d", "rank-4" and
"[2, 12]".

### ForwardRejectsAScaledWidthThatOverflows

With `K_OVERFLOWING_SCALE`, `forward()` over a [1, 1, 5, 3] input throws with a message
containing "Upsample2d" and "fit in 64 bits". The height is 1, so the width is the product
that overflows.

### ForwardRejectsAnNchwInput

A [1, 3, 2, 2] input is converted to NCHW, and `forward()` throws with a message containing
"upsample2d" and "NHWC". The lower-case name is the builder's: `GraphBuilder::upsample2d`
raises this message.

### TheParameterWalkVisitsNothing

After a trace the walk visits nothing and `parameter_count` is 0.

## test_version.cpp

Tests that the runtime version agrees with the compile-time macros.
