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
with three truths, two assigned to head 0 and one to head 1. An `AnchorTrainingStep` runs the
training graph around its host stage, with plain SGD at a rate of 0.1.

### TheLeavesFedAreTheDeltasOfTheHeads

After one run, the delta of each head is computed on the host from the head the step published
with `compute_head_delta`. The leaf the training graph was fed equals that delta element for element,
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

### AHostStageThatReadsAValueTheGraphLacksIsRefused

`Train::TrainingStep::make` over the training graph, with a host stage that reads the id one
behind the last value, fails with `InvalidDescription`, "The host stage reads value" and
"values".

### AHostStageBehindAReaderOfItsLeafIsRefused

The stage reads the loss and writes the delta leaf of head 0. The loss is computed from that
leaf, so `make` fails with `InvalidDescription`, "before the host stage" and "which the stage
writes".

### AStepWithoutAHostStageRefusesHostWork

A `Train::TrainingStep` made with no host stage reports `has_host_stage()` false and a null
`lead()`. `run` with host work fails with `InvalidDescription` and "without a host stage",
the work is not called and the run count stays 0.

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
without its files.

The run of 2026-10-12 on this model set the bounds of the step cases, by the owner's ruling of that
day: each bound that run exceeded is now about 1.5 times the largest difference it measured. The
heads were 0.176, 0.228 and 0.243 from the dump on values near 30 to 49, the [yolo] outputs at most
0.0261, a box entry of the delta 0.00235 and any other entry 0.0039, the costs 1.24e-3 relative,
and an update of a head convolution 0.0091 of the dump's largest magnitude. The two differences
`test_darknet_yolov3_tiny_training.cpp` above describes are the likely cause; where the difference
grows, layer by layer, is not measured on this model.

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

Each head of the Separate model under batch statistics is within 3.6e-1 absolute plus 1e-4 relative
of the dump, and the [yolo] output computed from it within 4e-2 absolute plus 1e-4 relative.

### TheDeltasOfTheStepMatchDarknet

The deltas the step fed are compared with the dumped ones: a box entry within 3.5e-3 absolute plus
1.5e-4 relative, any other entry within 6e-3. The summed cost and its mean over the heads are
compared with the layer costs and the network cost to 2e-3 relative. The bounds do not cover a slot
that crosses the ignore threshold.

### TheUpdatesOfTheHeadConvolutionsMatchDarknet

The weight and bias gradients of the convolution before each head, times minus the batch, are
compared with the dumped updates. The bound is 1.4e-2 of the dump's largest magnitude plus 1e-3 of
the entry.

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
case is skipped without its files.

The run of 2026-10-13 on this model, the first under the default memory limit, set the bounds of
the step cases by the owner's ruling of that day: each bound that run exceeded is now about 1.5
times the largest difference it measured. The heads were 0.205, 0.104 and 0.065 from the dump on
values near 40, the [yolo] outputs at most 0.0316, a box entry of the delta 0.0020 and any other
entry 0.0043, and an update of a head convolution 0.0047 of the dump's largest magnitude. The
cause is not known; YOLOv3 above differs by as much.

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

### TheCostOfOneBatchFallsUnderAThird

200 steps on the batch and the truth of `yolov7_tiny_train_step` bring the stored cost of the
Separate model below 0.33 of the first step's.

The share is 0.33 where the other five models state 0.1, by the owner's ruling of 2026-10-12.
CPU Darknet, run for 200 steps on the same batch, weights and learning rate with no burn-in,
takes its summed layer cost from 30.72 to 6.93, which is 0.226 of the first, and its lowest
step holds 4.27. SushiAI goes from 30.71 to 6.63, which is 0.216. Neither reaches a tenth.
The two curves agree at step 0 and are not compared step by step after it. Why the cost
stops there is not known.

## test_darknet_yolov7_training.cpp

Suites: `Integration_DarknetYolov7Training`, in `sushiai_vision_tests`.

The fixture `DarknetTrainingTest` reads `yolov7.cfg`, `yolov7.weights` and two runs under
`references/` in the directory `SA_REFERENCE_DATA_DIR` names: `yolov7_plain_pool_thresh_0p01`, one
image of YOLOv7 with its deltas and assignments, and `yolov7_train_step`, one training step of two
images. The box loss is `BoxLoss::CompleteIoU` and the objectness target is
`ObjectnessTarget::BestIoU`. Every case is skipped without its files.

The run of 2026-10-13 on this model, the first under the default memory limit, set three bounds
of the step cases by the owner's ruling of that day, each about 1.5 times the largest difference
that run measured: the heads were at most 0.0052 from the dump, the [yolo] outputs the same, and
an entry of the delta that is no box entry 0.00103. The other bounds held. The cause of the
differences is not known.

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
