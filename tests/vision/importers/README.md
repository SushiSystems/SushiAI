# tests/vision/importers

The tests of the module `vision/importers`, one file per header of `modules/vision/importers/include/`. The suite prefix of a case gives its ctest label.

## test_darknet_config.cpp

Suites: `Unit_DarknetConfig`.

Tests `parse_darknet_config`, which reads a config text into a `DarknetConfig` of one network
section and a list of layer sections, each with its options and line numbers, and
`load_darknet_config`. Every refusal of the parser has the code `InvalidDescription`.

### ParsesEveryLexicalFormOfTheGrammar

A text with `#` and `;` comment lines, a blank line, indented lines, tabs and spaces around
`=`, and spaces inside comma lists parses to a "network" section on line 4 with four options
and to three layers, "convolutional", "route" and "yolo". Each option has its key, its value
and its line. A list value loses its inner spaces ("10,14,23,27"), an empty value stays empty,
and a value keeps a second `=` ("a=b").

### IgnoresCarriageReturnsAtLineEnds

A text with CRLF line ends parses to the same keys and values, with "width" on line 2, the
layer on line 3 and "filters" on line 4.

### AcceptsOneKeyInTwoSections

"width" in the network section and again in a layer section parses, and the layer's value is
"8" on line 4.

### RefusesAKeyBeforeAnySection

A key on line 1 before any section fails with a message holding "test.cfg:1" and
"key 'width'".

### RefusesALineThatIsNeitherSectionNorOption

A line without `=` fails with a message holding "test.cfg:2".

### RefusesAnEmptyKey

A line that starts with `=` fails with a message holding "test.cfg:2".

### RefusesAKeyRepeatedWithinASection

"width" twice in one section fails with a message holding "test.cfg:3", "key 'width'" and
"on line 2".

### RefusesAFirstSectionThatIsNotNetOrNetwork

A text whose first section is `[convolutional]` fails with a message holding "test.cfg:1" and
"[convolutional]".

### RefusesAMalformedSectionLine

A section line without its closing bracket fails with a message holding "test.cfg:1" and
"']'". An empty section name and text after the closing bracket each fail on "test.cfg:1".

### RefusesATextWithoutASection

A text of one comment line fails with a message holding "test.cfg" and "[net]".

### MissingFileGivesFileNotFound

`load_darknet_config` of a path that does not exist fails with `FileNotFound` and names the
path.

## test_darknet_fold.cpp

Suites: `Unit_DarknetFold`.

Tests `fold_darknet_normalisation`, which folds the scales, rolling means and rolling
variances of a `DarknetConvolution` into its weights and biases with the epsilon 0.00001.

### OneFilterFoldMatchesTheFormulaInLongDouble

For one filter of eight weights with the bias 0.75, the scale 1.5, the mean 0.25 and the
variance 0.5, each folded weight equals the weight times 1.5 / sqrt(0.5 + epsilon), and the
bias equals 0.75 - 1.5 * 0.25 / sqrt(0.5 + epsilon). Both are computed in long double and
compared within 1e-6 relative plus 1e-9 absolute. After the fold the scales, means and
variances are empty.

### FoldedAndUnfoldedConvolutionsAgreeOnTheHost

For three filters over a two-channel 5x5 input with values drawn from the seed 20261009, the
folded convolution plus its bias and the unfolded convolution followed by the normalisation
differ by at most 1e-5 at every output pixel.

### FoldingTwiceStopsTheProcess

A second fold of the same convolution stops the process with a message holding
"not normalised".

### NonPositiveVarianceStopsTheProcess

A rolling variance of -1 stops the process with a message holding "positive variance".

## test_darknet_import.cpp

Suites: `Unit_DarknetImport`.

Tests `import_darknet_model` and `import_darknet_files` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_import.hpp`. The sample
`.cfg` text describes a 32 by 32 network of three channels: a normalised convolution of 4
filters, a plain convolution of 18 filters and one `[yolo]` head of one class with the mask
0, 1, 2 over the anchors 4,8, 8,16 and 16,24. The cases parse it with `parse_darknet_config`
and translate it with `translate_darknet_model`.

### KeepsTheRowsAndTheOutputsOfTheTranslation

Imported without weights, the model has the translation's input shape and the same type in
every row. It has one output, named "head.0", at the node the translation's first output
names.

### StatesTheHeadsAsAnAnchorDetectionTask

The task section has the type "anchor_detection" and reads through `read_anchor_task` as one
class and one head on the output "head.0" with `HeadCoding::Logit` and the mask {0, 1, 2}.
It has three anchors; the first has the width 4/32 and the third the height 24/32, within
1e-6.

### HoldsNoTensorWithoutWeights

An import with a null weights pointer has an empty `weights` list.

### NamesEveryTensorAsItsTargetDoes

With weights of the lengths the translation's shapes state, the names of the imported tensors
are the set of the non-empty names in the translation's targets, and there are seven tensors.

### CarriesTheRecipeAndItsNotes

The imported `training` value has the keys `objective`, `optimizer`, `schedule`, and the
import carries five notes.

### AnotherCountOfFilesIsRefused

`import_darknet_files` fails with `InvalidDescription` for no file ("'darknet'", "0 files")
and for three files (".cfg", "3 files").

### AMissingCfgIsRefusedByName

`import_darknet_files` of a `.cfg` path that does not exist fails with `FileNotFound` and the
path's name.

Suites: `Integration_DarknetImport`, in `sushiai_vision_importers_tests`.

The suite runs once for each model of the parity table in `modules/vision/darknet_forward_cases/source/darknet_forward.cpp`:
`yolov3-tiny`, `yolov3`, `yolov4-tiny`, `yolov4`, `yolov7-tiny` and `yolov7`. The fixture
`DarknetImportTest` imports the model in memory from its `.cfg` and `.weights` in the directory
`SA_REFERENCE_DATA_DIR` names. A case is skipped when a file of its model is missing.

### TheImportedRowsAndTaskAreThoseOfTheTranslation

The imported model file, without its task, is the text of the translated model. The task reads back
as the anchors, the heads and the suppression criterion.

### TheImportedWeightsAreThoseOfTheDirectStore

A state filled from the imported tensors equals, byte for byte, the state `store_darknet_weights`
fills. The imported tensors supply every entry exactly once.

## test_darknet_model.cpp

Suites: `Unit_DarknetModel`, `Unit_DarknetModelState`.

Tests the translation of a config text into a `DarknetModel`: the rows of its model spec, the
row each layer ends on, its heads, its anchor layout, the shape of each convolution and the
state names its weights go to. Most cases translate a 24-layer text with two heads that the
file holds, under `DarknetNormalisation::Separate` or `DarknetNormalisation::Folded`. The
refusal cases use a small 32x32 network. `Unit_DarknetModelState` is an engine fixture.

### SeparateFormHasTheCountedRowsAndSources

The Separate form has 43 rows, the input shape [416, 416, 3] and the `row_of_layer` list the
case states. Rows 0 to 2 are `conv2d`, `batchnorm2d` and `leaky_relu`, with a slope of 0.1, a
`bn_epsilon` of 1e-5 and no convolution bias; row 33 has a bias. Row 34 is a `conv2d` that reads
row 29, and row 38 is a `concat` of rows 37 and 18. Row 23 is a `maxpool2d` of kernel 2, stride 1
and padding 0 with a trailing padding of 1 and 1, and row 3 has a trailing padding of 0 and 0.

### FoldedFormHasNoNormalisationRow

The Folded form has 32 rows, none of the type `batchnorm2d`, and its own `row_of_layer` list.
Row 0 has a bias, row 25 reads row 21 and row 28 reads rows 27 and 13. There are 13 targets,
and the first names "0.weight" and "0.bias" and leaves scale, mean and variance empty.

### HeadsCarryTheirOutputsOptionsAndAnchors

The model has two heads and two outputs, "head.0" on layer 16 and row 33 and "head.1" on layer
23 and row 42, each with 80 classes, the Logit coding and a `scale_xy` of 1. The criterion is
`SuppressionCriterion::IoU` with the exponent 0.6. The six anchors are their pixel sizes
divided by 416, and the masks are {3, 4, 5} and {0, 1, 2}.

### ShapesCountTheFloatsOfTheWeightsFile

The Folded form lists 13 convolution shapes on the layers 0, 2, 4, 6, 8, 10, 12, 13, 14, 15,
18, 21 and 22. A shape is normalised unless it has 255 filters. Summed over the shapes, the
biases, normalisation vectors and weights come to 8858734 floats, and the twelfth convolution
reads 384 channels.

### SeparateFormBuildsAndItsTargetsAreStateEntries

In the Separate form the first target names "0.weight", "1.bias", "1.weight", "1.running_mean"
and "1.running_var", and there are 11 * 5 + 2 * 2 target names. The model builds, and every
target names an entry of its state.

### FoldedFormBuildsAndItsTargetsAreStateEntries

In the Folded form there are 13 * 2 target names. The model builds, and every target names an
entry of its state.

### AnActivatedHeadReadsItsStretchAndItsSuppression

A head with `new_coords=1`, `scale_x_y = 2.0`, `nms_kind=diounms` and `beta_nms=0.6` among its
keys gives one head with the Activated coding, a `scale_xy` of 2 and one class. The criterion is
`SuppressionCriterion::DistanceIoU` with the exponent 0.6. The model has two rows, the second a
`sigmoid`, and the output is published from row 1.

### OnlyDiounmsReadsBetaNMS

With `nms_kind=greedynms` and `beta_nms=0.3` the criterion is `DistanceIoU` and the exponent
stays 0.6. With `nms_kind=diounms` and `beta_nms=0.3` the exponent is 0.3.

### TheLastHeadDecidesTheCriterion

A text whose first head sets `nms_kind=diounms` and whose second head sets nothing has two
heads, the criterion `IoU` and the exponent 0.6.

### AnUnknownSectionIsRefusedNamingItsLine

A `[reorg]` section fails with `Unsupported`, "line 11", "layer 1" and "[reorg]".

### AnUnknownKeyIsRefusedNamingItsLine

Three unknown keys fail with `Unsupported`: `groups` in a convolution ("line 7",
"[convolutional]", "'groups'"), `label_smooth_eps` in a head ("line 16", "[yolo]",
"'label_smooth_eps'") and `crop` in the first section ("line 5", "'crop'").

### UntranslatedValuesAreRefusedNamingTheirLine

Three values fail with `Unsupported`: `letter_box=1` ("line 5"), an upsample `scale=2`
("line 11", "scale 2") and `activation=relu` ("line 6", "'relu'").

### ValuesThatDoNotFitAreRefusedNamingTheirLine

Five texts fail with `InvalidDescription`: `filters=many` ("line 6", "'filters'", "'many'"), a
route to a later layer ("line 12", "names layer 1"), a head over 12 channels that needs 18
("line 8", "reads 18 channels", "has 12"), a mask that names anchor 3 of three ("line 12",
"names anchor 3"), and a text with no head ("no [yolo] section").

Suites: `Unit_DarknetRouteGroups`.

Tests how the translation of a config text turns a `[route]` section with `groups` and
`group_id` into model rows. Each case starts from a network whose first convolution has 64
filters and ends with a head, and reads the rows, their sources and the channels of the
convolutions of the `DarknetModel`.

### OneSourceGroupIsOneSliceRow

A route of the layer before it with `groups=2` and `group_id=1` gives three rows: a `conv2d`, a
`slice_channels` that reads row 0 with the offset 32 and the count 32, and a row that reads the
slice. The second convolution reads 32 channels.

### EverySourceIsSlicedBeforeTheJoin

A grouped route of two layers gives six rows. Rows 2 and 3 are `slice_channels` rows over rows
1 and 0, each with the offset 32 and the count 32. Row 4 is a `concat` of rows 2 and 3, and row
5 reads it. The third convolution reads 64 channels.

### UngroupedSourceAddsNoRow

A route of one layer without groups gives two rows, the second a `conv2d` that reads row 0, and
the second convolution reads 64 channels.

### UnequalSplitsAndGroupsOutOfRangeAreRefused

Four routes fail with `InvalidDescription`: `groups=3` over 64 channels ("line 12", "groups 3",
"64 channels"), `group_id=2` with two groups ("line 11", "group_id 2"), `groups=0` ("line 11",
"groups 0"), and a `group_id` of 1 without `groups` ("line 11", "group_id 1").

Suites: `Unit_DarknetShortcut`.

Tests how the translation of a config text turns a `[shortcut]` section into a model row. Each
case starts from an 8x8 three-channel network with two convolutions.

### NegativeFromAddsTheLayerBeforeAndTheNamedLayer

`from=-2` gives four rows. Row 2 has the type "add", no attribute key, and the sources 1 and 0.

### AbsoluteFromAddsTheSameTwoRows

`from=0` with `activation=linear` gives the same "add" row with the sources 1 and 0.

### FromAboveTheShortcutIsRefusedNamingItsLine

`from=2` fails with `InvalidDescription`, "line 18" and "only layers 0 to 1".

### DifferentChannelCountsAreRefusedNamingItsLine

A shortcut over convolutions of 4 and 5 filters fails with `InvalidDescription`, "line 18" and
"share their shape".

### NonLinearActivationIsRefusedNamingItsLine

`activation=leaky` fails with `Unsupported`, "line 19" and "'leaky'".

### WeightsTypeIsRefusedNamingItsLine

`weights_type=per_feature` fails with `Unsupported`, "line 19" and "weights_type".

### MissingFromIsRefusedNamingTheSection

A shortcut without `from` fails with `InvalidDescription` and "no 'from'".

### MoreThanOneFromIsRefused

`from=-1,-2` fails with `InvalidDescription` and "names one layer".

## test_darknet_recipe.cpp

Suites: `Unit_DarknetRecipe`.

Tests `write_darknet_recipe` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_recipe.hpp`. Each case
parses `.cfg` text, reads it with `read_darknet_training` and writes the recipe. The sample
`[net]` section states `batch=8`, `subdivisions=2`, `momentum=0.9`, `decay=0.0005`,
`learning_rate=0.001`, `burn_in=100`, `max_batches=2000`, `policy=steps`, `steps=1600,1800`
and `scales=.1,.1`. A `[yolo]` section states a mask, one class, `jitter=.3` and an
`ignore_thresh`.

### WritesTheSolverAsSGD

The `optimizer` section has three keys: the type "sgd", a `momentum` of 0.9 and a
`weight_decay` of 0.0005, each within 1e-12. It has no `learning_rate`.

### WritesTheRateAsAStepSchedule

The `schedule` section reads through `read_step_schedule_section` as a base rate of 0.001, 100
warm-up iterations, the steps 1600 and 1800 and the scales 0.1 and 0.1.

### WritesTheSharedHeadOptionsAsTheAnchorObjective

With two heads of `ignore_thresh=.7`, the `objective` section has the type "anchor" and reads
through `read_head_training_section` with an ignore threshold of 0.7 and
`BoxGradient::DarknetCompatible`.

### HoldsTheObjectiveTheOptimizerAndTheSchedule

The recipe's `training` value has the keys `objective`, `optimizer`, `schedule`.

### NotesEveryKeyWithoutAPlace

The recipe has four notes. One holds "line 2", "batch=8", "[net]" and "batch_size"; one
"line 3", "subdivisions=2" and "[net]"; one "line 8", "max_batches=2000" and "epochs"; one
"line 15", "jitter=.3" and "[yolo]".

### LeavesNoNoteWhenEveryKeyHasAPlace

A `.cfg` of `learning_rate=0.01` and a head of `mask=0` and `classes=1` gives a recipe with no
note.

### HeadsThatDifferAreRefused

Heads with `ignore_thresh` .7 and .5 fail with `InvalidDescription`, "head 0 (line 12)",
"head 1 (line 17)", "'ignore_threshold'" and "one set".

### AFileWithoutAHeadIsRefused

The `[net]` section alone fails with `InvalidDescription`, "[yolo]" and "no objective".

## test_darknet_tensors.cpp

Suites: `Unit_DarknetTensors`.

Tests `list_darknet_tensors` in
`modules/vision/importers/include/SushiAI/vision/importers/darknet_tensors.hpp`. The weights
are a plain convolution of 2 filters over 3 channels with a 2 by 2 kernel, whose weights are
0 to 23, and a normalised convolution of 2 filters over 1 channel with a 1 by 1 kernel. The
targets name `0.weight` and `0.bias` for the first, and `1.weight`, `2.bias`, `2.weight`,
`2.running_mean` and `2.running_var` for the second.

### ListsOneTensorPerNamedTarget

Seven tensors are listed. Each is `Float32` and its payload has the byte count its record
states.

### WeightsMoveToKernelLast

The tensor `0.weight` has the shape [2, 2, 2, 3]. Its 24 values open with 0, 4, 8, 1 and end
with 23.

### EveryVectorLandsUnderItsTarget

`2.weight` has the shape [2] and holds the scales 2, 3. `2.bias` holds 1.5, -1.5,
`2.running_mean` 0.25, 0.75 and `2.running_var` 4, 9. No tensor is named `0.running_mean`.

### AVectorOfTheWrongLengthIsRefused

With a third scale on the second convolution, the listing fails with `Incompatible`,
"layer 1", "'2.weight'" and "3 values".

### AnEntryNamedTwiceIsRefused

With the second target's bias named `0.bias`, the listing fails with `Incompatible`,
"layer 1", "'0.bias'" and "already fills".

### AShapeThatIsNotPositiveIsRefused

With 0 channels in the first shape, the listing fails with `InvalidDescription`, "layer 0"
and "channels 0".

### MismatchedCountsAreRefused

With one target for two convolutions, the listing fails with `Incompatible`,
"2 convolutions" and "the targets 1".

## test_darknet_training.cpp

Suites: `Unit_DarknetTraining`.

Tests `read_darknet_training` over texts parsed with `parse_darknet_config`. A head's options
are checked as seven values: box loss, objectness target, box weight, object weight,
`max_delta`, match threshold and ignore threshold. Floats are compared to 1e-6 and doubles to
1e-12.

### ReadsTheV3HeadSettingsWhenTheTrainingKeysAreAbsent

A head with only `ignore_thresh=0.7` and `truth_thresh=1` among the training keys reads as
`BoxLoss::SquaredError`, `ObjectnessTarget::One`, a box weight of 0.75, an object weight of 1,
a `max_delta` of 0, a match threshold of 1 and an ignore threshold of 0.7.

### ReadsTheV4TinyHeadSettings

With `iou_loss=ciou` and `iou_normalizer=0.07`, the head reads as `BoxLoss::CompleteIoU` with a
box weight of 0.07. The other five values are as in the case above.

### ReadsTheV4HeadSettings

With `iou_thresh=0.213` and `max_delta=5` added, the head reads a match threshold of 0.213 and
a `max_delta` of 5.

### ReadsTheV7HeadSettings

With `objectness_smooth=1`, `iou_normalizer=0.05`, `iou_thresh=0.2` and `max_delta=2`, the head
reads `ObjectnessTarget::BestIoU`, a box weight of 0.05, a match threshold of 0.2 and a
`max_delta` of 2. `cls_normalizer=0.5` is accepted.

### KeepsTheSolverDefaultsWhenTheKeysAreAbsent

With no solver key, the solver has a learning rate of 0.001, a momentum of 0.9, a decay of
0.0005, a `burn_in` of 0, no steps and no scales, a batch of 1, 1 subdivision and a
`max_batches` of 0.

### ReadsEverySolverKeyTheFileSpells

Batch 64, subdivisions 16, momentum 0.949, decay 0.0005, learning rate 0.00261, `burn_in` 1000,
`max_batches` 500200, the steps 400000 and 450000 and the scales 0.1 and 0.1 are each read to
the value the text states.

### KeepsTheHeadsInFileOrderAndSkipsOtherSections

A text with a convolution, a head, a route and a second head gives two heads, with the box
weights 0.07 and 0.05 in that order.

### ReadsMSEAndObjectnessSmoothZeroAsTheDefaults

`iou_loss=mse` with `objectness_smooth=0` gives the seven values of a head without training
keys.

### SelectsTheDarknetCompatibleBoxGradientForEveryHead

A head with `iou_loss=ciou` and a head with `iou_loss=mse` both read
`BoxGradient::DarknetCompatible`. A default `HeadTrainingOptions` has `BoxGradient::Analytic`.

### AcceptsTruthThreshOneAndLabelSmoothZero

`truth_thresh=1.0` with `label_smooth_eps=0` reads as one head.

### RefusesAPolicyOtherThanSteps

`policy=exp` fails with `Unsupported`, "line 2" and "policy".

### RefusesATruthThreshOtherThanOne

`truth_thresh=0.9` fails with `Unsupported`, "line 3" and "truth_thresh".

### RefusesANonZeroLabelSmooth

`label_smooth_eps=0.1` fails with `Unsupported`, "line 3" and "label_smooth_eps".

### RefusesAnIoULossOtherThanMSEAndCIoU

`iou_loss=giou` fails with `Unsupported`, "line 3" and "iou_loss".

### RefusesAClsNormalizerThatIsNotANumber

`cls_normalizer=half` fails with `InvalidDescription`, "line 3" and "cls_normalizer".

### RefusesAnObjectnessSmoothOtherThanZeroOrOne

`objectness_smooth=2` fails with `InvalidDescription`, "line 3" and "objectness_smooth".

### RefusesAHeadWhoseMatchThresholdIsZero

`iou_thresh=0` fails with `InvalidDescription`, "line 2", the head section's line, and
"match threshold".

### RefusesAHeadWhoseIgnoreThresholdIsAboveOne

`ignore_thresh=1.5` fails with `InvalidDescription`, "line 2" and "ignore threshold".

### RefusesALearningRateThatIsNotANumber

`learning_rate=abc` fails with `InvalidDescription`, "line 2", "learning_rate" and "abc".

### RefusesANonFiniteLearningRate

`learning_rate=nan` fails with `InvalidDescription`, "line 2" and "nan".

### RefusesABatchBelowOne

`batch=0` fails with `InvalidDescription` and a message holding "batch".

### RefusesAnEmptyEntryInAList

`steps=1,,2` fails with `InvalidDescription`, "line 2" and "steps".

### RefusesStepsAndScalesOfDifferentLengths

Two steps with one scale fail with `InvalidDescription`, "line 1", "'steps' holds 2" and
"'scales' holds 1".

## test_darknet_weights.cpp

Suites: `Unit_DarknetWeights`.

Tests `read_darknet_header` over byte streams built in memory and `read_darknet_weights` over
files the cases write under the system's temporary directory. The sample file holds a 20-byte
header and index-valued floats for two convolutions, a plain one on layer 0 and a normalised
one on layer 2.

### ReadsAVersionZeroTwoHeaderWithA64BitSeen

A header of major 0, minor 2 and revision 5 reads an eight-byte seen count, and the stream
stands at byte 20.

### ReadsAVersionZeroOneHeaderWithA32BitSeen

A header of major 0 and minor 1 reads a four-byte seen count, and the stream stands at byte
16.

### ReadsAMajorOneHeaderWithA64BitSeen

A header of major 1 and minor 0 reads an eight-byte seen count, and the stream stands at byte
20.

### AcceptsAMajorAtTheLimitOfOneThousand

A header of major 1000 is read.

### RefusesAMajorAboveOneThousandAsUnsupported

A major of 1001 fails with `Unsupported` and a message holding "1001".

### RefusesAMinorAboveOneThousandAsUnsupported

A minor of 1001 fails with `Unsupported` and a message holding "1001".

### RefusesAHeaderThatEndsEarly

A stream that ends two bytes into the revision fails with `InvalidFile` and a message holding
"revision".

### ReadsEveryVectorInFileOrder

With floats 0 to 21 in the file, the plain convolution's biases start at 0 and its weights at
2. The normalised one's biases start at 4, its scales at 7, its rolling means at 10, its
rolling variances at 13 and its weights at 16. The header reads major 0 and minor 2.

### KeepsTheBatchNormVectorsOnlyForNormalisedLayers

The plain convolution has empty scales, means and variances and two weights. The normalised
one has three of each vector and six weights.

### DecodesEveryBitPatternAsItIs

Biases of negative zero and the smallest subnormal, and weights of a NaN with payload 1 and of
1.0, come back with the bits 0x80000000, 0x00000001, 0x7FC00001 and 0x3F800000.

### RefusesOneFloatShortWithTheLayerAndBothByteCounts

A file one float short fails with `InvalidFile`, "104 bytes" and
"convolution layer 2 needs bytes up to 108".

### RefusesOneFloatOverWithTheLayerAndBothByteCounts

A file one float over fails with `InvalidFile`, "112 bytes", "108 bytes" and
"4 bytes are left over after convolution layer 2".

### RefusesBytesAfterAHeaderWithNoConvolution

A file with one float after its header and an empty shape list fails with `InvalidFile`,
"no convolution is listed" and "4 bytes are left over".

### RefusesAShapeThatIsNotPositive

A shape with zero filters fails with `InvalidDescription`, "convolution layer 0" and
"filters 0".

### RefusesAShapeWhoseCountOverflows

A shape of 2^40 filters over 2^40 channels fails with `InvalidDescription`,
"convolution layer 0" and "too large to read".

### ReportsAMissingFileAsFileNotFound

A path that does not exist fails with `FileNotFound` and "could not open".

## test_darknet_import/test_decode.cpp

Suites: `Integration_DarknetDecode`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov3_tiny_layers.cpp

Suites: `Integration_DarknetYolov3TinyLayers`, `Integration_DarknetLayers`, in
`sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov3_forward.cpp

Suites: `Integration_DarknetYolov3Forward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov3_tiny_forward.cpp

Suites: `Integration_DarknetYolov3TinyForward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov4_forward.cpp

Suites: `Integration_DarknetYolov4Forward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov4_tiny_forward.cpp

Suites: `Integration_DarknetYolov4TinyForward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov7_forward.cpp

Suites: `Integration_DarknetYolov7Forward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_import/test_yolov7_tiny_forward.cpp

Suites: `Integration_DarknetYolov7TinyForward`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov3.cpp

Suites: `Integration_DarknetYolov3Training`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov3_tiny.cpp

Suites: `Integration_DarknetYolov3TinyTraining`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov4.cpp

Suites: `Integration_DarknetYolov4Training`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov4_tiny.cpp

Suites: `Integration_DarknetYolov4TinyTraining`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov7.cpp

Suites: `Integration_DarknetYolov7Training`, in `sushiai_vision_importers_tests`.

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

## test_darknet_training/test_yolov7_tiny.cpp

Suites: `Integration_DarknetYolov7TinyTraining`, in `sushiai_vision_importers_tests`.

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

## test_darknet_store.cpp

Suites: `Integration_DarknetStore`, in `sushiai_vision_importers_tests`.

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
