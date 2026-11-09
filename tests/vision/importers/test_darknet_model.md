# tests/vision/importers/test_darknet_model.cpp


Suites: `Small_DarknetModel`, `Small_DarknetModelHeads`, `Small_DarknetModelAnchors`,
`Small_DarknetModelSuppression`, `Small_DarknetModelUnknownKeys`,
`Small_DarknetModelUntranslated`, `Small_DarknetModelMisfits` and `Small_DarknetModelState`.

Tests the translation of a config text into a `DarknetModel`: the rows of its model spec, the
row each layer ends on, its heads, its anchor layout, the shape of each convolution and the
state names its weights go to. Most cases translate a 24-layer text with two heads that the
file holds, under `DarknetNormalisation::Separate` or `DarknetNormalisation::Folded`. The
refusal cases use a small 32x32 network, one table row per input, and name the row in the test
name. `Small_DarknetModelState` runs on a `HostExecutionBackend` the fixture owns, so no case
needs a device, a file, a thread or a clock.

## SeparateFormHasTheCountedRows

The Separate form has 43 rows and the `row_of_layer` list the case states.

## SeparateFormReadsTheInputOfTheNetSection

The input shape is [416, 416, 3], the input fit is "resize" and the network is 416 high and 416
wide.

## ANormalisedConvolutionIsThreeRowsReadingTheInput

Rows 0 to 2 are `conv2d`, `batchnorm2d` and `leaky_relu`, and row 0 reads the model input.

## SeparateFormWritesTheSlopeAndTheEpsilon

The `leaky_relu` row has a slope of 0.1 and the `batchnorm2d` row a `bn_epsilon` of 1e-5.

## OnlyAPlainConvolutionHasABiasInTheSeparateForm

Row 0, behind a batch normalisation, has no convolution bias; row 33, a head convolution, has
one.

## ARouteOfOneLayerAddsNoRow

Row 34 is a `conv2d` that reads row 29, the row of layer 13.

## ARouteOfTwoLayersIsAConcatInFileOrder

Row 38 is a `concat` of rows 37 and 18.

## AStrideOnePoolPadsTheTrailingEdge

Row 23 is a `maxpool2d` of kernel 2, stride 1 and padding 0 with a trailing padding of 1 and 1.

## AStrideTwoPoolOfAnEvenExtentNeedsNoTrailingPad

Row 3 has a trailing padding of 0 and 0.

## FoldedFormHasTheCountedRows

The Folded form has 32 rows and its own `row_of_layer` list.

## FoldedFormHasNoNormalisationRow

No row of the Folded form has the type `batchnorm2d`.

## FoldedFormConvolutionsHaveABias

Row 0 has a bias.

## FoldedFormKeepsTheSourcesOfTheRoutes

Row 25 reads row 21 and row 28 reads rows 27 and 13.

## FoldedFormTargetsNameAWeightAndABiasOnly

There are 13 targets, and the first names "0.weight" and "0.bias" and leaves scale, mean and
variance empty.

## TwoHeadsPublishTwoOutputs

The model has two heads and two outputs.

## Small_DarknetModelHeads

One row per head: "First" is `head.0` on layer 16 and row 33, "Second" is `head.1` on layer 23
and row 42.

- `NamesItsOutputAndLayer`: the head carries that output name and layer.
- `OutputPublishesTheRowOfItsLayer`: the output reads that row from the port "output" under that
  name.
- `ReadsTheOptionsOfTheFile`: 80 classes, the Logit coding and a `scale_xy` of 1.

## SuppressionDefaultsToIoUWithTheDefaultExponent

The criterion is `SuppressionCriterion::IoU` with the exponent 0.6.

## Small_DarknetModelAnchors

One row per anchor, six in all: `IsItsPixelSizeOverTheNetworkSize` holds the pixel sizes 10x14,
23x27, 37x58, 81x82, 135x169 and 344x319 divided by 416, within four units in the last place.

## MasksListTheAnchorsOfEachHead

The masks are {3, 4, 5} and {0, 1, 2}.

## ShapesNameTheLayerOfEachConvolution

The Folded form lists 13 convolution shapes on the layers 0, 2, 4, 6, 8, 10, 12, 13, 14, 15,
18, 21 and 22.

## EveryShapeIsNormalisedExceptTheHeadConvolutions

A shape is normalised unless it has 255 filters.

## ShapesCountTheFloatsOfTheWeightsFile

Summed over the shapes, the biases, normalisation vectors and weights come to 8858734 floats.

## TheConvolutionAfterTheJoinReadsBothChannelCounts

The twelfth convolution reads 384 channels, the 128 upsampled ones and the 256 of layer 8.

## Small_DarknetModelState

`SeparateFormBuildsAndItsTargetsAreStateEntries`: the first target names "0.weight", "1.bias",
"1.weight", "1.running_mean" and "1.running_var", and there are 11 * 5 + 2 * 2 target names.
`FoldedFormBuildsAndItsTargetsAreStateEntries`: there are 13 * 2 target names. In both the model
builds, publishes the two heads, and every target names an entry of its state.

## AnActivatedHeadHasTheActivatedCoding

A head with `new_coords=1`, `scale_x_y = 2.0`, `nms_kind=diounms` and `beta_nms=0.6` among its
keys has the Activated coding. The next five cases read the same text.

## AnActivatedHeadReadsItsStretch

The `scale_xy` is 2.

## AnActivatedHeadReadsItsClassCount

The head has one class.

## DiounmsSelectsTheDistanceCriterion

The criterion is `SuppressionCriterion::DistanceIoU`.

## DiounmsReadsItsExponentFromBetaNMS

The exponent is 0.6.

## AnActivatedHeadPublishesASigmoidRow

The model has two rows, the second a `sigmoid`, and the output is published from row 1.

## Small_DarknetModelSuppression

`OnlyDiounmsReadsBetaNMS`: with `nms_kind=greedynms` and `beta_nms=0.3` (row
`GreedynmsKeepsTheDefault`) the criterion is `DistanceIoU` and the exponent stays 0.6; with
`nms_kind=diounms` and `beta_nms=0.3` (row `DiounmsTakesBetaNMS`) the exponent is 0.3.

## TheLastHeadDecidesTheCriterion

A text whose first head sets `nms_kind=diounms` and whose second head sets nothing has two
heads, the criterion `IoU` and the exponent 0.6.

## AnUnknownSectionIsRefusedNamingItsLine

A `[reorg]` section fails with `Unsupported`, "line 11", "layer 1" and "[reorg]".

## Small_DarknetModelUnknownKeys

`IsRefusedNamingItsLine` fails with `Unsupported` for `groups` in a convolution
(`AGroupedKeyOfAConvolution`: "line 7", "[convolutional]", "'groups'"), `label_smooth_eps` in a
head (`AKeyOfAHead`: "line 16", "[yolo]", "'label_smooth_eps'") and `crop` in the first section
(`AKeyOfTheFirstSection`: "line 5", "'crop'").

## LetterBoxOneSelectsTheLetterboxFit

The small network with `letter_box=1` in its `[net]` section translates to an input fit of
"letterbox".

## LetterBoxTwoIsRefusedNamingItsLine

With `letter_box=2` the translation fails with `InvalidDescription`, "line 5" and
"letter_box=2".

## Small_DarknetModelUntranslated

`IsRefusedNamingItsLine` fails with `Unsupported` for an upsample `scale=2`
(`AnUpsampleScale`: "line 11", "scale 2") and `activation=relu` (`AnActivationOfRelu`: "line 6",
"'relu'").

## Small_DarknetModelMisfits

`IsRefusedNamingItsLine` fails with `InvalidDescription` for `filters=many` (`AWordForANumber`:
"line 6", "'filters'", "'many'"), a route to a later layer (`ARouteToALaterLayer`: "line 12",
"names layer 1"), a head over 12 channels that needs 18 (`AHeadOverTooFewChannels`: "line 8",
"reads 18 channels", "has 12"), a mask that names anchor 3 of three (`AMaskPastTheAnchors`:
"line 12", "names anchor 3"), and a text with no head (`ATextWithoutAHead`: "no [yolo]
section").

Suites: `Small_DarknetRouteGroups` and `Small_DarknetRouteSplits`.

Tests how the translation of a config text turns a `[route]` section with `groups` and
`group_id` into model rows. Each case starts from a network whose first convolution has 64
filters and ends with a head, and reads the rows, their sources and the channels of the
convolutions of the `DarknetModel`.

## OneSourceGroupIsOneSliceRow

A route of the layer before it with `groups=2` and `group_id=1` gives three rows: a `conv2d`, a
`slice_channels` that reads row 0 with the offset 32 and the count 32, and a row that reads the
slice. The second convolution reads 32 channels.

## EverySourceIsSlicedBeforeTheJoin

A grouped route of two layers gives six rows. Rows 2 and 3 are `slice_channels` rows over rows
1 and 0, each with the offset 32 and the count 32. Row 4 is a `concat` of rows 2 and 3, and row
5 reads it. The third convolution reads 64 channels.

## UngroupedSourceAddsNoRow

A route of one layer without groups gives two rows, the second a `conv2d` that reads row 0, and
the second convolution reads 64 channels.

## Small_DarknetRouteSplits

`IsRefusedNamingItsLine` fails with `InvalidDescription` for `groups=3` over 64 channels
(`GroupsThatDoNotDivideTheChannels`: "line 12", "groups 3", "64 channels"), `group_id=2` with
two groups (`AGroupIdPastTheGroups`: "line 11", "group_id 2"), `groups=0` (`ZeroGroups`: "line
11", "groups 0"), and a `group_id` of 1 without `groups` (`AGroupIdWithoutGroups`: "line 11",
"group_id 1").

Suites: `Small_DarknetShortcut` and `Small_DarknetShortcutFrom`.

Tests how the translation of a config text turns a `[shortcut]` section into a model row. Each
case starts from an 8x8 three-channel network with two convolutions.

## Small_DarknetShortcutFrom

`AddsTheLayerBeforeAndTheNamedLayer` gives four rows for `from=-2` (`ANegativeFrom`) and for
`from=0` with `activation=linear` (`AnAbsoluteFrom`). Row 2 has the type "add", no attribute
key, and the sources 1 and 0.

## FromAboveTheShortcutIsRefusedNamingItsLine

`from=2` fails with `InvalidDescription`, "line 18" and "only layers 0 to 1".

## DifferentChannelCountsAreRefusedNamingItsLine

A shortcut over convolutions of 4 and 5 filters fails with `InvalidDescription`, "line 18" and
"share their shape".

## NonLinearActivationIsRefusedNamingItsLine

`activation=leaky` fails with `Unsupported`, "line 19" and "'leaky'".

## WeightsTypeIsRefusedNamingItsLine

`weights_type=per_feature` fails with `Unsupported`, "line 19" and "weights_type".

## MissingFromIsRefusedNamingTheSection

A shortcut without `from` fails with `InvalidDescription` and "no 'from'".

## MoreThanOneFromIsRefused

`from=-1,-2` fails with `InvalidDescription` and "names one layer".
