# tests/vision/importers/test_darknet_model.cpp


Suites: `Small_DarknetModel`, `Medium_DarknetModelState`.

Tests the translation of a config text into a `DarknetModel`: the rows of its model spec, the
row each layer ends on, its heads, its anchor layout, the shape of each convolution and the
state names its weights go to. Most cases translate a 24-layer text with two heads that the
file holds, under `DarknetNormalisation::Separate` or `DarknetNormalisation::Folded`. The
refusal cases use a small 32x32 network. `Medium_DarknetModelState` is an engine fixture.

## SeparateFormHasTheCountedRowsAndSources

The Separate form has 43 rows, the input shape [416, 416, 3] and the `row_of_layer` list the
case states. Rows 0 to 2 are `conv2d`, `batchnorm2d` and `leaky_relu`, with a slope of 0.1, a
`bn_epsilon` of 1e-5 and no convolution bias; row 33 has a bias. Row 34 is a `conv2d` that reads
row 29, and row 38 is a `concat` of rows 37 and 18. Row 23 is a `maxpool2d` of kernel 2, stride 1
and padding 0 with a trailing padding of 1 and 1, and row 3 has a trailing padding of 0 and 0.

## FoldedFormHasNoNormalisationRow

The Folded form has 32 rows, none of the type `batchnorm2d`, and its own `row_of_layer` list.
Row 0 has a bias, row 25 reads row 21 and row 28 reads rows 27 and 13. There are 13 targets,
and the first names "0.weight" and "0.bias" and leaves scale, mean and variance empty.

## HeadsCarryTheirOutputsOptionsAndAnchors

The model has two heads and two outputs, "head.0" on layer 16 and row 33 and "head.1" on layer
23 and row 42, each with 80 classes, the Logit coding and a `scale_xy` of 1. The criterion is
`SuppressionCriterion::IoU` with the exponent 0.6. The six anchors are their pixel sizes
divided by 416, and the masks are {3, 4, 5} and {0, 1, 2}.

## ShapesCountTheFloatsOfTheWeightsFile

The Folded form lists 13 convolution shapes on the layers 0, 2, 4, 6, 8, 10, 12, 13, 14, 15,
18, 21 and 22. A shape is normalised unless it has 255 filters. Summed over the shapes, the
biases, normalisation vectors and weights come to 8858734 floats, and the twelfth convolution
reads 384 channels.

## SeparateFormBuildsAndItsTargetsAreStateEntries

In the Separate form the first target names "0.weight", "1.bias", "1.weight", "1.running_mean"
and "1.running_var", and there are 11 * 5 + 2 * 2 target names. The model builds, and every
target names an entry of its state.

## FoldedFormBuildsAndItsTargetsAreStateEntries

In the Folded form there are 13 * 2 target names. The model builds, and every target names an
entry of its state.

## AnActivatedHeadReadsItsStretchAndItsSuppression

A head with `new_coords=1`, `scale_x_y = 2.0`, `nms_kind=diounms` and `beta_nms=0.6` among its
keys gives one head with the Activated coding, a `scale_xy` of 2 and one class. The criterion is
`SuppressionCriterion::DistanceIoU` with the exponent 0.6. The model has two rows, the second a
`sigmoid`, and the output is published from row 1.

## OnlyDiounmsReadsBetaNMS

With `nms_kind=greedynms` and `beta_nms=0.3` the criterion is `DistanceIoU` and the exponent
stays 0.6. With `nms_kind=diounms` and `beta_nms=0.3` the exponent is 0.3.

## TheLastHeadDecidesTheCriterion

A text whose first head sets `nms_kind=diounms` and whose second head sets nothing has two
heads, the criterion `IoU` and the exponent 0.6.

## AnUnknownSectionIsRefusedNamingItsLine

A `[reorg]` section fails with `Unsupported`, "line 11", "layer 1" and "[reorg]".

## AnUnknownKeyIsRefusedNamingItsLine

Three unknown keys fail with `Unsupported`: `groups` in a convolution ("line 7",
"[convolutional]", "'groups'"), `label_smooth_eps` in a head ("line 16", "[yolo]",
"'label_smooth_eps'") and `crop` in the first section ("line 5", "'crop'").

## UntranslatedValuesAreRefusedNamingTheirLine

Three values fail with `Unsupported`: `letter_box=1` ("line 5"), an upsample `scale=2`
("line 11", "scale 2") and `activation=relu` ("line 6", "'relu'").

## ValuesThatDoNotFitAreRefusedNamingTheirLine

Five texts fail with `InvalidDescription`: `filters=many` ("line 6", "'filters'", "'many'"), a
route to a later layer ("line 12", "names layer 1"), a head over 12 channels that needs 18
("line 8", "reads 18 channels", "has 12"), a mask that names anchor 3 of three ("line 12",
"names anchor 3"), and a text with no head ("no [yolo] section").

Suites: `Small_DarknetRouteGroups`.

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

## UnequalSplitsAndGroupsOutOfRangeAreRefused

Four routes fail with `InvalidDescription`: `groups=3` over 64 channels ("line 12", "groups 3",
"64 channels"), `group_id=2` with two groups ("line 11", "group_id 2"), `groups=0` ("line 11",
"groups 0"), and a `group_id` of 1 without `groups` ("line 11", "group_id 1").

Suites: `Small_DarknetShortcut`.

Tests how the translation of a config text turns a `[shortcut]` section into a model row. Each
case starts from an 8x8 three-channel network with two convolutions.

## NegativeFromAddsTheLayerBeforeAndTheNamedLayer

`from=-2` gives four rows. Row 2 has the type "add", no attribute key, and the sources 1 and 0.

## AbsoluteFromAddsTheSameTwoRows

`from=0` with `activation=linear` gives the same "add" row with the sources 1 and 0.

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
