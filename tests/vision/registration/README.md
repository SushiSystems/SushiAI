# tests/vision/registration

The tests of the module `vision/registration`, one folder for its one header, `modules/vision/registration/include/SushiAI/vision/registration/registration.hpp`. The suite prefix of a case gives its ctest label. The three files in `test_registration/` build the rows that header registers from the model files of `applications/examples/`; `test_registration/registration_test_support.hpp` holds the helpers they share: `composed`, `trace_outputs`, `count_row_parameters` and `count_parameters`.

## test_registration/test_registration.cpp

Suites: `Small_VisionRegistration`, `Small_VisionRegistrationRow`, `Small_VisionRegistrationPixelRow`,
`Small_VisionRegistrationData`, `Small_VisionMnist`, `Small_VisionMnistOption`, `Small_VisionMnistPaths`,
`Medium_VisionRegistration`, `Medium_VisionRegistrationNeck`, `Medium_VisionRegistrationNeckLevel`,
`Medium_VisionMnist`, `Medium_VisionMnistSplit`, `Medium_VisionMnistImage`, `Medium_Yolo26Forward`,
`Medium_Yolo26Whole`, `Medium_Yolo26FeatureRows`, `Medium_Yolo26InnerRows`, `Medium_Yolo26Ports`,
`Medium_Yolo26Decode`.

Tests what the vision registration adds: its rows, the MNIST loader and the YOLO26n model file
against the reference. The registry cases take no engine and read no file, so they are small. The
dataset refusals run on a `HostExecutionBackend`, so they are small too. The YOLO26n file cases read
`applications/examples/yolo26/model.json` and `tests/vision/registration/fixtures/yolo26/MANIFEST.json`;
the file is the subject, so they are medium. The `Medium_Yolo26*` fixtures open `forward.sacp` and
`weights.sushi` of the fixture directory in `SetUp` and skip the case, naming the file, when one is
missing. `Medium_Yolo26Whole` runs the forward pass of the whole model once for the program, in `SetUpTestSuite`, and every row reads it; a pass that cannot run skips the rows with its reason.

### RegisterNodesAddsTheTenBlocks

`Vision::register_nodes` adds the node rows "conv", "bottleneck", "c3k", "c3k2", "c2f", "c3",
"sppf", "c2psa", "detect" and "segment_head", in that order, and no layer row.

### RegisterDatasetsAddsMNISTThenReader

`Vision::register_datasets` adds "mnist" and "reader", in that order. The "mnist" row has a factory.

### TheReaderRowHoldsMakeReaderDataset

The row of `Config::K_READER_DATA_TYPE` has `Vision::Datasets::make_reader_dataset`.

### ARegistryWithoutTheBlocksRefusesConv

`NN::build_model` over the rows of `NN::register_layers` alone fails for a model with a "conv"
row. The code is `InvalidDescription`, and the message holds "layer 0 'conv'", "the type 'conv'"
and "does not have".

### SegmentHeadBuildsOnePortAtTheUpsampledSize

The "segment_head" row builds a node with one port of the shape its case expects.

### DetectBuildsFourPortsPerLevel

The "detect" row over its case's levels builds a node with four ports per level.

### BuildsFromAMinimalObject

Rows over the eight single-port rows of `register_nodes`: each builds, from its case's attributes,
a node with one port named `NN::K_SINGLE_PORT_NAME` and of the shape the case expects.

### RejectsAnUnknownKey

Rows over the eight single-port rows and the detect row: each given the key "no_such_key" fails with
`InvalidDescription`, and the message holds "layer 0 '<type>'" and "'no_such_key'".

### TheYOLO26FileListsTheTypesOfItsTwentyFourRows

The model file parses to 24 rows of the types the case lists in order, from "conv" to "detect".

### TheYOLO26FileTakesA640By640By3Input

The model file states the input shape [640, 640, 3].

### TheYOLO26FileReadsTheSourcesOfItsInputJoinsAndHead

Row 0 reads the model input. The four `concat` rows read rows 11 and 6, 14 and 4, 17 and 13, and 20
and 10, and the detect row reads rows 16, 19 and 22.

### TheYOLO26FileMergesItsDefaultsIntoTheRows

Row 0 has a `bn_epsilon` of 0.001, row 9 a `bn_momentum` of 0.03, and row 11 has no `bn_epsilon`.
The reals are compared for equality, since the file and the literal are the same decimal text.

### TheYOLO26FilePublishesTheDetectRowAsItsOneOutput

The one output is row 23 with an empty port.

### TheYOLO26FileBuildsWhole

The model file builds over the composed registry. Its trace publishes the detect row's twelve
outputs in port order, and its state is the 594 records the manifest lists for rows 0 to 23,
without the "model." prefix.

### YOLO26RowsTwoAndFourAddTheirBottlenecksInput

Rows 2 and 4 of the file have "shortcut" true, and the first five rows trace to exactly two ADD
nodes.

### YOLO26RowsZeroToNineMatchTheManifest

Rows 0 to 9 build and yield the 165 records the manifest lists under model.0 to model.9, and
the traced output is "output [2, 20, 20, 256]".

### YOLO26RowsBuildToTheReferenceShapes

With a `silu` row in place of row 10, the first 22 rows build, and the vector of each row's one-port
shape equals `K_YOLO_ROW_SHAPES`.

### TracesToTheManifestExtents

Rows over feature levels 0 and 1. With a `silu` row in place of row 10, row 16 and row 19 trace to
the height, width and channel count the manifest records for the level.

### StrideTimesHeightIsTheInputSide

Rows over feature levels 0 and 1: the level's stride times its height is the input height, equal as
doubles, since both are whole numbers.

### TheManifestRecordsThreeLevels

The manifest's forward.sacp context lists three levels.

### Row21JoinsToThe384Channels

Row 21 traces to [2, 20, 20, 384].

### StateIsTheManifestRecordsOfRows0To9And11To21

The neck's state is the manifest's records for rows 0 to 9 followed by its 145 records for rows 11
to 21.

### RegisterObjectivesAddsFourRowsAfterTheLayers

After `Train::register_objectives`, `Vision::register_objectives` adds four rows: "anchor",
"anchor_free", "pixel_cross_entropy" and "dice". Row 0 stays "cross_entropy".

### RegisterObjectivesTwiceRefusesAnchorAsRegistered

A second `Vision::register_objectives` fails with `InvalidDescription`, "'anchor'" and "already
registered".

### TheAnchorRowReadsTheTaskOfTheModel

`Train::make_objective` of "anchor" over an empty section fails with `InvalidDescription`, and
the message holds "the objective block" and "'task'".

### BuildsOverOneLogitMapAndIsNamedByItsType

Rows over "pixel_cross_entropy" and "dice": each builds an objective over one [2, 4, 4, 3] output
with an `ignore_index` of 255, and the objective's name is the row's type.

### RegisterImportersAddsTheDarknetRow

`Vision::register_importers` adds one row of the format "darknet".

### RegisterImportersTwiceRefusesDarknetAsRegistered

A second `Vision::register_importers` fails with `InvalidDescription`, "'darknet'" and "already
registered".

### TheDarknetRowRefusesACallWithoutFiles

`IO::import_model` of "darknet" with no file fails with `InvalidDescription`, and the message
holds "'darknet'", ".cfg" and "0 files".

### TheReaderRowRefusesAnUnknownReaderNamingTheKnownOnes

`Data::make_dataset` of the reader type with the reader "no_such_reader", the task "detection" and
an absent root, on a host engine, fails with `InvalidDescription`. The message holds "the data block
'reader'", "'no_such_reader'", "'coco_json'" and "'yolo_txt'".

### TheDatasetRegistryListsMNIST

`find("mnist")` returns a row.

### AnUnknownDatasetTypeIsRefusedNamingMNIST

`make_dataset` of the type "coco" fails with `InvalidDescription` and a message that holds
"'mnist'".

### AnUnknownMNISTKeyIsRejectedByItsOwnFactory

The key "batch_sizes" is refused with `InvalidDescription`, and the message holds "data",
"'batch_sizes'" and "'batch_size'". The directory is not read.

### MNISTRequiresADirectory

A section without "directory" is refused with `InvalidDescription`, and the message holds "data",
"has no" and "'directory'".

### AnUnknownValueIsRefusedNamingTheAllowedOnes

Rows over the option "split" with the value "validation" and the option "layout" with the value
"tiled": each is refused with `InvalidDescription` before any file is read. The message holds "data",
the key, the value and the allowed values: "'train'" and "'test'" for the split, "'flat'" and
"'image'" for the layout.

### RefusesADirectoryWithoutTheFiles

Rows over the train split and the test split: "mnist" over an absent directory fails with
`FileNotFound`. The message holds "data", the directory, the split's name ("MNIST train split",
"MNIST test split") and its image file. The train row also holds the other three IDX file names.
Medium, because the refusal looks at the file system.

### YieldsRankFourInputsAndRankOneLabels

With `MNISTLayout::Image`, a batch size of 16 and 32 samples, batch 0 holds a rank-1 `Int32` label
tensor of 16 entries and a [16, 28, 28, 1] input tensor. The fixture skips the case when the MNIST
IDX files are absent.

### StatesItsSampleShapeAndLabelTarget

With `MNISTLayout::Image`, the dataset's sample shape is [28, 28, 1]. It has one target, of the role
`Data::K_ROLE_LABELS`, a scalar sample shape, the element type `Int32` and 10 classes.

### AnAbsentMNISTDirectoryIsUnavailableAndRefusedByName

For a directory that does not exist, `mnist_available` is false and `try_load_mnist` fails with
`FileNotFound` and names the directory.

### TheExpectedPathsAreListed

`mnist_expected_paths` is not empty.

### MatchesTheReferenceLevel

Rows over rows 16, 19 and 22, the three feature maps the head reads: each equals the record
`features.level0`, `features.level1` or `features.level2`, to 2048 float unit roundoffs.

### MatchesTheReferenceRow

Rows over rows 9, 10 and 21 of the whole-model pass: each equals the record `features.row9`,
`features.row10` or `features.row21`.

### C2PSAAloneMapsTheReferenceRow9ToRow10

Row 10 is built alone, with the records of `weights.sushi` that rename to its entries and the
reference's row 9 as its input. Its output equals the reference's row 10.

### MatchesTheReferenceRawOutput

Rows over the twelve ports: the `boxes` and `scores` ports of the `one2many` and `one2one` branches at
each of the three levels equal the records of the same names.

### MatchesTheReferenceDecodes

Rows over the branches `one2many` and `one2one`. The raw outputs of the branch are joined over the
levels of 80, 40 and 20 cells a side and decoded on the host, `one2many` as centre and size and
`one2one` as corners. The boxes, to 4096 float unit roundoffs, and the sigmoid of the scores, to 32,
equal the `decoded.boxes` and `decoded.scores` records of the branch.

### TheTopScoresMatchTheReferenceDetections

`select_top_detections` over the `one2one` decode gives rows whose score column equals that of the
`one2one.detections` record to 32 float unit roundoffs.

## test_registration/test_ultralytics_files.cpp

Suites: `Medium_UltralyticsFiles`, `Medium_UltralyticsTaskSection`.

Tests the model files `applications/examples/yolov8n/model.json` and
`applications/examples/yolo11n/model.json`, and the task section of those two and of
`applications/examples/yolo26/model.json`. The cases read the files from `SA_REPOSITORY_ROOT`. Each
case of `Medium_UltralyticsFiles` runs once per file, and each of `Medium_UltralyticsTaskSection`
once per folder.

### HoldsItsRows

`yolov8n` has 23 rows and `yolo11n` has 24.

### TakesA640By640By3Input

The file takes a [640, 640, 3] input.

### PublishesOneOutput

The file names one output node.

### BuildToTheParameterCountOfTheReference

The built model holds the parameters of the summary line Ultralytics prints when it loads the
model, 3,157,200 for `yolov8n` and 2,624,080 for `yolo11n`, less 16. The 16 are the fixed ramp of
the distribution expectation, which Ultralytics counts as a parameter and this framework keeps as no
state record.

### PublishOnlyTheOneToManyPorts

Each model publishes six ports: the boxes and the scores of three levels. No `one2one` port exists.

### PublishTheBoxesOfLevelZeroAtStrideEight

The boxes of level 0 are [1, 80, 80, 64].

### PublishTheBoxesOfLevelOneAtStrideSixteen

The boxes of level 1 are [1, 40, 40, 64].

### PublishTheBoxesOfLevelTwoAtStrideThirtyTwo

The boxes of level 2 are [1, 20, 20, 64].

### PublishTheScoresOfLevelZeroAtStrideEight

The scores of level 0 are [1, 80, 80, 80].

### PublishTheScoresOfLevelOneAtStrideSixteen

The scores of level 1 are [1, 40, 40, 80].

### PublishTheScoresOfLevelTwoAtStrideThirtyTwo

The scores of level 2 are [1, 20, 20, 80].

### GiveARegMaxOfSixteen

`derive_free_model_facts` over the traced outputs gives a `reg_max` of 16.

### GiveEightyClasses

`derive_free_model_facts` over the traced outputs gives 80 classes.

### GiveNoOneToOneCopy

`derive_free_model_facts` over the traced outputs gives no one-to-one copy.

### GiveThreeLevelsAtTheLevelStrides

`derive_free_model_facts` over the traced outputs gives three levels at strides 8, 16 and 32.

### GiveSquareLevelsOfTheInputOverTheStride

The three levels are square, with the side 640 divided by the stride.

### ReadsBackTheDefaultScoreThreshold

Rows over `yolov8n`, `yolo11n` and `yolo26`: `read_free_task` reads the task section with a score
threshold of 0.25. The value is written in the test; `FreeTask{}` is the code under test.

### ReadsBackTheDefaultSuppressionThreshold

Rows over the three folders: the suppression threshold is 0.7.

### ReadsBackIntersectionOverUnionSuppression

Rows over the three folders: the suppression criterion is intersection over union.

## test_registration/test_yolov5n.cpp

Suites: `Medium_Yolov5nFile`, `Medium_Yolov5nLevel`, `Medium_Yolov5nAnchor`, `Medium_Yolov5nHead`,
`Small_Yolov5nBiasHead`, `Small_Yolov5nBiasInit`.

Tests the model file `applications/examples/yolov5n/model.json`, read from `SA_REPOSITORY_ROOT`. The
bias cases are small: they trace the graph and zero a bias on a `HostExecutionBackend` state, so
they use no device.

### HoldsItsRows

The file has 29 rows.

### TakesA640By640By3Input

The file takes a [640, 640, 3] input.

### PublishesThreeOutputsNamedHead0ToHead2

The file names three outputs; the first is "head_0" and the last "head_2".

### BuildsToTheParameterCountOfTheReference

The built model holds 1,872,157 parameters, the figure in the summary line Ultralytics prints when
it loads `yolov5n.yaml`.

### BuildsToTheSumOfTheParameterCountsOfItsRows

The parameter count of the built model equals the sum of the counts of its 29 rows.

### PublishesTheHeadOfItsName

Rows over the three levels: the output of the level is named "head_<level>".

### PublishesTheHeadAt255ChannelsAtTheLevelStride

Rows over the three levels: the head is 255 channels wide at the side 640 divided by the stride 8,
16 or 32.

### TheTaskSectionReadsBackEightyClasses

The task section reads as 80 classes.

### TheTaskSectionReadsBackNineAnchors

The task section reads as nine anchors.

### ReadsBackAsTheYOLOv5AnchorInPixels

Rows over the nine anchors: the width and the height times 640 equal the YOLOv5 anchor in pixels,
to 256 float unit roundoffs of the fraction.

### TheTaskSectionReadsBackThreeHeads

The task section reads as three heads.

### ReadsBackItsThreeAnchors

Rows over the three heads: head `h` reads the anchors 3h, 3h + 1 and 3h + 2.

### ReadsBackItsOutput

Rows over the three heads: the head names the output "head_<h>".

### ReadsBackAsLogitCoded

Rows over the three heads: the coding is logit.

### ReadsBackWithSquaredSigmoidSides

Rows over the three heads: the sides are squared sigmoids.

### ReadsBackWithAStretchOfTwo

Rows over the three heads: the centre stretch is 2.

### TheTaskSectionReadsBackTheShapeRatioRule

The assignment rule is the shape ratio.

### TheTaskSectionReadsBackTheDefaultRatioLimit

The ratio limit is `K_DEFAULT_RATIO_LIMIT`.

### NamesTheBiasOfItsRow

Rows over the three levels: `derive_anchor_bias_heads` names the bias of the head "26.bias",
"27.bias" or "28.bias".

### CarriesTheStrideOfItsLevel

Rows over the three levels: the head carries the stride 8, 16 or 32.

### StartsTheObjectnessEntryOfEachSlot

Rows over the three levels: over a zeroed bias, `initialize_anchor_biases` sets the objectness entry
of each slot to log(8 / (640 / stride)^2), to 256 float unit roundoffs.

### StartsEachClassEntry

Rows over the three levels: each class entry of each slot becomes log(0.6 / (80 - 0.99)), to 256 float
unit roundoffs.

### LeavesTheBoxEntriesAtZero

Rows over the three levels: the four box entries of each slot stay zero.
