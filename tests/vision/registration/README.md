# tests/vision/registration

The tests of the module `vision/registration`, one file per header of `modules/vision/registration/include/`. The suite prefix of a case gives its ctest label. `registration_test_support.hpp` holds the helpers the three files share: `composed`, `trace_outputs` and `count_parameters`.

## test_registration.cpp

Suites: `Medium_VisionRegistration`, `Medium_VisionRegistrationData`.

Tests the four registration entry points of the vision module and what builds from their rows. The
YOLO26n cases read `applications/examples/yolo26/model.json` and
`tests/vision/registration/fixtures/yolo26/MANIFEST.json` and report themselves skipped when the
repository root is not found. `Medium_VisionRegistrationData` is an engine fixture.

### RegisterNodesAddsTheTenBlocks

`Vision::register_nodes` adds the node rows "conv", "bottleneck", "c3k", "c3k2", "c2f", "c3",
"sppf", "c2psa", "detect" and "segment_head", in that order, and no layer row.

### RegisterDatasetsAddsMNISTAndReader

`Vision::register_datasets` adds "mnist" and "reader", in that order. The "mnist" row has a
factory, and the row of `Config::K_READER_DATA_TYPE` has `Vision::Datasets::make_reader_dataset`.

### ARegistryWithoutTheBlocksRefusesConv

`NN::build_model` over the rows of `NN::register_layers` alone fails for a model with a "conv"
row. The code is `InvalidDescription`, and the message holds "layer 0 'conv'", "the type 'conv'"
and "does not have".

### EveryRowHasAMinimalCase

The file's minimal cases, followed by its detect case and its segment head case, have the
types of the registered node rows in the same order and the same count.

### SegmentHeadBuildsOnePortAtTheUpsampledSize

The "segment_head" row builds a node with one port of the shape its case expects.

### DetectBuildsFourPortsPerLevel

The "detect" row over its case's levels builds a node with four ports per level.

### EachRowBuildsFromAMinimalObject

Each of the other rows builds, from its case's attributes, a node with one port named
`NN::K_SINGLE_PORT_NAME` and of the shape the case expects.

### EachRowRejectsAnUnknownKey

Each row given the key "no_such_key" fails with `InvalidDescription`, and the message holds
"layer 0 '<type>'" and "'no_such_key'". The same holds for "detect".

### TheYOLO26FileHoldsItsTwentyFourRows

The model file parses to 24 rows of the types the case lists in order, from "conv" to
"detect", with the input shape [640, 640, 3]. The four `concat` rows read rows 11 and 6, 14 and
4, 17 and 13, and 20 and 10, and the detect row reads rows 16, 19 and 22. Row 0 has a
`bn_epsilon` of 0.001, row 9 a `bn_momentum` of 0.03, and row 11 has no `bn_epsilon`. The one
output is row 23 with an empty port.

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

With a `silu` row in place of row 10, the first 22 rows build, and each row's one port has the
per-sample shape `K_YOLO_ROW_SHAPES` holds for it.

### YOLO26NeckTracesToTheReferenceLevels

With a `silu` row in place of row 10, rows 16 and 19 trace to the height, width and channel
count the manifest records for feature levels 0 and 1, and each level's stride times its height
is the input height. Row 21 traces to [2, 20, 20, 384]. The state is the manifest's records for
rows 0 to 9 followed by its 145 records for rows 11 to 21.

### RegistersTheAnchorAndThePixelObjectives

After `Train::register_objectives`, `Vision::register_objectives` adds three rows: "anchor",
"pixel_cross_entropy" and "dice". Row 0 stays "cross_entropy". A second call fails with
`InvalidDescription`, "'anchor'" and "already registered".

### TheAnchorRowReadsTheTaskOfTheModel

`Train::make_objective` of "anchor" over an empty section fails with `InvalidDescription`, and
the message holds "the objective block" and "'task'".

### ThePixelRowsBuildOverOneLogitMap

"pixel_cross_entropy" and "dice" each build an objective over one [2, 4, 4, 3] output with an
`ignore_index` of 255, and the objective's name is the row's type.

### RegistersTheDarknetImporter

`Vision::register_importers` adds one row of the format "darknet". A second call fails with
`InvalidDescription`, "'darknet'" and "already registered".

### TheDarknetRowRefusesACallWithoutFiles

`IO::import_model` of "darknet" with no file fails with `InvalidDescription`, and the message
holds "'darknet'", ".cfg" and "0 files".

### TheReaderRowBuildsADataSection

`Data::make_dataset` of the reader type with the reader "coco_json", the task "detection" and
an absent root fails with `InvalidDescription`. The message holds "the data block 'reader'",
"'coco_json'" and "'yolo_txt'".

Suites: `Medium_Yolo26Forward`, in `sushiai_vision_registration_tests`.

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

## test_registration.cpp

The "mnist" row of the vision module's dataset registry: the keys, splits and layouts it refuses, the files it names when they are absent, and the rank-4 batches of the "image" layout. Needs an engine.

Suites: `Medium_VisionMnist`.

Tests the "mnist" dataset row the vision module registers, through `Data::make_dataset`, and
`Vision::Datasets::try_load_mnist`. The suite's fixture is an `EngineTest`, so every case
needs an engine. It keeps the prefix `Integration_` although the file is in a unit test
folder. The refusal cases point at a directory that does not exist. The one case that reads
images reports itself skipped when the MNIST IDX files are absent.

### TheDatasetRegistryListsMNIST

`find("mnist")` returns a row. `make_dataset` of the type "coco" fails with
`InvalidDescription` and a message that holds "'mnist'".

### MNISTRefusesADirectoryWithoutTheFiles

"mnist" over an absent directory fails with `FileNotFound`. The message holds "data", the
directory, "MNIST train split" and the names of the four IDX files.

### MNISTTestSplitLooksForTheT10kFiles

With "split" set to "test" the failure holds "MNIST test split" and
"t10k-images-idx3-ubyte, labels".

### MNISTRequiresADirectory

A section without "directory" throws an error naming "data", "has no" and "'directory'".

### AnUnknownMNISTKeyIsRejectedByItsOwnFactory

The key "batch_sizes" throws an error naming "data", "'batch_sizes'" and "'batch_size'".

### MNISTRefusesASplitItDoesNotKnow

The split "validation" fails with `InvalidDescription`, and the message holds "'split'",
"'validation'", "'train'" and "'test'".

### AnMNISTLayoutOtherThanFlatOrImageIsRefused

The layout "tiled" fails with `InvalidDescription`, and the message holds "'layout'", "'tiled'",
"'flat'" and "'image'".

### AnImageLayoutYieldsRankFourBatches

With `MNISTLayout::Image`, a batch size of 16 and 32 samples, the dataset's sample shape is
[28, 28, 1]. It has one target, of the role `Data::K_ROLE_LABELS`, a scalar sample shape, the
element type `Int32` and 10 classes. Batch 0 holds a rank-1 `Int32` label tensor of 16 entries
and a [16, 28, 28, 1] input tensor.

### MNISTFallsBackToSyntheticWhenTheFilesAreAbsent

For a directory that does not exist, `mnist_available` is false, `try_load_mnist` fails with
`FileNotFound` and names the directory, and `mnist_expected_paths` is not empty.

## test_ultralytics_files.cpp

Suites: `Medium_UltralyticsFiles`.

Tests the model files `applications/examples/yolov8n/model.json` and
`applications/examples/yolo11n/model.json`, and the task section of those two and of
`applications/examples/yolo26/model.json`. The cases read the files from `SA_REPOSITORY_ROOT`.

### HoldTheirRowsAndTheirInput

`yolov8n` has 23 rows and `yolo11n` has 24; both take a [640, 640, 3] input and name one output node.

### BuildToTheParameterCountOfTheReference

The built model holds the parameters of the Ultralytics summary line, 3,157,200 for `yolov8n` and
2,624,080 for `yolo11n`, less 16. The 16 are the fixed ramp of the distribution expectation, which
Ultralytics counts as a parameter and this framework keeps as no state record.

### PublishTheOneToManyPortsAtTheLevelStrides

Each model publishes six ports: the boxes of the three levels at 80, 40 and 20 positions a side,
64 channels wide, and the scores at the same sides, 80 channels wide. No `one2one` port exists.

### GiveTheFactsOfAnAnchorFreeModel

`derive_free_model_facts` over the traced outputs gives strides 8, 16 and 32, a `reg_max` of 16,
80 classes and no one-to-one copy.

### TheTaskSectionReadsBackAsTheDefaults

`read_free_task` reads the task section of `yolov8n`, `yolo11n` and `yolo26` as the `FreeTask`
defaults: score threshold 0.25 and intersection-over-union suppression at 0.7.
