# tests/vision/datasets/test_reader_section.cpp


Suites: `Small_DatasetsReaderTable`, `Small_DatasetsTaskTable`, `Medium_DatasetsReaderSection`.

Tests `reader_table` and `find_reader`, `task_table` and `find_task`, and
`make_reader_dataset` in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The
`Medium_DatasetsReaderSection` fixture is an `EngineTest` that builds a dataset from a section
called "data" whose `root` is a directory under the system's temporary directory. Its YOLO
split is two white raw images of 2 rows and 4 columns under `images/train`, the second with
one object of class 2 in `labels/train/b.txt`. Its folder split is one file under each of
`train/cat` and `train/dog`.

## ItListsTheEightReaders

`reader_table` lists yolo_txt, image_folder, idx, cifar, coco_json, voc_xml, mask_png and csv
in that order. Every row has a `read` function and is the row `find_reader` returns for its
name, and `find_reader("tfrecord")` is null.

## ARowListsTheKeysItReads

The rows yolo_txt and cifar list no key. coco_json lists `annotations`, `images`, `polygons`;
voc_xml lists `class_names`, `keep_difficult`; mask_png lists `images`, `masks`,
`image_suffix`, `mask_suffix`; csv lists `file`, `features`, `targets`.

## ARowReadsItsKeysOfTheSection

The csv row, given `file`, the feature c and the targets a, b over a file with the header
`a,b,c`, reads the values 3, 1, 2. It returns `InvalidDescription` with a message holding "data 'features'" and "array"
when `features` is a string. The coco_json row raises one holding "data 'polygons'" and
"boolean" when `polygons` is 3.

## TheMaskPNGRowPassesTheClassCountAndTheFolders

With `images` "pictures" and `masks` "labels" and no class count, the mask_png row fails with
`InvalidDescription`, "read_mask_png" and "the class count is 0". With a class count of 5 it
reads one sample and reports 5 classes.

## TheVOCXMLRowPassesTheClassListAndTheFlag

With `class_names` cat, dog over an annotation whose one object is a difficult dog, the
voc_xml row reports 2 classes and no object. With `keep_difficult` true the sample has one
object of class 1.

## TheCIFARRowRefusesAnotherClassCount

The cifar row with a class count of 7 fails with `InvalidDescription`, "7 classes",
"CIFAR-10" and "coarse".

## EachTaskStatesItsFitAndItsKeys

Detection has `SampleFit::Letterbox` and the key `max_objects`. Classification has
`SampleFit::Resize` and no key. Binary has `SampleFit::Resize` and the key `positive`.
Segmentation has `SampleFit::Letterbox` and the keys `ignore_index`, `background`.
`task_table` lists detection, classification, binary, multi_label, regression and
segmentation in that order, and `find_task("pose")` is null.

## ARowBuildsTheCollatorOfItsTask

For a list of two named classes with a table of two features and one target, and a section
with `max_objects` 5, every row has a `build` function and its `make_collator` gives a
collator whose `task` is the row's name.

## TheBinaryRowReadsThePositiveClass

Over the classes cat, dog the binary row builds a collator for an empty section, for
`positive` "cat" and for `positive` 0. It fails with `InvalidDescription` for `positive`
"bird" ("data 'positive' is 'bird'", "'cat', 'dog'"), for `positive` 2
("the positive class is 2") and for a class count of 3 ("data 'task' is 'binary'",
"the samples have 3").

## TheSegmentationRowReadsTheIgnoreIndex

For three classes and a size of 4 by 4, `ignore_index` -1 builds a collator whose first target
has the sample shape [4, 4]. `ignore_index` 1 fails with `InvalidDescription` and
"the ignore index 1 is one of the 3", and 4294967296 with "data 'ignore_index' is 4294967296"
and "32 bits".

## ADetectionSectionBuildsLetterboxedBatches

A detection section over the YOLO split with `input` [4, 4], `max_objects` 2 and `classes` 3
gives a dataset named "yolo_txt" with a batch size of 2, one batch and the sample shape
[4, 4, 3]. Its boxes target has the sample shape [2, 4] and its classes target 3 classes. In
batch 0 the first input value is 114 / 255, the valid flags of the slots 0 and 2 are 0 and 1,
and the class of slot 2 is 2.

## ADetectionSectionStatingResizeStretchesTheImages

The same section with `fit` "resize" stretches each uniform 255 image to 4 by 4, so the first
input value of batch 0 is 1 and no padding is left.

## AFitOutsideTheListIsRefused

A detection section whose `fit` is "crop" fails with `InvalidDescription`, "data 'fit' is
'crop'" and the names "'resize', 'letterbox'".

## AClassificationSectionTakesTheClassesOfTheList

A classification section over the folder split with `input` [2, 3] and `channels` 1 gives the
sample shape [2, 3, 1] and one target of the role `K_ROLE_LABELS` with 2 classes.

## TheClassCountMayExceedTheListAndNotFallBelowIt

Over the folder split, `classes` 5 gives a target of 5 classes. `classes` 1 fails with
`InvalidDescription`, "data 'classes' is 1" and "name 2 classes".

## AnUnknownReaderListsTheKnownOnes

The reader "tfrecord" fails with `InvalidDescription`, "data 'reader'", "'tfrecord'" and the
quoted name of each of the eight readers.

## AnUnknownTaskListsTheKnownOnes

The task "pose" fails with `InvalidDescription`, "data 'task'", "'pose'" and the quoted name
of each of the six tasks.

## AKeyTheTaskDoesNotReadIsRefused

The key `batch_sizes` returns `InvalidDescription` with a message holding "data", "'batch_sizes'" and "'batch_size'". The
key `max_objects` on a classification section raises one holding "data" and "'max_objects'".

## ADetectionSectionNeedsMaxObjects

A detection section without `max_objects` returns `InvalidDescription` with a message holding "data" and "'max_objects'".

## ValuesOutsideTheirBoundsAreRefused

The build fails with `InvalidDescription` for an `input` of three entries ("data 'input'",
"3 entries", "[height, width]"), for `input` [4, 0] ("data 'input[1]' is 0") and for
`channels` 0 ("data 'channels' is 0").

## TheFailureOfTheReaderIsReturned

A yolo_txt section over an empty root fails with `FileNotFound`, "train.txt" and
"images/train".

## ARegressionSectionReadsTheRowsOfACSVFile

A regression section over a `train.csv` of three records with the target y and a batch size
of 2 gives a dataset named "csv" with one batch, the sample shape [2] and one target of the
role `K_ROLE_VALUES` and the sample shape [1]. Batch 0 holds the inputs 1, 2, 4, 5 and the
values 3, 6.

## ASegmentationSectionFitsTheMaskOfEachImage

A segmentation section over one image of 2 rows and 4 columns and its mask, with `input`
[4, 4], `classes` 3 and `ignore_index` -1, gives the sample shape [4, 4, 3] and one target of
the role `K_ROLE_MASK`, the sample shape [4, 4] and 3 classes. In batch 0 the mask's two rows
0, 1, 2, 0 and 1, 1, 2, 2 lie between two rows of -1.

## AMaskOfAnotherSizeThanItsImageIsRefused

With a 2 by 2 mask for an image of 2 rows and 4 columns, the build succeeds and `read(0,
batch)` fails with `Incompatible`, "masks/train/a.png", "fit_mask" and
"the mask is 2 by 2 pixels; its image is 2 by 4".

## ABinarySectionWritesOneForThePositiveFolder

A binary section over the folders cat and dog with `positive` "cat" has one target of
`Float32`. Batch 0 holds the labels 1 and 0.

## AMultiLabelSectionFlagsTheClassesOfTheObjects

A multi_label section over the YOLO split with `classes` 3 has a target of the sample shape
[3]. Batch 0 holds 0, 0, 0 for the first image and 0, 0, 1 for the second.

## ATaskAndAReaderOfAnotherSampleKindAreRefused

The task classification over the reader csv fails with `InvalidDescription`,
"data 'task' is 'classification'", "reads images" and
"the reader 'csv' lists rows of numbers". The task regression over image_folder fails with
"data 'task' is 'regression'", "reads rows of numbers" and
"the reader 'image_folder' lists none".

## AKeyOfAnotherReaderIsRefused

The key `features` on a yolo_txt detection section returns `InvalidDescription` with a message holding "data",
"'features'" and "'max_objects'".

## ACacheIsWrittenAndRead

With `cache` naming a directory, the build writes two `.saim` files into it. After the folder
`images` is removed, `read(0, batch)` succeeds and input value 12 is 1.0.

## AShuffledSectionWithThePipelineReadsABatch

A detection section with `input` [16, 16], `shuffle` true, `seed` 7 and `pipeline` "default"
builds. After `begin_epoch(dataset, 1)`, `read(0, batch)` succeeds, the sample shape is
[16, 16, 3] and the inputs hold 2 * 16 * 16 * 3 elements.

## APipelineTheTableLacksIsRefused

The pipeline "mixup" fails with `InvalidDescription`, "data 'pipeline' is 'mixup'", "'none'"
and "'default'". The pipeline "default" with `input` [4, 6] fails with `InvalidDescription`,
"square" and "4 by 6".

## WorkersReadTheSameBatchesAhead

With `workers` 2 and `prefetch` 2 the dataset is a `Data::ThreadedLoader`; without them it is
not. Both have the same batch count, and every input element of every batch is equal between
the two.

## KeysThatDoNotApplyAreRefused

The build fails with `InvalidDescription` for `prefetch` without workers ("data 'prefetch'",
"data 'workers' is 0"), for `cache_side` without a cache ("data 'cache_side'",
"data 'cache'") and for `shuffle` on a csv regression section ("data 'shuffle'",
"'regression'", "rows of numbers").

## ASegmentationSectionPaintsThePolygonsOfALabelFile

A segmentation section over the reader yolo_txt, with a label file whose one polygon of class
1 covers the top left quarter of a 4 by 4 image and with `background` 2, gives a mask in
batch 0 whose top left 2 by 2 pixels are 1 and whose other pixels are 2.
