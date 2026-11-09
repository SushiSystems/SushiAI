# tests/vision/datasets/test_reader_section.cpp


Suites: `Small_DatasetsReaderTable`, `Small_DatasetsReaderTableKeys`,
`Small_DatasetsReaderTableKeyKinds`, `Medium_DatasetsReaderTableFiles`,
`Small_DatasetsTaskTable`, `Small_DatasetsTaskTableStates`, `Small_DatasetsTaskTableCollators`,
`Small_DatasetsTaskTablePositive`, `Small_DatasetsTaskTablePositiveRefusals`,
`Small_DatasetsTaskTableIgnoreRefusals`, `Small_DatasetsReaderSectionRefusals` and
`Medium_DatasetsReaderSection`.

Tests `reader_table` and `find_reader`, `task_table` and `find_task`, and
`make_reader_dataset` in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The section
suites derive from a bench that owns a `HostExecutionBackend`, so no device and no
process-wide engine is reached, and that builds the dataset of a section called "data".

A case is `Small_` when a refusal decides it on the values of the section or of the request,
before a reader lists a file: those cases take a root no test creates. A case that lists files
or reads a batch from them is `Medium_` and writes them under a directory in the system's
temporary directory. The YOLO split is two white raw images of 2 rows and 4 columns under
`images/train`, the second with one object of class 2 in `labels/train/b.txt`. The folder split
is one file under each of `train/cat` and `train/dog`. The numbered split is five white raw
images, image i with one object of class i.

The refusals that were once separate cases are the rows of `section_refusals()`, each named
after what it refuses. The cases that the unit tests of the readers and collators already hold
are not repeated here: the mask size message is `test_fit_mask.cpp`, the failure of an absent
split is `test_yolo_txt.cpp`, the binary and multi-label labels are
`test_binary_collator.cpp` and `test_multi_label_collator.cpp`, and the row of each reader
pointing at a function is `test_reader_table.cpp`.

## ItListsTheEightReaders

`reader_table` lists yolo_txt, image_folder, idx, cifar, coco_json, voc_xml, mask_png and csv
in that order.

## ALookupOfAReaderTheTableLacksFindsNone

`find_reader("tfrecord")` is null.

## ARowListsTheKeysItReads

One test per row, named after its reader. The rows yolo_txt, image_folder, idx and cifar list
no key. coco_json lists `annotations`, `images`, `polygons`; voc_xml lists `class_names`,
`keep_difficult`; mask_png lists `images`, `masks`, `image_suffix`, `mask_suffix`; csv lists
`file`, `features`, `targets`.

## ARowNamesAKeyOfTheWrongKind

Over a root that does not exist, the csv row with `features` as a string fails with
`InvalidDescription`, "data 'features'" and "array", and the coco_json row with `polygons` as 3
fails with "data 'polygons'" and "boolean". Neither opens a file.

## TheCSVRowReadsItsKeysOfTheSection

The csv row, given `file`, the feature c and the targets a, b over a file with the header
`a,b,c`, reads the values 3, 1, 2.

## TheMaskPNGRowRefusesAClassCountOfZero

With `images` "pictures" and `masks` "labels" and no class count, the mask_png row fails with
`InvalidDescription`, "read_mask_png" and "the class count is 0", before it lists a folder.

## TheMaskPNGRowPassesTheClassCountAndTheFolders

With `images` "pictures", `masks` "labels" and a class count of 5, the mask_png row reads one
sample and reports 5 classes.

## TheVOCXMLRowPassesTheClassList

With `class_names` cat, dog over an annotation whose one object is a difficult dog, the
voc_xml row reports 2 classes and no object.

## TheVOCXMLRowPassesTheDifficultFlag

With `keep_difficult` true over the same annotation the sample has one object of class 1.

## TheCIFARRowRefusesAnotherClassCount

The cifar row with a class count of 7 fails with `InvalidDescription`, "7 classes",
"CIFAR-10" and "coarse".

## ARowStatesItsFitAndItsKeys

One test per task. Detection has `SampleFit::Letterbox` and the key `max_objects`.
Classification, multi_label and regression have `SampleFit::Resize` and no key. Binary has
`SampleFit::Resize` and the key `positive`. Segmentation has `SampleFit::Letterbox` and the
keys `ignore_index`, `background`. The names of the rows are in `test_task_table.cpp`.

## ARowBuildsTheCollatorOfItsTask

One test per row of `task_table`. For a list of two named classes with a table of two features
and one target, and a section with `max_objects` 5, the `make_collator` of the row gives a
collator whose `task` is the row's name.

## TheBinaryRowReadsThePositiveClass

Over the classes cat, dog the binary row builds a collator for an empty section, for
`positive` "cat" and for `positive` 0. One test per section text.

## TheBinaryRowRefusesAPositiveClassItLacks

It fails with `InvalidDescription` for `positive` "bird" ("data 'positive' is 'bird'",
"'cat', 'dog'"), for `positive` 2 ("the positive class is 2") and for a class count of 3
("data 'task' is 'binary'", "the samples have 3"). One test per refusal.

## TheSegmentationRowReadsTheIgnoreIndex

For three classes and a size of 4 by 4, `ignore_index` -1 builds a collator whose first target
has the sample shape [4, 4].

## TheSegmentationRowRefusesAnIgnoreIndex

`ignore_index` 1 fails with `InvalidDescription` and "the ignore index 1 is one of the 3", and
4294967296 with "data 'ignore_index' is 4294967296" and "32 bits". One test per refusal.

## ASectionIsRefusedBeforeAnyFileIsRead

One test per row, over a root that does not exist. Each fails with `InvalidDescription` and
names what its row lists:

- AFitOutsideTheList: `fit` "crop"; "data 'fit' is 'crop'" and "'resize', 'letterbox'".
- AnUnknownReader: "tfrecord"; "data 'reader'", "'tfrecord'" and the quoted name of each of the
  eight readers.
- AnUnknownTask: "pose"; "data 'task'", "'pose'" and the quoted name of each of the six tasks.
- AKeyNoSectionReads: `batch_sizes`; "data", "'batch_sizes'" and "'batch_size'".
- AKeyOfAnotherTask: `max_objects` on a classification section; "data" and "'max_objects'".
- AKeyOfAnotherReader: `features` on a yolo_txt detection section; "data", "'features'" and
  "'max_objects'".
- PrefetchWithoutWorkers: "data 'prefetch'" and "data 'workers' is 0".
- ACacheSideWithoutACache: "data 'cache_side'" and "data 'cache'".
- ShuffleOnASectionOfRows: `shuffle` on a csv regression section; "data 'shuffle'",
  "'regression'" and "rows of numbers".
- AnInputOfThreeEntries: "data 'input'", "3 entries" and "[height, width]".
- AnInputExtentOfZero: `input` [4, 0]; "data 'input[1]' is 0".
- ZeroChannels: "data 'channels' is 0".
- AnUnknownPipeline: "mixup"; "data 'pipeline' is 'mixup'", "'none'" and "'default'".
- TheDefaultPipelineOnANonSquareInput: `input` [4, 6]; "square" and "4 by 6".

## ADetectionSectionBuildsLetterboxedBatches

A detection section over the YOLO split with `input` [4, 4], `max_objects` 2 and `classes` 3
gives a dataset named "yolo_txt" with a batch size of 2, one batch and the sample shape
[4, 4, 3]. Its boxes target has the sample shape [2, 4] and its classes target 3 classes. In
batch 0 the first input value is 114 / 255, the valid flags of the slots 0 and 2 are 0 and 1,
and the class of slot 2 is 2.

## ADetectionSectionStatingResizeStretchesTheImages

The same section with `fit` "resize" stretches each uniform 255 image to 4 by 4, so the first
input value of batch 0 is 1 and no padding is left.

## AClassificationSectionTakesTheClassesOfTheList

A classification section over the folder split with `input` [2, 3] and `channels` 1 gives the
sample shape [2, 3, 1] and one target of the role `K_ROLE_LABELS` with 2 classes.

## TheClassCountMayExceedTheList

Over the folder split, `classes` 5 gives a target of 5 classes.

## TheClassCountMayNotFallBelowTheList

`classes` 1 over the folder split fails with `InvalidDescription`, "data 'classes' is 1" and
"name 2 classes". The refusal needs the folders, since the count of the list comes from them.

## ADetectionSectionNeedsMaxObjects

A detection section over the YOLO split without `max_objects` returns `InvalidDescription` with
a message holding "data" and "'max_objects'". The collator reads the key after the reader has
listed the samples, so the split is written.

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

## AnImageTaskOverARowReaderIsRefused

The task classification over the reader csv fails with `InvalidDescription`,
"data 'task' is 'classification'", "reads images" and
"the reader 'csv' lists rows of numbers".

## ARowTaskOverAnImageReaderIsRefused

The task regression over image_folder fails with "data 'task' is 'regression'",
"reads rows of numbers" and "the reader 'image_folder' lists none".

## ACacheHoldsOneRawImagePerSample

With `cache` naming a directory, the build writes two `.saim` files into it.

## ACacheServesTheBatchesWhenTheSourcesAreGone

After the folder `images` is removed, `read(0, batch)` succeeds and input value 12 is 1.0, so
the batch came from the cache.

## AShuffledSectionReadsTheBatchesInTheOrderOfItsSeed

A detection section over the numbered split with a batch size of 1, `shuffle` true and `seed`
7, after `begin_epoch(dataset, 1)`, reads batches whose first object classes are the order
`Data::EpochPermutation` gives for the seed 7, the epoch 1 and 5 samples. Five samples make
the identity order a one in 120 chance.

## ThePipelineKeyChangesTheInputsOfABatch

Over the YOLO split with `input` [16, 16], the sample shape is [16, 16, 3] and the inputs of
batch 0 under `pipeline` "default" differ from those under `pipeline` "none", whose
letterboxed images the default pipeline does not leave as they are.

## WorkersPutTheDatasetBehindALoader

With `workers` 2 and `prefetch` 2 the dataset is a `Data::ThreadedLoader`; without them it is
not.

## WorkersReadTheSameBatchesAhead

Both datasets have the same batch count, and the inputs of every batch are equal between the
two.

## ASegmentationSectionPaintsThePolygonsOfALabelFile

A segmentation section over the reader yolo_txt, with a label file whose one polygon of class
1 covers the top left quarter of a 4 by 4 image and with `background` 2, gives a mask in
batch 0 whose top left 2 by 2 pixels are 1 and whose other pixels are 2.
