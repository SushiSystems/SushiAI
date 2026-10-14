# tests/vision/datasets

The tests of the module `vision/datasets`, one file per header of `modules/vision/datasets/include/`. The suite prefix of a case gives its ctest label.

## test_cifar.cpp

Suites: `Unit_DatasetsCifar`.

Tests `read_cifar` in `modules/vision/datasets/include/SushiAI/vision/datasets/cifar.hpp` over
batch files the cases write under the system's temporary directory. A record is its label
bytes followed by three planes of 32 by 32 bytes, each plane filled with one level.

### TheCIFAR10TrainingSplitListsItsFiveFilesInOrder

With `data_batch_1.bin` to `data_batch_5.bin` each holding one record labelled with its batch
number, the "train" split of CIFAR-10 has 10 classes, five files and five samples. Sample i
names file i, the file is `data_batch_<i + 1>.bin` and the class is i + 1.

### ASampleRefersToThePlanesAfterItsLabel

A `test_batch.bin` of two records gives the classes 3 and 9. The second sample's stored pixels
start at byte 3074, are `K_CIFAR_SIDE` square with three channels and are `PlaneLayout::Planar`.
`read_stored_pixels` over them gives 3072 bytes whose sampled pixels hold 40, 50, 60.

### CIFAR100TakesTheFineOrTheCoarseLabel

A `train.bin` record with the label bytes 19 and 87 reads as class 87 of 100 under
`CIFARVariant::CIFAR100Fine`, with its pixels at offset 2, and as class 19 of 20 under
`CIFARVariant::CIFAR100Coarse`.

### ALabelOutsideTheClassesNamesItsRecord

A CIFAR-10 file whose second record has the label 10 fails with `InvalidFile`,
"test_batch.bin", "record 2 of 2" and "label 10". A CIFAR-100 record with the coarse label 20
fails with `InvalidFile`, "test.bin", "record 1 of 1" and "label 20".

### AFileThatIsNotWholeRecordsIsRefused

A record one byte short fails with `InvalidFile`, "test_batch.bin", "3072 bytes" and
"3073 bytes". An empty file fails with `InvalidFile`, "test_batch.bin" and "0 bytes".

### AnAbsentFileIsNotFound

With only `data_batch_1.bin` present, the "train" split fails with `FileNotFound` and
"data_batch_2.bin".

### AnotherSplitIsRefused

The split "val" fails with `InvalidDescription` and "'val'".

## test_coco_json.cpp

Suites: `Unit_DatasetsCocoJson`.

Tests `read_coco_json` in
`modules/vision/datasets/include/SushiAI/vision/datasets/coco_json.hpp` over files the cases
write under the system's temporary directory. Unless a case says otherwise the annotation file
is `annotations/instances_train.json` beside a folder `train`. It lists `b.jpg` with id 9 and
`sub/a.jpg` with id 4, both 200 by 100 pixels, and the categories brush (90), person (1) and
cat (17).

### BoxesAndDenseClassesAreRead

The class names come out as person, cat, brush, and the samples as `train/b.jpg` with no object
and `train/sub/a.jpg`. Of that image's four annotations the crowd one and the one of width 0
are left out; the other two have the classes 2 and 0 and no polygon. The box [10, 20, 50, 20]
reads as the centre (35/200, 30/100) and the size (50/200, 20/100), and the sample has no class
index.

### TheFileAndTheFolderMayBeNamed

With `annotation_file` "labels.json" and `image_folder` "pictures", the first sample's file is
`pictures/b.jpg`.

### PolygonsAreKeptWhenAskedFor

By default an annotation with two polygon parts keeps no polygon. With
`COCOSegmentation::Polygons` the image has one object, whose polygon is the part of four
vertices, and its third vertex is (100/200, 50/100). The crowd annotation with a run-length
mask is left out.

### ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor

A file whose second annotation holds a run-length mask reads as two objects by default. With
`COCOSegmentation::Polygons` it fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "annotation 1 holds a run-length mask".

### AMalformedPolygonIsRefused

A polygon part of seven numbers fails with `InvalidFile`, "annotation 0",
"'segmentation' part" and "at least 3 pairs".

### AnUnknownImageOrCategoryNamesTheAnnotation

An annotation of image 5 fails with `InvalidFile`, "instances_train.json", "read_coco_json"
and "annotation 1 has the 'image_id' 5, which 'images' lacks". A crowd annotation of category
2 fails with "annotation 0 has the 'category_id' 2, which 'categories' lacks".

### AMalformedBoxNamesTheAnnotation

A `bbox` of three numbers, and one whose fourth entry is a string, each fail with
`InvalidFile` and "annotation 0 has no 'bbox' of 4 numbers". An `image_id` given as a string
fails with "annotation 0 has no whole number 'image_id'".

### TheImagesAndTheCategoriesAreChecked

Each of these fails with `InvalidFile`: two categories of id 3
("category 0 and category 2 share the id 3"), an image of width 0
("image 1 is 2 by 0 pixels") and an image without `file_name`
("image 0 has no string 'file_name'").

### TheFileMustExistAndHoldItsThreeLists

Without the folder `train` the read fails with `FileNotFound`, "train", "read_coco_json" and
"no such image folder". Without the annotation file it fails with `FileNotFound`,
"annotations/instances_train.json" and "no such file". A file cut after `"images": [` fails
with `InvalidFile` and "not valid JSON", and a file without `categories` with
"the file has no list 'categories'".

### TheListsMayComeInAnyOrder

One file states `images`, `categories` and `annotations` in that order. A second states an
`info` object, then `annotations`, `categories` and `images`. With `COCOSegmentation::Polygons`
both give the same class names, files, classes, box centre x, box height and polygon vertices.
The second sample has the class 2 and a polygon of four vertices, and the first has no polygon.

### AListStatedTwiceIsRefused

A file that states `images` a second time fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "states 'images' twice".

## test_csv.cpp

Suites: `Unit_DatasetsCsv`.

Tests `read_csv` in `modules/vision/datasets/include/SushiAI/vision/datasets/csv.hpp` over
files the cases write under the system's temporary directory. Unless a case names another
file, the reader takes `train.csv` with the default columns.

### TheLastColumnIsTheTargetByDefault

A file with the header `a,b,y` and two records gives one file, two samples and a class count
of 0. The table has two features and one target, the columns a, b, y and the values 1, 2, 3,
-0.5, 4, 10; `features(1)[1]` is 4 and `targets(1)[0]` is 10.

### NamedColumnsAreTakenInTheOrderNamed

Over `data/set.csv` with the header `id,x,y,z`, the features z, x and the targets y, z fail
with `InvalidDescription`, "set.csv", "read_csv" and
"the column 'z' is named as a feature and as a target". With the target y alone the columns
are z, x, y and the values 3, 1, 2, 6, 4, 5.

### QuotedFieldsAndCarriageReturnsAreRead

The file opens with a byte order mark, ends its lines with CRLF and has an empty line. Its
header holds a quoted name with a doubled quote and a comma, a quoted name with a line break,
and ` y ` between blanks. With the first column as the feature, the columns are
`the "x", one` and `y`, and the values are 1.5, 2, 3, 4.

### ANonNumericCellNamesItsLineAndColumn

The cell `five` fails with `InvalidFile`, "train.csv", "line 3", "column 2 ('b')", "read_csv"
and "the cell 'five'". An empty target cell fails with "line 2", "column 3 ('y')" and
"the cell ''". After a record whose quoted field spans lines 2 and 3, the cell `inf` fails
with "line 4", "column 3 ('y')" and "the cell 'inf'".

### ARecordOfAnotherFieldCountIsRefused

A record of two fields on line 4, after an empty line, fails with `InvalidFile`, "train.csv",
"line 4", "read_csv" and "the record has 2 fields; the header has 3".

### MalformedQuotesAreRefused

Each of these fails with `InvalidFile` and "line 2": a quote left open ("column 2",
"the quoted field is not closed"), a letter after a closing quote ("column 1",
"text follows the closing quote") and a quote at the end of an unquoted field ("column 2",
"a quote stands inside a field").

### AColumnTheHeaderLacksIsRefused

Over the header `a,b,y`, the target z fails with `InvalidDescription`, "train.csv",
"no target column 'z'" and "'a', 'b', 'y'". The feature c fails with `InvalidDescription` and
"no feature column 'c'".

### AHeaderMustNameItsColumnsOnceAndLeaveAFeature

The header `a, b ,a` on line 2 fails with `InvalidFile`, "line 2", "column 3" and
"the name 'a' is also the name of column 1". A file of the single column y fails with
`InvalidDescription` and "no column is left as a feature".

### AFileWithoutRecordsIsRefused

With the split "val" and no file, the read fails with `FileNotFound`, "val.csv" and
"read_csv". A `val.csv` of two empty lines fails with `InvalidFile` and
"holds no header row", and one with a header alone with "holds no record below its header".

## test_fit_mask.cpp

Suites: `Unit_DatasetsFitMask`, `Unit_DatasetsLoadSampleMask`.

Tests `fit_mask` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_mask.hpp` and
`load_sample_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_mask.hpp`. The
`load_sample_mask` cases that read a file write it under the system's temporary directory.

### AResizeRepeatsTheNearestClass

A 2 by 2 mask of 0, 1, 2, 3 resized to 4 by 4 has one channel and holds each class in a 2 by 2
block: the rows 0 0 1 1, 0 0 1 1, 2 2 3 3, 2 2 3 3.

### AReductionTakesThePixelUnderEachCentre

A 4 by 4 mask of the bytes 0 to 15 resized to 2 by 2 holds 5, 7, 13, 15.

### ALetterboxPadsWithTheVoidByte

A mask of 2 rows and 4 columns letterboxed to 4 by 4 keeps its two rows in the middle, with a
row of `K_MASK_VOID` above and below.

### ALetterboxThatReducesSamplesTheNearestPixel

A mask of 4 rows and 8 columns holding the bytes 0 to 31, letterboxed to 4 by 4, has the rows
9, 11, 13, 15 and 25, 27, 29, 31 between two rows of `K_MASK_VOID`.

### AMaskMustMatchItsImage

`fit_mask` fails with `Incompatible` and "fit_mask" for a mask of three channels
("the mask has 3 channels") and for a 2 by 2 mask whose image is 2 by 4
("the mask is 2 by 2 pixels; its image is 2 by 4"). A target size of 0 by 4 fails with
`InvalidDescription`, "fit_mask" and "0 by 4 pixels".

### TheMaskFileIsDecoded

The sample names file 1 of the list as its mask, a raw image file of 2 rows and 3 columns.
`load_sample_mask` gives an image of that size and one channel holding the bytes written: 0,
3, 255, 1, 2, 0.

### ASampleMustNameAMaskOfTheList

Over a list of one file and one sample, `load_sample_mask` fails with `InvalidDescription`
for sample 1 ("load_sample_mask", "sample 1 is outside a list of 1 samples"), for a sample
without a mask ("sample 0 has no mask") and for the mask index 4
("sample 0 names the mask file 4 of a list of 1 files").

### AnAbsentMaskFileIsReported

A mask file that does not exist fails with `FileNotFound` and "absent.png".

## test_fit_sample.cpp

Suites: `Unit_DatasetsFitSample`.

Tests `fit_sample` in `modules/vision/datasets/include/SushiAI/vision/datasets/fit_sample.hpp`.
A box corner may differ from its expected value by 1e-4 pixels.

### ResizeScalesTheLabelsWithEachAxis

A gray image of 2 rows and 4 columns at level 100, resized to 4 by 4, gives 16 bytes of 100
in one channel. The fitted sample keeps the file index 6 and the class index 3. Its one
object keeps the class 1, and the box of centre (0.5, 0.5) and size (0.5, 0.5) becomes the
corners (1, 1) and (3, 3).

### LetterboxPadsAndMovesTheLabels

The same shape at level 200, letterboxed to 4 by 4, has rows 1 and 2 at 200 and rows 0 and 3
at 114. The sample has no class index, and the box of centre (0.5, 0.5), width 0.5 and height
1.0 becomes the corners (1, 1) and (3, 3).

### AnObjectWithoutAreaIsRemoved

Of two objects, the one of class 2 with a width of 0 is removed and the one of class 5 is
kept.

### AnInputSizeOutsideTheBoundsIsRefused

A target size of 0 by 4 fails with `InvalidDescription`, "fit_sample" and "0 by 4".

### ASourceOfNoPixelsIsAFailure

A default `ImageView` as the source fails with `InvalidFile` and "fit_sample".

## test_idx.cpp

Suites: `Unit_DatasetsIdx`.

Tests `read_idx` in `modules/vision/datasets/include/SushiAI/vision/datasets/idx.hpp` over IDX
files the cases write under the system's temporary directory. The training split of the first
two cases is three images of 2 rows and 3 columns holding the bytes 100 to 117, with the
labels 4, 0 and 2.

### ASampleHasItsLabelAndThePlaceOfItsPixels

The list has one file, `train-images-idx3-ubyte`, no class names, a class count of 5 and three
samples. Sample i names file 0, has the class of its label and no object, and its stored
pixels start at byte 16 + 6i, are 2 by 3 with one channel and are `PlaneLayout::Interleaved`.

### ASamplesPixelsReadBackFromTheImageFile

`read_stored_pixels` over the third sample gives the bytes 112 to 117.

### TheTestSplitReadsTheT10kFiles

With the split "test", `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` are read: one
sample of class 7 and a class count of 8.

### NamedFilesAreReadForAnySplit

With the split "balanced" and the files `images.idx` and `labels.idx` named, two samples are
read. The image file has four dimensions, the last being 3; the second sample has three
channels and its pixels start at byte 23.

### ASplitWithoutAFileNameIsRefused

The split "validation" with no file named fails with `InvalidDescription` and "validation".

### AnAbsentFileIsNotFound

With the image file alone present, the read fails with `FileNotFound` and
"train-labels-idx1-ubyte".

### CountsThatDifferAreRefused

Two images with three labels fail with `InvalidFile`, "train-images-idx3-ubyte", "2 images",
"train-labels-idx1-ubyte" and "3 labels".

### AnImageFileOfAnotherFormIsRefused

An image file of two dimensions fails with `InvalidFile` and "2 dimensions", and one of
element code 0x0C with `InvalidFile` and "code 12". One whose fourth dimension is 5 fails with
`Unsupported` and "5 channels". Each message names "train-images-idx3-ubyte".

### AnImageFileWithoutAnImageIsRefused

An image file whose first extent is 0 fails with `InvalidFile`, "train-images-idx3-ubyte" and
"no image".

## test_idx_file.cpp

Suites: `Unit_DatasetsIdxFile`.

Tests `idx_element_bytes`, `read_idx_header` and `read_idx_payload` in
`modules/vision/datasets/include/SushiAI/vision/datasets/idx_file.hpp` over files the cases
write under the system's temporary directory.

### ElementBytesFollowTheFormat

`idx_element_bytes` is 1 for `UInt8` and `Int8`, 2 for `Int16`, 4 for `Int32` and `Float32`,
and 8 for `Float64`.

### AHeaderGivesTheElementTheExtentsAndThePayloadSize

A file of element code 0x08 with the extents 2, 3, 4 and 24 payload bytes reads as `UInt8`,
those extents, a header of 16 bytes and a payload of 24 bytes.

### AnExtentIsReadMostSignificantByteFirst

A file with the single extent 258 reads that extent as 258.

### ThePayloadSizeCountsTheElementBytes

A file of element code 0x0C with the extent 3 reads as `Int32` with a payload of 12 bytes.

### ThePayloadIsEveryByteAfterTheHeader

`read_idx_payload` over a file of five bytes gives 7, 0, 9, 3, 1.

### AnAbsentFileIsNotFound

`read_idx_header` of a path that does not exist fails with `FileNotFound` and "absent.idx".

### AMalformedHeaderIsRefused

`read_idx_header` fails with `InvalidFile` and the file's name for a file of three bytes
("leading bytes"), a non-zero second byte ("two zero bytes"), the element code 10
("element code 10"), a rank of 0 ("no dimension") and a second extent cut after one byte
("dimension 1 of 2").

### AFileOfAnotherSizeThanItsHeaderDeclaresIsRefused

A file with the extents 2, 2 and three payload bytes fails with `InvalidFile`, "short.idx",
"declares 4 bytes" and "holds 15 bytes". A file with the extent 2 and three payload bytes
fails with "long.idx", "declares 2 bytes" and "holds 11 bytes".

### ExtentsThatOverflowAreRefused

A file of element code 0x0E with three extents of 0xFFFFFFFF fails with `InvalidFile`,
"huge.idx" and "overflow".

## test_image_folder.cpp

Suites: `Unit_DatasetsImageFolder`.

Tests `read_image_folder` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_folder.hpp` and
`has_image_extension` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp`. The folder
cases write one-byte files under the system's temporary directory.

### ImageExtensionsIgnoreCase

`has_image_extension` is true for `a.jpg`, `a.JPEG`, `dir/a.PNG` and `a.saim`, and false for
`a.txt`, `a.jpg.bak` and `jpg`.

### ClassesAreTheSortedFolderNames

The folders zebra, ant and moth give the class names ant, moth, zebra and a class count of 3.
Sample i of the three has the class i, no stored pixels and no object.

### AClassListsItsImageFilesSortedAtAnyDepth

The list holds four files in the order `cat/a.JPG`, `cat/b.png`, `cat/more/c.saim`,
`dog/a.png`, and leaves `cat/notes.txt` out. The third sample has the class 0 and the fourth
the class 1.

### ASplitNamesAFolderOfTheRoot

The split "val" gives the classes no and yes and two samples. The file `val/stray.png` and the
folder `train` add nothing.

### AFolderWithoutAnImageIsStillAClass

With an empty folder `a` and one image in `b`, the class count is 2 and the one sample has the
class 1.

### AnAbsentSplitIsNotFound

The split "train" in an empty root fails with `FileNotFound` and "train".

### ASplitWithoutAClassOrAnImageIsRefused

A split holding only a file fails with `InvalidFile`, "flat" and "no class folder". A split
whose one class folder holds only a text file fails with `InvalidFile`, "text" and
"no image file".

## test_load_sample_image.cpp

Suites: `Unit_DatasetsLoadSampleImage`.

Tests `load_sample_image` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_image.hpp`. The cases
that read a file write it under the system's temporary directory.

### StoredPixelsAreReadFromTheirOffset

Over a file of the bytes 0 to 9, a sample whose stored pixels are 2 by 2 with one channel at
offset 2 loads as a 2 by 2 image of one channel holding 2, 3, 4, 5.

### AnImageFileIsDecoded

A sample without stored pixels, whose file is a raw RGB image of 2 rows and 3 columns written
by `write_raw_image_file`, loads as an image of that size and three channels holding the bytes
written.

### ASampleOutsideTheListIsRefused

Sample 3 of a list of one sample fails with `InvalidDescription`, "load_sample_image",
"sample 3" and "1 samples".

### AFileIndexOutsideTheListIsRefused

A sample naming file 5 of a list of one file fails with `InvalidDescription`,
"load_sample_image" and "file 5".

### AnAbsentFileIsNamed

An image file that does not exist fails with `FileNotFound` and "absent.saim".

## test_mask_png.cpp

Suites: `Unit_DatasetsMaskPng`.

Tests `read_mask_png` in `modules/vision/datasets/include/SushiAI/vision/datasets/mask_png.hpp`
over one-byte files the cases write under the system's temporary directory. Unless a case says
otherwise the options name four classes and the split "train", so the images are under
`images/train` and the masks under `masks/train`.

### EachImageRefersToTheMaskOfItsStem

The list has a class count of 4, no class names, two samples and four files. The first sample
is `images/train/a/c.jpeg` with the mask `masks/train/a/c.png`, and has no class index and no
object. The second is `images/train/b.jpg` with `masks/train/b.png`. The files `notes.md` and
`index.txt` are not listed.

### TheFoldersMayBeNamedAndTheSplitLeftOut

With an empty split, `image_folder` "leftImg8bit" and `mask_folder` "gtFine", the one sample
is `leftImg8bit/a.png` with the mask `gtFine/a.png`.

### AnImageWithoutAMaskIsRefused

An image `b.jpg` without a mask fails with `FileNotFound`, "images/train/b.jpg",
"read_mask_png", "has no mask" and "masks/train/b.png".

### AMaskWithoutAnImageIsRefused

A mask `z.png` without an image fails with `InvalidFile`, "masks/train/z.png", "read_mask_png"
and "the mask has no image".

### TwoImagesOfOneStemAreRefused

The images `a.jpg` and `a.png` with one mask `a.png` fail with `InvalidFile`,
"images/train/a.png", "shares the mask" and "masks/train/a.png".

### TheOptionsAndTheFoldersAreChecked

A class count of 0 fails with `InvalidDescription`, "read_mask_png" and
"the class count is 0". An absent image folder fails with `FileNotFound`, "images/train" and
"no such directory", and an absent mask folder with `FileNotFound`, "masks/train" and
"no such directory". Two empty folders fail with `InvalidFile`, "images/train" and
"holds no image".

### TheSuffixesPairNamesThatDiffer

With `image_suffix` "_leftImg8bit" and `mask_suffix` "_gtFine_labelIds" under the folders
`leftImg8bit` and `gtFine`, the two images pair with their `_gtFine_labelIds.png` masks, and
the list has two samples and four files. The mask `aachen_000000_gtFine_color.png` is not
listed.

### AnImageWithoutTheImageSuffixIsRefused

With `image_suffix` "_left", the image `b.png` fails with `InvalidFile`, "images/train/b.png",
"read_mask_png" and "does not end in '_left'".

### AMaskOfTheSuffixWithoutAnImageIsRefused

With `mask_suffix` "_ids", the mask `b_ids.png` without an image fails with `InvalidFile`,
"masks/train/b_ids.png" and "the mask has no image".

## test_polygon_mask.cpp

Suites: `Unit_DatasetsPolygonMask`.

Tests `paint_polygon_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/polygon_mask.hpp` over masks of 4 by
4 pixels. Each object is a rectangle given as a polygon and as its box.

### APolygonPaintsThePixelsWhoseCentresItHolds

A rectangle of class 2 from (1, 1) to (3, 3) over the background `K_MASK_VOID` gives a mask of
one channel whose four middle pixels are 2 and whose other twelve are 255.

### TheLaterOfTwoObjectsWinsAnOverlap

A rectangle of class 0 from (0, 0) to (3, 2), then one of class 1 from (2, 0) to (4, 4), over
the background 7 give the rows 0 0 1 1, 0 0 1 1, 7 7 1 1, 7 7 1 1.

### NoObjectIsTheBackgroundEverywhere

An empty label set over the background 3 gives sixteen bytes of 3.

### RefusesAnObjectItCannotPaint

An object with a box and no polygon fails with `InvalidDescription`, "paint_polygon_mask",
"object 0" and "no polygon". An object of class 255 fails with `InvalidDescription`,
"the class 255" and "0 to 254".

## test_reader_dataset.cpp

Suites: `Unit_DatasetsReaderDataset`.

Tests `ReaderDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/reader_dataset.hpp`. The fixture is an
`EngineTest`. Its list stores gray samples of 2 by 2 pixels in one file `pixels.bin` of rising
bytes, written under the system's temporary directory, so sample i opens with the byte 4i and
has the class i modulo 2. The default options are batches of two such inputs under the name
"tiny", with a classification collator over two classes.

### ItDescribesItsBatches

Over five samples the dataset reports a batch size of 2, the sample shape [2, 2, 1], two
batches, the name "tiny" and one target of the role `K_ROLE_LABELS`.

### ABatchHoldsItsSamplesInListOrder

Over four samples, batch 1 holds the inputs (8 + i) / 255 for i from 0 to 7, each within 1e-6,
and the labels 0 and 1.

### ADetectionBatchLetterboxesImagesAndBoxes

Two raw RGB files of 2 rows and 4 columns at level 255 are letterboxed to 4 by 4, with a
detection collator of 3 classes and 2 slots. The first input value is 114 / 255 and value 12
is 1.0. The second image's object fills slot 2: valid is 0, 1, 0 at the slots 0, 2 and 3, the
class is 2 and the box is (1, 1, 3, 3) within 1e-4.

### MakeRefusesOptionsOutsideTheirBounds

`ReaderDataset::make` fails with `InvalidDescription` for a null collator ("ReaderDataset",
"tiny", "no collator"), a size of 0 by 2 ("0 by 2"), five channels ("5 channels"), a batch
size of 0 ("batch size of 0") and a batch size of 3 over two samples ("2 samples",
"one batch of 3").

### ABatchOutsideTheEpochStopsTheProcess

Over two samples, `read(1, batch)` is fatal with "ReaderDataset", "1 batches" and "batch 1".

### AReadReusesTheTensorsItFilledBefore

After `read(0, batch)` and `read(1, batch)` into one batch, the inputs and the labels have the
data pointers of the first read, and the first input value is 8 / 255.

### AReadDoesNotWriteATensorSomethingElseHolds

With a copy of the input tensor kept after `read(0, batch)`, `read(1, batch)` gives inputs at
another data pointer. The kept tensor still opens with 0 and the batch opens with 8 / 255.

### ItIsADatasetATrainerCanRead

Through a `Data::Dataset` reference, `read(1, batch)` gives as many targets as `targets()`
lists, the first with the same role, and inputs whose first extent is `batch_size()`.

### AnImageOfAnotherChannelCountNamesItsFile

With the options asking for three channels over the gray list, `read(0, batch)` fails with
`Incompatible`, "pixels.bin", "ReaderDataset", "1 channels" and "reads 3".

### AFileAbsentAtReadTimeIsNamed

With the list's file replaced by a path that does not exist, `make` succeeds and
`read(0, batch)` fails with `InvalidFile` and "gone.bin".

### AShuffledEpochReadsEverySampleOnceInTheOrderOfTheSeed

With `shuffle` on and the seed 42 over eight samples, the rows read across the four batches
are the samples in the order `Data::EpochPermutation(42, 0, 8)` gives, and each of the eight
is read once.

### BeginEpochSelectsTheOrderOfTheEpoch

After `Data::begin_epoch(dataset, 3)` the samples are read in the order of
`EpochPermutation(42, 3, 8)`, which differs from the order of epoch 0. After
`begin_epoch(dataset, 0)` they are read in the order of epoch 0.

### WithoutShuffleEveryEpochKeepsTheOrderOfTheList

Without `shuffle`, six samples are read as 0, 1, 2, 3, 4, 5, and again in that order after
`begin_epoch(dataset, 5)`.

### MakeRefusesANullPipeline

The form of `make` that takes a pipeline fails for a null one with `InvalidDescription`,
"ReaderDataset", "tiny" and "no pipeline".

## test_reader_section.cpp

Suites: `Unit_DatasetsReaderTable`, `Unit_DatasetsTaskTable`, `Unit_DatasetsReaderSection`.

Tests `reader_table` and `find_reader`, `task_table` and `find_task`, and
`make_reader_dataset` in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The
`Unit_DatasetsReaderSection` fixture is an `EngineTest` that builds a dataset from a section
called "data" whose `root` is a directory under the system's temporary directory. Its YOLO
split is two white raw images of 2 rows and 4 columns under `images/train`, the second with
one object of class 2 in `labels/train/b.txt`. Its folder split is one file under each of
`train/cat` and `train/dog`.

### ItListsTheEightReaders

`reader_table` lists yolo_txt, image_folder, idx, cifar, coco_json, voc_xml, mask_png and csv
in that order. Every row has a `read` function and is the row `find_reader` returns for its
name, and `find_reader("tfrecord")` is null.

### ARowListsTheKeysItReads

The rows yolo_txt and cifar list no key. coco_json lists `annotations`, `images`, `polygons`;
voc_xml lists `class_names`, `keep_difficult`; mask_png lists `images`, `masks`,
`image_suffix`, `mask_suffix`; csv lists `file`, `features`, `targets`.

### ARowReadsItsKeysOfTheSection

The csv row, given `file`, the feature c and the targets a, b over a file with the header
`a,b,c`, reads the values 3, 1, 2. It raises an error holding "data 'features'" and "array"
when `features` is a string. The coco_json row raises one holding "data 'polygons'" and
"boolean" when `polygons` is 3.

### TheMaskPNGRowPassesTheClassCountAndTheFolders

With `images` "pictures" and `masks` "labels" and no class count, the mask_png row fails with
`InvalidDescription`, "read_mask_png" and "the class count is 0". With a class count of 5 it
reads one sample and reports 5 classes.

### TheVOCXMLRowPassesTheClassListAndTheFlag

With `class_names` cat, dog over an annotation whose one object is a difficult dog, the
voc_xml row reports 2 classes and no object. With `keep_difficult` true the sample has one
object of class 1.

### TheCIFARRowRefusesAnotherClassCount

The cifar row with a class count of 7 fails with `InvalidDescription`, "7 classes",
"CIFAR-10" and "coarse".

### EachTaskStatesItsFitAndItsKeys

Detection has `SampleFit::Letterbox` and the key `max_objects`. Classification has
`SampleFit::Resize` and no key. Binary has `SampleFit::Resize` and the key `positive`.
Segmentation has `SampleFit::Letterbox` and the keys `ignore_index`, `background`.
`task_table` lists detection, classification, binary, multi_label, regression and
segmentation in that order, and `find_task("pose")` is null.

### ARowBuildsTheCollatorOfItsTask

For a list of two named classes with a table of two features and one target, and a section
with `max_objects` 5, every row has a `build` function and its `make_collator` gives a
collator whose `task` is the row's name.

### TheBinaryRowReadsThePositiveClass

Over the classes cat, dog the binary row builds a collator for an empty section, for
`positive` "cat" and for `positive` 0. It fails with `InvalidDescription` for `positive`
"bird" ("data 'positive' is 'bird'", "'cat', 'dog'"), for `positive` 2
("the positive class is 2") and for a class count of 3 ("data 'task' is 'binary'",
"the samples have 3").

### TheSegmentationRowReadsTheIgnoreIndex

For three classes and a size of 4 by 4, `ignore_index` -1 builds a collator whose first target
has the sample shape [4, 4]. `ignore_index` 1 fails with `InvalidDescription` and
"the ignore index 1 is one of the 3", and 4294967296 with "data 'ignore_index' is 4294967296"
and "32 bits".

### ADetectionSectionBuildsLetterboxedBatches

A detection section over the YOLO split with `input` [4, 4], `max_objects` 2 and `classes` 3
gives a dataset named "yolo_txt" with a batch size of 2, one batch and the sample shape
[4, 4, 3]. Its boxes target has the sample shape [2, 4] and its classes target 3 classes. In
batch 0 the first input value is 114 / 255, the valid flags of the slots 0 and 2 are 0 and 1,
and the class of slot 2 is 2.

### AClassificationSectionTakesTheClassesOfTheList

A classification section over the folder split with `input` [2, 3] and `channels` 1 gives the
sample shape [2, 3, 1] and one target of the role `K_ROLE_LABELS` with 2 classes.

### TheClassCountMayExceedTheListAndNotFallBelowIt

Over the folder split, `classes` 5 gives a target of 5 classes. `classes` 1 fails with
`InvalidDescription`, "data 'classes' is 1" and "name 2 classes".

### AnUnknownReaderListsTheKnownOnes

The reader "tfrecord" fails with `InvalidDescription`, "data 'reader'", "'tfrecord'" and the
quoted name of each of the eight readers.

### AnUnknownTaskListsTheKnownOnes

The task "pose" fails with `InvalidDescription`, "data 'task'", "'pose'" and the quoted name
of each of the six tasks.

### AKeyTheTaskDoesNotReadIsRefused

The key `batch_sizes` raises an error holding "data", "'batch_sizes'" and "'batch_size'". The
key `max_objects` on a classification section raises one holding "data" and "'max_objects'".

### ADetectionSectionNeedsMaxObjects

A detection section without `max_objects` raises an error holding "data" and "'max_objects'".

### ValuesOutsideTheirBoundsAreRefused

The build fails with `InvalidDescription` for an `input` of three entries ("data 'input'",
"3 entries", "[height, width]"), for `input` [4, 0] ("data 'input[1]' is 0") and for
`channels` 0 ("data 'channels' is 0").

### TheFailureOfTheReaderIsReturned

A yolo_txt section over an empty root fails with `FileNotFound`, "train.txt" and
"images/train".

### ARegressionSectionReadsTheRowsOfACSVFile

A regression section over a `train.csv` of three records with the target y and a batch size
of 2 gives a dataset named "csv" with one batch, the sample shape [2] and one target of the
role `K_ROLE_VALUES` and the sample shape [1]. Batch 0 holds the inputs 1, 2, 4, 5 and the
values 3, 6.

### ASegmentationSectionFitsTheMaskOfEachImage

A segmentation section over one image of 2 rows and 4 columns and its mask, with `input`
[4, 4], `classes` 3 and `ignore_index` -1, gives the sample shape [4, 4, 3] and one target of
the role `K_ROLE_MASK`, the sample shape [4, 4] and 3 classes. In batch 0 the mask's two rows
0, 1, 2, 0 and 1, 1, 2, 2 lie between two rows of -1.

### AMaskOfAnotherSizeThanItsImageIsRefused

With a 2 by 2 mask for an image of 2 rows and 4 columns, the build succeeds and `read(0,
batch)` fails with `Incompatible`, "masks/train/a.png", "fit_mask" and
"the mask is 2 by 2 pixels; its image is 2 by 4".

### ABinarySectionWritesOneForThePositiveFolder

A binary section over the folders cat and dog with `positive` "cat" has one target of
`Float32`. Batch 0 holds the labels 1 and 0.

### AMultiLabelSectionFlagsTheClassesOfTheObjects

A multi_label section over the YOLO split with `classes` 3 has a target of the sample shape
[3]. Batch 0 holds 0, 0, 0 for the first image and 0, 0, 1 for the second.

### ATaskAndAReaderOfAnotherSampleKindAreRefused

The task classification over the reader csv fails with `InvalidDescription`,
"data 'task' is 'classification'", "reads images" and
"the reader 'csv' lists rows of numbers". The task regression over image_folder fails with
"data 'task' is 'regression'", "reads rows of numbers" and
"the reader 'image_folder' lists none".

### AKeyOfAnotherReaderIsRefused

The key `features` on a yolo_txt detection section raises an error holding "data",
"'features'" and "'max_objects'".

### ACacheIsWrittenAndRead

With `cache` naming a directory, the build writes two `.saim` files into it. After the folder
`images` is removed, `read(0, batch)` succeeds and input value 12 is 1.0.

### AShuffledSectionWithThePipelineReadsABatch

A detection section with `input` [16, 16], `shuffle` true, `seed` 7 and `pipeline` "default"
builds. After `begin_epoch(dataset, 1)`, `read(0, batch)` succeeds, the sample shape is
[16, 16, 3] and the inputs hold 2 * 16 * 16 * 3 elements.

### APipelineTheTableLacksIsRefused

The pipeline "mixup" fails with `InvalidDescription`, "data 'pipeline' is 'mixup'", "'none'"
and "'default'". The pipeline "default" with `input` [4, 6] fails with `InvalidDescription`,
"square" and "4 by 6".

### WorkersReadTheSameBatchesAhead

With `workers` 2 and `prefetch` 2 the dataset is a `Data::ThreadedLoader`; without them it is
not. Both have the same batch count, and every input element of every batch is equal between
the two.

### KeysThatDoNotApplyAreRefused

The build fails with `InvalidDescription` for `prefetch` without workers ("data 'prefetch'",
"data 'workers' is 0"), for `cache_side` without a cache ("data 'cache_side'",
"data 'cache'") and for `shuffle` on a csv regression section ("data 'shuffle'",
"'regression'", "rows of numbers").

### ASegmentationSectionPaintsThePolygonsOfALabelFile

A segmentation section over the reader yolo_txt, with a label file whose one polygon of class
1 covers the top left quarter of a 4 by 4 image and with `background` 2, gives a mask in
batch 0 whose top left 2 by 2 pixels are 1 and whose other pixels are 2.

## test_sample_cache.cpp

Suites: `Unit_DatasetsSampleCache`.

Tests `write_sample_cache`, `use_sample_cache` and `sample_cache_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/sample_cache.hpp`. The source images
are raw image files the cases write under the system's temporary directory, and the cache is
the folder `cache` beside them.

### TheFirstPassWritesAndTheSecondReuses

Over two RGB images of 2 by 3 and 4 by 2 pixels, the first `write_sample_cache` reports 2
written, 0 reused and the byte count of the two raw image files. The second reports 0 written,
2 reused and 0 bytes.

### AListPointedAtTheCacheLoadsTheSamePixels

After a write, `use_sample_cache` gives a list whose one file lies in the cache directory, has
the extension `.saim` and a name that starts with "a-". `load_sample_image` over it gives the
size, the three channels and every byte of the source image.

### ALongestSideReducesOnlyWhatExceedsIt

With `longest_side` 4, an image of 4 rows and 8 columns at level 200 loads from the cache as 2
rows and 4 columns with a first byte of 200. A 2 by 2 image loads as 2 by 2.

### AnImageThatChangedIsStale

After the source is rewritten with another size, `use_sample_cache` fails with `FileNotFound`,
"images/a.jpg", "sample cache" and "changed since". A second write reports 1 written and 0
reused, and the image then loads from the cache with a width of 3 and a first byte of 2.

### StoredPixelsAndMasksAreLeftAlone

For a list of one sample with stored pixels and one sample with an image and a mask, the write
reports 1 written. In the list `use_sample_cache` returns, the file of stored pixels and the
mask keep their paths and the image's path differs.

### ANameFollowsTheSourceAndTheOptions

`sample_cache_path` returns the same path twice for one source and one set of options, and its
file name is 23 characters long. With `longest_side` 320 the path differs. A source that does
not exist fails with `FileNotFound` and "gone.jpg".

### RefusesOptionsItCannotServe

`write_sample_cache` fails with `InvalidDescription` for options without a directory
("no directory"), for `longest_side` -1 ("a longest side of -1") and for `longest_side` 1 over
a list whose sample has a mask ("a longest side of 1", "masks").

## test_stored_pixels.cpp

Suites: `Unit_DatasetsStoredPixels`.

Tests `StoredPixels` and `read_stored_pixels` in
`modules/vision/datasets/include/SushiAI/vision/datasets/stored_pixels.hpp`. Three cases read
a file written under the system's temporary directory and the others read a string stream.

### ByteCountIsTheProductOfTheExtents

`byte_count` of 4 by 5 pixels with three channels is 60.

### InterleavedBytesAreCopiedFromTheOffset

Over a file of the bytes 0 to 19, interleaved pixels of 2 rows, 3 columns and two channels at
offset 5 read as an image of that shape holding the bytes 5 to 16.

### PlanarBytesAreInterleaved

Planar pixels of 2 by 2 with three channels at offset 1, stored as the planes 1, 2, 3, 4, then
11, 12, 13, 14, then 21, 22, 23, 24, read as 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24.

### APlaneLongerThanOneChunkIsCopiedWhole

Planar pixels of 100 by 100 with two channels read so that, at every 997th pixel, channel 0
holds the byte of the first plane and channel 1 the byte of the second.

### AFileThatEndsEarlyIsInvalid

Twelve bytes asked at offset 4 of a file of ten fail with `InvalidFile`, "short.bin" and
"6 of 12".

### AFileThatCannotBeOpenedIsInvalid

A path that does not exist fails with `InvalidFile` and "absent.bin".

### ExtentsOutsideAnImageAreRefused

A size of 0 by 2 fails with `InvalidDescription` and "0 by 2", and five channels with
`InvalidDescription` and "5 channels".

## test_table_dataset.cpp

Suites: `Unit_DatasetsTableDataset`.

Tests `TableDataset` in
`modules/vision/datasets/include/SushiAI/vision/datasets/table_dataset.hpp`. The fixture is an
`EngineTest`. Its list names the file `rows.csv` and holds rows of two features and one
target, row i being 10i, 10i + 1 and 10i + 2. The default options are batches of two rows
under the name "rows", with a `RegressionCollator` of two features and one target.

### ItDescribesItsBatches

Over five rows the dataset reports the name "rows", a batch size of 2, two batches, the sample
shape [2] and one target of the role `K_ROLE_VALUES` and the sample shape [1].

### ABatchHoldsItsRowsInListOrder

Through a `Data::Dataset` reference, batch 1 holds the inputs 20, 21, 30, 31 and the values
22, 32. A following `read(0, batch)` writes the same input tensor, whose first value is then 0.

### MakeRefusesWhatItCannotBatch

`TableDataset::make` fails with `InvalidDescription` for a null collator ("TableDataset",
"'rows' has no collator"), a list without a table ("a table of 0 features"), a table one
number short ("4 samples and 11 numbers", "a row holds 3"), a batch size of 0
("a batch size of 0") and one row for batches of two ("1 samples", "one batch of 2").

### ABatchOutsideTheEpochStopsTheProcess

Over two rows, `read(1, batch)` is fatal with "TableDataset", "1 batches" and "batch 1".

### TheFailureOfTheCollatorIsReturned

With a collator of two features and two targets over the list of one target, `make` succeeds
and `read(0, batch)` fails with `InvalidDescription`, "rows.csv", "RegressionCollator" and
"2 features and 1 targets".

## test_voc_xml.cpp

Suites: `Unit_DatasetsVocXml`.

Tests `read_voc_xml` in `modules/vision/datasets/include/SushiAI/vision/datasets/voc_xml.hpp`
over files the cases write under the system's temporary directory. Unless a case says
otherwise the split is "train", listed in `ImageSets/Main/train.txt`. The sample annotation
names `first.jpg`, a size of 200 by 100 and some objects, each with a name, a difficult flag
and the box edges 11, 21, 60, 40.

### TheListedAnnotationsAreReadWithSortedClasses

The list file names second, an empty line, then first followed by " 1", with a CRLF on its
first line. The class names come out as "cat & co", "dog", and the samples as
`JPEGImages/second.jpg` with no object and `JPEGImages/first.jpg`. That sample has no class
index and two objects of the classes 1 and 0; its first box has the centre (35/200, 30/100)
and the size (50/200, 20/100). `Annotations/unlisted.xml` is not read.

### AnEmptySplitReadsEveryAnnotation

With an empty split the annotations `a.xml` and `b.xml` are read in that order, giving
`JPEGImages/a.jpg` and `JPEGImages/first.jpg`. `Annotations/readme.txt` is not read.

### ADifficultObjectIsDroppedUnlessKept

Over a difficult dog and a cat, the default options give the class names cat and one object.
`VOCDifficult::Kept` gives cat, dog and two objects.

### AStatedClassListGivesTheIndices

With `class_names` person, dog, cat the class count is 3 and a dog has the class 1. With
person, cat the read fails with `InvalidFile`, "Annotations/first.xml", "line 4",
"read_voc_xml" and "the class 'dog'". With dog, cat, dog it fails with `InvalidDescription`,
"read_voc_xml", "names 'dog' twice" and "at 0 and at 2".

### AMalformedAnnotationNamesItsFileAndLine

An annotation that closes `annotation` inside an open `size` fails with `InvalidFile`,
"Annotations/first.xml", "line 3", "parse_xml" and "closes '<size>'".

### AMissingElementOrNumberIsRefused

Each of these fails with `InvalidFile` and "first.xml": an annotation without `size`
("line 1", "read_voc_xml", "'annotation' has no 'size'"), a root named `image` ("line 1",
"the root element is 'image'"), a width of "wide" ("line 3", "'width' holds 'wide'"), an
object without `bndbox` ("line 4", "'object' has no 'bndbox'") and a box whose maximum lies
before its minimum ("line 7", "the box ends at (5, 2), before its start (9, 1)").

### TheListAndItsStemsMustExist

Without the list file the read fails with `FileNotFound`, "read_voc_xml", "the split 'train'"
and "ImageSets/Main/train.txt". A stem on line 3 without an annotation fails with
`FileNotFound`, "ImageSets/Main/train.txt", "line 3" and "Annotations/missing.xml". A list of
one empty line fails with `InvalidFile` and "holds no annotation".

## test_xml_element.cpp

Suites: `Unit_DatasetsXmlElement`.

Tests `parse_xml` and `XMLElement` in
`modules/vision/datasets/include/SushiAI/vision/datasets/xml_element.hpp` over strings.

### ElementsTextAndLinesAreRead

A root `a` on line 1 with the children `b`, `c` and an empty `b` reads with no text of its own
and three children. The first `b` has the text "one" and the line 2, the `d` inside `c` has
the text "two", and the last `b` has the line 4 and no text. `find_child` returns the first
`b` for "b", the `c` for "c" and null for "e".

### TheFiveEntitiesAreReplaced

The text `a &amp; b &lt;c&gt; &quot;d&quot; &apos;e&apos;` reads as `a & b <c> "d" 'e'`.

### TheDeclarationCommentsAndAttributesAreSkipped

A document with a byte order mark, an XML declaration, comments before, inside and after the
root, and attributes on the root and its child reads as the root `a` on line 3 with the text
"t" and one child `b`.

### AClosingTagOfAnotherNameIsRefused

`</a>` on line 3 after `<b>` on line 2 fails with `InvalidFile`, "line 3", "parse_xml",
"'</a>' closes '<b>'" and "line 2".

### AnElementLeftOpenIsRefused

Each of these fails with `InvalidFile`: an element `b` left open ("line 3",
"the element 'b', opened on line 2, is not closed"), a comment left open ("line 2",
"a comment is not closed") and a tag without its end ("line 2", "has no '>'").

### AnUnknownEntityIsRefused

`&nbsp;` on line 3 fails with `InvalidFile`, "line 3" and "'&' opens none of". `&#65;` fails
with `InvalidFile` and "line 1".

### MarkupOutsideTheSubsetIsRefused

Each of these fails with `InvalidFile`: a CDATA section ("line 2", "CDATA"), a DOCTYPE
("line 2", "declaration") and a processing instruction inside the root ("line 3",
"processing instruction").

### AMalformedAttributeIsRefused

`k=v` without quotes fails with `InvalidFile`, "line 2", "attribute 'k'" and "not quoted".
`k` without a value fails with `InvalidFile`, "line 1" and "attribute 'k' has no '='".

### ADocumentNeedsExactlyOneRoot

Each of these fails with `InvalidFile`: two empty lines ("line 3", "holds no element"), bare
text ("line 1", "text stands where an element must open"), a second root ("line 2",
"content follows the root element 'a'") and the name `1a` ("line 1",
"a name must open with").

### NestingBeyondTheDepthIsRefused

`K_XML_MAX_DEPTH` + 1 elements opened one per line fail with `InvalidFile`, "line 65" and
"more than 64 elements are open at once".

## test_yolo_txt.cpp

Suites: `Unit_DatasetsYoloTxt`.

Tests `read_yolo_txt` and `yolo_label_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/yolo_txt.hpp`. The reader cases write
one-byte image files and label text files under the system's temporary directory and read
with a class count of 3.

### TheLabelPathSwapsTheLastImagesFolder

`yolo_label_path` maps `data/images/train/a.jpg` to `data/labels/train/a.txt`,
`images/x/images/a.png` to `images/x/labels/a.txt` and `images/a.b.jpeg` to `labels/a.b.txt`.

### TheLabelPathKeepsAFolderThatIsNotImages

It maps `data/train/a.jpg` to `data/train/a.txt`, `data/images` to `data/images.txt` and
`a.jpg` to `a.txt`.

### TheImageFolderIsListedWithItsLabels

The split "train" lists `images/train/a.png`, `b.jpg` and `c.jpg` in that order and leaves
`notes.md` out, with a class count of 3 and no class names. The first sample has no stored
pixels, no class index and two objects of the classes 2 and 0; its first box is (0.5, 0.25,
0.2, 0.1) and has no polygon. The second sample has one object of class 1 with a polygon of
three vertices, the second being (0.9, 0.1).

### AnImageWithoutALabelFileHasNoObject

An image without a label file and an image with an empty one both have no object.

### AnImageListIsReadInItsOwnOrder

A `train.txt` naming `./images/all/z.jpg` with a CRLF, an empty line and `images/all/a.jpg`
between blanks gives two samples in that order. The first has no object and the second one
object of class 1. `unlisted.jpg` is not read.

### AnImageListWinsOverAnImageFolder

With `images/train` holding two images and `train.txt` naming one, the list has the one sample
`images/train/b.jpg`.

### AListedImageThatIsAbsentNamesItsLine

A listed image that does not exist fails with `FileNotFound`, "train.txt", "line 3" and
"gone.jpg".

### ABadLabelFileNamesItsPathAndLine

A label file whose second line holds "abc" as a coordinate fails with `InvalidFile`,
"labels/train/b.txt", "line 2" and "abc".

### AClassOutsideTheClassCountIsRefused

A label of class 3 fails with `InvalidFile`, "labels/train/a.txt" and "line 1".

### AClassCountBelowOneIsRefused

A class count of 0 fails with `InvalidDescription` and "class count is 0".

### ASplitWithNeitherFormIsNotFound

A split with no list and no folder fails with `FileNotFound`, "train.txt" and "images/train".

### ASplitWithoutAnImageIsRefused

An image folder holding only `readme.txt` fails with `InvalidFile`, "'train'" and "no image".

## test_batch_inputs.cpp, test_classification_collator.cpp, test_detection_collator.cpp

The scaled inputs and the roles the classification and detection collators write, and the batches they refuse. Needs an engine.

Suites: `Unit_DatasetsCollators`.

Tests `write_batch_inputs`, `ClassificationCollator` and `DetectionCollator` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`. It
holds two gray samples of one row and two columns, the first of level 51 from `first.png` and
the second of level 255 from `crowded.png`, and allocates batches of two such images.

### InputsAreBytesDividedBy255

`write_batch_inputs` writes 0.2, 0.2, 1.0, 1.0, each within 1e-6.

### AnImageOfAnotherSizeIsRefused

With the second image 2 by 2, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs", "image 1" and "2 by 2".

### ATensorOfAnotherImageCountIsRefused

With one sample and a tensor for two, `write_batch_inputs` fails with `Incompatible`,
"write_batch_inputs" and "holds 2 images".

### ClassificationPublishesLabels

`ClassificationCollator::make(0)` fails with `InvalidDescription`, "ClassificationCollator" and
"0 classes". A collator of 7 classes reports the task `K_TASK_CLASSIFICATION` and one target:
the role `K_ROLE_LABELS`, an empty sample shape, `Int32` and 7 classes.

### ClassificationWritesTheClassOfEachSample

With the class indices 6 and 0, `collate` writes the labels 6 and 0, and the third input value
is 1.0.

### ClassificationNamesTheFileOfASampleWithoutAClass

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"crowded.png", "ClassificationCollator" and "no class index". With the class index 7 over 7
classes it fails with "crowded.png", "class index 7" and "[0, 7)".

### ABatchWithoutTheRoleIsRefused

Over a batch allocated without targets, `collate` fails with `Incompatible`,
"ClassificationCollator" and "no role 'labels'".

### DetectionPublishesThreeRoles

`DetectionCollator::make` fails with `InvalidDescription` for 0 classes ("0 classes") and for
0 objects ("0 objects"). A collator of 3 classes and 4 objects reports the task
`K_TASK_DETECTION` and three targets: `K_ROLE_BOXES` of shape [4, 4] and `Float32`,
`K_ROLE_CLASSES` of shape [4], `Int32` and 3 classes, and `K_ROLE_VALID` of shape [4] and
`Float32`.

### DetectionWritesCornersAndPadsWithZero

Every target slot is first filled with 9. With one object of class 2 and the box (0.25, 0.5,
1.5, 1.0) on the second image and two slots per image, `collate` writes that box, the class 2
and the valid flag 1 into the third slot and zero into every other slot.

### DetectionNamesTheFileOfAnImageWithTooManyObjects

Three objects on the second image with two slots fail with `InvalidDescription`,
"crowded.png", "DetectionCollator", "3 objects" and "at most 2".

### DetectionRefusesAClassOutsideTheSet

An object of class 3 on the first image, over 3 classes, fails with `InvalidDescription`,
"first.png", "object 0", "class 3" and "[0, 3)".

## test_pipeline_table.cpp, test_fit_pipeline.cpp, test_augment_pipeline.cpp

The pipeline table, the fit the same in every epoch, a training sample identical alone, in a batch and reversed, the mosaic epochs, masks, and the refusals.

Suites: `Unit_DatasetsPipelineTable`, `Unit_DatasetsFitPipeline`,
`Unit_DatasetsAugmentPipeline`.

Tests `pipeline_table` and `find_pipeline`, `FitPipeline` and `AugmentPipeline` in
`modules/vision/datasets/include/SushiAI/vision/datasets/`. The pipeline cases read four raw
RGB images of the sizes 20 by 16, 16 by 20, 16 by 16 and 10 by 20, written under the system's
temporary directory; sample i holds one object of class i. The training pipelines have a size
of 16 and the seed 20261011.

### ItListsNoneThenDefault

`pipeline_table` has two rows named "none" and "default". `find_pipeline` returns each row for
its name and null for "mixup".

### ARowBuildsThePipelineOfItsName

For a request of 16 by 16 and three channels, each row's `make` gives a pipeline whose `name`
is the row's name.

### DefaultRefusesAnInputItDoesNotMake

The "default" row fails with `InvalidDescription` and "'default'" for a size of 4 by 6
("square", "4 by 6") and for one channel ("3 channels", "has 1").

### FitsEachSampleTheSameInEveryEpoch

A `FitPipeline` that letterboxes to 8 by 8 is named "none". For the samples 3 and 0 it gives
images of 8 by 8 and three channels with one object each, of the classes 3 and 0. Epoch 7
gives the same image, mask, file, classes and boxes as epoch 0.

### RefusesWhatItCannotFit

`FitPipeline::make` with a size of 0 by 8 fails with `InvalidDescription`, "FitPipeline" and
"0 by 8". `produce` with two sample indices and one fitted slot fails with
`InvalidDescription`, "2 samples" and "1 fitted".

### MakesSquareSamplesOfThreeChannels

An `AugmentPipeline` with 1000 mosaic epochs is named "default". In epoch 0 each of the four
samples is 16 by 16 with three channels and an empty mask, and every box lies within
[0, 16] on both axes.

### ASampleDependsOnItsIndexAndItsEpochAlone

In epoch 2, each sample comes out the same when the four are produced twice, when they are
produced in reverse order and when sample 2 is produced alone. In epoch 3 the image of at
least one sample differs from its image in epoch 2.

### PastTheMosaicEpochsASampleHoldsItsOwnObjects

With 0 mosaic epochs, in the epochs 0, 1 and 2 every object of sample i has the class i.

### AMaskTakesTheSizeOfItsSample

Each sample is given a mask of its image's size that holds its index. In epoch 1 every
produced mask is 16 by 16 with one channel, and each byte is below 4 or is `K_MASK_VOID`.

### AnImageThatCannotBeReadIsTheFailure

With the file of sample 1 replaced by a path that does not exist, `produce` fails with
`FileNotFound` and "gone.saim".

### RefusesWhatItCannotMake

`AugmentPipeline::make` fails with `InvalidDescription` for a size of 0 ("AugmentPipeline",
"a training size of 0") and for -1 mosaic epochs ("-1 epochs of mosaic"). `produce` fails
with `InvalidDescription` for the epoch -1 ("epoch -1") and for an empty list
("a list of 0 samples").

## test_binary_collator.cpp, test_multi_label_collator.cpp, test_regression_collator.cpp, test_segmentation_collator.cpp

The roles the binary, multi-label, regression and segmentation collators write, and the samples they refuse. Needs an engine.

Suites: `Unit_DatasetsTaskCollators`.

Tests `BinaryCollator`, `MultiLabelCollator`, `RegressionCollator` and `SegmentationCollator`
in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`.
It holds two gray samples of one row and two columns, from `first.png` and `second.png`, and
allocates batches of two samples.

### BinaryPublishesLabels

`BinaryCollator::make` fails with `InvalidDescription` for the positive class 2
("BinaryCollator", "the positive class is 2") and for -1 ("the positive class is -1"). A
collator of the positive class 1 reports the task `K_TASK_BINARY` and one target: the role
`K_ROLE_LABELS`, an empty sample shape, `Float32` and 0 classes.

### BinaryWritesOneForThePositiveClass

With the class indices 1 and 0, a collator of the positive class 1 writes the labels 1, 0 and
a collator of the positive class 0 writes 0, 1.

### BinaryNamesTheFileOfASampleItCannotLabel

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"second.png", "BinaryCollator" and "no class index". With the class index 2 it fails with
"second.png", "class index 2" and "[0, 2)".

### MultiLabelWritesTheSetOfClassesPresent

`MultiLabelCollator::make(0)` fails with `InvalidDescription`, "MultiLabelCollator" and
"0 classes". A collator of 4 classes reports the task `K_TASK_MULTI_LABEL` and one target of
the role `K_ROLE_LABELS`, the sample shape [4] and `Float32`. The first sample has objects of
the classes 2, 0 and 2 and the class index 3; the second has neither. Over a target first
filled with 9, `collate` writes 1, 0, 1, 1 for the first sample and four zeros for the second.

### MultiLabelRefusesAClassOutsideTheSet

An object of class 4 on the second sample, over 4 classes, fails with `InvalidDescription`,
"second.png", "MultiLabelCollator", "the class 4" and "[0, 4)".

### RegressionCopiesFeaturesAndTargets

`RegressionCollator::make` fails with `InvalidDescription` for 0 features ("0 features") and
for 0 targets ("0 targets"), each with "RegressionCollator". A collator of 3 features and 2
targets reports the task `K_TASK_REGRESSION` and one target of the role `K_ROLE_VALUES`, the
sample shape [2] and `Float32`. For samples with the features 1, 2, 3 and 4, 5, 6 and the
targets 10, 20 and 40, 50, `collate` writes the inputs 1 to 6 and the values 10, 20, 40, 50.

### RegressionRefusesWhatHoldsNoRow

Over samples that hold images and no row, `collate` fails with `InvalidDescription`,
"first.png", "RegressionCollator" and "0 features and 0 targets". Over inputs of the sample
shape [2] it fails with `Incompatible`, "RegressionCollator" and "float32 of [2, 3]".

### SegmentationPublishesTheMask

`SegmentationCollator::make` fails with `InvalidDescription` for 0 classes
("SegmentationCollator", "0 classes"), for 256 classes ("256 classes", "1 to 255"), for the
ignore index 2 over 3 classes ("the ignore index 2 is one of the 3 classes") and for a size
of 0 by 2 ("0 by 2 pixels"). A collator of 3 classes reports the task `K_TASK_SEGMENTATION`
and one target of the role `K_ROLE_MASK`, the sample shape [1, 2], `Int32` and 3 classes.

### SegmentationWritesClassesAndTheIgnoreIndex

For the masks 2, `K_MASK_VOID` and 0, 1, `collate` writes 2, the ignore index, 0, 1, with the
ignore index `K_DEFAULT_IGNORE_INDEX` and with -100.

### SegmentationRefusesAByteThatIsNoClass

A mask byte of 3 over 3 classes fails with `InvalidDescription`, "second.png",
"SegmentationCollator", "holds 3 at row 0, column 1" and "below 3 and 255 names none".

### SegmentationRefusesASampleWithoutAMask

With no mask on the second sample, `collate` fails with `InvalidDescription`, "second.png",
"SegmentationCollator" and "the sample's mask is 0 by 0".

### SegmentationPaintsThePolygonsOfASample

With `MaskOrigin::Polygons` and the ignore index -1, the first sample has a polygon of class 2
over its second pixel and the second sample one of class 0 over its first pixel. `collate`
writes -1, 2, 0, -1. With `background` 1 it writes 1, 2, 0, 1.

### SegmentationRefusesAnObjectWithoutAPolygon

With `MaskOrigin::Polygons`, an object with a box and no polygon on the second sample fails
with `InvalidDescription`, "second.png", "paint_polygon_mask" and "no polygon".

### SegmentationRefusesABackgroundItCannotWrite

`SegmentationCollator::make` fails with `InvalidDescription` for `background` 1 without
`MaskOrigin::Polygons` ("a background of 1", "masks come from files") and for `background` 3
over 3 classes with it ("the background 3 is none of the 3 classes").
