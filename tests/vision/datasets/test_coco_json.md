# tests/vision/datasets/test_coco_json.cpp


Suites: `Small_DatasetsCocoJson`, `Small_DatasetsCocoJsonUnknown`, `Small_DatasetsCocoJsonBox`,
`Small_DatasetsCocoJsonLists` and `Medium_DatasetsCocoJson`.

Tests `read_coco_json` in
`modules/vision/datasets/include/SushiAI/vision/datasets/coco_json.hpp`. The `Small_` cases read
from a `MemorySource` under the folder `set`; the `Medium_` case writes a set under the system's
temporary directory. Unless a case says otherwise the annotation file is
`annotations/instances_train.json` beside a folder `train`. It lists `b.jpg` with id 9 and
`sub/a.jpg` with id 4, both 200 by 100 pixels, and the categories brush (90), person (1) and
cat (17). A fraction compares within `K_FRACTION_TOLERANCE`, 32 times the float unit roundoff.

## ABoxBecomesAFractionOfItsImage

The box [10, 20, 50, 20] of image 4 has the centre (35 / 200, 30 / 100) and the size
(50 / 200, 20 / 100).

## CategoryIdsBecomeDenseClassesInIdOrder

The class names are person, cat, brush and the class count is 3. The sample has no class index, and
its two objects, of the categories 90 and 1, have the classes 2 and 0.

## ACrowdAnnotationAndABoxOfNoWidthAreLeftOut

Of three annotations of image 4, the crowd one and the one of width 0 are dropped: the sample holds
one object of class 2 with no outline, and the sample of image 9 holds none.

## ImageNamesResolveInTheImageFolder

The samples are `train/b.jpg` and `train/sub/a.jpg`, in the order of the image list.

## TheFileAndTheFolderMayBeNamed

With `labels.json` and the folder `pictures` named, the first sample is `pictures/b.jpg`.

## APolygonIsKeptOnlyWhenAskedFor

By default the object holds no outline; with `COCOSegmentation::Polygons` it holds one of four
vertices.

## TheLargestPartOfAPolygonIsTaken

Of a three-vertex and a four-vertex part, the outline is the four-vertex one, and its third vertex
is (100 / 200, 50 / 100).

## ARunLengthMaskIsAcceptedByDefault

Two annotations, the second with a run-length mask, both read when polygons are not asked for.

## ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor

The same file with `COCOSegmentation::Polygons` fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "annotation 1 holds a run-length mask".

## AMalformedPolygonIsRefused

A part of seven numbers fails with `InvalidFile`, "annotation 0", "'segmentation' part" and "at
least 3 pairs".

## AnUnknownImageOrCategoryNamesTheAnnotation

One row each: an annotation of the image id 5 fails with "annotation 1 has the 'image_id' 5, which
'images' lacks"; one of the category id 2 fails with "annotation 0 has the 'category_id' 2, which
'categories' lacks". Each fails with `InvalidFile`, "instances_train.json" and "read_coco_json".

## AMalformedBoxNamesTheAnnotation

One row each: a box of three numbers and a box with a string fail with `InvalidFile` and
"annotation 0 has no 'bbox' of 4 numbers".

## AnImageIdThatIsNotAWholeNumberNamesTheAnnotation

An image id written as the string "9" fails with `InvalidFile` and "annotation 0 has no whole number
'image_id'".

## TheImagesAndTheCategoriesAreChecked

One row each, failing with `InvalidFile`: two categories of the id 3 ("category 0 and category 2
share the id 3"); an image of width 0 ("image 1 is 2 by 0 pixels"); an image without a file name
("image 0 has no string 'file_name'").

## AnAbsentImageFolderIsNotFound

Without a folder `train`, the read fails with `FileNotFound`, "train", "read_coco_json" and "no such
image folder".

## AnAbsentAnnotationFileIsNotFound

With the folder and without the file, the read fails with `FileNotFound`,
"annotations/instances_train.json", "read_coco_json" and "no such file".

## TextThatIsNoJsonIsRefused

The text `{"images": [` fails with `InvalidFile`, "instances_train.json", "read_coco_json" and "not
valid JSON".

## ADocumentWithoutAListIsRefused

A document with `images` and `annotations` and no `categories` fails with `InvalidFile` and "the
file has no list 'categories'".

## TheListsMayComeInAnyOrder

A document that lists the annotations, then the categories, then the images, after an `info` object,
gives the same class names and the same description of each sample (file, class, box and outline,
compared exactly) as one that lists images, categories, annotations.

## AnAnnotationListedBeforeItsImageStillReachesIt

With the categories and the images before the annotation, image 4 (the second sample) holds one
object of class 2 with an outline of four vertices, and the first sample holds none.

## AListStatedTwiceIsRefused

A file with `images` twice fails with `InvalidFile`, "instances_train.json", "read_coco_json" and
"states 'images' twice".

## ASetOnDiskIsRead

The file-system form of `read_coco_json` reads a written set: the second sample is
`train/sub/a.jpg` and holds one object.
