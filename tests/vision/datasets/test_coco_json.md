# tests/vision/datasets/test_coco_json.cpp


Suites: `Small_DatasetsCocoJson`.

Tests `read_coco_json` in
`modules/vision/datasets/include/SushiAI/vision/datasets/coco_json.hpp` over files the cases
write under the system's temporary directory. Unless a case says otherwise the annotation file
is `annotations/instances_train.json` beside a folder `train`. It lists `b.jpg` with id 9 and
`sub/a.jpg` with id 4, both 200 by 100 pixels, and the categories brush (90), person (1) and
cat (17).

## BoxesAndDenseClassesAreRead

The class names come out as person, cat, brush, and the samples as `train/b.jpg` with no object
and `train/sub/a.jpg`. Of that image's four annotations the crowd one and the one of width 0
are left out; the other two have the classes 2 and 0 and no polygon. The box [10, 20, 50, 20]
reads as the centre (35/200, 30/100) and the size (50/200, 20/100), and the sample has no class
index.

## TheFileAndTheFolderMayBeNamed

With `annotation_file` "labels.json" and `image_folder` "pictures", the first sample's file is
`pictures/b.jpg`.

## PolygonsAreKeptWhenAskedFor

By default an annotation with two polygon parts keeps no polygon. With
`COCOSegmentation::Polygons` the image has one object, whose polygon is the part of four
vertices, and its third vertex is (100/200, 50/100). The crowd annotation with a run-length
mask is left out.

## ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor

A file whose second annotation holds a run-length mask reads as two objects by default. With
`COCOSegmentation::Polygons` it fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "annotation 1 holds a run-length mask".

## AMalformedPolygonIsRefused

A polygon part of seven numbers fails with `InvalidFile`, "annotation 0",
"'segmentation' part" and "at least 3 pairs".

## AnUnknownImageOrCategoryNamesTheAnnotation

An annotation of image 5 fails with `InvalidFile`, "instances_train.json", "read_coco_json"
and "annotation 1 has the 'image_id' 5, which 'images' lacks". A crowd annotation of category
2 fails with "annotation 0 has the 'category_id' 2, which 'categories' lacks".

## AMalformedBoxNamesTheAnnotation

A `bbox` of three numbers, and one whose fourth entry is a string, each fail with
`InvalidFile` and "annotation 0 has no 'bbox' of 4 numbers". An `image_id` given as a string
fails with "annotation 0 has no whole number 'image_id'".

## TheImagesAndTheCategoriesAreChecked

Each of these fails with `InvalidFile`: two categories of id 3
("category 0 and category 2 share the id 3"), an image of width 0
("image 1 is 2 by 0 pixels") and an image without `file_name`
("image 0 has no string 'file_name'").

## TheFileMustExistAndHoldItsThreeLists

Without the folder `train` the read fails with `FileNotFound`, "train", "read_coco_json" and
"no such image folder". Without the annotation file it fails with `FileNotFound`,
"annotations/instances_train.json" and "no such file". A file cut after `"images": [` fails
with `InvalidFile` and "not valid JSON", and a file without `categories` with
"the file has no list 'categories'".

## TheListsMayComeInAnyOrder

One file states `images`, `categories` and `annotations` in that order. A second states an
`info` object, then `annotations`, `categories` and `images`. With `COCOSegmentation::Polygons`
both give the same class names, files, classes, box centre x, box height and polygon vertices.
The second sample has the class 2 and a polygon of four vertices, and the first has no polygon.

## AListStatedTwiceIsRefused

A file that states `images` a second time fails with `InvalidFile`, "instances_train.json",
"read_coco_json" and "states 'images' twice".
