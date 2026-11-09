# tests/vision/datasets/test_yolo_txt.cpp


Suites: `Small_DatasetsYoloTxt` and `Medium_DatasetsYoloTxt`.

Tests `read_yolo_txt` and `yolo_label_path` in
`modules/vision/datasets/include/SushiAI/vision/datasets/yolo_txt.hpp`. The `Small_`
reader cases read one-byte image files and label text files served by a `MemorySource` under the
folder `data`; the three `Medium_` cases write them under the system's temporary directory and
read them through the file-system overload. Every case reads with a class count of 3.

## TheLabelPathSwapsTheLastImagesFolder

`yolo_label_path` maps `data/images/train/a.jpg` to `data/labels/train/a.txt`,
`images/x/images/a.png` to `images/x/labels/a.txt` and `images/a.b.jpeg` to `labels/a.b.txt`.

## TheLabelPathKeepsAFolderThatIsNotImages

It maps `data/train/a.jpg` to `data/train/a.txt`, `data/images` to `data/images.txt` and
`a.jpg` to `a.txt`.

## TheImageFolderIsListedSorted

The split "train" lists `images/train/a.png`, `b.jpg` and `c.jpg` in that order and leaves
`notes.md` out.

## TheListHoldsAClassCountAndNoClassName

For the same folder the list has a class count of 3 and no class names, and the first sample
has no stored pixels and no class index.

## ALabelFileOfBoxesGivesOneObjectPerLine

The first sample has two objects of the classes 2 and 0; its first box is (0.5, 0.25, 0.2, 0.1)
within the typed tolerance of a parsed coordinate and has no polygon.

## ALabelLineOfPointsGivesAPolygon

The second sample has one object of class 1 with a polygon of three vertices, the second being
(0.9, 0.1).

## AnImageWithoutALabelFileHasNoObject

An image without a label file and an image with an empty one both have no object.

## AnImageListIsReadInItsOwnOrder

A `train.txt` naming `./images/all/z.jpg` with a CRLF, an empty line and `images/all/a.jpg`
between blanks gives two samples in that order. The first has no object and the second one
object of class 1. `unlisted.jpg` is not read.

## AnImageListWinsOverAnImageFolder

With `images/train` holding two images and `train.txt` naming one, the list has the one sample
`images/train/b.jpg`.

## AListedImageThatIsAbsentNamesItsLine

A listed image that does not exist fails with `FileNotFound`, "train.txt", "line 3" and
"gone.jpg".

## ABadLabelFileNamesItsPathAndLine

A label file whose second line holds "abc" as a coordinate fails with `InvalidFile`,
"labels/train/b.txt", "line 2" and "abc".

## AClassOutsideTheClassCountIsRefused

A label of class 3 fails with `InvalidFile`, "labels/train/a.txt" and "line 1".

## AClassCountBelowOneIsRefused

A class count of 0 fails with `InvalidDescription` and "class count is 0".

## ASplitWithNeitherFormIsNotFound

A split with no list and no folder fails with `FileNotFound`, "train.txt" and "images/train".

## ASplitWithoutAnImageIsRefused

An image folder holding only `readme.txt` fails with `InvalidFile`, "'train'" and "no image".

## ImagesAndLabelsOnDiskArePaired

`Medium_DatasetsYoloTxt`. Two images and the label file of `b.jpg` written under the temporary
directory give the samples `images/train/a.png`, with no object, and `images/train/b.jpg`, with
one object of class 2, through the file-system overload.

## AListedImageAbsentFromTheDiskNamesItsLine

`Medium_DatasetsYoloTxt`. The list on disk names `images/gone.jpg` on line 3, and the read fails
with `FileNotFound`, "train.txt", "line 3" and "gone.jpg".

## ASplitWithNeitherFormOnDiskIsNotFound

`Medium_DatasetsYoloTxt`. A temporary directory with neither `train.txt` nor `images/train` fails
with `FileNotFound`, "train.txt" and "images/train".
