# tests/vision/datasets/test_voc_xml.cpp


Suites: `Small_DatasetsVocXml` and `Medium_DatasetsVocXml`.

Tests `read_voc_xml` in `modules/vision/datasets/include/SushiAI/vision/datasets/voc_xml.hpp`.
The `Small_` cases read files served by a `MemorySource` under the folder `data`; the two
`Medium_` cases write them under the system's temporary directory and read them through the
file-system overload. Unless a case says otherwise the split is "train", listed in
`ImageSets/Main/train.txt`. The sample annotation names `first.jpg`, a size of 200 by 100 and
some objects, each with a name, a difficult flag and the box edges 11, 21, 60, 40.

## TheListedAnnotationsAreReadInTheOrderOfTheList

The list file names second, an empty line, then first followed by " 1", with a CRLF on its
first line. The samples come out as `JPEGImages/second.jpg` with no object and
`JPEGImages/first.jpg`. `Annotations/unlisted.xml` is not read.

## TheClassesAreSortedByName

For the same files the class names come out as "cat & co", "dog". The second sample has no
class index and two objects of the classes 1 and 0.

## ABoxBecomesACentreAndASizeInFractions

For the same files the first box of the second sample has the centre (35/200, 30/100) and the
size (50/200, 20/100), each within the typed tolerance of a quotient in float.

## AnEmptySplitReadsEveryAnnotation

With an empty split the annotations `a.xml` and `b.xml` are read in that order, giving
`JPEGImages/a.jpg` and `JPEGImages/first.jpg`. `Annotations/readme.txt` is not read.

## ADifficultObjectIsDroppedByDefault

Over a difficult dog and a cat, the default options give the class names cat and one object.

## ADifficultObjectIsKeptWhenAsked

For the same annotation, `VOCDifficult::Kept` gives cat, dog and two objects.

## AStatedClassListGivesTheIndices

With `class_names` person, dog, cat the class count is 3 and a dog has the class 1.

## AStatedClassListThatLacksAnObjectNameIsRefused

With person, cat the read fails with `InvalidFile`, "Annotations/first.xml", "line 4",
"read_voc_xml" and "the class 'dog'".

## AStatedClassListThatNamesAClassTwiceIsRefused

With dog, cat, dog the read fails with `InvalidDescription`, "read_voc_xml", "names 'dog' twice"
and "at 0 and at 2".

## AMalformedAnnotationNamesItsFileAndLine

An annotation that closes `annotation` inside an open `size` fails with `InvalidFile`,
"Annotations/first.xml", "line 3", "parse_xml" and "closes '<size>'".

## AnAnnotationWithoutASizeIsRefused

An annotation without `size` fails with `InvalidFile`, "first.xml", "line 1", "read_voc_xml" and
"'annotation' has no 'size'".

## ARootThatIsNoAnnotationIsRefused

A root named `image` fails with `InvalidFile`, "first.xml", "line 1" and
"the root element is 'image'".

## ASizeThatIsNoNumberIsRefused

A width of "wide" fails with `InvalidFile`, "first.xml", "line 3" and "'width' holds 'wide'".

## AnObjectWithoutABoxIsRefused

An object without `bndbox` fails with `InvalidFile`, "first.xml", "line 4" and
"'object' has no 'bndbox'".

## ABoxThatEndsBeforeItStartsIsRefused

A box whose maximum lies before its minimum fails with `InvalidFile`, "first.xml", "line 7" and
"the box ends at (5, 2), before its start (9, 1)".

## ASplitWithoutAListIsRefused

Without the list file the read fails with `FileNotFound`, "read_voc_xml", "the split 'train'"
and "ImageSets/Main/train.txt".

## AStemWithoutAnAnnotationIsRefused

A stem on line 3 without an annotation fails with `FileNotFound`, "ImageSets/Main/train.txt",
"line 3" and "Annotations/missing.xml".

## AListWithoutAStemIsRefused

A list of one empty line fails with `InvalidFile` and "holds no annotation".

## AnnotationsOnDiskAreReadInTheOrderOfTheList

`Medium_DatasetsVocXml`. The list and two annotations written under the temporary directory give
the samples `JPEGImages/second.jpg` and `JPEGImages/first.jpg` and the class `dog` through the
file-system overload.

## ASplitWithoutAListOnDiskIsNotFound

`Medium_DatasetsVocXml`. A temporary directory without the list fails with `FileNotFound`,
"read_voc_xml", "the split 'train'" and "ImageSets/Main/train.txt".
