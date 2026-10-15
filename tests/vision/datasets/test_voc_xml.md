# tests/vision/datasets/test_voc_xml.cpp


Suites: `Unit_DatasetsVocXml`.

Tests `read_voc_xml` in `modules/vision/datasets/include/SushiAI/vision/datasets/voc_xml.hpp`
over files the cases write under the system's temporary directory. Unless a case says
otherwise the split is "train", listed in `ImageSets/Main/train.txt`. The sample annotation
names `first.jpg`, a size of 200 by 100 and some objects, each with a name, a difficult flag
and the box edges 11, 21, 60, 40.

## TheListedAnnotationsAreReadWithSortedClasses

The list file names second, an empty line, then first followed by " 1", with a CRLF on its
first line. The class names come out as "cat & co", "dog", and the samples as
`JPEGImages/second.jpg` with no object and `JPEGImages/first.jpg`. That sample has no class
index and two objects of the classes 1 and 0; its first box has the centre (35/200, 30/100)
and the size (50/200, 20/100). `Annotations/unlisted.xml` is not read.

## AnEmptySplitReadsEveryAnnotation

With an empty split the annotations `a.xml` and `b.xml` are read in that order, giving
`JPEGImages/a.jpg` and `JPEGImages/first.jpg`. `Annotations/readme.txt` is not read.

## ADifficultObjectIsDroppedUnlessKept

Over a difficult dog and a cat, the default options give the class names cat and one object.
`VOCDifficult::Kept` gives cat, dog and two objects.

## AStatedClassListGivesTheIndices

With `class_names` person, dog, cat the class count is 3 and a dog has the class 1. With
person, cat the read fails with `InvalidFile`, "Annotations/first.xml", "line 4",
"read_voc_xml" and "the class 'dog'". With dog, cat, dog it fails with `InvalidDescription`,
"read_voc_xml", "names 'dog' twice" and "at 0 and at 2".

## AMalformedAnnotationNamesItsFileAndLine

An annotation that closes `annotation` inside an open `size` fails with `InvalidFile`,
"Annotations/first.xml", "line 3", "parse_xml" and "closes '<size>'".

## AMissingElementOrNumberIsRefused

Each of these fails with `InvalidFile` and "first.xml": an annotation without `size`
("line 1", "read_voc_xml", "'annotation' has no 'size'"), a root named `image` ("line 1",
"the root element is 'image'"), a width of "wide" ("line 3", "'width' holds 'wide'"), an
object without `bndbox` ("line 4", "'object' has no 'bndbox'") and a box whose maximum lies
before its minimum ("line 7", "the box ends at (5, 2), before its start (9, 1)").

## TheListAndItsStemsMustExist

Without the list file the read fails with `FileNotFound`, "read_voc_xml", "the split 'train'"
and "ImageSets/Main/train.txt". A stem on line 3 without an annotation fails with
`FileNotFound`, "ImageSets/Main/train.txt", "line 3" and "Annotations/missing.xml". A list of
one empty line fails with `InvalidFile` and "holds no annotation".
