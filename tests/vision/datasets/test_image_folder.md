# tests/vision/datasets/test_image_folder.cpp


Suites: `Small_DatasetsImageFolder` and `Medium_DatasetsImageFolder`.

Tests `read_image_folder` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_folder.hpp`. The `Small_` cases list
one-byte files from a `MemorySource` under the folder `data`; the `Medium_` cases write them under
the system's temporary directory. The extensions a file may have are tested in
`test_image_extensions.cpp`.

## ClassesAreTheSortedFolderNames

The folders `zebra`, `ant` and `moth` give the class names ant, moth, zebra, a class count of 3 and
the sample classes 0, 1, 2.

## ASampleNamesNoStoredPixelsAndNoObject

A listed sample has no `stored` pixels and no object, since the image is decoded when read.

## AClassListsItsImageFilesSortedAtAnyDepth

The folder `cat` holds `b.png`, `a.JPG`, `notes.txt` and `c.saim` in a folder `more`; `dog` holds `a.png`. The
four samples are `cat/a.JPG`, `cat/b.png`, `cat/more/c.saim` and `dog/a.png`, of the classes 0, 0, 0
and 1; the text file is left out.

## ASplitNamesAFolderOfTheRoot

With the split "val", the class names are no and yes and there are two samples, so the folder
`train` is not read.

## AFileDirectlyInTheSplitIsNoClass

A file `val/stray.png` beside the class folder `no` adds no class and no sample.

## AFolderWithoutAnImageIsStillAClass

An empty folder `a` and a folder `b` with one image give a class count of 2 and a sample of class 1.

## ASplitWithoutAClassFolderIsRefused

A split holding only a file fails with `InvalidFile`, "flat" and "no class folder".

## ASplitWithoutAnImageIsRefused

A split whose class folder holds only a text file fails with `InvalidFile`, "text" and "no image
file".

## AnAbsentSplitIsNotFound

A split that is not a directory fails with `FileNotFound` and "train". It reaches the file system,
so it is `Medium_`.

## ClassFoldersOnDiskBecomeSortedClasses

The file-system form of `read_image_folder` lists written folders `dog` (one image, one below
`deeper`) and `cat`: the classes cat and dog, and the sample classes 0, 1, 1.
