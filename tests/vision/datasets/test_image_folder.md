# tests/vision/datasets/test_image_folder.cpp


Suites: `Unit_DatasetsImageFolder`.

Tests `read_image_folder` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_folder.hpp` and
`has_image_extension` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp`. The folder
cases write one-byte files under the system's temporary directory.

## ImageExtensionsIgnoreCase

`has_image_extension` is true for `a.jpg`, `a.JPEG`, `dir/a.PNG` and `a.saim`, and false for
`a.txt`, `a.jpg.bak` and `jpg`.

## ClassesAreTheSortedFolderNames

The folders zebra, ant and moth give the class names ant, moth, zebra and a class count of 3.
Sample i of the three has the class i, no stored pixels and no object.

## AClassListsItsImageFilesSortedAtAnyDepth

The list holds four files in the order `cat/a.JPG`, `cat/b.png`, `cat/more/c.saim`,
`dog/a.png`, and leaves `cat/notes.txt` out. The third sample has the class 0 and the fourth
the class 1.

## ASplitNamesAFolderOfTheRoot

The split "val" gives the classes no and yes and two samples. The file `val/stray.png` and the
folder `train` add nothing.

## AFolderWithoutAnImageIsStillAClass

With an empty folder `a` and one image in `b`, the class count is 2 and the one sample has the
class 1.

## AnAbsentSplitIsNotFound

The split "train" in an empty root fails with `FileNotFound` and "train".

## ASplitWithoutAClassOrAnImageIsRefused

A split holding only a file fails with `InvalidFile`, "flat" and "no class folder". A split
whose one class folder holds only a text file fails with `InvalidFile`, "text" and
"no image file".
