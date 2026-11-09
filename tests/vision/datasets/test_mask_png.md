# tests/vision/datasets/test_mask_png.cpp


Suites: `Small_DatasetsMaskPng` and `Medium_DatasetsMaskPng`.

Tests `read_mask_png` in `modules/vision/datasets/include/SushiAI/vision/datasets/mask_png.hpp`.
The `Small_` cases read one-byte files served by a `MemorySource` under the folder `data`; the
two `Medium_` cases write them under the system's temporary directory and read them through the
file-system overload. Unless a case says otherwise the options name four classes and the split
"train", so the images are under `images/train` and the masks under `masks/train`.

## EachImageRefersToTheMaskOfItsStem

Two images and two masks below the images and masks folders give two samples and four files. The
first sample is `images/train/a/c.jpeg` with the mask `masks/train/a/c.png`, and the second is
`images/train/b.jpg` with `masks/train/b.png`. The files `notes.md` and `index.txt` are not
listed.

## TheListHoldsAClassCountAndNoLabel

The list has a class count of 4, no class names, and a sample with no class index and no object.

## TheFoldersMayBeNamedAndTheSplitLeftOut

With an empty split, `image_folder` "leftImg8bit" and `mask_folder` "gtFine", the one sample
is `leftImg8bit/a.png` with the mask `gtFine/a.png`.

## AnImageWithoutAMaskIsRefused

An image `b.jpg` without a mask fails with `FileNotFound`, "images/train/b.jpg",
"read_mask_png", "has no mask" and "masks/train/b.png".

## AMaskWithoutAnImageIsRefused

A mask `z.png` without an image fails with `InvalidFile`, "masks/train/z.png", "read_mask_png"
and "the mask has no image".

## TwoImagesOfOneStemAreRefused

The images `a.jpg` and `a.png` with one mask `a.png` fail with `InvalidFile`,
"images/train/a.png", "shares the mask" and "masks/train/a.png".

## AClassCountBelowOneIsRefused

A class count of 0 fails with `InvalidDescription`, "read_mask_png" and
"the class count is 0".

## AnAbsentImageFolderIsReported

A source without `images/train` fails with `FileNotFound`, "images/train" and
"no such directory".

## AnAbsentMaskFolderIsReported

A source with `images/train` and without `masks/train` fails with `FileNotFound`,
"masks/train" and "no such directory".

## AnImageFolderWithoutAnImageIsRefused

Two empty folders fail with `InvalidFile`, "images/train" and "holds no image".

## TheSuffixesPairNamesThatDiffer

With `image_suffix` "_leftImg8bit" and `mask_suffix` "_gtFine_labelIds" under the folders
`leftImg8bit` and `gtFine`, the two images pair with their `_gtFine_labelIds.png` masks, and
the list has two samples and four files. The mask `aachen_000000_gtFine_color.png` is not
listed.

## AnImageWithoutTheImageSuffixIsRefused

With `image_suffix` "_left", the image `b.png` fails with `InvalidFile`, "images/train/b.png",
"read_mask_png" and "does not end in '_left'".

## AMaskOfTheSuffixWithoutAnImageIsRefused

With `mask_suffix` "_ids", the mask `b_ids.png` without an image fails with `InvalidFile`,
"masks/train/b_ids.png" and "the mask has no image".

## ImagesAndMasksOnDiskArePairedByTheirStems

`Medium_DatasetsMaskPng`. The files of the first case, written under the temporary directory,
give the same two samples through the file-system overload.

## AnImageFolderAbsentFromTheDiskIsNotFound

`Medium_DatasetsMaskPng`. A temporary directory without `images/train` fails with `FileNotFound`,
"images/train" and "no such directory".
