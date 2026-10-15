# tests/vision/datasets/test_mask_png.cpp


Suites: `Unit_DatasetsMaskPng`.

Tests `read_mask_png` in `modules/vision/datasets/include/SushiAI/vision/datasets/mask_png.hpp`
over one-byte files the cases write under the system's temporary directory. Unless a case says
otherwise the options name four classes and the split "train", so the images are under
`images/train` and the masks under `masks/train`.

## EachImageRefersToTheMaskOfItsStem

The list has a class count of 4, no class names, two samples and four files. The first sample
is `images/train/a/c.jpeg` with the mask `masks/train/a/c.png`, and has no class index and no
object. The second is `images/train/b.jpg` with `masks/train/b.png`. The files `notes.md` and
`index.txt` are not listed.

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

## TheOptionsAndTheFoldersAreChecked

A class count of 0 fails with `InvalidDescription`, "read_mask_png" and
"the class count is 0". An absent image folder fails with `FileNotFound`, "images/train" and
"no such directory", and an absent mask folder with `FileNotFound`, "masks/train" and
"no such directory". Two empty folders fail with `InvalidFile`, "images/train" and
"holds no image".

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
