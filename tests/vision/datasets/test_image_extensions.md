# tests/vision/datasets/test_image_extensions.cpp


Suites: `Small_DatasetsImageExtensions`, `Small_DatasetsImageExtensionsUndecoded`,
`Small_DatasetsImageExtensionsTaken` and `Small_DatasetsImageExtensionsNoExtension`.

Tests `has_image_extension` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp`. Every case passes a
path that names no file.

## ImageExtensionsIgnoreCase

One row each: "a.jpg", "a.JPEG", "dir/a.PNG" and "a.saim" are taken, whatever the case of the
letters and the folders before the name.

## ANameWithoutAnImageExtensionIsRefused

One row each: "a.txt", "a.jpg.bak" and "jpg" are refused.

## FormatsTheCodecsDoNotDecodeAreRefused

One row each: ".gif", ".bmp", ".tiff" and ".webp" are refused, since vision/codecs decodes none of
them.

## TheFileIsNotOpened

A path below a directory that does not exist, ending in ".png", is taken.
