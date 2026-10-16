# tests/vision/datasets/test_image_extensions.cpp


Suites: `Small_DatasetsImageExtensions`.

Tests `has_image_extension` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_extensions.hpp`. The four
extensions it takes, the case of their letters and the refusal of ".txt" and "a.jpg.bak" are
tested in `test_image_folder.cpp`, under `Small_DatasetsImageFolder`. This file pins the two
promises left: other image formats are refused, and no file is opened. Every case passes a path
that names no file.

## FormatsTheCodecsDoNotDecodeAreRefused

".gif", ".bmp", ".tiff" and ".webp" are refused, since vision/codecs decodes none of them.

## TheFileIsNotOpened

A path below a directory that does not exist, ending in ".png", is taken.
