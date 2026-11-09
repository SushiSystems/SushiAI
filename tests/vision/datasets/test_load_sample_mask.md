# tests/vision/datasets/test_load_sample_mask.cpp


Suites: `Medium_DatasetsLoadSampleMask`.

Tests `load_sample_mask` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_mask.hpp`. The refusals of
a sample outside the list, a sample without a mask and a mask index outside the files, and the
report of an absent file, are tested in `test_fit_mask.cpp`, under
`Small_DatasetsLoadSampleMaskRefusal` and `Medium_DatasetsLoadSampleMask`. This file holds the
decode of a mask file: its channels and bytes are kept as stored. The mask is written under the
system's temporary directory with the helpers of `datasets_test_support.hpp`.

## AColorMaskKeepsTheChannelsItStores

A raw RGB image of 1 by 2 pixels holding 1, 2, 3, 40, 50, 60 is the mask of sample 0. The loaded
mask has that size, three channels and those six bytes, so a mask that is not a palette image
keeps the channels it stores and is not reduced to one.

## AGrayMaskKeepsItsClassBytesAndTheVoidByte

A raw gray image of 2 rows and 3 columns holding 0, 3, 255, 1, 2, 0 is the mask of sample 0.
The loaded mask has that size, one channel and those six bytes, the void byte 255 among them.
