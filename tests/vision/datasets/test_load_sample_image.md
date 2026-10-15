# tests/vision/datasets/test_load_sample_image.cpp


Suites: `Unit_DatasetsLoadSampleImage`.

Tests `load_sample_image` in
`modules/vision/datasets/include/SushiAI/vision/datasets/load_sample_image.hpp`. The cases
that read a file write it under the system's temporary directory.

## StoredPixelsAreReadFromTheirOffset

Over a file of the bytes 0 to 9, a sample whose stored pixels are 2 by 2 with one channel at
offset 2 loads as a 2 by 2 image of one channel holding 2, 3, 4, 5.

## AnImageFileIsDecoded

A sample without stored pixels, whose file is a raw RGB image of 2 rows and 3 columns written
by `write_raw_image_file`, loads as an image of that size and three channels holding the bytes
written.

## ASampleOutsideTheListIsRefused

Sample 3 of a list of one sample fails with `InvalidDescription`, "load_sample_image",
"sample 3" and "1 samples".

## AFileIndexOutsideTheListIsRefused

A sample naming file 5 of a list of one file fails with `InvalidDescription`,
"load_sample_image" and "file 5".

## AnAbsentFileIsNamed

An image file that does not exist fails with `FileNotFound` and "absent.saim".
