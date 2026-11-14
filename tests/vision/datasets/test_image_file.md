# tests/vision/datasets/test_image_file.cpp


Suite: `Small_DatasetsImageFile`.

Tests `read_image_file` in
`modules/vision/datasets/include/SushiAI/vision/datasets/image_file.hpp`, over files held by a
`MemorySource`.

## TheImageIsTheOneSampleWithNoObjectAndNoClass

`cat.JPG` gives one sample that names that file, has no object and no class, and a class count of
80.

## AFileWithNoImageExtensionIsRefused

`notes.txt` fails with `InvalidDescription` and "no image extension".

## AFileThatIsAbsentIsNotFound

A file the source does not hold fails with `FileNotFound` and its name.

## ANoClassCountIsRefused

A class count of 0 fails with `InvalidDescription` and "class count".
