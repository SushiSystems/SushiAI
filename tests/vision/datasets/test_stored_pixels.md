# tests/vision/datasets/test_stored_pixels.cpp


Suites: `Small_DatasetsStoredPixels` and `Medium_DatasetsStoredPixels`.

Tests `StoredPixels` and `read_stored_pixels` in
`modules/vision/datasets/include/SushiAI/vision/datasets/stored_pixels.hpp`. The `Medium_`
cases read a file written under the system's temporary directory and the `Small_` cases read a
string stream.

## ByteCountIsTheProductOfTheExtents

`byte_count` of 4 by 5 pixels with three channels is 60.

## InterleavedBytesAreCopiedFromTheOffset

Over a stream of the bytes 0 to 19, interleaved pixels of 2 rows, 3 columns and two channels at
offset 5 read as an image of that shape holding the bytes 5 to 16.

## AFileIsOpenedAndItsPixelsAreCopied

The file form of `read_stored_pixels` over a written file of the bytes 0 to 19 gives the same
bytes 5 to 16.

## PlanarBytesAreInterleaved

Planar pixels of 2 by 2 with three channels at offset 1, stored as the planes 1, 2, 3, 4, then
11, 12, 13, 14, then 21, 22, 23, 24, read as 1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24.

## APlaneLongerThanOneChunkIsCopiedWhole

Planar pixels of 100 by 100 with two channels read so that every pixel holds, in channel 0, the
byte of the first plane and, in channel 1, the byte of the second at the same place.

## AFileThatEndsEarlyIsInvalid

Twelve bytes asked at offset 4 of a file of ten fail with `InvalidFile`, "short.bin" and
"6 of 12".

## AFileThatCannotBeOpenedIsInvalid

A path that does not exist fails with `InvalidFile` and "absent.bin".

## ExtentsOutsideAnImageAreRefused

A size of 0 by 2 fails with `InvalidDescription` and "0 by 2", and five channels with
`InvalidDescription` and "5 channels".
