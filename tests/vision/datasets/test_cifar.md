# tests/vision/datasets/test_cifar.cpp


Suites: `Small_DatasetsCifar`, `Small_DatasetsCifarLabel`, `Small_DatasetsCifarSize` and
`Medium_DatasetsCifar`.

Tests `read_cifar` in `modules/vision/datasets/include/SushiAI/vision/datasets/cifar.hpp`. The
`Small_` cases read batch files from a `MemorySource` under the folder `data`; the `Medium_` cases
write them under the system's temporary directory. A record is its label bytes followed by three
planes of 32 by 32 bytes, each plane filled with one level.

## TheCIFAR10TrainingSplitListsItsFiveFilesInOrder

With `data_batch_1.bin` to `data_batch_5.bin` each holding one record labelled with its batch
number, the "train" split of CIFAR-10 has 10 classes, the five files in order, the files 0 to 4 for
its five samples and the classes 1 to 5.

## ASampleRefersToThePlanesAfterItsLabel

In `test_batch.bin` with two records labelled 3 and 9, the samples have the classes 3 and 9. The
second sample's pixels start at byte 3074, are 32 by 32 with three channels and are
`PlaneLayout::Planar`.

## TheStoredPlanesReadBackAsOneInterleavedImage

`read_stored_pixels` over the second sample, from a stream holding the batch file, gives an image
of 32 by 32 pixels that each hold the bytes 40, 50, 60.

## CIFAR100TakesTheFineLabelFromTheSecondByte

A record labelled 19 and 87 gives 100 classes and one sample of class 87 whose pixels start at
byte 2.

## CIFAR100TakesTheCoarseLabelFromTheFirstByte

The same record read as the coarse variant gives 20 classes and one sample of class 19.

## ALabelOutsideTheClassesNamesItsRecord

One row each: a CIFAR-10 file whose second of two records is labelled 10 fails with `InvalidFile`,
"test_batch.bin", "record 2 of 2" and "label 10"; a CIFAR-100 coarse file whose only record has the
coarse label 20 fails with "test.bin", "record 1 of 1" and "label 20".

## AFileThatIsNotWholeRecordsIsRefused

One row each: a file cut one byte short of a record fails with `InvalidFile`, "test_batch.bin",
"3072 bytes" and "3073 bytes"; an empty file fails with "test_batch.bin" and "0 bytes".

## AnotherSplitIsRefused

The split "val" over an empty source fails with `InvalidDescription` and "'val'".

## AnAbsentFileIsNotFound

With only `data_batch_1.bin` on disk, the training split fails with `FileNotFound` and
"data_batch_2.bin". It reaches the file system, so it is `Medium_`.

## ABatchOnDiskListsItsRecords

The file-system form of `read_cifar` reads a written `test_batch.bin` of two records into two
samples, the second of class 9 starting at byte 3074.
