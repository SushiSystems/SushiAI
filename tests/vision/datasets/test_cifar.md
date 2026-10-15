# tests/vision/datasets/test_cifar.cpp


Suites: `Unit_DatasetsCifar`.

Tests `read_cifar` in `modules/vision/datasets/include/SushiAI/vision/datasets/cifar.hpp` over
batch files the cases write under the system's temporary directory. A record is its label
bytes followed by three planes of 32 by 32 bytes, each plane filled with one level.

## TheCIFAR10TrainingSplitListsItsFiveFilesInOrder

With `data_batch_1.bin` to `data_batch_5.bin` each holding one record labelled with its batch
number, the "train" split of CIFAR-10 has 10 classes, five files and five samples. Sample i
names file i, the file is `data_batch_<i + 1>.bin` and the class is i + 1.

## ASampleRefersToThePlanesAfterItsLabel

A `test_batch.bin` of two records gives the classes 3 and 9. The second sample's stored pixels
start at byte 3074, are `K_CIFAR_SIDE` square with three channels and are `PlaneLayout::Planar`.
`read_stored_pixels` over them gives 3072 bytes whose sampled pixels hold 40, 50, 60.

## CIFAR100TakesTheFineOrTheCoarseLabel

A `train.bin` record with the label bytes 19 and 87 reads as class 87 of 100 under
`CIFARVariant::CIFAR100Fine`, with its pixels at offset 2, and as class 19 of 20 under
`CIFARVariant::CIFAR100Coarse`.

## ALabelOutsideTheClassesNamesItsRecord

A CIFAR-10 file whose second record has the label 10 fails with `InvalidFile`,
"test_batch.bin", "record 2 of 2" and "label 10". A CIFAR-100 record with the coarse label 20
fails with `InvalidFile`, "test.bin", "record 1 of 1" and "label 20".

## AFileThatIsNotWholeRecordsIsRefused

A record one byte short fails with `InvalidFile`, "test_batch.bin", "3072 bytes" and
"3073 bytes". An empty file fails with `InvalidFile`, "test_batch.bin" and "0 bytes".

## AnAbsentFileIsNotFound

With only `data_batch_1.bin` present, the "train" split fails with `FileNotFound` and
"data_batch_2.bin".

## AnotherSplitIsRefused

The split "val" fails with `InvalidDescription` and "'val'".
