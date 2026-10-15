# tests/vision/datasets: test_binary_collator.cpp, test_multi_label_collator.cpp, test_regression_collator.cpp, test_segmentation_collator.cpp


The roles the binary, multi-label, regression and segmentation collators write, and the samples they refuse. Needs an engine.

Suites: `Unit_DatasetsTaskCollators`.

Tests `BinaryCollator`, `MultiLabelCollator`, `RegressionCollator` and `SegmentationCollator`
in `modules/vision/datasets/include/SushiAI/vision/datasets/`. The fixture is an `EngineTest`.
It holds two gray samples of one row and two columns, from `first.png` and `second.png`, and
allocates batches of two samples.

## BinaryPublishesLabels

`BinaryCollator::make` fails with `InvalidDescription` for the positive class 2
("BinaryCollator", "the positive class is 2") and for -1 ("the positive class is -1"). A
collator of the positive class 1 reports the task `K_TASK_BINARY` and one target: the role
`K_ROLE_LABELS`, an empty sample shape, `Float32` and 0 classes.

## BinaryWritesOneForThePositiveClass

With the class indices 1 and 0, a collator of the positive class 1 writes the labels 1, 0 and
a collator of the positive class 0 writes 0, 1.

## BinaryNamesTheFileOfASampleItCannotLabel

With no class index on the second sample, `collate` fails with `InvalidDescription`,
"second.png", "BinaryCollator" and "no class index". With the class index 2 it fails with
"second.png", "class index 2" and "[0, 2)".

## MultiLabelWritesTheSetOfClassesPresent

`MultiLabelCollator::make(0)` fails with `InvalidDescription`, "MultiLabelCollator" and
"0 classes". A collator of 4 classes reports the task `K_TASK_MULTI_LABEL` and one target of
the role `K_ROLE_LABELS`, the sample shape [4] and `Float32`. The first sample has objects of
the classes 2, 0 and 2 and the class index 3; the second has neither. Over a target first
filled with 9, `collate` writes 1, 0, 1, 1 for the first sample and four zeros for the second.

## MultiLabelRefusesAClassOutsideTheSet

An object of class 4 on the second sample, over 4 classes, fails with `InvalidDescription`,
"second.png", "MultiLabelCollator", "the class 4" and "[0, 4)".

## RegressionCopiesFeaturesAndTargets

`RegressionCollator::make` fails with `InvalidDescription` for 0 features ("0 features") and
for 0 targets ("0 targets"), each with "RegressionCollator". A collator of 3 features and 2
targets reports the task `K_TASK_REGRESSION` and one target of the role `K_ROLE_VALUES`, the
sample shape [2] and `Float32`. For samples with the features 1, 2, 3 and 4, 5, 6 and the
targets 10, 20 and 40, 50, `collate` writes the inputs 1 to 6 and the values 10, 20, 40, 50.

## RegressionRefusesWhatHoldsNoRow

Over samples that hold images and no row, `collate` fails with `InvalidDescription`,
"first.png", "RegressionCollator" and "0 features and 0 targets". Over inputs of the sample
shape [2] it fails with `Incompatible`, "RegressionCollator" and "float32 of [2, 3]".

## SegmentationPublishesTheMask

`SegmentationCollator::make` fails with `InvalidDescription` for 0 classes
("SegmentationCollator", "0 classes"), for 256 classes ("256 classes", "1 to 255"), for the
ignore index 2 over 3 classes ("the ignore index 2 is one of the 3 classes") and for a size
of 0 by 2 ("0 by 2 pixels"). A collator of 3 classes reports the task `K_TASK_SEGMENTATION`
and one target of the role `K_ROLE_MASK`, the sample shape [1, 2], `Int32` and 3 classes.

## SegmentationWritesClassesAndTheIgnoreIndex

For the masks 2, `K_MASK_VOID` and 0, 1, `collate` writes 2, the ignore index, 0, 1, with the
ignore index `K_DEFAULT_IGNORE_INDEX` and with -100.

## SegmentationRefusesAByteThatIsNoClass

A mask byte of 3 over 3 classes fails with `InvalidDescription`, "second.png",
"SegmentationCollator", "holds 3 at row 0, column 1" and "below 3 and 255 names none".

## SegmentationRefusesASampleWithoutAMask

With no mask on the second sample, `collate` fails with `InvalidDescription`, "second.png",
"SegmentationCollator" and "the sample's mask is 0 by 0".

## SegmentationPaintsThePolygonsOfASample

With `MaskOrigin::Polygons` and the ignore index -1, the first sample has a polygon of class 2
over its second pixel and the second sample one of class 0 over its first pixel. `collate`
writes -1, 2, 0, -1. With `background` 1 it writes 1, 2, 0, 1.

## SegmentationRefusesAnObjectWithoutAPolygon

With `MaskOrigin::Polygons`, an object with a box and no polygon on the second sample fails
with `InvalidDescription`, "second.png", "paint_polygon_mask" and "no polygon".

## SegmentationRefusesABackgroundItCannotWrite

`SegmentationCollator::make` fails with `InvalidDescription` for `background` 1 without
`MaskOrigin::Polygons` ("a background of 1", "masks come from files") and for `background` 3
over 3 classes with it ("the background 3 is none of the 3 classes").
