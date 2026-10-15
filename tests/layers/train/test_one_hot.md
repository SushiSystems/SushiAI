# tests/layers/train/test_one_hot.cpp


Suites: `Unit_OneHot`.

Tests `Train::fill_one_hot`. The fixture needs an engine. The labels are an int32 tensor of one
extent, and the rows hold 7 in every element before a fill.

## EveryRowHoldsOneInItsLabelsColumn

The labels 2, 0, 1, 2 fill a [4, 3] float32 tensor with 1 in each row's label column and 0
elsewhere.

## AFloat64LeafReceivesTheSameRows

The same labels fill a [4, 3] float64 tensor with the same rows.

## ASecondFillReplacesTheFirst

After a fill with the labels 0, 0 and a second with 1, 0, a [2, 2] tensor holds 0, 1, 1, 0.

## ALabelOutsideTheClassesIsRefusedAndNothingIsWritten

Over three classes the labels 0, 3 fail with `Incompatible`, "row 1", "is 3" and "3 classes",
and the labels -1, 0 fail with "row 0", "is -1" and "3 classes". The rows still hold six 7s.

## AnotherLabelCountIsRefusedWithBothShapes

Three labels for a [4, 3] tensor fail with `Incompatible`, "[3]" and "[4, 3]".

## LabelsOfAnotherElementTypeAreRefused

Float32 labels fail with `Incompatible`, "float32" and "int32".

## RowsThatAreNotRealAreRefused

An int32 [2, 3] tensor of rows fails with `Incompatible`, "int32", "float32" and "float64".
