# tests/layers/train/test_one_hot.cpp


Suites: `Small_OneHot`, `Small_OneHotDType`, `Small_OneHotOutOfRange`.

Tests `Train::fill_one_hot`. The tensors are allocated from the engine of a
`HostExecutionBackend`. The labels are an int32 tensor of one extent, and the rows hold 7 in every
element before a fill.

## DTypeRow

`DTypeRow` holds the element type of the leaf a fill writes.

## OutOfRangeRow

`OutOfRangeRow` holds labels with one out of range and the two texts of the refusal.

## EveryRowHoldsOneInItsLabelsColumn

The labels 2, 0, 1, 2 fill a [4, 3] tensor with 1 in each row's label column and 0 elsewhere.
The rows are a float32 leaf and a float64 leaf. 0 and 1 are exact in both types, so the
comparison is equality.

## ASecondFillReplacesTheFirst

After a fill with the labels 0, 0 and a second with 1, 0, a [2, 2] tensor holds 0, 1, 1, 0.

## ALabelOutsideTheClassesIsRefusedAndNothingIsWritten

Over three classes the fill fails with `Incompatible` and "3 classes", and the rows still hold
six 7s. The rows are the labels 0, 3 ("row 1", "is 3") and -1, 0 ("row 0", "is -1").

## AnotherLabelCountIsRefusedWithBothShapes

Three labels for a [4, 3] tensor fail with `Incompatible`, "[3]" and "[4, 3]".

## LabelsOfAnotherElementTypeAreRefused

Float32 labels fail with `Incompatible`, "float32" and "int32".

## RowsThatAreNotRealAreRefused

An int32 [2, 3] tensor of rows fails with `Incompatible`, "int32", "float32" and "float64".
