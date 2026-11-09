# tests/vision/training/test_detection_truth_check.cpp


Suites: `Small_DetectionTruthCheck`, `Small_DetectionTruthCheckSizes` and
`Small_DetectionTruthCheckLabels`.

Tests `check_truth_layout` and `check_truth_labels` of `detection_truth_check.hpp`, the checks of a
ground truth that the anchor and the anchor-free steps both run. The truth holds two images of two
rows, the last row padding, over three classes. No case needs a device.

The steps call the checks, and the checks are tested here. The free step's file keeps one case
that shows a refused truth counts no run.

### AcceptsATruthOfTheDeclaredBatch

A truth of the declared batch with arrays of the sizes it needs passes both checks.

### ATruthOfAnotherBatchNamesBothBatchesAndTheOrigin

A truth of 3 images against a batch of 2 fails with `Incompatible` naming "3 images" and "ports
were traced for a batch of 2".

### ANegativeRowCountIsRefused

A row count of -1 fails with `Incompatible` naming "-1 rows" and "cannot be negative".

### AnArrayOfAnotherSizeIsRefusedWithTheSizesItNeeds

Rows: one label, one flag or one box value fewer. Each fails with `Incompatible` naming "needs 4
labels, 4 flags and 16 box values" and what the truth holds: "3, 4 and 16", "4, 3 and 16" or "4,
4 and 15".

### AValidLabelOutsideTheClassesNamesTheRowAndTheHolder

Rows: label 3 on row 2 against three classes held by `head.1`, and label -1 on row 0 held by "the
ports". Each fails with `InvalidDescription` naming the row and the label; the first also names
"head.1 has 3 classes".

### APaddingRowMayHoldAnyLabel

A label of 99 on the padding row passes, since no step reads it.
