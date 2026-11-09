# tests/layers/train/test_binary_confusion.cpp


Suites: `Small_BinaryConfusion`, `Small_BinaryConfusionColumn`, `Small_BinaryConfusionCut`,
`Small_BinaryConfusionRatios`, `Small_DecisionThreshold`, `Small_DecisionThresholdLogit`,
`Small_DecisionThresholdRead`, `Small_DecisionThresholdRefused`.

Tests `Train::count_confusion`, the four ratios over a `BinaryConfusion`, `Train::logit_of`,
`Train::read_threshold` and `Train::require_host_reals`. The tensors are allocated on the engine of
a `HostExecutionBackend`, so no suite needs a device.

## HostTensors

`HostTensors` owns a `HostExecutionBackend` and `tensor_of`, which allocates a tensor of a shape
and an element type on its engine and writes the given values into it. `Small_BinaryConfusion`,
`Small_BinaryConfusionColumn` and `Small_BinaryConfusionCut` derive from it.

## ColumnRow

`ColumnRow` holds a column of the [2, 2] logits and the true positives, false positives and false
negatives that column gives.

## CutRow

`CutRow` holds a threshold probability and the true positives and false negatives the one row of
logit 1 gives at that cut.

## EachRowLandsInOneOfTheFourCounts

Five rows of one column with the logits 2, 1, -1, -3, 0 and the truth 1, 0, 1, 0, 1, cut at
`logit_of(0.5)`, give 2 true positives, 1 false positive, 1 false negative and 1 true negative.
The ratios of those counts are `TheRatiosAreTheQuotientsOfTheCounts`.

## AColumnIsCountedAlone

Over a [2, 2] float64 matrix of logits and a [2, 2] truth, the first column counts 2 true positives
and nothing else. The second column counts 0 true positives, 1 false positive and 1 false negative.

## TheThresholdMovesTheCut

One row with the logit 1 and the truth 1 is a true positive at the threshold 0.5 and a false
negative at the threshold 0.9.

## ARatioOverNothingIsZero

For a `BinaryConfusion` of four zeros the accuracy, the precision, the recall and the F1 are each 0.

## RejectionsAloneGiveFullAccuracyAndNoF1

With 4 true negatives and nothing else the accuracy is 1 and the F1 is 0.

## TheRatiosAreTheQuotientsOfTheCounts

For 2 true positives and 1 of each other count the accuracy is 3/5, and the precision, the recall and
the F1 are 2/3, 2/3 and 4/6. Each is within one rounded operation of double of the quotient: the counts are exact integers and
the division rounds once.

## LogitOfOneHalfIsZero

`logit_of(0.5)` is exactly 0.

## LogitOfInvertsTheSigmoid

For 0.1, 0.731 and 0.9 the sigmoid of `logit_of(p)` is p within 5 + ceil(|x|) rounded operations of
double, x being the logit: the subtraction and the quotient of `logit_of` round twice, its logarithm
rounds x by |x| unit roundoffs once the exponential reads it, and the exponential, the sum and the
quotient of the sigmoid round three times.

## ReadRow

`ReadRow` holds a JSON document and the threshold `read_threshold` returns for it.

## TheReaderDefaultsToOneHalf

`read_threshold` over `{}` returns `K_DEFAULT_THRESHOLD`, and over `{"threshold": 0.25}` it returns
0.25.

## RefusedRow

`RefusedRow` holds a JSON document and up to three texts the refusal holds.

## TheReaderRefusesWhatIsNoProbability

The key `cut` fails with `InvalidDescription` and a message naming "objective", `'cut'` and
`'threshold'`. The value `"high"` fails with a message containing "objective" and "number". The
thresholds 0, 1, -0.5 and 1.5 each fail with "above 0 and below 1".

## TheHostReadAcceptsAFloatTensorOfTheAskedCount

`require_host_reals` is ok for a float64 tensor of four elements asked for 4 values, and
`host_real(reals, 2)` is 2.5.

## TheHostReadRefusesAnIntegerTensor

An int32 tensor fails with `InvalidDescription` and a message containing "host read of the
logits" and "int32".

## TheHostReadRefusesAnotherElementCount

Asking a float64 tensor of four elements for 6 values fails with `Incompatible`, "6 values" and
"holds 4".
