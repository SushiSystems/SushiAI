# tests/layers/train/test_binary_confusion.cpp


Suites: `Medium_BinaryConfusion`, `Small_BinaryConfusionRatios`, `Small_DecisionThreshold`.

Tests `Train::count_confusion`, the four ratios over a `BinaryConfusion`, `Train::logit_of`,
`Train::read_threshold` and `Train::require_host_reals`. The `Medium_BinaryConfusion` fixture
needs an engine to allocate the logits and the truth. The other two suites run on the host.

## EachRowLandsInOneOfTheFourCounts

Five rows of one column with the logits 2, 1, -1, -3, 0 and the truth 1, 0, 1, 0, 1, cut at
`logit_of(0.5)`, give 2 true positives, 1 false positive, 1 false negative and 1 true negative.
The accuracy is 3/5, and the precision, the recall and the F1 are each 2/3.

## AColumnIsCountedAlone

Over a [2, 2] float64 matrix of logits and a [2, 2] truth, column 0 counts 2 true positives.
Column 1 counts 0 true positives, 1 false negative and 1 false positive.

## TheThresholdMovesTheCut

One row with the logit 1 and the truth 1 is a true positive at the threshold 0.5 and a false
negative at the threshold 0.9.

## ARatioOverNothingIsZero

For a `BinaryConfusion` of four zeros the accuracy, the precision, the recall and the F1 are
each 0. With 4 true negatives and nothing else the accuracy is 1 and the F1 is 0.

## LogitOfInvertsTheSigmoid

`logit_of(0.5)` is exactly 0. For 0.1, 0.731 and 0.9 the sigmoid of `logit_of(p)` is p within
1e-12.

## TheReaderDefaultsToOneHalf

`read_threshold` over `{}` returns `K_DEFAULT_THRESHOLD`, and over `{"threshold": 0.25}` it
returns 0.25.

## TheReaderRefusesWhatIsNoProbability

The key `cut` fails with `InvalidDescription` and a message naming "objective", `'cut'` and
`'threshold'`. The value `"high"` fails with a message containing "number". The thresholds 0,
1, -0.5 and 1.5 each fail with "above 0 and below 1".

## TheHostReadRefusesWhatItCannotRead

`require_host_reals` is ok for a float64 tensor of four elements asked for 4 values, and
`host_real(reals, 2)` is 2.5. An int32 tensor fails with `InvalidDescription` and a message
containing "host read of the logits" and "int32". Asking the float64 tensor for 6 values fails
with `Incompatible`, "6 values" and "holds 4".
