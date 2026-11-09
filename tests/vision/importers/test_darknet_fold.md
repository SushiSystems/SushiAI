# tests/vision/importers/test_darknet_fold.cpp


Suites: `Small_DarknetFold` and `Medium_DarknetFoldFatal`.

Tests `fold_darknet_normalisation`, which folds the scales, rolling means and rolling
variances of a `DarknetConvolution` into its weights and biases with the epsilon 0.00001.
The two death tests run a child process and are `Medium_`; the rest are `Small_`. No tolerance
is typed: the first case measures each value against one rounding of float with
`TestSupport::measure_relative_error`, and the second uses 256 times the float unit roundoff.

## OneFilterFoldMatchesTheFormulaInLongDouble

For one filter of eight weights with the bias 0.75, the scale 1.5, the mean 0.25 and the
variance 0.5, each folded weight equals the weight times 1.5 / sqrt(0.5 + epsilon), and the
bias equals 0.75 - 1.5 * 0.25 / sqrt(0.5 + epsilon). Both are computed in long double and
compared with `TestSupport::measure_relative_error<float>` and one rounded operation, since
the fold computes in double and stores a float. After the fold the scales, means and variances
are empty.

## FoldedAndUnfoldedConvolutionsAgreeOnTheHost

For three filters over a two-channel 5x5 input with values drawn from the seed 20261009, the
folded convolution plus its bias and the unfolded convolution followed by the normalisation
differ by at most 256 times the float unit roundoff at every output pixel.

## FoldingTwiceStopsTheProcess

`Medium_DarknetFoldFatal`. A second fold of the same convolution stops the process with a message holding
"not normalised".

## NonPositiveVarianceStopsTheProcess

`Medium_DarknetFoldFatal`. A rolling variance of -1 stops the process with a message holding "positive variance".
