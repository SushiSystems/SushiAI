# tests/vision/importers/test_darknet_fold.cpp


Suites: `Small_DarknetFold`.

Tests `fold_darknet_normalisation`, which folds the scales, rolling means and rolling
variances of a `DarknetConvolution` into its weights and biases with the epsilon 0.00001.

## OneFilterFoldMatchesTheFormulaInLongDouble

For one filter of eight weights with the bias 0.75, the scale 1.5, the mean 0.25 and the
variance 0.5, each folded weight equals the weight times 1.5 / sqrt(0.5 + epsilon), and the
bias equals 0.75 - 1.5 * 0.25 / sqrt(0.5 + epsilon). Both are computed in long double and
compared within 1e-6 relative plus 1e-9 absolute. After the fold the scales, means and
variances are empty.

## FoldedAndUnfoldedConvolutionsAgreeOnTheHost

For three filters over a two-channel 5x5 input with values drawn from the seed 20261009, the
folded convolution plus its bias and the unfolded convolution followed by the normalisation
differ by at most 1e-5 at every output pixel.

## FoldingTwiceStopsTheProcess

A second fold of the same convolution stops the process with a message holding
"not normalised".

## NonPositiveVarianceStopsTheProcess

A rolling variance of -1 stops the process with a message holding "positive variance".
