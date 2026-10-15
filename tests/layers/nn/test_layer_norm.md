# tests/layers/nn/test_layer_norm.cpp


Suites: `Medium_LayerNorm`.

The file runs LAYER_NORM and the `LayerNorm` layer on an engine with an epsilon of 1e-5 and compares
with a host loop in double: FLOAT32 results within 2e-5 and FLOAT64 results within 1e-12. The
gradient cases normalise a FLOAT64 `[2, 3, 4]` value, reshape it to `[6, 4]`, and compare the
analytic gradient of one leaf with central differences while the other leaves hold seeded noise.

## EachRowIsNormalisedByItsOwnStatistics

A FLOAT32 `[2, 3, 5]` value with a gain and a bias of five elements is normalised and the step is
run twice. The output equals the host's rows.

## TheSavedStatisticsAreTheRowMeanAndInverseDeviation

For the FLOAT64 rows `{1, 2, 3, 4}` and `{-4, -2, 0, 2}` the saved means are 2.5 and -1, and the
saved inverse deviations are `1 / sqrt(1.25 + eps)` and `1 / sqrt(5 + eps)`.

## TheInputGradientMatchesCentralDifferences

The gradient at the `[2, 3, 4]` input.

## TheGainGradientMatchesCentralDifferences

The gradient at the gain of four elements.

## TheBiasGradientMatchesCentralDifferences

The gradient at the bias of four elements.

## ALayerNormalisesWithItsGainAndItsBias

A `LayerNorm` of width 4 over a FLOAT32 `[2, 3, 4]` value keeps the shape, and with its weight and
bias written from noise its output equals the host's rows.

Suites: `Medium_MaxPoolTrailing`.

A `MaxPool2d` with a 2x2 window, a stride of one and one trailing row and column of padding pools a
`[1, 3, 3, 1]` input into nine values. Each is compared with the maximum of its truncated window
within 1e-6.

## TrailingWindowsTakeTheMaximaOfTheirTruncatedTaps

The input `[[5, -1, 2], [-3, 8, 0], [4, -7, 6]]` pools to `[[8, 8, 2], [8, 8, 6], [4, 6, 6]]`.

## AllNegativeTrailingWindowsKeepTheirTrueMaximum

The input runs from -18 to -10. The result is `[[-14, -13, -13], [-11, -10, -10], [-11, -10, -10]]`;
a pad that took part in the maximum as zero would show as 0.

Suites: `Medium_MishValues`.

The `Mish` layer runs over one FLOAT32 row on the engine. Each result is compared with `x *
tanh(log1p(exp(x)))`, computed in double, within 1e-6 of the exact value's magnitude. A zero must be
matched exactly.

## MatchesTheExactFormAtTheAcceptancePoints

The inputs are -20, -1, 0, 1 and 20.

## SweepMatchesTheExactFormAndStatesTheDarknetDifference

The inputs run from -30 to 30 in steps of 0.5, 121 values. The engine's results match the exact
form. The test also evaluates, on the host in double, the form with a softplus cut-off at a
magnitude of 20, records its largest difference from the exact form as the property
`darknet_cut_off_largest_difference`, and expects that difference to be at most 1e-12.

## test_layer_norm.cpp


The LayerNorm layer's trace, its gain and bias, its refusals and its factory.

Suites: `Small_LayerNorm`, `Medium_LayerNormDeathTest`.

Tests `NN::LayerNorm` in `modules/layers/nn/include/SushiAI/nn/layer_norm.hpp` and its factory
`NN::make_layer_norm_node`. Every layer normalises over a last extent of 4. A `static_assert`
holds that `LayerNorm` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

## ATraceDeclaresTheGainAndTheBiasAndEmitsOneNode

Over a [2, 3, 4] input a training trace is ok and adds three nodes. The node that writes the
result is LAYER_NORM, reads the input, the weight and the bias in that order and holds
`K_LAYER_NORM_EPSILON` in attribute slot 0. The result is a float32 value of shape [2, 3, 4].

## TheGainIsAScaleAndTheBiasABias

After a trace over a [5, 4] float64 input, the weight and the bias are declared with the shape
[4] and the type float64, with the roles `ParameterRole::Scale` and `ParameterRole::Bias`. A
layer whose `epsilon` is 1e-3 puts 1e-3 in the node's attribute slot 0.

## TheWalkYieldsTheGainThenTheBias

`NN::walk_state` under the prefix `norm` visits `norm.weight` and then `norm.bias`.

## AnInputThatDoesNotFitFails

A [2, 5] input fails with `InvalidDescription`, "LayerNorm(4)", "[..., 4] input" and "[2, 5]".
An int32 input fails with "half, float32 or float64 input" and "int32". `forward` returns its
input, the graph gains no node, the weight stays undeclared and the graph has a refusal.

## ALayerDescribedWrongFails

A default layer, whose `dim` is 0, fails with "positive dim" and "got 0". An `epsilon` of 0
fails with "LayerNorm(4)" and "positive epsilon", and that layer's weight stays undeclared.

## OutputShapeIsTheInputShape

`output_shape([4])` is [4] and `output_shape([6, 4])` is [6, 4].

## TheFactoryReadsTheWidthAndTheEpsilon

`make_layer_norm_node` over one [6, 4] input and `{}` returns a `ModuleNode<LayerNorm>` whose
layer has the `dim` 4 and the epsilon `K_LAYER_NORM_EPSILON`, and whose one port has the shape
[6, 4]. With `{"ln_epsilon": 0.001}` the layer's epsilon is 0.001.

## TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `momentum` key, an
`ln_epsilon` of -1 ("positive"), an `ln_epsilon` of "small" ("a number"), two inputs ("reads
one input", "given 2") and a rank-6 input ("rank 1 to 5").

## ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"LayerNorm(4)", "value 7" and "holds 1 values".
