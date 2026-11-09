# tests/layers/nn/test_layer_norm.cpp

The LayerNorm layer's trace, its gain and bias, its refusals, its factory and its values on an
engine.

Suites: `Small_LayerNorm`, `Small_LayerNormFactoryRefusal`, `Medium_LayerNormDeathTest`,
`Medium_LayerNorm`.

Tests `NN::LayerNorm` in `modules/layers/nn/include/SushiAI/nn/layer_norm.hpp` and its factory
`NN::make_layer_norm_node`. The small cases normalise over a last extent of 4. A `static_assert`
holds that `LayerNorm` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

## ATraceDeclaresTheGainAndTheBiasAndEmitsOneNode

Over a [2, 3, 4] input a training trace is ok and adds three nodes. The node that writes the
result is LAYER_NORM, reads the input, the weight and the bias in that order and holds
`K_LAYER_NORM_EPSILON` in attribute slot 0. The result is a float32 value of shape [2, 3, 4].

## TheGainIsAScaleOfTheInputType

After a trace over a [5, 4] float64 input, the weight is declared with the shape [4], the type
float64 and the role `ParameterRole::Scale`.

## TheBiasIsABiasOfTheInputType

After the same trace, the bias is declared with the shape [4], the type float64 and the role
`ParameterRole::Bias`.

## TheEpsilonIsStoredInTheNode

A layer whose `epsilon` is 1e-3 puts 1e-3 in the node's attribute slot 0. The record asked for the
epsilon inside the gain and bias tests; it has a test of its own here, so each of the three names
one promise.

## TheWalkYieldsTheGainThenTheBias

`NN::walk_state` under the prefix `norm` visits `norm.weight` and then `norm.bias`.

## AnInputOfAnotherWidthFails

A [2, 5] input fails with `InvalidDescription`, "LayerNorm(4)", "[..., 4] input" and "[2, 5]".
`forward` returns its input, the graph gains no node, the weight stays undeclared and the graph has
a refusal.

## AnIntegerInputFails

An int32 input fails with "half, float32 or float64 input" and "int32", with the same effects.

## ADimThatIsNotPositiveFails

A default layer, whose `dim` is 0, fails with "positive dim" and "got 0".

## AnEpsilonThatIsNotPositiveFails

An `epsilon` of 0 fails with "LayerNorm(4)" and "positive epsilon", and that layer's weight stays
undeclared.

## OutputShapeIsTheInputShape

`output_shape([4])` is [4] and `output_shape([6, 4])` is [6, 4].

## TheFactoryReadsTheWidthAndTheEpsilon

`make_layer_norm_node` over one [6, 4] input and `{}` returns a `ModuleNode<LayerNorm>` whose
layer has the `dim` 4 and the epsilon `K_LAYER_NORM_EPSILON`, and whose one port has the shape
[6, 4]. With `{"ln_epsilon": 0.001}` the layer's epsilon is 0.001.

## Small_LayerNormFactoryRefusal.TheFactoryRefusesTheRow

One row per refusal; each fails with `InvalidDescription` and a message naming "layer 3":

| Row | Input and attributes | The message names |
| --- | --- | --- |
| AStrayKey | [6, 4], `momentum` | 'momentum' |
| AnEpsilonThatIsNotPositive | [6, 4], `ln_epsilon` -1 | 'ln_epsilon', "positive" |
| AnEpsilonThatIsNotANumber | [6, 4], `ln_epsilon` "small" | 'ln_epsilon', "a number" |
| TwoInputs | [4] and [4] | "reads one input", "given 2" |
| ARankOutsideOneToFive | rank 6 | "rank 1 to 5" |

## ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"LayerNorm(4)", "value 7" and "holds 1 values".

## Medium_LayerNorm

The value cases run LAYER_NORM and the `LayerNorm` layer on an engine with an epsilon of 1e-5 and
compare with a host loop in double. The bounds are derived: `K_FLOAT32_TOLERANCE` is 32 rounded
operations, times an amplification of 16 by the inverse deviation, times the float unit roundoff;
`K_FLOAT64_TOLERANCE` is that count times 32 with the double unit roundoff. The gradient cases
normalise a FLOAT64 [2, 3, 4] value, reshape it to [6, 4], and compare the analytic gradient of one
leaf with central differences while the other leaves hold seeded noise. The `Probe` stops the test
when its step does not compile, and `cycling_one_hot_targets` builds the targets.

### EachRowIsNormalisedByItsOwnStatistics

A FLOAT32 [2, 3, 5] value with a gain and a bias of five elements is normalised and the step is
run twice. The output equals the host's rows.

### TheSavedMeanIsTheRowMean

For the FLOAT64 rows {1, 2, 3, 4} and {-4, -2, 0, 2} the saved means are 2.5 and -1.

### TheSavedInverseDeviationIsOneOverTheRowDeviation

For the same rows the saved inverse deviations are `1 / sqrt(1.25 + eps)` and `1 / sqrt(5 + eps)`.

### TheInputGradientMatchesCentralDifferences

The gradient at the [2, 3, 4] input.

### TheGainGradientMatchesCentralDifferences

The gradient at the gain of four elements.

### TheBiasGradientMatchesCentralDifferences

The gradient at the bias of four elements.

### ALayerNormalisesWithItsGainAndItsBias

A `LayerNorm` of width 4 over a FLOAT32 [2, 3, 4] value keeps the shape, and with its weight and
bias written from noise its output equals the host's rows.
