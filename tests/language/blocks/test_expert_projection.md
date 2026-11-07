# tests/language/blocks/test_expert_projection.cpp


Suites: `Small_BlockExpertProjection`, `Medium_BlockExpertProjection`.

Tests `Language::Blocks::ExpertProjection` in
`modules/language/blocks/include/SushiAI/language/blocks/expert_projection.hpp`. Every case
builds a host graph with a [5, 24] gated value and a [5, 3] gate, for 3 experts of hidden width 8
and a result width of 6. Only the numeric case needs the engine.

## make_projection

Returns a projection of those three extents with neither parameter declared.

## AWalkYieldsThePrefixedWeightThenTheBias

`NN::walk_state` over a mixed projection under the prefix "ep" yields "ep.weight" first and
"ep.bias" second.

## AMixDeclaresTheWeightAndTheBiasAndReturnsRowsOfDim

`mix` returns a [5, 6] value. The weight is declared as [6, 24] with the role `Weight` and the
bias as [3, 6] with the role `Bias`.

## AMixSumsTheGatedProjectionAndTheGatedBias

`Medium_BlockExpertProjection`, which needs the engine: the compiled step of a mix over small
integer values returns, for each row, the gated row times the transposed weight plus the gate
row times the bias, equal to the host sum bit for bit (`mixed_on_host`).

## AnEmptyExtentIsRefused

A projection of zero experts fails with `InvalidDescription` naming "ExpertProjection" and
"experts=0", adds no node and leaves the weight undeclared.

## AGatedValueOfAnotherWidthIsRefused

A gated value of [5, 8] does not match the [6, 24] weight, and `mix` fails with
`InvalidDescription`.
