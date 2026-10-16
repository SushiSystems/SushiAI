# tests/language/blocks/test_expert_projection.cpp


Suites: `Small_BlockExpertProjection`.

Tests `Language::Blocks::ExpertProjection` in
`modules/language/blocks/include/SushiAI/language/blocks/expert_projection.hpp`. Every case
builds a host graph with a [5, 24] gated value and a [5, 3] gate, for 3 experts of hidden width 8
and a result width of 6. No case needs a device.

## make_projection

Returns a projection of those three extents with neither parameter declared.

## TheFieldsAreTheWeightAndTheBias

`fields()` names "weight" first and "bias" second.

## AMixSumsTheGatedProjectionAndTheGatedBias

`mix` returns a [5, 6] value whose producer is a K_ADD of two K_MATMUL nodes, the first reading
the gated value and the second the gate. The weight is declared as [6, 24] with the role
`Weight` and the bias as [3, 6] with the role `Bias`.

## AnEmptyExtentIsRefused

A projection of zero experts fails with `InvalidDescription` naming "ExpertProjection" and
"experts=0", adds no node and leaves the weight undeclared.

## AGatedValueOfAnotherWidthIsRefused

A gated value of [5, 8] does not match the [6, 24] weight, and `mix` fails with
`InvalidDescription`.
