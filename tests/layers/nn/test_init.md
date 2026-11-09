# tests/layers/nn/test_init.cpp


Suites: `Medium_InitState`, `Medium_InitStateSeed`.

The cases check the fill that `NN::initialize()` writes over a `NN::ModelState`: the constants of a
batch norm, the zeroed final layer, and the determinism of the default seed.

## Helpers

`expect_every_element_equal` reads a tensor of the state and expects every element to equal a
value. `sample_mean` and `sample_deviation` give the statistics of a vector. `mean_bound` and
`deviation_bound` give the bounds of the normal-fill case: `K_STANDARD_ERRORS` (6.1, a two-sided
normal tail of 1.0e-9) standard errors of the statistic, which are sigma over the root of the
count for the mean and sigma over the root of twice the count for the deviation.
`prepare_batch_norm` builds and traces a three-channel batch norm.

## AScaleStartsAtOne

A training trace of a batch norm named `bn` is initialised. Afterwards `bn.weight` holds only
ones.

## AShiftStartsAtZero

After the same initialisation `bn.bias` holds only zeros.

## ABufferHoldsItsInitialValue

After the same initialisation `bn.running_mean` holds only zeros and `bn.running_var` holds only
ones.

## ZeroFinalLayerZeroesTheConvWeightAndKeepsTheNormScaleAtOne

A convolution named `conv`, followed by a batch norm named `bn`, is initialised with the default
policy. Every element of `conv.weight` is zero, and every element of `bn.weight` is one.

## WithoutZeroFinalLayerTheLastWeightIsNotZero

An MLP initialised with `zero_final_layer` set to false has at least one nonzero element in
`2.weight`.

## Medium_InitStateSeed.TheSameSeedGivesTheSameBits

Two MLPs of the same architecture are each created as a state and initialised with the default
policy. The entry of the row has the same bytes in both states. The rows are TheFirstWeight
(`0.weight`), TheFirstBias (`0.bias`), TheLastWeight (`2.weight`) and TheLastBias (`2.bias`).

## ANormalFillHasTheDeviationItStates

The table of an `Embedding` of 64 rows and width 32 is initialised with the default policy. Over its
2048 values the mean is within `mean_bound` of zero and the deviation within
`deviation_bound` of `NN::K_EMBEDDING_DEVIATION`. The seed is the fixed default of the policy, so
the draw is the same on every run.

## ANormalFillOfNoDeviationIsRefused

An `Embedding` whose weight asks for `ParameterInit::normal(0.0)` is refused by `NN::initialize` as
`InvalidDescription`, with "positive, finite deviation" in the message.
