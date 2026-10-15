# tests/layers/nn/test_init.cpp


Suites: `Integration_InitState`.

The cases check the fill that `NN::initialize()` writes over a `NN::ModelState`: the constants of a
batch norm, the zeroed final layer, and the determinism of the default seed.

## AScaleIsOneAShiftIsZeroAndABufferHoldsItsInitialValue

A training trace of a batch norm named `bn` is initialised. Afterwards `bn.weight` holds only ones,
`bn.bias` and `bn.running_mean` hold only zeros, and `bn.running_var` holds only ones.

## ZeroFinalLayerZeroesOnlyTheLastFanScaledEntry

A convolution named `conv`, followed by a batch norm named `bn`, is initialised with the default
policy. Every element of `conv.weight` is zero, and `bn.weight` is still one.

## WithoutZeroFinalLayerTheLastWeightIsNotZero

An MLP initialised with `zero_final_layer` set to false has at least one nonzero element in
`2.weight`.

## TheSameSeedGivesTheSameBits

Two MLPs of the same architecture are each created as a state and initialised with the default
policy. Every entry has the same bytes in both states.

## ANormalFillHasTheDeviationItStates

The table of an `Embedding` of 64 rows and width 32 is initialised with the default policy. Over its
2048 values the mean is within 0.0025 of zero and the deviation within 0.002 of
`NN::K_EMBEDDING_DEVIATION`.

## ANormalFillOfNoDeviationIsRefused

An `Embedding` whose weight asks for `ParameterInit::normal(0.0)` is refused by `NN::initialize` as
`InvalidDescription`, with "positive, finite deviation" in the message.
