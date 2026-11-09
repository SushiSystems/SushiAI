# tests/layers/nn/test_dropout.cpp


That Dropout emits one DROPOUT node in training with its rate and site seed, returns its input in evaluation and at rate 0, gives each site its own seed, its rate check and its factory.

Suites: `Small_Dropout`, `Small_DropoutRate`, `Small_DropoutFactoryRefusal`.

Tests `NN::Dropout` in `modules/layers/nn/include/SushiAI/nn/dropout.hpp` and its factory
`NN::make_dropout_node`. A `static_assert` at the top of the file holds that `Dropout` is a
`ModalModule` and is neither a `Module` nor a `TraceModule`. The training traces state the
seed 5.

## TheTrainingFormEmitsOneDropoutNode

A layer of rate 0.25 over a [2, 3, 4] input adds one DROPOUT node that reads the input. The
node holds 0.25 in `K_DROPOUT_RATE_SLOT`, its seed is `dropout_site_seed(5, 0)`, and the result
has the input's shape. The layer has no parameter and no buffer.

## TheEvaluationFormReturnsItsInputAndEmitsNothing

Under `Trace::evaluation` the layer returns its input, the trace is ok, and the node count and
the value count do not change.

## ARateOfZeroEmitsNothingInTraining

A default `Dropout` under a training trace returns its input and adds no node. The next
`claim_random_site()` of the trace returns 0.

## site_seeds

`site_seeds` traces two Dropout layers of rate 0.5 one after the other on a fresh training trace
of seed 5 and returns the seeds of the two nodes.

## EachSiteOfATraceGetsItsOwnSeed

The two layers get the seeds `dropout_site_seed(5, 0)` and `dropout_site_seed(5, 1)`.

## ASecondTraceOfTheSameSeedRepeatsTheSiteSeeds

Two separate traces of the seed 5 give their sites the same two seeds.

## ATraceWithoutAStatedSeedHasSeedZero

`Trace::training` without a seed and `Trace::evaluation` report the seed 0. `Trace::training`
given 5 reports 5.

## Small_DropoutRate.ARateOutsideItsIntervalFails

For the rate of the row, `trace_training` and `trace_evaluation` return their input and both
traces fail with `InvalidDescription` and a message containing "rate in [0, 1)". The graph has a
refusal. The rows are One (1.0), Negative (-0.1) and NaN.

## OutputShapeIsTheInputShape

`output_shape([6, 4])` is [6, 4].

## TheFactoryReadsTheRate

`make_dropout_node` over one [6, 4] input and `{"rate": 0.1}` returns a `ModuleNode<Dropout>`
whose layer has the rate 0.1 and whose one port has the shape [6, 4].

## Small_DropoutFactoryRefusal.TheFactoryRefuses

Each row fails with `InvalidDescription` and a message naming "layer 3" and the row's
substrings. The rows are AnAbsentRate ("'rate'", "absent"), ARateOfOne ("'rate'", "[0, 1)"),
AStrayKey (a `seed` key: "'seed'") and TwoInputs ("reads one input", "given 2").
