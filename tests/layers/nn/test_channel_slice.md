# tests/layers/nn/test_channel_slice.cpp


The ChannelSlice layer's shape inference, the one SLICE_CHANNELS node it emits with its offset, the offsets, counts and ranks it rejects, and its empty state.

Suites: `Unit_ChannelSlice`.

Tests `NN::ChannelSlice` in `modules/layers/nn/include/SushiAI/nn/channel_slice.hpp`, which
keeps `count` channels of an image batch from `offset` on.

Two `static_assert` lines at the top of the file hold that `ChannelSlice` is a `Module` and is
not a `ModuleContainer`.

## TheOffsetAndCountFieldsAreSetInOrder

`ChannelSlice{32, 16}` has `offset` 32 and `count` 16.

## OutputShapeKeepsHeightAndWidthAndTakesTheRange

With offset 32 and count 32, `output_shape([5, 3, 64])` is [5, 3, 32].

## OutputShapeAcceptsTheFirstLastAndWholeRanges

Over [4, 2, 4], the ranges {0, 1} and {3, 1} give [4, 2, 1], {1, 3} gives [4, 2, 3] and {0, 4}
gives [4, 2, 4].

## TheResultIsTaggedNHWC

The traced result over a [2, 5, 3, 64] input has rank 4 and the layout `TensorLayout::NHWC`.

## ForwardEmitsOneSliceNodeCarryingTheOffset

`forward` with offset 32 and count 32 adds one node. Its operation is `Ops::K_SLICE_CHANNELS`,
its one input is the value passed in, its attribute 0 is 32, and the result has the shape
[2, 5, 3, 32].

## TheWholeChannelRangeTracesToTheInputShape

`ChannelSlice{0, 64}` over a [2, 5, 3, 64] input traces to [2, 5, 3, 64].

## OutputShapeAgreesWithTheTracedShape

For {32, 32} over 64 channels and for {1, 2} over 6 channels, `output_shape` equals the traced
shape with its batch axis dropped.

## OutputShapeRejectsANegativeOffset

Offset -1 with count 4 throws an error naming "ChannelSlice", "offset=-1, count=4" and
"64 channels".

## OutputShapeRejectsAZeroCount

Offset 0 with count 0 throws an error naming "ChannelSlice", "offset=0, count=0" and
"64 channels".

## OutputShapeRejectsANegativeCount

Offset 8 with count -2 throws an error naming "ChannelSlice", "offset=8, count=-2" and
"64 channels".

## OutputShapeRejectsARangePastTheChannels

Offset 32 with count 40 over 64 channels throws an error naming "offset=32, count=40" and
"64 channels".

## OutputShapeRejectsAnOffsetAtTheChannelCount

Offset 64 with count 1 over 64 channels throws an error naming "offset=64, count=1".

## OutputShapeRejectsExtremeValuesWithoutOverflow

An offset of the largest `int64_t` with count 1, and offset 1 with a count of the largest
`int64_t`, each throw an error naming "ChannelSlice" and "64 channels".

## OutputShapeRejectsARankOnePerSampleShape

`output_shape([16])` throws an error naming "ChannelSlice", "[H, W, C]" and "[16]".

## OutputShapeRejectsARankFourShape

`output_shape([2, 3, 5, 7])` throws `Error`.

## ForwardRejectsARankTwoInput

`forward` over a [2, 12] input throws an error naming "ChannelSlice", "rank-4" and "[2, 12]".

## ForwardRejectsARangePastTheChannels

`forward` with offset 2 and count 2 over a three-channel input throws an error naming
"ChannelSlice", "offset=2, count=2" and "3 channels".

## ForwardRejectsAnNCHWInput

`forward` over a value converted to NCHW throws an error naming "slice_channels" and "NHWC".

## TheParameterWalkVisitsNothing

After a trace, `NN::for_each_parameter` calls its visitor zero times and
`NN::parameter_count` is 0.
