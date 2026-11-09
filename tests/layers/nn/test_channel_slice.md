# tests/layers/nn/test_channel_slice.cpp


The ChannelSlice layer's shape inference, the one SLICE_CHANNELS node it emits with its offset, the offsets, counts and ranks it rejects, and its empty state.

Suites: `Small_ChannelSlice`, `Small_ChannelSliceAccepted`, `Small_ChannelSliceRefused`,
`Small_ChannelSliceForwardRefusal`.

Tests `NN::ChannelSlice` in `modules/layers/nn/include/SushiAI/nn/channel_slice.hpp`, which
keeps `count` channels of an image batch from `offset` on.

Two `static_assert` lines at the top of the file hold that `ChannelSlice` is a `Module` and is
not a `ModuleContainer`.

## Small_ChannelSliceAccepted.OutputShapeTakesTheRange

`output_shape` keeps height and width and takes the range of channels. The rows are AMiddleRange
(offset 32, count 32 over [5, 3, 64] gives [5, 3, 32]) and, over [4, 2, 4], TheFirstChannel
(0, 1), TheLastChannel (3, 1), AllButTheFirst (1, 3) and TheWholeRange (0, 4).

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

## Small_ChannelSliceRefused.OutputShapeRefuses

`output_shape` returns `InvalidDescription` and the message holds the row's substrings. Over
[5, 3, 64] the rows are ANegativeOffset (-1, 4), AZeroCount (0, 0), ANegativeCount (8, -2),
ARangePastTheChannels (32, 40) and AnOffsetAtTheChannelCount (64, 1), each naming "ChannelSlice",
its "offset=..., count=..." and "64 channels"; AnExtremeOffsetWithoutOverflow and
AnExtremeCountWithoutOverflow (the largest `int64_t` as offset or count) name "ChannelSlice" and
"64 channels", so the range test does not overflow. ARankOnePerSampleShape ([16]) names
"ChannelSlice", "[H, W, C]" and "[16]". ARankFourShape ([2, 3, 5, 7]), a shape that still carries
its batch axis, fails with the code alone.

## Small_ChannelSliceForwardRefusal.ForwardRefuses

`forward` returns `InvalidDescription` and the message holds the row's substrings. The rows are
ARankTwoInput ([2, 12]: "ChannelSlice", "rank-4", "[2, 12]"), ARangePastTheChannels (offset 2,
count 2 over [1, 2, 2, 3]: "ChannelSlice", "offset=2, count=2", "3 channels") and AnNCHWInput
(a [1, 2, 2, 3] input converted to NCHW: "slice_channels", "NHWC").
