# tests/vision/blocks/test_focus.cpp

Suites: `Small_BlockFocus`. Every case traces on the host and runs on `Graph::HostEvaluator`
where it compares numbers, so no case waits on a device, a file, a thread or a clock.

Tests `Focus::stack_phases`, `Focus`'s trace and state names, and `make_focus_node`. The YOLOX
order is top-left, bottom-left, top-right, bottom-right.

## TheStackFollowsTheYoloxPhaseOrder

A 1x4x4x1 input holding 1 to 16 row by row stacks to the 1x2x2x4 value
1 5 2 6, 3 7 4 8, 9 13 10 14, 11 15 12 16. The window of rows 0 and 1 and columns 0 and 1 holds
1 2 over 5 6, read as top-left 1, bottom-left 5, top-right 2, bottom-right 6. The values are
integers, compared exactly.

## APhaseKeepsItsChannelsTogether

A 1x2x2x2 input of 1 to 8 stacks to 1 2 5 6 3 4 7 8: each phase carries both of its channels.

## EachImageOfABatchStacksOnItsOwn

A 2x2x2x1 input of 1 to 8 stacks to 1 3 2 4 for the first image and 5 7 6 8 for the second.

## TheStackHalvesTheExtentsAndQuadruplesTheChannels

A [2, 6, 8, 3] input stacks to [2, 3, 4, 12].

## TheStackRefusesAnOddExtent

An odd height and an odd width each fail with `InvalidDescription`, "even H and W" and the shape.

## TheOutputShapeOfTheYoloxXStem

`Focus(3, 80, 3)` maps [640, 640, 3] to [320, 320, 80]. 80 is the width of the YOLOX-X stem.

## TheOutputShapeRefusesAnOddExtent

[639, 640, 3] fails with "even H and W" and the shape.

## ATracedStemYieldsTheOutputWidthAtHalfTheExtents

A traced stem over [2, 8, 8, 3] with 16 filters yields [2, 4, 4, 16].

## ATracedStemOverAnOddInputFailsTheTrace

A stem traced over [1, 5, 4, 3] leaves a failed trace status naming "even H and W".

## StateHoldsTheConvolutionOverTwelveChannels

Walked under "stem", the stem lists "stem.conv.conv.weight" of [80, 3, 3, 12] and the four norm
records of 80, the names of YOLOX's `stem.conv`.

## TheFactoryBuildsTheStemWithAKernelOfThree

A row of `{"filters": 16}` over [8, 8, 3] has one port of [4, 4, 16], and its first record is
"conv.conv.weight" of [16, 3, 3, 12]: the kernel is 3 when absent.

## TheFactoryRefusesAnEvenKernel

`{"filters": 16, "kernel": 2}` fails with "layer 0 'focus'", "'kernel'", "odd" and "2".

## TheFactoryRefusesAnOddInput

An input of [7, 8, 3] fails with "even H and W" and the shape.

## TheFactoryRefusesAnUnknownKey

An unknown key "stride" fails with "layer 0 'focus'" and "'stride'".

## TheFactoryRefusesAnAbsentFilterCount

A row without "filters" fails with "'filters'" and "absent".
