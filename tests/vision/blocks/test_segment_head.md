# tests/vision/blocks/test_segment_head.cpp


Suites: `Unit_BlockSegmentHead`.

Tests `Vision::Blocks::SegmentHead` and `make_segment_head_node` in
`modules/vision/blocks/include/SushiAI/vision/blocks/segment_head.hpp`. Every head takes 16
input channels and five classes, with an epsilon of 1e-3 and a momentum of 0.03 for its
normalisation.

## OutputShapeScalesTheExtentsAndSetsTheClasses

For the input [6, 10, 16], a head of scale 4 reports [24, 40, 5] and a head of scale 1 reports
[6, 10, 5].

## TheTraceEndsAtTheUpsample

A training trace of a head with 8 hidden channels and scale 2 over [2, 4, 4, 16] holds one
BATCHNORM2D, one SILU and one UPSAMPLE2D. The result has the shape [2, 8, 8, 5] and the
UPSAMPLE2D node produces it.

## StateIsListedUnderCv1AndCv2

After a training trace the state walk gives seven records in this order: `cv1.conv.weight`
[8, 3, 3, 16], `cv1.bn.weight`, `cv1.bn.bias`, `cv1.bn.running_mean` and `cv1.bn.running_var`,
each [8], then `cv2.weight` [5, 1, 1, 8] and `cv2.bias` [5].

## TheFactoryBuildsTheRow

Over the input [4, 4, 16], the row with `classes` 5, `hidden` 8 and `scale` 2 gives a node with
one port of shape [8, 8, 5]. The row with only `classes` 5 gives one port of shape [4, 4, 5];
after an evaluation trace it has seven pieces of state, the first being `cv1.conv.weight`
[16, 3, 3, 16].

## TheFactoryRefusesARowThatDoesNotFit

`make_segment_head_node` fails with `InvalidDescription` for a row with the key `mode`
("layer 3 'segment_head'", "'mode'"), for a row without `classes` ("'classes'", "absent") and
for `scale` 0 ("'scale'", "positive").
