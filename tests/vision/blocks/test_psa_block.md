# tests/vision/blocks/test_psa_block.cpp


Suites: `Unit_BlockPsaBlock`.

Tests `PSABlock` at the width of row 10 of YOLO26n, 128 channels over two heads, traced in
training mode over a [2, 4, 4, 128] input.

## OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

## EmitsThreeAddsOneSiLUFiveNormsAndTenUpdates

The trace holds three ADD nodes, one SILU node and five BATCHNORM2D nodes, and the buffer
update list holds ten updates. The result has the shape [2, 4, 4, 128].

## TheResidualAddsReadTheInputThenAttentionThenTheFeedForward

The second ADD of the graph reads the block's input and a BATCHNORM2D result. The third ADD
reads the second ADD's result and a BATCHNORM2D result, and its output is the value `forward`
returned.

## StateMatchesManifestRow10Block

Walked under the prefix "model.10.m.0", the traced block yields 25 records in order: five each
for `attn.qkv`, `attn.proj`, `attn.pe`, `ffn.0` and `ffn.1`. `ffn.0` has a [256, 1, 1, 128]
weight and `ffn.1` a [128, 1, 1, 256] weight.

## AWidthTheHeadCountDoesNotDivideThrows

`PSABlock(100, 3)` throws an error naming "head count divides", "100 channels" and "3 heads".
