# tests/vision/blocks/test_psa_block.cpp


Suites: `Small_BlockPsaBlock` and `Medium_BlockPsaBlockDeath`.

Tests `PSABlock` at the width of row 10 of YOLO26n, 128 channels over two heads, traced in
training mode over a [2, 4, 4, 128] input.


## OutputShapeIsTheInputShape

`output_shape([20, 20, 128])` is [20, 20, 128].

## ATrainingTraceKeepsTheInputShapeAndUpdatesEveryNorm

A training trace yields the shape [2, 4, 4, 128] and requests two updates, a mean and a variance, for each norm the state walk lists.

## TheInputIsAddedBackOnceAndTheResultIsAResidualSum

Exactly one ADD node reads the block's input as its first operand, and the value `forward` returned is produced by an ADD.

## StateMatchesManifestRow10Block

Walked under the prefix "model.10.m.0", the traced block yields 25 records in order: five each
for `attn.qkv`, `attn.proj`, `attn.pe`, `ffn.0` and `ffn.1`. `ffn.0` has a [256, 1, 1, 128]
weight and `ffn.1` a [128, 1, 1, 256] weight.

## AWidthTheHeadCountDoesNotDivideIsFatal

`PSABlock(100, 3)` stops the process naming "head count divides", "100 channels" and "3 heads". This is a death test, so it is `Medium_BlockPsaBlockDeath`. The rule itself runs small in `test_attention.cpp` as `check_attention_heads`.
