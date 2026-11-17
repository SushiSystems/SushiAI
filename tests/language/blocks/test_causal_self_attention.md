# tests/language/blocks/test_causal_self_attention.cpp

Suites of the key-padding cases: `Small_BlockCausalSelfAttentionLengths`,
`Small_BlockCausalSelfAttentionLengthsInput` and `Small_BlockCausalSelfAttention`. The other cases
of the file are described in `tests/language/blocks/README.md`.

Tests `forward_with_lengths` of `CausalSelfAttention` and of `TransformerBlock`, in
`modules/language/blocks/include/SushiAI/language/blocks/`. A length per sequence keeps every key at
or past it out of the softmax. Every case runs on `Graph::HostEvaluator` over the state of
`blocks_test_support.hpp`, with a width of 8, two heads and a padded batch of two texts of 5
positions: a short text of 3 tokens with two padded positions, and a text that fills its sequence.

## APaddedTextGivesTheRowsOfTheTextAtItsOwnLength

Promise: a text padded to T gives the same rows as the same text at its own length. Oracle: the block
run alone over each text without lengths, a text of 3 positions and a text of 5, compared through the
tolerance helper. Rows: `None`, `Causal`. The batch holds both texts, so the condition has to read
the length of its own row.

## APaddedTextGivesTheRowsOfTheTextAtItsOwnLengthThroughABlock

The same promise through a `TransformerBlock` without the causal mask, which routes the lengths to
its attention and leaves the normalisations and the feed-forward to read each row alone.

## APaddedValueChangesNoRowWithinItsLength

Two batches that differ only in the padded rows give the same rows within the length, bit for bit:
a padded key scores the fill, and `exp` of the fill is exactly 0. Rows: `None`, `Causal`.

## LengthsThatDoNotFitFail

Lengths that do not fit are refused with `InvalidDescription`, the trace hands back its input and
emits no `SELECT`.

Rows: TwoLengthsForOneText, IntegerLengths.

## AHalfBlockFillsAPaddedKeyWithTheLowestFiniteHalf

A half block fills a padded key with -65504, the lowest finite half. The fill of -1e9 the causal mask
writes is minus infinity in half, and a row of padded keys alone would then softmax to NaN.

## AFloat32BlockFillsAPaddedKeyWithTheMaskedScore

A float32 block fills a padded key with `K_MASKED_ATTENTION_SCORE`, the score the causal mask uses.
