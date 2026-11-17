# tests/language/blocks/test_text_encoder.cpp

Suites: `Small_BlockTextEncoder`, `Small_BlockTextEncoderFactory`, `Medium_BlockTextEncoder`.

Tests `Language::Blocks::TextEncoder` and `make_text_encoder_node` in
`modules/language/blocks/include/SushiAI/language/blocks/text_encoder.hpp`. Every encoder has 11
tokens, a width of 8, two heads, a hidden width of 16, two blocks and a position table of 6 rows. The
numeric cases run on an engine, in `Medium_`, because the pool divides by the length and
`Graph::HostEvaluator` has no rule for `DIV` or `MAXIMUM`. They fill every parameter with the
eighths of `blocks_test_support.hpp`, in walk order, so two encoders of one description hold the same
state.

## ATraceGivesOneRowPerText

A trace over [N, T] tokens and N lengths gives an [N, dim] value and records no failure.

## ATextLongerThanThePositionTableFails

A text of 7 positions over a table of 6 rows fails with both counts and hands back the tokens.

## TheResultShapeIsTheWidth

The per-sample result is [dim], and a per-sample input of rank 2 is refused.

## TheWalkNamesTheTableThePositionsAndEachBlock

The walk yields 26 entries: the token table, the positions, then 12 for each of the two blocks, named
`enc.embedding.weight`, `enc.positions.weight`, `enc.blocks.0.norm1.weight` and so on.

## TheFactoryBuildsANodeOfTwoInputsAndOnePort

The node reads two values and has the one port `output` of shape [dim], and tracing it over [2, 5]
tokens and 2 lengths appends one [2, dim] value.

## TheFactoryRefusesARowThatDoesNotFit

The factory refuses a stray key, a missing `vocabulary`, a missing `blocks`, uneven heads, a `length`
below the tokens, one input, rank-2 tokens and two lengths for one text.

Rows: StrayKey, NoVocabulary, NoBlocks, UnevenHeads, LengthBelowTheTokens, OneInput, RankTwoTokens,
TwoLengthsForOneText.

## APaddedTextEncodesAsTheTextAtItsOwnLength

A text padded to T encodes as the same text at its own length. Oracle: the encoder run alone over
each text, the short one at T of 3 and the full one at T of 5, compared through the tolerance helper
with a normwise bound of 512 operations on the longest path. The two texts share one padded batch.

## ATokenBeyondTheLengthOfItsRowChangesNothing

Changing the tokens past the length of row 0 changes no value of the result, bit for bit: a padded key
scores the fill, `exp` of it is exactly 0, and the pool weighs a padded position 0.
