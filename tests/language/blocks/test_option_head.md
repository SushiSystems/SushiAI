# tests/language/blocks/test_option_head.cpp

Suites: `Small_BlockOptionHead`, `Small_BlockOptionHeadScore`, `Small_BlockOptionHeadEncode`,
`Medium_BlockOptionHead`, `Medium_BlockOptionHeadMask`.

Tests `Language::Blocks::OptionHead` in
`modules/language/blocks/include/SushiAI/language/blocks/option_head.hpp`. The traces and refusals
run on the graph alone, in `Small_`. The numeric cases run on an engine, in `Medium_`, because the
encoder pools through `DIV` and `Graph::HostEvaluator` has no rule for it, and the gradient cases
differentiate a graph the engine runs. The score cases read a [2, 4] question state and [2, 3, 4]
option states in float64; the encoding cases use an encoder of 11 tokens, a width of 8, two heads, one
block and a position table of 6 rows, with the eighths of `blocks_test_support.hpp` in every
parameter.

## ATraceGivesOneLogitPerOption

`encode` of [2, 3, 4] tokens gives a [2, 3, 8] value and `score` of that value gives [2, 3], with no
failure and no refusal on the graph.

## ForwardEncodesThenScores

`forward` over the question state, the tokens, the lengths and the flags gives the [2, 3] logits.

## AScoreThatDoesNotFitFails

A score of operands that misfit fails with `InvalidDescription` and hands back the options.

Rows: RankThreeQuestion, AnotherWidth, AnotherBatch, FlagsOfAnotherShape, IntegerFlags.

## AnEncodingThatDoesNotFitFails

An encoding that misfits fails with `InvalidDescription` and hands back the tokens.

Rows: RankTwoTokens, FloatTokens, OneLengthForEachText, LongerThanThePositionTable. The last is the
encoder's own refusal, which `encode` hands on unchanged.

## TheResultShapeIsTheOptionCount

The per-sample result is [K] for a [K, T] input, and a per-sample input of rank 1 is refused.

## ALogitIsTheDotProductOfTheQuestionAndTheOption

With every option held, logit (n, k) is the dot product of question n and option (n, k). Oracle: the
product worked on the host in double precision from the same eighths, compared through the tolerance
helper with a bound of four terms of the sum of the absolute products.

## APaddedOptionWeighsZeroAndTheArgmaxIsHeld

For each mask, a padded option holds exactly `K_MASKED_ATTENTION_SCORE`, a softmax over its row
weighs it exactly 0, the weights of the held options sum to 1 within the tolerance of the helper and
the argmax of each row is a held option. The point gives every padded option the largest score of its
row, 5/4 against a held score of at most -1/8, so a mask that failed to replace it would be caught by
the argmax.

Rows: NothingPadded, LastPadded, FirstPadded, OneHeld. A row with no held option is outside the
contract: the softmax of its fill is uniform.

## TheScoreDifferentiatesAtTheQuestionAndTheOptions

The full-Jacobian gradient check of a cross-entropy over the logits, at the question state and the
option states together, with one padded option in the second question.

## APaddedOptionReceivesNoGradient

The gradient at the channels of a padded option is exactly 0: the select passes the held logits their
gradient and the padded ones none.

## AnOptionEncodesAsItsTextAlone

An option padded to T encodes as the same text run alone at its own length. Oracle: the encoder run
over the one option, option (0, 0) at 2 of 4 tokens and option (1, 2) at 4 of 4, compared through the
tolerance helper with a normwise bound of 512 operations on the longest path.
