# tests/language/blocks/test_sequence_pool.cpp

Suites: `Small_BlockSequencePool`, `Small_BlockSequencePoolInput`, `Medium_BlockSequencePool`.

Tests `Language::Blocks::SequencePool` in
`modules/language/blocks/include/SushiAI/language/blocks/sequence_pool.hpp`. The numeric cases run
on an engine, in `Medium_`, because the pool divides by the length and `Graph::HostEvaluator` has no
rule for `DIV` or `MAXIMUM`. The cases read a [2, 5, 4] sequence of dyadic eighths, with lengths of 3
and 5.

## ATraceDropsTheSequenceAxis

A [N, T, dim] value pools to a [N, dim] value and records no failure.

## AnInputThatDoesNotFitFails

A rank-2 sequence, an integer sequence and lengths of the wrong count each fail with
`InvalidDescription` and hand back the sequence.

Rows: RankTwoSequence, IntegerSequence, OneLengthForTwoTexts.

## TheResultShapeIsTheWidthOfAPosition

The per-sample result is [dim], and a per-sample input of rank 1 is refused.

## ARowIsTheMeanOverTheValidPositionsOfItsText

Row n is the mean of the first lengths[n] positions of text n. Oracle: the mean worked on the host in
double precision from the same eighths, compared through the tolerance helper. The first text has two
padded positions, which must add nothing to the sum and count for nothing in the divisor.

## ALengthOfZeroGivesARowOfZeros

A length of 0 gives a row of zeros, and the other row keeps its mean. The zero row compares exactly:
a zero weight times a finite value is 0.

## ThePoolDifferentiatesAtItsSequence

The full-Jacobian gradient check of a cross-entropy over the pooled rows, differentiated at the
sequence, on lengths of 3 and 5.
