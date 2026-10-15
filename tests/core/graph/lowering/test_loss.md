# tests/core/graph/lowering/test_loss.cpp


Suites: `Medium_LogSoftmaxStability`.

This file tests a log-softmax that was stabilised by the wrong maximum.

Shifting the whole batch by the batch's global maximum rests on the argument that
log-softmax is invariant to subtracting any single scalar from a row. The argument is sound in real
arithmetic and false in binary32. The exponential destroys the invariance, not the subtraction: a
row sitting a long way below the batch maximum has every `exp(x - global_max)` collapse into the
subnormals and then to zero, its row sum becomes exactly zero, `log(0)` is `-inf`, and the row's
log-probabilities come back `+inf`.

The figures in the table were measured on this project's OpenCL CPU device, in FLOAT32, on the
[2, 3] input `logits_with_gap` builds, whose true log-probabilities are about -0.41 to -2.41.

The quantity is the worst absolute deviation of row 1's log-probabilities from the answer the same
code gives at a gap of 0. It is the one figure measured directly on both sides of the fix, and the
assertion that produces it is the EXPECT_EQ in TheAnswerDoesNotDependOnTheGapAtAll:

| row-to-batch-max gap | global shift (before) | per-row shift (after) |
|---------------------:|----------------------:|----------------------:|
|                   90 |              2.15e-06 |           0 (exactly) |
|                  100 |              2.70e-02 |           0 (exactly) |
|                  104 |                   inf |           0 (exactly) |
|                  200 |                   inf |           0 (exactly) |

Measured against the hand-computed expectation, and not against the gap-0 answer, the gap-100 case
came out 2.7033e-02 off. That is 6.6% of the largest true log-probability (-0.40761), and it is the
"about 7% low at 100" that SushiBLAS's GPU_VALIDATION.md recorded for this defect before the fix.

The residual at the gaps that pass is absent from both columns, on both sides of the fix, because
it was never measured. A passing EXPECT_NEAR bounds it under 1e-4 and says nothing more. Do not
quote a figure for it. An earlier draft of the file header did, and the invented numbers survived a
full run unchallenged because nothing in the suite fails when a comment is wrong.

The file is a regression test and not an integration test because of how the defect fails. `+inf`
log-probabilities do not stop a run: the loss becomes `inf` or `nan`, the optimizer writes `nan`
parameters, and the training loop reports completion having learnt nothing. Nothing throws, and no
test that only checks "did it run" notices. This tier exists for that failure mode.

The expectations are hand-computed and not read off another path in this repository. Each row is
{a, a+1, a+2}, whose log-softmax is {-2 - k, -1 - k, -k} with k = ln(1 + e^-1 + e^-2), for any a.
The expected values are therefore the same for every gap under test, and any dependence on `a` is
the defect showing.

## ARowFarBelowTheBatchMaximumStaysFiniteAndCorrect

This test reproduces the defect itself. At a gap of 104 the old code returned `+inf` for row 1. At
100 it was 6.6% low on the largest log-probability (-0.43464 measured against a true -0.40761). At
90 it had drifted 2.15e-06 off its own gap-0 answer. The per-row maximum removes the gap from the
arithmetic, so the answer is the same at every gap, and the single shared expectation asserts that.

The tolerance in this test is 1e-4, which is wide enough to pass at gap 90 both before and after
the fix. That is intended. This case exists to catch `+inf` and gross error.
TheAnswerDoesNotDependOnTheGapAtAll makes the finer claim, that the gap changes nothing, at exact
equality.

## CrossEntropyOverSuchARowIsFiniteAndCorrect

This test checks the same defect one node downstream. Cross-entropy is `-sum(L * Y) / N` over the
log-probabilities, so an infinite row makes the loss infinite. The loss is the number the training
loop prints and the number every stopping rule reads.

The expectation is hand-computed: with the target on class 0 of both rows, the loss is
`-(2 * (-2 - k)) / 2 = 2 + k`, whatever the gap.

## TheCrossEntropyGradientOverSuchARowIsFinite

This is the failure that destroys a run, where the two earlier tests cover a bad reported number.
The fused cross-entropy gradient is `(exp(L) - Y) / N`, so an infinite `L` makes the gradient
infinite, the optimizer writes non-finite parameters, and every later step trains on NaN while the
loop reports success.

The expectation is hand-computed: `exp(L)` is the softmax of {a, a+1, a+2}, which is
{e^-2, e^-1, 1} / (1 + e^-1 + e^-2), the same for both rows. The test subtracts the one-hot target
and divides by the batch size of 2.

## TheAnswerDoesNotDependOnTheGapAtAll

This test states the property that makes the fix a fix and not a wider threshold: nothing in the
answer depends on the gap. Two batches differing only by a constant offset applied to one row must
produce identical log-probabilities for the row that did not move, and identical ones for the row
that did.

A shift by the global maximum fails this at any gap large enough to matter. A shift by each row's
own maximum passes it at every gap, because the gap never enters the arithmetic.

The test asserts equality and not a tolerance, and the arithmetic justifies it. Every logit in the
test is a small integer, so each row's own maximum is exact, the shift `x - row_max` is exact, and
both rows are reduced to the same three floats {-2, -1, 0} before anything transcendental happens.
The gap cannot reach the answer, so any difference is a real one. A tolerance here would have been
the same mistake in miniature: a threshold wide enough to hide the thing under test.
