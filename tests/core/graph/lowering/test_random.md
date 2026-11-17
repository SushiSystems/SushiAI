# tests/core/graph/lowering/test_random.cpp


Suites: `Small_DropoutOps`, `Small_DropoutSeed`, `Small_DropoutRate`, `Medium_DropoutDraw`.

Tests `GraphBuilder::dropout`: the node it emits, the seed it stores, the operands it refuses
and its row in the lowering table. The accepted cases pass the rate 0.25. The `Medium_DropoutDraw`
cases compile the node on an engine: its row records one position prelude and one Bernoulli draw
that reads the position, and the oracle is a relation between runs and between stream seeds. An
engine is made once per program, so "two engines seeded alike" is one engine whose stream is
reseeded to the same value before each step runs; the position holds the stream's seed, read at
each run, and the run index.

## DropoutWritesItsValueAndTheMaskItSaves

Over a [2, 3, 4] float64 input the call is ok and emits one DROPOUT node that reads the input
and writes the output and then the saved mask. The node holds 0.25 in `K_DROPOUT_RATE_SLOT`.
Both outputs have the shape [2, 3, 4], the type float64 and the lifetime `Activation`, and the
graph has no refusal.

## TheSiteSeedSurvivesItsAttributeSlots

One row per seed: 0xFEDCBA9876543210, 0 and all 64 bits set. `Graph::dropout_seed` of the emitted
node's attributes returns the seed that was passed.

## DropoutRefusesAnInputOfAnIntegerType

An int32 input fails with `InvalidDescription` and "half, float32 or float64 input". The node
count is unchanged and `has_refusal()` is true.

## DropoutRefusesARateOutsideZeroToOne

One row per rate: 0, 1 and -0.5. Each fails with `InvalidDescription` and "rate in (0, 1)". The
node count is unchanged and `has_refusal()` is true.

## DropoutHasTheRowOfTheRandomTable

`Graph::K_RANDOM_LOWERINGS` has one row, for DROPOUT. `find_lowering` returns
`Graph::lower_dropout`, `lowering_row_count` is 1 and `is_unlowered` is false.

## OneCompiledStepDrawsADifferentMaskAtEachRun

A dropout over a [64, 64] leaf is compiled once and run twice. The two saved masks differ,
`compile_count()` is 1 and the step holds one replay prelude.

## TwoStepsOnAStreamSeededAlikeGiveEqualMasksAtEqualRunIndices

Two steps of one graph shape run twice each, the stream reseeded to the same value before each
step. The masks of run 0 are equal, and so are those of run 1.

## AnotherStreamSeedGivesAnotherMaskAtTheSameRunIndex

Two steps run once under stream seeds that differ by one. Their masks at run 0 differ.

## TheKeptFractionOfALargeMaskIsTheKeepProbability

The fraction of non-zero elements of a [1024, 1024] mask at rate 0.25 is within ten standard
deviations of 0.75. One deviation of that fraction is at most 0.5 / 1024, so the bound is 0.00488,
under the 0.005 the design names; the seed is fixed, and a fair draw fails it with a rate below 1e-22.
