# tests/language/blocks

The tests of the module `language/blocks`, one file per header of `modules/language/blocks/include/`. The suite prefix of a case gives its ctest label.

Where a case compares numbers, it runs the traced graph on `Graph::HostEvaluator` over the state
that `blocks_test_support.hpp` writes (`write_state`, `eighths`), so no engine starts. A case that
needs an engine or a stopped process runs in a `Medium_` suite.

## test_causal_self_attention.cpp

Suites: `Small_BlockCausalSelfAttention`, `Small_BlockCausalSelfAttentionInput`,
`Small_BlockCausalSelfAttentionDescription`, `Small_BlockCausalSelfAttentionFactory`,
`Small_BlockCausalSelfAttentionCache`, `Small_BlockCausalSelfAttentionRotary`,
`Small_BlockCausalSelfAttentionPositions`, `Medium_AttentionGradient`, `Medium_CausalAttentionMask`,
`Medium_CausalAttention`.

Tests `Language::Blocks::CausalSelfAttention` in
`modules/language/blocks/include/SushiAI/language/blocks/causal_self_attention.hpp` and its node
factory `make_causal_self_attention_node`: the ordinary trace, the incremental trace of one
position, the rotary form of both, the state walk and the refusals, then the gradient of the
attention and the causal mask on an engine. Every case uses a width of 8 and two heads. Most
ordinary traces read a [2, 5, 8] input, and the incremental ones read a [1, 1, 8] input over a
per-sample shape of [5, 8]. The one numeric `Small_` case reads [1, 3, 8] and compares an
unmasked block with the attention worked on the host; the mask has no host kernel, so every
case that computes through it is a `Medium_` case.

### ACausalTraceKeepsTheSequenceShape

A causal trace keeps the [N, T, dim] shape and records no failure.

### TheScoresOfEachHeadAreScaledByTheRootOfItsWidth

The scores of each head are scaled by one over the square root of the head width: 0.5 for a width of
8 and two heads. The block has no mask, so the host evaluator runs it. The reference is the
attention worked on the host from the same state and the same rows.

### ATraceWithoutAMaskEmitsNone

A block without a mask adds no CAUSAL_MASK node.

### TheWalkYieldsTheTwoProjections

`qkv` is [3 * dim, dim], proj [dim, dim], and the walk names the four.

### AnInputThatDoesNotFitFails

A rank-2 input, another width and an integer input fail and emit nothing.

Rows: RankTwo, AnotherWidth, Int32.

### ABlockDescribedWrongFails

A head count that does not divide the width, and no width, each fail.

Rows: UnevenHeads, EmptyBlock.

### TheFactoryReadsTheHeadsAndTheMask

The factory takes dim from the input and reads "heads" and "causal".

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses a stray key, no heads, uneven heads and a bad input.

Rows: StrayKey, NoHeads, UnevenHeads, RankThreeInput.

### TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow

One position keeps the output shape and records no failure.

### TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues

The block declares the position, the mask and two carried [T, dim] leaves.

### TwoBlocksOfOneTraceShareTheirFedLeaves

Two blocks of one trace share the fed leaves and carry four values.

### TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm

The incremental form declares the state the ordinary form declares.

### TheIncrementalFormRefusesWhatItCannotCache

An unmasked block, a longer slice, a plain trace and a flat sample fail.

Rows: NoMask, ThreePositions, PlainTrace, FlatSample.

### ARotaryTraceTurnsTheQueriesAndTheKeysOnce

Rotary positions keep the shape and fill one table of cosines and of sines.

### ARotaryBlockHoldsTheStateOfALearnedOne

The walk of a rotary block yields the two projections and no table.

### ARotaryTraceRefusesWhatItCannotTurn

A trace no step fills tables for and an odd head are each refused.

Rows: NoTables, OddHeadWidth.

### TheFactoryReadsThePositions

The factory reads "positions", learned or rotary, into the layer.

### TheFactoryRefusesWhatItCannotRead

The factory refuses an unknown word, a non-string and an odd head width.

Rows: UnknownWord, NotAString, OddHeadWidth.

### TheRotaryIncrementalFormReplacesARowOfEachRing

One rotary position keeps the output shape and carries two windows.

### TheRotaryIncrementalFormIsFedASlotAndTheAngles

A rotary position is fed its slot, the mask and its angles, no position.

### AnUnfusedLogSoftmaxMatchesItsClosedForm

The unfused rule against its closed form, with targets that are not one-hot. A SCALE by one sits
between LOG_SOFTMAX and NLL_LOSS, so the fused rule does not apply and the gradient is (softmax(x) *
rowsum(Y) - Y) / N, which the host computes.

### ASoftmaxGradientMatchesCentralDifferences

The gradient of a [3, 4] softmax that a matrix product reads.

### ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes

The gradient of a softmax over the last axis of a [2, 3, 4] value.

### ACausalAttentionDifferentiatesAtItsInput

The gradient of a causal attention block at the sequence it reads.

### ACausalAttentionDifferentiatesAtItsProjectionWeight

The gradient of a causal attention block at the weight of its qkv.

### AnUnmaskedAttentionDifferentiatesAtItsInput

The gradient of an attention block without the mask at the sequence it reads.

### TheMaskReplacesTheEntriesAboveTheDiagonal

The mask keeps the lower triangle and writes the fill above, in FLOAT32 and in FLOAT64, where the
mask vectors are converted from FLOAT32.

### TheMaskPassesItsGradientThroughTheKeptEntriesAlone

The gradient of the scores is the incoming gradient on and below the diagonal and zero above it. The
loss is the cross-entropy of the masked [4, 4] rows, so the incoming gradient is (softmax - target)
/ 4, which the host computes from the masked scores.

### AnEarlierPositionIgnoresALaterInput

Positions 0 to 2 of a masked attention keep their output exactly.

### WithoutTheMaskAnEarlierPositionReadsALaterInput

Without the mask an earlier position does read the later input.

### ATransformerBlockKeepsEarlierPositions

A whole causal transformer block keeps the same property.

## test_expert_transformer_block.cpp

Suites: `Small_BlockExpertTransformerBlock`, `Small_BlockExpertTransformerBlockOnePort`,
`Small_BlockExpertTransformerBlockFactory`.

Tests `Language::Blocks::ExpertTransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/expert_transformer_block.hpp`, and the
`transformer_block` row of `make_transformer_block_node` when it states `experts`. The block of
every case is causal, has a width of 8, two heads and four experts of hidden width 16.

### ATraceAddsANormalisedAttentionThenANormalisedMixture

A trace yields its three values and adds the mixture to the first sum.

### TheWalkYieldsTheFourChildren

The walk names the fourteen parameters under the four children.

### AnInputThatDoesNotFitFails

An input of another width fails once, with the first child's message.

### TheTransformerRowWithExpertsHasThreePorts

A row with experts builds a node of the output, balance and usage ports.

### ARowWithoutExpertsHasOnePort

A row without experts, or with none, builds a node of one port.

Rows: NoExpertsKey, ZeroExperts.

### TheExpertRowWritesThreeOutputsOfTheirShapes

The node of a row with experts writes three outputs of their shapes.

### AnExpertRowTakesFourTimesTheWidthAsHidden

A row leaving out "hidden" gives each expert four times the width.

### TheTransformerRowRefusesCountsThatDoNotFit

The row refuses a top_k without experts, above them and negative experts.

Rows: TopKWithoutExperts, TopKAboveExperts, NegativeExperts.

### TheIncrementalFormTracesEachChildOverOnePosition

One position yields its output and usage and carries the two caches.

### TheIncrementalRowWritesThreeOutputs

The node of an expert row writes three outputs over one position.

### TheBlockPassesThePositionsToTheAttention

The block hands learned or rotary positions to its attention.

### ARotaryExpertRowDeclaresTwoTables

The one attention of a rotary expert row declares two host tables.

### ARotaryExpertRowOverAnOddHeadWidthIsRefused

A rotary expert row over an odd head width is refused.

## test_feed_forward.cpp

Suites: `Small_BlockFeedForward`, `Small_BlockFeedForwardInput`, `Small_BlockFeedForwardFactory`.

Tests `Language::Blocks::FeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/feed_forward.hpp` and its node factory
`make_feed_forward_node`. Every case uses a width of 6 and a hidden width of 24.

### ATraceKeepsTheSequenceShape

A trace keeps the [N, T, dim] shape and records no failure.

### TheWalkYieldsTheTwoLinearLayers

The walk names the two linear layers, expand before project.

### AnInputThatDoesNotFitIsRefused

An input of another rank or width fails and leaves the graph as it was.

Rows: RankTwo, WidthSeven.

### ABlockOfNoWidthIsRefused

A block of no width fails the trace and names both widths.

### TheFactoryReadsTheHiddenWidth

The factory reads "hidden" into a one-port node of the input's shape.

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses a row without "hidden" and a row with a stray key.

Rows: NoHidden, StrayKey.

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

The incremental output of one position is the ordinary output at it. Both graphs run on the host
evaluator over the same state and the same rows, so the row of the position is computed by the same
operations in the same order and compares exactly.

### TheIncrementalFormFeedsAndCarriesNothing

The incremental form declares no fed leaf and carries no value.

## test_language_head.cpp

Suites: `Small_BlockLanguageHead`, `Small_BlockLanguageHeadTable`, `Small_BlockLanguageHeadFactory`,
`Small_BlockLanguageHeadTie`, `Medium_BlockLanguageHeadDeathTest`.

Tests `Language::Blocks::LanguageHead` in
`modules/language/blocks/include/SushiAI/language/blocks/language_head.hpp` and its node factory
`make_language_head_node`, with and without a weight tied to an embedding table. Every case
uses a width of 6 and a vocabulary of 11.

### ATraceNormalisesThenMapsToTheVocabulary

A trace maps [N, T, dim] to [N, T, vocabulary] and declares no bias.

### TheWalkYieldsTheNormalisationAndTheWeight

The walk names the normalisation's two parameters and the one weight.

### AnInputOfAnotherWidthIsRefused

An input of another width fails and leaves the graph as it was.

### AHeadOfNoWidthIsRefused

A head of no width fails the trace and names both widths.

### ATiedHeadScoresWithTheEmbeddingsTable

A tied head scores the normalised rows with the table, transposed. Each row has mean 0 and variance
1, so with a gain of 1, a bias of 0 and an epsilon of 3 it normalises to half of itself, and every
logit is exact.

### ATiedHeadDeclaresNoWeightOfItsOwn

A tied head keeps the [N, T, vocabulary] shape and declares no weight.

### ATiedTableThatDoesNotFitFails

A table no layer has declared, or of another shape, fails the trace.

Rows: Undeclared, Narrow.

### OutputShapeReplacesTheWidthByTheVocabulary

`output_shape` replaces the width of a [T, C] shape by the vocabulary.

### TheFactoryReadsTheVocabulary

The factory reads "vocabulary" and "ln_epsilon" into a one-port node.

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses an absent vocabulary and a rank-one input.

Rows: NoVocabulary, RankOne.

### TheFactoryTiesTheHeadToAnEmbeddingRow

"tie" binds the head to the table of the embedding row it names.

### TheFactoryRefusesATieTheRowsCannotServe

A "tie" the rows above cannot serve is refused with its reason.

Rows: NotAnEmbedding, OtherVocabulary, NotAbove.

### ATieWithNoRowsAboveIsRefused

A tie with no rows above the head is refused.

### OutputShapeOverARankOneSampleIsFatal

`output_shape` over a per-sample shape of rank one stops the process.

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

The incremental logits of one position are the ordinary logits at it. Both graphs run on the host
evaluator over the same state and the same rows, so the row of the position is computed by the same
operations in the same order and compares exactly.

### TheIncrementalFormFeedsAndCarriesNothing

The incremental form declares no fed leaf and carries no value.

## test_moe_feed_forward.cpp

Suites: `Small_BlockMoeFeedForward`, `Small_BlockMoeFeedForwardTopK`,
`Small_BlockMoeFeedForwardFactory`.

Tests `Language::Blocks::MoEFeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/moe_feed_forward.hpp` and its node
factory `make_moe_feed_forward_node`. Every case uses a width of 6 and four experts of hidden
width 8. A `top_k` of 0 keeps every expert.

### ATraceMixesEveryExpertByItsGate

A trace yields the output, the balance and the usage in their shapes.

### KeepingTheTopScoresKeepsTheSequenceShape

A block keeping the top scores keeps the sequence shape, one row per count.

### TheWalkYieldsTheRouterAndTheStackedLayers

The walk names the router, then the two stacked layers of the experts.

### AnInputOfAnotherRankIsRefused

An input of another rank fails and leaves the graph as it was.

### ABlockOfNoWidthIsRefused

A block of no width fails the trace and names its three counts.

### ATopKAboveTheExpertsIsRefused

`trace_experts` refuses a top_k above the experts and adds no node.

### TheFactoryBuildsANodeOfThreePorts

The factory builds a node of the output, balance and usage ports.

### TheNodeWritesThreeOutputsOfTheirShapes

The node writes the output, the balance and the usage in their shapes.

### ARefusedInputLeavesTheOutputsAsTheyWere

The node leaves its outputs as they were when it refuses an input.

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses a row whose keys or counts do not fit.

Rows: NoExperts, NoHidden, StrayKey, TopKAboveExperts, NegativeTopK, FractionalTopK.

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

The incremental output of one position is the ordinary output at it. Both graphs keep one expert and
run on the host evaluator over the same state and rows, so the row of the position is computed by
the same operations in the same order and compares exactly.

### TheIncrementalFormFeedsAndCarriesNothing

The incremental form declares no fed leaf and carries no value.

## test_position_embedding.cpp

Suites: `Small_BlockPositionEmbedding`, `Small_BlockPositionEmbeddingTable`,
`Small_BlockPositionEmbeddingFactory`, `Small_BlockPositionEmbeddingWindow`.

Tests `Language::Blocks::PositionEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/position_embedding.hpp` and its node
factory `make_position_embedding_node`. The sequence of every case has five positions and a
width of 6.

### ATraceAddsTheTableToEveryRow

A table of T rows adds row t to position t of every sample.

### AShorterSequenceReadsTheLeadingRows

A sequence shorter than the table adds the table's leading rows.

### ATableThatCannotFitTheInputIsRefused

A table that cannot fit the input fails and stays undeclared.

Rows: FewerRows, OtherWidth, NoExtent.

### TheWalkYieldsTheTable

The walk names the one table.

### AnAbsentLengthTakesThePositions

An absent "length" takes T and gives a one-port node of the input.

### AnExplicitLengthIsRead

An explicit "length" above T is read into the layer.

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses a length below T, a stray key and a rank-one input.

Rows: ShorterLength, StrayKey, RankOne.

### TheIncrementalFormAddsTheRowOfTheFedPosition

One position adds the table row of the position fed to it.

### TheIncrementalFormIsFedItsPositionAlone

One position is fed an int32 [1, 1] index and carries no value.

### TheIncrementalFormRefusesAWindowItsTableCannotServe

A window longer than the table and a longer slice are refused.

Rows: LongerWindow, LongerSlice.

## test_rotary_embedding.cpp

Suites: `Small_BlockRotaryEmbedding`, `Small_BlockRotaryEmbeddingTable`,
`Small_BlockRotaryEmbeddingWidths`, `Small_RotaryEmbeddingTurn`, `Small_RotaryEmbedding`,
`Medium_RotaryEmbedding`.

Tests `Language::Blocks::RotaryEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_embedding.hpp` and
`Language::Blocks::check_rotary_widths`. Every case uses a width of 8 cut into two groups, so
each group has two pairs. The ordinary traces read five positions.

### ATraceWeighsTheValueAndItsSwappedHalves

A trace keeps the shape and collects a table of cosines and of sines.

### TheTablesAreLeavesOfARowPerPosition

The two tables are float32 leaves of a row of pairs per position.

### TheTablesHoldTheAnglesOfEveryPosition

A table entry holds the cosine and the sine of its hand-worked angle.

Rows: PositionZero, PositionOnePairZero, PositionTwoPairZero, PositionFourPairOne.

### ADoubleValueReadsDoubleTables

A float64 value reads float64 tables.

### TheBlockHoldsNoState

The block lists no parameter and no buffer.

### TheIncrementalFormIsFedTheAnglesOfItsPosition

One position is fed one row of cosines and one of sines, and no table.

### TheWidthCheckAcceptsWholePairs

The width check accepts a width of whole pairs in every group.

### TheWidthCheckRefusesWidthsWithoutWholePairs

The width check refuses widths that leave no whole pair.

Rows: OddGroup, UnevenGroups, NoWidth.

### AnInputOfAnotherWidthIsRefused

An input of another width fails and collects no table.

### ATraceWithoutTablesIsRefused

A trace that collects no tables is refused, naming the cosine table.

### ANonCarryingIncrementalTraceIsRefused

An incremental trace that carries nothing is refused.

### ATurnMatchesTheHostFormula

Each turned element of three rows against its hand-worked value. The rows are (1, 2, 3, 4), (1, -1,
3, 0.25) and (-3, 1, 0, 2); pair 0 of row r turns by r radians and pair 1 by r / 100, so pair 0 of
row 1 is (cos 1 - 3 sin 1, sin 1 + 3 cos 1).

Rows: RowZeroPairZeroA, RowZeroPairZeroB, RowOnePairZeroA, RowOnePairZeroB, RowOnePairOneA,
RowOnePairOneB, RowTwoPairZeroA, RowTwoPairZeroB, RowTwoPairOneA, RowTwoPairOneB.

### AScoreDependsOnTheDistanceOfThePositionsAlone

The score of a turned query and key depends on their distance alone. Two positions apart, the query
leads by 2 and 0.02 radians, so the score is sum over pairs of (ac + bd) cos t + (ad - bc) sin t =
-3.1601117306901636.

### PositionZeroIsTheIdentity

Position 0 turns nothing: its row leaves the compiled step as it entered.

### EveryGroupTurnsByTheSameAngles

Two groups of one row holding the same values turn to the same values.

## test_rotary_positions.cpp

Suites: `Small_BlockRotaryAngle`, `Small_BlockRotaryRow`, `Small_BlockRotaryPositions`,
`Small_BlockRotaryUnitCircle`, `Small_BlockRotarySpanRefusal`, `Medium_BlockRotaryPositionsDeath`.

Tests `Language::Blocks::rotary_angle`, `Language::Blocks::write_rotary_row` and
`Language::Blocks::check_rotary_row_spans` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_positions.hpp`. Angles and rows
are compared within bounds derived from the unit roundoff of their type.

### AnAngleIsThePositionTimesTheFrequencyOfThePair

Pair i of n turns by position * 10000^(-i / n).

### PositionZeroTurnsNothing

Position 0 writes a cosine of 1 and a sine of 0 for every pair.

### ARowHoldsTheCosineAndTheSineOfEachAngle

The first pair of position 11 against the values worked by hand. Pair 0 turns by 11 radians: cos 11
= 0.0044257 and sin 11 = -0.9999902.

### ALargePositionWritesAPointOfTheUnitCircle

A position past the int32 range still writes a row of unit length.

### SpansOfOneNonzeroLengthAreAccepted

Spans of one nonzero length are accepted.

### SpansThatDoNotMatchAreRefused

An empty cosine span and spans of different lengths are refused.

### ARowWithoutMatchingSpansIsFatal

`write_rotary_row` stops the process on the refusal of the span check.

## test_transformer_block.cpp

Suites: `Small_BlockTransformerBlock`, `Small_BlockTransformerBlockFactory`.

Tests `Language::Blocks::TransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/transformer_block.hpp` and its node
factory `make_transformer_block_node`. Every block has a width of 8, two heads and a
feed-forward of hidden width 16, and is causal unless the case says otherwise.

### ATraceAddsANormalisedAttentionThenANormalisedFeedForward

A trace keeps the shape and adds each normalised branch to what it read.

### ABlockWithoutAMaskEmitsNone

The mask option reaches the attention: no CAUSAL_MASK without it.

### TheWalkYieldsTheFourChildren

The walk names the twelve parameters under the four children, in order.

### AnInputThatDoesNotFitFails

An input of another width fails once, with the first child's message.

### TheFactoryTakesTheDefaultsOfAbsentKeys

A row of heads alone takes the defaults of the other keys.

### TheFactoryReadsTheStatedKeys

Stated keys override the defaults.

### TheFactoryRefusesARowThatDoesNotFit

The factory refuses a stray key, no heads and heads that do not divide.

Rows: StrayKey, NoHeads, UnevenHeads.

### TheIncrementalFormTracesEachChildOverOnePosition

One position of a block is fed two leaves and carries the two caches.

### BothFormsDeclareTheSameParameters

The incremental form declares the parameters the ordinary form declares.

### AnUnmaskedBlockHasNoIncrementalForm

A block without a mask fails the trace through its attention.

### TheFactoryPassesThePositionsToItsAttention

The factory hands "positions" to the attention.

### RotaryPositionsOverAnOddHeadWidthAreRefused

Rotary positions over an odd head width are refused.
