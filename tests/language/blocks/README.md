# tests/language/blocks

The tests of the module `language/blocks`, one file per header of `modules/language/blocks/include/`. The suite prefix of a case gives its ctest label.

## test_causal_self_attention.cpp

Suites: `Unit_BlockCausalSelfAttention`.

Tests `Language::Blocks::CausalSelfAttention` in
`modules/language/blocks/include/SushiAI/language/blocks/causal_self_attention.hpp` and its node
factory `make_causal_self_attention_node`: the ordinary trace, the incremental trace of one
position, the rotary form of both, the state walk and the refusals. Every case uses a width of
8 and two heads. Most ordinary traces read a [2, 5, 8] input, and the incremental ones read a
[1, 1, 8] input over a per-sample shape of [5, 8].

### ACausalTraceMasksEveryHead

A causal block traced over [2, 5, 8] returns a [2, 5, 8] value, records no failure and no
refusal, and adds 45 nodes: 15 around the heads and 15 per head. The graph holds two
CAUSAL_MASK, two LOG_SOFTMAX, six SLICE_CHANNELS, one CONCAT_CHANNELS, six MATMUL and two SCALE
nodes.

### EachMaskReadsTheScaledScoresOfItsHead

The first input of every CAUSAL_MASK node has the shape [2, 5, 5] and comes from a SCALE node
whose attribute is 0.5 and whose input comes from a MATMUL node. The mask's own attribute is
below -1e8.

### ATraceWithoutAMaskEmitsNone

A block built with `AttentionMask::None` adds no CAUSAL_MASK node and 43 nodes in total, 15
around the heads and 14 per head.

### TheWalkYieldsTheTwoProjections

After a trace, `NN::walk_state` under the prefix "attn" yields "attn.qkv.weight",
"attn.qkv.bias", "attn.proj.weight" and "attn.proj.bias" in that order. The `qkv` weight has
the shape [24, 8] and the `proj` weight [8, 8].

### AnInputThatDoesNotFitFails

`forward` returns its input and the trace fails with `InvalidDescription` for a [5, 8] input
("CausalSelfAttention", "rank-3 [N, T, 8] input", "[5, 8]"), a [2, 5, 10] input
("CausalSelfAttention", "[2, 5, 10]") and an `Int32` input ("half, float32 or float64 input",
"int32"). The three traces add no node, the `qkv` weight stays undeclared, and the graph holds
a refusal.

### ABlockDescribedWrongFails

A block of width 8 with three heads fails with `InvalidDescription` and a message containing
"head count divides" and "dim=8 heads=3". A default-constructed block fails with "dim=0
heads=0". Each `forward` returns its input.

### TheFactoryReadsTheHeadsAndTheMask

Over a [5, 8] sample, the row `{"heads": 4}` builds a `ModuleNode<CausalSelfAttention>` whose
layer has `dim` 8, `heads` 4 and the mask `Causal`, with one port of the shape [5, 8]. The row
`{"heads": 2, "causal": false}` builds a layer whose mask is `None`.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and a message containing "layer 3" for a `dropout`
key ("'dropout'"), a row without `heads` ("'heads'", "absent"), three heads over a width of 8
("dim=8 heads=3") and a [4, 4, 8] sample ("rank-2 [T, C] input", "[4, 4, 8]").

### TheIncrementalFormWritesOneRowPerCacheAndReadsEveryRow

`forward_incremental` over a carrying trace returns a [1, 1, 8] value with no failure and no
refusal. The graph holds two SCATTER_ADD_ROWS, no CAUSAL_MASK, two LOG_SOFTMAX, nine
SLICE_CHANNELS, one CONCAT_CHANNELS, six MATMUL, two SCALE and four ADD nodes.

### TheIncrementalFormDeclaresTwoFedLeavesAndTwoCarriedValues

The trace declares two fed leaves: `K_POSITION_LEAF`, an `Int32` value of the shape [1, 1],
then `K_ATTENTION_MASK_LEAF` of the shape [1, 5]. It carries two values. The leaf and the next
value of each have the shape [5, 8], the leaf has the lifetime `Input`, and an ADD node
produces the next value.

### TwoBlocksOfOneTraceShareTheirFedLeaves

Two blocks traced incrementally one after the other in one carrying trace leave two fed leaves
and four carried values.

### TheIncrementalFormDeclaresTheParametersOfTheOrdinaryForm

After an incremental trace, `NN::parameter_count` of the block is 4 and `NN::buffer_count` is 0.

### TheIncrementalFormRefusesWhatItCannotCache

`forward_incremental` returns its input and the trace fails with `InvalidDescription` in four
cases: a block without a mask ("no mask", "no incremental form"), a [1, 3, 8] input ("one
position", "[1, 3, 8]"), an evaluation trace in place of a carrying one ("'position'",
"carrying trace") and a per-sample shape of [5] ("per-sample", "[5]").

### ARotaryTraceTurnsTheQueriesAndTheKeysOnce

A causal block with `AttentionPositions::Rotary`, traced with a `HostTableList` collecting
tables, returns a [2, 5, 8] value and adds 68 nodes, 23 more than the learned form. The graph
holds two CHANNEL_AFFINE, nine SLICE_CHANNELS, four CONCAT_CHANNELS and two CAUSAL_MASK nodes.
The list holds two tables, each a [5, 2] leaf of the lifetime `Buffer` that requires no
gradient and has ten values. The first value of the first table is 1 and of the second is 0.

### ARotaryBlockHoldsTheStateOfALearnedOne

After a rotary trace, `NN::walk_state` under "attn" yields the same four names as the learned
block: the weight and the bias of `qkv`, then of `proj`.

### ARotaryTraceRefusesWhatItCannotTurn

A rotary block traced without `collect_tables` fails with `InvalidDescription` and a message
containing "'rotary_cosine'" and "collect_tables". A rotary block of width 6 with two heads
fails with "rotary positions" and "3 channels". Each `forward` returns its input.

### TheFactoryReadsThePositions

The row `{"heads": 2}` builds a layer whose `positions` is `Learned`, and `"positions":
"rotary"` builds one whose `positions` is `Rotary`. The factory fails with `InvalidDescription`
and "layer 3" for `"positions": "sinusoidal"` ("'positions'", "'sinusoidal'", "'learned'",
"'rotary'"), for `"positions": 1` ("'positions'", "must be a string") and for a rotary row
over a [5, 6] sample ("rotary positions", "3 channels").

### TheRotaryIncrementalFormReplacesARowOfEachRing

The incremental trace of a rotary block returns a [1, 1, 8] value with no failure and no
refusal. The graph holds three SCATTER_ADD_ROWS, two SELECT, three ADD, two CHANNEL_AFFINE and
no CAUSAL_MASK nodes. Two values are carried; each leaf has the shape [5, 8], a SELECT node
produces each next value, and the third input of that node is the leaf.

### TheRotaryIncrementalFormIsFedASlotAndTheAngles

Two rotary blocks traced incrementally in one carrying trace declare four fed leaves in the
order `K_CACHE_SLOT_LEAF` (`Int32`, [1, 1]), `K_ATTENTION_MASK_LEAF` ([1, 5]),
`K_ROTARY_COSINE_LEAF` ([1, 2]) and `K_ROTARY_SINE_LEAF` ([1, 2]), and carry four values.

## test_expert_transformer_block.cpp

Suites: `Unit_BlockExpertTransformerBlock`.

Tests `Language::Blocks::ExpertTransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/expert_transformer_block.hpp`, and the
`transformer_block` row of `make_transformer_block_node` when it states `experts`. The block of
every case is causal, has a width of 8, two heads and four experts of hidden width 16.

### ATraceAddsANormalisedAttentionThenANormalisedMixture

`trace_experts` over a [2, 5, 8] input returns an output of the shape [2, 5, 8], a balance of
[1, 1] and a usage of [1, 4], with no failure and no refusal. The trace adds 84 nodes, among
them two LAYER_NORM, two CAUSAL_MASK and one UPSAMPLE2D. An ADD node produces the output; its
first input comes from another ADD node whose first input is the block's input.

### TheWalkYieldsTheFourChildren

After a trace, `NN::walk_state` under "h" yields fourteen names: the weight and the bias of
`norm1`, of `attn.qkv`, of `attn.proj`, of `norm2`, of `ff.router`, of `ff.expand` and of
`ff.project`, in that order.

### AnInputThatDoesNotFitFails

Over a [2, 5, 9] input `forward` returns the input and the trace fails with
`InvalidDescription` and a message containing "LayerNorm(8)" and "[2, 5, 9]". The graph holds
no LAYER_NORM node and the router's weight stays undeclared.

### TheTransformerRowWithExpertsHasThreePorts

Over a [5, 8] sample, the rows `{"heads": 2}` and `{"heads": 2, "experts": 0}` build nodes of
one port. The row with `"experts": 4, "top_k": 1` builds a node of three ports:
`K_SINGLE_PORT_NAME` of the shape [5, 8], `K_BALANCE_PORT`, and `K_USAGE_PORT` of the shape [4].
Traced over [2, 5, 8], that node writes three outputs, the third of the shape [1, 4]; the graph
holds one MAXPOOL2D node and a GELU node whose result has the shape [10, 128].

### TheTransformerRowRefusesCountsThatDoNotFit

The row fails with `InvalidDescription` and "layer 3" for a `top_k` without `experts`
("'top_k'", "no 'experts'"), for `"experts": 2, "top_k": 3` ("top_k=3") and for `"experts": -1`
("'experts'", "zero or more").

### TheIncrementalFormTracesEachChildOverOnePosition

`trace_experts_incremental` of a block that keeps one expert, over a [1, 1, 8] input and a
per-sample shape of [5, 8], returns an output of [1, 1, 8] and a usage of [1, 4]. The graph
holds two LAYER_NORM, two SCATTER_ADD_ROWS, no CAUSAL_MASK and one MAXPOOL2D node, and two
values are carried. The node of the row `{"heads": 2, "experts": 4}` writes three outputs
through `forward_incremental`, the first of the shape [1, 1, 8], and carries two values.

### TheBlockAndItsRowPassThePositionsToTheAttention

A block built without positions has `attn.positions` equal to `Learned`, and one built with
`AttentionPositions::Rotary` has `Rotary`. The node of a row with four experts and `"positions":
"rotary"`, traced with tables collected, leaves two host tables and two CHANNEL_AFFINE nodes.
The same row with eight heads fails with `InvalidDescription` and a message containing "layer
3", "rotary positions" and "1 channels".

## test_feed_forward.cpp

Suites: `Unit_BlockFeedForward`.

Tests `Language::Blocks::FeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/feed_forward.hpp` and its node factory
`make_feed_forward_node`. Every case uses a width of 6 and a hidden width of 24.

### ATraceMapsTheRowsThroughBothLayers

A trace over a [2, 5, 6] input returns a [2, 5, 6] value and adds 11 nodes. A RESHAPE node
produces the result; walking back through first inputs gives BIAS_ADD, MATMUL and GELU, and
the GELU result has the shape [10, 24].

### TheWalkYieldsTheTwoLinearLayers

After a trace, `NN::walk_state` under "ff" yields "ff.expand.weight", "ff.expand.bias",
"ff.project.weight" and "ff.project.bias" in that order. The `expand` weight has the shape
[24, 6] and the `project` weight [6, 24].

### AnInputOrADescriptionThatDoesNotFitFails

A [5, 6] input fails the trace with `InvalidDescription` and a message containing
"FeedForward", "rank-3 [N, T, 6] input" and "[5, 6]". A default-constructed block fails with
"positive widths" and "dim=0 hidden=0". Each `forward` returns its input, no node is added and
the `expand` weight stays undeclared.

### TheFactoryReadsTheHiddenWidth

Over a [5, 6] sample, the row `{"hidden": 24}` builds a `ModuleNode<FeedForward>` whose layer
has `dim` 6 and `hidden` 24, with one port of the shape [5, 6]. The factory fails with
`InvalidDescription` and "layer 3" for a row without `hidden` ("'hidden'", "absent") and for a
`bias` key ("'bias'").

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 6] value. Its
graph has as many nodes as the graph of `forward` over a [1, 1, 6] input, with the same
operation at every index. The trace declares no fed leaf and carries no value.

## test_language_head.cpp

Suites: `Unit_BlockLanguageHead`, `Unit_BlockLanguageHeadDeathTest`.

Tests `Language::Blocks::LanguageHead` in
`modules/language/blocks/include/SushiAI/language/blocks/language_head.hpp` and its node factory
`make_language_head_node`, with and without a weight tied to an embedding table. Every case
uses a width of 6 and a vocabulary of 11.

### ATraceNormalisesThenMapsToTheVocabulary

A trace over a [2, 5, 6] input returns a [2, 5, 11] value and adds 7 nodes. Walking back from
the result through first inputs gives RESHAPE, MATMUL, RESHAPE and LAYER_NORM, and the
LAYER_NORM node reads the head's input. The bias of `linear` stays undeclared.

### TheWalkYieldsTheNormalisationAndTheWeight

After a trace, `NN::walk_state` under "head" yields "head.norm.weight", "head.norm.bias" and
"head.linear.weight". The `linear` weight has the shape [11, 6].

### AnInputOrADescriptionThatDoesNotFitFails

A [2, 5, 7] input fails the trace with `InvalidDescription` and a message containing
"LanguageHead", "rank-3 [N, T, 6] input" and "[2, 5, 7]". A default-constructed head fails
with "positive widths" and "dim=0 vocabulary=0". Each `forward` returns its input and no node
is added.

### ATiedHeadScoresWithTheEmbeddingsTable

A head tied with `tie_to` to the weight of an `NN::Embedding` of 11 rows and 6 columns, traced
over that embedding's [2, 5, 6] result, returns a [2, 5, 11] value and adds 6 nodes, one fewer
than the untied head. The second input of its MATMUL node is the embedding's weight. The
head's own `linear` weight stays undeclared, `NN::parameter_count` of the head is 2 and of the
embedding is 1.

### ATiedTableThatDoesNotFitFails

A head tied to a parameter that was never declared fails the trace with `InvalidDescription`
and "tied to a table no layer has declared". A head tied to a declared [11, 5] table fails with
"table of [11, 6]" and "tied to a float32 table of [11, 5]". Each `forward` returns its input.

### OutputShapeReplacesTheWidthByTheVocabulary

`output_shape([5, 6])` is [5, 11].

### TheFactoryReadsTheVocabulary

Over a [5, 6] sample, the row `{"vocabulary": 11, "ln_epsilon": 0.001}` builds a
`ModuleNode<LanguageHead>` whose layer has `dim` 6, `vocabulary` 11 and a normalisation epsilon
of 0.001, with one port of the shape [5, 11]. The factory fails with `InvalidDescription` and
"layer 3" for an empty row ("'vocabulary'", "absent") and for a [6] sample ("rank-2 [T, C]
input").

### TheFactoryTiesTheHeadToAnEmbeddingRow

Above the head stands a `ModelGraph` whose row 0 is an 11 by 6 embedding and whose row 1 is a
layer normalisation. With `NN::BuiltRows` over that model, the row `{"tie": 0}` builds a head
of vocabulary 11 whose `tied_weight` is the address of the embedding's weight, with a port of
the shape [5, 11]. The factory fails with `InvalidDescription` for `"tie": -1` ("layer 2",
"'tie'", "row 1", "no 'embedding' row"), for `"tie": 0, "vocabulary": 12` ("layer 2", "12
tokens", "[11, 6] table"), for `"tie": 2` ("'tie'", "row 2", "not above") and for a tied row
built without the rows above it ("layer 0", "not above").

### OutputShapeOverARankOneSampleIsFatal

`output_shape([6])` stops the process with a report containing "LanguageHead::output_shape"
and "[6]". The case is in the suite `Unit_BlockLanguageHeadDeathTest`.

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 11] value.
Its graph has as many nodes as the graph of `forward` over a [1, 1, 6] input, with the same
operation at every index. The trace declares no fed leaf and carries no value.

## test_moe_feed_forward.cpp

Suites: `Unit_BlockMoeFeedForward`.

Tests `Language::Blocks::MoEFeedForward` in
`modules/language/blocks/include/SushiAI/language/blocks/moe_feed_forward.hpp` and its node
factory `make_moe_feed_forward_node`. Every case uses a width of 6 and four experts of hidden
width 8. A `top_k` of 0 keeps every expert.

### ATraceMixesEveryExpertByItsGate

`trace_experts` of a block with `top_k` 0 over a [2, 5, 6] input returns an output of the
shape [2, 5, 6], a balance of [1, 1] and a usage of [1, 4], and adds 31 nodes. The graph holds
one LOG_SOFTMAX, no MAXPOOL2D, one UPSAMPLE2D, one GELU and five MATMUL nodes. Walking back
from the output through first inputs gives RESHAPE, ADD, MATMUL and MUL, and the MUL result
has the shape [10, 32].

### KeepingTheTopScoresAddsOneWindowPerLevel

For `top_k` 1, 2 and 4, the trace returns a [2, 5, 6] output and adds 31 + 7 + 4 * `top_k`
nodes. The graph holds `top_k` MAXPOOL2D, `top_k` LESS_EQUAL and `top_k` SELECT nodes and two
LOG_SOFTMAX nodes, and a RESHAPE node produces the usage.

### TheWalkYieldsTheRouterAndTheStackedLayers

After a trace, the state walk under "ff" yields the weight and the bias of `router`, of
`expand` and of `project`, in that order. The shapes are [4, 6] for the router's weight,
[32, 6] for the `expand` weight, [6, 32] for the `project` weight and [4, 6] for the `project`
bias, whose role is `ParameterRole::Bias`.

### AnInputOrADescriptionThatDoesNotFitFails

`forward` over a [5, 6] input returns the input and fails the trace with `InvalidDescription`
("MoEFeedForward", "rank-3 [N, T, 6] input", "[5, 6]"). A default-constructed block fails with
"at least one expert" and "dim=0 hidden=0 experts=0". `trace_experts` of a block with `top_k`
5 returns a failure containing "1 to 4 of them" and "top_k=5" and leaves the trace's own
status ok. No node is added and the router's weight stays undeclared.

### TheFactoryBuildsANodeOfThreePorts

Over a [5, 6] sample, the row `{"experts": 4, "hidden": 8, "top_k": 2}` builds a node of one
input and three ports: `K_SINGLE_PORT_NAME` of [5, 6], `K_BALANCE_PORT` of [1] and
`K_USAGE_PORT` of [4]. Traced over [2, 5, 6], the node writes three outputs of the shapes
[2, 5, 6], [1, 1] and [1, 4], the graph holds two MAXPOOL2D nodes, and the state walk yields
six names. `forward` over a [5, 6] input fails with `InvalidDescription` ("MoEFeedForward",
"[5, 6]") and leaves the three outputs in the vector.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and "layer 3" for a row without `experts`
("'experts'", "absent"), a row without `hidden` ("'hidden'", "absent"), a `noise` key
("'noise'"), a `top_k` of 5 ("top_k=5"), a `top_k` of -1 ("'top_k'", "zero or more") and a
`top_k` of 0.5 ("'top_k'", "whole number").

### TheIncrementalFormIsTheOrdinaryTraceOfOnePosition

`forward_incremental` of a block with `top_k` 1 over a [1, 1, 6] input in a carrying trace
returns a [1, 1, 6] value. Its graph has as many nodes as the graph of `forward` over a
[1, 1, 6] input, with the same operation at every index. The trace declares no fed leaf and
carries no value.

## test_position_embedding.cpp

Suites: `Unit_BlockPositionEmbedding`.

Tests `Language::Blocks::PositionEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/position_embedding.hpp` and its node
factory `make_position_embedding_node`. The sequence of every case has five positions and a
width of 6.

### ATraceAddsTheTableToEveryRow

A block of five rows traced over a [2, 5, 6] input returns a [2, 5, 6] value and adds 5 nodes.
Its weight has the shape [5, 6] and the role `ParameterRole::Embedding`. A RESHAPE node
produces the result from a BIAS_ADD node whose inputs have the shapes [2, 30] and [30].

### AShorterSequenceReadsTheLeadingRows

A block of eight rows traced over a [2, 5, 6] input adds 7 nodes and declares a [8, 6] weight.
The second input of the BIAS_ADD node comes from a RESHAPE node that reads a SLICE_CHANNELS
node; the slice's result has the shape [1, 1, 1, 30] and its first attribute is 0.

### ASequenceLongerThanTheTableIsRefused

Over a [2, 5, 6] input, `forward` returns the input and the trace fails with
`InvalidDescription` for a block of four rows ("5 positions", "4 rows"), whose weight stays
undeclared, for a block of width 7 ("PositionEmbedding", "[N, T, 7]") and for a
default-constructed block ("positive extents").

### TheWalkYieldsTheTable

After a trace, `NN::for_each_parameter` under "positions" yields the one name
"positions.weight".

### TheFactoryReadsTheLength

Over a [5, 6] sample, an empty row builds a `ModuleNode<PositionEmbedding>` whose layer has
`length` 5 and `dim` 6, with one port of the sample's shape. The row `{"length": 9}` builds a
layer of `length` 9. The factory fails with `InvalidDescription` and "layer 1" for `"length":
4` ("'length' is 4", "5 positions"), for a `rows` key ("'rows'") and for a [6] sample ("rank-2
[T, C] input").

### TheIncrementalFormAddsTheRowOfTheFedPosition

`forward_incremental` over a [1, 1, 6] input in a carrying trace returns a [1, 1, 6] value.
The graph holds one GATHER_ROWS, one ADD and no BIAS_ADD node, and the weight's value has the
shape [5, 6]. The trace declares one fed leaf, `K_POSITION_LEAF`, an `Int32` value of the
shape [1, 1], and carries no value.

### TheIncrementalFormRefusesAWindowItsTableCannotServe

`forward_incremental` returns its input and the trace fails with `InvalidDescription` for a
per-sample shape of [6, 6] over a table of five rows ("6 positions", "5 rows"), which declares
no fed leaf, and for a [1, 2, 6] input ("one position", "[1, 2, 6]").

## test_rotary_embedding.cpp

Suites: `Unit_BlockRotaryEmbedding`.

Tests `Language::Blocks::RotaryEmbedding` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_embedding.hpp` and
`Language::Blocks::check_rotary_widths`. Every case uses a width of 8 cut into two groups, so
each group has two pairs. The ordinary traces read five positions.

### ATraceWeighsTheValueAndItsSwappedHalves

A trace over a [2, 5, 8] input with tables collected returns a [2, 5, 8] value, with no
failure and no refusal, and adds 20 nodes: two table leaves and 18 others. The graph holds two
CHANNEL_AFFINE, two SLICE_CHANNELS, three CONCAT_CHANNELS, one SCALE and one ADD node. Each
CHANNEL_AFFINE node reads a [2, 1, 1, 40] value and a [40] value that requires no gradient.

### TheTablesHoldTheAnglesOfEveryPosition

The trace leaves two host tables, cosines then sines. Each leaf has the shape [5, 2], the
cosine leaf is `Float32`, and each table has ten values. For every row and pair the two values
are within 1e-15 of the cosine and the sine of `rotary_angle(row, pair, 2)`.

### ADoubleValueReadsDoubleTables

Over a `Float64` input of the shape [1, 5, 8] the result is `Float64`, the first of the two
table leaves is `Float64` and the graph holds no CAST node.

### TheBlockHoldsNoState

After a trace, `NN::walk_state` yields no parameter and no buffer, and the graph's
`state_leaves()` is empty. `output_shape([5, 8])` is [5, 8].

### TheIncrementalFormIsFedTheAnglesOfItsPosition

`forward_incremental` over a [1, 1, 8] input in a carrying trace returns a [1, 1, 8] value and
adds 20 nodes, with no failure and no refusal. The trace carries no value and declares two
fed leaves, `K_ROTARY_COSINE_LEAF` then `K_ROTARY_SINE_LEAF`, each a `Float32` value of the
shape [1, 2].

### ABlockOrAnInputThatDoesNotFitIsRefused

`check_rotary_widths(8, 2)` is ok. It fails with `InvalidDescription` for (6, 2) ("pairs",
"width=6 groups=2"), (8, 3) ("width=8 groups=3") and (0, 0) ("width=0 groups=0"). `forward`
over a [2, 5, 10] input returns the input, fails with "RotaryEmbedding" and "[2, 5, 10]" and
collects no table. `forward` in a trace that collects no tables fails with "'rotary_cosine'"
and "collect_tables", and `forward_incremental` in an evaluation trace fails with
"'rotary_cosine'" and "carrying trace".

## test_rotary_positions.cpp

Suites: `Unit_BlockRotaryPositions`, `Unit_BlockRotaryPositionsDeathTest`.

Tests `Language::Blocks::rotary_angle` and `Language::Blocks::write_rotary_row` in
`modules/language/blocks/include/SushiAI/language/blocks/rotary_positions.hpp`. Angles are
compared within 1e-12 and `float` rows within 1e-6.

### AnAngleIsThePositionTimesTheFrequencyOfThePair

`K_ROTARY_BASE` is 10000. `rotary_angle(position, pair, pairs)` is 0 for (0, 1, 4), 7 for
(7, 0, 4), 0.03 for (3, 1, 2), 0.5 for (5, 1, 4), 0.05 for (5, 2, 4) and 0.005 for (5, 3, 4).

### PositionZeroTurnsNothing

`write_rotary_row` for position 0 over four pairs writes a cosine of exactly 1 and a sine of
exactly 0 for every pair.

### ARowHoldsTheCosineAndTheSineOfEachAngle

The row of position 11 over four pairs holds, for every pair, the cosine and the sine of
`rotary_angle(11, pair, 4)` converted to `float`. The cosine of pair 0 is 0.0044257 and its
sine is -0.9999902.

### ALargePositionWritesAPointOfTheUnitCircle

For position 2^40 over four pairs, the squared cosine plus the squared sine of every pair is 1
within 1e-6.

### ARowWithoutMatchingSpansIsFatal

`write_rotary_row` stops the process for two empty spans, with a report containing "0
cosines", and for a span of two cosines beside a span of three sines, with "2 cosines" and "3
sines". The case is in the suite `Unit_BlockRotaryPositionsDeathTest`.

## test_transformer_block.cpp

Suites: `Unit_BlockTransformerBlock`.

Tests `Language::Blocks::TransformerBlock` in
`modules/language/blocks/include/SushiAI/language/blocks/transformer_block.hpp` and its node
factory `make_transformer_block_node`. Every block has a width of 8, two heads and a
feed-forward of hidden width 16, and is causal unless the case says otherwise.

### ATraceAddsANormalisedAttentionThenANormalisedFeedForward

A trace over a [2, 5, 8] input returns a [2, 5, 8] value and adds 64 nodes, among them two
LAYER_NORM, two ADD, two CAUSAL_MASK and one GELU. An ADD node produces the result; its first
input comes from the other ADD node, whose first input is the block's input. Each LAYER_NORM
node reads either the block's input or the result of the first ADD node.

### ABlockWithoutAMaskEmitsNone

A block built with `AttentionMask::None` traces without failure into a graph of no CAUSAL_MASK
node and two LOG_SOFTMAX nodes.

### TheWalkYieldsTheFourChildren

After a trace, `NN::walk_state` under "h" yields twelve names: the weight and the bias of
`norm1`, of `attn.qkv`, of `attn.proj`, of `norm2`, of `ff.expand` and of `ff.project`, in that
order.

### AnInputThatDoesNotFitFails

Over a [2, 5, 9] input `forward` returns the input and the trace fails with
`InvalidDescription` and a message containing "LayerNorm(8)" and "[2, 5, 9]". The graph holds
no LAYER_NORM node and holds a refusal.

### TheFactoryReadsItsKeysOverTheirDefaults

Over a [5, 8] sample, the row `{"heads": 2}` builds a `ModuleNode<TransformerBlock>` with two
heads, a hidden width of 32, the mask `Causal` and `K_LAYER_NORM_EPSILON` in `norm1`, and a
port of the shape [5, 8]. The row `{"heads": 4, "hidden": 16, "causal": false, "ln_epsilon":
0.001}` builds one with four heads, a hidden width of 16, the mask `None` and an epsilon of
0.001 in `norm2`.

### TheFactoryRefusesARowThatDoesNotFit

The factory fails with `InvalidDescription` and "layer 3" for a `depth` key ("'depth'"), a row
without `heads` ("'heads'", "absent") and five heads over a width of 8 ("dim=8 heads=5").

### TheIncrementalFormTracesEachChildOverOnePosition

`forward_incremental` over a [1, 1, 8] input and a per-sample shape of [5, 8] returns a
[1, 1, 8] value with no failure and no refusal. The graph holds two LAYER_NORM, two
SCATTER_ADD_ROWS, no CAUSAL_MASK and one GELU node. The trace declares two fed leaves and
carries two values.

### BothFormsDeclareTheSameParameters

A block traced with `forward` over [1, 5, 8] and a second block traced with
`forward_incremental` over [1, 1, 8] have the same `NN::parameter_count`, and the second has a
`NN::buffer_count` of 0.

### AnUnmaskedBlockHasNoIncrementalForm

`forward_incremental` of a block built with `AttentionMask::None` returns its input, and the
trace fails with `InvalidDescription` and a message containing "no mask".

### TheFactoryPassesThePositionsToItsAttention

The row `{"heads": 2}` builds a layer whose `attn.positions` is `Learned`, and `"positions":
"rotary"` builds one whose `attn.positions` is `Rotary`. A rotary row over a [5, 6] sample
fails with `InvalidDescription` and a message containing "layer 3", "rotary positions" and "3
channels".

Suites: `Integration_AttentionGradient`, in `sushiai_language_blocks_tests`.

Every graph is FLOAT64. The file's `Probe` builds a graph, takes the cross-entropy of the `[N, C]`
value its build function names against targets whose row r names class r mod C, differentiates it at
one leaf and compiles it. Every other leaf is filled from the seeded host stream.
`Autograd::check_full_jacobian` compares the analytic gradient with central differences. The block
cases read a `[1, 4, 4]` sequence through a `CausalSelfAttention` of two heads.

### AnUnfusedLogSoftmaxMatchesItsClosedForm

A SCALE by one sits between LOG_SOFTMAX and NLL_LOSS over a `[3, 4]` parameter, and the graph holds
one LOG_SOFTMAX_BACKWARD node. The targets are positive weights whose rows do not sum to one. After
two runs the gradient equals `(softmax(x) * rowsum(Y) - Y) / N`, computed on the host, within 1e-12.

### ASoftmaxGradientMatchesCentralDifferences

A `[3, 4]` softmax is read by a matrix product with a `[4, 3]` input. The graph holds one
LOG_SOFTMAX_BACKWARD node, and the gradient at the softmax's operand is checked against central
differences.

### ASoftmaxOverRankThreeDifferentiatesThroughItsReshapes

A softmax over the last axis of a `[2, 3, 4]` parameter is reshaped to `[6, 4]`. The graph holds one
LOG_SOFTMAX_BACKWARD node, and the gradient is checked against central differences.

### ACausalAttentionDifferentiatesAtItsInput

The gradient of a causal attention block is taken at the sequence it reads. The graph holds two
LOG_SOFTMAX_BACKWARD nodes, one for each head, and the gradient is checked against central
differences.

### ACausalAttentionDifferentiatesAtItsProjectionWeight

The same block is differentiated at the weight of its `qkv` projection, with the same node count and
the same check.

### AnUnmaskedAttentionDifferentiatesAtItsInput

A block built with `AttentionMask::None` is differentiated at its input, with the same node count
and the same check.

Suites: `Integration_CausalAttention`, in `sushiai_language_blocks_tests`.

The mask cases run CAUSAL_MASK with a fill of -1e9 on an engine and compare with a host loop. The
block cases trace a block over a `[1, 5, 8]` input with two heads, fill every parameter from the
seeded host stream, and run it twice: once on an input, once with positions 3 and 4 replaced. Values
that must be the same number in both runs are compared with a tolerance of 0.

### TheMaskReplacesTheEntriesAboveTheDiagonal

A FLOAT32 `[2, 4, 4]` value of dyadic numbers is masked and the step is run twice. The lower
triangle of each matrix is unchanged and every entry above the diagonal holds the fill.

### TheMaskReplacesTheEntriesAboveTheDiagonalInFloat64

The same check in FLOAT64.

### TheMaskPassesItsGradientThroughTheKeptEntriesAlone

The loss is the cross-entropy of the masked FLOAT64 `[4, 4]` rows against class 0. The gradient at
the scores equals `(softmax - target) / 4` on and below the diagonal within 1e-12, where the softmax
is taken over the kept entries, and is exactly zero above it.

### AnEarlierPositionIgnoresALaterInput

Through a causal `CausalSelfAttention`, no element of positions 0 to 2 differs between the two runs,
and at least one element of positions 3 and 4 does. Both outputs hold 40 elements and the first is
finite.

### WithoutTheMaskAnEarlierPositionReadsALaterInput

With `AttentionMask::None`, at least one element of positions 0 to 2 differs between the two runs.

### ATransformerBlockKeepsEarlierPositions

A causal `TransformerBlock` with a feed-forward width of 32 keeps positions 0 to 2 unchanged and
changes positions 3 and 4.

Suites: `Integration_RotaryEmbedding`, in `sushiai_language_blocks_tests`.

A `RotaryEmbedding` is run through a compiled step over rows of float32 values in groups of four
channels, two pairs each. Results are compared with the host's double within 1e-5 absolute.

### ATurnMatchesTheHostFormula

Over three rows of one group, each pair `(a, b)` becomes `(a cos - b sin, a sin + b cos)` for the
angle `rotary_angle` gives its row and pair. Pair 0 of row 1, `(1, 3)` turned by one radian, is
-1.984110649 and 2.462377903.

### PositionZeroIsTheIdentity

Row 0 leaves the block with exactly the values it entered with. Row 1 of the same values differs by
more than 1 in its first channel.

### EveryGroupTurnsByTheSameAngles

Two groups of one row that hold the same values turn to exactly the same values, on each of three
rows.

### AScoreDependsOnTheDistanceOfThePositionsAlone

A query and a key are turned at six positions. The score of positions 2 and 0 equals the scores of 3
and 1, 4 and 2, and 5 and 3, and equals the number the host computes from a distance of 2. The score
of positions 3 and 0 differs from it by more than 1.
