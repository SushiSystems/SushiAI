# tests/language/generation

The tests of the module `language/generation`, one file per header of `modules/language/generation/include/`. The suite prefix of a case gives its ctest label.

## test_generation_task.cpp

Suites: `Small_GenerationTask`.

Tests `Language::Generation::read_generation_task` in
`modules/language/generation/include/SushiAI/language/generation/generation_task.hpp`, the
reader of the `next_token_generation` task section. Every call passes the context "the task
section".

### EveryKeyIsRead

A section with `vocabulary` "vocab.savb", `window` 256, `tokens` 200, `seed` 7 and a `top_k`
sampler reads back with those four values and a sampler whose name is `K_TOP_K_SAMPLER`.

### TheOptionalKeysHaveDefaults

A section with only `type`, `vocabulary` and `window` reads `tokens` as
`K_DEFAULT_GENERATED_TOKENS`, `seed` as 0 and a sampler whose name is `K_GREEDY_SAMPLER`.

### AKeyThatDoesNotFitIsRefused

The reader fails with `InvalidDescription` for the type "anchor_detection"
("'anchor_detection'", "'next_token_generation'"), a section without `vocabulary` ("the task
section", "has no 'vocabulary'"), one without `window` ("has no 'window'"), a `window` of 0
("'window'", "positive"), a `tokens` of 0 ("'tokens'", "positive"), a `seed` of -1 ("'seed'"),
a `prompt` key ("the task section", "'prompt'", "does not have") and a sampler of the type
"beam" ("the task section 'sampler'", "'beam'").

### TheCacheIsReadAndDefaultsToTrue

`K_DEFAULT_GENERATION_CACHE` is true. `cache` reads as true from a section without the key,
false from `"cache": false` and true from `"cache": true`.

### ACacheThatIsNoFlagIsRefused

`"cache": 1` fails with `InvalidDescription` and a message containing "the task section",
"'cache'" and "a boolean".

## test_greedy_sampler.cpp

Suites: `Small_GenerationGreedySampler`, `Small_GenerationGreedySamplerDeathTest`.

Tests `Language::Generation::GreedySampler` in
`modules/language/generation/include/SushiAI/language/generation/greedy_sampler.hpp`. Every
generator is seeded with 20261011.

### ThePickIsTheLargestLogitAndTheFirstOfEquals

`sample` returns 1 for the logits 0.5, 2, -1, returns 1 for 0.5, 2, 2, -1 and returns 0 for the
single logit -3. `name()` is `K_GREEDY_SAMPLER`.

### APickDrawsNothing

After one `sample`, the generator compares equal to a second generator built from the same
seed and never used.

### NoLogitIsFatal

`sample` over an empty span stops the process with a report containing "no logit". The case is
in the suite `Small_GenerationGreedySamplerDeathTest`.

## test_nucleus_sampler.cpp

Suites: `Small_GenerationNucleusSampler`.

Tests `Language::Generation::NucleusSampler` in
`modules/language/generation/include/SushiAI/language/generation/nucleus_sampler.hpp`. The
distribution cases count 20000 draws under the seed 20261011 over the logits ln 2, 0, 0, whose
probabilities are 1/2, 1/4 and 1/4, and compare each share within 0.02.

### TheFewestTokensThatReachTheMassAreDrawn

With a mass of 0.6 and a temperature of 1, token 0 has a share of 2/3, token 1 a share of 1/3
and token 2 is never drawn. `name()` is `K_NUCLEUS_SAMPLER`.

### AMassTheFirstTokenReachesKeepsItAlone

With a mass of 0.4, token 0 is drawn all 20000 times.

### AMassOfOneKeepsEveryToken

With a mass of 1, the shares are 1/2, 1/4 and 1/4.

### ASeedFixesTheSequence

A sampler of mass 0.9 and temperature 1.2 draws the same 64 tokens twice from the same seed
over eight logits from 0 to 0.7.

### MakeRefusesAMassOutsideItsRangeAndABadTemperature

`NucleusSampler::make` fails with `InvalidDescription` for a mass of 0 ("'nucleus'", "above
0"), a mass of 1.5 ("at most 1"), a mass that is not a number ("'nucleus'") and a temperature
of -2 ("temperature").

## test_sampler_section.cpp

Suites: `Small_GenerationSamplerSection`.

Tests `Language::Generation::make_sampler` in
`modules/language/generation/include/SushiAI/language/generation/sampler_section.hpp`, which
builds a sampler from a section. Every call passes the context "the task section 'sampler'".

### EachTypeBuildsItsSampler

The sections `{"type": "greedy"}`, `{"type": "temperature", "temperature": 0.7}`, `{"type":
"temperature"}`, `{"type": "top_k", "k": 40, "temperature": 2}` and `{"type": "nucleus", "p":
0.9}` build samplers whose names are `K_GREEDY_SAMPLER`, `K_TEMPERATURE_SAMPLER` twice,
`K_TOP_K_SAMPLER` and `K_NUCLEUS_SAMPLER`.

### AnUnknownTypeAndAnAbsentKeyAreRefused

`make_sampler` fails with `InvalidDescription` for the type "beam" ("'beam'", "'greedy',
'temperature', 'top_k', 'nucleus'"), a `top_k` section without `k` ("'sampler'", "has no
'k'"), a `nucleus` section without `p` ("has no 'p'"), a section without `type` ("has no
'type'") and a string in place of the section ("an object").

### AKeyTheTypeDoesNotReadIsRefused

`make_sampler` fails with `InvalidDescription` for a `greedy` section with `temperature`
("'temperature'", "does not have"), a `temperature` section with `k` ("'k'"), a `k` of 0
("keep 0 tokens"), a `k` of "many" ("'k'", "whole number") and a `p` of 2 ("at most 1").

## test_temperature_sampler.cpp

Suites: `Small_GenerationTemperatureSampler`.

Tests `Language::Generation::TemperatureSampler` in
`modules/language/generation/include/SushiAI/language/generation/temperature_sampler.hpp`.
The distribution cases count 20000 draws under the seed 20261011 and compare each share
within 0.02.

### TheSharesAreTheSoftmax

At a temperature of 1 over the logits 0, ln 2, 0, the shares are 1/4, 1/2 and 1/4. `name()`
is `K_TEMPERATURE_SAMPLER`.

### ALowerTemperatureSharpensTheShares

At a temperature of 0.5 over the logits 0 and ln 2, the shares are 0.2 and 0.8.

### ASeedFixesTheSequence

Over eight logits from 0 to 0.7, two sequences of 64 draws from the same seed are equal, and
the sequence from the seed plus 1 differs from them.

### MakeRefusesATemperatureThatIsNotPositiveAndFinite

`TemperatureSampler::make` fails with `InvalidDescription` for 0 ("'temperature'",
"positive"), for -1 ("positive"), for infinity ("finite") and for a value that is not a number
("finite").

## test_top_k_sampler.cpp

Suites: `Small_GenerationTopKSampler`.

Tests `Language::Generation::TopKSampler` in
`modules/language/generation/include/SushiAI/language/generation/top_k_sampler.hpp`. The
distribution cases count 20000 draws under the seed 20261011 and compare each share within
0.02.

### OnlyTheKLargestAreDrawnAndTheLowerIndexWinsATie

With `k` 2 and a temperature of 1 over the logits 0, ln 2, 0, -5, token 0 has a share of 1/3,
token 1 a share of 2/3, and tokens 2 and 3 are never drawn. `name()` is `K_TOP_K_SAMPLER`.

### KeepingOneTokenIsTheGreedyPick

With `k` 1 over the logits 0.3, 0.2, 0.9, 0.9, token 2 is drawn all 20000 times.

### AKAboveTheRowKeepsEveryToken

With `k` 50 over the logits 0 and ln 2, the shares are 1/3 and 2/3.

### ASeedFixesTheSequence

A sampler of `k` 4 and temperature 0.8 draws the same 64 tokens twice from the same seed over
eight logits from 0 to 0.7.

### MakeRefusesAKBelowOneAndABadTemperature

`TopKSampler::make` fails with `InvalidDescription` for a `k` of 0 ("'top_k'", "keep 0
tokens") and for a temperature of 0 ("'top_k'", "temperature").

## test_position_feed.cpp

Suites: `Medium_GenerationPositionFeed`.

Tests `Language::Generation::Detail::make_position_feed`, which picks a `PositionFeed` from
the leaves a step is fed. The fixture is `Support::FedStepTest` of
`position_feed_test_support.hpp`; its `build_step(has_position, has_ring)` builds a step fed
"feed_position", or "feed_slot" with "feed_cosine" and "feed_sine", or both, or neither.

### AStepFedAPositionGetsTheTableFeed

Over a step fed only a position, the feed lists one leaf, named `K_FEED_POSITION`, and
`keeps_rows_on_slide()` is false.

### AStepFedACacheSlotGetsTheRingFeed

Over a step fed only the ring leaves, the feed lists three leaves, the first named
`K_FEED_SLOT`, and `keeps_rows_on_slide()` is true.

### LeavesWithoutARingPickTheTableFeed

With `CachedStepLeaves` that name a position and a mask and no ring, a step fed a position
gets a feed of one leaf.

### AStepFedBothOrNeitherIsRefused

`make_position_feed` fails with `InvalidDescription` over a step fed both ("CachedGenerator",
"both", "'feed_position'", "'feed_slot'") and over a step fed neither ("neither",
"'feed_position'", "'feed_slot'").

## test_ring_position_feed.cpp

Suites: `Medium_GenerationRingPositionFeed`.

Tests `Language::Generation::Detail::RingPositionFeed`. The fixture is `Support::FedStepTest`;
the feed is made over a window of four rows and three pairs, with an angle function that
writes position + pair as each cosine and the negation of that as each sine.

### TheFeedListsTheSlotThenTheAngles

`feeds()` lists three leaves in the order `K_FEED_SLOT`, `K_FEED_COSINE`, `K_FEED_SINE`. The
slot tensor is `Int32` with one element, the cosine tensor is `Float32` with three elements
and the sine tensor has three elements.

### PlaceWrapsTheSlotAndKeepsThePositionOfTheAngles

`place(11)` writes the slot 3, the cosines 11, 12, 13 and the sines -11, -12, -13. `place(4)`
writes the slot 0 and a first cosine of 4. `place(2)` writes the slot 2.

### EveryRowStaysRightAfterASlide

`keeps_rows_on_slide()` is true.

### ARingThatDoesNotFitIsRefused

Over a step fed both the position and the ring leaves, `RingPositionFeed::make` fails with
`InvalidDescription` for a null angle function ("no function", "'feed_cosine'",
"'feed_sine'"), for a sine leaf named "feed_tangent" ("'feed_tangent'"), for the cosine leaf
named as the slot ("'feed_cosine'", "int32 leaf of [1, 1]") and for the position leaf named
as the sine ("'feed_position'", "float32 leaf of [1, 3]").

## test_table_position_feed.cpp

Suites: `Medium_GenerationTablePositionFeed`.

Tests `Language::Generation::Detail::TablePositionFeed`. The fixture is
`Support::FedStepTest`.

### PlaceWritesThePositionIntoItsOneLeaf

Over a step fed a position, `feeds()` lists one leaf named `K_FEED_POSITION` whose tensor is
`Int32` with one element. `place(0)` writes 0 into it and `place(5)` writes 5.

### NoRowStaysRightAfterASlide

`keeps_rows_on_slide()` is false.

### ALeafThatDoesNotFitIsRefused

Over a step fed only the ring leaves, `TablePositionFeed::make` fails with
`InvalidDescription` for the leaf "feed_position" ("'feed_position'") and for the leaf
"feed_cosine" ("CachedGenerator", "'feed_cosine'", "int32 leaf of [1, 1]").

## test_cached_generator.cpp

Suites: `Medium_CachedGeneration`, in `sushiai_language_generation_tests`.

The fixture builds the decoder of `decoder_model.hpp`: a learned position table, two blocks, a
window of 8 and a vocabulary of 32. It traces the `[1, 8]` window graph and the `[1, 1]` incremental
graph, initialises one state, and builds a `WindowGenerator` and a `CachedGenerator` over it. Tokens
come from a greedy sampler. Float32 logits of the two passes are compared with an absolute tolerance
of 1e-4.

### TheGeneratorReadsTheExtentsOfTheStep

The cached generator reports a window of 8 and a vocabulary of 32. Its step carries state and has
three inputs: `position` is one of them and `cache_slot` is refused as `InvalidDescription`. The
incremental graph carries four values, two for each block.

### TheCachesAreNoEntryOfTheModelState

A state created again from the model after both traces has as many entries as the first one, and
every entry is a parameter.

### APromptThatFitsGivesTheTokensOfTheWindowGenerator

A prompt of three ids and four generated tokens stay inside the window. The cached generator gives
the tokens of the window generator, in 6 runs of one compile.

### TheLogitsEqualThoseOfTheWindowAtTheLastPosition

After a prompt of five ids, the 32 logits of the one-position step equal the window step's logits at
position 4 within 1e-4.

### AContextThatSlidesGivesTheTokensOfTheWindowGenerator

A prompt of three ids and twelve generated tokens pass the window. The tokens equal the window
generator's, and the step ran 56 times from one compile: 3 for the prompt, 5 to fill the window, and
8 for each of 6 slides.

### APromptLongerThanTheWindowKeepsItsLastIDs

A prompt of eleven ids gives the same four tokens from both generators.

### ASecondPromptMatchesAFreshGenerator

A generator that has produced twelve tokens for one prompt is given a second prompt. Its four tokens
equal those of a generator compiled fresh over the same state and those of the window generator.
Both steps report one compile.

### AStepThatDoesNotFitIsRefused

`CachedGenerator::make` over the window step is refused as `InvalidDescription` with "keeps nothing
between runs". A mask leaf named `scores` is refused with a message that names 'scores' and
'attention_mask'.

### APromptOrACountThatDoesNotFitIsRefused

An empty prompt, the id 32 and a count of zero tokens are each refused as `InvalidDescription`; the
messages contain "no id", "0 to 31" and "0 tokens". Nothing is generated and the step has run zero
times.

Suites: `Medium_RotaryGeneration`, in `sushiai_language_generation_tests`.

The fixture builds rotary decoders from the model files of `decoder_model.hpp`: two blocks over a
window of 8, one block over a window of 8, and two blocks over a window of 64. For each it can
compile the window step and the one-position step and build a `WindowGenerator` and a
`CachedGenerator`. Float32 logits of two passes are compared with an absolute tolerance of 1e-4. A
ring of two or more blocks does not equal `WindowGenerator` after a slide; see
`modules/language/generation/README.md`.

### TheStepIsFedTheRingAndNoPosition

The cached generator of the two-block decoder reports a window of 8 and a vocabulary of 32. Its step
has five inputs, among them `cache_slot`, `rotary_cosine` and `rotary_sine`, and it refuses
`position` as `InvalidDescription`. The incremental graph carries four values and holds no host
table; the window graph holds four.

### TheStateHoldsParametersOnly

The state of the two-block decoder has 27 entries. Each is a parameter, and no name contains
"rotary".

### TheLogitsEqualThoseOfTheWindowAtTheLastPosition

After a prompt of five ids, the 32 logits of the one-position step equal the window step's at
position 4 within 1e-4.

### APromptThatFitsGivesTheTokensOfTheWindowGenerator

With two blocks, a prompt of three ids and four tokens give the same tokens from both generators.

### AOneBlockRingGivesTheTokensOfTheWindowGeneratorAcrossSlides

With one block, 26 greedy tokens after a prompt of three ids equal the window generator's. The
one-position step ran 28 times from one compile.

### TheLogitsOfAOneBlockRingEqualTheWindowsAtEveryPosition

A text of 30 ids is fed to the one-block ring one position at a time. At each position the window
step is run over the last 8 ids, and the 32 logits of the ring equal the window's at its last held
position within 1e-4.

### ARingEqualsAWideCacheReadThroughItsLastPositions

The state of the two-block decoder is copied into the decoder with a window of 64, whose cache never
wraps in this run and is masked to its last 8 positions. Greedy tokens sampled from that step equal
the 26 tokens the ring's generator produced.

### AContextThatSlidesTakesOneRunPerToken

Twelve tokens after a prompt of three ids take 14 runs of one compile.

### ASlotWrittenTwiceHoldsTheSecondRow

The one-block ring is fed 8 positions and then position 8, which wraps to row 0. A second step is
fed position 8 alone, into zeros. In both caches of the block, the keys and the values, row 0 of the
first step equals row 0 of the second exactly, and row 1 keeps the values it held before the wrap.

### ASecondPromptMatchesAFreshGenerator

A generator that has produced 26 tokens for one prompt gives, for a second prompt, the 26 tokens of
a generator compiled fresh over the same state.

### AStepOrARingThatDoesNotFitIsRefused

A ring without its angle function is refused as `InvalidDescription` with "no function" and
'rotary_cosine'. A decoder with a learned position table before a rotary block, whose step is fed
both leaves, is refused with a message that contains "both", 'position' and 'cache_slot'.

## test_window_generator.cpp

Suites: `Medium_Generation`, in `sushiai_language_generation_tests`.

The fixture builds the decoder of `decoder_model.hpp`, with a window of 8 and a vocabulary of 32,
initialises its state, compiles the `[1, 8]` forward pass and builds a `WindowGenerator` over it.

### TheGeneratorReadsTheExtentsOfTheStep

The generator reports a window of 8 and a vocabulary of 32.

### ARunEqualsOneTokenAtATimeOverTheGrownPrompt

Twelve greedy tokens after a prompt of three ids equal twelve runs of one token each over the prompt
grown by the tokens so far. Every token lies in 0 to 31, and the step reports one compile.

### ASeedFixesADrawnContinuation

A `TemperatureSampler` at 1.0 with two random generators of one seed gives the same twelve tokens
twice.

### APromptOrACountThatDoesNotFitIsRefused

An empty prompt, the id 32 and a count of zero tokens are each refused as `InvalidDescription`; the
messages contain "no id", "0 to 31" and "0 tokens". Nothing is generated.
