# tests/language/training

The tests of the module `language/training`, one file per header of `modules/language/training/include/`. The suite prefix of a case gives its ctest label.

## test_expert_balance_objective.cpp

Suites: `Unit_LanguageTrainingExpertBalanceObjective`,
`Unit_LanguageTrainingExpertBalanceMetrics`.

Tests `Language::Training::ExpertBalanceObjective` in
`modules/language/training/include/SushiAI/language/training/expert_balance_objective.hpp` and
its factory `make_expert_balance_objective`. The blocks the cases describe have four experts.
The fixture of the metrics suite provides an engine and allocates `Float32` usage tensors of
the shape [1, 4].

### MakeKeepsItsOutputsAndRefusesNone

An objective made of the balance "balance" and the usage "usage" of four experts returns that
name from `balances()` and one usage of four experts from `usages()`. Its name is
"expert_balance" and `is_fed_by_batch()` is true. `make` fails with `InvalidDescription` for no
balance ("at least one balance output", "0 and 1"), no usage ("1 and 0"), an empty balance
name ("no name") and a usage of 0 experts ("'usage' of 0").

### ItDeclaresNoLeafAndReadsNoRole

After `declare`, the leaves list no leaf and no required role, and the graph has no node.

### TheLossIsTheMeanOfTheBalanceOutputs

The model outputs "balance2" and "balance3" are [1, 1] values. The loss of an objective over
"balance3" alone has the shape [1] and comes from a RESHAPE node that reads that output, with
no SCALE node in the graph. The loss of an objective over both has the shape [1] and comes
from a SCALE node whose attribute is 0.5 and whose input comes from an ADD node.

### TheFactoryReadsTheOutputsByTheirNames

Over a context that publishes "output", "balance2", "usage2", "balance3" and "usage3", the
factory builds an `ExpertBalanceObjective` whose balances are "balance2" and "balance3" and
whose usages are "usage2" and "usage3", the second of four experts.

### TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `weight` key ("objective states the key
'weight'"), for a context that publishes only "output" ("begin with 'balance'", "'usage'", "0
and 0", "among its 1") and for a "balance2" output of the shape [1, 2] ("'balance2'", "one
element", "[1, 2]").

### MeasureReportsTheSmallestAndLargestShare

The step publishes "usage2" with four shares of 0.25 and "usage3" with the shares 0.125, 0.5,
0.0625 and 0.3125. `measure` writes two metrics, "expert_usage_min" of 0.0625 and
"expert_usage_max" of 0.5, each within 1e-7 and each marked as a fraction.

### MeasureRefusesAUsageItCannotRead

Over a step that publishes only "usage2" with four shares, `measure` fails with
`InvalidDescription` for an objective that reads "usage9" ("'usage9'", "did not publish") and
with `Incompatible` for one that expects eight experts in "usage2" ("the usage of the
experts", "8 values", "holds 4"). No metric is written.

## test_next_token_objective.cpp

Suites: `Unit_LanguageTrainingNextTokenObjective`.

Tests `Language::Training::NextTokenObjective` in
`modules/language/training/include/SushiAI/language/training/next_token_objective.hpp` and its
factory `make_next_token_objective`. Unless a case says otherwise the batch has two rows of
three positions over a vocabulary of five, and the factory's context publishes one output
"out" of the shape [2, 3, 5].

### MakeKeepsItsExtentsAndRefusesNone

`batch()`, `length()` and `vocabulary()` return 2, 3 and 5, and the name is "next_token".
`make` fails with `InvalidDescription` and "at least 1" for a batch of 0 and for a vocabulary
of 0.

### MakeRefusesWhatInt32CannotIndex

`make(64, 1024, 50257)` fails with `InvalidDescription` and a message containing "int32 index"
and "64 * 1024 * 50257". `make(32, 256, 8192)` and `make(1, 1, 2147483647)` are ok, and
`make(1, 2, 2147483647)` fails with "int32 index".

### ItDeclaresAnIndexLeafAndReadsLabels

`declare` adds one leaf, "targets", an `Int32` value of the shape [6]. It requires one role,
`K_ROLE_LABELS`, with the sample shape [3], the type `Int32` and 0 classes.

### TheLossGathersOneLogProbabilityPerPosition

`Train::build_training_graph` over an `NN::Embedding` of five rows and five columns, with an
`Int32` input of the shape [2, 3], gives a loss of one element and `Int32` targets. The graph
holds no NLL_LOSS, one LOG_SOFTMAX, two GATHER_ROWS, two SCATTER_ADD_ROWS and one
LOG_SOFTMAX_BACKWARD node. The one parameter has a gradient and the role `Embedding`.

### TheFactoryReadsTheModelsOutput

The objective the factory builds from an empty section declares a "targets" leaf of the shape
[6] and has the name "next_token".

### TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `smoothing` key ("objective states the key
'smoothing'"), a second published output ("one [N, T, V] output", "publishes 2"), an output of
the shape [2, 5] ("[2, 3, V] logits", "'out'"), an input sample of [4, 4, 3] ("[T] token ids
per sample") and a context of six classes ("6 token ids", "scores 5").

### ABatchAloneFeedsTheObjective

`is_fed_by_batch()` is true.

### MeasureReportsThePerplexityOfTheLoss

Over a step whose loss is ln 8, `measure` writes one metric, "perplexity", that is not a
fraction and equals 8 within a relative 1e-12.

### TheOutputKeyNamesTheLogitsAmongSeveral

Over a context that also publishes "balance" of the shape [1, 1], the section `{"output":
"out"}` builds an objective. Its loss over model outputs "balance" and "out" has one element,
and the graph holds one LOG_SOFTMAX node.

### TheFactoryRefusesAnOutputItCannotFind

Over the same two outputs, the factory fails with `InvalidDescription` for an empty section
("publishes 2", "name the logits with 'output'"), `"output": "logits"` ("the output 'logits'",
"none of that name among its 2"), `"output": 3` ("'output' names a published output"),
`"output": ""` ("'output'", "is empty") and `"output": "balance"` ("[2, 3, V] logits",
"'balance'", "[1, 1]").

## test_next_token_targets.cpp

Suites: `Unit_LanguageTrainingNextTokenTargets`, `Unit_LanguageTrainingPerplexity`.

Tests `Language::Training::fill_next_token_targets` in
`modules/language/training/include/SushiAI/language/training/next_token_targets.hpp`, and
`Language::Training::perplexity` and `Language::Training::next_token_metrics`. The fixture of
the first suite provides an engine, `Int32` labels of the shape [2, 3] and a leaf of six
elements that starts at -7 everywhere. The vocabulary has five tokens.

### APositionIndexesItsLabelAmongEveryScore

The labels 4, 0, 2, 1, 1, 3 fill the leaf with 4, 5, 12, 16, 21, 28.

### ALabelOutsideTheVocabularyIsRefused

`fill_next_token_targets` fails with `Incompatible` for a label of 5 at position 4 ("position
4 is 5", "holds 5 tokens") and for a label of -1 at position 2 ("position 2 is -1"). Every
element of the leaf is still -7.

### AFillThatDoesNotFitIsRefused

The fill fails with `Incompatible` for labels of the shape [2, 2] ("holds 4 labels", "expects
6"), for `Float32` labels ("float32", "int32"), for a vocabulary of 2147483647 ("6 positions",
"cannot be indexed") and for a vocabulary of 0 ("cannot be indexed").

### ThePerplexityIsTheExponentialOfTheLoss

`perplexity(0)` is exactly 1, `perplexity(ln 64)` is 64 within a relative 1e-12 and
`perplexity(1e6)` is infinite. The case is in the suite `Unit_LanguageTrainingPerplexity`.

### TheMetricsAreTheLossAndThePerplexity

`next_token_metrics(ln 8)` returns "loss" with exactly ln 8 and "perplexity" with 8 within a
relative 1e-12, neither marked as a fraction. The case is in the suite
`Unit_LanguageTrainingPerplexity`.

Suites: `Integration_NextTokenObjective`, in `sushiai_language_training_tests`.

The model is an `Embedding` of five rows and width five, so the logits of a position are the table
row of its token. The table holds `0.3 r - 0.2 c + 0.1 r c`. The batch is two rows of three tokens,
in which token 3 occurs twice with two different labels. The step runs with SGD at a rate of 0, so
the table can be read as it was written.

### TheLossAndItsGradientEqualTheHostComputation

The graph binds one parameter slot. After one step the loss equals the mean of `-log p(label)` over
the six positions, and each element of the table's gradient equals `(softmax - one_hot) / 6` summed
by table row, both within 1e-5.

Suites: `Integration_NextTokenTrainer`, in `sushiai_language_training_tests`.

The fixture fits a bigram table, an `Embedding` of five rows and width five, to one batch of two
rows of three tokens through `Train::Trainer` and the next-token objective's own feed, for 30
epochs.

### FitLowersTheLossAndReportsPerplexity

The report holds 30 steps and one compile, and the last loss is below the first. The last metrics
hold one entry, `perplexity`, which is not a fraction and equals `exp` of the last loss within 1e-9.
The report's text names it.

### ALabelOutsideTheVocabularyStopsTheFirstStep

A label of 9 with five tokens is refused as `Incompatible`. The message contains "position 3", "is
9" and "holds 5 tokens".

Suites: `Integration_DecoderOverfit`, in `sushiai_language_training_tests`.

The model is a decoder with a token table of 64 rows, a learned position table over 16 positions,
two causal blocks of width 32 with two heads and a feed-forward width of 64, and a head tied to the
token table. The test writes a token file into the system's temporary directory, reads it back as
one batch of four rows through `TokenDataset`, and removes the file afterwards. Training is AdamW
with a weight decay of 0.01, gradients clipped to a norm of 1, and a cosine schedule from 3e-3 to
3e-4 with 20 warm-up iterations. The parameters start from Xavier values.

### ATwoBlockDecoderOverfitsOneBatch

The dataset holds one batch. The state has no `head.linear.weight`; the token table, the position
table, the head's normalisation weight and a `qkv` bias do not decay, and a `qkv` weight does. Over
400 updates every loss is finite. The first loss is within 0.5 of ln(64), the last is below 0.1 of
the first, and its perplexity is below the first's. The step reports one compile and the optimizer
400 steps.

Suites: `Integration_RotaryDecoderOverfit`, in `sushiai_language_training_tests`.

The setup is that of `test_decoder_overfit.cpp` without the position table: a token table of 64
rows, two rotary blocks of width 32 with two heads and a feed-forward width of 64, one batch of four
rows of 16 tokens read from a token file, and AdamW with clipping and a cosine schedule.

### ATwoBlockRotaryDecoderOverfitsOneBatch

The dataset holds one batch. The training graph holds four host tables, the cosines and the sines of
each block. Every state entry is a parameter, and one slot is bound for each entry. Over 400 updates
every loss is finite; the first is within 0.5 of ln(64) and the last is below 0.1 of the first. The
step reports one compile.
