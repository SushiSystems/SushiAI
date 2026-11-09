# tests/language/training/test_next_token_objective.cpp


Suites: `Small_LanguageTrainingNextTokenObjective`, `Small_LanguageTrainingNextTokenObjectiveExtents`,
`Small_LanguageTrainingNextTokenObjectiveIndex`, `Small_LanguageTrainingNextTokenObjectiveLargest`,
`Small_LanguageTrainingNextTokenObjectiveFactory`, `Small_LanguageTrainingNextTokenObjectiveOutputKey`,
`Medium_NextTokenObjective`, `Medium_NextTokenTrainer`, `Medium_DecoderOverfit`.

Tests `Language::Training::NextTokenObjective` in
`modules/language/training/include/SushiAI/language/training/next_token_objective.hpp` and its factory `make_next_token_objective`. Unless a
case says otherwise the batch has two rows of three positions over a vocabulary of five, and the
factory's context publishes one output "out" of the shape [2, 3, 5]. The small suites build
graphs and read descriptions on the host and use no device. The medium suites run on the engine.

## MakeRefusesAnExtentBelowOne

`make` fails with `InvalidDescription` and "at least 1" for a batch of 0, a length of 0 and a
vocabulary of 0, naming all three extents ("got 0, 3 and 5" and so on). The three are rows. The
getters of a made objective are not asserted.

## MakeRefusesWhatInt32CannotIndex

`make(64, 1024, 50257)` and `make(1, 2, 2147483647)` fail with `InvalidDescription`, "int32
index" and the product ("64 * 1024 * 50257", "1 * 2 * 2147483647"). Rows.

## MakeAcceptsWhatInt32CanIndex

`make(32, 256, 8192)` and `make(1, 1, 2147483647)` are ok. Rows.

## ItDeclaresAnIndexLeafAndReadsLabels

`declare` adds one leaf, "targets", an `Int32` value of the shape [6]. It requires one role,
`K_ROLE_LABELS`, with the sample shape [3], the type `Int32` and 0 classes.

## TheLossGathersOneLogProbabilityPerPosition

`Train::build_training_graph` over an `NN::Embedding` of five rows and five columns, with an
`Int32` input of the shape [2, 3], gives a loss of one element and `Int32` targets. The one
parameter has a gradient and the role `Embedding`. The operations that build the loss are not
counted; `TheLossAndItsGradientEqualTheHostComputation` checks its numbers.

## TheFactoryReadsTheModelsOutput

The objective the factory builds from an empty section declares a "targets" leaf of the shape
[6] and has the name "next_token".

## TheFactoryRefusesWhatDoesNotFit

The factory fails with `InvalidDescription` for a `smoothing` key ("objective states the key
'smoothing'"), a second published output ("one [N, T, V] output", "publishes 2"), an output of
the shape [2, 5] ("[2, 3, V] logits", "'out'"), an input sample of [4, 4, 3] ("[T] token ids
per sample") and a context of six classes ("6 token ids", "scores 5"). Rows.

## MeasureReportsThePerplexityOfTheLoss

Over a step whose loss is ln 8, `measure` writes one metric, "perplexity", that is not a
fraction and equals 8 within six rounded operations (`tools/testing/support/tolerance.hpp`).

## TheOutputKeyNamesTheLogitsAmongSeveral

Over a context that also publishes "balance" of the shape [1, 1], the section `{"output":
"out"}` builds an objective. Its loss over model outputs "balance" and "out" has one element.

## TheFactoryRefusesAnOutputItCannotFind

Over the same two outputs, the factory fails with `InvalidDescription` for an empty section
("publishes 2", "name the logits with 'output'"), `"output": "logits"` ("the output 'logits'",
"none of that name among its 2"), `"output": 3` ("'output' names a published output"),
`"output": ""` ("'output'", "is empty") and `"output": "balance"` ("[2, 3, V] logits",
"'balance'", "[1, 1]"). Rows.

## TheLossAndItsGradientEqualTheHostComputation

The model is an `Embedding` of five rows and width five, so the logits of a position are the
table row of its token. The table holds `0.3 r - 0.2 c + 0.1 r c`. The batch is two rows of three
tokens, in which token 3 occurs twice with two different labels. The step runs with SGD at a rate
of 0, so the table can be read as it was written. After one step the loss equals the mean of
`-log p(label)` over the six positions, and each element of the table's gradient equals
`(softmax - one_hot) / 6` summed by table row. The host computation is `work_on_host`; the
comparison is a float32 absolute bound from `tools/testing/support/tolerance.hpp`.

## FitLowersTheLossAndReportsPerplexity

The fixture fits the bigram table to one batch for 30 epochs through `Train::Trainer` and the
objective's own feed. The report holds 30 steps and the last loss is below the first. The last
metrics hold one entry, `perplexity`, which is not a fraction and equals `exp` of the last loss.
The compile count and the report's text are not asserted.

## ALabelOutsideTheVocabularyStopsTheFirstStep

A label of 9 with five tokens is refused as `Incompatible`. The message contains "position 3", "is
9" and "holds 5 tokens".

## ATwoBlockDecoderOverfitsOneBatch

A typed case over two decoders, `Learned` and `Rotary`. Each has a token table of 64 rows, two
causal blocks of width 32 with two heads and a feed-forward width of 64, and a head tied to the
token table; `Learned` adds a learned position table over 16 positions and `Rotary` uses rotary
attentions. The case writes a token file into the system's temporary directory, reads it back as
one batch of four rows through `TokenDataset`, and removes the file. Training is AdamW with a
weight decay of 0.01, gradients clipped to a norm of 1, and a cosine schedule from 3e-3 to 3e-4
with 20 warm-up iterations; the parameters start from Xavier values. Over 400 updates every loss
is finite, the first is within 0.5 of ln(64), the last is below 0.1 of the first, and its
perplexity is below the first's. The step reports one compile and the optimizer 400 steps. The
state names and the decay flags are not asserted.
