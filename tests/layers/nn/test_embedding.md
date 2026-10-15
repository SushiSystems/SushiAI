# tests/layers/nn/test_embedding.cpp


Suites: `Medium_GatherRows`.

The file runs GATHER_ROWS, SCATTER_ADD_ROWS and the `Embedding` layer on an engine. The tables hold
the dyadic values 0.25, 0.5, 0.75 and so on, whose copies and sums are exact, so results are
compared with a host loop at a tolerance of 0. Indices are written into INT32 leaves.

## GatherCopiesTheRowsItsIndexNames

A `[2, 3]` index over a `[5, 3]` table, with row 2 named three times, gives the rows a host loop
copies.

## AReplayReadsTheIndexOfItsOwnRun

One compiled step over a FLOAT64 `[6, 2]` table is run with two different indices of four. Each run
returns the rows of the index written before it.

## ScatterAddSumsTheRowsOfARepeatedIndex

Six FLOAT64 rows are scattered into five table rows with the index `{3, 1, 3, 3, 0, 1}`, and the
step is run twice. The result equals the host sums, and rows 2 and 4, which no index names, are
zero.

## TheTableGradientIsBitIdenticalAcrossReplays

A FLOAT32 `[13, 8]` table is gathered with 256 indices, so each row is named about twenty times, and
the loss is differentiated at the table. The bytes of the gradient are the same after each of four
runs.

## AnEmbeddingYieldsTheRowsOfItsTable

An `Embedding` of 7 rows and width 4 over a `[2, 3]` batch of ids has the output shape `[2, 3, 4]`,
and its output holds the table rows the ids name.

## test_embedding.cpp


The Embedding layer's trace over an int32 index, its table parameter, its shape rule and its factory.

Suites: `Small_Embedding`, `Medium_EmbeddingDeathTest`.

Tests `NN::Embedding` in `modules/layers/nn/include/SushiAI/nn/embedding.hpp` and its factory
`NN::make_embedding_node`. Every layer has a table of 7 rows of length 3. A `static_assert`
holds that `Embedding` is a `TraceModule` and is neither a `Module` nor a `ModalModule`.

## ATraceDeclaresTheTableAndEmitsOneGather

Over a [2, 5] int32 input the trace is ok and adds two nodes, one of them GATHER_ROWS. That node
reads the weight and then the indices, and its result is a float32 value of shape [2, 5, 3].

## TheTableIsAnEmbeddingDrawnFromANormalOfFixedDeviation

After a trace of a float64 layer the weight is declared with the shape [7, 3], the type float64
and the role `ParameterRole::Embedding`. Its initialiser is `Normal` with the deviation
`K_EMBEDDING_DEVIATION`, which is 0.02, and its fan-in and fan-out are both 3. The weight's
value has the lifetime `Parameter` and requires a gradient, and the result is float64.

## TheWalkYieldsTheTable

`NN::walk_state` under the prefix `tokens` visits one name, `tokens.weight`, as a parameter of
shape [7, 3], and visits no buffer.

## TwoGraphsNameTheSameTable

After traces of one layer into two graphs, `weight.state()` is the same `StateID` and is not
`K_NO_STATE`, and `weight.id_in` returns a value in each graph.

## AnInputThatIsNotInt32Fails

A float32 input and an int64 input each fail with `InvalidDescription` and a message containing
"Embedding(7, 3)", "int32 input of indices" and the type given. `forward` returns its input,
the graph gains no value and no node, the weight stays undeclared and the graph has a refusal.

## AnInputOfTheHighestRankFails

A rank-6 int32 input fails with `InvalidDescription`, "rank 1 to 5" and
"[1, 1, 1, 1, 1, 1]", and the weight stays undeclared.

## ALayerDescribedWrongFails

`num_embeddings` 0 and `embedding_dim` -1 each fail with "positive extents" and the field with
its value. An int32 table fails with "half, float32 or float64 table" and "int32". The weights
of the first and the third layer stay undeclared.

## OutputShapeAppendsTheRowLength

`output_shape` maps [5] to [5, 3], [4, 2] to [4, 2, 3] and [1, 1, 1, 1] to [1, 1, 1, 1, 3].

## TheFactoryReadsTheTwoExtents

`make_embedding_node` over one [5] input with `num_embeddings` 11 and `embedding_dim` 4 returns
a `ModuleNode<Embedding>` whose layer holds 11, 4 and float32. The node reads one input and its
one port has the shape [5, 4].

## TheFactoryRefusesARowThatDoesNotFit

Five rows fail with `InvalidDescription` and a message naming "layer 3": a `padding` key, an
absent `embedding_dim`, a `num_embeddings` of 0 ("positive"), two inputs ("reads one input",
"given 2") and a rank-5 input ("[1, 1, 1, 1, 1]", "rank 1 to 4").

## ATraceOverAValueTheGraphLacksIsFatal

`forward` over value id 7 in a graph of one value stops the process with a report containing
"Embedding(7, 3)", "value 7" and "holds 1 values".

## OutputShapeOverARankFiveSampleIsFatal

`output_shape([1, 1, 1, 1, 1])` stops the process with a report containing
"Embedding::output_shape" and "at most 4".
