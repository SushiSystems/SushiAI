# tests/layers/train/test_trainer/test_token_feed.cpp


Suites: `Integration_TokenFeed`.

A materialised dataset named "tokens" holds int32 ids, two rows of five per batch, all below 7. The
cases feed it to an int32 input leaf and to a float32 one.

## AnInt32BatchOfADatasetFeedsAnEmbedding

An `Embedding` of 7 rows and width 4 is traced for a `[2, 5]` int32 input, and its state has one
entry. The input leaf describes itself as int32 and `match_input` accepts the dataset. After one
evaluation step over the batch, each output row equals the table row of its id, with a tolerance of
0.

## FitRefusesAnInt32DatasetForAFloat32Leaf

A classifier MLP whose input leaf is float32 `[5]` is fitted to the same dataset. `fit` is refused
as `Incompatible`; the message names 'tokens', "as int32 [5] per sample" and "compiled for float32
[5] per sample". The trainer has taken zero steps.
