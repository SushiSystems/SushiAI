# tests/language/decoder_models/test_decoder_model.cpp


Suites: `Small_DecoderModel`.

Tests the model files of
`modules/language/decoder_models/include/SushiAI/language/decoder_models/decoder_model.hpp`
against the constants beside them, and that each builds through `build_decoder`. The int32 input
of `K_DECODER_MODEL` is pinned in `tests/language/registration/test_registration.cpp`. No case
needs a device.

## count_rows

Counts the rows of a parsed model whose type is a given one.

## TheTokenTableHasTheVocabularyAndTheWidth

The first row of `decoder_spec()` is an "embedding" whose `num_embeddings` equals
`K_DECODER_VOCABULARY` and whose `embedding_dim` equals `K_DECODER_DIM`. The file holds
`K_DECODER_BLOCKS` "transformer_block" rows and one "position_embedding" row.

## TheRotaryFilesHoldNoPositionTable

`K_ROTARY_DECODER_MODEL` holds no "position_embedding" row, `K_DECODER_BLOCKS` blocks and an
input of `K_DECODER_WINDOW` positions. `K_WIDE_ROTARY_DECODER_MODEL` has an input of
`K_WIDE_DECODER_WINDOW` positions and `K_SHALLOW_ROTARY_DECODER_MODEL` one block.

## EveryFileBuilds

`build_decoder` succeeds for each of the four model files.
