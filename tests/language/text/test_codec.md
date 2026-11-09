# tests/language/text/test_codec.cpp


Suites: `Small_TextCodec`, `Small_TextCodecEdgeText`, `Small_TextCodecRefusal`.

Tests `Language::Text::encode_text` and `Language::Text::decode_tokens` in
`modules/language/text/include/SushiAI/language/text/codec.hpp`. A round trip encodes a text,
checks that every id is in the vocabulary and is not a special token, and checks that the ids
decode to the same bytes.

## RandomBytesSurviveARoundTrip

A property test with the fixed seed 20261012 and 64 rounds. With a vocabulary of 400 tokens
trained on 20000 drawn bytes, the round trip holds for 64 texts of random bytes of up to 1999
bytes each. The loop is in the helper `expect_random_texts_round_trip`.

## AnEdgeTextSurvivesARoundTrip

Rows over a fixed text: the training corpus, the empty text and eight bytes that are not valid
UTF-8. Each makes the round trip over the vocabulary of the case above.

## ATextLongerThanOneChunkSurvivesARoundTrip

The vocabulary has eight merges, the first (97, 97) and seven that each join the token before
with itself. A text of 5000 'a' bytes makes the round trip.

## ATextLongerThanOneChunkEncodesChunkByChunk

Over the same vocabulary, 5000 'a' bytes encode to 21 ids: the first is 264, the twentieth is 263
and the last is 259.

## TheFirstMergeAppliesFirstFromTheLeft

With the merges (97, 98) and (257, 257), "abababc" encodes to 258, 257, 99.

## AnEarlierMergeWinsAnOverlapOverALaterOne

With the merges (98, 99) and (97, 98), "abc ab" encodes to 97, 257, 32, 258.

## ATrainedCorpusEncodesToItsTrainedTokens

With the vocabulary trained on "ab ab cd cd" to a size of 300, that text encodes to 258, 32,
258, 259, 259.

## EncodeAppendsToTheIDsItIsGiven

Encoding "A" into a vector that holds 7 leaves 7, 65.

## ASpecialTokenDecodesToNoByte

Decoding the ids 72, `K_END_OF_TEXT`, 105 into a string that holds ">" leaves ">Hi".

## DecodeRefusesAnIDOutsideTheVocabulary

Rows over the ids. Over a vocabulary of 257 tokens, `decode_tokens` fails with `InvalidDescription`
for the ids 72, 9999, 105 ("decode_tokens", "9999", "index 1", "[0, 257)") and for the id -3
("decode_tokens", "-3", "index 0").
