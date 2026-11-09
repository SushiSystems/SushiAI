# tests/language/text/test_stream_encoder.cpp


Suites: `Small_TextStreamEncoder`, `Small_TextStreamEncoderPieces`.

Tests `Language::Text::encode_stream` in
`modules/language/text/include/SushiAI/language/text/stream_encoder.hpp` against
`encode_text`. The stream is a `std::istringstream`, so no file is read.

## EveryPieceSizeGivesTheIDsOfTheWholeText

Rows over the piece size: 1, 7, 64, 1000 and `K_STREAM_PIECE_BYTES`. The text is 40 repetitions
of two lines with runs of blanks, a tab and a carriage return, then a word of 600 'x' bytes and
" the end". With a vocabulary of 300 tokens trained on it, `encode_stream` gives the ids of
`encode_text`.

## AnEmptyStreamAppendsNothing

`encode_stream` over an empty stream leaves a vector that holds 7 unchanged.

## AStreamAppendsTheIDsOfItsBytes

Over a stream of "ab", `encode_stream` leaves a vector that held 7 as 7, 'a', 'b'.
