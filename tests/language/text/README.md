# tests/language/text

The tests of the module `language/text`, one file per header of `modules/language/text/include/`. The suite prefix of a case gives its ctest label.

## test_bpe_trainer.cpp

Suites: `Unit_TextBpeTrainer`.

Tests `Language::Text::train_bpe` in
`modules/language/text/include/SushiAI/language/text/bpe_trainer.hpp`. Unless a case says
otherwise the options state a vocabulary size and no special name, so the first merged token
has the id 257. The generated corpora are drawn with the seed 20261011.

### ItLearnsTheHandWorkedMergesOfATinyCorpus

Training on "abababc" to a size of 300 learns two merges, (97, 98) then (257, 257). The
vocabulary has 259 tokens and token 258 stands for "abab".

### PairsOfOneCountMergeBySmallerLeftThenRightID

Training on "ab ab cd cd" learns three merges in the order (32, 99), (97, 98), (257, 100).

### NoMergeCrossesTheStartOfABlankRun

Training on "a b a b a b" learns two merges, (32, 98) then (32, 97).

### TheSameCorpusGivesTheSameVocabulary

Two runs to a size of 400 over one corpus of 20000 bytes drawn from "abcdef", the blank and
the newline learn at least one merge and the same merges in the same order.

### ARunStoppedEarlyLearnsTheFirstMergesOfALongerRun

Over a corpus of 4000 bytes drawn from "abc" and the blank, a run to a size of 270 learns at
least one and at most 13 merges, and each equals the merge at the same index of a run to a
size of 290.

### ItStopsAtTheSizeAskedFor

Training on "abababc" to a size of 258 learns one merge, (97, 98).

### APairThatOccursOnceIsNotMerged

Training on "abcdef" to a size of 300 gives a vocabulary of 257 tokens, and so does training
on the empty text.

### SpecialTokensPrecedeTheMergedTokens

With the special name "<pad>", training on "abababc" learns two merges, the second of which
is (258, 258), and token 259 stands for "abab".

### ItRefusesASizeOutOfBounds

`train_bpe` fails with `InvalidDescription` for a size of 256 ("train_bpe", "256 tokens", "257
to"), a size of `K_MAX_VOCABULARY_SIZE + 1` ("train_bpe", "asked for") and an empty special
name ("special token 1").

## test_codec.cpp

Suites: `Unit_TextCodec`.

Tests `Language::Text::encode_text` and `Language::Text::decode_tokens` in
`modules/language/text/include/SushiAI/language/text/codec.hpp`. A round trip encodes a text,
checks that every id is in the vocabulary and is not a special token, and checks that the ids
decode to the same bytes.

### RandomBytesSurviveARoundTrip

With a vocabulary of 400 tokens trained on 20000 drawn bytes, the round trip holds for 64
texts of random bytes of up to 1999 bytes each, for the training corpus, for the empty text
and for eight bytes that are not valid UTF-8.

### ATextLongerThanOneChunkSurvivesARoundTrip

The vocabulary has the merge (97, 97) and seven merges that each join the token before with
itself. A text of 5000 'a' bytes makes the round trip and encodes to 21 ids: the first is 264,
the twentieth is 263 and the last is 259.

### MergesAreAppliedInTheOrderTheyWereLearned

With the merges (97, 98) and (257, 257), "abababc" encodes to 258, 257, 99. With the merges
(98, 99) and (97, 98), "abc ab" encodes to 97, 257, 32, 258.

### ATrainedCorpusEncodesToItsTrainedTokens

With the vocabulary trained on "ab ab cd cd" to a size of 300, that text encodes to 258, 32,
258, 259, 259.

### EncodeAppendsToTheIDsItIsGiven

Encoding "A" into a vector that holds 7 leaves 7, 65.

### ASpecialTokenDecodesToNoByte

Decoding the ids 72, `K_END_OF_TEXT`, 105 into a string that holds ">" leaves ">Hi".

### DecodeRefusesAnIDOutsideTheVocabulary

Over a vocabulary of 257 tokens, `decode_tokens` fails with `InvalidDescription` for the ids
72, 9999, 105 ("decode_tokens", "9999", "index 1", "[0, 257)") and for the id -3 ("-3", "index
0").

## test_stream_encoder.cpp

Suites: `Unit_TextStreamEncoder`.

Tests `Language::Text::encode_stream` in
`modules/language/text/include/SushiAI/language/text/stream_encoder.hpp` against
`encode_text`.

### EveryPieceSizeGivesTheIDsOfTheWholeText

The text is 40 repetitions of two lines with runs of blanks, a tab and a carriage return, then
a word of 600 'x' bytes and " the end". With a vocabulary of 300 tokens trained on it,
`encode_stream` gives the ids of `encode_text` for pieces of 1, 7, 64 and 1000 bytes and of
`K_STREAM_PIECE_BYTES`.

### AnEmptyStreamAppendsNothing

`encode_stream` over an empty stream leaves a vector that holds 7 unchanged. Over a stream of
"ab" it then leaves 7, 'a', 'b'.

## test_vocabulary.cpp

Suites: `Unit_TextVocabulary`.

Tests `Language::Text::Vocabulary` in
`modules/language/text/include/SushiAI/language/text/vocabulary.hpp`. A merge is written here
as (left, right).

### AVocabularyWithoutMergesHoldsTheBytesAndEndOfText

A vocabulary made of no merges has 257 tokens and a `first_merged()` of 257. The ids 0 and 255
are of the kind `Byte`, id 65 stands for "A" and id 255 for the byte 0xFF. `K_END_OF_TEXT` is
of the kind `Special` and stands for no byte, and `find_special(K_END_OF_TEXT_NAME)` is 256.
`contains` is true for 256 and false for 257 and -1.

### MergedTokensTakeIDsInTheOrderOfTheirMerges

With the merges (97, 98) and (257, 99) the vocabulary has 259 tokens, id 257 is of the kind
`Merged` and stands for "ab", and id 258 stands for "abc". `find_merge` returns 257 for
(97, 98) and 258 for (257, 99), and no value for (98, 97) and (-1, 97). `merges()` holds two
merges, the second of which is (257, 99).

### SpecialTokensFollowEndOfText

With one merge and the special names "<pad>" and "<mask>" the vocabulary has 260 tokens.
`find_special` returns 257 and 258 for the two names and no value for "<absent>".
`first_merged()` is 259, id 259 stands for "ab", and `special_names()` holds three names, the
first of which is `K_END_OF_TEXT_NAME`.

### MakeRefusesAMergeOfATokenItMayNotJoin

`Vocabulary::make` fails with `InvalidDescription` for the merge (97, 257) at index 0
("Vocabulary", "merge 0", "97 and 257"), (259, 97) at index 1 ("merge 1", "259 and 97"),
(`K_END_OF_TEXT`, 97) ("merge 0", "256 and 97") and (-1, 97) ("merge 0", "-1 and 97").

### MakeRefusesAPairNamedTwice

Two merges of (97, 98) fail with `InvalidDescription` and a message containing "merge 1", "97
and 98" and "earlier merge".

### MakeRefusesATokenAboveTheByteLimit

Eight merges that each double the token before build a vocabulary whose id 264 stands for 256
'a' bytes. A ninth such merge fails with `InvalidDescription` and a message containing "merge
8", "512 bytes" and "at most 256".

### MakeRefusesASpecialNameOutOfBounds

`Vocabulary::make` fails with `InvalidDescription` for an empty name ("special token 1", "0
bytes"), a name of 256 bytes ("256 bytes", "1 to 255"), "<pad>" given twice ("'<pad>'", "named
twice"), `K_END_OF_TEXT_NAME` given as a name ("named twice") and 256 names ("257 special
tokens", "at most 256").

### AnIDOutsideTheVocabularyStopsTheProcess

Over a vocabulary of 257 tokens, `bytes_of(257)` stops the process with a report containing
"Vocabulary", "257" and "[0, 257)", and `kind_of(-1)` with "Vocabulary" and "-1".

## test_vocabulary_file.cpp

Suites: `Unit_TextVocabularyFile`.

Tests `write_vocabulary_file`, `read_vocabulary_file` and `parse_vocabulary_file` in
`modules/language/text/include/SushiAI/language/text/vocabulary_file.hpp`. Most cases start
from a hand-written file of 49 bytes: a header of 24 bytes with the magic "SAVB" and version
1, a special section of 17 bytes that holds the name "<end_of_text>", and the one merge
(97, 98) in 8 bytes.

### TheWriterProducesTheDocumentedBytes

`write_vocabulary_file` of a vocabulary with the one merge (97, 98) writes exactly those 49
bytes.

### AVocabularySurvivesItsFile

A vocabulary of three merges and the special names "<pad>" and "<mask>" is written and read
back. The read vocabulary has the same size, the same special names and the same three
merges, and its id 261 stands for " abc".

### ATruncatedFileIsRefusedAtEveryLength

For every length from 0 to 48 bytes, `parse_vocabulary_file` over that prefix fails with
`InvalidFile` and a message containing "read_vocabulary_file" and "ends at byte". A prefix of
10 bytes names "ends at byte 10" and "header of 24 bytes"; one of 30 names "ends at byte 30"
and "declares 49 bytes".

### BytesAfterTheDeclaredEndAreRefused

With three bytes appended, the parse fails with `InvalidFile` and a message containing "3
bytes follow" and "byte 49".

### AnotherMagicOrVersionIsRefused

The parse fails with `InvalidFile` for a first byte of 'X' ("'SAVB'", "no vocabulary file"),
for a version of 2 ("byte 4", "format version 2", "versions 1 to 1") and for a version of 0
("format version 0").

### AHeaderCountOutOfBoundsNamesItsOffset

The parse fails with `InvalidFile` for a byte-token count of 257 ("byte 8", "257 byte
tokens"), a special-token count of 0 ("byte 12", "0 special tokens") and a merge count whose
last byte is 0x7F ("byte 16", "merges").

### ASpecialTokenPastItsSectionNamesItsOffset

The parse fails with `InvalidFile` for a name length of 14 ("special token 0", "byte 28", "14
bytes", "ends at byte 41"), a name length of 12 ("byte 28", "'<end_of_text'") and a name
length of 0 ("byte 24", "0 bytes").

### AMergeOfALaterTokenIsRefused

With byte 42 set to 1 the merge reads as (353, 98) and the parse fails with
`InvalidDescription` and a message containing "merge 0" and "353 and 98". With byte 44 set to
0xFF the parse fails with `InvalidFile` and a message containing "merge 0" and "byte 41".

### TheReaderNamesThePathOfAFileItRefuses

`read_vocabulary_file` fails with `FileNotFound` for a path that does not exist ("absent.savb",
"read_vocabulary_file") and with `InvalidFile` for a file cut to 45 bytes ("cut.savb", "ends
at byte 45", "declares 49 bytes").
