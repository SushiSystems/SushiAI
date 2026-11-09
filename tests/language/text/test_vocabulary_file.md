# tests/language/text/test_vocabulary_file.cpp


Suites: `Small_TextVocabularyFile`, `Small_TextVocabularyFileTruncation`,
`Small_TextVocabularyFileMagic`, `Small_TextVocabularyFileHeaderCount`,
`Small_TextVocabularyFileSpecialRecord`, `Medium_TextVocabularyFile`,
`Medium_TextVocabularyFileRefusal`.

Tests `serialise_vocabulary_file`, `write_vocabulary_file`, `read_vocabulary_file` and
`parse_vocabulary_file` in
`modules/language/text/include/SushiAI/language/text/vocabulary_file.hpp`. Most cases start
from a hand-written file of 49 bytes: a header of 24 bytes with the magic "SAVB" and version
1, a special section of 17 bytes that holds the name "<end_of_text>", and the one merge
(97, 98) in 8 bytes. Only the cases that read or write a file are `Medium_`.

## TheSerialiserProducesTheDocumentedBytes

`serialise_vocabulary_file` of a vocabulary with the one merge (97, 98) returns exactly those 49
bytes, in memory.

## AVocabularySurvivesItsFile

A vocabulary of three merges and the special names "<pad>" and "<mask>" is written and read
back. The read vocabulary has the same size, the same special names and the same three
merges, and its id 261 stands for " abc".

## ATruncatedFileIsRefusedAtEveryLength

Rows over the length, from 0 to 48 bytes. `parse_vocabulary_file` over that prefix fails with
`InvalidFile` and a message containing "read_vocabulary_file" and "ends at byte" followed by the
length. A prefix shorter than the header names "header of 24 bytes"; a longer one names "declares 49
bytes".

## BytesAfterTheDeclaredEndAreRefused

With three bytes appended, the parse fails with `InvalidFile` and a message containing "3
bytes follow" and "byte 49".

## AnotherMagicOrVersionIsRefused

Rows over a changed byte. The parse fails with `InvalidFile` for a first byte of 'X' ("'SAVB'", "no
vocabulary file"), for a version of 2 ("byte 4", "format version 2", "versions 1 to 1") and for a
version of 0 ("format version 0").

## AHeaderCountOutOfBoundsNamesItsOffset

Rows over a changed byte. The parse fails with `InvalidFile` for a byte-token count of 257 ("byte 8",
"257 byte tokens"), a special-token count of 0 ("byte 12", "0 special tokens") and a merge count
whose last byte is 0x7F ("byte 16", "merges").

## ASpecialTokenPastItsSectionNamesItsOffset

Rows over the name length. The parse fails with `InvalidFile` for a length of 14 ("special token
0", "byte 28", "14 bytes", "ends at byte 41"), a length of 12 ("byte 28", "'<end_of_text'") and a
length of 0 ("byte 24", "0 bytes").

## AMergeOfALaterTokenIsRefusedByTheVocabulary

With byte 42 set to 1 the merge reads as (353, 98) and the parse fails with `InvalidDescription`
and a message containing "merge 0" and "353 and 98".

## AMergeIDAboveATokenIDIsRefusedByTheFile

With byte 44 set to 0xFF the parse fails with `InvalidFile` and a message containing "merge 0" and
"byte 41".

## TheReaderNamesThePathOfAFileItRefuses

Rows over the file. `read_vocabulary_file` fails with `FileNotFound` for a path that does not
exist ("absent.savb", "read_vocabulary_file") and with `InvalidFile` for a file cut to 45 bytes
("cut.savb", "ends at byte 45", "declares 49 bytes"). Both read from a file, so the case is `Medium_`.
