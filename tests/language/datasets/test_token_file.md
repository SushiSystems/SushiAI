# tests/language/datasets/test_token_file.cpp


Suites: `Small_LanguageDatasetsTokenFile`, `Small_LanguageDatasetsTokenFileWriterRefusal`,
`Small_LanguageDatasetsTokenFileTruncation`, `Small_LanguageDatasetsTokenFileHeaderRefusal`,
`Small_LanguageDatasetsTokenFileIdRefusal`, `Medium_LanguageDatasetsTokenFile`.

Tests `write_token_file`, `read_token_file_header` and `read_token_range` in
`modules/language/datasets/include/SushiAI/language/datasets/token_file.hpp`. The format is
tested over `std::ostringstream` and `std::istringstream`, which are the in-memory adapters of
the stream overloads, so those cases run `Small_`. The `Medium_` cases reach the disk: the path
overloads, the file that a refusal leaves behind or does not, and the name of the file in a
message. Most cases start from a hand-written file of 36 bytes: a header of 24 bytes (the magic
"SATK", version 1, a vocabulary of 300, an id width of 4 and a count of 3) and the ids 1, 258
and 0.

## TheWriterProducesTheDocumentedBytes

`write_token_file` of the ids 1, 258 and 0 over 300 tokens writes exactly those 36 bytes to a
stream.

## TheHeaderSurvivesTheStream

A stream of the ids 0 to 39999 over 40000 tokens has a header with a vocabulary size of 40000
and a token count of 40000.

## ARangeOfIDsComesBackAtItsOffset

Over that stream, `read_token_range` of five ids from 16382 returns 16382 to 16386, the range
from 39995 ends at 39999, and an empty range at 40000 is ok.

## TheWriterRefusesAnIDOutsideTheVocabulary

One promise over three rows: `write_token_file` to a stream fails with `InvalidDescription` for
the id 300 over 300 tokens ("write_token_file", "id 300", "index 1", "[0, 300)"), for the id -2
("id -2", "index 0") and for a vocabulary of 0 ("vocabulary of 0 tokens"), and writes no byte.

## ATruncatedFileIsRefusedAtEveryLength

One promise over the lengths 0 to 35: `read_token_file_header` over the stream cut to that length
fails with `InvalidFile` and a message containing "read_token_file_header" and "ends at byte"
and the length.

## ACutNamesTheHeaderOrTheDeclaredBytes

A cut at 10 bytes names "ends at byte 10" and "header of 24 bytes"; a cut at 30 names "ends at
byte 30" and "declares 36 bytes".

## BytesAfterTheDeclaredEndAreRefused

With two bytes appended, `read_token_file_header` fails with `InvalidFile` and a message
containing "2 bytes follow" and "byte 36".

## AHeaderFieldOutOfBoundsNamesItsOffset

One promise over five rows: `read_token_file_header` fails with `InvalidFile` for a first byte of
'X' ("'SATK'", "no token file"), a version of 2 ("byte 4", "format version 2", "versions 1 to
1"), a vocabulary of 0 ("byte 8", "vocabulary of 0 tokens"), an id width of 8 ("byte 12", "ids of
8 bytes") and a count whose last byte is 0x7F ("byte 16", "ids").

## AnIDOutsideTheVocabularyNamesItsOffset

One promise over two rows. With the vocabulary size lowered to 258, `read_token_range` of three
ids fails with `InvalidFile` ("read_token_range", "id 258", "byte 28"). In a stream whose third
id is the bytes FF FF FF FF, the range at id 2 fails with "id -1" and "byte 32".

## AValidIDReadsBackAsWritten

With the vocabulary size lowered to 258, `read_token_range` of the first id alone returns 1.

## ARangeOutsideTheFileIsRefused

`read_token_range` of two ids from id 2 of a file of three fails with `InvalidDescription` and a
message containing "2 ids from id 2" and "file of 3".

## AFileSurvivesTheDisk

`Medium_`. `write_token_file` of the ids 1, 258 and 0 to a path writes the 36 bytes. The header
reads back with a vocabulary of 300 and 3 ids, and the range of two ids from id 1 is 258 and 0.

## TheWriterLeavesNoFileForARefusedID

`Medium_`. `write_token_file` to a path fails with `InvalidDescription` for the id 300 over 300
tokens, and no file exists afterwards.

## ARefusalOfTheHeaderNamesTheFile

`Medium_`. With two bytes appended to a file, `read_token_file_header` of its path fails with
`InvalidFile` and a message containing "long.satk", "2 bytes follow" and "byte 36".

## AnAbsentFileIsRefused

`Medium_`. Over a path that does not exist, `read_token_file_header` fails with `FileNotFound`
("absent.satk") and `read_token_range` with `InvalidFile` ("absent.satk", "cannot be opened").
