# tests/language/datasets/test_decision_file.cpp

Suites: `Small_LanguageDatasetsDecisionFile`, `Small_LanguageDatasetsDecisionFileWriterRefusal`,
`Small_LanguageDatasetsDecisionFileTruncation`, `Small_LanguageDatasetsDecisionFileRefusal`,
`Medium_LanguageDatasetsDecisionFile`.

Tests `write_decision_file` and `read_decision_file` in
`modules/language/datasets/include/SushiAI/language/datasets/decision_file.hpp`. The format is
tested over `std::ostringstream` and `std::istringstream`, the in-memory adapters of the stream
overloads, so those cases run `Small_`. The `Medium_` cases reach the disk. Most cases start from a
sample file of four records and 148 bytes: a header of 20 bytes, then a choice of 64 bytes, a yes
of 20, a number of 24 and a no of 20. Its offsets are the ones the refusal rows edit.

## TheWriterProducesTheDocumentedBytes

One yes-or-no record over a vocabulary of 10 writes exactly the 40 bytes the module README gives:
the magic, version 1, the vocabulary, a count of 1, then type 1, answer 1, one id, no option, the
id 5.

## ANumberIsWrittenAsTheBitsOfItsFloat

A number of 1.0 writes the answer word `00 00 80 3F`, the little-endian bits of `0x3F800000`.

## AWrittenFileReadsBackEqual

The four sample records read back equal to those written, with the vocabulary of 20.

## AFileOfNoRecordReadsBackEmpty

A file of no record over a vocabulary of 7 reads back with that vocabulary and no record.

## TheExtremesOfAFloatAndAnIDReadBackAsWritten

The largest float, the smallest subnormal and -1, each as a number, and the id 19 of a vocabulary of
20 read back equal.

## TheWriterRefusesARecordThatBreaksARule

One promise over thirteen rows: `write_decision_file` to a stream fails with `InvalidDescription`
naming the record and the rule, and writes no byte. The rows are an empty text, an id at or above the
vocabulary, a negative id, a vocabulary of 0, a choice of no option, an option of no id, an index
past the options or below 0, options on a yes-or-no, a number that is NaN or infinite, and an answer
of another type on a yes-or-no or a number.

## ATruncatedFileIsRefusedAtEveryLength

One promise over the lengths 0 to 147: `read_decision_file` over the sample file cut to that length
fails with `InvalidFile` naming `read_decision_file`.

## ACutInsideTheHeaderNamesTheHeader

A cut at 10 bytes names "ends at byte 10" and "header of 20 bytes".

## AFieldOutOfItsBoundsNamesItsOffset

One promise over ten rows: `read_decision_file` fails with `InvalidFile` for a first byte of 'X', a
version of 2, a vocabulary of 0, a count whose last byte is 0x7F (byte 12), a record type of 3 (byte
20), a yes-or-no answer of 2 (byte 88), an option count of 0x7F000003 (byte 32), a number whose bits
are the NaN `0x7FC00000` (record 2), an id of 3 over a vocabulary lowered to 3 (record 0), and two
bytes after the last record (byte 148).

## AFileSurvivesTheDisk

`Medium_`. `write_decision_file` to a path writes the same 148 bytes as the stream, and
`read_decision_file` of that path returns the four records and the vocabulary.

## TheWriterLeavesNoFileForARefusedRecord

`Medium_`. A refused record leaves no file at the path.

## ARefusalNamesTheFile

`Medium_`. With two bytes appended to a file, `read_decision_file` of its path fails with
`InvalidFile` and a message containing "long.sadc" and "2 bytes follow".

## AnAbsentFileIsRefused

`Medium_`. A path that names nothing is `FileNotFound`, naming "absent.sadc".
