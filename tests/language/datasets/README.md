# tests/language/datasets

The tests of the module `language/datasets`, one file per header of `modules/language/datasets/include/`. The suite prefix of a case gives its ctest label.

## test_next_token_collator.cpp

Suites: `Medium_LanguageDatasetsNextTokenCollator`.

Tests `Language::Datasets::NextTokenCollator` in
`modules/language/datasets/include/SushiAI/language/datasets/next_token_collator.hpp`. The
fixture provides an engine and allocates a batch of an input tensor and an `Int32` labels
tensor. Every collator has a window of three positions over a vocabulary of 50.

### ItDescribesTheTask

`task()` is `K_TASK_NEXT_TOKEN`, `window_length()` is 3, `tokens_per_window()` is 4 and
`vocabulary_size()` is 50. `targets()` holds one description: the role `K_ROLE_LABELS`, the
sample shape [3], the type `Int32` and 50 classes.

### LabelsHoldTheNextTokenOfEachPosition

`collate` over the ids 10, 11, 12, 13, 20, 21, 22, 23 writes the inputs 10, 11, 12, 20, 21, 22
and the labels 11, 12, 13, 21, 22, 23 into a batch of two rows.

### MakeRefusesExtentsOutOfBounds

`NextTokenCollator::make` fails with `InvalidDescription` for a window of 0
("NextTokenCollator", "window of 0 positions") and for a vocabulary of 0 ("vocabulary of 0
tokens").

### CollateRefusesWhatDoesNotFit

`collate` into a batch of two rows fails with `Incompatible` for seven ids
("NextTokenCollator", "7 ids", "windows of 4") and for twelve ids ("inputs", "6 elements",
"writes 9"). It also fails for a batch whose inputs are `Float32` ("inputs", "int32") and for
a batch without tensors ("one role 'labels'").

## test_prompt_dataset.cpp

Suites: `Medium_LanguageDatasetsPromptDataset`.

Tests `Language::Datasets::PromptDataset` in
`modules/language/datasets/include/SushiAI/language/datasets/prompt_dataset.hpp` and
`make_prompt_dataset`, which builds one from a data block. The fixture provides an engine and
reads the row and the stated prompt length of batch 0.

### AShortPromptIsFollowedByZeros

A dataset of the ids 3, 1, 4, 1, 5 over a window of 8 has a batch size of 1, one batch, the
sample shape [8], the type `Int32`, the name `K_PROMPT_DATA_TYPE` and one target of the role
`K_ROLE_PROMPT_LENGTH`. Its row is 3, 1, 4, 1, 5, 0, 0, 0 and its stated length is 5.

### ALongPromptKeepsItsLastIDs

A dataset of the ids 0 to 9 over a window of 4 has the row 6, 7, 8, 9 and the stated length 4.

### MakeRefusesAPromptOrAWindowThatDoesNotFit

`PromptDataset::make` fails with `InvalidDescription` for no id ("no id"), for the ids 1 and
-2 ("-2") and for a window of 0 ("window of 0").

### TheDataBlockEncodesItsText

A vocabulary made of no merges is written to a file in a scratch directory. A block with the
text "hi", a window of 4 and that file as `vocabulary` builds a dataset whose row is 'h', 'i',
0, 0 and whose stated length is 2. `make_prompt_dataset` fails with `InvalidDescription` for a
block without `text` ("the data block", "has no 'text'") and for a block with a `tokens` key
("'tokens'").

## test_token_dataset.cpp

Suites: `Medium_LanguageDatasetsTokenDataset`.

Tests `Language::Datasets::TokenDataset` in
`modules/language/datasets/include/SushiAI/language/datasets/token_dataset.hpp`. The fixture
provides an engine and writes `rising.satk` to a scratch directory: the ids 0 to 19 over a
vocabulary of 20, so each id equals its position. Unless a case says otherwise the dataset
has a next-token collator of four positions, a batch size of 2 and the name "tiny".

### ItDescribesItsBatches

The dataset has a batch size of 2, the sample shape [4], the type `Int32`, two batches, the
name "tiny" and a vocabulary size of 20. Its one target has the role `K_ROLE_LABELS`, the
sample shape [4] and 20 classes.

### ABatchHoldsItsWindowsAsInt32

`read(1)` fills `Int32` inputs of the shape [2, 4] and one `Int32` target of eight elements.
The inputs are the ids 8 to 15 and each label is its input plus 1.

### EveryLabelIsTheTokenAfterItsInput

With a stride of 3 the dataset has three batches. In every batch, row `r` of batch `b` starts
at the id `(2b + r) * 3`: input `t` is that id plus `t`, label `t` is that id plus `t + 1`, and
each label but the last equals the next input of its row.

### MakeRefusesOptionsOutsideTheirBounds

`TokenDataset::make` fails with `InvalidDescription` for a null collator ("TokenDataset",
"tiny", "no collator"), a batch size of 0 ("batch size of 0"), a stride of 0 ("TokenDataset",
"tiny", "stride of 0") and a batch size of 5 ("rising.satk", "4 windows of 5", "one batch of
5"). It fails with `Incompatible` for a collator over 19 tokens ("rising.satk", "vocabulary of
20", "one of 19") and with `FileNotFound` for a path that does not exist ("absent.satk").

### AnIDOutsideTheVocabularyNamesItsFileOffset

Byte 8 of the file is overwritten with 5, which makes its vocabulary size 5. The dataset is
still made, and `read(0)` fails with `InvalidFile` and a message containing "rising.satk", "id
5", "byte 44" and "[0, 5)".

### AFileCutAfterMakeNamesWhereItEnds

The file is resized to 64 bytes after the dataset is made. `read(0)` is ok and `read(1)` fails
with `InvalidFile` and a message containing "rising.satk", "ends at byte 64" and "end at byte
76".

### ABatchOutsideTheEpochStopsTheProcess

`read(2)` on a dataset of two batches stops the process with a report containing
"TokenDataset", "2 batches" and "batch 2".

### AReadReusesTheTensorsItFilledBefore

After `read(0)` and `read(1)` into the same batch, the inputs and the labels have the data
pointers they had after the first read, and the first input is 8.

### AReadDoesNotWriteATensorSomethingElseHolds

A copy of the inputs tensor is kept after `read(0)`. After `read(1)` the batch's inputs have
another data pointer, the first id of the kept tensor is 0 and the first id of the batch is 8.

### ItIsADatasetATrainerCanRead

Read through a `Data::Dataset` reference, `read(1)` fills as many targets as `targets()`
lists, the first target has the role the description states, and the first extent of the
inputs equals `batch_size()`.

## test_token_file.cpp

Suites: `Small_LanguageDatasetsTokenFile`.

Tests `write_token_file`, `read_token_file_header` and `read_token_range` in
`modules/language/datasets/include/SushiAI/language/datasets/token_file.hpp`. Most cases start
from a hand-written file of 36 bytes: a header of 24 bytes (the magic "SATK", version 1, a
vocabulary of 300, an id width of 4 and a count of 3) and the ids 1, 258 and 0. Files are
written to a scratch directory.

### TheWriterProducesTheDocumentedBytes

`write_token_file` of the ids 1, 258 and 0 over 300 tokens writes exactly those 36 bytes.

### TheHeaderAndARangeOfIDsSurviveTheFile

A file of the ids 0 to 39999 over 40000 tokens has a header with a vocabulary size of 40000
and a token count of 40000. `read_token_range` of five ids from 16382 returns 16382 to 16386,
the range from 39995 ends at 39999, and an empty range at 40000 is ok.

### TheWriterRefusesAnIDOutsideTheVocabulary

`write_token_file` fails with `InvalidDescription` for the id 300 over 300 tokens
("write_token_file", "id 300", "index 1", "[0, 300)"), for the id -2 ("id -2", "index 0") and
for a vocabulary of 0 ("vocabulary of 0 tokens"). No file exists afterwards.

### ATruncatedFileIsRefusedAtEveryLength

For every length from 0 to 35 bytes, `read_token_file_header` over the file cut to that length
fails with `InvalidFile` and a message containing "cut.satk", "read_token_file_header" and
"ends at byte". A cut at 10 bytes names "ends at byte 10" and "header of 24 bytes"; a cut at
30 names "ends at byte 30" and "declares 36 bytes".

### BytesAfterTheDeclaredEndAreRefused

With two bytes appended, `read_token_file_header` fails with `InvalidFile` and a message
containing "long.satk", "2 bytes follow" and "byte 36".

### AHeaderFieldOutOfBoundsNamesItsOffset

`read_token_file_header` fails with `InvalidFile` for a first byte of 'X' ("'SATK'", "no token
file"), a version of 2 ("byte 4", "format version 2", "versions 1 to 1"), a vocabulary of 0
("byte 8", "vocabulary of 0 tokens"), an id width of 8 ("byte 12", "ids of 8 bytes") and a
count whose last byte is 0x7F ("byte 16", "ids").

### AnIDOutsideTheVocabularyNamesItsOffset

With the vocabulary size lowered to 258, the header reads and `read_token_range` of three ids
fails with `InvalidFile` ("above.satk", "read_token_range", "id 258", "byte 28", "[0, 258)"),
while a range of the first id alone returns 1. In a file whose third id is the bytes FF FF FF
FF, the range at id 2 fails with "id -1" and "byte 32".

### ARangeOutsideTheFileAndAnAbsentFileAreRefused

`read_token_range` of two ids from id 2 fails with `InvalidDescription` and a message
containing "2 ids from id 2" and "file of 3". Over a path that does not exist,
`read_token_file_header` fails with `FileNotFound` ("absent.satk") and `read_token_range` with
`InvalidFile` ("absent.satk", "cannot be opened").

## test_tokens_section.cpp

Suites: `Medium_LanguageDatasetsTokensSection`.

Tests `Language::Datasets::make_tokens_dataset` in
`modules/language/datasets/include/SushiAI/language/datasets/tokens_section.hpp`, which builds
a dataset from a `tokens` data block. The fixture provides an engine and writes a token file
of the ids 0 to 39 over a vocabulary of 40.

### TheKeysDescribeTheDataset

A block with `window` 4, `stride` 5 and `batch_size` 2 builds a dataset with a batch size of
2, the sample shape [4], the type `Int32` and four batches. Its `K_ROLE_LABELS` target has 40
classes. In batch 1 the first input is 10 and the fifth is 15.

### AnAbsentBatchSizeIsTheDefault

A block with `window` 1 and `stride` 1 builds a dataset whose batch size equals
`TokenDatasetOptions{}.batch_size`.

### AKeyThatDoesNotFitIsRefused

`make_tokens_dataset` fails with `InvalidDescription` for a block without `window` ("the data
block", "has no 'window'"), a block without `file` ("the data block", "has no 'file'"), a
`window` of "four" ("'window'", "whole number"), a `window` of 0 ("'window'", "positive") and
a `shuffle` key ("the data block", "'shuffle'", "does not have").

### AMissingFileFails

A block whose `file` does not exist returns a status that is not ok. The case checks neither
the code nor the message.

## test_window_layout.cpp

Suites: `Small_LanguageDatasetsWindowLayout`.

Tests `count_windows` and `first_token_of` over a `WindowLayout` in
`modules/language/datasets/include/SushiAI/language/datasets/window_layout.hpp`. A layout is
written here as {window, stride}.

### WindowsBeginOneStrideApart

`count_windows(10, {4, 3})` is 3, and `first_token_of` is 0, 3 and 6 for the windows 0, 1 and
2.

### OnlyWholeWindowsAreCounted

`count_windows` is 2 for 8 ids over {4, 4}, 2 for 11 ids over {4, 4}, 1 for 4 ids over {4, 1}
and 0 for 3 ids over {4, 1}.

### ALayoutOutOfBoundsIsRefused

`count_windows` fails with `InvalidDescription` for a window of 0 ("count_windows", "window of
0 tokens"), a stride of 0 ("stride of 0 tokens") and a window of `K_MAX_WINDOW_TOKENS + 1`
("window of").
