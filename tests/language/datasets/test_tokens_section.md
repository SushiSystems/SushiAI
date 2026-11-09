# tests/language/datasets/test_tokens_section.cpp


Suites: `Small_LanguageDatasetsTokensSection`, `Small_LanguageDatasetsTokensSectionRefusal`,
`Medium_LanguageDatasetsTokensSection`.

Tests `read_tokens_section` and `Language::Datasets::make_tokens_dataset` in
`modules/language/datasets/include/SushiAI/language/datasets/tokens_section.hpp`, which build a
dataset from a `tokens` data block. The keys are host logic and run `Small_`. The `Medium_`
fixture provides an engine and writes a token file of the ids 0 to 39 over a vocabulary of 40.

## TheKeysOfABlockAreRead

A block with `file` "a.satk", `window` 4, `stride` 5 and `batch_size` 2 reads back with those
four values.

## AnAbsentBatchSizeIsSixteen

A block with only `file` and `window` 1 reads a batch size of 16 and no stride. 16 is the
documented default of `TokenDatasetOptions::batch_size`, written here as the value.

## AKeyThatDoesNotFitIsRefused

One promise over five rows: `read_tokens_section` fails with `InvalidDescription` for a block
without `window` ("the data block", "has no 'window'"), a block without `file` ("the data block",
"has no 'file'"), a `window` of "four" ("'window'", "whole number"), a `window` of 0 ("'window'",
"positive") and a `shuffle` key ("the data block", "'shuffle'", "does not have").

## TheKeysDescribeTheDataset

`Medium_`. A block with `window` 4, `stride` 5 and `batch_size` 2 builds a dataset with a batch
size of 2, the sample shape [4], the type `Int32` and four batches. Its `K_ROLE_LABELS` target
has 40 classes.

## AReadReturnsTheWindowsAtTheStride

`Medium_`. In batch 1 of that dataset the first input is 10 and the fifth is 15.

## AMissingFileFails

`Medium_`. A block whose `file` does not exist returns a status that is not ok. The case checks
neither the code nor the message.
