# tests/language/datasets/test_prompt_dataset.cpp


Suites: `Small_LanguageDatasetsPromptDataset`, `Small_LanguageDatasetsPromptDatasetLayout`,
`Small_LanguageDatasetsPromptDatasetBlock`, `Medium_LanguageDatasetsPromptDataset`,
`Medium_LanguageDatasetsPromptDatasetRange`.

Tests `Language::Datasets::PromptDataset`, the row layout `lay_out_prompt`, the key reader
`read_prompt_section` and `make_prompt_dataset` in
`modules/language/datasets/include/SushiAI/language/datasets/prompt_dataset.hpp`. The row layout
and the keys of a data block are host logic and run `Small_`. A dataset holds an engine and its
batch is engine storage, so what reads through `PromptDataset` runs `Medium_`.

## TheDatasetDescribesItsOneBatch

A dataset of the ids 3, 1, 4, 1, 5 over a window of 8 has a batch size of 1, one batch, the
sample shape [8], the type `Int32`, the name `K_PROMPT_DATA_TYPE` and one target of the role
`K_ROLE_PROMPT_LENGTH`.

## AReadWritesTheRowAndItsLengthIntoTheBatch

`read(0)` of that dataset writes the row 3, 1, 4, 1, 5, 0, 0, 0 and the stated length 5.

## AShortPromptIsFollowedByZeros

`lay_out_prompt` of the ids 3, 1, 4, 1, 5 over a window of 8 returns the row 3, 1, 4, 1, 5, 0, 0,
0 and the length 5.

## ALongPromptKeepsItsLastIDs

`lay_out_prompt` of the ids 0 to 9 over a window of 4 returns the row 6, 7, 8, 9 and the length 4.

## LayOutRefusesAPromptOrAWindowThatDoesNotFit

One promise over three rows: `lay_out_prompt` fails with `InvalidDescription` for no id ("no
id"), for the ids 1 and -2 ("-2") and for a window of 0 ("window of 0").

## TheDataBlockEncodesItsText

A vocabulary made of no merges is written to a file in a scratch directory. A block with the
text "hi", a window of 4 and that file as `vocabulary` builds a dataset whose row is 'h', 'i', 0,
0 and whose stated length is 2.

## TheKeysOfADataBlockAreRead

`read_prompt_section` of a block with `text` "hi", `vocabulary` "bytes.savb" and `window` 4
returns those three values.

## AKeyThatDoesNotFitIsRefused

One promise over two rows: `read_prompt_section` fails with `InvalidDescription` for a block
without `text` ("the data block", "has no 'text'") and for a block with a `tokens` key
("'tokens'").

## ARangeReadHoldsTheRowOfTheWholeBatch

Over the ids 3, 1, 4, 1, 5 and a window of 8, `read_rows(0, {0, 1})` holds the row and the stated
length that `read(0)` holds.

## RegressionABadRangeIsRefusedWithAStatus

One promise over two rows: `read_rows` returns `InvalidDescription` rather than stopping, for the
rows 0 to 2 of batch 0 ("Rows 0 to 2", "of batch 0", "a batch of 1") and for batch 99 ("Batch 99
is out of range").
