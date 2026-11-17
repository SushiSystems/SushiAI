# tests/language/datasets/test_decision_dataset.cpp

Suites: `Small_LanguageDatasetsDecisionDataset`, `Small_LanguageDatasetsDecisionDatasetFit`,
`Small_LanguageDatasetsDecisionDatasetBlock`, `Medium_LanguageDatasetsDecisionDataset`,
`Medium_LanguageDatasetsDecisionDatasetRefusal`.

Tests `lay_out_decisions`, `DecisionDataset`, `read_decisions_section` and
`make_decisions_dataset` in
`modules/language/datasets/include/SushiAI/language/datasets/decision_dataset.hpp`. The layout of the
rows is a function of values, so it runs `Small_`; the batch tensors need an engine and the file a
disk, so the dataset cases run `Medium_`. Expected arrays are written out by hand from the
records. The extents are T = 4 and K = 3.

## TheRowsOfThreeTypesAreLaidOutAsDocumented

A choice (text 1 2 3, options 4 5 and 6, answer 1), a yes (text 7) and a number (text 8 9 10 11,
value 2.5) lay out as: tokens `1 2 3 0 | 7 0 0 0 | 8 9 10 11`, float lengths 3 1 4, option rows
`4 5 0 0` and `6 0 0 0` for the choice and zeros elsewhere, float option lengths 2 1 0 and six
zeros, option flags 1 1 0 and six zeros, `choice` 1 0 0, `yes_no` 0 1 0, `number` 0 0 2.5 and the
type mask rows `1 0 0`, `0 1 0`, `0 0 1`. Each answer role holds 0 on the rows of the other types.

## ARecordThatFillsTheExtentsFits

A choice of a text and an option of exactly T ids and K options lays out with their lengths, the
three flags and the index 2.

## AnOptionOfNoIdIsFlaggedAsPadding

A choice with the options `{4}`, `{}` and `{5, 6}` lays out the option lengths 1 0 2 and the flags
1 0 1: the flag follows the length, so an option of no id is read as padding, as it was when the
model derived the flag from the length.

## NoRecordLaysOutAsNoRow

No record gives empty arrays.

## LayOutRefusesWhatDoesNotFit

One promise over five rows: `lay_out_decisions` fails with `InvalidDescription` for a text longer
than T (naming record 1), an option longer than T, more options than K, T of 0 and K of 0.

## TheKeysOfADataBlockAreRead

`read_decisions_section` returns the file, T, K and the batch size of a block that states them all.

## AnAbsentBatchSizeKeepsTheDefault

A block without `batch_size` keeps the default of `DecisionDatasetOptions`.

## AKeyThatDoesNotFitIsRefused

One promise over five rows: a block without `file`, `window` or `options`, with the key `stride`, or
with `options` of 0 fails with `InvalidDescription` naming the key.

## TheDatasetDescribesItsBatches

`Medium_`. Four records at a batch size of 2 give a batch size of 2, two batches, the sample shape
[4], `Int32`, the vocabulary 20 and the name "decisions".

## TheEightRolesAreDescribedInOrder

`Medium_`. `targets()` states, in order, `lengths` (per sample [1], float32), `options` ([3, 4],
int32), `option_lengths` ([3], float32), `option_valid` ([3], float32), `choice` ([], int32, 3
classes), `yes_no` ([1], float32), `number` ([1], float32) and `type_mask` ([3], float32).

## TheRolesLabelsAndValuesAreNotPublished

`Medium_`. `find_target` of `labels` and of `values` over `targets()` finds nothing: a dataset that
published them would be read by an objective that names no `target`, and its yes-or-no and number
rows would share one role.

## ABatchHasTheEightRolesWithTheirShapes

`Medium_`. Batch 0 has an int32 input of 8 elements and eight roles of 2, 24, 6, 6, 2, 2, 2 and 6
elements; `lengths`, `yes_no` and `number` are [2, 1], `options` is [2, 3, 4] and `type_mask` is
[2, 3].

## ABatchHoldsTheValuesOfItsRecords

`Medium_`. Batch 0, the choice and the yes, holds the arrays of the layout case for those two
rows, as floats and int32 read back from the tensors.

## TheSecondBatchHoldsTheLaterRecords

`Medium_`. Batch 1 holds the number and the no: tokens `8 9 10 11 3 3 0 0`, lengths 4 2, `yes_no`
0 0, `number` 2.5 0 and type mask `0 0 1 | 0 1 0`.

## ARangeReadHoldsTheRowOfTheWholeBatch

`Medium_`. `read_rows` of row 1 of batch 0 holds that row of `read`.

## MakeRefusesOptionsOutsideTheirBounds

`Medium_`. One promise over six rows: `DecisionDataset::make` fails for a batch size of 0, T of 0, a
text longer than T (record 2, with T of 3), a record with more options than K (record 0, with K of
1), fewer records than a batch (4 for a batch of 5), and an absent file (`FileNotFound`).

## ADataBlockBuildsTheDataset

`Medium_`. `make_decisions_dataset` over a block of `file`, `window` 4, `options` 3 and
`batch_size` 2 returns a dataset of batch size 2, two batches, sample shape [4] and eight roles.
