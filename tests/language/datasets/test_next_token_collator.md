# tests/language/datasets/test_next_token_collator.cpp


Suites: `Small_LanguageDatasetsNextTokenCollator`, `Small_LanguageDatasetsNextTokenCollatorMake`,
`Small_LanguageDatasetsNextTokenCollatorRefusal`, `Medium_LanguageDatasetsNextTokenCollator`.

Tests `Language::Datasets::NextTokenCollator` in
`modules/language/datasets/include/SushiAI/language/datasets/next_token_collator.hpp`. Every
collator has a window of three positions over a vocabulary of 50. Only the case that writes
batch tensors runs `Medium_`: its fixture provides an engine and allocates a batch of an input
tensor and an `Int32` labels tensor. The refusals of `collate` decide on the dtype and the
element count of a tensor description, so they build tensors with no storage and run `Small_`.

## ItDescribesTheSizesOfTheTask

`task()` is `K_TASK_NEXT_TOKEN`, `window_length()` is 3, `tokens_per_window()` is 4 and
`vocabulary_size()` is 50.

## ItDescribesTheLabelsTarget

`targets()` holds one description: the role `K_ROLE_LABELS`, the sample shape [3], the type
`Int32` and 50 classes.

## LabelsHoldTheNextTokenOfEachPosition

`collate` over the ids 10, 11, 12, 13, 20, 21, 22, 23 writes the inputs 10, 11, 12, 20, 21, 22
and the labels 11, 12, 13, 21, 22, 23 into a batch of two rows.

## MakeRefusesExtentsOutOfBounds

One promise over two rows: `NextTokenCollator::make` fails with `InvalidDescription` for a
window of 0 ("NextTokenCollator", "window of 0 positions") and for a vocabulary of 0
("vocabulary of 0 tokens").

## CollateRefusesWhatDoesNotFit

One promise over four rows: `collate` fails with `Incompatible` for seven ids ("NextTokenCollator",
"7 ids", "windows of 4"), for twelve ids into a batch of six elements ("inputs", "6 elements",
"writes 9"), for inputs of `Float32` ("inputs", "int32") and for a batch without the labels role
("one role 'labels'").
