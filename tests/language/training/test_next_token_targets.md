# tests/language/training/test_next_token_targets.cpp


Suites: `Small_LanguageTrainingNextTokenTargets`, `Small_LanguageTrainingNextTokenTargetsRange`,
`Small_LanguageTrainingNextTokenTargetsFit`.

Tests `Language::Training::fill_next_token_targets` in
`modules/language/training/include/SushiAI/language/training/next_token_targets.hpp`. The cases call the overload that takes spans of int32, so
they allocate no tensor and need no engine. The perplexity cases that sat here moved to
`test_perplexity.cpp`. The vocabulary has five tokens, and a leaf of six elements starts at -7.

## APositionIndexesItsLabelAmongEveryScore

The labels 4, 0, 2, 1, 1, 3 fill the leaf with 4, 5, 12, 16, 21, 28.

## ALabelOutsideTheVocabularyIsRefused

`fill_next_token_targets` fails with `Incompatible` for a label of 5 at position 4 ("position
4 is 5", "holds 5 tokens") and for a label of -1 at position 2 ("position 2 is -1"). Every
element of the leaf is still -7. The two inputs are rows.

## AFillThatDoesNotFitIsRefused

The fill fails with `Incompatible` for four labels over a leaf of six ("holds 4 labels",
"expects 6"), for a vocabulary of 2147483647 ("6 positions", "cannot be indexed") and for a
vocabulary of 0 ("cannot be indexed"). The three are rows.

## LabelsOfAnotherElementTypeAreRefused

The tensor overload fails with `Incompatible` for `Float32` labels over an `Int32` leaf
("float32", "int32"). Both tensors are shape-only, since the check reads the element type and
touches no element.
