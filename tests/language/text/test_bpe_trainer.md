# tests/language/text/test_bpe_trainer.cpp


Suites: `Small_TextBpeTrainer`, `Small_TextBpeTrainerSizeRefusal`.

Tests `Language::Text::train_bpe` in
`modules/language/text/include/SushiAI/language/text/bpe_trainer.hpp`. Unless a case says
otherwise the options state a vocabulary size and no special name, so the first merged token
has the id 257. The generated corpora are drawn with the seed 20261011.

## ItLearnsTheHandWorkedMergesOfATinyCorpus

Training on "abababc" to a size of 300 learns two merges, (97, 98) then (257, 257). The
vocabulary has 259 tokens and token 258 stands for "abab".

## PairsOfOneCountMergeBySmallerLeftThenRightID

Training on "ab ab cd cd" learns three merges in the order (32, 99), (97, 98), (257, 100).

## NoMergeCrossesTheStartOfABlankRun

Training on "a b a b a b" learns two merges, (32, 98) then (32, 97).

## TheSameCorpusGivesTheSameVocabulary

Two runs to a size of 400 over one corpus of 20000 bytes drawn from "abcdef", the blank and
the newline learn at least one merge and the same merges in the same order.

## ARunStoppedEarlyLearnsTheFirstMergesOfALongerRun

Over a corpus of 4000 bytes drawn from "abc" and the blank, a run to a size of 270 learns at
least one merge, and each equals the merge at the same index of a run to a size of 290.

## ItStopsAtTheSizeAskedFor

Training on "abababc" to a size of 258 learns one merge, (97, 98).

## APairThatOccursOnceIsNotMerged

Training on "abcdef" to a size of 300 gives a vocabulary of 257 tokens.

## AnEmptyCorpusTrainsToTheBaseVocabulary

Training on the empty text to a size of 300 gives a vocabulary of 257 tokens.

## SpecialTokensPrecedeTheMergedTokens

With the special name "<pad>", training on "abababc" learns two merges, the second of which
is (258, 258), and token 259 stands for "abab".

## ItRefusesASizeOutOfBounds

Rows over the size. `train_bpe` fails with `InvalidDescription` for a size of 256 ("train_bpe",
"256 tokens", "257 to") and for a size of `K_MAX_VOCABULARY_SIZE + 1` ("train_bpe", "asked for").
The refusal of an empty special name is tested in `test_vocabulary.cpp`, in
`MakeRefusesASpecialNameOutOfBounds`.
