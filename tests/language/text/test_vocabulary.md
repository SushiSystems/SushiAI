# tests/language/text/test_vocabulary.cpp


Suites: `Small_TextVocabulary`, `Small_TextVocabularyMergeRefusal`,
`Small_TextVocabularyNameRefusal`, `Medium_TextVocabularyDeath`.

Tests `Language::Text::Vocabulary` in
`modules/language/text/include/SushiAI/language/text/vocabulary.hpp`. A merge is written here
as (left, right).

## AVocabularyWithoutMergesHoldsTheByteTokens

The ids 0 and 255 are of the kind `Byte`, id 65 stands for "A" and id 255 for the byte 0xFF.

## AVocabularyWithoutMergesHoldsEndOfText

`K_END_OF_TEXT` is of the kind `Special` and stands for no byte, and
`find_special(K_END_OF_TEXT_NAME)` is 256.

## AVocabularyWithoutMergesHoldsTheIDsUpToEndOfText

A vocabulary made of no merges has 257 tokens and a `first_merged()` of 257. `contains` is true
for 256 and false for 257 and -1.

## MergedTokensTakeIDsInTheOrderOfTheirMerges

With the merges (97, 98) and (257, 99) the vocabulary has 259 tokens, id 257 is of the kind
`Merged` and stands for "ab", and id 258 stands for "abc". `merges()` holds two merges, the second
of which is (257, 99).

## FindMergeLooksUpAPairAndAnAbsentPair

`find_merge` returns 257 for (97, 98) and 258 for (257, 99), and no value for (98, 97) and (-1, 97).

## SpecialTokensFollowEndOfText

With one merge and the special names "<pad>" and "<mask>" the vocabulary has 260 tokens.
`find_special` returns 257 and 258 for the two names. `first_merged()` is 259, id 259 stands for
"ab", and `special_names()` holds three names, the first of which is `K_END_OF_TEXT_NAME`.

## FindSpecialLooksUpAnAbsentName

`find_special("<absent>")` returns no value.

## MakeRefusesAMergeOfATokenItMayNotJoin

Rows over the merges. `Vocabulary::make` fails with `InvalidDescription` for the merge (97, 257)
at index 0 ("Vocabulary", "merge 0", "97 and 257"), (259, 97) at index 1 ("merge 1", "259 and 97"),
(`K_END_OF_TEXT`, 97) ("merge 0", "256 and 97") and (-1, 97) ("merge 0", "-1 and 97").

## MakeRefusesAPairNamedTwice

Two merges of (97, 98) fail with `InvalidDescription` and a message containing "merge 1", "97
and 98" and "earlier merge".

## MakeRefusesATokenAboveTheByteLimit

Nine merges that each double the token before fail with `InvalidDescription` and a message
containing "merge 8", "512 bytes" and "at most 256".

## MakeHoldsATokenOfTheByteLimit

Eight such merges build a vocabulary whose id 264 stands for 256 'a' bytes.

## MakeRefusesASpecialNameOutOfBounds

Rows over the special names. `Vocabulary::make` fails with `InvalidDescription` for an empty name
("special token 1", "0 bytes"), a name of 256 bytes ("256 bytes", "1 to 255"), "<pad>" given twice
("'<pad>'", "named twice"), `K_END_OF_TEXT_NAME` given as a name ("named twice") and 256 names
("257 special tokens", "at most 256").

## AnIDOutsideTheVocabularyStopsTheProcess

Over a vocabulary of 257 tokens, `bytes_of(257)` stops the process with a report containing
"Vocabulary", "257" and "[0, 257)", and `kind_of(-1)` with "Vocabulary" and "-1". The case is
`Medium_` because a death test starts a child process.
