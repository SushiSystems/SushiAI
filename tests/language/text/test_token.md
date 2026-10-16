# tests/language/text/test_token.cpp


Suites: `Small_LanguageTextToken`.

Tests the constants of `modules/language/text/include/SushiAI/language/text/token.hpp`: the type
`TokenID` and the limits of a vocabulary. The header declares no function, so the cases pin the
relations between the constants that a vocabulary and a token file depend on.

## ATokenIDIsAnInt32

`TokenID` is `std::int32_t`.

## EndOfTextFollowsTheByteTokens

`K_BYTE_TOKEN_COUNT` is 256 and `K_END_OF_TEXT` equals it, the first id after the byte tokens.
`K_END_OF_TEXT_NAME` is "<end_of_text>" and fits `K_MAX_SPECIAL_NAME_BYTES`.

## TheLargestVocabularyFitsATokenID

`K_MAX_VOCABULARY_SIZE` is 2 to the 20th. It is larger than the byte tokens and
`K_MAX_SPECIAL_TOKENS` together and no larger than the largest `TokenID`. `K_MAX_TOKEN_BYTES` is
at least 1.
