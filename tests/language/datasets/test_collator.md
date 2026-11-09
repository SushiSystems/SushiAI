# tests/language/datasets/test_collator.cpp


Suites: `Small_LanguageDatasetsCollator`.

Tests that a caller sizes a batch from the interface `Collator` of
`modules/language/datasets/include/SushiAI/language/datasets/collator.hpp` alone. No in-memory
`Collator` exists beside the module, and the repository's own collator is host logic, so the case
reads `NextTokenCollator::make(3, 50)` through a `const Collator&`. The collator of the next-token
task is tested in `test_next_token_collator.cpp`.

## ACallerSizesTheTokensFromTheInterface

Three rows need 12 tokens, the rows times `tokens_per_window()`. The collator describes one role,
`K_ROLE_LABELS`, whose class count equals `vocabulary_size()`.
