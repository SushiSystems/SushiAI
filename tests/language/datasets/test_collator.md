# tests/language/datasets/test_collator.cpp


Suites: `Small_LanguageDatasetsCollator`.

Tests that a caller reads a task and collates a batch through the interface `Collator` of
`modules/language/datasets/include/SushiAI/language/datasets/collator.hpp` alone. The cases use a
fake collator and need no device. The collator of the next-token task is tested in
`test_next_token_collator.cpp`.

## CountingCollator

A fake of a window of 4 positions, 5 tokens per window and a vocabulary of 10. It describes one
labels role, records the token count of every `collate()` call, refuses a count that is not whole
windows with `Incompatible`, and sets a flag when it is destroyed.

## collate_rows

Collates a number of windows of zeros through a `const Collator&`, sizing the tokens from
`tokens_per_window()` as a dataset does.

## ACallerSizesTheTokensFromTheInterface

Three rows reach `collate()` as 15 tokens. The task, the one labels role and its class count,
equal to `vocabulary_size()`, are read through the interface.

## ARefusalReachesTheCaller

Seven tokens are not whole windows; the call through the interface fails with `Incompatible`
naming "not whole windows".

## DestroyingThroughTheInterfaceRunsTheDerivedDestructor

A `std::unique_ptr<Collator>` that owns the fake runs the fake's destructor when it is reset.
