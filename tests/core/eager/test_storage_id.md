# tests/core/eager/test_storage_id.cpp


Suites: `Small_StorageID`.

Tests that `StorageIDSource` hands out each id once and never the empty id.

This file needs no device. A source is a counter, so every case draws ids and compares them.
`modules/core/eager/README.md`, under `storage_id.hpp`, says what a storage id names.

## EveryDrawFromOneSourceIsNewAndNeverEmpty

A thousand draws from one source, checked by `expect_draws_are_new_and_never_empty`, are all
different and none is `K_NO_STORAGE`. The first draw of a new source is one of them.

## TwoSourcesCountIndependently

The first draw of a second source equals the first draw of the first one. Ids are unique per
source only, which is why a session owns exactly one source.
