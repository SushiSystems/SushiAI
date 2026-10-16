# tests/core/eager/test_storage_id.cpp


Suites: `Small_StorageID`.

Tests that `StorageIDSource` hands out each id once and never the empty id.

This file needs no device. A source is a counter, so every case draws ids and compares them.
`modules/core/eager/README.md`, under `storage_id.hpp`, says what a storage id names.

## TheEmptyIDIsZero

`K_NO_STORAGE` equals zero, so a table of storage ids that starts zero-filled names no buffer.

## AFreshSourceNeverReturnsTheEmptyID

The first `next()` of a new source is not `K_NO_STORAGE`.

## EveryDrawFromOneSourceIsNew

A thousand draws from one source are all different and none is `K_NO_STORAGE`.

## TwoSourcesCountIndependently

The first draw of a second source equals the first draw of the first one. Ids are unique per
source only, which is why a session owns exactly one source.
