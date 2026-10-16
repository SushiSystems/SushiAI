# tests/core/eager/test_region_table.cpp


Suites: `Small_RegionTable`.

Tests what `RegionTable` keeps, records again, builds and drops from one step to the next.

This file needs no device: the table works on storage ids. Each case plans two or three traces
of `encoder_head_loss` on one table, with the presets of `eager_test_support.hpp` as the storage
of the leaves and the gradients. `modules/core/eager/README.md`, under `region_table.hpp` and
`slot_ledger.hpp`, states the rules.

## TheFirstStepBuildsEveryRegion

Every region is built, none has a previous key, and the plan creates slots.

## TheSameTraceKeepsEveryRegion

A second trace of the same model keeps every region and creates and releases nothing. The
encoder's result and the loss are in the storage the first plan gave them.

## AChangedScopeRebuildsItsTwoRegionsAndNoOther

The head ends in K_TANH in the first step and in K_RELU in the second. Its forward and backward
regions are built. The encoder's two regions and the loss region are kept: each reads the
head's results from the slots of the first step, because it names them by its own import places
and its own structure did not change.

## AReaderKeepsTheStorageOfAResultWhoseProducerChanged

The head's result has one storage id in both steps. This is the single fact the case above
rests on, asserted on its own.

## AScopeThatAppearsRenumbersARegionWithoutRebuildingIt

Three segments give the encoder's backward region key 5 and four give it key 7. The region is
recorded again under the new key with its old key as `previous_key`. The encoder's forward
region keeps key 1 and is kept, and the new scope's region is built with no previous key.

## AScopeThatDisappearsIsDroppedByItsOldKeys

The plan lists the two keys the "extra" scope's regions held, forward before backward, and
releases the slots only they used.

## ALeafWithoutStorageIsFatal

A plan with no preset stops the process, naming the region and the value it could not place.

## ForgetBuildsEveryRegionAgain

`forget()` returns one id per slot the first plan created, and the next plan builds every region
with nothing to drop.
