# tests/core/eager/test_region_table.cpp


Suites: `Small_RegionTable`, `Medium_RegionTableFatal`.

Tests what `RegionTable` keeps, records again, builds and drops from one step to the next.

This file needs no device: the table works on storage ids. Each case plans two or three traces
of `encoder_head_loss` on one table, with the presets of `eager_test_support.hpp` as the storage
of the leaves and the gradients. `modules/core/eager/README.md`, under `region_table.hpp` and
`slot_ledger.hpp`, states the rules.

## TheFirstStepBuildsEveryRegion

Every region is built, none has a previous key, and the plan creates slots.

## TheSameTraceKeepsEveryRegion

A second trace of the same model keeps every region and creates, releases and drops nothing.

## AKeptRegionKeepsItsStorage

The encoder's result and the loss are in the storage the first plan gave them.

## AChangedScopeRebuildsItsTwoRegionsAndNoOther

The head ends in K_TANH in the first step and in K_RELU in the second. Its forward and backward
regions are built. The encoder's two regions and the loss region are kept: each reads the
head's results from the slots of the first step, because it names them by its own import places
and its own structure did not change.

## AReaderKeepsTheStorageOfAResultWhoseProducerChanged

The head's result has one storage id in both steps. This is the single fact the case above
rests on, asserted on its own.

## AScopeThatAppearsIsBuilt

Adding a scope "extra" builds its forward region, and that region has no previous key.

## AScopeThatAppearsLeavesTheRegionsBeforeItKept

The encoder's forward region is kept when the scope "extra" appears after it.

## AScopeThatAppearsRenumbersARegionWithoutRebuildingIt

The encoder's backward region gets another key when the fourth segment is added. The region is
recorded again under the new key with its old key as `previous_key`, and is not built.

## AScopeThatDisappearsIsDroppedByItsOldKeys

The plan lists the two keys the "extra" scope's regions held, forward before backward.

## AScopeThatDisappearsReleasesItsStorage

The plan releases the slots only the vanished scope used.

## ALeafWithoutStorageIsFatal

A plan with no preset stops the process, naming the region and the value it could not place.

## ForgetReleasesEverySlotAndLeavesNoLiveRegion

`forget()` returns one id per slot the first plan created, and the table holds no live region.

## ThePlanAfterForgetBuildsEveryRegion

The next plan builds every region with nothing to drop.
