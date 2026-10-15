# tests/core/eager/test_slot_ledger.cpp


Suites: `Unit_SlotLedger`.

Tests which slot `SlotLedger` gives a shared buffer from one step to the next.

This file needs no device. A case opens a step, asks for slots and closes it, twice or three
times. `K_PRODUCER` writes the buffer and `K_READER` reads it; a reader names it by its own id
and the place of the value among its imports. `modules/core/eager/README.md`, under
`slot_ledger.hpp`, gives the three questions the ledger answers.

## match, first_step

The file-local `match` calls `match_readers` with every reader trusted and falls back to
`place`, the order `RegionTable` uses for a changed producer. `first_step` runs one step that
creates a slot.

## TheFirstMatchCreatesASlot

The slot is listed in `created()` with the extents and the element type asked for.

## AnUnchangedProducerKeepsItsSlot, KeepFindsNoSlotAtAnotherPosition

`keep` returns the slot written at the same position and nothing for another position.

## AChangedProducerReusesTheSlotItsReaderRead

The producer grew and the value moved from node 2 to node 5. The reader still names it as its
first import, so the buffer takes the old slot. The third step finds it by `keep` at the new
position, which shows that a matched slot is filed under its new name.

## AReaderThatIsNotTrustedNamesNoSlot

With no trusted reader `match_readers` returns `K_NO_STORAGE`, although a reader of that name
read the slot in the step before.

## PlaceTakesTheSlotWrittenAtTheSamePosition

`place` reuses the slot at the position once; the second call for the same position creates one.

## AChangedShapeGetsANewSlotAndReleasesTheOld, AChangedElementTypeGetsANewSlot

A slot is reused only for the same extents and element type. The slot left behind is returned by
`end_step`.

## ASlotIsClaimedOnce

Two buffers that the same reader slot names get two slots, and the first is not handed out again
through `keep`.

## ASlotOfAnotherProducerIsNotTaken

Another region now feeds the reader's first import. It gets a new slot, and the first producer
keeps its own.

## AnUnclaimedSlotIsReleasedAtTheEndOfTheStep

A step that asks for nothing releases every slot of the step before.

## ForgetReturnsEverySlotOnce

The second step is left open after one `keep` and one creation, as a step that failed after
planning leaves it. `forget` returns the kept slot and the created one, each once.
