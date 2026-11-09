# tests/language/generation/test_ring_position_feed.cpp


Suites: `Small_GenerationRingPositionFeed`, `Small_GenerationRingSlot`, `Small_GenerationRingAngles`, `Small_GenerationRingRefusal`.

Tests `RingPositionFeed` in `modules/language/generation/include/SushiAI/language/generation/ring_position_feed.hpp` over an `InMemoryFedLeaves` and a host
engine. The angle function writes position + pair as each cosine and its negation as each sine.
That every row stays after a slide is not tested alone: it is tested by the ring cases of
`test_cached_generator.cpp`.

## TheFeedListsTheSlotThenTheAngles

The feed lists the slot, the cosines and the sines in that order, with the types and the counts.

## TheSlotIsThePositionModuloT

Rows: position 11 over four rows is row 3, position 2 is row 2.

## TheAnglesAreThoseOfThePosition

Rows over the three pairs: at position 11 the cosine of pair i is 11 + i and the sine is its negation.

## TheFirstWrapWritesOverTheRowOfPositionZero

Position T writes the slot 0 and the angles of T.

## ARingThatDoesNotFitIsRefused

Four rows: a ring without its angle function, an unnamed sine, a slot of the wrong type and a narrow leaf each fail with `InvalidDescription`.
