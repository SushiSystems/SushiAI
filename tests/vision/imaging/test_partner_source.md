# tests/vision/imaging/test_partner_source.cpp


Suites: `Small_ImagingPartnerSource`.

Tests the partner count of a mosaic and a fake `PartnerSource` used through the interface.

This file needs no device. The header declares an interface, so the cases exercise a fake. The
two real sources have their own files, `tests/vision/imaging/test_dataset_partners.cpp` and
`tests/vision/imaging/test_recent_partners.cpp`, and `tests/vision/imaging/test_mosaic.cpp`
checks that a mosaic asks its source once per sample.

## DrawingPartners

`DrawingPartners` returns three draws of `Random::index` below `K_POOL`, in draw order, and sets
a flag the test owns when it is destroyed.

## AMosaicHasThreePartners

`K_MOSAIC_PARTNERS` is 3, and `MosaicPartners` holds that many indices.

## ASourceAskedThroughTheInterfaceReturnsItsDraws

A `DrawingPartners` held as a `std::unique_ptr` to `PartnerSource` is asked once. Its three
partners equal three draws of a second generator of the same seed, and both generators end in
the same state, so the call made exactly three draws.

## DestroyingThroughTheInterfaceRunsTheSourcesDestructor

Resetting the base pointer runs the destructor of `DrawingPartners`, which the virtual
destructor of `PartnerSource` guarantees.
