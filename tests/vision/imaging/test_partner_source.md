# tests/vision/imaging/test_partner_source.cpp


Suites: `Small_ImagingPartnerSourceContract`.

Tests the contract every `PartnerSource` keeps, asked through the interface.

This file needs no device. The header declares an interface, so the suite is one typed contract
instantiated for `InMemoryPartnerSource`, `DatasetPartners` and `RecentPartners`, each made by a
small maker struct. The real sources have their own files, `tests/vision/imaging/test_dataset_partners.cpp`
and `tests/vision/imaging/test_recent_partners.cpp`, and `tests/vision/imaging/test_mosaic.cpp`
checks that a mosaic asks its source once per sample. That a source is destroyed through the
interface is a compile-time check in `compile_partner_source.cpp`.

## InMemoryMaker, DatasetMaker and RecentMaker

Each maker returns a `PartnerSource` held by a `std::unique_ptr`: the in-memory source with one
scripted entry, a `DatasetPartners` over 10 images and a `RecentPartners` over 10 images with a
batch of 4. `SourceNames` names each instantiation.

## ASourceAskedThroughTheInterfaceMakesThreeDraws

A source asked once through the interface leaves its generator in the state of a second
generator of the same seed after three `next_u32` draws, so the call made exactly three draws.
