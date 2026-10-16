# tests/core/eager/test_region_names.cpp


Suites: `Small_RegionNames`.

Tests that `RegionNames` gives one id per name and direction, whatever the key.

This file needs no device. Every case interns `RegionInfo` values by hand and reads the ids
back. `modules/core/eager/README.md`, under `region_names.hpp`, says why a region needs an id
that outlives its key.

## ANameKeepsItsIDWhateverItsKey

The same name and direction interned under key 3 and then key 7 give one id, and `size()` is 1.

## TheTwoDirectionsOfANameAreTwoRegions

"head#0" forward and "head#0" backward get two ids, and `find()` returns the backward one for
the backward direction.

## FindingAnUnknownNameGivesNoID

`find()` returns `K_NO_REGION_ID` for a name never interned and for a known name in the other
direction, and adds no id.

## ALabelNamesTheRegionAndItsDirection

The label of a backward "head#0" is "head#0 backward".

## TheLabelOfAnIDNeverGivenIsFatal

`label_of(5)` on an object that gave one id stops the process, and the report names "Region id 5" and
"never given".
