# tests/core/profile_records/test_profile_records.cpp

Suites: `Small_ProfileRecords`, `Medium_ProfileRecords`.

Tests the helpers in `modules/core/profile_records/include/SushiAI/profile_records/profile_records.hpp`. None starts a session.

## FollowingTimelineSetsTheWindowAndNoLinger

`following_timeline("train.step", 1, 3)` follows `train.step`, with a warm-up of 1, 3 active periods and a linger of 0.

## ZonesOfMapsEachSiteNameToItsGroup

A record with the sites `data.read` and `host.feed`, and a period whose one zone tree holds a node of each, gives the map of the two names to the groups `data` and `host`.

## SushiAIZonesOfLeavesOutTheLibraryGroups

A period with the zones `device.run` of group `device`, `chain.wait` of group `Runtime` and `blas.record` of group `BLAS` gives `sushiai_zones_of` the one entry `device.run` to `device`: the groups of SushiRuntime and SushiBLAS are not SushiAI's.

## SushiAIZonesOfKeepsEveryGroupOfTheSharedList

A record with one zone `<group>.stage` for each group of `K_ZONE_GROUPS` (`SushiAI/profile/zone_groups.hpp`) gives `sushiai_zones_of` all of them. A group added to the list is covered with no edit to the case.

## TheConfigurationTextIsReadBySushiProfiler

`read_profiler_config` accepts a configuration text in the format `write_profile_config` writes. The read configuration follows `train.step`, has 2 active periods and one writer. The case reads the text and touches no file.

## TheWrittenFileReadsBackThroughSushiProfiler

Suite `Medium_ProfileRecords`: the case writes a file in a scratch folder, so it is `medium`. `write_profile_config` writes a file that `read_profiler_config` accepts, after `read_text` reads it back. The read configuration follows `train.step`, has 2 active periods and one writer.
