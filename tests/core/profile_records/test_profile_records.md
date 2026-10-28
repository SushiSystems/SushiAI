# tests/core/profile_records/test_profile_records.cpp

Suites: `Small_ProfileRecords`.

Tests the helpers in `modules/core/profile_records/include/SushiAI/profile_records/profile_records.hpp`. None starts a session.

## FollowingTimelineSetsTheWindowAndNoLinger

`following_timeline("train.step", 1, 3)` follows `train.step`, with a warm-up of 1, 3 active periods and a linger of 0.

## ZonesOfMapsEachSiteNameToItsGroup

A record with the sites `data.read` and `host.feed`, and a period whose one zone tree holds a node of each, gives the map of the two names to the groups `data` and `host`.

## SushiAIZonesOfLeavesOutTheLibraryGroups

A period with the zones `device.run` of group `device`, `chain.wait` of group `Runtime` and `blas.record` of group `BLAS` gives `sushiai_zones_of` the one entry `device.run` to `device`: the groups of SushiRuntime and SushiBLAS are not SushiAI's.

## TheWrittenConfigurationIsReadBySushiProfiler

`write_profile_config` writes a file that `read_profiler_config` accepts. The read configuration follows `train.step`, has 2 active periods and one writer.
