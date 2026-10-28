# tests/core/profile/test_zone_groups.cpp

Suites: `Small_ZoneGroups`.

Tests `K_ZONE_GROUPS` and `is_zone_group` in
`modules/core/profile/include/SushiAI/profile/zone_groups.hpp`, and the group check that
`SA_ZONE` and `SA_ZONE_FINE` run through `SA_REQUIRE_ZONE_GROUP` in `profile.hpp`. The file
names each of the ten groups in a `SA_REQUIRE_ZONE_GROUP` at namespace scope, so a group removed
from the list stops the compile of this program. A call site with a group outside the list stops
the compile of its own file, in a build with the profiler and in one without.

## TheListHoldsTheSpecifiedGroups

`K_ZONE_GROUPS` equals `compile`, `copy`, `data`, `device`, `eager`, `eval`, `host`, `io`,
`optim` and `train`, the groups of section 3 of
`docs/agent/2026_10_26_PROFILER_ADOPTION/SPEC.md`. Adding a group changes this case in the same
commit.

## NoGroupIsEmptyOrRepeated

Every entry is non-empty and occurs once.

## IsZoneGroupAcceptsOnlyTheListedGroups

Each listed group passes `is_zone_group`; the empty string, `hos`, `hosts`, `Host` and `BLAS`
do not.
