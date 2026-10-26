# tests/core/profile/test_profile.cpp

Suites: `Medium_Profile`.

Tests `SA_ZONE` in `modules/core/profile/include/SushiAI/profile/profile.hpp`. The case runs a
real SushiProfiler session that follows one timeline.

## AZoneCarriesItsGroupAndStageName

`SA_ZONE("data", "read")` in each of three periods records a zone site named `data.read` in the
group `data`. The case skips itself when `SA_PROFILE_LEVEL` is 0.
