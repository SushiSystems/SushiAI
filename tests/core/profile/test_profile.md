# tests/core/profile/test_profile.cpp

Suites: `Medium_Profile`.

Tests `SA_ZONE` in `modules/core/profile/include/SushiAI/profile/profile.hpp`. The case runs a
real SushiProfiler session that follows one timeline.

## AZoneCarriesItsGroupAndStageName

`SA_ZONE("data", "read")` in each of three periods records a zone site named `data.read` in the
group `data`. The test is compiled only when `SA_PROFILE_LEVEL` is at least 1, since below that
level `SA_ZONE` records nothing.
