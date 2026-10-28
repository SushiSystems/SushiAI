# profile_records

An internal module: only the test programs link it, its headers are never installed, and it is built only when `SA_BUILD_TESTS` and `SA_ENABLE_PROFILER` are on.

## Owns

The helpers that tests of other modules repeated when they read a profiling record: `following_timeline`, which builds a `ProfilerConfig` that follows one timeline for a window; `zones_of`, which maps the zone sites of a period to their groups; `zones_in_groups`, which keeps the zones of named groups; `sushiai_zones_of`, which keeps the zones of `Profile::K_ZONE_GROUPS`, SushiAI's own groups, and leaves out those of SushiRuntime and SushiBLAS; `write_text` and `read_text`; and `write_profile_config`, which writes a configuration file that follows `train.step` for two periods and writes one record.

## Depends on

SushiProfiler, as a public link, because its headers name `ProfilerConfig` and `RunRecord`. Its header includes `SushiAI/profile/zone_groups.hpp` of the `profile` module for the one list of zone groups; the test program reaches that folder through the library it links, and `CMakeLists.txt` does not yet name `profile` in `PUBLIC_DEPENDS`.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/profile_records/profile_records.hpp` | `following_timeline`, `zones_of`, `zones_in_groups`, `sushiai_zones_of`, `write_text`, `read_text`, `write_profile_config` |
