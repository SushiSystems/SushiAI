# profile

## Owns

SushiAI's front on SushiProfiler and the periods of its loops. It is the one module that
includes SushiProfiler. `profile.hpp` defines the `SA_ZONE`, `SA_ZONE_FINE`, `SA_COUNTER` and
`SA_VALUE` macros, which forward to SushiProfiler's `SP_ZONE_AT`, `SP_COUNTER_AT` and
`SP_VALUE_AT` under the ceiling `SA_PROFILE_LEVEL`. `timelines.hpp` names the two timelines,
`train.step` and `inference.batch`, and `mark_period` opens the next period of one. A program
that runs a session creates a `TimelineBinding`, which points each `Timeline` at the session's
timeline of the same name; with no binding, `mark_period` does nothing.

`SA_ENABLE_PROFILER` and `SA_PROFILE_LEVEL` come from `profile_config.hpp`, which the build
generates from `profile_config.hpp.in`. Under `SA_ENABLE_PROFILER` 0 every macro expands to
nothing and `mark_period` and `TimelineBinding` do nothing. Nothing is defined on the command
line, so a change of level rebuilds the sources that include `profile.hpp` and nothing else.

The tests of other modules read a record through `modules/core/profile_records`.

A zone is a group and a stage, both string literals: `SA_ZONE("host", "readback")` records the
site `host.readback` in the group `host`. The groups and stages are in
[`docs/agent/2026_10_26_PROFILER_ADOPTION/SPEC.md`](../../../docs/agent/2026_10_26_PROFILER_ADOPTION/SPEC.md),
section 3, and the topic is [`docs/design/PROFILING/README.md`](../../../docs/design/PROFILING/README.md).

## Depends on

The module names `failure` privately, for `SA_ASSERT` in the binding, and links SushiProfiler
privately through `EXTERNAL_PRIVATE_LINKS`. A module that expands the macros names `profile`
in `PRIVATE_DEPENDS` and links SushiProfiler privately. No public header of SushiAI includes a
`profile` header or a SushiProfiler header, so a zone added in a `.cpp` rebuilds that file only.

## Public entry points

Each header is named by its include path under `include/`.

| Header | Declares |
| --- | --- |
| `SushiAI/profile/profile.hpp` | `SA_ZONE`, `SA_ZONE_FINE`, `SA_COUNTER`, `SA_VALUE`; included by source files only |
| `SushiAI/profile/timelines.hpp` | `Timeline`, `mark_period`, `timeline_name` |
| `SushiAI/profile/timeline_binding.hpp` | `TimelineBinding`; included by the programs only |
