# Plan: the SushiAI logging front

SushiAI logs through SushiLog, the one logger every Sushi library links, and keeps its own
front: `SA_LOG_*` macros, `sushiai.<module>` categories and `SA_LOG_LEVEL`. This is task 9 of
SushiSkills `docs/agent/2026_10_16_SUSHILOG_REPOSITORY/PLAN.md`; the rules are in the `logging`
and `dependencies` skills.

## Dependency

- SushiLog v0.1.2 (`01da0e7464040a02025f44e50cfca64579215a22`) and `sushicore` 0.9.1 are
  released.
- Task 7, SushiRuntime's front over SushiLog, is on SushiRuntime `main` (`e59249f`).
- SushiBLAS's front must also move to SushiLog (the SushiBLAS half of task 9). SushiAI includes
  SushiBLAS headers that log, so it cannot build until SushiBLAS builds against SushiLog.

## Open question for the owner

The `dependencies` skill pins each Sushi library with a release tag and its commit. SushiLog
has one: `v0.1.2`. SushiRuntime's latest tag, `v0.3.0`, predates the SushiRuntime that adopts
SushiLog, and SushiBLAS has no tag. Until both tag a release, the fragment can pin only
SushiLog. Pinning a commit with no tag breaks rule 1 of that section.

## Tasks

Each task has one acceptance criterion and one build (3 threads).

**T1. Tool refresh.** Done (`ffa9ee1`, `f4ac032`).

**T2. Resolve and pin SushiLog.**
- `cli/sushistack.deps.toml`: `sushilog` in `[module] depends_on`, and `[sushi.sushilog]` with
  `version = "v0.1.2"` and the full commit. SushiRuntime and SushiBLAS are pinned the same way
  once they have tags (see the open question).
- `cli/pyproject.toml`: `sushicore==0.9.1`.
- `cmake/Log.cmake`, included before `Runtime.cmake`: if no `SushiLog::SushiLog` target exists
  yet, `add_subdirectory` the path from `SUSHILOG_DIR`, the sibling `../sushilog` or the
  copy `sa setup` fetched, in that order. CMake never downloads. The target guard lets
  SushiRuntime and SushiBLAS reuse the same target.
- CI checks SushiLog out at the pin, as it does SushiRuntime and SushiBLAS.
- Acceptance: `sa setup` on a clone with no sibling fetches SushiLog at `v0.1.2`, and
  `sa build` configures with one `SushiLog::SushiLog` target.

**T3. The front, in `core/core`.**
- `SushiAI/core/log.hpp`: `SA_LOG_TRACE` through `SA_LOG_FATAL` on `SL_LOG_FRONT_AT`, with a
  per-translation-unit ceiling from `SushiLog::log_level_ceiling(SA_ENABLE_LOGGING != 0)`.
- `SushiAI/core/sushiai_log_project.hpp`: `K_SUSHIAI_LOG_PROJECT{"sushiai", "SA_LOG_LEVEL"}`.
- `SushiAI/core/core_log_category.hpp`: `K_CORE_LOG_CATEGORY`, module `core`.
- `core/core` links `SushiLog::SushiLog` in `EXTERNAL_PUBLIC`. It includes SushiLog headers
  only, never SushiRuntime's for logging.
- The umbrella names the three headers.
- Acceptance: `test_log.cpp` sets a capturing sink and checks two things. An `SA_LOG_WARN`
  record carries `sushiai.core`. Setting the `sushiai` project to `Off` silences it and leaves
  the `sushiruntime` project at its level.

**T4. The fatal handler.**
- `write_fatal_report` in `check.cpp` writes through `SA_LOG_FATAL(K_CORE_LOG_CATEGORY, ...)`.
- `test_status.cpp` matches `[FATAL] [sushiai.core]` instead of `sushiai: fatal:`.
- Acceptance: `check_all` is clean, `sa test --label small` and `--label medium` pass, every
  `...IsFatal` test still passes, and one test process shows SushiAI and SushiRuntime records in
  one stream.

**T5. Docs close.**
- Write the changelog lines, one per scope.
- Update the `core/core` README and `tests/core/core/README.md`.
- Update KNOWN_ISSUES: remove the `check.cpp:43` row.
- Acceptance: `check_docs_truth.py` and `check_changelog.py` are clean.

Later modules gain their own `<module>_log_category.hpp` when they first log. This plan adds
no log call beyond the fatal report.

## Work in the tree

T2 to T5 are written on SushiLog and uncommitted; nothing is built. The build waits for the
owner's order and for SushiBLAS's adoption on its `main`.
- `check_all` is clean, `rule_front_reexport` included.
- 99 CLI tests pass. The 33 that spawn `sa.exe` or `sushiai.exe` from `~/.local/bin` fail
  there: that Python has no `sushicore`.
- The engine fixture's `main` calls `SushiLog::set_log_synchronous(true)` once, in place of the
  SushiRuntime and SushiBLAS calls.

## Boundary items with the owner

- `core/core` holds the failure contract, `Shape`, allocation, the op identities and the seam;
  the front is a sixth responsibility.
- `core/core` links all of `SushiRuntime::SushiRuntime`.
