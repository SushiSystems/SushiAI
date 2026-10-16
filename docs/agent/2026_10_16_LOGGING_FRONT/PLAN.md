# Plan: the SushiAI logging front

SushiAI logs through SushiRuntime's logger and keeps its own front: `SA_LOG_*` macros,
`sushiai.<module>` categories and `SA_LOG_LEVEL`. The rules are in the `logging` skill
(SushiSkills 2b7b432).

## Dependency

SushiRuntime `feature/logging_standard` must provide three things:
- `make_log_category(project, module)`;
- per-project levels through `set_log_level(project, level)` and the project's environment
  variable;
- the project and the module in the record and in the sink.

None of the tasks below starts until that lands on SushiRuntime `main`, or the owner allows
building against the branch.

## Tasks

Each task has one acceptance criterion and one build (3 threads).

**T1. Tool refresh.**
- Files: `tools/` from SushiSkills, `tools/settings.py`, `modules/vision/datasets/README.md`.
- Settings: `K_LOG_PREFIX = "SA"`, and `K_CONSOLE_PATHS` lists `applications/`, the CLI
  command layer, `modules/vision/darknet_forward_cases/` and
  `modules/vision/darknet_training_cases/`. The `[ MEASURED ]` lines are test reports.
- Acceptance: `python tools/check_all.py .` reports only `check.cpp:43`.

**T2. The front, in `core/core`.**
- `SushiAI/core/log.hpp` defines `SA_LOG_TRACE` through `SA_LOG_FATAL`, each forwarding to
  `SR_LOG_AT`. It includes only SushiRuntime's public logging headers.
- `SushiAI/core/core_log_category.hpp` declares `K_CORE_LOG_CATEGORY`, made with
  `make_log_category("sushiai", "core")`.
- `SA_LOG_LEVEL` is read through SushiRuntime's per-project entry point. SushiAI writes no
  parser, queue, thread or sink, and SushiRuntime gains no SushiAI-specific code.
- Acceptance: a small test sets a capturing sink and checks two things. An `SA_LOG_WARN`
  record carries `[sushiai.core]`. Setting the `sushiai` project to `off` silences it and
  leaves a `sushiruntime` category at its level.

**T3. The fatal handler.**
- `write_fatal_report` in `check.cpp` writes through
  `SA_LOG_FATAL(K_CORE_LOG_CATEGORY, ...)`.
- The `SA_EXPECT_FATAL` matchers and the tests that match the old `sushiai: fatal:` text
  move to the record shape.
- Acceptance: `check_all` is clean, `sa test --label small` and `--label medium` pass, and
  every `...IsFatal` test still passes.

**T4. Docs close.**
- Write the changelog line.
- Update the `core/core` README for the front and the categories.
- Update `docs/architecture/` where it names stderr fatal output.
- Remove the KNOWN_ISSUES row if one was opened.
- Acceptance: `check_docs_truth.py` and `check_changelog.py` are clean.

Later modules gain their own `<module>_log_category.hpp` when they first log. This plan adds
no log call beyond the fatal report.

## Interim

The CI checkers fail today on the README path that T1 fixes. Until T2 lands,
`rule_console_output` on `check.cpp:43` keeps them red. A job is never disabled to get green.
