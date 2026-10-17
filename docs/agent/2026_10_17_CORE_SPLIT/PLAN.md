# Plan: split core/core by responsibility

The owner's decisions on the task 9 report (relayed by sushiskills-3a, 2026-10-17):
- The logging front gets its own module, `core/log`.
- `core/core` is split now, one module per responsibility, with its edges declared. Each module
  links only what it uses.
- The CLI tests stop running a CLI from outside the repository.
- SushiRuntime and SushiBLAS stay unpinned until the owner tags them.

This plan replaces `core/core` and absorbs the uncommitted work of
`docs/agent/2026_10_16_LOGGING_FRONT/`.

## What `core/core` holds today

It has 12 headers and 4 sources, 2 587 lines in all. The "users" column counts the files outside
`core/core` that include each header.

| Header | Holds | Users |
| --- | --- | --- |
| `check.hpp`, `status.hpp`, `result.hpp` | `SA_FATAL_IF`, `Status`, `Result<T>`, the fatal handler | 101, 476, 358 |
| `common.hpp` | `DType`, `Layout`, `K_MAX_RANK` and the dtype helpers; `Error` and `SA_THROW_IF`; the alias `sa` | 281 |
| `shape.hpp`, `tensor_shape.hpp` | `Shape`, `shape_of` | 311, 11 |
| `ops.hpp` | the operation identities | 118 |
| `seam.hpp` | the guards that turn SushiRuntime and SushiBLAS failures into a `Status` | 89 |
| `allocation.hpp` | tensor and arena allocation | 53 |
| `log.hpp`, `sushiai_log_project.hpp`, `core_log_category.hpp` | the logging front (uncommitted) | 1 |

## Modules after the split

Each header moves to `SushiAI/<module>/<name>.hpp`. Tier `core` throughout.

| Module | Headers | Depends on | External |
| --- | --- | --- | --- |
| `log` | `log.hpp`, `sushiai_log_project.hpp` | none | `SushiLog::SushiLog` |
| `failure` | `check.hpp`, `status.hpp`, `result.hpp`, `error.hpp` (`Error`, `SA_THROW_IF` out of `common.hpp`), `failure_log_category.hpp` | `log` (private: `check.cpp`) | SushiBLAS `format.hpp` |
| `dtype` | `dtype.hpp` (`DType`, `Layout`, `K_MAX_RANK`, the helpers, out of `common.hpp`) | none | `SushiBLAS::sushiblas` |
| `shape` | `shape.hpp`, `tensor_shape.hpp` | `dtype`, `failure` | `SushiBLAS::sushiblas`, `SushiRuntime::SushiRuntime` |
| `ops` | `ops.hpp` | none | `SushiRuntime::SushiRuntime` |
| `seam` | `seam.hpp` | `failure` | `SushiBLAS::sushiblas`, `SushiRuntime::SushiRuntime` |
| `allocation` | `allocation.hpp` | `failure`, `dtype`, `shape`, `seam` | `SushiBLAS::sushiblas`, `SushiRuntime::SushiRuntime` |

- **`dtype` is a sixth module that was not on the owner's list.** `DType`, `Layout` and
  `K_MAX_RANK` share `common.hpp` with `Error`, but they describe elements, not failures. `ops`
  and `seam` need neither, and a file that needs only `DType` stops pulling in `Error`.
- **The alias `sa` moves to the umbrella `SushiAI/SushiAI.hpp`.** Six files outside the library
  use it: the examples, the programs and `tests/package/vision_main.cpp`, and each reaches
  the umbrella, directly or through `SushiAI/vision.hpp`. Removing it would change the public surface, which is the owner's
  decision.
- Every module that names `core` today in `PUBLIC_DEPENDS` or `PRIVATE_DEPENDS` instead names
  the modules whose headers it includes, public or private by where the include sits.
- `SushiRuntime::SushiRuntime` narrows to SushiRuntime's per-module targets once sushiruntime-d1
  exports them. That is a later task with a build of its own.

## A question for the owner before T1

The decision put "the log category headers" in `core/log`. The `logging` skill and
`check_logging.py` (`rule_missing_category_header`, `rule_foreign_category`) want the opposite:
a module that logs holds its own `<module>_log_category.hpp` and logs only under
`K_<MODULE>_LOG_CATEGORY`. A category inside `core/log` would fail both rules. This plan follows
the checker: `core/log` holds the project, the ceiling and the macros, and `failure` holds
`failure_log_category.hpp`, so the fatal record reads `[sushiai.failure]`.

## Tasks

Hierarchy first: a module exists, with its edges, before anything includes it. Every task ends
with `python tools/check_all.py .` clean. A build is 3 threads, `sa test --label small` and
`--label medium`.

**T1. `core/log` and `failure`.**
- Create both modules and move the front and the failure headers into them.
- Split `Error` and `SA_THROW_IF` out of `common.hpp`.
- Rewrite every include of `check.hpp`, `status.hpp` and `result.hpp`, plus the `Error` uses of
  `common.hpp`. Declare the dependencies.
- Acceptance: build 1 green, and the fatal tests match `[FATAL] [sushiai.failure]`.

**T2. `dtype`, `shape` and `ops`.**
- Create the three modules and move their headers.
- Rewrite the remaining `common.hpp` users to `dtype.hpp` and remove `common.hpp`.
- Acceptance: build 2 green.

**T3. `seam` and `allocation`; `core/core` goes.**
- Create the two modules and move their headers and sources.
- Delete `modules/core/core` and `tests/core/core`, whose tests move to their modules' test
  folders.
- Update the layer table and the umbrella.
- Acceptance: build 3 green, and no file names `SushiAI/core/` or the module `core`.

**T4. The CLI tests run this checkout's CLI.**
- `test_demo.py`, `test_train.py`, `test_import.py` and `test_tokenize.py` call
  `[sys.executable, "-m", "sushiai", ...]` in place of `sa` or `sushiai` from PATH.
- The `shutil.which` skips go: they hid the dependency instead of removing it.
- Acceptance: `pytest cli/tests` passes with `~/.local/bin` off PATH. The tests that run a built
  program need build 3's tree; no C++ build of their own.

**T5. Per-module SushiRuntime targets.** This waits on sushiruntime-d1.
- Each module links only the SushiRuntime targets it includes.
- Acceptance: build 4 green.

**T6. Docs close.**
- Write one module README per new module and update `docs/architecture/` for the layer table.
- Write the changelog lines and close `2026_10_16_LOGGING_FRONT`.
- Acceptance: `check_docs_truth.py` and `check_changelog.py` are clean.

## Build count

There are four full builds: one each for T1, T2 and T3, then one for T5 when SushiRuntime's
targets land. T4 and T6 build nothing. Each build recompiles almost the whole tree, because
every module's include paths change. None starts before the owner's order and before SushiBLAS's
adoption is on its `main`.

T1 to T3 can instead be written in one pass and built once. That is one build in place of three,
but a failure points at a larger change.

## Pins

Only SushiLog is pinned, at `v0.1.2`. SushiRuntime and SushiBLAS are pinned when the owner tags
them.
