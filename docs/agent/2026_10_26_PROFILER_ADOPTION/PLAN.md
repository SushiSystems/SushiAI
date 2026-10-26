# SushiAI on SushiProfiler: plan

**Status:** Open — 2026-10-26; written, under review, no code yet
**Implements:** PROFILING#P1, PROFILING#P2, PROFILING#P3, PROFILING#P4, PROFILING#P5

Spec: `SPEC.md` in this folder. Every task closes on one `sa build` and `sa test` at one job
(the owner moved every build to the end of the programme), its tests passing, and
`python tools/check_all.py .` clean. Workers write no build file and run no build; each task lists
its build lines for the orchestrator.

## Global constraints

- SushiProfiler v0.2.0 names only; nothing from v0.3.0.
- No public header of SushiAI includes `<SushiAI/profile/...>` or `<SushiProfiler/...>`.
- Zone names exactly as SPEC section 3; timeline names `train.step` and `inference.batch`.
- Breaking removals carry one changelog entry each, naming the replacement.
- Code needs SushiRuntime's R15 and R16 (`Graph::set_label`, `NodeOptions::origin`) and SushiBLAS's
  B-O1 (`TaskRecorder::origin`); a task that uses them is written against those signatures exactly.

## Review focus

1. A profiled run with no `--profile`: no session, no record file, no measurable cost. T1's test
   calls `mark_period` with no session.
2. A `--profile` path that does not exist or holds an unknown key: the program stops before
   training, naming the file. T5's test.
3. Two compiled steps with equal node indices: the record keeps them apart by `graph`. T4's test.
4. A lowering rule that records no task (a view): no origin is left open for the next rule. T4's
   test.
5. A run that ends with an error mid-step: `ProfileSession` still stops its session and writes the
   record. T5's test.

## Tasks

| Id | Task | Who | Acceptance |
| --- | --- | --- | --- |
| A1 | Build lines: tier entry `profile core`; `modules/core/profile/CMakeLists.txt` `sushiai_add_module(NAME profile TIER core SOURCES timelines.cpp EXTERNAL_PUBLIC SushiProfiler::SushiProfiler)` (external dropped when `SA_ENABLE_PROFILER` is OFF); options `SA_ENABLE_PROFILER` ON and `SA_PROFILE_LEVEL` 2 as compile definitions on every module; `cmake/Profiler.cmake` with `find_package(SushiProfiler CONFIG REQUIRED)`, included from the root after `include(Log)`; `[sushisystems.sushiprofiler]` v0.2.0 and `depends_on` in `cli/sushistack.deps.toml`; `tests/core/profile/CMakeLists.txt` | orchestrator | Configure succeeds with the option ON and OFF |
| T1 | `<SushiAI/profile/profile.hpp>` and `<SushiAI/profile/timelines.hpp>` as SPEC section 2; `source/timelines.cpp`; module README; `tests/core/profile/test_timelines.cpp` | worker | `Small_Timelines.NamesAreTrainStepAndInferenceBatch` and `Small_Timelines.MarkWithNoSessionDoesNothing` pass |
| T2 | Remove SPEC section 4's names: `trainer.hpp`, `trainer.cpp` (`find_total`, `add_node_timings`, the timing lines of `to_string`, `duration_ms`), `session.hpp`/`session.cpp` (`Stopwatch`, the five `_ns` fields); rewrite `test_convergence.cpp:528-548` to check `to_string` names steps and losses only; `example.cpp` and `demo_mlp.cpp:726-759` stop reading `duration_ms`; `git grep` the names over `*.md` and correct each hit; changelog entries | worker | Every test that named a removed member compiles against the new surface, and `git grep -n "node_timings\|duration_ms\|partition_ns" -- modules tests applications` is empty |
| T3 | Zones and periods of SPEC section 3 in the files it names, with `#include <SushiAI/profile/profile.hpp>` and `timelines.hpp` in those `.cpp` files only; `DEPENDS profile` lines listed for `train`, `graph`, `eager`, `vision/training`, `programs`; READMEs name the zones of their module | worker | `Medium_TrainingZones.AStepWritesOnePeriodWithItsGroups` (a two-step classifier fit under a test session gives two `train.step` periods, each with `data.read`, `copy.input` and `device.run`) passes |
| T4 | `CompiledStep::compile(..., std::string_view label)` and its callers with SPEC section 5's labels; `Lowering::record` opens `recorder_->origin("<id>:<op>")` per rule; `RegionExecutor` labels `eager.region.<n>`; tests in `tests/core/graph/` attach a `SushiRuntime::RunObserver` and read `RunObservation::table` | worker | `Medium_CompiledStepAttribution.EveryTaskOfANodeCarriesItsOrigin` (a node lowering to three tasks gives three rows with one origin), `Medium_CompiledStepAttribution.TwoStepsHaveTwoLabels` and `Medium_CompiledStepAttribution.AViewLeavesNoOriginOpen` pass |
| T5 | `ProfileSession` (SPEC section 6) in `applications/programs`; `--profile` on `train.cpp` and `inference.cpp`; `sa train --profile`, `sa inference --profile` and `sa profile` in `cli/sushiai/cli.py` with tests under `cli/tests/`; `docs/guides/` page on profiling a run | worker | `Medium_ProfileSession.AMissingConfigNamesTheFile`, `Medium_ProfileSession.ARunThatFailsStillWritesItsRecord` and the CLI tests `test_train_forwards_profile`, `test_profile_runs_sp` pass |
| T6 | SushiCore's O5 lines applied: `cmake/Log.cmake`, `Runtime.cmake`, `BLAS.cmake`, `SushiAISyclTarget.cmake`, `services/project.py`'s `-D..._DIR` replaced by `sushi_packages` | orchestrator | `sa build` configures against the packages and compiles no SushiRuntime or SushiBLAS source |
| T7 | P6: two YOLOv5n runs, one on the production path with `level` 1 and one with device timing; `sp compare`; the write-up in `REPORT.md` and the design topic | orchestrator | The report names each group's share of a step and the kernels and IR nodes that bound it |

Order: A1, T1, then T2, T3 and T4 in one wave with disjoint files (T3 and T4 both touch
`compiled_step.cpp`: T3 owns its zones, T4 its label, so T4 runs after T3), then T5, T6 and T7.

## What the first build must confirm

- `SushiProfiler::SushiProfiler` is found from its package, and its DLL is on the CLI's path.
- `Graph::set_label`, `NodeOptions::origin` and `TaskRecorder::origin` exist with the signatures
  used.
- A level-0 build (`SA_PROFILE_LEVEL=0`) compiles every zone away.
