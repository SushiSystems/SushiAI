# SushiAI on SushiProfiler: specification

**Status:** Open — 2026-10-26; rewritten after review, under its second review, no code yet
**Implements:** PROFILING#P1, PROFILING#P2, PROFILING#P3, PROFILING#P4, PROFILING#P5

The topic is `docs/design/PROFILING/README.md`; its Decision table holds the rulings. This page
fixes the interfaces and what each changes. A first review returned "rewrite needed"; its findings
are answered in place below.

## 1. What this rests on

| Fact | Where |
| --- | --- |
| `Trainer::fit` sums `node_timings` only when `TrainerOptions::profile` is set | `modules/layers/train/source/trainer.cpp:502-511`, `trainer.hpp:101-102` |
| `RunReport::to_string` prints a wall clock and per-node rows | `trainer.cpp:186-224` |
| `fit` times itself with `steady_clock` into `duration_ms` | `trainer.cpp:480`, `:515-517` |
| The eager session times five stages with a private `Stopwatch` into `StepCost` | `modules/core/eager/source/session.cpp:52-73`, `:338-428`; `session.hpp:98-102` |
| Lowering calls one rule per IR node through SushiBLAS's `TaskRecorder` | `modules/core/graph/source/lowering.cpp:52-80`, `lowering/context.hpp:99-100` |
| A compiled step owns its `SushiRuntime::Graph`; the trailing work (optimizer, gradient exchange, carries) is recorded after lowering, through `StepRecorder::record_into_step` | `compiled_step.hpp:180`, `compiled_step.cpp:108-116` |
| No lowering rule records through `record_late_pair`; every task a rule records is inserted during the rule call | `git grep record_late_pair modules/core/graph` is empty |
| The eager path lowers through the same `Lowering::record` | `modules/core/eager/source/region_emitter.cpp:58-64` |
| The step loops | `epoch_loop.cpp:42-59`, `trainer.cpp:485-513`, `anchor_training_loop.cpp:76-103`, `free_training_loop.cpp:76-108`, `inference_program.cpp:95-118`, `domains_vision.cpp:488-512` and `:644-662` |
| Each program makes its runtime and engine before it runs | `applications/programs/source/train.cpp:78-79`, `inference.cpp:79-80` |
| A module names its edges with `PUBLIC_DEPENDS` and `PRIVATE_DEPENDS` | `cmake/SushiAIModule.cmake:176` |

What SushiProfiler v0.2.0 provides, read from its headers:

- `Session::start(const ProfilerConfig&, const Registry&)` returns a `SessionStart` holding the
  session or an `error` string (`session/session.hpp:59-67`, `:96-97`).
- `Session::timeline(std::string_view)` makes and owns a timeline; the window follows the one the
  configuration's `timeline` names (`session.hpp:107-108`, `config/profiler_config.hpp:76`). A
  timeline the session did not make is never observed.
- `Session::stop()` returns a `SessionStop` with the `record` and `problems`; `Session::report(record)`
  runs the configured analyses and record writers. `~Session()` drops an unstopped record.
- `read_profiler_config(std::string_view text)` returns a `ConfigReadResult` with an optional
  configuration or an `error` (`profiler_config.hpp:86-104`).
- `breakdown` groups zones by the site's `group` and its parents (`record_queries.cpp:133-145`); it
  does not split names.
- `DeviceSpanKeys::K_GRAPH` and `K_ORIGIN`, and `roofline` grouping by them (task C10).

What the layers below provide, fixed in their own plans:

- SushiRuntime: `RuntimeProfilerParts(SushiProfiler::Registry&, const Runtime&)`, neither copyable
  nor movable; `Graph::set_label(std::string)` (R15); `NodeOptions::origin` (R16).
- SushiBLAS: `[[nodiscard]] OriginScope TaskRecorder::origin(std::string_view label)` (B-O1), which
  stamps every node recorded inside it and restores the previous label when it ends.

## 2. The `profile` module

`modules/core/profile/`, tier `core`, one source file, the only module that includes SushiProfiler.
Consumers name it in `PRIVATE_DEPENDS`, never `PUBLIC_DEPENDS`.

`<SushiAI/profile/profile.hpp>`:

| Name | Expands to |
| --- | --- |
| `SA_PROFILE_LEVEL` | Compile-time ceiling, default 2; 0 removes every site |
| `SA_ZONE(group, stage)` | `SP_ZONE_AT(SA_PROFILE_LEVEL, 1, group "." stage, group)`; both arguments are string literals |
| `SA_ZONE_FINE(group, stage)` | The same at level 2 |
| `SA_COUNTER(name, unit, op, value)`, `SA_VALUE(name, unit, value)` | `SP_COUNTER_AT` and `SP_VALUE_AT` at level 1 |

Under `SA_ENABLE_PROFILER` 0 every macro expands to nothing.

`<SushiAI/profile/timelines.hpp>`:

```cpp
enum class Timeline : std::uint8_t
{
    TrainStep,      /**< "train.step": one period per optimizer step. */
    InferenceBatch, /**< "inference.batch": one period per batch run. */
};

/// @brief Opens the next period of @p timeline when a session is bound; otherwise does nothing.
void mark_period(Timeline timeline) noexcept;

/// @brief Returns the name of @p timeline, as a session's configuration names it.
[[nodiscard]] std::string_view timeline_name(Timeline timeline) noexcept;
```

`<SushiAI/profile/timeline_binding.hpp>`, included by the programs only:

```cpp
class TimelineBinding final
{
    public:
        /// @brief Points every Timeline at the session's timeline of the same name.
        explicit TimelineBinding(SushiProfiler::Session& session);
        TimelineBinding(const TimelineBinding&) = delete;
        TimelineBinding& operator=(const TimelineBinding&) = delete;
        TimelineBinding(TimelineBinding&&) = delete;
        TimelineBinding& operator=(TimelineBinding&&) = delete;
        /// @brief Unbinds every Timeline; mark_period does nothing afterwards.
        ~TimelineBinding();
};
```

`source/timelines.cpp` holds one `std::atomic<SushiProfiler::PeriodTimeline*>` per timeline,
null while unbound; `mark_period` loads it with acquire order and marks when it is not null. A
process runs one session (SushiProfiler's rule), so the binding is process-wide; a second
`TimelineBinding` while one lives is a programming error, caught by `SA_ASSERT`. Binding and
unbinding happen on the thread that runs the step loop, with no mark in flight, which the
programs' order in section 6 gives.

## 3. Zones and periods

| Group | Stage zones and where |
| --- | --- |
| `data` | `read`: every `dataset.read` in the loops of section 1 |
| `host` | `feed`: `objective_->feed` (`trainer.cpp:371`) and the truth-from-roles calls; `readback`: `outputs_view_` (`anchor_training_step.cpp:217-219`), `floats_of` and `widen` (`free_training_step.cpp:310-311`, `:376-377`); `assign`: `feed_deltas`, `branch.stage.assign`, `build_target_leaves`; `loss`: `current_loss`, `measure_terms`; `measure`: `objective_->measure` |
| `copy` | `input`: every `blas.copy` of a batch into its input buffer |
| `device` | `run`: `step_->run()`, `training_->run`, `run.step.run` |
| `optim` | `schedule`: `learning_rate` and `apply_step_schedule` |
| `eval` | `decode`, `nms`, `ap`: `decode_batch` or `decoder.decode`, `select_nms_detections`, `add_batch` |
| `io` | `checkpoint`: `save_run_checkpoint`, `IO::load_checkpoint` |
| `compile` | `lower`, `runtime`: `lowering.record` and `graph_.compile()` in `CompiledStep::compile` |
| `eager` | `partition`, `plan`, `compile`, `execute`, `flush`: the five stopwatch laps of the eager session |
| `train` | `fit`: the whole of `Trainer::fit` |

`mark_period(Timeline::TrainStep)` opens each training step, before its `data.read`, in
`run_epochs`, `Trainer::fit`, the anchor loop and the anchor-free loop.
`mark_period(Timeline::InferenceBatch)` opens each batch in the classifier loop and both detector
evaluation loops. The step's `device.run` waits for its device work, so its device spans and the
anchor-free host pause fall inside the period that opened the step.

## 4. What leaves the library

| Removed | Replacement |
| --- | --- |
| `TrainerOptions::profile` | A profiling session of the program |
| `RunReport::NodeTimings`, `RunReport::node_timings`, `find_total`, `add_node_timings` | The `roofline` analysis of a record |
| `RunReport::duration_ms` and the wall-clock line of `to_string` | The `train.fit` zone |
| `StepCost::partition_ns`, `plan_ns`, `compile_ns`, `execute_ns`, `total_ns` and `Stopwatch` | The `eager.*` zones; the counts of `StepCost` stay |

Consumers that change with them:

- `tests/layers/train/test_trainer/test_convergence.cpp:508-549`, the test
  `ProfilingOnAnUnprofiledContextReportsNothingRatherThanZeroes`, is deleted with its paragraph in
  `test_convergence.md:52`.
- `tests/core/eager/test_session.cpp:452-485` stops summing the stage times and keeps its count
  assertions.
- `applications/examples/source/demo_mlp.cpp:737` stops summing `duration_ms`.
- The `last_report()` comment in `compiled_step.hpp:151-152` and the label comment in `ir.hpp:184`
  stop naming node timing and a profiling report.

These are breaking changes to public names, made without a deprecation release (before 1.0,
`api-stability` allows it); each has a changelog entry naming its replacement.

## 5. Graph labels and node origins

- `CompiledStep::compile` gains a last parameter `std::string_view label = {}`, handed to
  `graph_.set_label` before `graph_.compile()`, so the first plan carries it. Callers and labels:
  `TrainingStep` gives `train.lead` to `lead_` and `train.step` to `step_`
  (`training_step.cpp:166-175`), `EvaluationStep` gives `eval.step` (`evaluation_step.cpp:192`),
  `ReplicaStep` gives `replica.step` (`replica_step.cpp:125`); the last two pass `nodes`
  explicitly. Callers that pass no label (the tests) keep compiling. The eager `RegionExecutor`
  labels its graph `eager.region.<n>`, `n` its region index, before its first run.
- `Lowering::record` opens `recorder_->origin(label)` around each rule call, `label` being
  `<node id>:<op name>` with the op name from `Ops::name_of`; the eager path gets the same through
  `region_emitter.cpp`.
- `CompiledStep::compile` opens `recorder_.origin("step.trailing")` around
  `trailing->record_into_step`, so the optimizer, gradient exchange and carries are named.
- Labels are built once per compile and never on a run.

## 6. The programs' session

`applications/programs/source/profile_session.hpp` and `.cpp`:

```cpp
class ProfileSession final
{
    public:
        /// @brief Reads the SushiProfiler configuration at @p config_path and starts a session.
        [[nodiscard]] static Result<std::unique_ptr<ProfileSession>> open(
            const std::filesystem::path& config_path, const SushiRuntime::Runtime& runtime);

        ProfileSession(const ProfileSession&) = delete;
        ProfileSession& operator=(const ProfileSession&) = delete;
        ProfileSession(ProfileSession&&) = delete;
        ProfileSession& operator=(ProfileSession&&) = delete;

        /// @brief Unbinds, stops the session and hands the record to its writers and analyses.
        [[nodiscard]] Status close();

        /// @brief Calls close() if it was not called, logging its problems.
        ~ProfileSession();
};
```

- `open` reads the file, calls `read_profiler_config`, builds a `SushiProfiler::Registry`,
  constructs `SushiRuntime::RuntimeProfilerParts` in place, calls `Session::start` and makes a
  `TimelineBinding`. An unreadable file, a configuration refusal or a start error returns an error
  that names the file and quotes SushiProfiler's text.
- Members, in declaration order: the registry, the runtime parts, the session, the binding; so
  destruction drops the binding first and the parts last.
- `close` destroys the binding, calls `stop()`, then `report(record)`, and logs every entry of
  both `problems` lists through `SA_LOG_WARN` on the `programs` category; it returns an error when
  the session could not be stopped.
- `train.cpp` and `inference.cpp` take `--profile <path>`, open the session right after the engine
  (`train.cpp:79`, `inference.cpp:80`), so the runtime and engine outlive it, and call `close()`
  after the run, whether it succeeded or not. A run that fails still writes its record.
- `sa train --profile` and `sa inference --profile` forward the flag. `sa profile <args>` runs
  `sp <args>` and, when `sp` is not on the path, says to install SushiProfiler's CLI.

## 7. Dependencies and build

- `cli/sushistack.deps.toml`: `[sushisystems.sushiprofiler]` pinned to v0.2.0; `depends_on` gains
  `sushiprofiler`.
- Options `SA_ENABLE_PROFILER` (ON) and `SA_PROFILE_LEVEL` (2) are compile definitions of the
  `profile` module only, `PUBLIC` on it, so they reach the modules that name it and no other; a
  change of level rebuilds those sources and nothing else.
- `cmake/Profiler.cmake`: `find_package(SushiProfiler CONFIG REQUIRED)` when `SA_ENABLE_PROFILER`
  is ON.
- The package consumption of SushiLog, SushiRuntime and SushiBLAS is task O5 of SushiCore's
  `docs/agent/2026_10_26_PREBUILT_SUSHI_PACKAGES/PLAN.md`, applied here as T6.

## 8. Build time

| Change | What rebuilds |
| --- | --- |
| A zone added in a `.cpp` | That file |
| `profile.hpp` or `SA_PROFILE_LEVEL` | The sources of the modules that name `profile` |
| `timeline_binding.hpp` | `profile_session.cpp` and `timelines.cpp` |
| A SushiRuntime or SushiBLAS `.cpp` | Nothing in SushiAI compiles; the executables relink |

No public header of SushiAI includes a `profile` header or a SushiProfiler header.

## 9. Not in this work

- P6, the YOLOv5n write-up, runs after the build.
- The distributed path's timing (`cluster_run.cpp`) has the zones of its own loop only; SushiRuntime
  times the distributed layer in its own later phase.
