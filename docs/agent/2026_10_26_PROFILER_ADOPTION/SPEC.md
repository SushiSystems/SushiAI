# SushiAI on SushiProfiler: specification

**Status:** Open — 2026-10-26; written, under review, no code yet
**Implements:** PROFILING#P1, PROFILING#P2, PROFILING#P3, PROFILING#P4, PROFILING#P5

The topic is `docs/design/PROFILING/README.md`; its Decision table holds the rulings. This page
fixes the interfaces and what each changes.

## 1. What this rests on

| Fact | Where |
| --- | --- |
| `Trainer::fit` sums `node_timings` only when `TrainerOptions::profile` is set | `modules/layers/train/source/trainer.cpp:502-511`, `trainer.hpp:101-102` |
| `RunReport::to_string` prints a wall clock and per-node rows | `trainer.cpp:186-222` |
| `fit` times itself with `steady_clock` into `duration_ms` | `trainer.cpp:480`, `:515-517` |
| The eager session times five stages with a private `Stopwatch` into `StepCost` | `modules/core/eager/source/session.cpp:52-73`, `:338-428`; `session.hpp:98-102` |
| Lowering calls one rule per IR node through SushiBLAS's `TaskRecorder` | `modules/core/graph/source/lowering.cpp:52-80`, `lowering/context.hpp:99-100` |
| A compiled step owns its `SushiRuntime::Graph` | `compiled_step.hpp:180` |
| The step loops | `epoch_loop.cpp:42-59`, `trainer.cpp:485-513`, `anchor_training_loop.cpp:76-103`, `free_training_loop.cpp:76-108`, `inference_program.cpp:95-118`, `domains_vision.cpp:488-512` and `:646-660` |
| No SushiProfiler reference exists in SushiAI's CMake or fragment | `cmake/`, `cli/sushistack.deps.toml` |

What the layers below provide, fixed in their own plans:

- SushiProfiler v0.2.0: `PeriodTimeline`, the `SP_ZONE` family, `Session`, `Registry`,
  `DeviceSpanKeys::K_GRAPH` and `K_ORIGIN`, and the `breakdown`, `device_idle` and `roofline`
  analyses.
- SushiRuntime: `register_profiler_parts(SushiProfiler::Registry&, const Runtime&)` returning a
  `RuntimeProfilerParts` that is neither copyable nor movable; `Graph::set_label(std::string)`;
  `NodeOptions::origin`. `RunReport` no longer carries any timing.
- SushiBLAS: `[[nodiscard]] OriginScope TaskRecorder::origin(std::string_view label)`, which stamps
  every node recorded inside it.

## 2. The `profile` module

`modules/core/profile/`, tier `core`, header-only, the only module that includes SushiProfiler.

`<SushiAI/profile/profile.hpp>`:

| Name | Meaning |
| --- | --- |
| `SA_PROFILE_LEVEL` | Compile-time ceiling, default 2; 0 removes every site |
| `SA_ZONE(name)` | `SP_ZONE_AT(SA_PROFILE_LEVEL, 1, name, nullptr)`, or nothing when `SA_ENABLE_PROFILER` is 0 |
| `SA_ZONE_FINE(name)` | The same at level 2 |
| `SA_COUNTER(name, unit, op, value)`, `SA_VALUE(name, unit, value)` | Forward to `SP_COUNTER_AT` and `SP_VALUE_AT` at level 1 |

`<SushiAI/profile/timelines.hpp>`:

```cpp
enum class Timeline : std::uint8_t
{
    TrainStep,      /**< "train.step": one period per optimizer step. */
    InferenceBatch, /**< "inference.batch": one period per batch run. */
};

/// @brief Opens the next period of @p timeline; does nothing when the profiler is compiled out.
void mark_period(Timeline timeline) noexcept;

/// @brief Returns the name of @p timeline, as a session's window names it.
[[nodiscard]] std::string_view timeline_name(Timeline timeline) noexcept;
```

`mark_period` is defined in `source/timelines.cpp`, which holds one function-local
`SushiProfiler::PeriodTimeline` per timeline. The set of timelines is closed, so a switch is used.
Under `SA_ENABLE_PROFILER` 0 the module is still built, and `mark_period` does nothing.

## 3. Zones and periods

Zone names are `<group>.<stage>`; the group is what `breakdown` sums.

| Zone | Where |
| --- | --- |
| `data.read` | Around every `dataset.read` in the loops of section 1 |
| `host.feed` | `objective_->feed` (`trainer.cpp:371`) and the truth-from-roles calls |
| `copy.input` | Every `blas.copy` of a batch into its input buffer |
| `device.run` | `step_->run()`, `training_->run`, `run.step.run` |
| `host.readback` | Port reads: `outputs_view_` (`anchor_training_step.cpp:217-219`), `floats_of` and `widen` (`free_training_step.cpp:310-311`, `:376-377`) |
| `host.assign` | Target assignment: `feed_deltas`, `branch.stage.assign`, `build_target_leaves` |
| `host.loss` | `current_loss`, `measure_terms` |
| `host.measure` | `objective_->measure` |
| `optim.schedule` | `learning_rate` and `apply_step_schedule` |
| `eval.decode`, `eval.nms`, `eval.ap` | `decode_batch` / `decoder.decode`, `select_nms_detections`, `add_batch` |
| `io.checkpoint` | `save_run_checkpoint`, `IO::load_checkpoint` |
| `compile.lower`, `compile.runtime` | `lowering.record` and `graph_.compile()` in `CompiledStep::compile` |
| `eager.partition`, `eager.plan`, `eager.compile`, `eager.execute`, `eager.flush` | The five stopwatch laps of the eager session |
| `train.fit` | The whole of `Trainer::fit` |

`mark_period(Timeline::TrainStep)` opens each training step, before its `data.read`, in
`run_epochs`, `Trainer::fit`, the anchor loop and the anchor-free loop.
`mark_period(Timeline::InferenceBatch)` opens each batch in the classifier and both detector
evaluation loops.

## 4. What leaves the library

| Removed | Replacement |
| --- | --- |
| `TrainerOptions::profile` | A profiling session of the program |
| `RunReport::NodeTimings`, `RunReport::node_timings`, `find_total`, `add_node_timings` | The `roofline` analysis of a record |
| `RunReport::duration_ms` and the wall-clock line of `to_string` | The `train.fit` zone |
| `StepCost::partition_ns`, `plan_ns`, `compile_ns`, `execute_ns`, `total_ns` and `Stopwatch` | The `eager.*` zones; the counts of `StepCost` stay |

These are breaking changes to public names, made without a deprecation release (before 1.0,
`api-stability` allows it); each has a changelog entry naming its replacement.

## 5. Graph labels and node origins

- `CompiledStep::compile` gains a last parameter `std::string_view label`, which it hands to
  `graph_.set_label` before `graph_.compile()`, so the first plan already carries it. Callers and
  labels: `TrainingStep` gives `train.lead` to `lead_` and `train.step` to `step_`
  (`training_step.cpp:166-175`), `EvaluationStep` gives `eval.step` (`evaluation_step.cpp:192`),
  `ReplicaStep` gives `replica.step` (`replica_step.cpp:125`). The eager `RegionExecutor` labels its
  graph `eager.region.<n>`, `n` its region index, before its first run.
- `Lowering::record` opens `recorder_->origin(label)` around each rule call, where `label` is
  `<node id>:<op name>` with the op name from `Ops::name_of`. The label strings are built once per
  compile and never on a run.

## 6. The programs' session

`applications/programs/source/profile_session.hpp` and `.cpp`:

```cpp
class ProfileSession final
{
    public:
        /// @brief Opens a session from the SushiProfiler configuration at @p config_path.
        [[nodiscard]] static Result<std::unique_ptr<ProfileSession>> open(
            const std::filesystem::path& config_path, const SushiRuntime::Runtime& runtime);

        ProfileSession(const ProfileSession&) = delete;
        ProfileSession& operator=(const ProfileSession&) = delete;
        ProfileSession(ProfileSession&&) = delete;
        ProfileSession& operator=(ProfileSession&&) = delete;
        ~ProfileSession();   // stops the session, then drops the runtime parts
};
```

It builds a `SushiProfiler::Registry`, registers SushiRuntime's parts with
`register_profiler_parts`, starts a `SushiProfiler::Session` from the configuration and owns
both, members declared so the session is destroyed before the parts. A configuration error is a
`Result` error naming the file. The programs take `--profile <path>`; `sa train` and
`sa inference` forward it. `sa profile <args>` runs `sp <args>`, and says to install SushiProfiler's
CLI when `sp` is not on the path.

## 7. Dependencies and build

- `cli/sushistack.deps.toml`: `[sushisystems.sushiprofiler]` pinned to v0.2.0; `depends_on` gains
  `sushiprofiler`.
- Options `SA_ENABLE_PROFILER` (ON) and `SA_PROFILE_LEVEL` (2), given to every module as compile
  definitions.
- `cmake/Profiler.cmake`: `find_package(SushiProfiler CONFIG REQUIRED)` when `SA_ENABLE_PROFILER`
  is ON.
- The package consumption of SushiLog, SushiRuntime and SushiBLAS is task O5 of SushiCore's
  `docs/agent/2026_10_26_PREBUILT_SUSHI_PACKAGES/PLAN.md`, applied here as P5.

## 8. Build time

| Change | What rebuilds |
| --- | --- |
| A zone added in a `.cpp` | That file |
| `profile.hpp` | The files that include it, all sources |
| A SushiRuntime or SushiBLAS `.cpp` | Nothing in SushiAI compiles; the packages' libraries relink |

No public header of SushiAI includes `profile.hpp` or any SushiProfiler header, so a SushiProfiler
upgrade recompiles only the sources with zones.

## 9. Not in this work

- P6, the YOLOv5n write-up, runs after the build.
- The distributed path's timing (`cluster_run.cpp`) has the zones of its own loop only; SushiRuntime
  times the distributed layer in its own later phase.
