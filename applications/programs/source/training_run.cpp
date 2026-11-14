/**************************************************************************/
/* training_run.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file training_run.cpp
 * @brief Implements the run registry and the choice of the run of an objective.
 * @author Mustafa Garip
 */

#include "training_run.hpp"

#include <cstdint>
#include <string>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Joins the objective types of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_objectives(SushiRuntime::span<const RunEntry> rows)
            {
                std::string listed;
                for (const RunEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.objective).append("'");
                }

                return listed;
            }

        } // namespace

        std::optional<IO::RunProgress> progress_to_checkpoint(const CheckpointSchedule& schedule,
                                                              const UpdateAt& at,
                                                              std::size_t step_runs) noexcept
        {
            const Config::CheckpointEvery& every = schedule.every;
            if (every.unit == Config::CheckpointUnit::None)
                return std::nullopt;

            const bool ends_epoch = at.batch + 1 == schedule.batches;
            const bool ends_run = ends_epoch && at.epoch == schedule.epochs;
            if (ends_run)
                return std::nullopt;

            const bool due = every.unit == Config::CheckpointUnit::Batches
                ? at.updates % every.count == 0
                : ends_epoch && at.epoch % every.count == 0;
            if (!due)
                return std::nullopt;

            IO::RunProgress progress;
            progress.seed = schedule.start.seed;
            progress.step_runs = static_cast<std::uint64_t>(step_runs);
            const auto current_epoch = schedule.start.epoch + static_cast<std::uint64_t>(at.epoch - 1);
            progress.epoch = ends_epoch ? current_epoch + 1 : current_epoch;
            progress.batch = ends_epoch ? 0 : static_cast<std::uint64_t>(at.batch + 1);
            return progress;
        }

        Status checkpoint_after_update(const PreparedRun& run, const UpdateAt& at,
                                       std::size_t step_runs)
        {
            CheckpointSchedule schedule;
            schedule.every = run.spec.checkpoint_every;
            schedule.epochs = run.spec.epochs;
            schedule.batches = run.dataset.batch_count();
            schedule.start = run.start;

            const std::optional<IO::RunProgress> progress =
                progress_to_checkpoint(schedule, at, step_runs);
            if (!progress)
                return Status();

            SA_ZONE("io", "checkpoint");
            return IO::save_run_checkpoint(run.spec.checkpoint, run.state, run.optimizer, *progress);
        }

        IO::RunProgress finished_progress(const PreparedRun& run,
                                          std::size_t step_runs) noexcept
        {
            IO::RunProgress progress;
            progress.seed = run.start.seed;
            progress.step_runs = static_cast<std::uint64_t>(step_runs);
            progress.epoch = run.start.epoch + static_cast<std::uint64_t>(run.spec.epochs);
            progress.batch = 0;
            return progress;
        }

        Status check_start_batch(const PreparedRun& run)
        {
            const std::size_t batches = run.dataset.batch_count();
            SA_FAIL_IF(run.start.batch != 0 && run.start.batch >= batches,
                       StatusCode::Incompatible,
                       "the run resumes at batch {} of epoch {}, but the '{}' data holds {} "
                       "batches an epoch",
                       run.start.batch, run.start.epoch, run.dataset.name(), batches);
            return Status();
        }

        Status RunRegistry::add(const RunEntry& row)
        {
            SA_FATAL_IF(row.objective.empty(), "A run row needs a non-empty objective type");
            SA_FATAL_IF(row.run == nullptr, "The run row '{}' has no function", row.objective);

            SA_FAIL_IF(find(row.objective) != nullptr, StatusCode::InvalidDescription,
                       "a run for the objective type '{}' is already registered",
                       row.objective);

            return Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const RunEntry* RunRegistry::find(std::string_view objective) const noexcept
        {
            for (const RunEntry& row : rows_)
            {
                if (row.objective == objective)
                    return &row;
            }

            return nullptr;
        }

        Status RunRegistry::set_fallback(RunFunction run)
        {
            SA_FATAL_IF(run == nullptr, "The fallback run has no function");
            SA_FAIL_IF(fallback_ != nullptr, StatusCode::InvalidDescription,
                       "a fallback run is already registered");

            fallback_ = run;
            return Status();
        }

        RunFunction RunRegistry::fallback() const noexcept
        {
            return fallback_;
        }

        SushiRuntime::span<const RunEntry> RunRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const RunEntry>(rows_.data(), rows_.size());
        }

        Result<RunSummary> run_training(const RunRegistry& registry, std::string_view objective,
                                        ExecutionBackend& backend, PreparedRun& run)
        {
            const RunEntry* const row = registry.find(objective);
            if (row != nullptr)
                return row->run(backend, run);

            const RunFunction fallback = registry.fallback();
            SA_FAIL_IF(fallback == nullptr, StatusCode::InvalidDescription,
                       "no training run is registered for the objective type '{}'; this build "
                       "trains: {}",
                       objective, list_objectives(registry.rows()));

            return fallback(backend, run);
        }

    } // namespace Apps
} // namespace SushiAI
