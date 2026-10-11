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
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file training_run.cpp
 * @brief Implements the run registry and the choice of the run of an objective.
 * @author Mustafa Garip
 */

#include "training_run.hpp"

#include <cstdint>
#include <string>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/seam.hpp>

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
                                        PreparedRun& run)
        {
            const RunEntry* const row = registry.find(objective);
            if (row != nullptr)
                return row->run(run);

            const RunFunction fallback = registry.fallback();
            SA_FAIL_IF(fallback == nullptr, StatusCode::InvalidDescription,
                       "no training run is registered for the objective type '{}'; this build "
                       "trains: {}",
                       objective, list_objectives(registry.rows()));

            return fallback(run);
        }

    } // namespace Apps
} // namespace SushiAI
