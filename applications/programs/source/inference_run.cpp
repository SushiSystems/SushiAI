/**************************************************************************/
/* inference_run.cpp                                                      */
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
 * @file inference_run.cpp
 * @brief Implements the report registry, its lookup by task type and the shared header.
 * @author Mustafa Garip
 */

#include "inference_run.hpp"

#include <iostream>
#include <string>
#include <SushiLog/text/text_format.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Joins the task types of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_tasks(SushiRuntime::span<const InferenceEntry> rows)
            {
                std::string listed;
                for (const InferenceEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.task).append("'");
                }

                return listed;
            }

        } // namespace

        Status InferenceRegistry::add(const InferenceEntry& row)
        {
            SA_FATAL_IF(row.task.empty(), "A report row needs a non-empty task type");
            SA_FATAL_IF(row.report == nullptr, "The report row '{}' has no function", row.task);

            SA_FAIL_IF(find(row.task) != nullptr, StatusCode::InvalidDescription,
                       "a report for the task type '{}' is already registered", row.task);

            return Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const InferenceEntry* InferenceRegistry::find(std::string_view task) const noexcept
        {
            for (const InferenceEntry& row : rows_)
            {
                if (row.task == task)
                    return &row;
            }

            return nullptr;
        }

        SushiRuntime::span<const InferenceEntry> InferenceRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const InferenceEntry>(rows_.data(), rows_.size());
        }

        void print_inference_header(const PreparedInference& run, std::string_view outputs)
        {
            std::cout << "sushiai inference\n"
                      << "----------------------------------------------------------------\n"
                      << SushiLog::format_text("Model      : {} ({} layers -> {})\n",
                                                 run.model_path.string(), run.layers, outputs)
                      << SushiLog::format_text("Checkpoint : {}\n", run.weights_path.string())
                      << SushiLog::format_text("Data       : {} ({} x {})\n\n",
                                                 run.dataset.name(), run.dataset.batch_count(),
                                                 run.dataset.batch_size());
        }

        Status report_inference(const InferenceRegistry& registry, std::string_view task,
                                PreparedInference& run)
        {
            const InferenceEntry* const row = registry.find(task);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "'{}' has the task type '{}', and no report is registered for it; this "
                       "build reports: {}",
                       run.model_path.string(), task, list_tasks(registry.rows()));

            return row->report(run);
        }

    } // namespace Apps
} // namespace SushiAI
