/**************************************************************************/
/* exporter_registry.cpp                                                  */
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
 * @file exporter_registry.cpp
 * @brief Implements the exporter registry and the export of a model by format name.
 * @author Mustafa Garip
 */

#include <SushiAI/io/exporter_registry.hpp>

#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/exported_model.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Joins the formats of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_formats(SushiRuntime::span<const ExporterEntry> rows)
            {
                std::string listed;
                for (const ExporterEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.format).append("'");
                }

                return listed;
            }

        } // namespace

        Status ExporterRegistry::add(const ExporterEntry& row) noexcept
        {
            SA_FATAL_IF(row.format.empty(), "An exporter row needs a non-empty format");
            SA_FATAL_IF(row.write == nullptr, "The exporter row '{}' has no function",
                        row.format);

            SA_FAIL_IF(row.format == K_WEIGHT_EXPORT_WORD, StatusCode::InvalidDescription,
                       "export format '{}' is the word of the weight export", row.format);
            SA_FAIL_IF(find(row.format) != nullptr, StatusCode::InvalidDescription,
                       "export format '{}' is already registered", row.format);

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const ExporterEntry* ExporterRegistry::find(std::string_view format) const noexcept
        {
            for (const ExporterEntry& row : rows_)
            {
                if (row.format == format)
                    return &row;
            }

            return nullptr;
        }

        SushiRuntime::span<const ExporterEntry> ExporterRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const ExporterEntry>(rows_.data(), rows_.size());
        }

        Result<ExportedModel> export_model(const ExporterRegistry& registry,
                                           std::string_view format, const ImportedModel& model,
                                           std::string_view stem) noexcept
        {
            const ExporterEntry* const row = registry.find(format);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "unknown export format '{}'; this build has: {}", format,
                       list_formats(registry.rows()));

            return row->write(model, stem);
        }

    } // namespace IO
} // namespace SushiAI
