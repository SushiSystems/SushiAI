/**************************************************************************/
/* importer_registry.cpp                                                  */
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
 * @file importer_registry.cpp
 * @brief Implements the importer registry and the import of a model by format name.
 * @author Mustafa Garip
 */

#include <SushiAI/io/importer_registry.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Joins the formats of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_formats(SushiRuntime::span<const ImporterEntry> rows)
            {
                std::string listed;
                for (const ImporterEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.format).append("'");
                }

                return listed;
            }

        } // namespace

        Status ImporterRegistry::add(const ImporterEntry& row)
        {
            SA_FATAL_IF(row.format.empty(), "An importer row needs a non-empty format");
            SA_FATAL_IF(row.read == nullptr, "The importer row '{}' has no function",
                        row.format);

            SA_FAIL_IF(find(row.format) != nullptr, StatusCode::InvalidDescription,
                       "import format '{}' is already registered", row.format);

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const ImporterEntry* ImporterRegistry::find(std::string_view format) const noexcept
        {
            for (const ImporterEntry& row : rows_)
            {
                if (row.format == format)
                    return &row;
            }

            return nullptr;
        }

        SushiRuntime::span<const ImporterEntry> ImporterRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const ImporterEntry>(rows_.data(), rows_.size());
        }

        Result<ImportedModel> import_model(const ImporterRegistry& registry,
                                           std::string_view format,
                                           SushiRuntime::span<const std::filesystem::path> files)
        {
            const ImporterEntry* const row = registry.find(format);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "unknown import format '{}'; this build has: {}", format,
                       list_formats(registry.rows()));

            return row->read(files);
        }

    } // namespace IO
} // namespace SushiAI
