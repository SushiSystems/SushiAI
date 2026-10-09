/**************************************************************************/
/* dataset_registry.cpp                                                   */
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
 * @file dataset_registry.cpp
 * @brief Implements the dataset registry, its "rings" row and make_dataset over a registry.
 * @author Mustafa Garip
 */

#include <SushiAI/data/dataset_registry.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /** @brief Joins the types of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_types(SushiRuntime::span<const DatasetEntry> rows)
            {
                std::string listed;
                for (const DatasetEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.type).append("'");
                }

                return listed;
            }

        } // namespace

        Status DatasetRegistry::add(const DatasetEntry& row)
        {
            SA_FATAL_IF(row.type.empty(), "A dataset row needs a non-empty type");
            SA_FATAL_IF(row.make == nullptr, "The dataset row '{}' has no factory", row.type);

            SA_FAIL_IF(find(row.type) != nullptr, StatusCode::InvalidDescription,
                       "dataset type '{}' is already registered", row.type);

            return Detail::guard(Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const DatasetEntry* DatasetRegistry::find(std::string_view type) const noexcept
        {
            for (const DatasetEntry& row : rows_)
            {
                if (row.type == type)
                    return &row;
            }

            return nullptr;
        }

        SushiRuntime::span<const DatasetEntry> DatasetRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const DatasetEntry>(rows_.data(), rows_.size());
        }

        Status register_datasets(DatasetRegistry& registry)
        {
            return registry.add(DatasetEntry{"rings", &make_rings});
        }

        Result<std::unique_ptr<Dataset>> make_dataset(const DatasetRegistry& registry,
                                                      SushiBLAS::Engine& engine,
                                                      std::string_view type,
                                                      const Config::Value& attributes,
                                                      std::string_view context)
        {
            const DatasetEntry* const row = registry.find(type);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "unknown dataset type '{}'; this build has: {}", type,
                       list_types(registry.rows()));

            return row->make(engine, attributes, context);
        }

    } // namespace Data
} // namespace SushiAI
