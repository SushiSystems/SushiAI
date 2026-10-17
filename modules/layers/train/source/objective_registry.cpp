/**************************************************************************/
/* objective_registry.cpp                                                 */
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
 * @file objective_registry.cpp
 * @brief Implements the registry of the objectives a training file may name.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/sum_objective.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Joins the types of @p rows into "'a', 'b'" for a diagnostic. */
            [[nodiscard]] std::string list_types(SushiRuntime::span<const ObjectiveEntry> rows)
            {
                std::string listed;
                for (const ObjectiveEntry& row : rows)
                {
                    if (!listed.empty())
                        listed.append(", ");

                    listed.append("'").append(row.type).append("'");
                }

                return listed;
            }

        } // namespace

        Status ObjectiveRegistry::add(const ObjectiveEntry& row)
        {
            SA_FATAL_IF(row.type.empty(), "An objective row needs a non-empty type");
            SA_FATAL_IF(row.make == nullptr, "The objective row '{}' has no factory", row.type);

            SA_FAIL_IF(find(row.type) != nullptr, StatusCode::InvalidDescription,
                       "objective type '{}' is already registered", row.type);

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                rows_.push_back(row);
                return Status();
            });
        }

        const ObjectiveEntry* ObjectiveRegistry::find(std::string_view type) const noexcept
        {
            for (const ObjectiveEntry& row : rows_)
            {
                if (row.type == type)
                    return &row;
            }

            return nullptr;
        }

        SushiRuntime::span<const ObjectiveEntry> ObjectiveRegistry::rows() const noexcept
        {
            return SushiRuntime::span<const ObjectiveEntry>(rows_.data(), rows_.size());
        }

        Status register_objectives(ObjectiveRegistry& registry)
        {
            SA_TRY(registry.add(ObjectiveEntry{"cross_entropy", &make_cross_entropy}));
            SA_TRY(registry.add(
                ObjectiveEntry{K_BINARY_CROSS_ENTROPY_TYPE, &make_binary_cross_entropy}));
            SA_TRY(registry.add(ObjectiveEntry{K_MULTI_LABEL_TYPE, &make_multi_label}));
            SA_TRY(registry.add(
                ObjectiveEntry{K_MEAN_SQUARED_ERROR_TYPE, &make_mean_squared_error}));

            return registry.add(ObjectiveEntry{K_SUM_TYPE, &make_sum_objective});
        }

        Result<std::unique_ptr<Objective>> make_objective(const ObjectiveRegistry& registry,
                                                          std::string_view type,
                                                          const ObjectiveContext& given,
                                                          const Config::Value& attributes,
                                                          std::string_view context)
        {
            const ObjectiveEntry* const row = registry.find(type);
            SA_FAIL_IF(row == nullptr, StatusCode::InvalidDescription,
                       "unknown objective type '{}'; this build has: {}", type,
                       list_types(registry.rows()));

            if (given.registry == &registry)
                return row->make(given, attributes, context);

            // A composite builds its terms from the registry it was itself found in.
            SA_TRY_ASSIGN(const ObjectiveContext seen,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&given, &registry]()
            {
                ObjectiveContext copy = given;
                copy.registry = &registry;
                return copy;
            }));

            return row->make(seen, attributes, context);
        }

    } // namespace Train
} // namespace SushiAI
