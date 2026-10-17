/**************************************************************************/
/* objective_registry.hpp                                                 */
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
 * @file objective_registry.hpp
 * @brief Declares the registry of the objectives a training file may name.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Builds one objective from its configuration attributes.
         * @param given What the factory may read of the run besides its own section.
         * @param attributes The objective's own keys; the factory refuses any it does not know.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, never null, or the failure that kept it from being built.
         */
        using ObjectiveFactory = Result<std::unique_ptr<Objective>> (*)(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context);

        /**
         * @struct ObjectiveEntry
         * @brief One row of an ObjectiveRegistry.
         */
        struct ObjectiveEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            ObjectiveFactory make;
        };

        /** @brief Holds the objective rows a training file may name, by type. */
        class ObjectiveRegistry final
        {
            public:
                /**
                 * @brief Appends an objective row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            factory is fatal.
                 * @return InvalidDescription when a row holds the type.
                 */
                [[nodiscard]] Status add(const ObjectiveEntry& row);

                /** @brief Returns the row of @p type, or null when none holds it. */
                [[nodiscard]] const ObjectiveEntry* find(std::string_view type) const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const ObjectiveEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<ObjectiveEntry> rows_;
        };

        /**
         * @brief Adds the "cross_entropy", "binary_cross_entropy", "multi_label",
         *        "mean_squared_error" and "sum" rows to @p registry.
         * @param registry The registry the rows join.
         * @return Success, or InvalidDescription when one of the five types is already held.
         */
        [[nodiscard]] Status register_objectives(ObjectiveRegistry& registry);

        /**
         * @brief Builds the objective named @p type from the rows of @p registry.
         * @param given What the type's factory may read of the run; the factory sees it
         *              with @p registry as its registry, whatever @p given names.
         * @param context What to call the objective in a diagnostic.
         * @return The objective, never null; InvalidDescription, naming every type the registry
         *         holds, when @p type is no row; or the failure the type's factory returns.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_objective(
            const ObjectiveRegistry& registry,
            std::string_view type,
            const ObjectiveContext& given,
            const Config::Value& attributes,
            std::string_view context);

    } // namespace Train
} // namespace SushiAI
