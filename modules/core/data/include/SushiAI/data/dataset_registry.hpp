/**************************************************************************/
/* dataset_registry.hpp                                                   */
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
 * @file dataset_registry.hpp
 * @brief Declares the dataset registry, its rows and the builder of a dataset from a type.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @brief Builds one dataset from its configuration attributes.
         * @param engine The engine that allocates and fills the tensors.
         * @param attributes The dataset's own keys; the factory rejects any it does not know.
         * @param context What to call this dataset in a diagnostic.
         * @return The dataset, never null, or the failure that kept it from being built.
         */
        using DatasetFactory = Result<std::unique_ptr<Dataset>> (*)(SushiBLAS::Engine& engine,
                                                                    const Config::Value& attributes,
                                                                    std::string_view context);

        /**
         * @struct DatasetEntry
         * @brief One row of a DatasetRegistry.
         */
        struct DatasetEntry
        {
            /** @brief The "type" a configuration file spells. */
            std::string_view type;

            /** @brief What builds it. */
            DatasetFactory make;
        };

        /** @brief Holds the dataset rows a configuration file may name, by type. */
        class DatasetRegistry final
        {
            public:
                /**
                 * @brief Appends a dataset row.
                 * @param row Its type text must outlive the registry; an empty type or a null
                 *            factory is fatal.
                 * @return InvalidDescription when a row holds the type.
                 */
                [[nodiscard]] Status add(const DatasetEntry& row);

                /** @brief Returns the row of @p type, or null when none holds it. */
                [[nodiscard]] const DatasetEntry* find(std::string_view type) const noexcept;

                /**
                 * @brief Returns the rows in the order they were added.
                 * @return A view the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const DatasetEntry> rows() const noexcept;

            private:
                /** @brief Holds the rows in insertion order. */
                std::vector<DatasetEntry> rows_;
        };

        /**
         * @brief Adds the "rings" row to @p registry.
         * @param registry The registry the row joins.
         * @return Success, or InvalidDescription when "rings" is already held.
         */
        [[nodiscard]] Status register_datasets(DatasetRegistry& registry);

        /**
         * @brief Builds the dataset named @p type from the rows of @p registry.
         * @param engine The engine that allocates and fills the tensors.
         * @param context What to call the dataset in a diagnostic.
         * @return The dataset, never null; InvalidDescription, naming every type the registry
         *         holds, when @p type is no row; or the failure the type's factory returns.
         */
        [[nodiscard]] Result<std::unique_ptr<Dataset>> make_dataset(
            const DatasetRegistry& registry,
            SushiBLAS::Engine& engine,
            std::string_view type,
            const Config::Value& attributes,
            std::string_view context);

    } // namespace Data
} // namespace SushiAI
